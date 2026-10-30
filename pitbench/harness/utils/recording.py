"""Standalone native-hook recorder staged into disposable agent environments.

This file intentionally uses only the standard library: agent images need not
install PitBench. All canonical logs are still written by the host AgentTrace.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import select
import shlex
import struct
import sys
import tarfile
import time
import uuid
from pathlib import Path


def durable_json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _store_content(objects: Path, content: bytes) -> str:
    digest = hashlib.sha256(content).hexdigest()
    destination = objects / digest
    if not destination.exists():
        with destination.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    return digest


def _read_content(objects: Path, digest: str) -> bytes:
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("invalid transcript object digest")
    content = (objects / digest).read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError(f"corrupt transcript object: {digest}")
    return content


def read_transcript(objects: Path, reference: dict) -> bytes:
    """Recover an exact transcript version, including legacy full-file objects."""
    if reference.get("format") != "transcript-chunks-v1":
        content = _read_content(objects, reference["sha256"])
        if len(content) != reference["bytes"]:
            raise ValueError("transcript size mismatch")
        return content
    chunks = []
    digest = reference["sha256"]
    while digest is not None:
        manifest = json.loads(_read_content(objects, digest))
        content = _read_content(objects, manifest["chunk"]["sha256"])
        if len(content) != manifest["chunk"]["bytes"]:
            raise ValueError("transcript chunk size mismatch")
        chunks.append((manifest["offset"], content))
        digest = manifest["previous"]
    result = bytearray()
    for offset, content in reversed(chunks):
        if offset != len(result):
            raise ValueError("noncontiguous transcript chunks")
        result.extend(content)
    if len(result) != reference["total_bytes"]:
        raise ValueError("transcript size mismatch")
    return bytes(result)


def _capture_transcript(spool: Path, source: str) -> dict:
    objects = spool / "objects"
    objects.mkdir(exist_ok=True)
    state_path = spool / ".transcripts.json"
    states = json.loads(state_path.read_text()) if state_path.exists() else {}
    key = str(Path(source).resolve())
    content = Path(source).read_bytes()
    previous = states.get(key)
    offset = previous["bytes"] if previous else 0
    appending = (
        previous is not None
        and len(content) >= offset
        and hashlib.sha256(content[:offset]).hexdigest() == previous["content_sha256"]
    )
    if appending and len(content) == offset:
        return previous["reference"]
    if not appending:
        offset = 0
    chunk = content[offset:]
    manifest = {
        "previous": previous["reference"]["sha256"] if appending else None,
        "offset": offset,
        "chunk": {"sha256": _store_content(objects, chunk), "bytes": len(chunk)},
    }
    encoded = json.dumps(manifest, sort_keys=True).encode()
    reference = {
        "format": "transcript-chunks-v1",
        "sha256": _store_content(objects, encoded),
        "bytes": len(encoded),
        "source": source,
        "total_bytes": len(content),
    }
    states[key] = {
        "bytes": len(content),
        "content_sha256": hashlib.sha256(content).hexdigest(),
        "reference": reference,
    }
    durable_json(state_path, states)
    return reference


def record_hook(
    spool: Path,
    provider: str,
    payload: dict,
    root: Path | None,
    *,
    hook_event: str | None = None,
) -> dict:
    # The canonical trace keeps the tool payload. Repository changes are captured
    # as Git patches by AgentTrace, so copying target files and the full workspace
    # here only stores duplicate source and build artifacts.
    del root
    spool.mkdir(parents=True, exist_ok=True)
    with (spool / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        event = {
            "schema_version": 1,
            "id": uuid.uuid4().hex,
            "provider": provider,
            "timestamp_ns": time.time_ns(),
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "payload": payload,
            "hook_event": hook_event,
        }
        transcript = payload.get("transcriptPath") or payload.get("transcript_path")
        if transcript and Path(transcript).is_file():
            event["transcript"] = _capture_transcript(spool, transcript)
        event["completed_ns"] = time.time_ns()
        durable_json(spool / f"event-{event['timestamp_ns']}-{event['id']}.json", event)
        return event


def export_events(spool: Path, after: int) -> None:
    if not spool.exists():
        with tarfile.open(fileobj=sys.stdout.buffer, mode="w|"):
            return
    with (spool / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        included = set()
        selected = []
        for path in sorted(spool.glob("event-*.json")):
            event = json.loads(path.read_text())
            digests = set()
            if event.get("transcript"):
                digests.add(event["transcript"]["sha256"])
            if event["completed_ns"] <= after:
                included.update(digests)
            else:
                selected.append((path, event, digests))
        with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:
            for path, event, digests in selected:
                archive.add(path, arcname=path.name)
                pending = [
                    (
                        digest,
                        event["transcript"].get("format") == "transcript-chunks-v1",
                    )
                    for digest in sorted(digests)
                ]
                while pending:
                    digest, chunked = pending.pop()
                    if digest not in included:
                        archive.add(
                            spool / "objects" / digest, arcname=f"objects/{digest}"
                        )
                        included.add(digest)
                        if chunked:
                            manifest = json.loads(
                                _read_content(spool / "objects", digest)
                            )
                            pending.append((manifest["chunk"]["sha256"], False))
                            if manifest["previous"] is not None:
                                pending.append((manifest["previous"], True))


def watch_workspace(root: Path, spool: Path) -> None:
    """Journal Linux filesystem notifications, with explicit overflow/gap events."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.inotify_init1.argtypes = [ctypes.c_int]
    libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
    descriptor = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
    if descriptor < 0:
        raise OSError(ctypes.get_errno(), "inotify_init1")
    paths = {}
    root = root.resolve()
    mask = (
        0x00000004
        | 0x00000008
        | 0x00000040
        | 0x00000080
        | 0x00000100
        | 0x00000200
        | 0x00000400
        | 0x00000800
    )

    def add(directory):
        for parent, directories, _ in os.walk(directory, followlinks=False):
            directories[:] = [
                name
                for name in directories
                if name != ".git" and not (Path(parent) / name).is_symlink()
            ]
            watch = libc.inotify_add_watch(descriptor, os.fsencode(parent), mask)
            if watch < 0:
                raise OSError(ctypes.get_errno(), f"inotify_add_watch: {parent}")
            paths[watch] = Path(parent)

    try:
        add(root)
        spool.mkdir(parents=True, exist_ok=True)
        print("READY", flush=True)
        while not (spool / "stop-watcher").exists():
            if not select.select([descriptor], [], [], 0.1)[0]:
                continue
            data = os.read(descriptor, 1024 * 1024)
            offset = 0
            while offset < len(data):
                watch, flags, cookie, length = struct.unpack_from("iIII", data, offset)
                offset += 16
                name = os.fsdecode(data[offset : offset + length].split(b"\0", 1)[0])
                offset += length
                if flags & 0x00004000:
                    record_hook(
                        spool,
                        "filesystem",
                        {"operation": "overflow", "history_complete": False},
                        None,
                    )
                    continue
                if watch not in paths or name == ".git":
                    continue
                path = paths[watch] / name if name else paths[watch]
                if flags & 0x40000000 and flags & (0x00000100 | 0x00000080):
                    try:
                        add(path)
                    except OSError as error:
                        record_hook(
                            spool,
                            "filesystem",
                            {"operation": "watch_gap", "error": str(error)},
                            None,
                        )
                for bit, operation in [
                    (4, "attributes"),
                    (8, "close_write"),
                    (64, "move_from"),
                    (128, "move_to"),
                    (256, "create"),
                    (512, "delete"),
                    (1024, "delete_self"),
                    (2048, "move_self"),
                ]:
                    if flags & bit:
                        record_hook(
                            spool,
                            "filesystem",
                            {
                                "operation": operation,
                                "path": str(path.relative_to(root)),
                                "absolute_path": str(path),
                                "cookie": cookie,
                                "is_directory": bool(flags & 0x40000000),
                            },
                            None,
                        )
    finally:
        os.close(descriptor)


def hook_configuration(provider: str, command: str) -> dict:
    if provider in {"codex", "claude", "antigravity"}:
        events = ["PreToolUse", "PostToolUse", "PostToolUseFailure"]
        if provider != "antigravity":
            events += [
                "SubagentStart",
                "SubagentStop",
                "UserPromptSubmit",
                "SessionEnd",
            ]
        return {
            "hooks": {
                event: [
                    {
                        "matcher": "*",
                        "hooks": [
                            {
                                "type": "command",
                                "command": command + " --event " + event,
                                "timeout": 120,
                            }
                        ],
                    }
                ]
                for event in events
            }
        }
    if provider == "gemini":
        return {
            "hooks": {
                event: [
                    {
                        "matcher": "*",
                        "hooks": [
                            {
                                "name": "pitbench-recording",
                                "type": "command",
                                "command": command + " --event " + event,
                                "timeout": 120000,
                            }
                        ],
                    }
                ]
                for event in [
                    "BeforeTool",
                    "AfterTool",
                    "BeforeModel",
                    "AfterModel",
                    "PreCompress",
                    "SessionStart",
                    "SessionEnd",
                ]
            }
        }
    if provider == "cursor":
        return {
            "version": 1,
            "hooks": {
                event: [{"command": command + " --event " + event, "timeout": 120}]
                for event in [
                    "preToolUse",
                    "postToolUse",
                    "postToolUseFailure",
                    "subagentStart",
                    "subagentStop",
                    "preCompact",
                    "sessionEnd",
                ]
            },
        }
    raise ValueError(f"native hooks are not implemented for {provider}")


def install_hooks(path: Path, provider: str, command: str) -> None:
    path = path.expanduser()
    config = json.loads(path.read_text()) if path.exists() else {}
    added = hook_configuration(provider, command)
    hooks = config.setdefault("hooks", {})
    for event, definitions in added["hooks"].items():
        existing = hooks.setdefault(event, [])
        # Avoid overwriting user hooks, including on retries in the same runtime.
        for definition in definitions:
            if definition not in existing:
                existing.append(definition)
    if "version" in added:
        config.setdefault("version", added["version"])
    path.parent.mkdir(parents=True, exist_ok=True)
    durable_json(path, config)


def configure_mounted_hooks(path: Path, provider: str) -> bool:
    script = Path("/opt/pitbench/recording.py")
    spool = Path("/opt/pitbench/recording")
    if not script.is_file() or not spool.is_dir():
        return False
    install_hooks(
        path,
        provider,
        shlex.join(
            [
                sys.executable,
                str(script),
                "hook",
                "--spool",
                str(spool),
                "--provider",
                provider,
            ]
        ),
    )
    return True


def finalize_mounted_recording(provider: str) -> None:
    spool = Path("/opt/pitbench/recording")
    if not spool.is_dir():
        return
    transcripts = {}
    for path in spool.glob("event-*.json"):
        event = json.loads(path.read_text())
        payload = event["payload"]
        transcript = payload.get("transcriptPath") or payload.get("transcript_path")
        if transcript:
            transcripts[transcript] = payload
    for path, payload in transcripts.items():
        record_hook(
            spool,
            provider,
            {**payload, "transcriptPath": path},
            None,
            hook_event="SessionEnd",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=["hook", "install", "export", "watch", "stop-watch"],
    )
    parser.add_argument("--spool", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--provider", default="pitbench")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--command")
    parser.add_argument("--after", type=int, default=0)
    parser.add_argument("--reason", default="boundary")
    parser.add_argument("--call-id")
    parser.add_argument("--event")
    args = parser.parse_args()
    if args.action == "install":
        install_hooks(args.config, args.provider, args.command)
    elif args.action == "export":
        export_events(args.spool, args.after)
    elif args.action == "watch":
        watch_workspace(args.root, args.spool)
    elif args.action == "stop-watch":
        (args.spool / "stop-watcher").touch()
    else:
        record_hook(
            args.spool,
            args.provider,
            json.load(sys.stdin),
            args.root,
            hook_event=args.event,
        )
        # An observation does not grant permission, rewrite input or add context.
        if args.provider != "antigravity":
            print("{}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
