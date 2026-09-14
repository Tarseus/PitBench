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
import stat
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


def capture_workspace(root: Path, spool: Path) -> dict:
    """Capture actual files, including ignored builds and initialized submodules.

    Every recorded regular file is opened once and checked for concurrent
    modification. A changed file is reported as unstable, never as an exact
    version. Git administration directories and the recorder spool are excluded.
    """
    objects = spool / "objects"
    objects.mkdir(parents=True, exist_ok=True)
    entries = {}
    errors = []
    new_objects = []
    cache_path = spool / "files.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    started = time.monotonic()
    root = root.resolve()
    spool_root = spool.resolve()
    if not root.is_dir():
        raise ValueError(f"workspace is not a directory: {root}")
    for directory, directories, filenames in os.walk(
        root,
        followlinks=False,
        onerror=lambda error: errors.append(
            {"path": str(error.filename), "error": str(error)}
        ),
    ):
        directories[:] = sorted(name for name in directories if name != ".git")
        for name in sorted([*directories, *filenames]):
            path = Path(directory) / name
            if name == ".git" or path == spool_root or spool_root in path.parents:
                if name in directories:
                    directories.remove(name)
                continue
            relative = str(path.relative_to(root))
            try:
                before = path.lstat()
                item = {"mode": stat.S_IMODE(before.st_mode)}
                if stat.S_ISLNK(before.st_mode):
                    item.update(kind="symlink", target=os.readlink(path))
                elif stat.S_ISDIR(before.st_mode):
                    item.update(kind="directory")
                elif stat.S_ISREG(before.st_mode):
                    previous = cache.get(relative, {})
                    signature = [
                        before.st_dev,
                        before.st_ino,
                        before.st_size,
                        before.st_mtime_ns,
                        before.st_ctime_ns,
                    ]
                    if (
                        previous.get("signature") == signature
                        and (objects / previous["sha256"]).is_file()
                    ):
                        entries[relative] = {**previous, "mode": item["mode"]}
                        continue
                    temporary = objects / f".blob-{uuid.uuid4().hex}"
                    digest = hashlib.sha256()
                    size = 0
                    try:
                        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                        with (
                            os.fdopen(fd, "rb") as source,
                            temporary.open("xb") as target,
                        ):
                            opened = os.fstat(source.fileno())
                            while chunk := source.read(1024 * 1024):
                                digest.update(chunk)
                                target.write(chunk)
                                size += len(chunk)
                            after = os.fstat(source.fileno())
                            target.flush()
                            os.fsync(target.fileno())
                        object_id = digest.hexdigest()
                        destination = objects / object_id
                        if destination.exists():
                            temporary.unlink()
                        else:
                            os.replace(temporary, destination)
                            new_objects.append(object_id)
                        stable = (
                            before.st_dev,
                            before.st_ino,
                            before.st_size,
                            before.st_mtime_ns,
                            before.st_ctime_ns,
                        ) == (
                            opened.st_dev,
                            opened.st_ino,
                            opened.st_size,
                            opened.st_mtime_ns,
                            opened.st_ctime_ns,
                        ) == (
                            after.st_dev,
                            after.st_ino,
                            after.st_size,
                            after.st_mtime_ns,
                            after.st_ctime_ns,
                        ) and size == after.st_size
                        item.update(
                            kind="file",
                            sha256=object_id,
                            bytes=size,
                            stable=stable,
                            inode=after.st_ino,
                            device=after.st_dev,
                            mtime_ns=after.st_mtime_ns,
                            ctime_ns=after.st_ctime_ns,
                            signature=signature if stable else None,
                        )
                        if not stable:
                            errors.append(
                                {"path": relative, "error": "changed_during_capture"}
                            )
                    finally:
                        temporary.unlink(missing_ok=True)
                else:
                    item.update(kind="special", stable=False)
                    errors.append(
                        {"path": relative, "error": "special_file_not_captured"}
                    )
                entries[relative] = item
            except OSError as error:
                errors.append(
                    {"path": relative, "error": f"{type(error).__name__}: {error}"}
                )
    # Identity describes content/mode, not sampling timestamps or inode numbers.
    identity = {
        path: {
            key: value
            for key, value in item.items()
            if key not in {"inode", "device", "mtime_ns", "ctime_ns", "signature"}
        }
        for path, item in entries.items()
    }
    state_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    descriptor = os.open(objects, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    durable_json(cache_path, entries)
    return {
        "state_id": state_id,
        "root": str(root),
        "entries": entries,
        "errors": errors,
        "capture_elapsed_sec": time.monotonic() - started,
        "consistency": "file_versions_checked; directory_walk_not_atomic",
        "includes_ignored_files": True,
        "excludes": [".git", "recorder spool"],
        "new_objects": new_objects,
    }


def record_hook(
    spool: Path,
    provider: str,
    payload: dict,
    root: Path | None,
    *,
    hook_event: str | None = None,
) -> dict:
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
        event_name = (
            hook_event or payload.get("hook_event_name") or payload.get("event") or ""
        )
        targets = {}
        tool_input = (
            payload.get("tool_input")
            or (payload.get("toolCall") or {}).get("args")
            or {}
        )
        for key, value in tool_input.items() if isinstance(tool_input, dict) else []:
            if key.lower() not in {
                "path",
                "file_path",
                "filepath",
                "absolutepath",
                "targetfile",
            } or not isinstance(value, str):
                continue
            path = Path(value)
            if not path.is_absolute() and payload.get("cwd"):
                path = Path(payload["cwd"]) / path
            if not path.is_absolute():
                continue
            if path.is_file() and not path.is_symlink():
                content = path.read_bytes()
                digest = hashlib.sha256(content).hexdigest()
                objects = spool / "objects"
                objects.mkdir(exist_ok=True)
                destination = objects / digest
                if not destination.exists():
                    with destination.open("xb") as stream:
                        stream.write(content)
                        stream.flush()
                        os.fsync(stream.fileno())
                targets[str(path)] = {
                    "sha256": digest,
                    "bytes": len(content),
                    "exists": True,
                }
            elif not path.exists():
                targets[str(path)] = {"exists": False}
        if targets:
            event["tool_targets"] = {
                "entries": targets,
                "state_id": hashlib.sha256(
                    json.dumps(targets, sort_keys=True).encode()
                ).hexdigest(),
                "scope": "explicit native tool file targets; not the entire task repository",
            }
        if root is not None and event_name in {
            "PreToolUse",
            "PostToolUse",
            "PostToolUseFailure",
            "BeforeTool",
            "AfterTool",
            "preToolUse",
            "postToolUse",
            "postToolUseFailure",
            "WorkspaceSnapshot",
        }:
            try:
                event["workspace"] = capture_workspace(root, spool)
            except Exception as error:
                event["capture_error"] = f"{type(error).__name__}: {error}"
        transcript = payload.get("transcriptPath") or payload.get("transcript_path")
        if transcript and Path(transcript).is_file():
            content = Path(transcript).read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            objects = spool / "objects"
            objects.mkdir(exist_ok=True)
            destination = objects / digest
            if not destination.exists():
                with destination.open("xb") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
            event["transcript"] = {
                "sha256": digest,
                "bytes": len(content),
                "source": transcript,
            }
        if provider == "filesystem" and payload.get("operation") == "close_write":
            path = Path(payload["absolute_path"])
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(fd, "rb") as stream:
                    before = os.fstat(stream.fileno())
                    if not stat.S_ISREG(before.st_mode):
                        raise OSError(
                            "file notification does not refer to a regular file"
                        )
                    content = stream.read()
                    after = os.fstat(stream.fileno())
                digest = hashlib.sha256(content).hexdigest()
                objects = spool / "objects"
                objects.mkdir(exist_ok=True)
                destination = objects / digest
                if not destination.exists():
                    with destination.open("xb") as stream:
                        stream.write(content)
                        stream.flush()
                        os.fsync(stream.fileno())
                event["file_version"] = {
                    "sha256": digest,
                    "bytes": len(content),
                    "stable": (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                    == (after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                    "association": "read_after_kernel_notification",
                }
            except OSError as error:
                event["capture_error"] = f"file version unavailable: {error}"
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
            digests = {
                item["sha256"]
                for item in (event.get("workspace") or {}).get("entries", {}).values()
                if item.get("kind") == "file"
            }
            if event.get("transcript"):
                digests.add(event["transcript"]["sha256"])
            if event.get("file_version"):
                digests.add(event["file_version"]["sha256"])
            digests.update(
                item["sha256"]
                for item in (event.get("tool_targets") or {})
                .get("entries", {})
                .values()
                if item.get("sha256")
            )
            if event["completed_ns"] <= after:
                included.update(digests)
            else:
                selected.append((path, event, digests))
        with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:
            for path, event, digests in selected:
                archive.add(path, arcname=path.name)
                for digest in sorted(digests):
                    if digest not in included:
                        archive.add(
                            spool / "objects" / digest, arcname=f"objects/{digest}"
                        )
                        included.add(digest)


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
        choices=["hook", "snapshot", "install", "export", "watch", "stop-watch"],
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
    elif args.action == "snapshot":
        event = record_hook(
            args.spool,
            "pitbench",
            {
                "hook_event_name": "WorkspaceSnapshot",
                "reason": args.reason,
                "call_id": args.call_id,
            },
            args.root,
        )
        print(json.dumps({"id": event["id"], "completed_ns": event["completed_ns"]}))
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
