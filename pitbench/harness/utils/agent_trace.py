"""Agent-independent collection of observable execution evidence.

Only the harness opens a trace. Tools and adapters publish into that trace; they
do not choose filenames, event schemas, or persistence policies. Raw evidence
is content-addressed and is never shortened to fit the event index.
"""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import os
import shlex
import tarfile
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from pitbench.harness.utils import recording
from pitbench.harness.utils.pipeline_trace import _json_safe
from pitbench.harness.utils.repository import (
    archive_repository_head,
    capture_repository_patch,
)

_CURRENT: ContextVar[AgentTrace | None] = ContextVar("agent_trace", default=None)
_CALL: ContextVar[str | None] = ContextVar("agent_trace_call", default=None)


def current_trace() -> AgentTrace | None:
    return _CURRENT.get()


def close_event_stream(stream) -> None:
    # Docker leaves response ownership to callers of exec_start(stream=True).
    # Close the HTTP response while its socket is still alive; GC otherwise
    # closes them in the wrong order on Python 3.13.
    response = getattr(stream, "_response", None)
    if response is not None:
        response.close()
    elif hasattr(stream, "close"):
        stream.close()


def _encode(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, type) and issubclass(value, BaseModel):
        return value.model_json_schema()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseException):
        return {"type": type(value).__name__, "message": str(value)}
    if isinstance(value, bytes):
        raise TypeError("Store bytes with AgentTrace.content()")
    return repr(value)


class AgentTrace:
    """One execution, one append-only index, host-owned evidence files."""

    def __init__(self, root: Path, *, context: dict[str, Any]) -> None:
        self.execution_id = uuid.uuid4().hex
        self.root = (root / self.execution_id).resolve()
        self.root.mkdir(parents=True, mode=0o700)
        (self.root / "objects").mkdir(mode=0o700)
        self.path = self.root / "events.jsonl"
        self._stream = self.path.open("x", encoding="utf-8")
        self.context = _json_safe(context)
        self._lock = threading.RLock()
        self._sequence = 0
        self._started = time.monotonic()
        self._closed = False
        self._container = None
        self._workdir: str | None = None
        self._state_lock = threading.Lock()
        self._bases: dict[str, dict] = {}
        self._last_state: str | None = None
        self._stop = threading.Event()
        self._collector: threading.Thread | None = None
        self._sources: dict[str, Path] = {}
        self._files: dict[str, tuple[int, int, int, str, int]] = {}
        self._collection_errors: set[str] = set()
        self._remote_sources: tuple[str, ...] = ()
        self._remote_hashes: dict[str, str] = {}
        self._workspace_state: str | None = None
        self._recorder_spool = f"/tmp/pitbench-recording-{self.execution_id}"
        self._native_spool = f"/tmp/pitbench-native-recording-{self.execution_id}"
        self._recorder_after: dict[str, int] = {}
        self._harvest_lock = threading.RLock()
        self._native_calls: dict[tuple[str, str, str], dict] = {}
        self._native_events: set[str] = set()
        self._active_calls: set[str] = set()
        self._workspace_events: dict[str, str | None] = {}
        self._call_states: dict[str, dict] = {}
        self._watcher: threading.Thread | None = None
        self._source_lock = threading.RLock()
        self._transcript_calls: dict[tuple, str] = {}
        self._transcript_results: set[tuple] = set()
        self.native_inbox = self.root / "native-inbox"
        self.native_inbox.mkdir(mode=0o700)

    @property
    def latest_state_id(self) -> str | None:
        return self._last_state

    @property
    def collection_gap_count(self) -> int:
        return len(self._collection_errors)

    @contextmanager
    def activate(self):
        token = _CURRENT.set(self)
        try:
            yield self
        finally:
            _CURRENT.reset(token)

    def content(self, value: bytes | str) -> dict[str, Any]:
        data = value.encode("utf-8") if isinstance(value, str) else value
        digest = hashlib.sha256(data).hexdigest()
        path = self.root / "objects" / digest
        with self._lock:
            if not path.exists():
                with path.open("xb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        return {"path": f"objects/{digest}", "sha256": digest, "bytes": len(data)}

    def record(self, event: str, *, call_id: str | None = None, **data) -> str:
        with self._lock:
            if self._closed:
                raise RuntimeError("agent trace is already closed")
            self._sequence += 1
            event_id = f"{self.execution_id}:{self._sequence}"
            payload = json.dumps(data, ensure_ascii=True, default=_encode)
            row = {
                "schema_version": 1,
                "event_id": event_id,
                "sequence": self._sequence,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "elapsed_sec": time.monotonic() - self._started,
                "execution_id": self.execution_id,
                **self.context,
                "event": event,
                "call_id": call_id if call_id is not None else _CALL.get(),
                "data": self.content(payload),
            }
            self._stream.write(json.dumps(row, ensure_ascii=True) + "\n")
            self._stream.flush()
            os.fsync(self._stream.fileno())
            return event_id

    @contextmanager
    def call(self, kind: str, *, name: str, inputs: Any, checkpoint=False):
        call_id = uuid.uuid4().hex
        parent_call_id = _CALL.get()
        self.record(
            f"{kind}.started",
            call_id=call_id,
            parent_call_id=parent_call_id,
            name=name,
            inputs=inputs,
        )
        token = _CALL.set(call_id)
        started = time.monotonic()
        before_state = None
        failed = False
        with self._lock:
            overlapping = sorted(self._active_calls)
            self._active_calls.add(call_id)
        with self.activate():
            try:
                if checkpoint:
                    before_state = self.checkpoint("before_call", call_id=call_id)
                yield call_id
            except BaseException as error:
                failed = True
                self.record(f"{kind}.failed", name=name, error=error)
                raise
            finally:
                try:
                    after_state = (
                        self.checkpoint("after_call", call_id=call_id)
                        if checkpoint
                        else None
                    )
                    self.record(
                        f"{kind}.finished",
                        name=name,
                        elapsed_sec=time.monotonic() - started,
                        status="failed" if failed else "returned",
                        before_state_id=before_state,
                        after_state_id=after_state,
                        overlapping_call_ids=overlapping,
                        boundary="instrumented_callable",
                    )
                    self._call_states[call_id] = {
                        "before_state_id": before_state,
                        "after_state_id": after_state,
                    }
                finally:
                    with self._lock:
                        self._active_calls.discard(call_id)
                    _CALL.reset(token)

    def recorder_command(self, action: str, *, native=False, **arguments) -> list[str]:
        command = [
            "python3",
            "-c",
            Path(recording.__file__).read_text(),
            action,
            "--spool",
            self._native_spool if native else self._recorder_spool,
        ]
        for name, value in arguments.items():
            if value is not None:
                command.extend(["--" + name.replace("_", "-"), str(value)])
        return command

    def install_native_hooks(
        self, container, *, config_path: str, provider: str, root: str | None
    ) -> None:
        command = self.recorder_command(
            "hook", native=True, provider=provider, root=root
        )
        installed = container.exec_run(
            self.recorder_command(
                "install",
                native=True,
                provider=provider,
                config=config_path,
                command=shlex.join(command),
            )
        )
        if installed.exit_code != 0:
            raise RuntimeError(
                f"could not install {provider} recording hooks: {installed.output!r}"
            )
        self.record(
            "instrumentation.hooks_installed",
            provider=provider,
            config_path=config_path,
        )

    def _capture_workspace(self, reason: str, call_id: str | None) -> str | None:
        try:
            result = self._container.exec_run(
                self.recorder_command(
                    "snapshot",
                    root=self._workdir or ".",
                    reason=reason,
                    call_id=call_id,
                ),
                workdir=self._workdir,
            )
            if result.exit_code != 0:
                raise RuntimeError(f"workspace capture failed: {result.output!r}")
            expected = json.loads(result.output)["id"]
            self.harvest_recorder(native=False)
            return self._workspace_events.get(expected)
        except Exception as error:
            self._gap("workspace", error)
            return None

    def harvest_recorder(self, *, native=True) -> dict[str, str | None]:
        if self._container is None:
            return {}
        spool = self._native_spool if native else self._recorder_spool
        with self._harvest_lock:
            result = self._container.exec_run(
                self.recorder_command(
                    "export", native=native, after=self._recorder_after.get(spool, 0)
                ),
                workdir=self._workdir,
            )
            if result.exit_code != 0 or not isinstance(result.output, bytes):
                raise RuntimeError("could not export native recorder evidence")
            pending = []
            with tarfile.open(fileobj=io.BytesIO(result.output)) as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    data = archive.extractfile(member).read()
                    if member.name.startswith("objects/"):
                        reference = self.content(data)
                        if member.name != reference["path"]:
                            raise ValueError(
                                "native recorder object failed its checksum"
                            )
                    elif member.name.startswith("event-") and "/" not in member.name:
                        pending.append(json.loads(data))
            for event in sorted(pending, key=lambda value: value["completed_ns"]):
                self.ingest_hook(event)
                self._recorder_after[spool] = event["completed_ns"]
            return {
                event["id"]: (event.get("workspace") or {}).get("state_id")
                for event in pending
            }

    def ingest_hook(self, event: dict) -> None:
        with self._harvest_lock:
            self._ingest_hook(event)

    def _ingest_hook(self, event: dict) -> None:
        if event["id"] in self._native_events:
            return
        payload = event["payload"]
        provider = event["provider"]
        if provider == "filesystem":
            self.record("filesystem.mutation", native=event)
            if event.get("capture_error") or payload.get("operation") in {
                "overflow",
                "watch_gap",
            }:
                self._gap(
                    "filesystem",
                    RuntimeError(event.get("capture_error") or str(payload)),
                )
            self._native_events.add(event["id"])
            return
        workspace = event.get("workspace")
        state_id = None
        targets = event.get("tool_targets")
        if targets:
            self.record("tool.target_snapshot", producer=provider, **targets)
            state_id = targets["state_id"]
        if workspace is not None:
            for item in workspace["entries"].values():
                if (
                    item.get("kind") == "file"
                    and not (self.root / "objects" / item["sha256"]).is_file()
                ):
                    raise ValueError(
                        "workspace manifest refers to a missing content object"
                    )
            state_id = workspace["state_id"]
            self.record(
                "workspace.snapshot",
                call_id=payload.get("call_id"),
                producer=provider,
                reason=payload.get("reason") or payload.get("hook_event_name"),
                **workspace,
            )
            for error in workspace["errors"]:
                self._gap("workspace.file", RuntimeError(str(error)))
            if provider == "pitbench":
                self._workspace_state = state_id
                self._workspace_events[event["id"]] = state_id
        if event.get("capture_error"):
            self._gap("native.workspace", RuntimeError(event["capture_error"]))
        if provider != "pitbench":
            self.record("native.hook", producer=provider, native=event)
            event_name = (
                event.get("hook_event")
                or payload.get("hook_event_name")
                or payload.get("event")
                or ""
            ).lower()
            session_id = str(
                payload.get("session_id")
                or payload.get("conversation_id")
                or payload.get("conversationId")
                or ""
            )
            tool_id = payload.get("tool_use_id") or payload.get("tool_call_id")
            if tool_id is None:
                tool_id = payload.get("stepIdx")
            native_call = payload.get("toolCall") or {}
            tool_name = payload.get("tool_name") or native_call.get("name")
            tool_input = payload.get("tool_input", native_call.get("args"))
            key = (provider, session_id, str(tool_id))
            if event_name in {"pretooluse", "beforetool"}:
                call_id = uuid.uuid4().hex
                if tool_id is None:
                    key = (provider, session_id, event["id"])
                self._native_calls[key] = {
                    "call_id": call_id,
                    "state": state_id,
                    "tool_id": tool_id,
                    "name": tool_name,
                    "input": tool_input,
                }
                self._transcript_calls[(provider, session_id, tool_id)] = call_id
                self.record(
                    "tool.started",
                    call_id=call_id,
                    producer=provider,
                    boundary="native_hook",
                    provider_call_id=tool_id,
                    provider_session_id=session_id,
                    name=tool_name,
                    inputs=tool_input,
                    before_state_id=state_id,
                )
            elif event_name in {"posttooluse", "aftertool", "posttoolusefailure"}:
                previous = (
                    self._native_calls.pop(key, None) if tool_id is not None else None
                )
                correlation = "provider_call_id"
                if tool_id is None:
                    matches = [
                        candidate
                        for candidate, value in self._native_calls.items()
                        if candidate[:2] == (provider, session_id)
                        and value["name"] == tool_name
                        and value["input"] == tool_input
                    ]
                    if len(matches) == 1:
                        previous = self._native_calls.pop(matches[0])
                    correlation = (
                        "unique_arguments_without_provider_id"
                        if previous
                        else "unresolved"
                    )
                call_id = previous["call_id"] if previous else uuid.uuid4().hex
                result_present = any(
                    key in payload
                    for key in ("tool_response", "tool_result", "toolResult")
                ) or bool(payload.get("error"))
                response = payload.get(
                    "tool_response",
                    payload.get("tool_result", payload.get("toolResult")),
                )
                backend_id = previous.get("backend_call_id") if previous else None
                if isinstance(response, dict):
                    backend_id = (
                        response.get("_meta", {}).get("pitbench", {}) or {}
                    ).get("call_id") or backend_id
                backend_states = self._call_states.get(backend_id, {})
                self.record(
                    "tool.result",
                    call_id=call_id,
                    producer=provider,
                    provider_call_id=tool_id,
                    result=payload.get(
                        "tool_response",
                        payload.get(
                            "tool_result",
                            payload.get("toolResult", payload.get("error")),
                        ),
                    ),
                    result_present=result_present,
                    visibility="native_hook_payload",
                )
                self.record(
                    "tool.finished",
                    call_id=call_id,
                    producer=provider,
                    boundary="native_hook",
                    status="failed"
                    if event_name == "posttoolusefailure"
                    else "returned",
                    before_state_id=backend_states.get("before_state_id")
                    or (previous["state"] if previous else None),
                    after_state_id=backend_states.get("after_state_id") or state_id,
                    missing_request=previous is None,
                    provider_call_id=tool_id,
                    correlation=correlation,
                    backend_call_id=backend_id,
                )
            elif event_name in {"subagentstart", "subagentstop"}:
                self.record("agent.child", producer=provider, native=payload)
            transcript = event.get("transcript")
            if provider == "antigravity" and transcript is not None:
                for line in (
                    (self.root / "objects" / transcript["sha256"])
                    .read_text()
                    .splitlines()
                ):
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    index = row.get("step_index")
                    result_key = (provider, session_id, index)
                    call_id = self._transcript_calls.get(result_key)
                    if (
                        call_id
                        and row.get("status") in {"DONE", "ERROR"}
                        and "content" in row
                        and result_key not in self._transcript_results
                    ):
                        self.record(
                            "tool.result",
                            call_id=call_id,
                            producer=provider,
                            result=row["content"],
                            result_present=True,
                            visibility="native_transcript",
                            provider_step_index=index,
                            transcript=transcript,
                        )
                        self._transcript_results.add(result_key)
        self._native_events.add(event["id"])

    def bind_native_call(
        self, name: str, arguments: dict | None, backend_call_id: str
    ) -> None:
        # Pre hooks persist before the CLI sends the MCP request. Read those
        # records at receipt, without relying on the background polling interval.
        self.collect_files()
        with self._harvest_lock:
            matches = []
            for call in self._native_calls.values():
                if call.get("backend_call_id"):
                    continue
                inputs = call["input"] or {}
                if call["name"] == f"mcp__pitbench__{name}" and inputs == (
                    arguments or {}
                ):
                    matches.append(call)
                elif (
                    call["name"] == "call_mcp_tool"
                    and inputs.get("ServerName") == "pitbench"
                    and inputs.get("ToolName") == name
                ):
                    supplied = inputs.get("Arguments", {})
                    if isinstance(supplied, str):
                        try:
                            supplied = json.loads(supplied)
                        except json.JSONDecodeError:
                            continue
                    if supplied == (arguments or {}):
                        matches.append(call)
            if len(matches) == 1:
                matches[0]["backend_call_id"] = backend_call_id
                self.record(
                    "mcp.native_binding",
                    call_id=backend_call_id,
                    native_call_id=matches[0]["call_id"],
                    method="unique_active_hook_and_exact_arguments",
                )
            elif len(matches) > 1:
                self.record(
                    "mcp.native_binding",
                    call_id=backend_call_id,
                    candidates=[call["call_id"] for call in matches],
                    method="ambiguous; not_bound",
                )

    def _gap(self, source: str, error: Exception) -> None:
        # Keep a concrete failure visible without flooding a long-running trace.
        key = f"{source}:{type(error).__name__}:{error}"
        if key not in self._collection_errors:
            self._collection_errors.add(key)
            self.record("collection.gap", source=source, error=error)

    def _git(self, *args: str, allowed=(0,)) -> bytes:
        result = self._container.exec_run(
            ["timeout", "10s", "git", "--no-optional-locks", *args],
            workdir=self._workdir,
            demux=True,
        )
        if isinstance(result.output, tuple):
            output, error = result.output
        else:
            output, error = result.output, b""
        if result.exit_code not in allowed:
            detail = (error or output or b"").decode(errors="replace")
            raise RuntimeError(f"git {args[0]} exited {result.exit_code}: {detail}")
        if not isinstance(output, (bytes, type(None))):
            raise TypeError("repository capture requires byte output")
        return output or b""

    def checkpoint(self, reason: str, *, call_id: str | None = None) -> str | None:
        if self._container is None:
            return
        with self._state_lock:
            started = time.monotonic()
            try:
                head = self._git("rev-parse", "HEAD").decode().strip()
                if head not in self._bases:
                    self._bases[head] = self.content(
                        archive_repository_head(
                            self._container,
                            workdir=self._workdir,
                            head=head,
                        )
                    )
                patch_ref = self.content(
                    capture_repository_patch(
                        self._container,
                        workdir=self._workdir,
                        timeout_sec=10,
                    )
                )
                state_id = hashlib.sha256(
                    f"{head}:{patch_ref['sha256']}".encode()
                ).hexdigest()
                self.record(
                    "repository.checkpoint",
                    call_id=call_id,
                    reason=reason,
                    state_id=state_id,
                    previous_state_id=self._last_state,
                    head=head,
                    base_archive=self._bases[head],
                    patch=patch_ref,
                    consistency="non_atomic_observation",
                    scope="tracked and non-ignored untracked files; excludes .pitbench",
                    capture_elapsed_sec=time.monotonic() - started,
                )
                self._last_state = state_id
            except Exception as error:
                self._gap("repository", error)
            if reason != "sampled":
                return self._capture_workspace(reason, call_id)
        return None

    def collect_files(self) -> None:
        with self._source_lock:
            self._collect_files()

    def _collect_files(self) -> None:
        for label, root in self._sources.items():
            if label == "native-hooks":
                for path in (root / "objects").glob("*"):
                    if path.is_symlink() or not path.is_file() or len(path.name) != 64:
                        continue
                    if not (self.root / "objects" / path.name).exists():
                        reference = self.content(path.read_bytes())
                        if reference["sha256"] != path.name:
                            raise ValueError("native hook object failed its checksum")
            for path in sorted(root.rglob("*")) if root.is_dir() else []:
                if path.is_symlink() or not path.is_file():
                    continue
                name = f"{label}/{path.relative_to(root)}"
                try:
                    stat = path.stat()
                    previous = self._files.get(name)
                    if (
                        previous
                        and (stat.st_ino, stat.st_mtime_ns, stat.st_size)
                        == previous[:3]
                    ):
                        continue
                    offset, generation = (
                        (previous[2], previous[4]) if previous else (0, 0)
                    )
                    with path.open("rb") as stream:
                        # Detect replacement, truncation and rewrites, including
                        # rewritten JSON documents which happen to grow in size.
                        digest = hashlib.sha256()
                        remaining = offset
                        while remaining > 0:
                            chunk = stream.read(min(remaining, 64 * 1024))
                            if not chunk:
                                break
                            digest.update(chunk)
                            remaining -= len(chunk)
                        prefix_hash = digest.hexdigest()
                        reset = previous and (
                            stat.st_ino != previous[0]
                            or stat.st_size <= offset
                            or prefix_hash != previous[3]
                        )
                        if reset:
                            offset, generation = 0, generation + 1
                            digest = hashlib.sha256()
                            self.record(
                                "source.reset", source=name, generation=generation
                            )
                        stream.seek(offset)
                        while chunk := stream.read(64 * 1024):
                            self.record(
                                "source.chunk",
                                source=name,
                                generation=generation,
                                offset=offset,
                                content=self.content(chunk),
                                visibility="provider_artifact; model_visibility_unspecified",
                            )
                            offset += len(chunk)
                            digest.update(chunk)
                        prefix_hash = digest.hexdigest()
                    if (
                        label == "native-hooks"
                        and path.name.startswith("event-")
                        and path.suffix == ".json"
                    ):
                        self.ingest_hook(json.loads(path.read_text()))
                    self._files[name] = (
                        stat.st_ino,
                        stat.st_mtime_ns,
                        offset,
                        prefix_hash,
                        generation,
                    )
                except OSError as error:
                    self._gap(name, error)

    def collect_remote_files(self) -> None:
        # Remote Docker logs are copied to the host only at container teardown.
        # Preserve their archives during execution as well, without extracting
        # any provider-controlled paths onto the host.
        for source in self._remote_sources:
            try:
                chunks, _ = self._container.get_archive(source)
                content = self.content(b"".join(chunks))
                if self._remote_hashes.get(source) != content["sha256"]:
                    self.record(
                        "source.archive", source=source, content=content, format="tar"
                    )
                    self._remote_hashes[source] = content["sha256"]
            except Exception as error:
                self._gap(source, error)

    def start_collection(
        self,
        *,
        container,
        workdir,
        sources,
        snapshot_interval=1.0,
        remote_sources=(),
    ):
        self._container, self._workdir = container, workdir
        self._sources = sources
        self._sources["native-hooks"] = self.native_inbox
        self._remote_sources = remote_sources
        self.checkpoint("initial")
        self.start_filesystem_journal()

        def collect():
            next_snapshot = time.monotonic() + snapshot_interval
            while not self._stop.wait(0.25):
                try:
                    self.collect_files()
                    if time.monotonic() >= next_snapshot:
                        self.harvest_recorder()
                        self.harvest_recorder(native=False)
                        self.collect_remote_files()
                        self.checkpoint("sampled")
                        next_snapshot = time.monotonic() + snapshot_interval
                except Exception as error:
                    self._gap("collector", error)

        self._collector = threading.Thread(
            target=collect,
            name=f"agent-trace-{self.execution_id[:8]}",
            daemon=True,
        )
        self._collector.start()

    def start_filesystem_journal(self) -> None:
        api = getattr(getattr(self._container, "client", None), "api", None)
        if api is None:
            self.record(
                "instrumentation.filesystem",
                available=False,
                reason="container streaming API unavailable",
            )
            return
        ready = threading.Event()

        def watch():
            try:
                created = api.exec_create(
                    self._container.id,
                    self.recorder_command("watch", root=self._workdir or "."),
                    workdir=self._workdir,
                    stdout=True,
                    stderr=True,
                )
                stream = api.exec_start(created["Id"], stream=True, demux=True)
                try:
                    for chunk in stream:
                        stdout, stderr = (
                            chunk if isinstance(chunk, tuple) else (chunk, None)
                        )
                        if stdout and b"READY" in stdout:
                            self.record(
                                "instrumentation.filesystem",
                                available=True,
                                mechanism="inotify",
                            )
                            ready.set()
                        if stderr:
                            self._gap(
                                "filesystem.watcher",
                                RuntimeError(stderr.decode(errors="replace")),
                            )
                finally:
                    close_event_stream(stream)
            except Exception as error:
                self._gap("filesystem.watcher", error)
            finally:
                ready.set()

        self._watcher = threading.Thread(
            target=watch, name="pitbench-filesystem-journal", daemon=True
        )
        self._watcher.start()
        if not ready.wait(10):
            self._gap(
                "filesystem.watcher",
                RuntimeError("watcher startup did not acknowledge readiness"),
            )

    def finish_collection(self) -> None:
        self._stop.set()
        if self._collector is not None:
            self._collector.join()
        if self._watcher is not None:
            self._container.exec_run(self.recorder_command("stop-watch"))
            self._watcher.join(timeout=10)
            if self._watcher.is_alive():
                self._gap("filesystem.watcher", RuntimeError("watcher did not stop"))
        self.collect_files()
        self.collect_remote_files()
        self.harvest_recorder()
        self.harvest_recorder(native=False)
        self.checkpoint("final")

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._stream.close()


def traced_call(kind: str, *, checkpoint: bool = False):
    """Observe a shared synchronous tool boundary, including rejected inputs."""

    def decorate(function):
        signature = inspect.signature(function)

        @wraps(function)
        def wrapped(self, *args, **kwargs):
            trace = getattr(self, "_agent_trace", None) or current_trace()
            if trace is None:
                return function(self, *args, **kwargs)
            try:
                inputs = signature.bind(self, *args, **kwargs)
                inputs.apply_defaults()
                arguments = {
                    key: value
                    for key, value in inputs.arguments.items()
                    if key != "self"
                }
            except TypeError:
                arguments = {"args": args, "kwargs": kwargs}
            with trace.call(
                kind,
                name=function.__qualname__,
                inputs=arguments,
                checkpoint=checkpoint or kind in {"tool", "terminal"},
            ):
                cancelled = getattr(self, "_trace_cancelled", None)
                if (
                    function.__name__ == "send_keys"
                    and cancelled is not None
                    and cancelled.is_set()
                ):
                    raise RuntimeError("agent terminal was cancelled")
                result = function(self, *args, **kwargs)
                trace.record(
                    f"{kind}.result", result=result, visibility="returned_to_caller"
                )
                return result

        return wrapped

    return decorate


def model_completion(function, **kwargs):
    """Capture the actual request after adapter transformations, without auth."""
    trace = current_trace()
    if trace is None:
        return function(**kwargs)
    request = {
        key: value
        for key, value in kwargs.items()
        if key not in {"api_key", "headers", "api_base", "logger_fn"}
    }
    with trace.call("model", name="completion", inputs=request):
        response = function(**kwargs)
        trace.record("model.result", response=response)
        return response


def read_trace(path: Path):
    """Read a live or interrupted index and resolve its event payloads.

    Ignore an unfinished last line, as a killed writer may leave one. Complete
    corrupt lines and missing/corrupt payloads raise instead of losing evidence.
    Raw content references in the payload remain references, avoiding eagerly
    loading complete process outputs or source archives during analysis.
    """
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.endswith("\n"):
                return
            row = json.loads(line)
            ref = row["data"]
            digest = ref["sha256"]
            if len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest
            ):
                raise ValueError("invalid agent trace object digest")
            data = (path.parent / "objects" / digest).read_bytes()
            if hashlib.sha256(data).hexdigest() != digest or len(data) != ref["bytes"]:
                raise ValueError(f"corrupt agent trace object: {digest}")
            row["data"] = json.loads(data)
            yield row
