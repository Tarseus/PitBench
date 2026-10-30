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
import shutil
import tarfile
import threading
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from pitbench.harness.utils import recording
from pitbench.harness.utils.pipeline_trace import _json_safe
from pitbench.harness.utils.repository import capture_repository_patch

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
        self._last_state: str | None = None
        self._stop = threading.Event()
        self._collector: threading.Thread | None = None
        self._sources: dict[str, Path] = {}
        self._collection_errors: set[str] = set()
        self._remote_sources: tuple[str, ...] = ()
        self._remote_hashes: dict[str, str] = {}
        self._recorder_spool = f"/tmp/pitbench-recording-{self.execution_id}"
        self._native_spool = f"/tmp/pitbench-native-recording-{self.execution_id}"
        self._recorder_after: dict[str, int] = {}
        self._harvest_lock = threading.RLock()
        self._native_calls: dict[tuple[str, str, str], dict] = {}
        self._native_events: set[str] = set()
        self._filesystem_operations: Counter[str] = Counter()
        self._active_calls: set[str] = set()
        self._call_states: dict[str, dict] = {}
        self._call_results: dict[str, Any] = {}
        self._watcher: threading.Thread | None = None
        self._source_lock = threading.RLock()
        self._transcript_calls: dict[tuple, str] = {}
        self._transcript_results: set[tuple] = set()
        self.native_inbox = self.root / "native-inbox"
        self.native_inbox.mkdir(mode=0o700)
        self._research_budget = self.context.get("research_budget")
        self._research_cpu_start: int | None = None
        self._research_cpu_sec: float | None = None
        self._research_memory_bytes: int | None = None
        self._research_checkpoints: set[float] = set()
        self._research_stats_available = False
        self._research_cpu_baseline_elapsed_sec: float | None = None
        self._research_stats_errors = 0

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
            effective_call_id = call_id if call_id is not None else _CALL.get()
            if event.endswith(".result") and effective_call_id is not None:
                self._call_results[effective_call_id] = _json_safe(
                    data.get("result", data)
                )
            row = {
                "schema_version": 1,
                "event_id": event_id,
                "sequence": self._sequence,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "elapsed_sec": time.monotonic() - self._started,
                "execution_id": self.execution_id,
                **self.context,
                "event": event,
                "call_id": effective_call_id,
                "data": self.content(payload),
            }
            self._stream.write(json.dumps(row, ensure_ascii=True) + "\n")
            self._stream.flush()
            os.fsync(self._stream.fileno())
            return event_id

    def record_if_open(
        self,
        event: str,
        *,
        call_id: str | None = None,
        raw_content: bytes | None = None,
        **data,
    ) -> str | None:
        """Record a late asynchronous event only while collection is active."""
        with self._lock:
            if self._closed:
                return None
            if raw_content is not None:
                data["content"] = self.content(raw_content)
            return self.record(event, call_id=call_id, **data)

    def record_contents_if_open(
        self,
        event: str,
        *,
        call_id: str | None = None,
        contents: dict[str, bytes],
        **data,
    ) -> str | None:
        """Record several raw values atomically while the trace remains open."""
        with self._lock:
            if self._closed:
                return None
            data.update({name: self.content(value) for name, value in contents.items()})
            return self.record(event, call_id=call_id, **data)

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
            return {event["id"]: self.latest_state_id for event in pending}

    def ingest_hook(self, event: dict) -> None:
        with self._harvest_lock:
            self._ingest_hook(event)

    def _ingest_hook(self, event: dict) -> None:
        if event["id"] in self._native_events:
            return
        payload = event["payload"]
        provider = event["provider"]
        if provider == "filesystem":
            operation = str(payload.get("operation", "unknown"))
            self._filesystem_operations[operation] += 1
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
        state_id = self.latest_state_id
        if event.get("capture_error"):
            self._gap("native.hook", RuntimeError(event["capture_error"]))
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
                state_id = self.checkpoint("native_tool", call_id=call_id)
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
                    recording.read_transcript(self.root / "objects", transcript)
                    .decode()
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

    def close_bound_native_calls(self) -> None:
        """Recover missing provider post-hooks from exact bound backend results."""
        with self._harvest_lock:
            completed = []
            for key, native_call in self._native_calls.items():
                backend_id = native_call.get("backend_call_id")
                if (
                    backend_id not in self._call_states
                    or backend_id not in self._call_results
                ):
                    continue
                backend_states = self._call_states[backend_id]
                result = self._call_results[backend_id]
                failed = bool(
                    isinstance(result, dict)
                    and (result.get("isError") or result.get("is_error"))
                )
                call_id = native_call["call_id"]
                self.record(
                    "tool.result",
                    call_id=call_id,
                    producer=key[0],
                    provider_call_id=native_call.get("tool_id"),
                    result=result,
                    result_present=True,
                    visibility="exact_bound_backend_result",
                )
                self.record(
                    "tool.finished",
                    call_id=call_id,
                    producer=key[0],
                    boundary="native_hook_with_backend_fallback",
                    status="failed" if failed else "returned",
                    before_state_id=backend_states.get("before_state_id"),
                    after_state_id=backend_states.get("after_state_id"),
                    missing_request=False,
                    provider_call_id=native_call.get("tool_id"),
                    correlation="exact_bound_backend_result_without_post_hook",
                    backend_call_id=backend_id,
                )
                self.record(
                    "instrumentation.recovered_completion",
                    native_call_id=call_id,
                    backend_call_id=backend_id,
                    provider=key[0],
                    reason="provider post-tool hook was not emitted",
                )
                completed.append(key)
            for key in completed:
                self._native_calls.pop(key, None)

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
                if state_id != self._last_state:
                    self.record(
                        "repository.checkpoint",
                        call_id=call_id,
                        reason=reason,
                        state_id=state_id,
                        previous_state_id=self._last_state,
                        head=head,
                        patch=patch_ref,
                        consistency="non_atomic_observation",
                        scope="tracked and non-ignored untracked files; excludes .pitbench",
                        capture_elapsed_sec=time.monotonic() - started,
                    )
                self._last_state = state_id
            except Exception as error:
                self._gap("repository", error)
                return None
            return state_id

    def collect_research_resources(self) -> None:
        """Host-owned whole-container counters; never infer experiment CPU."""
        if not self._research_budget:
            return
        elapsed = time.monotonic() - self._started
        try:
            sample = self._container.stats(stream=False, one_shot=True)
            total_ns = sample["cpu_stats"]["cpu_usage"]["total_usage"]
            if type(total_ns) is not int or total_ns < 0:
                raise ValueError("invalid Docker CPU counter")
            if self._research_cpu_start is None:
                self._research_cpu_start = total_ns
                self._research_cpu_baseline_elapsed_sec = (
                    time.monotonic() - self._started
                )
            if total_ns < self._research_cpu_start:
                raise ValueError("Docker CPU counter moved backwards")
            self._research_cpu_sec = (total_ns - self._research_cpu_start) / 1e9
            memory = sample.get("memory_stats", {}).get("usage")
            if type(memory) is int and memory >= 0:
                self._research_memory_bytes = max(
                    self._research_memory_bytes or 0, memory
                )
            self._research_stats_available = True
            self.record(
                "research.resources",
                elapsed_sec=elapsed,
                sandbox_cpu_sec=self._research_cpu_sec,
                sandbox_memory_bytes=memory,
                source="Docker daemon cumulative CPU delta; sampled memory",
            )
        except Exception as error:
            self._research_stats_errors += 1
            self._gap("research.resource_accounting", error)
        for target in self._research_budget.get("checkpoints_sec", []):
            if target in self._research_checkpoints or elapsed < target:
                continue
            state = self.checkpoint("research_budget_checkpoint")
            observed = time.monotonic() - self._started
            self.record(
                "research.checkpoint",
                target_elapsed_sec=target,
                observed_elapsed_sec=observed,
                capture_lateness_sec=observed - target,
                state_id=state,
                status="captured" if state else "capture_failed",
                consistency="non_atomic_observation_at_observed_time",
                hidden_feedback=False,
            )
            self._research_checkpoints.add(target)

    def record_research_summary(self, *, result=None, error=None) -> None:
        if not self._research_budget:
            return
        summary = {
            "budget": self._research_budget,
            "observed_wall_time_sec": time.monotonic() - self._started,
            "sandbox_cpu_time_sec": self._research_cpu_sec,
            "sampled_sandbox_memory_max_bytes": self._research_memory_bytes,
            "stats_available": self._research_stats_available,
            "stats_collection_errors": self._research_stats_errors,
            "cpu_baseline_elapsed_sec": self._research_cpu_baseline_elapsed_sec,
            "development_experiment_cpu_time_sec": None,
            "compile_cpu_time_sec": None,
            "model_host_or_sidecar_cpu_time_sec": None,
            "resource_enforcement": {
                "wall_time": "harness_deadline",
                "cpus_and_memory": "container_configuration_when_declared",
                "total_cpu_time": "recording_only_no_default_cap",
                "development_cpu_time": "recording_only_attribution_unavailable",
                "development_parallel_runs": "scheduler_preference_within_container_cpu_limit",
                "max_parallel_runs": "no_default_independent_quota",
            },
            "llm_usage": {
                "input_tokens": getattr(result, "total_input_tokens", None),
                "output_tokens": getattr(result, "total_output_tokens", None),
                "reported_cost_usd": getattr(result, "total_cost", None),
                "requests": None,
                "coverage": "AgentResult and retained provider evidence; defaults are not independent usage verification",
            },
            "unreached_checkpoints_sec": [
                point
                for point in self._research_budget.get("checkpoints_sec", [])
                if point not in self._research_checkpoints
            ],
            "missing_checkpoint_reason": "execution_ended_before_successful_scheduled_collection",
            "error": str(error) if error else None,
            "hidden_judge_included": False,
        }
        path = self.root / "research-budget.json"
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        self.record("research.budget_summary", summary=summary)

    def collect_files(self) -> None:
        with self._source_lock:
            self._collect_files()

    def _collect_files(self) -> None:
        """Ingest native hook events without copying provider logs into the trace.

        Agent logs and terminal recordings already live beside the trace in the trial
        artifacts. Copying each growing log in 64 KiB chunks duplicates those files
        and turns a complete action record into a repository-sized archive.
        """
        for path in sorted(self.native_inbox.glob("event-*.json")):
            try:
                self.ingest_hook(json.loads(path.read_text()))
            except (OSError, json.JSONDecodeError) as error:
                self._gap(f"native-hooks/{path.name}", error)

    def _record_local_artifacts(self) -> None:
        """Index retained provider logs without storing a second copy."""
        for source, root in self._sources.items():
            for path in sorted(root.rglob("*")) if root.is_dir() else []:
                if path.is_symlink() or not path.is_file():
                    continue
                try:
                    digest = hashlib.sha256()
                    with path.open("rb") as stream:
                        while chunk := stream.read(1024 * 1024):
                            digest.update(chunk)
                    self.record(
                        "source.artifact",
                        source=source,
                        path=str(path.relative_to(root)),
                        bytes=path.stat().st_size,
                        sha256=digest.hexdigest(),
                        retained_at=str(path),
                    )
                except OSError as error:
                    self._gap(f"{source}/{path.name}", error)

    def _record_filesystem_activity(self) -> None:
        if self._filesystem_operations:
            self.record(
                "filesystem.activity",
                events_observed=sum(self._filesystem_operations.values()),
                by_operation=dict(sorted(self._filesystem_operations.items())),
            )

    def collect_remote_files(self) -> None:
        # Remote Docker logs are unavailable on the host, so retain one final
        # archive. Local logs are indexed by _record_local_artifacts instead.
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
        self._sources = dict(sources)
        self._remote_sources = remote_sources
        self.collect_research_resources()
        self.checkpoint("initial")
        self.start_filesystem_journal()

        def collect():
            next_snapshot = time.monotonic() + snapshot_interval
            while not self._stop.wait(0.25):
                try:
                    self.collect_files()
                    if time.monotonic() >= next_snapshot:
                        self.collect_research_resources()
                        self.harvest_recorder()
                        self.harvest_recorder(native=False)
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
        self.harvest_recorder()
        self.harvest_recorder(native=False)
        self.checkpoint("final")
        self.close_bound_native_calls()
        self.collect_remote_files()
        self._record_local_artifacts()
        self._record_filesystem_activity()
        self.collect_research_resources()
        shutil.rmtree(self.native_inbox, ignore_errors=True)

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
