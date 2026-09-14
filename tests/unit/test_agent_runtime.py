from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from typer.testing import CliRunner

from pitbench.cli import auth
from pitbench.cli.doctor import CheckStatus, DoctorCheck
from pitbench.cli.main import app
from pitbench.harness.agents.agent_factory import AgentFactory
from pitbench.harness.agents.base_agent import AgentResult, BaseAgent
from pitbench.harness.agents.codex_mcp_agent import CodexMCPAgent
from pitbench.harness.agents.codex_relay import _RelayHandler
from pitbench.harness.agents.containers import CodexContainerRunner
from pitbench.harness.agents.host_mcp import LoopbackMCPServer, TaskTerminal
from pitbench.harness.agents.mcp_agents.goose_mcp_agent import GooseMCPAgent
from pitbench.harness.agents.null_agent import NopAgent
from pitbench.harness.agents.process_agent import ManagedProcessAgent
from pitbench.harness.agents.profiles import CodexProfile, ProfileError
from pitbench.harness.harness.harness import Harness
from pitbench.harness.harness.run_artifacts import RunArtifactStore
from pitbench.harness.utils import recording
from pitbench.harness.utils.agent_trace import (
    AgentTrace,
    current_trace,
    model_completion,
    read_trace,
)
from pitbench.harness.utils.trace_validation import inspect_trace

# Tests consolidated from tests/unit/utils/test_agent_trace.py


def events(root):
    rows = []
    for line in (root / "events.jsonl").read_text().splitlines():
        row = json.loads(line)
        ref = row["data"]
        data = (root / ref["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == ref["sha256"]
        row["data"] = json.loads(data)
        rows.append(row)
    return rows


class LocalContainer:
    """Exercise the actual Git/shell commands without requiring Docker."""

    def __init__(self, workdir):
        self.id = "local-test-container"
        self.attrs = {"Config": {"WorkingDir": str(workdir)}, "Image": "test-image"}

    def exec_run(self, command, *, workdir=None, demux=False, **kwargs):
        result = subprocess.run(command, cwd=workdir, capture_output=True, timeout=15)
        return SimpleNamespace(
            exit_code=result.returncode,
            output=(result.stdout, result.stderr) if demux else result.stdout,
        )


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "code.txt").write_text("before\n")
    (repo / "deleted.txt").write_text("delete me\n")
    (repo / ".gitignore").write_text("build/\n")
    (repo / ".gitattributes").write_text(
        "deleted.txt export-ignore\nsubst.txt export-subst\n"
    )
    (repo / "subst.txt").write_text("$Format:%H$\n")
    (repo / "binary.dat").write_bytes(b"\0\xff\x01")
    (repo / "link.txt").symlink_to("code.txt")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "initial",
        ],
        check=True,
    )
    return repo, LocalContainer(repo)


def run_agent(agent, container, logdir):
    harness = Harness.__new__(Harness)
    harness._run_id = "run-1"
    harness._remote_build = False
    return asyncio.run(
        harness._run_agent_with_timeout(
            trial_handler=SimpleNamespace(
                task_id="task-1",
                trial_name=logdir.parent.name,
                instruction="Improve this code",
            ),
            session=SimpleNamespace(container=container),
            logging_dir=logdir,
            timeout_sec=10,
            agent=agent,
            portkey_metadata=None,
            portkey_trace_id=None,
        )
    )


def test_shared_profile_and_container_runtime_validate_provider_contract(
    tmp_path, monkeypatch
):
    profile_root = tmp_path / "profile"
    content_root = profile_root / "codex-home"
    content_root.mkdir(parents=True)
    (profile_root / "profile.yaml").write_text(
        "schema_version: '1.0'\nname: test\ncodex_home: codex-home\n"
    )
    (content_root / "config.toml").write_text("model = 'test'\n")
    profile = CodexProfile.load(profile_root)

    codex = tmp_path / "codex"
    codex.write_text("")
    codex.chmod(0o755)
    (tmp_path / "codex-code-mode-host").write_text("")
    (tmp_path / "codex-code-mode-host").chmod(0o755)
    runner = CodexContainerRunner("runner-image", codex, profile)
    monkeypatch.setattr(runner, "validate_inputs", lambda: None)
    command = runner.command_prefix()
    assert command[-3:] == [
        "python3",
        "/opt/pitbench/container_entrypoint.py",
        "codex",
    ]
    assert "codex_container_runner.py" not in " ".join(command)
    assert runner.metadata()["profile"]["sha256"] == profile.sha256

    (profile_root / "profile.yaml").write_text(
        "schema_version: '1.0'\nname: wrong-provider\ngemini_config: codex-home\n"
    )
    with pytest.raises(ProfileError, match="unknown Codex profile fields"):
        CodexProfile.load(profile_root)


def test_managed_process_cancel_preserves_base_agent_cancellation():
    class ProcessAgent(ManagedProcessAgent):
        @staticmethod
        def name() -> str:
            return "process-test"

        def perform_task(self, **kwargs):
            return AgentResult()

    agent = ProcessAgent()
    agent.cancel()
    assert agent._trace_cancelled.is_set()


def test_trace_coverage_failure_does_not_replace_agent_failure(
    repository, tmp_path, monkeypatch
):
    agent = NopAgent()

    def fail(**kwargs):
        raise RuntimeError("original agent failure")

    agent.perform_task = fail
    monkeypatch.setattr(
        "pitbench.harness.utils.trace_validation.inspect_trace",
        Mock(side_effect=ValueError("coverage failure")),
    )
    with pytest.raises(RuntimeError, match="original agent failure"):
        BaseAgent.execute_task(
            agent,
            instruction="fail",
            session=SimpleNamespace(container=repository[1]),
            logging_dir=tmp_path / "agent-logs",
            trace_context={"run_id": "r", "task_id": "t", "trial_name": "n"},
        )


@pytest.mark.parametrize(
    "import_path", sorted(set(AgentFactory.AGENT_NAME_TO_IMPORT_PATH.values()))
)
def test_every_registered_agent_uses_harness_trace_entry(
    repository, tmp_path, import_path
):
    cls = AgentFactory.get_agent_from_import_path(import_path)
    agent = object.__new__(cls)
    BaseAgent.__init__(agent)

    def perform(self, **kwargs):
        assert current_trace() is self._agent_trace is kwargs["session"]._agent_trace
        self._agent_trace.record("custom.evidence", native_agent=import_path)
        return AgentResult()

    agent.perform_task = MethodType(perform, agent)
    # Even a custom adapter overriding execute_task cannot bypass the entry.
    agent.execute_task = Mock(side_effect=AssertionError("adapter bypass"))
    logdir = tmp_path / "trial" / "agent-logs"
    assert run_agent(agent, repository[1], logdir) == AgentResult()
    (root,) = (logdir.parent / "agent-trace").iterdir()
    rows = events(root)
    assert rows[0]["event"] == "execution.started"
    assert rows[-1]["event"] == "execution.finished"
    assert rows[-1]["data"]["status"] == "completed"
    assert any(row["event"] == "custom.evidence" for row in rows)
    assert all(row["run_id"] == "run-1" and row["task_id"] == "task-1" for row in rows)
    assert not list(logdir.iterdir())  # Host trace is outside the mounted logs.
    agent.execute_task.assert_not_called()


def test_live_stream_is_durable_before_child_exit_and_on_failure(repository, tmp_path):
    agent = NopAgent()
    ready = tmp_path / "observed"

    def perform(self, **kwargs):
        script = """
import sys, time
from pathlib import Path
sys.stdout.write('partial evidence'); sys.stdout.flush()
deadline = time.monotonic() + 4
while not Path(sys.argv[1]).exists():
    if time.monotonic() > deadline: raise SystemExit(3)
    time.sleep(.01)
print('stderr evidence', file=sys.stderr, flush=True)
"""
        process = subprocess.Popen(
            [sys.executable, "-c", script, str(ready)],
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        trace = self._agent_trace

        def inspect_live_log():
            import time

            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                for row in events(trace.root):
                    if row["event"] == "process.stdout":
                        assert process.poll() is None
                        assert (
                            trace.root / row["data"]["content"]["path"]
                        ).read_text() == "partial evidence"
                        ready.touch()
                        return
                time.sleep(0.01)

        observer = threading.Thread(target=inspect_live_log)
        observer.start()
        _, stderr, code = self._stream_process(
            process, "credential-payload", timeout_sec=5
        )
        observer.join()
        assert ready.exists() and code == 0 and "stderr evidence" in stderr
        raise RuntimeError("agent failed after output")

    agent.perform_task = MethodType(perform, agent)
    logdir = tmp_path / "failed" / "agent-logs"
    with pytest.raises(RuntimeError, match="agent failed"):
        run_agent(agent, repository[1], logdir)
    (root,) = (logdir.parent / "agent-trace").iterdir()
    rows = events(root)
    assert rows[-1]["data"]["status"] == "failed"
    assert any(row["event"] == "process.stderr" for row in rows)
    assert "credential-payload" not in "".join(
        path.read_text(errors="replace") for path in (root / "objects").iterdir()
    )


def test_tool_feedback_patch_failure_async_and_state_reconstruction(
    repository, tmp_path
):
    repo, container = repository
    trace = AgentTrace(tmp_path / "trace", context={"trial_name": "trial"})
    trace.start_collection(container=container, workdir=str(repo), sources={})
    patch = "--- a/code.txt\n+++ b/code.txt\n@@ -1 +1 @@\n-before\n+after\n"
    with trace.activate():
        terminal = TaskTerminal(container, workdir=str(repo))
        assert terminal.apply_patch(patch)["applied"]
        after_patch = trace.latest_state_id
        with pytest.raises(ValueError, match="path must stay"):
            terminal.read_file("../outside")
        result = terminal.run_command("printf '%060000d' 0")
        assert result["output_truncated"]
        started = terminal.start_command("printf async-output", timeout_sec=5)
        terminal._job(started["handle"]).thread.join(timeout=10)
        polled = terminal.poll_command(started["handle"])
        assert polled["done"] and polled["stdout"] == "async-output"
        terminal.run_command(
            "git restore code.txt; rm deleted.txt; printf new > new.txt"
        )
        final_state = trace.latest_state_id
        terminal.close()
    trace.finish_collection()
    trace.close()
    rows = events(trace.root)
    starts = [row for row in rows if row["event"] == "tool.started"]
    patch_start = next(
        row for row in starts if row["data"]["name"].endswith("apply_patch")
    )
    assert patch_start["data"]["inputs"]["patch"] == patch
    rejected = next(row for row in starts if row["data"]["name"].endswith("read_file"))
    assert any(
        row["event"] == "tool.failed" and row["call_id"] == rejected["call_id"]
        for row in rows
    )
    assert any(
        row["event"] == "tool.raw_output"
        and row["data"].get("stdout", {}).get("bytes") == 60000
        for row in rows
    )
    async_start = next(
        row for row in starts if row["data"]["name"].endswith("start_command")
    )
    raw_async = next(
        row
        for row in rows
        if row["event"] == "tool.raw_output" and row["data"].get("handle")
    )
    assert raw_async["call_id"] == async_start["call_id"]
    assert after_patch != final_state
    for state, expected in ((after_patch, "after\n"), (final_state, "before\n")):
        checkpoint = next(
            row["data"]
            for row in rows
            if row["event"] == "repository.checkpoint"
            and row["data"]["state_id"] == state
        )
        restored = tmp_path / state
        restored.mkdir()
        archive = trace.root / checkpoint["base_archive"]["path"]
        with tarfile.open(archive) as source:
            source.extractall(restored, filter="data")
        subprocess.run(["git", "init", "-q", str(restored)], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(restored),
                "apply",
                "--binary",
                str(trace.root / checkpoint["patch"]["path"]),
            ],
            check=True,
        )
        assert (restored / "code.txt").read_text() == expected
        assert (restored / "subst.txt").read_text() == "$Format:%H$\n"
        assert (restored / "binary.dat").read_bytes() == b"\0\xff\x01"
        assert (restored / "link.txt").is_symlink()
        if state == after_patch:
            assert (restored / "deleted.txt").read_text() == "delete me\n"
        if state == final_state:
            assert not (restored / "deleted.txt").exists()
            assert (restored / "new.txt").read_text() == "new"


def test_source_rewrite_append_and_remote_archives(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    path = logs / "native.jsonl"
    trace = AgentTrace(tmp_path / "trace", context={})
    trace._sources = {"native": logs}
    path.write_bytes(b"A" * 5000)
    trace.collect_files()
    path.write_bytes(b"A" * 4999 + b"B" + b"C" * 20)
    trace.collect_files()
    with path.open("ab") as stream:
        stream.write(b"tail")
    trace.collect_files()
    trace._remote_sources = ("/agent-logs",)
    trace._container = SimpleNamespace(
        get_archive=lambda path: ([b"remote tar bytes"], {})
    )
    trace.collect_remote_files()
    trace.collect_remote_files()
    trace.close()
    rows = events(trace.root)
    assert len([row for row in rows if row["event"] == "source.reset"]) == 1
    rebuilt = bytearray()
    for row in rows:
        if row["event"] == "source.chunk" and row["data"]["generation"] == 1:
            assert row["data"]["offset"] == len(rebuilt)
            rebuilt.extend((trace.root / row["data"]["content"]["path"]).read_bytes())
    assert rebuilt == path.read_bytes()
    assert len([row for row in rows if row["event"] == "source.archive"]) == 1


def test_concurrent_trials_do_not_mix_context_or_model_requests(repository, tmp_path):
    def execute(index):
        agent = NopAgent()

        def perform(self, **kwargs):
            model_completion(
                lambda **kwargs: {"content": f"reply-{index}"},
                model="test-model",
                messages=[{"role": "user", "content": f"prompt-{index}"}],
                api_key="do-not-store",
                headers={"authorization": "do-not-store"},
            )
            return AgentResult()

        agent.perform_task = MethodType(perform, agent)
        logdir = tmp_path / f"trial-{index}" / "agent-logs"
        run_agent(agent, repository[1], logdir)
        (root,) = (logdir.parent / "agent-trace").iterdir()
        return events(root)

    with ThreadPoolExecutor(max_workers=2) as pool:
        trials = list(pool.map(execute, range(2)))
    for index, rows in enumerate(trials):
        assert {row["trial_name"] for row in rows} == {f"trial-{index}"}
        request = next(row["data"] for row in rows if row["event"] == "model.started")
        assert request["inputs"]["messages"][0]["content"] == f"prompt-{index}"
        assert "do-not-store" not in json.dumps(rows)
        assert [row["sequence"] for row in rows] == list(range(1, len(rows) + 1))
    assert trials[0][0]["execution_id"] != trials[1][0]["execution_id"]


def test_resume_preserves_interrupted_trace_and_original_artifacts(tmp_path):
    trial = tmp_path / "task" / "trial"
    trial.mkdir(parents=True)
    (trial / "partial.log").write_text("failed evidence")
    trace = AgentTrace(trial / "agent-trace", context={})
    trace.record("execution.started")
    trace.close()
    store = RunArtifactStore(
        run_path=tmp_path,
        run_id=tmp_path.name,
        results_path=tmp_path / "results.json",
        metadata_path=tmp_path / "run_metadata.json",
        s3_bucket=None,
        logger=Mock(),
        trace=Mock(),
    )
    store.archive_interrupted_task("task")
    assert not (tmp_path / "task").exists()
    (saved,) = (tmp_path / "interrupted").glob("*/task/trial")
    assert (saved / "partial.log").read_text() == "failed evidence"
    assert list((saved / "agent-trace").glob("*/events.jsonl"))


class ImportedAgent(NopAgent):
    @staticmethod
    def name():
        return "external-agent"

    def perform_task(self, **kwargs):
        current_trace().record("external.operation", instruction=kwargs["instruction"])
        return AgentResult()


def test_custom_import_agent_is_observed_without_an_adapter(repository, tmp_path):
    agent = AgentFactory.get_agent(import_path=f"{__name__}:ImportedAgent")
    logdir = tmp_path / "custom" / "agent-logs"
    run_agent(agent, repository[1], logdir)
    (path,) = (logdir.parent / "agent-trace").glob("*/events.jsonl")
    assert any(row["event"] == "external.operation" for row in read_trace(path))


def test_timeout_cancels_before_executor_shutdown_and_flushes_partial_trace(
    repository, tmp_path
):
    class WaitingAgent(NopAgent):
        def __init__(self):
            super().__init__()
            self.cancelled = threading.Event()

        def perform_task(self, **kwargs):
            self._agent_trace.record("custom.waiting")
            assert self.cancelled.wait(3), "harness did not cancel the worker"
            self._agent_trace.record("custom.cancelled")
            return AgentResult(failure_mode="agent_timeout")

        def cancel(self):
            self.cancelled.set()

    harness = Harness.__new__(Harness)
    harness._run_id = "timeout-run"
    agent = WaitingAgent()
    logdir = tmp_path / "timeout" / "agent-logs"
    with pytest.raises(TimeoutError):
        asyncio.run(
            harness._run_agent_with_timeout(
                trial_handler=SimpleNamespace(
                    task_id="task", trial_name="timeout", instruction="wait"
                ),
                session=SimpleNamespace(container=repository[1]),
                logging_dir=logdir,
                timeout_sec=0.2,
                agent=agent,
                portkey_metadata=None,
                portkey_trace_id=None,
            )
        )
    (path,) = (logdir.parent / "agent-trace").glob("*/events.jsonl")
    rows = list(read_trace(path))
    assert agent.cancelled.is_set()
    assert any(row["event"] == "execution.timeout" for row in rows)
    assert any(row["event"] == "custom.cancelled" for row in rows)
    assert rows[-1]["data"]["status"] == "failed"


def test_reader_handles_torn_tail_but_rejects_corrupt_evidence(tmp_path):
    trace = AgentTrace(tmp_path, context={})
    trace.record("evidence", value="kept")
    trace.close()
    with trace.path.open("ab") as stream:
        stream.write(b'{"unfinished":')
    rows = list(read_trace(trace.path))
    assert len(rows) == 1 and rows[0]["data"]["value"] == "kept"
    ref = json.loads(trace.path.read_text().splitlines()[0])["data"]
    (trace.root / ref["path"]).write_text("corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        list(read_trace(trace.path))


def test_container_adapter_forwards_live_output_to_common_trace(tmp_path):
    trace = AgentTrace(tmp_path / "trace", context={})
    running = Mock(id="provider-container")

    def logs(**kwargs):
        yield b"first chunk"
        assert any(row["event"] == "process.output" for row in read_trace(trace.path))
        yield b"second chunk"

    running.logs.side_effect = logs
    running.wait.return_value = {"StatusCode": 0}
    agent = object.__new__(GooseMCPAgent)
    BaseAgent.__init__(agent)
    agent._agent_trace = trace
    agent._logger = Mock()
    agent._model_name, agent._provider = "test-model", "test-provider"
    agent._no_rebuild = True
    agent._docker_image_name = "test-image"
    agent._mcp_server = SimpleNamespace(
        server_container_name="mcp", MCP_CONTAINER_PORT=8000
    )
    agent._client = SimpleNamespace(containers=Mock())
    agent._client.containers.run.return_value = running
    with trace.activate():
        agent._run_agent("instruction", "test-network", logging_dir=tmp_path)
    trace.close()
    assert (tmp_path / "agent.log").read_bytes() == b"first chunksecond chunk"
    running.remove.assert_called_once_with(force=True)
    assert events(trace.root)[-1]["data"]["exit_code"] == 0


def test_relay_records_raw_and_delivered_stream_without_auth_headers(tmp_path):
    trace = AgentTrace(tmp_path / "trace", context={})
    handler = object.__new__(_RelayHandler)
    handler.server = SimpleNamespace(trace=trace)
    handler._trace_request_id = "request-1"
    handler.send_response = Mock()
    handler.send_header = Mock()
    handler.end_headers = Mock()
    handler.wfile = io.BytesIO()
    payload = b'data: {"type":"response.completed","response":{"usage":{"total_tokens":4}}}\n\n'
    copied, tokens = handler._copy_response(
        io.BytesIO(payload),
        {"OpenAI-Request-ID": "provider-1", "Authorization": "do-not-store"},
        200,
    )
    trace.close()
    assert copied and tokens == 4 and handler.wfile.getvalue() == payload
    rows = events(trace.root)
    assert all(row["call_id"] == "request-1" for row in rows)
    assert "do-not-store" not in json.dumps(rows)
    assert any(row["event"] == "model.response_chunk" for row in rows)
    delivered = b"".join(
        (trace.root / row["data"]["content"]["path"]).read_bytes()
        for row in rows
        if row["event"] == "model.delivered_chunk"
    )
    assert delivered == payload


# Tests consolidated from tests/unit/utils/test_recording.py


def test_workspace_includes_builds_submodules_and_detects_same_size_rewrite(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".gitignore").write_text("build/\n")
    (root / "build").mkdir()
    binary = root / "build/program"
    binary.write_bytes(b"binary-1")
    (root / "submodule").mkdir()
    (root / "submodule/.git").write_text("gitdir: ../.git/modules/submodule")
    (root / "submodule/code.c").write_text("int answer = 1;")
    spool = tmp_path / "spool"
    first = recording.capture_workspace(root, spool)
    old_time = binary.stat().st_mtime_ns
    binary.write_bytes(b"binary-2")
    os.utime(binary, ns=(old_time, old_time))
    second = recording.capture_workspace(root, spool)
    assert not first["errors"] and not second["errors"]
    assert (
        first["entries"]["build/program"]["sha256"]
        != second["entries"]["build/program"]["sha256"]
    )
    assert "submodule/code.c" in first["entries"]
    assert "submodule/.git" not in first["entries"]
    assert (
        spool / "objects" / first["entries"]["build/program"]["sha256"]
    ).read_bytes() == b"binary-1"


def test_hook_install_preserves_user_hooks_and_has_passive_provider_output(tmp_path):
    config = tmp_path / "hooks.json"
    existing = {"hooks": {"PreToolUse": [{"hooks": [{"command": "existing"}]}]}}
    config.write_text(json.dumps(existing))
    recording.install_hooks(config, "codex", "python3 recorder.py hook")
    recording.install_hooks(config, "codex", "python3 recorder.py hook")
    value = json.loads(config.read_text())
    assert value["hooks"]["PreToolUse"][0] == existing["hooks"]["PreToolUse"][0]
    assert len(value["hooks"]["PreToolUse"]) == 2
    assert "PostToolUseFailure" in value["hooks"]
    for provider, expected in (("codex", b"{}\n"), ("antigravity", b"")):
        result = subprocess.run(
            [
                sys.executable,
                recording.__file__,
                "hook",
                "--provider",
                provider,
                "--event",
                "PreToolUse",
                "--spool",
                str(tmp_path / provider),
            ],
            input=b'{"tool_use_id":"1"}',
            capture_output=True,
            check=True,
        )
        assert result.stdout == expected


def test_missing_provider_post_hook_closes_from_exact_backend_result(tmp_path):
    trace = AgentTrace(tmp_path / "trace", context={})
    arguments = {"patch": "invalid patch"}
    trace.ingest_hook(
        {
            "id": "pre",
            "provider": "codex",
            "payload": {
                "hook_event_name": "PreToolUse",
                "session_id": "session",
                "tool_use_id": "provider-call",
                "tool_name": "mcp__pitbench__apply_patch",
                "tool_input": arguments,
            },
        }
    )
    with trace.call("mcp", name="apply_patch", inputs=arguments) as backend_call_id:
        trace.bind_native_call("apply_patch", arguments, backend_call_id)
        trace.record("mcp.result", result={"isError": True, "content": ["failed"]})
    trace.close_bound_native_calls()
    trace.close()

    rows = list(read_trace(trace.path))
    native_start = next(
        row
        for row in rows
        if row["event"] == "tool.started" and row["data"].get("producer") == "codex"
    )
    native_rows = [row for row in rows if row["call_id"] == native_start["call_id"]]
    assert any(row["event"] == "tool.result" for row in native_rows)
    finished = next(row for row in native_rows if row["event"] == "tool.finished")
    assert finished["data"]["status"] == "failed"
    assert finished["data"]["backend_call_id"] == backend_call_id
    assert any(row["event"] == "instrumentation.recovered_completion" for row in rows)


def test_mcp_metadata_correlates_native_call_and_rejected_request(repository, tmp_path):
    repo, container = repository
    trace = AgentTrace(tmp_path / "trace", context={})
    trace.start_collection(container=container, workdir=str(repo), sources={})
    pre = {
        "id": "native-pre",
        "provider": "codex",
        "completed_ns": 1,
        "payload": {
            "hook_event_name": "PreToolUse",
            "session_id": "session",
            "tool_use_id": "native-1",
            "tool_name": "mcp__pitbench__run_command",
            "tool_input": {"command": "printf evidence"},
        },
    }
    trace.ingest_hook(pre)
    with trace.activate():
        terminal = TaskTerminal(container, workdir=str(repo))

    async def invoke(url):
        async with streamable_http_client(url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "run_command", {"command": "printf evidence"}
                )
                rejected = await session.call_tool("run_command", {})
                return result, rejected

    with LoopbackMCPServer(terminal) as server:
        result, rejected = asyncio.run(invoke(server.url))
    assert result.structuredContent["stdout"] == "evidence"
    assert rejected.isError
    assert result.meta["pitbench"]["before_state_id"]
    assert rejected.meta["pitbench"]["call_id"] != result.meta["pitbench"]["call_id"]
    trace.ingest_hook(
        {
            **pre,
            "id": "native-post",
            "completed_ns": 2,
            "payload": {
                **pre["payload"],
                "hook_event_name": "PostToolUse",
                "tool_response": result.model_dump(by_alias=True),
            },
        }
    )
    trace.finish_collection()
    trace.close()
    report = inspect_trace(trace.path)
    assert report["observed_operations_paired"]
    rows = list(read_trace(trace.path))
    native_end = next(
        row["data"]
        for row in rows
        if row["event"] == "tool.finished" and row["data"].get("producer") == "codex"
    )
    assert native_end["backend_call_id"] == result.meta["pitbench"]["call_id"]


def test_ambiguous_native_requests_are_not_assigned_a_backend(tmp_path):
    trace = AgentTrace(tmp_path, context={})
    for number in (1, 2):
        trace.ingest_hook(
            {
                "id": str(number),
                "provider": "codex",
                "payload": {
                    "hook_event_name": "PreToolUse",
                    "session_id": "session",
                    "tool_use_id": str(number),
                    "tool_name": "mcp__pitbench__run_command",
                    "tool_input": {"command": "pwd"},
                },
            }
        )
    trace.bind_native_call("run_command", {"command": "pwd"}, "backend")
    assert all("backend_call_id" not in call for call in trace._native_calls.values())
    assert list(read_trace(trace.path))[-1]["data"]["method"] == "ambiguous; not_bound"
    trace.close()


def test_antigravity_result_is_recovered_from_later_transcript(tmp_path):
    trace = AgentTrace(tmp_path, context={})
    payload = {
        "conversationId": "s",
        "stepIdx": 2,
        "toolCall": {"name": "view_file", "args": {"AbsolutePath": "/schema.json"}},
    }
    trace.ingest_hook(
        {
            "id": "before",
            "provider": "antigravity",
            "hook_event": "PreToolUse",
            "payload": payload,
        }
    )
    trace.ingest_hook(
        {
            "id": "after",
            "provider": "antigravity",
            "hook_event": "PostToolUse",
            "payload": {**payload, "error": ""},
        }
    )
    assert (
        next(
            row["data"]
            for row in read_trace(trace.path)
            if row["event"] == "tool.result"
        )["result_present"]
        is False
    )
    transcript = trace.content(
        json.dumps({"step_index": 2, "status": "DONE", "content": "actual feedback"})
        + "\n"
    )
    trace.ingest_hook(
        {
            "id": "end",
            "provider": "antigravity",
            "hook_event": "SessionEnd",
            "payload": payload,
            "transcript": transcript,
        }
    )
    results = [row for row in read_trace(trace.path) if row["event"] == "tool.result"]
    assert len({row["call_id"] for row in results}) == 1
    assert results[-1]["data"]["result"] == "actual feedback"
    trace.close()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux inotify journal")
def test_filesystem_journal_records_fast_create_delete_between_snapshots(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    spool = tmp_path / "spool"
    process = subprocess.Popen(
        [
            sys.executable,
            recording.__file__,
            "watch",
            "--root",
            str(root),
            "--spool",
            str(spool),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert process.stdout.readline() == b"READY\n"
        temporary = root / "short-lived.txt"
        temporary.write_text("transient")
        temporary.unlink()
        deadline = time.monotonic() + 3
        recorded = []
        while time.monotonic() < deadline:
            recorded = [
                json.loads(path.read_text()) for path in spool.glob("event-*.json")
            ]
            if any(event["payload"]["operation"] == "delete" for event in recorded):
                break
            time.sleep(0.01)
        assert {event["payload"]["operation"] for event in recorded} >= {
            "create",
            "delete",
        }
    finally:
        (spool / "stop-watcher").touch()
        process.communicate(timeout=5)
    assert process.returncode == 0


# Tests consolidated from tests/unit/pitbench/test_auth.py


CODEX_TOKEN = {
    "auth_mode": "chatgpt",
    "tokens": {
        "access_token": "private-access",
        "refresh_token": "private-refresh",
        "account_id": "account",
    },
}
AGY_TOKEN = {
    "auth_method": "consumer",
    "token": {"access_token": "private-access", "refresh_token": "private-refresh"},
}


@pytest.fixture
def environment(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    original_which = auth.shutil.which
    monkeypatch.setattr(
        auth.shutil,
        "which",
        lambda value: (
            f"/native/{value}" if value in {"codex", "agy"} else original_which(value)
        ),
    )
    root = tmp_path / "project"
    root.mkdir()
    return root, home


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def invoke(root, *args):
    return CliRunner().invoke(app, ["auth", *args, "--root", str(root)])


def test_reuse_codex_login_preserves_other_local_configuration(environment):
    root, home = environment
    credential = put(home / ".codex/auth.json", CODEX_TOKEN)
    config = root / "config/evaluate.local.yaml"
    config.parent.mkdir()
    existing = {
        "paths": {"output_path": "custom-runs"},
        "tasks": {"task": {"agent_image": "image:local"}},
        "agents": {
            "antigravity": {"model_name": "keep"},
            "codex": {
                "codex_binary": "/missing/old/codex",
                "codex_auth_path": "/missing/old/auth.json",
                "proxy_url": "http://old-proxy",
            },
        },
    }
    config.write_text(yaml.safe_dump(existing))
    for _ in range(2):
        result = invoke(root, "login", "codex", "--non-interactive")
        assert result.exit_code == 0, result.output
    actual = yaml.safe_load(config.read_text())
    assert actual["paths"] == existing["paths"]
    assert actual["tasks"] == existing["tasks"]
    assert actual["agents"]["antigravity"] == existing["agents"]["antigravity"]
    assert actual["agents"]["codex"]["codex_auth_path"] == str(credential)
    assert actual["agents"]["codex"]["codex_binary"] == "/native/codex"
    assert actual["agents"]["codex"]["proxy_url"] == "http://old-proxy"
    assert "private-access" not in result.output
    assert config.stat().st_mode & 0o777 == 0o600


def test_agy_keyring_is_exported_once_to_private_file(environment, monkeypatch):
    root, _ = environment
    read_keyring = Mock(return_value=AGY_TOKEN)
    monkeypatch.setattr(auth, "read_agy_keyring", read_keyring)
    for alias in ("agy", "antigravity"):
        result = invoke(
            root,
            "login",
            alias,
            "--non-interactive",
            "--proxy-url",
            "http://local:7897",
            "--model",
            "selected",
        )
        assert result.exit_code == 0, result.output
    read_keyring.assert_called_once()
    token = root / ".pitbench/credentials/antigravity-oauth-token"
    assert json.loads(token.read_text()) == AGY_TOKEN
    assert token.stat().st_mode & 0o777 == 0o600
    assert token.parent.stat().st_mode & 0o777 == 0o700
    assert (token.parent / ".gitignore").read_text() == "*\n"
    values = yaml.safe_load((root / "config/evaluate.local.yaml").read_text())[
        "agents"
    ]["antigravity"]
    assert values["auth_token_path"] == str(token)
    assert values["runner_backend"] == "container"
    assert values["model_name"] == "selected"
    assert values["proxy_url"] == "http://local:7897"
    assert "private-access" not in result.output


@pytest.mark.parametrize("agent,token", [("codex", CODEX_TOKEN), ("agy", AGY_TOKEN)])
def test_explicit_auth_file_does_not_invoke_keyring_or_login(
    environment, monkeypatch, agent, token
):
    root, _ = environment
    exported = put(root / "exported.json", token)
    keyring = Mock(side_effect=AssertionError("must not read keyring"))
    monkeypatch.setattr(auth, "read_agy_keyring", keyring)
    result = invoke(
        root, "login", agent, "--auth-file", str(exported), "--non-interactive"
    )
    assert result.exit_code == 0, result.output
    keyring.assert_not_called()


def test_codex_uses_configured_codex_home(environment, monkeypatch):
    root, _ = environment
    directory = root / "native-home"
    credential = put(directory / "auth.json", CODEX_TOKEN)
    monkeypatch.setenv("CODEX_HOME", str(directory))
    result = invoke(root, "login", "codex", "--non-interactive")
    assert result.exit_code == 0, result.output
    assert str(credential) in result.output


def test_no_credentials_in_noninteractive_mode_does_not_write_config(
    environment, monkeypatch
):
    root, _ = environment
    monkeypatch.setattr(auth, "read_agy_keyring", lambda: None)
    result = invoke(root, "login", "agy", "--non-interactive")
    assert result.exit_code == 2
    assert "No readable agy credential" in result.output
    assert not (root / "config/evaluate.local.yaml").exists()


def test_keyring_error_is_not_treated_as_signed_out(environment, monkeypatch):
    root, _ = environment
    monkeypatch.setattr(
        auth, "read_agy_keyring", Mock(side_effect=ValueError("unlock keyring"))
    )
    result = invoke(root, "login", "agy", "--non-interactive")
    assert result.exit_code == 2
    assert "unlock keyring" in result.output
    assert not (root / "config/evaluate.local.yaml").exists()


@pytest.mark.parametrize(
    "agent,token", [("codex", AGY_TOKEN), ("agy", {"access_token": "never-print-this"})]
)
def test_wrong_credential_format_is_rejected_without_echoing_token(
    environment, agent, token
):
    root, _ = environment
    path = put(root / "incorrect.json", token)
    result = invoke(root, "login", agent, "--auth-file", str(path))
    assert result.exit_code == 2
    assert "Expected" in result.output
    assert "never-print-this" not in result.output
    assert not (root / "config/evaluate.local.yaml").exists()


def test_codex_device_login_uses_native_cli_and_does_not_change_global_config(
    environment, monkeypatch
):
    root, home = environment
    monkeypatch.setattr(
        auth, "sys", SimpleNamespace(stdin=SimpleNamespace(isatty=lambda: True))
    )
    original_run = subprocess.run
    calls = []

    def run(command, **kwargs):
        if command[0] == "git":
            return original_run(command, **kwargs)
        calls.append((command, kwargs))
        put(home / ".codex/auth.json", CODEX_TOKEN)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(auth.subprocess, "run", run)
    result = invoke(
        root, "login", "codex", "--device-auth", "--proxy-url", "http://selected:1234"
    )
    assert result.exit_code == 0, result.output
    assert calls[0][0] == [
        "/native/codex",
        "-c",
        'cli_auth_credentials_store="file"',
        "login",
        "--device-auth",
    ]
    assert calls[0][1]["env"]["HTTPS_PROXY"] == "http://selected:1234"
    assert not (home / ".codex/config.toml").exists()


def test_status_needs_no_task_data_and_propagates_failure(environment, monkeypatch):
    root, _ = environment
    credential = DoctorCheck(CheckStatus.PASS, "credentials", "ready")
    runner = DoctorCheck(CheckStatus.WARN, "runner", "image missing")
    monkeypatch.setattr(auth, "_credential_check", lambda *args: (credential, True))
    monkeypatch.setattr(auth, "_runner_check", lambda *args: (runner, False))
    result = invoke(root, "status", "agy")
    assert result.exit_code == 1
    assert "image missing" in result.output
    assert not (root / "configs/tasks").exists()


def test_codex_status_checks_supplied_file_and_cleans_temporary_copy(
    tmp_path, monkeypatch
):
    credential = put(tmp_path / "exported.json", CODEX_TOKEN)
    directories = []

    def run(command, **kwargs):
        directory = Path(kwargs["env"]["CODEX_HOME"])
        directories.append(directory)
        assert json.loads((directory / "auth.json").read_text()) == CODEX_TOKEN
        assert (directory / "auth.json").stat().st_mode & 0o777 == 0o600
        assert "OPENAI_API_KEY" not in kwargs["env"]
        return subprocess.CompletedProcess(command, 0, "Logged in using ChatGPT", "")

    monkeypatch.setattr(subprocess, "run", run)
    result = CodexMCPAgent.check_login(
        "codex", credential, {"OPENAI_API_KEY": "must-not-use"}
    )
    assert result.returncode == 0
    assert not directories[0].exists()
    assert json.loads(credential.read_text()) == CODEX_TOKEN


def test_private_write_refuses_tracked_files(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    path = tmp_path / "public.yaml"
    path.write_text("unchanged")
    subprocess.run(["git", "add", path.name], cwd=tmp_path, check=True)
    with pytest.raises(ValueError, match="tracked file"):
        auth._write_private(path, "secret")
    assert path.read_text() == "unchanged"
