from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import anyio
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from pitbench.harness.agents import AgentFactory, AgentName
from pitbench.harness.agents.codex_container import CodexContainerRunner
from pitbench.harness.agents.codex_mcp_agent import CodexMCPAgent
from pitbench.harness.agents.host_mcp import LoopbackMCPServer, TaskTerminal


class FakeContainer:
    def __init__(
        self,
        *,
        exit_code: int = 0,
        stdout: bytes = b"ok\n",
        stderr: bytes = b"",
    ) -> None:
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.calls = []

    def exec_run(self, command, **kwargs):
        self.calls.append((command, kwargs))
        return SimpleNamespace(
            exit_code=self.exit_code,
            output=(self.stdout, self.stderr),
        )


class BlockingContainer(FakeContainer):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.released = threading.Event()

    def exec_run(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if len(self.calls) == 1:
            self.started.set()
            self.released.wait(timeout=2)
            return SimpleNamespace(exit_code=143, output=(b"partial\n", b""))
        self.released.set()
        return SimpleNamespace(exit_code=0, output=(b"", b""))


def test_task_terminal_runs_in_fixed_container_workdir(tmp_path):
    container = FakeContainer(stdout=b"changed\n", stderr=b"warning\n")
    terminal = TaskTerminal(
        container,
        workdir="/workspace/repo",
        log_path=tmp_path / "terminal.jsonl",
    )

    result = terminal.run_command("git status --short", timeout_sec=12)

    command, kwargs = container.calls[0]
    assert command[:5] == [
        "timeout",
        "--signal=TERM",
        "--kill-after=5s",
        "12.0s",
        "bash",
    ]
    assert command[-1] == "git status --short"
    assert kwargs == {"workdir": "/workspace/repo", "demux": True}
    assert result["exit_code"] == 0
    assert result["stdout"] == "changed\n"
    assert result["stderr"] == "warning\n"
    assert (tmp_path / "terminal.jsonl").is_file()


def test_task_terminal_reports_timeout():
    terminal = TaskTerminal(FakeContainer(exit_code=124), workdir="/workspace/repo")

    result = terminal.run_command("sleep 999", timeout_sec=9999)

    assert result["timed_out"] is True
    assert terminal._container.calls[0][0][3] == "900.0s"


def test_loopback_mcp_server_calls_bound_terminal():
    container = FakeContainer(stdout=b"/workspace/repo\n")
    terminal = TaskTerminal(container, workdir="/workspace/repo")

    async def call_tool(url: str):
        async with streamablehttp_client(url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                result = await session.call_tool(
                    "run_command",
                    {"command": "pwd", "timeout_sec": 1},
                )
                return result, {tool.name for tool in tools.tools}

    with LoopbackMCPServer(terminal) as server:
        assert server.url is not None
        with patch.dict(
            "os.environ",
            {
                "ALL_PROXY": "",
                "all_proxy": "",
                "NO_PROXY": "127.0.0.1,localhost",
                "no_proxy": "127.0.0.1,localhost",
            },
        ):
            result, tool_names = anyio.run(call_tool, server.url)

    assert result.isError is False
    assert container.calls[0][0][-1] == "pwd"
    assert tool_names == {
        "apply_patch",
        "cancel_command",
        "git_diff",
        "git_status",
        "list_files",
        "poll_command",
        "read_file",
        "run_command",
        "search_files",
        "start_command",
    }


def test_task_terminal_reads_file_range_and_rejects_escape():
    container = FakeContainer(stdout=b"11\naGVsbG8=")
    terminal = TaskTerminal(container, workdir="/workspace/repo")

    result = terminal.read_file("src/example.py", offset=2, limit=5)

    assert result == {
        "path": "src/example.py",
        "offset": 2,
        "content": "hello",
        "next_offset": 7,
        "size": 11,
        "eof": False,
    }
    assert "realpath -e" in container.calls[0][0][-1]

    with pytest.raises(ValueError, match="stay within"):
        terminal.read_file("../hidden.txt")


def test_task_terminal_applies_checked_patch_and_rejects_escape():
    container = FakeContainer()
    terminal = TaskTerminal(container, workdir="/workspace/repo")
    patch_text = """--- a/example.txt
+++ b/example.txt
@@ -1 +1 @@
-old
+new
"""

    result = terminal.apply_patch(patch_text)

    assert result["applied"] is True
    command = container.calls[0][0][-1]
    assert "git apply --check" in command
    assert 'git apply "$patch_file"' in command

    with pytest.raises(ValueError, match="stay within"):
        terminal.apply_patch("--- a/file\n+++ ../../outside\n")


def test_task_terminal_returns_structured_git_status_and_diff():
    status_container = FakeContainer(stdout=b" M src/main.py\n?? notes.txt\n")
    terminal = TaskTerminal(status_container, workdir="/workspace/repo")

    assert terminal.git_status() == {
        "clean": False,
        "entries": [
            {"index": " ", "worktree": "M", "path": "src/main.py"},
            {"index": "?", "worktree": "?", "path": "notes.txt"},
        ],
    }

    diff_container = FakeContainer(stdout=b"diff --git a/a b/a\n")
    diff = TaskTerminal(diff_container, workdir="/workspace/repo").git_diff(
        "src/main.py", staged=True
    )
    assert diff["diff"].startswith("diff --git")
    assert diff["path"] == "src/main.py"
    assert diff["staged"] is True
    assert "git diff --cached -- src/main.py" in diff_container.calls[0][0][-1]


def test_task_terminal_async_command_supports_incremental_polling():
    terminal = TaskTerminal(
        FakeContainer(stdout=b"first second", stderr=b"warning"),
        workdir="/workspace/repo",
    )

    started = terminal.start_command("pitbench check", timeout_sec=10)
    deadline = time.monotonic() + 2
    while True:
        result = terminal.poll_command(started["handle"], max_chars=5)
        if result["done"]:
            break
        assert time.monotonic() < deadline
        time.sleep(0.01)

    assert started["streaming"] is False
    assert result["stdout"] == "first"
    assert result["next_stdout_offset"] == 5
    remainder = terminal.poll_command(
        started["handle"], stdout_offset=result["next_stdout_offset"]
    )
    assert remainder["stdout"] == " second"
    assert remainder["stderr"] == "warning"
    assert remainder["exit_code"] == 0


def test_task_terminal_can_cancel_async_command():
    container = BlockingContainer()
    terminal = TaskTerminal(container, workdir="/workspace/repo")
    started = terminal.start_command("sleep 100")
    assert container.started.wait(timeout=1)

    cancelled = terminal.cancel_command(started["handle"])
    terminal._jobs[started["handle"]].thread.join(timeout=1)
    result = terminal.poll_command(started["handle"])

    assert cancelled["cancelled"] is True
    assert result["cancelled"] is True
    assert result["done"] is True


def test_codex_command_uses_isolated_runner_and_loopback_mcp():
    agent = CodexMCPAgent(
        model_name="openai/gpt-5",
        runner_path="/installed/pitbench-codex-runner",
        reasoning_effort="xhigh",
    )

    command = agent._build_command(
        mcp_url="http://127.0.0.1:43210/mcp",
        instruction="Improve it.",
    )

    assert command[:7] == [
        "sudo",
        "-n",
        "-u",
        "pitbench-codex",
        "--",
        "/installed/pitbench-codex-runner",
        "exec",
    ]
    assert "--ignore-user-config" in command
    assert "--ephemeral" in command
    assert ["--sandbox", "read-only"] == command[
        command.index("--sandbox") : command.index("--sandbox") + 2
    ]
    assert 'mcp_servers.pitbench.url="http://127.0.0.1:43210/mcp"' in command
    assert 'mcp_servers.pitbench.default_tools_approval_mode="approve"' in command
    assert 'model_reasoning_effort="xhigh"' in command
    assert not any("auth.json" in argument for argument in command)


def test_codex_control_plane_command_uses_runner_without_task_mcp():
    agent = CodexMCPAgent(
        model_name="openai/gpt-5.6-luna",
        runner_path="/installed/pitbench-codex-runner",
        reasoning_effort="xhigh",
    )

    command = agent._build_control_plane_command()

    assert command[:7] == [
        "sudo",
        "-n",
        "-u",
        "pitbench-codex",
        "--",
        "/installed/pitbench-codex-runner",
        "exec",
    ]
    assert command[command.index("--model") + 1] == "gpt-5.6-luna"
    assert 'model_reasoning_effort="xhigh"' in command
    assert not any("mcp_servers" in argument for argument in command)


def test_codex_container_backend_loads_ephemeral_profile_config(tmp_path):
    auth_path = tmp_path / "auth.json"
    auth_path.write_text('{"tokens":{"access_token":"test"}}')
    profile_home = tmp_path / "profile/codex-home"
    profile_home.mkdir(parents=True)
    (tmp_path / "profile/profile.yaml").write_text(
        'schema_version: "1.0"\n'
        "name: custom\n"
        "codex_home: codex-home\n"
        "allow_hooks: true\n"
    )
    (profile_home / "config.toml").write_text("# profile\n")
    agent = CodexMCPAgent(
        model_name="gpt-5.6-sol",
        codex_auth_path=str(auth_path),
        runner_backend="container",
        profile_path=str(tmp_path / "profile"),
    )
    container_runner = MagicMock(spec=CodexContainerRunner)
    container_runner.command_prefix.return_value = ["docker-runner"]
    agent._container_runner = container_runner

    command = agent._build_command(
        mcp_url="http://127.0.0.1:43210/mcp", instruction="Improve it."
    )
    payload = json.loads(agent._runner_payload({}))

    assert command[0:2] == ["docker-runner", "exec"]
    assert "--ignore-user-config" not in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert payload["allow_hooks"] is True
    assert payload["profile_sha256"] == agent._profile.sha256


def test_codex_workspace_backend_uses_native_tools_and_strict_relay(tmp_path):
    auth_path = tmp_path / "auth.json"
    auth_path.write_text('{"tokens":{"access_token":"secret"}}')
    profile_home = tmp_path / "profile/codex-home"
    profile_home.mkdir(parents=True)
    (tmp_path / "profile/profile.yaml").write_text(
        'schema_version: "1.0"\n'
        "name: plugins\n"
        "codex_home: codex-home\n"
        "allow_hooks: true\n"
    )
    (profile_home / "config.toml").write_text("# plugin config\n")
    agent = CodexMCPAgent(
        model_name="gpt-5.6-sol",
        codex_auth_path=str(auth_path),
        runner_backend="workspace",
        profile_path=str(tmp_path / "profile"),
        reasoning_effort="high",
    )
    runtime = SimpleNamespace(
        config_profile_name="pitbench_workspace_test",
        command_prefix=lambda: ["docker-exec", "token-from-environment"],
    )
    relay = SimpleNamespace(
        url="http://172.30.0.1:4321",
        container_ip="172.30.0.1",
        port=4321,
    )

    command = agent._build_workspace_command(
        runtime=runtime,
        relay=relay,
        instruction="Improve it.",
    )

    assert command[:2] == ["docker-exec", "token-from-environment"]
    assert command[command.index("--profile") + 1] == "pitbench_workspace_test"
    assert "--dangerously-bypass-hook-trust" in command
    assert "--sandbox" not in command
    assert "use_legacy_landlock" not in command
    assert 'model_provider="pitbench_relay"' in command
    assert 'model_providers.pitbench_relay.base_url="http://172.30.0.1:4321"' in command
    assert not any("env_key" in argument for argument in command)
    assert 'web_search="disabled"' in command
    assert "features.unified_exec=true" in command
    assert "--strict-config" in command
    assert "shell_environment_policy.ignore_default_excludes=false" in command
    assert not any("secret" in argument for argument in command)
    assert agent._runner_payload({}) == ""


def test_codex_host_backend_rejects_profile(tmp_path):
    with pytest.raises(
        ValueError, match="profiles require runner_backend=container or workspace"
    ):
        CodexMCPAgent(
            model_name="gpt-5.6-sol",
            runner_backend="host",
            profile_path=str(tmp_path),
        )


def test_codex_omits_reasoning_effort_when_unspecified():
    agent = CodexMCPAgent(model_name="gpt-5.5")

    command = agent._build_control_plane_command()

    assert not any("model_reasoning_effort" in argument for argument in command)


def test_runner_payload_copies_only_auth_and_proxy(tmp_path):
    auth_path = tmp_path / "auth.json"
    auth_path.write_text('{"tokens":{"access_token":"secret"}}')
    agent = CodexMCPAgent(
        model_name="gpt-5",
        codex_auth_path=str(auth_path),
    )
    payload = json.loads(
        agent._runner_payload(
            {
                "HTTPS_PROXY": "http://127.0.0.1:7897",
                "UNRELATED": "must-not-cross",
            }
        )
    )

    assert payload["auth_json"] == auth_path.read_text()
    assert payload["proxy_env"] == {"HTTPS_PROXY": "http://127.0.0.1:7897"}
    assert "UNRELATED" not in payload["proxy_env"]


def test_subscription_env_removes_api_key_and_defaults_to_no_proxy():
    with patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "must-not-be-used",
            "HTTPS_PROXY": "http://ambient-proxy:8080",
            "NO_PROXY": "example.test",
        },
        clear=True,
    ):
        env = CodexMCPAgent._subscription_env()

    assert "OPENAI_API_KEY" not in env
    assert "HTTPS_PROXY" not in env
    assert env["NO_PROXY"] == "127.0.0.1,localhost"


def test_runtime_env_applies_explicit_proxy_to_codex_runner(tmp_path):
    auth_path = tmp_path / "auth.json"
    auth_path.write_text('{"tokens":{"access_token":"secret"}}')
    agent = CodexMCPAgent(
        model_name="gpt-5",
        codex_auth_path=str(auth_path),
        proxy_url="http://127.0.0.1:7897",
    )

    with patch.dict("os.environ", {}, clear=True):
        env = agent._runtime_env()
        payload = json.loads(agent._runner_payload(env))

    assert payload["proxy_env"]["HTTP_PROXY"] == "http://127.0.0.1:7897"
    assert payload["proxy_env"]["HTTPS_PROXY"] == "http://127.0.0.1:7897"
    assert payload["proxy_env"]["ALL_PROXY"] == "http://127.0.0.1:7897"
    assert env["NO_PROXY"] == "127.0.0.1,localhost"


def test_codex_jsonl_usage_is_aggregated():
    output = "\n".join(
        [
            '{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":3}}',
            "not json",
            '{"type":"turn.completed","usage":{"input_tokens":7,"output_tokens":2}}',
        ]
    )

    assert CodexMCPAgent._parse_usage(output) == (17, 5)


def test_control_plane_preflight_requires_completed_turn(tmp_path):
    agent = CodexMCPAgent(model_name="gpt-5", control_plane_timeout_sec=3)
    completed = MagicMock()
    completed.returncode = 0
    completed.communicate.return_value = (
        '{"type":"turn.completed","usage":{}}\n',
        "",
    )

    with patch("subprocess.Popen", return_value=completed) as popen:
        agent._check_control_plane('{"auth_json":"{}"}', tmp_path)

    assert popen.call_args.kwargs["start_new_session"] is True
    completed.communicate.assert_called_once_with(input='{"auth_json":"{}"}', timeout=3)
    assert "turn.completed" in (tmp_path / "codex-preflight.jsonl").read_text()


def test_control_plane_preflight_reports_backend_timeout(tmp_path):
    agent = CodexMCPAgent(model_name="gpt-5", control_plane_timeout_sec=2)
    process = MagicMock()
    process.communicate.side_effect = [
        subprocess.TimeoutExpired(
            cmd="codex", timeout=2, output=b'{"type":"turn.started"}\n'
        ),
        ('{"type":"turn.started"}\n', "transport timeout\n"),
    ]

    with (
        patch("subprocess.Popen", return_value=process),
        patch.object(agent, "_terminate_process") as terminate,
        pytest.raises(RuntimeError, match="control-plane preflight timed out"),
    ):
        agent._check_control_plane('{"auth_json":"{}"}', tmp_path)

    terminate.assert_called_once_with(process)
    assert "turn.started" in (tmp_path / "codex-preflight.jsonl").read_text()


def test_control_plane_preflight_preserves_backend_error(tmp_path):
    agent = CodexMCPAgent(model_name="gpt-5")
    failed = MagicMock()
    failed.returncode = 1
    failed.communicate.return_value = (
        (
            '{"type":"error","message":"Reconnecting"}\n'
            '{"type":"turn.failed","error":{"message":"request timed out"}}\n'
        ),
        "transport closed",
    )

    with (
        patch("subprocess.Popen", return_value=failed),
        pytest.raises(RuntimeError, match="request timed out"),
    ):
        agent._check_control_plane('{"auth_json":"{}"}', tmp_path)


def test_codex_jsonl_requires_completed_mcp_call():
    failed = (
        '{"type":"item.completed","item":{"type":"mcp_tool_call",'
        '"status":"failed","error":{"message":"cancelled"}}}'
    )
    completed = (
        '{"type":"item.completed","item":{"type":"mcp_tool_call",'
        '"status":"completed","error":null}}'
    )

    assert CodexMCPAgent._has_successful_mcp_call(failed) is False
    assert CodexMCPAgent._has_successful_mcp_call(completed) is True


def test_codex_jsonl_accepts_completed_workspace_command():
    failed = (
        '{"type":"item.completed","item":{"type":"command_execution",'
        '"status":"failed","exit_code":1}}'
    )
    completed = (
        '{"type":"item.completed","item":{"type":"command_execution",'
        '"status":"completed","exit_code":0}}'
    )

    assert CodexMCPAgent._has_successful_workspace_call(failed) is False
    assert CodexMCPAgent._has_successful_workspace_call(completed) is True


def test_workspace_relay_uses_no_solver_credential():
    agent = CodexMCPAgent(model_name="gpt-5", runner_backend="workspace")
    relay_process = MagicMock(returncode=0)
    relay_process.communicate.return_value = ("PITBENCH_RELAY_REACHABLE\n", "")
    public_process = MagicMock(returncode=0)
    public_process.communicate.return_value = ("PITBENCH_PUBLIC_BLOCKED\n", "")
    runtime = SimpleNamespace(
        config_profile_name="pitbench_workspace_test",
        permission_profile_name="pitbench_workspace_relay_test",
        command_prefix=lambda: ["docker", "exec"],
    )
    relay = SimpleNamespace(
        url="http://172.30.0.1:4321",
        container_ip="172.30.0.1",
        port=4321,
    )

    with patch(
        "subprocess.Popen", side_effect=[relay_process, public_process]
    ) as popen:
        agent._check_workspace_isolation(
            runtime=runtime,
            relay=relay,
            env={"PATH": "/usr/bin"},
            logging_dir=None,
        )

    commands = [call.args[0] for call in popen.call_args_list]
    assert all(
        not any("env_key" in argument for argument in command) for command in commands
    )
    assert all(
        call.kwargs["env"] == {"PATH": "/usr/bin"} for call in popen.call_args_list
    )


def test_codex_jsonl_reports_live_agent_interactions():
    agent = CodexMCPAgent(model_name="gpt-5")
    progress = []
    agent.set_progress_callback(progress.append)
    counts = {"tool_calls": 0, "messages": 0}

    agent._update_live_progress(
        '{"type":"item.completed","item":{"type":"agent_message"}}', counts
    )
    agent._update_live_progress(
        '{"type":"item.completed","item":{"type":"mcp_tool_call",'
        '"status":"completed"}}',
        counts,
    )

    assert progress == [
        "Agent: tool calls 0 · messages 1",
        "Agent: tool calls 1 · messages 1",
    ]


def test_codex_process_streams_jsonl_and_preserves_output():
    agent = CodexMCPAgent(model_name="gpt-5", timeout_sec=5)
    progress = []
    agent.set_progress_callback(progress.append)
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                'print(\'{"type":"item.completed","item":{"type":'
                '"mcp_tool_call"}}\', flush=True); '
                "print('diagnostic', file=sys.stderr, flush=True)"
            ),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stdout, stderr, return_code = agent._stream_process(process, "{}")

    assert return_code == 0
    assert '"mcp_tool_call"' in stdout
    assert stderr == "diagnostic\n"
    assert progress == ["Agent: tool calls 1 · messages 0"]


def test_factory_registers_host_codex_agent():
    assert AgentFactory.get_agent_class(AgentName.CODEX) is CodexMCPAgent
