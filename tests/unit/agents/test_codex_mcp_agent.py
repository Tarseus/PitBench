from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import anyio
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from pitbench.harness.agents import AgentFactory, AgentName
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
                return await session.call_tool(
                    "run_command",
                    {"command": "pwd", "timeout_sec": 1},
                )

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
            result = anyio.run(call_tool, server.url)

    assert result.isError is False
    assert container.calls[0][0][-1] == "pwd"


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


def test_factory_registers_host_codex_agent():
    assert AgentFactory.get_agent_class(AgentName.CODEX) is CodexMCPAgent
