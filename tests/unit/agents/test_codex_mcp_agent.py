from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import anyio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from fceval.agents import AgentFactory, AgentName
from fceval.agents.codex_mcp_agent import CodexMCPAgent
from fceval.agents.host_mcp import LoopbackMCPServer, TaskTerminal


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
    assert not any("auth.json" in argument for argument in command)


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


def test_subscription_env_removes_api_key_and_keeps_loopback_off_proxy():
    with patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "must-not-be-used", "NO_PROXY": "example.test"},
        clear=True,
    ):
        env = CodexMCPAgent._subscription_env()

    assert "OPENAI_API_KEY" not in env
    assert "example.test" in env["NO_PROXY"]
    assert "127.0.0.1" in env["NO_PROXY"]
    assert "localhost" in env["NO_PROXY"]


def test_codex_jsonl_usage_is_aggregated():
    output = "\n".join(
        [
            '{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":3}}',
            "not json",
            '{"type":"turn.completed","usage":{"input_tokens":7,"output_tokens":2}}',
        ]
    )

    assert CodexMCPAgent._parse_usage(output) == (17, 5)


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
