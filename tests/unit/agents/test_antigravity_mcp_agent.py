from __future__ import annotations

import json
from unittest.mock import patch

from fceval.agents import AgentFactory, AgentName
from fceval.agents.antigravity_mcp_agent import AntigravityMCPAgent


def test_antigravity_command_uses_isolated_runner_and_loopback_mcp():
    agent = AntigravityMCPAgent(
        model_name="google/gemini-3.1-pro-high",
        runner_path="/installed/pitbench-antigravity-runner",
        effort="high",
    )

    command = agent._build_command(
        mcp_url="http://127.0.0.1:43210/mcp",
        instruction="Improve it.",
    )

    assert command[:7] == [
        "sudo",
        "-n",
        "-u",
        "pitbench-agy",
        "--",
        "/installed/pitbench-antigravity-runner",
        "run",
    ]
    assert command[7:10] == [
        "--mcp-url",
        "http://127.0.0.1:43210/mcp",
        "--",
    ]
    assert ["--output-format", "stream-json"] == command[
        command.index("--output-format") : command.index("--output-format") + 2
    ]
    assert "--sandbox" in command
    assert "--disable-slash-commands" in command
    assert ["--effort", "high"] == command[-2:]
    assert "--dangerously-skip-permissions" not in command
    assert not any("oauth_creds.json" in argument for argument in command)


def test_runner_payload_copies_only_antigravity_auth_and_proxy(tmp_path):
    auth_token_path = tmp_path / "antigravity-oauth-token"
    settings_path = tmp_path / "settings.json"
    auth_token_path.write_text(
        '{"auth_method":"oauth-personal","token":{"access_token":"secret"}}'
    )
    settings_path.write_text('{"security":{"auth":{"selectedType":"oauth-personal"}}}')
    agent = AntigravityMCPAgent(
        model_name="gemini-3.7-flash-low",
        auth_token_path=str(auth_token_path),
        settings_path=str(settings_path),
    )

    payload = json.loads(
        agent._runner_payload(
            {
                "HTTPS_PROXY": "http://127.0.0.1:7897",
                "UNRELATED": "must-not-cross",
            }
        )
    )

    assert json.loads(payload["auth_token_json"])["token"]["access_token"] == "secret"
    assert json.loads(payload["settings_json"])["security"]["auth"]
    assert payload["proxy_env"] == {"HTTPS_PROXY": "http://127.0.0.1:7897"}
    assert "UNRELATED" not in payload["proxy_env"]


def test_subscription_env_forces_oauth_and_keeps_loopback_off_proxy():
    with patch.dict(
        "os.environ",
        {
            "GEMINI_API_KEY": "must-not-be-used",
            "GOOGLE_API_KEY": "must-not-be-used",
            "NO_PROXY": "example.test",
        },
        clear=True,
    ):
        env = AntigravityMCPAgent._subscription_env()

    assert "GEMINI_API_KEY" not in env
    assert "GOOGLE_API_KEY" not in env
    assert "example.test" in env["NO_PROXY"]
    assert "127.0.0.1" in env["NO_PROXY"]
    assert "localhost" in env["NO_PROXY"]


def test_antigravity_stream_usage_and_result_are_parsed():
    output = "\n".join(
        [
            '{"event":"step_update","step_update":{"usage":'
            '{"input_tokens":4,"output_tokens":1}}}',
            '{"event":"result","result":{"status":"SUCCESS","usage":'
            '{"input_tokens":10,"output_tokens":3,"thinking_tokens":2}}}',
        ]
    )

    assert AntigravityMCPAgent._parse_usage(output) == (10, 3)
    assert AntigravityMCPAgent._has_successful_result(output) is True


def test_antigravity_requires_completed_pitbench_mcp_call():
    failed = json.dumps(
        {
            "event": "step_update",
            "step_update": {
                "state": "DONE",
                "tool_info": {
                    "name": "call_mcp_tool",
                    "parameters": {
                        "server_name": "pitbench",
                        "tool_name": "run_command",
                    },
                    "output": {"isError": True},
                },
            },
        }
    )
    completed = json.dumps(
        {
            "event": "step_update",
            "step_update": {
                "state": "DONE",
                "tool_info": {
                    "name": "call_mcp_tool",
                    "parameters": {
                        "server_name": "pitbench",
                        "tool_name": "run_command",
                    },
                    "output": {"exit_code": 0},
                },
            },
        }
    )
    native = json.dumps(
        {
            "event": "step_update",
            "step_update": {
                "state": "DONE",
                "tool_info": {
                    "name": "run_command",
                    "parameters": {"command": "pwd"},
                    "output": "empty host workspace",
                },
            },
        }
    )

    assert AntigravityMCPAgent._has_successful_mcp_call(failed) is False
    assert AntigravityMCPAgent._has_successful_mcp_call(native) is False
    assert AntigravityMCPAgent._has_successful_mcp_call(completed) is True

    permission_denied = json.dumps(
        {
            "event": "step_update",
            "step_update": {
                "state": "DONE",
                "tool_info": {
                    "name": "call_mcp_tool",
                    "parameters": {
                        "ServerName": "pitbench",
                        "ToolName": "run_command",
                    },
                    "error": {"message": "permission denied"},
                },
            },
        }
    )
    assert AntigravityMCPAgent._has_successful_mcp_call(permission_denied) is False


def test_factory_registers_host_antigravity_agent():
    assert AgentFactory.get_agent_class(AgentName.ANTIGRAVITY) is AntigravityMCPAgent
