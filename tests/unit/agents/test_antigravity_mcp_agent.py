from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from pitbench.harness.agents import AgentFactory, AgentName
from pitbench.harness.agents.antigravity_container import (
    AntigravityContainerRunner,
)
from pitbench.harness.agents.antigravity_mcp_agent import AntigravityMCPAgent
from pitbench.harness.agents.failure_mode import FailureMode


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
    prompt = command[command.index("--print") + 1]
    assert "pitbench validate" in prompt
    assert "pitbench check" in prompt
    assert "Never change a test" in prompt


def test_antigravity_container_backend_loads_profile(tmp_path):
    auth_path = tmp_path / "antigravity-oauth-token"
    settings_path = tmp_path / "settings.json"
    auth_path.write_text('{"auth_method":"consumer","token":{"access_token":"test"}}')
    settings_path.write_text("{}")
    profile_config = tmp_path / "profile/gemini-config"
    profile_config.mkdir(parents=True)
    (tmp_path / "profile/profile.yaml").write_text(
        'schema_version: "1.0"\n'
        "name: custom\n"
        "gemini_config: gemini-config\n"
        "allow_hooks: true\n"
    )
    (profile_config / "config.json").write_text("{}\n")
    agent = AntigravityMCPAgent(
        model_name="gemini-3.7-flash-high",
        auth_token_path=str(auth_path),
        settings_path=str(settings_path),
        runner_backend="container",
        profile_path=str(tmp_path / "profile"),
    )
    container_runner = MagicMock(spec=AntigravityContainerRunner)
    container_runner.command_prefix.return_value = ["docker-runner"]
    agent._container_runner = container_runner

    command = agent._build_command(
        mcp_url="http://127.0.0.1:43210/mcp", instruction="Improve it."
    )
    payload = json.loads(agent._runner_payload({}))

    assert command[:2] == ["docker-runner", "run"]
    assert "--disable-slash-commands" not in command
    assert payload["allow_hooks"] is True
    assert payload["profile_sha256"] == agent._profile.sha256


def test_antigravity_host_backend_rejects_profile(tmp_path):
    with pytest.raises(ValueError, match="profiles require runner_backend=container"):
        AntigravityMCPAgent(
            model_name="gemini-3.7-flash-high",
            runner_backend="host",
            profile_path=str(tmp_path),
        )


def test_runner_payload_copies_only_antigravity_auth_and_proxy(tmp_path):
    auth_token_path = tmp_path / "antigravity-oauth-token"
    settings_path = tmp_path / "settings.json"
    auth_token_path.write_text(
        '{"auth_method":"consumer","token":{"access_token":"secret"}}'
    )
    settings_path.write_text('{"agentMode":"auto","toolPermission":"always-proceed"}')
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
    runner_settings = json.loads(payload["settings_json"])
    assert runner_settings["agentMode"] == "auto"
    assert runner_settings["security"]["auth"]["selectedType"] == "consumer"
    assert "security" not in json.loads(settings_path.read_text())
    assert payload["proxy_env"] == {"HTTPS_PROXY": "http://127.0.0.1:7897"}
    assert "UNRELATED" not in payload["proxy_env"]


def test_subscription_env_forces_oauth_and_defaults_to_no_proxy():
    with patch.dict(
        "os.environ",
        {
            "GEMINI_API_KEY": "must-not-be-used",
            "GOOGLE_API_KEY": "must-not-be-used",
            "HTTPS_PROXY": "http://ambient-proxy:8080",
            "NO_PROXY": "example.test",
        },
        clear=True,
    ):
        env = AntigravityMCPAgent._subscription_env()

    assert "GEMINI_API_KEY" not in env
    assert "GOOGLE_API_KEY" not in env
    assert "HTTPS_PROXY" not in env
    assert env["NO_PROXY"] == "127.0.0.1,localhost"


def test_runtime_env_applies_explicit_proxy_to_antigravity_runner():
    agent = AntigravityMCPAgent(
        model_name="gemini-3.7-flash-high",
        proxy_url="http://127.0.0.1:17898",
    )

    with patch.dict("os.environ", {}, clear=True):
        env = agent._runtime_env()

    assert env["HTTP_PROXY"] == "http://127.0.0.1:17898"
    assert env["HTTPS_PROXY"] == "http://127.0.0.1:17898"
    assert env["ALL_PROXY"] == "http://127.0.0.1:17898"
    assert env["NO_PROXY"] == "127.0.0.1,localhost"


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


def test_antigravity_combines_retry_usage_and_detects_network_errors():
    failed = json.dumps(
        {
            "event": "result",
            "result": {
                "status": "ERROR",
                "error": "There was a network issue connecting to the server",
                "usage": {"input_tokens": 10, "output_tokens": 3},
            },
        }
    )
    succeeded = json.dumps(
        {
            "event": "result",
            "result": {
                "status": "SUCCESS",
                "usage": {"input_tokens": 7, "output_tokens": 2},
            },
        }
    )

    assert AntigravityMCPAgent._has_retryable_network_error(failed, "") is True
    assert AntigravityMCPAgent._has_retryable_network_error(succeeded, "") is False
    assert AntigravityMCPAgent._parse_usage(f"{failed}\n{succeeded}") == (17, 5)


def test_antigravity_rejects_more_than_one_network_retry():
    with pytest.raises(ValueError, match="network_retries must be 0 or 1"):
        AntigravityMCPAgent(model_name="gemini-3.7-flash-high", network_retries=2)


def test_antigravity_retries_network_failure_in_existing_workspace(tmp_path):
    failed = json.dumps(
        {
            "event": "result",
            "result": {
                "status": "ERROR",
                "error": "The stream was interrupted. Please continue the task.",
                "usage": {"input_tokens": 10, "output_tokens": 3},
            },
        }
    )
    completed_tool = json.dumps(
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
    succeeded = "\n".join(
        [
            completed_tool,
            json.dumps(
                {
                    "event": "result",
                    "result": {
                        "status": "SUCCESS",
                        "usage": {"input_tokens": 7, "output_tokens": 2},
                    },
                }
            ),
        ]
    )
    first_process = MagicMock(returncode=0)
    first_process.communicate.return_value = (failed, "")
    second_process = MagicMock(returncode=0)
    second_process.communicate.return_value = (succeeded, "")
    agent = AntigravityMCPAgent(model_name="gemini-3.7-flash-high")
    agent._resolved_agy_binary = MagicMock(return_value="/usr/bin/agy")
    agent._runtime_env = MagicMock(return_value={})
    agent._preflight = MagicMock()
    agent._runner_payload = MagicMock(return_value="{}")

    with (
        patch(
            "pitbench.harness.agents.antigravity_mcp_agent.LoopbackMCPServer"
        ) as server_class,
        patch(
            "pitbench.harness.agents.antigravity_mcp_agent.subprocess.Popen",
            side_effect=[first_process, second_process],
        ) as popen,
    ):
        server_class.return_value.__enter__.return_value.url = (
            "http://127.0.0.1:43210/mcp"
        )
        result = agent.perform_task(
            "Improve it.",
            SimpleNamespace(container=MagicMock()),
            logging_dir=tmp_path,
        )

    assert popen.call_count == 2
    retry_command = popen.call_args_list[1].args[0]
    retry_prompt = retry_command[retry_command.index("--print") + 1]
    assert "Continue from the existing task repository state" in retry_prompt
    assert result.failure_mode == FailureMode.NONE
    assert (result.total_input_tokens, result.total_output_tokens) == (17, 5)


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
