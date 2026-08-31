from types import SimpleNamespace

from pitbench.harness.agents.codex_relay import _RelayBudget, _RelayHandler


def test_relay_rejects_server_side_network_tools() -> None:
    assert _RelayHandler._has_disallowed_remote_tool({"tools": []}) is False
    assert (
        _RelayHandler._has_disallowed_remote_tool(
            {"tools": [{"type": "web_search_preview"}]}
        )
        is True
    )
    assert (
        _RelayHandler._has_disallowed_remote_tool(
            {"tools": [{"type": "computer_use_preview"}]}
        )
        is True
    )
    assert _RelayHandler._has_disallowed_remote_tool({"tools": "invalid"}) is True


def test_relay_rejects_remote_model_input_resources() -> None:
    assert _RelayHandler._has_remote_resource_reference(
        {"input": [{"type": "input_image", "image_url": "https://example.com/x"}]}
    )
    assert _RelayHandler._has_remote_resource_reference(
        {"input": [{"type": "input_file", "file_url": "http://example.com/x"}]}
    )
    assert not _RelayHandler._has_remote_resource_reference(
        {"input": [{"type": "input_image", "image_url": "data:image/png;base64,AA=="}]}
    )


def test_relay_loads_only_required_oauth_fields(tmp_path) -> None:
    auth = tmp_path / "auth.json"
    auth.write_text(
        '{"tokens":{"access_token":"access","account_id":"account",'
        '"refresh_token":"refresh"}}'
    )
    handler = object.__new__(_RelayHandler)
    handler.server = SimpleNamespace(auth_path=auth)

    assert handler._load_auth() == ("access", "account")


def test_relay_removes_shell_escalation_from_model_schema() -> None:
    tools = [
        {
            "type": "function",
            "name": "exec_command",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string"},
                    "sandbox_permissions": {
                        "type": "string",
                        "enum": ["use_default", "require_escalated"],
                    },
                },
            },
        }
    ]

    assert _RelayHandler._remove_shell_escalation(tools) == 1
    assert tools[0]["parameters"]["properties"]["sandbox_permissions"] == {
        "type": "string",
        "enum": ["use_default"],
        "description": "PitBench requires the existing workspace sandbox.",
    }


def test_relay_blocks_only_escalating_tool_call_events() -> None:
    reasoning = b'data: {"type":"response.reasoning.delta","delta":"require_escalated"}'
    tool_call = (
        b'data: {"type":"response.custom_tool_call_input.delta",'
        b'"delta":"require_escalated"}'
    )

    assert _RelayHandler._event_requests_shell_escalation(reasoning) is False
    assert _RelayHandler._event_requests_shell_escalation(tool_call) is True


def test_relay_extracts_streamed_token_usage() -> None:
    event = (
        b'data: {"type":"response.completed","response":{"usage":'
        b'{"input_tokens":12,"output_tokens":7,"total_tokens":19}}}'
    )

    assert _RelayHandler._event_total_tokens(event) == 19


def test_relay_budget_enforces_requests_tokens_and_concurrency() -> None:
    budget = _RelayBudget(
        max_requests=2,
        max_total_tokens=100,
        max_concurrent_requests=1,
        max_duration_sec=60,
    )

    assert budget.reserve() is None
    assert budget.reserve() == "relay concurrency budget exhausted"
    budget.finish(40)
    assert budget.reserve() is None
    budget.finish(60)
    assert budget.reserve() == "relay request budget exhausted"
    assert budget.snapshot()["total_tokens"] == 100


def test_relay_budget_enforces_token_and_duration_limits() -> None:
    token_budget = _RelayBudget(
        max_requests=3,
        max_total_tokens=10,
        max_concurrent_requests=1,
        max_duration_sec=60,
    )
    assert token_budget.reserve() is None
    token_budget.finish(10)
    assert token_budget.reserve() == "relay token budget exhausted"

    duration_budget = _RelayBudget(
        max_requests=3,
        max_total_tokens=100,
        max_concurrent_requests=1,
        max_duration_sec=1,
    )
    duration_budget.started_at -= 2
    assert duration_budget.reserve() == "relay duration budget exhausted"
