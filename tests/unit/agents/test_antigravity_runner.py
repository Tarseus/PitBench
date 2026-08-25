from __future__ import annotations

import json
import runpy
import stat
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

RUNNER = runpy.run_path(
    str(Path(__file__).parents[3] / "scripts" / "pitbench-antigravity-runner")
)


def test_runner_accepts_only_loopback_pitbench_mcp_url():
    validate = RUNNER["_validate_mcp_url"]

    assert validate("http://127.0.0.1:43210/mcp") == "http://127.0.0.1:43210/mcp"
    for invalid in (
        "https://127.0.0.1:43210/mcp",
        "http://localhost:43210/mcp",
        "http://127.0.0.1:43210/other",
        "http://example.test:43210/mcp",
    ):
        with pytest.raises(ValueError):
            validate(invalid)


def test_runner_rejects_unknown_operation_and_unbounded_agy_flags():
    parse_command = RUNNER["_parse_command"]
    base = [
        "run",
        "--mcp-url",
        "http://127.0.0.1:43210/mcp",
        "--",
        "--print",
        "do it",
        "--output-format",
        "stream-json",
        "--model",
        "gemini-3.7-flash-low",
        "--mode",
        "accept-edits",
        "--sandbox",
        "--disable-slash-commands",
        "--print-timeout",
        "5m",
    ]

    _, arguments = parse_command(base)
    assert "--sandbox" in arguments
    with pytest.raises(ValueError):
        parse_command(["exec"])
    with pytest.raises(SystemExit):
        parse_command([*base, "--dangerously-skip-permissions"])
    with pytest.raises(SystemExit):
        parse_command([*base, "--add-dir", "/home/gsd"])


def test_runner_writes_private_json_with_owner_only_permissions(tmp_path):
    target = tmp_path / "nested" / "token.json"

    RUNNER["_write_private_json"](target, {"secret": "value"})

    assert target.stat().st_mode & 0o777 == 0o600
    assert target.parent.stat().st_mode & 0o777 == 0o700


def test_runner_installs_exact_mcp_tool_permission(tmp_path, monkeypatch):
    binary = tmp_path / "agy"
    binary.touch(mode=stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    runner_home = tmp_path / "runner-home"
    payload = {
        "auth_token_json": (
            '{"auth_method":"consumer","token":{"access_token":"test"}}'
        ),
        "settings_json": (
            '{"agentMode":"auto","toolPermission":"always-proceed",'
            '"trustedWorkspaces":["/host/workspace"],'
            '"security":{"auth":{"selectedType":"consumer"}}}'
        ),
    }
    invocation = {}

    def fake_run(arguments, **kwargs):
        invocation["arguments"] = arguments
        invocation.update(kwargs)
        return SimpleNamespace(returncode=0)

    runner_globals = RUNNER["_run_agy"].__globals__
    monkeypatch.setitem(runner_globals, "AGY_BINARY", binary)
    monkeypatch.setitem(
        runner_globals,
        "sys",
        SimpleNamespace(stdin=StringIO(json.dumps(payload))),
    )
    monkeypatch.setitem(
        runner_globals,
        "tempfile",
        SimpleNamespace(
            TemporaryDirectory=lambda **_: _TemporaryDirectory(runner_home)
        ),
    )
    monkeypatch.setitem(
        runner_globals,
        "subprocess",
        SimpleNamespace(run=fake_run, DEVNULL=-1),
    )

    assert RUNNER["_run_agy"]("http://127.0.0.1:43210/mcp", ["--print", "test"]) == 0

    shared = json.loads(
        (runner_home / ".gemini" / "config" / "config.json").read_text()
    )
    cli = json.loads(
        (runner_home / ".gemini" / "antigravity-cli" / "settings.json").read_text()
    )
    token = json.loads(
        (
            runner_home
            / ".gemini"
            / "antigravity-cli"
            / "antigravity-oauth-token"
        ).read_text()
    )
    assert shared == {"permissions": {"allow": ["mcp(pitbench/run_command)"]}}
    assert cli["toolPermission"] == "request-review"
    assert "permissions" not in cli
    assert token["auth_method"] == "consumer"
    legacy = json.loads((runner_home / ".gemini" / "settings.json").read_text())
    assert legacy["security"]["auth"]["selectedType"] == "consumer"
    assert invocation["stdin"] == -1
    assert all(
        path.stat().st_mode & 0o777 == 0o600
        for path in (
            runner_home / ".gemini" / "config" / "config.json",
            runner_home / ".gemini" / "antigravity-cli" / "settings.json",
        )
    )


class _TemporaryDirectory:
    def __init__(self, path: Path):
        self.path = path

    def __enter__(self) -> str:
        self.path.mkdir()
        return str(self.path)

    def __exit__(self, *_args) -> None:
        return None
