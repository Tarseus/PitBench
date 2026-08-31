import json
import stat
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import docker.errors
from typer.testing import CliRunner

from pitbench.cli.doctor import (
    CheckStatus,
    DoctorCheck,
    _credential_check,
    _docker_check,
    _image_reference_check,
    _registry_and_proxy_checks,
    _runner_check,
    run_doctor,
)
from pitbench.cli.main import app
from pitbench.harness.agents.antigravity_container import (
    AntigravityContainerRunner,
)
from pitbench.harness.agents.codex_container import CodexContainerRunner


def test_docker_check_calls_api_and_explains_stale_process_groups() -> None:
    def denied():
        raise docker.errors.DockerException(
            "Connection aborted: PermissionError(13, 'Permission denied')"
        )

    client, _, result = _docker_check(denied)

    assert client is None
    assert result.status == CheckStatus.FAIL
    assert "current process" in result.detail
    assert "stale supplementary groups" in result.recovery
    assert "newgrp docker" in result.recovery


def test_docker_check_requires_a_real_server_round_trip() -> None:
    client = Mock()
    client.info.return_value = {"ServerVersion": "28.3.3"}

    actual, info, result = _docker_check(lambda: client)

    assert actual is client
    assert info == {"ServerVersion": "28.3.3"}
    client.ping.assert_called_once_with()
    assert result == DoctorCheck(
        CheckStatus.PASS, "Docker API", "server 28.3.3 answered ping"
    )


def test_image_references_match_agent_and_judge_runtime_constraints() -> None:
    digest = "a" * 64

    agent = _image_reference_check("agent", f"sha256:{digest}")
    judge = _image_reference_check("judge", "pitbench/judge:latest")

    assert agent is not None and agent.status == CheckStatus.FAIL
    assert "Dockerfile FROM" in agent.detail
    assert judge is not None and judge.status == CheckStatus.FAIL
    assert _image_reference_check("agent", "pitbench/agent:local") is None
    assert _image_reference_check("judge", f"sha256:{digest}") is None
    assert _image_reference_check("judge", f"ghcr.io/org/judge@sha256:{digest}") is None


def test_runner_self_test_rejects_dedicated_user_with_docker_access(
    tmp_path: Path,
) -> None:
    runner = tmp_path / "runner"
    runner.write_text("runner")
    metadata = SimpleNamespace(st_uid=0, st_mode=stat.S_IFREG | 0o755)
    completed = SimpleNamespace(
        returncode=1,
        stdout=json.dumps(
            {"uid": 123, "groups": ["docker"], "docker_socket_access": True}
        ),
        stderr="",
    )

    with (
        patch.object(Path, "stat", return_value=metadata),
        patch("pitbench.cli.doctor.os.access", return_value=True),
        patch(
            "pitbench.cli.doctor.pwd.getpwnam",
            return_value=SimpleNamespace(pw_uid=123),
        ),
        patch("pitbench.cli.doctor.subprocess.run", return_value=completed),
    ):
        result, ready = _runner_check(
            "codex", {"runner_path": str(runner), "runner_user": "isolated"}
        )

    assert ready is False
    assert result.status == CheckStatus.WARN
    assert result.detail == "dedicated user isolated still has Docker access"


def test_codex_container_runner_needs_no_administrator(tmp_path: Path) -> None:
    codex = tmp_path / "codex"
    codex.write_text("binary")
    profile_home = tmp_path / "profile/codex-home"
    profile_home.mkdir(parents=True)
    (tmp_path / "profile/profile.yaml").write_text(
        'schema_version: "1.0"\n'
        "name: reproducible\n"
        "codex_home: codex-home\n"
        "allow_hooks: false\n"
    )

    with (
        patch("pitbench.cli.doctor.shutil.which", return_value=str(codex)),
        patch.object(
            CodexContainerRunner,
            "prepare",
            return_value={"runner_image_id": "sha256:runner"},
        ) as prepare,
    ):
        result, ready = _runner_check(
            "codex",
            {
                "runner_backend": "container",
                "container_runner_image": "python:test",
                "profile_path": str(tmp_path / "profile"),
            },
        )

    assert ready is True
    assert result.status == CheckStatus.PASS
    assert "sha256:runner" in result.detail
    assert "reproducible@" in result.detail
    assert result.recovery is None
    prepare.assert_called_once_with(pull=False)


def test_codex_workspace_runner_accepts_profile_plugins(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle/bin"
    bundle.mkdir(parents=True)
    codex = bundle / "codex"
    codex.write_text("binary")
    codex.chmod(0o755)
    code_mode_host = bundle / "codex-code-mode-host"
    code_mode_host.write_text("host")
    code_mode_host.chmod(0o755)
    resources = tmp_path / "bundle/codex-resources"
    resources.mkdir()
    bubblewrap = resources / "bwrap"
    bubblewrap.write_text("bwrap")
    bubblewrap.chmod(0o755)
    profile_home = tmp_path / "profile/codex-home"
    profile_home.mkdir(parents=True)
    (tmp_path / "profile/profile.yaml").write_text(
        'schema_version: "1.0"\n'
        "name: plugins\n"
        "codex_home: codex-home\n"
        "allow_hooks: false\n"
    )
    (profile_home / "plugins.json").write_text("{}\n")

    with patch("pitbench.cli.doctor.shutil.which", return_value=str(codex)):
        result, ready = _runner_check(
            "codex",
            {
                "runner_backend": "workspace",
                "profile_path": str(tmp_path / "profile"),
            },
        )

    assert ready is True
    assert result.status == CheckStatus.PASS
    assert "profile plugins@" in result.detail


def test_antigravity_container_runner_needs_no_administrator(
    tmp_path: Path,
) -> None:
    agy = tmp_path / "agy"
    agy.write_text("binary")
    profile_config = tmp_path / "profile/gemini-config"
    profile_config.mkdir(parents=True)
    (tmp_path / "profile/profile.yaml").write_text(
        'schema_version: "1.0"\n'
        "name: reproducible-agy\n"
        "gemini_config: gemini-config\n"
        "allow_hooks: false\n"
    )

    with (
        patch("pitbench.cli.doctor.shutil.which", return_value=str(agy)),
        patch.object(
            AntigravityContainerRunner,
            "prepare",
            return_value={"runner_image_id": "sha256:runner"},
        ) as prepare,
    ):
        result, ready = _runner_check(
            "antigravity",
            {
                "runner_backend": "container",
                "container_runner_image": "python:test",
                "profile_path": str(tmp_path / "profile"),
            },
        )

    assert ready is True
    assert result.status == CheckStatus.PASS
    assert "sha256:runner" in result.detail
    assert "reproducible-agy@" in result.detail
    assert result.recovery is None
    prepare.assert_called_once_with(pull=False)


def test_antigravity_credentials_use_supported_models_command(
    tmp_path: Path,
) -> None:
    token = tmp_path / "token.json"
    settings = tmp_path / "settings.json"
    token.write_text(json.dumps({"auth_method": "oauth", "token": {}}))
    settings.write_text("{}")
    completed = SimpleNamespace(returncode=0, stdout="gemini\n", stderr="")

    with (
        patch("pitbench.cli.doctor.shutil.which", return_value="/usr/bin/agy"),
        patch("pitbench.cli.doctor.subprocess.run", return_value=completed) as run,
    ):
        result, ready = _credential_check(
            "antigravity",
            {
                "auth_token_path": str(token),
                "settings_path": str(settings),
            },
        )

    assert ready is True
    assert result.status == CheckStatus.PASS
    assert run.call_args.args[0] == ["agy", "models"]


def test_docker_proxy_accepts_engine_api_key_casing() -> None:
    client = Mock()

    with patch.dict("pitbench.cli.doctor.os.environ", {}, clear=True):
        checks = _registry_and_proxy_checks(
            client, {"HttpProxy": "http://127.0.0.1:17898"}, []
        )

    assert checks[-1] == DoctorCheck(
        CheckStatus.PASS,
        "Docker proxy",
        "daemon proxy: http://127.0.0.1:17898",
    )


def test_doctor_rejects_unknown_profile(tmp_path: Path) -> None:
    try:
        run_doctor("unknown", tmp_path)
    except ValueError as error:
        assert str(error) == "unsupported doctor profile: unknown"
    else:
        raise AssertionError("unknown profile was accepted")


def test_doctor_cli_returns_nonzero_when_any_check_fails() -> None:
    runner = CliRunner()
    checks = [
        DoctorCheck(CheckStatus.PASS, "platform", "Linux x86_64"),
        DoctorCheck(
            CheckStatus.FAIL,
            "Docker API",
            "permission denied",
            "Start a fresh login.",
        ),
    ]

    with patch("pitbench.cli.main.run_doctor", return_value=checks):
        result = runner.invoke(app, ["doctor", "pyvrp"])

    assert result.exit_code == 1
    assert "PASS platform: Linux x86_64" in result.output
    assert "FAIL Docker API: permission denied" in result.output
    assert "Recovery: Start a fresh login." in result.output
    assert "NOT READY: 1 failure(s)" in result.output


def test_doctor_cli_succeeds_with_warnings() -> None:
    runner = CliRunner()
    checks = [DoctorCheck(CheckStatus.WARN, "antigravity runner", "not installed")]

    with patch("pitbench.cli.main.run_doctor", return_value=checks):
        result = runner.invoke(app, ["doctor", "pyvrp"])

    assert result.exit_code == 0
    assert (
        "READY: PyVRP evaluation prerequisites passed (1 warning(s))" in result.output
    )
