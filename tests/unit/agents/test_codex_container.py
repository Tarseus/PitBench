import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pitbench.harness.agents import codex_container_runner as runner_module
from pitbench.harness.agents.codex_container import CodexContainerRunner
from pitbench.harness.agents.codex_profile import CodexProfile


def _executables(root: Path) -> Path:
    root.mkdir(parents=True)
    binary = root / "codex"
    binary.write_text("codex")
    binary.chmod(0o755)
    host = root / "codex-code-mode-host"
    host.write_text("host")
    host.chmod(0o755)
    return binary


def _profile(root: Path) -> CodexProfile:
    home = root / "codex-home"
    home.mkdir(parents=True)
    (root / "profile.yaml").write_text(
        'schema_version: "1.0"\n'
        "name: hooks\n"
        "codex_home: codex-home\n"
        "allow_hooks: true\n"
    )
    (home / "config.toml").write_text("# custom\n")
    return CodexProfile.load(root)


def test_container_command_mounts_only_runner_inputs(tmp_path: Path) -> None:
    codex = _executables(tmp_path / "bin")
    profile = _profile(tmp_path / "profile")
    runner = CodexContainerRunner("python:test", codex, profile)

    command = runner.command_prefix()

    assert command[:3] == ["docker", "run", "--rm"]
    assert ["--network", "host"] == command[
        command.index("--network") : command.index("--network") + 2
    ]
    assert "--read-only" in command
    assert "/var/run/docker.sock" not in " ".join(command)
    mounts = [
        command[index + 1]
        for index, argument in enumerate(command)
        if argument == "--mount"
    ]
    assert mounts == [
        runner._mount(codex, runner.CONTAINER_CODEX),
        runner._mount(
            codex.parent / "codex-code-mode-host", runner.CONTAINER_CODE_MODE_HOST
        ),
        runner._mount(runner.runner_script, runner.CONTAINER_RUNNER),
        runner._mount(profile.codex_home, runner.CONTAINER_PROFILE),
    ]
    assert command[-3:] == [
        "python:test",
        "python3",
        "/opt/pitbench/codex_container_runner.py",
    ]


def test_container_prepare_records_image_and_rejects_socket(tmp_path: Path) -> None:
    codex = _executables(tmp_path / "bin")
    runner = CodexContainerRunner("python:test", codex)
    responses = [
        SimpleNamespace(returncode=0, stdout="sha256:image\n", stderr=""),
        SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "docker_socket_present": False,
                    "docker_socket_access": False,
                    "profile_mounted": False,
                }
            ),
            stderr="",
        ),
    ]

    with patch(
        "pitbench.harness.agents.codex_container.subprocess.run", side_effect=responses
    ):
        metadata = runner.prepare(pull=False)

    assert metadata["runner_image_id"] == "sha256:image"
    assert metadata["profile"] is None


def test_container_entrypoint_copies_profile_and_injects_oauth(tmp_path: Path) -> None:
    codex = _executables(tmp_path / "bin")
    profile = _profile(tmp_path / "profile")
    observed: dict[str, object] = {}

    def execute(command, *, cwd, env, stdin, check):
        codex_home = Path(env["CODEX_HOME"])
        observed["command"] = command
        observed["auth"] = json.loads((codex_home / "auth.json").read_text())
        observed["config"] = (codex_home / "config.toml").read_text()
        observed["home"] = env["HOME"]
        return SimpleNamespace(returncode=0)

    payload = {
        "auth_json": '{"tokens":{"access_token":"secret"}}',
        "proxy_env": {},
        "allow_hooks": True,
    }
    with (
        patch.object(runner_module, "CODEX_BINARY", codex),
        patch.object(
            runner_module,
            "CODE_MODE_HOST",
            codex.parent / "codex-code-mode-host",
        ),
        patch.object(runner_module, "PROFILE_ROOT", profile.codex_home),
        patch.object(runner_module.sys, "stdin", io.StringIO(json.dumps(payload))),
        patch.object(runner_module.subprocess, "run", side_effect=execute),
    ):
        assert runner_module._run_codex(["exec", "--json", "--", "work"]) == 0

    assert observed["auth"] == {"tokens": {"access_token": "secret"}}
    assert observed["config"] == "# custom\n"
    assert "--dangerously-bypass-hook-trust" in observed["command"]
    assert "secret" not in " ".join(observed["command"])
