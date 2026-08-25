import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pitbench.harness.agents import antigravity_container_runner as runner_module
from pitbench.harness.agents.antigravity_container import (
    AntigravityContainerRunner,
)
from pitbench.harness.agents.antigravity_profile import AntigravityProfile


def _executable(root: Path) -> Path:
    root.mkdir(parents=True)
    binary = root / "agy"
    binary.write_text("agy")
    binary.chmod(0o755)
    return binary


def _profile(root: Path) -> AntigravityProfile:
    config = root / "gemini-config"
    config.mkdir(parents=True)
    (root / "profile.yaml").write_text(
        'schema_version: "1.0"\n'
        "name: custom-agy\n"
        "gemini_config: gemini-config\n"
        "allow_hooks: true\n"
    )
    (config / "config.json").write_text(
        '{"plugins":{"custom":{"enabled":true}},'
        '"permissions":{"allow":["custom(tool)"]}}\n'
    )
    (config / "mcp_config.json").write_text(
        '{"mcpServers":{"custom":{"url":"http://127.0.0.1:9000/mcp"}}}\n'
    )
    (config / "hooks.json").write_text("{}\n")
    return AntigravityProfile.load(root)


def test_antigravity_container_mounts_only_runner_inputs(tmp_path: Path) -> None:
    agy = _executable(tmp_path / "bin")
    profile = _profile(tmp_path / "profile")
    runner = AntigravityContainerRunner("python:test", agy, profile)

    command = runner.command_prefix()
    mounts = [
        command[index + 1]
        for index, argument in enumerate(command)
        if argument == "--mount"
    ]

    assert command[:3] == ["docker", "run", "--rm"]
    assert "--read-only" in command
    assert ["--network", "host"] == command[
        command.index("--network") : command.index("--network") + 2
    ]
    assert "/var/run/docker.sock" not in " ".join(command)
    assert mounts == [
        runner._mount(agy, runner.CONTAINER_AGY),
        runner._mount(runner.runner_script, runner.CONTAINER_RUNNER),
        runner._mount(profile.gemini_config, runner.CONTAINER_PROFILE),
    ]


def test_antigravity_container_prepare_records_image(tmp_path: Path) -> None:
    agy = _executable(tmp_path / "bin")
    runner = AntigravityContainerRunner("python:test", agy)
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
        "pitbench.harness.agents.antigravity_container.subprocess.run",
        side_effect=responses,
    ):
        metadata = runner.prepare(pull=False)

    assert metadata["runner_image_id"] == "sha256:image"
    assert metadata["profile"] is None


def test_antigravity_entrypoint_merges_profile_and_injects_oauth(
    tmp_path: Path,
) -> None:
    agy = _executable(tmp_path / "bin")
    profile = _profile(tmp_path / "profile")
    observed: dict[str, object] = {}

    def execute(command, *, cwd, env, stdin, check):
        gemini = Path(env["HOME"]) / ".gemini"
        observed["command"] = command
        observed["token"] = json.loads(
            (gemini / "antigravity-cli/antigravity-oauth-token").read_text()
        )
        observed["config"] = json.loads((gemini / "config/config.json").read_text())
        observed["mcp"] = json.loads((gemini / "config/mcp_config.json").read_text())
        observed["hooks"] = (gemini / "config/hooks.json").read_text()
        return SimpleNamespace(returncode=0)

    payload = {
        "auth_token_json": (
            '{"auth_method":"consumer","token":{"access_token":"secret"}}'
        ),
        "settings_json": '{"security":{"auth":{"selectedType":"consumer"}}}',
        "proxy_env": {},
        "allow_hooks": True,
    }
    with (
        patch.object(runner_module, "AGY_BINARY", agy),
        patch.object(runner_module, "PROFILE_ROOT", profile.gemini_config),
        patch.object(runner_module.sys, "stdin", io.StringIO(json.dumps(payload))),
        patch.object(runner_module.subprocess, "run", side_effect=execute),
    ):
        assert (
            runner_module._run_agy("http://127.0.0.1:43210/mcp", ["--print", "work"])
            == 0
        )

    assert observed["token"]["token"]["access_token"] == "secret"
    assert observed["config"]["plugins"]["custom"]["enabled"] is True
    assert observed["config"]["permissions"]["allow"] == [
        "custom(tool)",
        "mcp(pitbench/run_command)",
    ]
    assert observed["mcp"]["mcpServers"]["custom"] == {
        "url": "http://127.0.0.1:9000/mcp"
    }
    assert observed["mcp"]["mcpServers"]["pitbench"] == {
        "url": "http://127.0.0.1:43210/mcp"
    }
    assert observed["hooks"] == "{}\n"
    assert "secret" not in " ".join(observed["command"])
