import subprocess
from pathlib import Path
from unittest.mock import Mock

import docker.errors
import pytest

from pitbench.harness.terminal.docker_compose_manager import DockerComposeManager


def _make_manager() -> DockerComposeManager:
    manager = object.__new__(DockerComposeManager)
    manager._client_container_name = "test-container"
    manager._docker_compose_path = Path("docker-compose.yaml")
    manager._logger = Mock()
    manager.env = {}
    return manager


def test_run_docker_compose_command_raises_runtime_error_with_stderr(monkeypatch):
    manager = _make_manager()

    def _raise_called_process_error(*args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=args[0],
            output="compose stdout",
            stderr="compose stderr",
        )

    monkeypatch.setattr(subprocess, "run", _raise_called_process_error)

    with pytest.raises(RuntimeError, match="compose stderr"):
        manager._run_docker_compose_command(["up", "-d"])

    assert manager._logger.error.call_count == 1


def test_run_docker_compose_command_adds_address_pool_hint(monkeypatch):
    manager = _make_manager()

    def _raise_called_process_error(*args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=args[0],
            stderr=(
                "failed to create network some-network: Error response from daemon: "
                "all predefined address pools have been fully subnetted"
            ),
        )

    monkeypatch.setattr(subprocess, "run", _raise_called_process_error)

    with pytest.raises(RuntimeError) as exc_info:
        manager._run_docker_compose_command(["up", "-d"])

    message = str(exc_info.value)
    assert "docker network prune -f" in message
    assert "--n-concurrent-trials" in message


def test_image_label_returns_cached_value() -> None:
    manager = _make_manager()
    manager._client_image_name = "pitbench-task"
    manager._client = Mock()
    manager._client.images.get.return_value.labels = {"revision": "current"}

    assert manager.image_label("revision") == "current"
    manager._client.images.get.assert_called_once_with("pitbench-task")


def test_image_label_returns_none_when_image_is_missing() -> None:
    manager = _make_manager()
    manager._client_image_name = "pitbench-task"
    manager._client = Mock()
    manager._client.images.get.side_effect = docker.errors.ImageNotFound("missing")

    assert manager.image_label("revision") is None


def test_nested_sandbox_uses_scoped_compose_override() -> None:
    manager = _make_manager()
    manager._nested_sandbox = True
    manager._compose_override_path = None

    command = manager.get_docker_compose_command(["up", "-d"])
    override = manager._compose_override_path

    assert override is not None
    assert command.count("-f") == 2
    assert str(override) in command
    content = override.read_text()
    assert "cap_drop:\n      - ALL" in content
    assert "SETFCAP" in content
    assert "no-new-privileges:true" in content
    assert "seccomp=unconfined" in content
    assert "apparmor=unconfined" in content
    override.unlink()


def test_agent_identity_cannot_be_overridden_to_root(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("T_BENCH_AGENT_UID", "0")
    monkeypatch.setenv("T_BENCH_AGENT_GID", "0")
    monkeypatch.setattr("docker.from_env", Mock(return_value=Mock()))

    manager = DockerComposeManager(
        client_container_name="test-container",
        client_image_name="test-image",
        docker_compose_path=tmp_path / "docker-compose.yaml",
    )

    assert manager.env["T_BENCH_AGENT_UID"] != "0"
    assert manager.env["T_BENCH_AGENT_GID"] != "0"
