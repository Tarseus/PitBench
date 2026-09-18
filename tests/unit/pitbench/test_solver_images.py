from __future__ import annotations

import subprocess
from pathlib import Path

from pitbench.cli.evaluate_config import SolverImageBuildConfig
from pitbench.cli.solver_images import build_solver_image, docker_build_command

ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = ROOT / "docker/solver-images/ortools/Dockerfile"
BASE_COMMIT = "551ad10d94835c99e5e1e684500d3db398c0e345"


def _command(config: SolverImageBuildConfig) -> list[str]:
    return docker_build_command(
        repository_root=ROOT,
        dockerfile=DOCKERFILE,
        source_repository="https://github.com/google/or-tools.git",
        base_commit=BASE_COMMIT,
        image_tag="pitbench-ortools:test",
        config=config,
    )


def test_solver_image_build_defaults_to_unmodified_docker_network() -> None:
    command = _command(SolverImageBuildConfig())

    assert "--network" not in command
    assert not any("PROXY=" in value for value in command)
    assert "SOURCE_REPOSITORY=https://github.com/google/or-tools.git" in command
    assert f"BASE_COMMIT={BASE_COMMIT}" in command


def test_solver_image_build_uses_host_network_for_configured_loopback_proxy() -> None:
    command = _command(
        SolverImageBuildConfig(
            proxy_url="http://127.0.0.1:7897", no_proxy="localhost,127.0.0.1"
        )
    )

    assert command[command.index("--network") + 1] == "host"
    assert "HTTP_PROXY=http://127.0.0.1:7897" in command
    assert "HTTPS_PROXY=http://127.0.0.1:7897" in command
    assert "NO_PROXY=localhost,127.0.0.1" in command


def test_solver_image_build_leaves_default_network_for_non_loopback_proxy() -> None:
    command = _command(SolverImageBuildConfig(proxy_url="http://proxy.internal:3128"))

    assert "--network" not in command
    assert "HTTPS_PROXY=http://proxy.internal:3128" in command


def test_solver_image_build_uses_task_release_and_reports_local_image_id() -> None:
    commands: list[list[str]] = []
    progress: list[str] = []

    def run(command, **kwargs):
        commands.append(command)
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, "sha256:test-image\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    built = build_solver_image(
        "ortools_v9_15",
        ROOT,
        config=SolverImageBuildConfig(proxy_url="http://127.0.0.1:7897"),
        progress=progress.append,
        run=run,
    )

    assert built.image_tag == "pitbench-ortools:551ad10d9483"
    assert built.image_id == "sha256:test-image"
    assert commands[0][0:2] == ["docker", "build"]
    assert "BASE_COMMIT=551ad10d94835c99e5e1e684500d3db398c0e345" in commands[0]
    assert commands[0][commands[0].index("--network") + 1] == "host"
    assert "configured proxy" in progress[0]
