from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

from pitbench.cli.evaluate_config import SolverImageBuildConfig
from pitbench.repositories.base import RepositoryPluginRegistry
from pitbench.tasks import TaskCatalog

_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


@dataclass(frozen=True)
class BuiltSolverImage:
    task_id: str
    image_tag: str
    image_id: str


def _docker_network(config: SolverImageBuildConfig) -> str | None:
    if config.docker_network:
        return config.docker_network
    if config.proxy_url and urlsplit(config.proxy_url).hostname in _LOOPBACK_HOSTS:
        return "host"
    return None


def docker_build_command(
    *,
    repository_root: Path,
    dockerfile: Path,
    source_repository: str,
    base_commit: str,
    image_tag: str,
    config: SolverImageBuildConfig,
) -> list[str]:
    """Create a source-image build command without changing Docker defaults."""
    command = [
        "docker",
        "build",
        "--file",
        str(dockerfile),
        "--tag",
        image_tag,
        "--build-arg",
        f"SOURCE_REPOSITORY={source_repository}",
        "--build-arg",
        f"BASE_COMMIT={base_commit}",
    ]
    network = _docker_network(config)
    if network is not None:
        command.extend(["--network", network])
    if config.proxy_url:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            command.extend(["--build-arg", f"{key}={config.proxy_url}"])
    if config.no_proxy:
        for key in ("NO_PROXY", "no_proxy"):
            command.extend(["--build-arg", f"{key}={config.no_proxy}"])
    command.append(str(repository_root))
    return command


def build_solver_image(
    task_id: str,
    repository_root: Path,
    *,
    config: SolverImageBuildConfig,
    image_tag: str | None = None,
    progress: Callable[[str], None] = print,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> BuiltSolverImage:
    """Build one task's source image using optional machine-local network settings."""
    root = repository_root.resolve()
    record = TaskCatalog(root).validate_one(task_id)
    plugin = RepositoryPluginRegistry.load(record.task.repository.plugin)
    dockerfile = root / "docker" / "solver-images" / plugin.name / "Dockerfile"
    if not dockerfile.is_file():
        raise ValueError(f"{task_id}: missing solver image definition {dockerfile}")

    tag = image_tag or f"pitbench-{plugin.name}:{record.task.release.base_commit[:12]}"
    command = docker_build_command(
        repository_root=root,
        dockerfile=dockerfile,
        source_repository=record.task.repository.clone_url,
        base_commit=record.task.release.base_commit,
        image_tag=tag,
        config=config,
    )
    network = _docker_network(config)
    if config.proxy_url:
        mode = f"network={network}" if network is not None else "default network"
        progress(f"Building {task_id} as {tag} with configured proxy ({mode})")
    else:
        progress(f"Building {task_id} as {tag} with Docker defaults")
    run(command, cwd=root, check=True)
    inspected = run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", tag],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    image_id = inspected.stdout.strip()
    if not image_id:
        raise RuntimeError(f"{task_id}: Docker did not return an image ID for {tag}")
    return BuiltSolverImage(task_id=task_id, image_tag=tag, image_id=image_id)
