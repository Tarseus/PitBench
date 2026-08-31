from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from adapters.pitbench.adapter import (
    IMAGE_REVISION,
    IMAGE_REVISION_LABEL,
    IMAGE_SOURCE_LABEL,
    IMAGE_TOOLS_LABEL,
)
from pitbench.agent_tools import AGENT_TOOLS_METADATA, agent_tools_label
from pitbench.harness.handlers.trial_handler import TrialHandler
from pitbench.harness.terminal.docker_compose_manager import DockerComposeManager


def prepare_task_image(
    task_path: Path,
    *,
    rebuild: bool,
    progress: Callable[[str], None] = print,
    trial_handler_type: Callable[..., Any] = TrialHandler,
    manager_type: Callable[..., Any] = DockerComposeManager,
) -> None:
    trial = trial_handler_type(
        trial_name="pitbench-image-prepare", input_path=task_path
    )
    manager = manager_type(
        client_container_name=trial.client_container_name,
        client_image_name=trial.client_image_name,
        docker_image_name_prefix=trial.docker_image_name_prefix,
        docker_compose_path=trial.task_paths.docker_compose_path,
        sessions_logs_path=task_path / ".image-prepare/sessions",
        agent_logs_path=task_path / ".image-prepare/agent-logs",
    )
    expected_source = (
        (task_path / "Dockerfile").read_text().splitlines()[0].removeprefix("FROM ")
    )
    tools_metadata = task_path / AGENT_TOOLS_METADATA
    expected_tools = agent_tools_label(
        json.loads(tools_metadata.read_text()).get("agent_tools", [])
        if tools_metadata.is_file()
        else []
    )
    cached_revision = manager.image_label(IMAGE_REVISION_LABEL)
    cached_source = manager.image_label(IMAGE_SOURCE_LABEL)
    cached_tools = manager.image_label(IMAGE_TOOLS_LABEL)
    if (
        rebuild
        or cached_revision != IMAGE_REVISION
        or cached_source != expected_source
        or cached_tools != expected_tools
    ):
        progress(f"Building task image {trial.client_image_name}")
        manager.build()
    else:
        progress(f"Reusing task image {trial.client_image_name}")
