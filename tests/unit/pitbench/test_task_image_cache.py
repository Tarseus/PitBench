from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from adapters.pitbench.adapter import (
    IMAGE_CONTEXT_LABEL,
    IMAGE_REVISION,
    IMAGE_REVISION_LABEL,
    IMAGE_SOURCE_LABEL,
    IMAGE_TOOLS_LABEL,
    image_context_digest,
)
from pitbench.cli.task_image import prepare_task_image


@pytest.mark.parametrize(
    ("source", "cached_source", "force_rebuild", "should_build"),
    [
        (
            "registry.example/solver@sha256:" + "b" * 64,
            "registry.example/solver@sha256:" + "b" * 64,
            False,
            False,
        ),
        ("sha256:" + "a" * 64, "sha256:" + "a" * 64, False, True),
        ("solver:latest", "solver:latest", False, True),
        (
            "registry.example/solver@sha256:" + "b" * 64,
            "registry.example/solver@sha256:" + "c" * 64,
            False,
            True,
        ),
        (
            "registry.example/solver@sha256:" + "b" * 64,
            "registry.example/solver@sha256:" + "b" * 64,
            True,
            True,
        ),
    ],
)
def test_task_image_reuses_only_matching_immutable_sources(
    tmp_path: Path,
    source: str,
    cached_source: str,
    force_rebuild: bool,
    should_build: bool,
) -> None:
    (tmp_path / "Dockerfile").write_text(f"FROM {source}\n")
    (tmp_path / ".dockerignore").write_text("*\n!Dockerfile\n")
    (tmp_path / "agent_tools.json").write_text(json.dumps({"agent_tools": ["bench"]}))
    context_digest = image_context_digest(tmp_path)
    with (tmp_path / "Dockerfile").open("a") as dockerfile:
        dockerfile.write(f'LABEL {IMAGE_CONTEXT_LABEL}="{context_digest}"\n')
    labels = {
        IMAGE_REVISION_LABEL: IMAGE_REVISION,
        IMAGE_SOURCE_LABEL: cached_source,
        IMAGE_TOOLS_LABEL: "bench",
        IMAGE_CONTEXT_LABEL: context_digest,
    }
    builds: list[bool] = []

    class Trial:
        def __init__(self, *, trial_name: str, input_path: Path) -> None:
            self.client_container_name = "test-container"
            self.client_image_name = "test-image"
            self.docker_image_name_prefix = "test"
            self.task_paths = SimpleNamespace(
                docker_compose_path=input_path / "docker-compose.yaml"
            )

    class Manager:
        def __init__(self, **_: object) -> None:
            pass

        def image_label(self, label: str) -> str | None:
            return labels.get(label)

        def build(self) -> None:
            builds.append(True)

    prepare_task_image(
        tmp_path,
        rebuild=force_rebuild,
        progress=lambda _: None,
        trial_handler_type=Trial,
        manager_type=Manager,
    )
    assert bool(builds) is should_build
    if not should_build:
        (tmp_path / "agent_config.json").write_text('{"changed": true}\n')
        with pytest.raises(ValueError, match="context changed"):
            prepare_task_image(
                tmp_path,
                rebuild=False,
                progress=lambda _: None,
                trial_handler_type=Trial,
                manager_type=Manager,
            )
