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
from pitbench.harness.handlers.trial_handler import TrialHandler


def test_parallel_conditions_of_same_task_use_distinct_images(tmp_path: Path) -> None:
    trials = []
    for condition in ["source", "algorithm_space"]:
        task_path = tmp_path / condition / "same_task"
        task_path.mkdir(parents=True)
        (task_path / "task.yaml").write_text(
            "instruction: Optimize the solver\ndifficulty: hard\n"
            "evaluator_import_path: pitbench.evaluator.evaluator:PitBenchEvaluator\n"
        )
        trials.append(TrialHandler(trial_name=condition, input_path=task_path))
    assert trials[0].task_id == trials[1].task_id
    assert trials[0].client_image_name != trials[1].client_image_name
    prepared = TrialHandler(
        trial_name="pitbench-image-prepare", input_path=trials[1].task_paths.input_path
    )
    assert prepared.client_image_name == trials[1].client_image_name


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


def test_public_output_mount_is_inside_workspace_and_excludes_protected_agent_logs():
    import yaml

    from adapters.pitbench.adapter import PitBenchAdapter

    client = yaml.safe_load(PitBenchAdapter._compose(["bench"]))["services"]["client"]
    path = client["environment"]["PITBENCH_RUNS"]
    assert path == "/workspace/repo/.pitbench/runs"
    assert "${T_BENCH_TASK_AGENT_LOGS_PATH}/development-runs:" + path in client["volumes"]
    assert not any(v.endswith(":/workspace/repo/.pitbench") for v in client["volumes"])
    assert "PITBENCH_RUNS" not in yaml.safe_load(PitBenchAdapter._compose([]))["services"]["client"]["environment"]


def test_development_volumes_preserve_image_contents_outside_docker_overlay():
    import yaml

    from adapters.pitbench.adapter import PitBenchAdapter

    compose = yaml.safe_load(PitBenchAdapter._compose(["bench"]))
    client = compose["services"]["client"]
    assert "development-workspace:/workspace/repo" in client["volumes"]
    assert "development-home:/home/pitbench-agent" in client["volumes"]
    for name in ("workspace", "home"):
        volume = compose["volumes"][f"development-{name}"]
        assert volume["driver_opts"] == {
            "type": "none", "o": "bind",
            "device": "${T_BENCH_TASK_DEVELOPMENT_STORAGE_PATH}/" + name,
        }
    for target in ("/tmp", "/var/tmp"):
        assert any(v.endswith(":" + target) for v in client["volumes"])
    assert client["environment"]["TMPDIR"] == "/tmp"
