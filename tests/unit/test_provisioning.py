"""Functional unit/component contracts consolidated by responsibility."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from adapters.pitbench.adapter import (
    IMAGE_CONTEXT_LABEL,
    IMAGE_REVISION,
    IMAGE_REVISION_LABEL,
    IMAGE_SOURCE_LABEL,
    IMAGE_TOOLS_LABEL,
    PitBenchAdapter,
    dockerfile_base_reference,
    image_context_digest,
)
from pitbench.cli.evaluate_config import EvaluationConfig, SolverImageBuildConfig
from pitbench.cli.main import app
from pitbench.cli.solver_images import build_solver_image, docker_build_command
from pitbench.cli.task_image import prepare_task_image
from pitbench.harness.handlers.trial_handler import TrialHandler
from pitbench.instances.storage import prepare_public_inputs
from pitbench.schema.task import PitBenchTask
from pitbench.tasks import TaskCatalog
from tests.support import ROOT


def bundle_bytes(name="instances/example.json"):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        member = tarfile.TarInfo(name)
        content = b'{"customers": [1, 2]}'
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
    return data.getvalue()


@pytest.mark.parametrize("download", [False, True])
def test_prepare_inputs_accepts_local_and_downloaded_bundles(
    tmp_path, monkeypatch, download
):
    archive = bundle_bytes()
    if download:
        source = "https://example.test/public-inputs.tar.gz"

        def open_url(url, *, timeout):
            assert url == source
            return io.BytesIO(archive)

        monkeypatch.setattr("pitbench.instances.storage.urlopen", open_url)
    else:
        source = str(tmp_path / "inputs.tar.gz")
        (tmp_path / "inputs.tar.gz").write_bytes(archive)
    destination = tmp_path / "dataset"
    destination.mkdir()
    (destination / "existing.txt").write_text("retain unrelated data")
    assert prepare_public_inputs(source, destination) == 1
    assert (
        destination / "instances/example.json"
    ).read_bytes() == b'{"customers": [1, 2]}'
    assert (destination / "existing.txt").read_text() == "retain unrelated data"


def test_prepare_inputs_rejects_archive_path_escape_before_installation(tmp_path):
    source = tmp_path / "inputs.tar.gz"
    source.write_bytes(bundle_bytes("../outside.json"))
    destination = tmp_path / "dataset"
    with pytest.raises(tarfile.FilterError):
        prepare_public_inputs(str(source), destination)
    assert not destination.exists()
    assert not (tmp_path / "outside.json").exists()


def test_prepare_inputs_cli_resolves_paths_from_repository_root(tmp_path):
    (tmp_path / "inputs.tar.gz").write_bytes(bundle_bytes())
    result = CliRunner().invoke(
        app,
        [
            "tasks",
            "prepare-inputs",
            "--source",
            "inputs.tar.gz",
            "--output",
            "dataset/public",
            "--root",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Prepared 1 public input files" in result.output
    assert (tmp_path / "dataset/public/instances/example.json").is_file()


def test_imported_tasks_are_explicit_and_work_with_the_validation_cli(tmp_path):
    import yaml

    root = ROOT
    task = PitBenchTask.from_yaml(root / "configs/tasks/vroom_v1_15_0.yaml")
    task.task_id = "research_vroom"
    repository = tmp_path / "repository"
    configs = repository / "configs/instance_sets"
    configs.mkdir(parents=True)
    shutil.copy2(root / "configs/instance_sets/vroom_v1_15_0_dev.yaml", configs)
    source = tmp_path / "research_vroom.yaml"
    source.write_text(yaml.safe_dump(task.model_dump(mode="json")))
    archive = tmp_path / "task-bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as target:
        target.add(source, arcname="tasks/research_vroom.yaml")
    destination = repository / "dataset/task-bundles/research"
    prepare_public_inputs(str(archive), destination)
    assert TaskCatalog(repository).records() == []
    result = CliRunner().invoke(
        app,
        [
            "tasks",
            "validate",
            "--root",
            str(repository),
            "--task-dir",
            "dataset/task-bundles/research/tasks",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "OK research_vroom" in result.output
    assert "Validated 1 task configs" in result.output


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=repository, text=True
    ).strip()


def test_source_override_is_rejected_by_evaluation_config() -> None:
    with pytest.raises(ValidationError, match="repository_source"):
        EvaluationConfig.model_validate(
            {"tasks": {"solver": {"repository_source": "/tmp/solver"}}}
        )


def test_pinned_source_is_fetched_once_then_reused_without_origin(
    tmp_path: Path,
) -> None:
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "--quiet")
    _git(origin, "config", "user.name", "PitBench Test")
    _git(origin, "config", "user.email", "pitbench@example.invalid")
    (origin / "solver.txt").write_text("release source\n")
    (origin / "solver-link.txt").symlink_to("solver.txt")
    _git(origin, "add", "solver.txt", "solver-link.txt")
    _git(origin, "commit", "--quiet", "-m", "release")
    commit = _git(origin, "rev-parse", "HEAD")
    task = SimpleNamespace(
        task_id="test_solver",
        release=SimpleNamespace(base_commit=commit, tree_sha=None),
        repository=SimpleNamespace(clone_url=str(origin)),
    )
    adapter = PitBenchAdapter(tmp_path, tmp_path / "private")
    first = tmp_path / "first"
    adapter._materialize_cached_repository(task, first, tmp_path / "first.bundle")
    cache = adapter.source_cache_path(tmp_path, task)
    assert cache.is_dir()
    assert _git(first, "rev-parse", "HEAD") == commit
    assert (first / "solver-link.txt").is_symlink()
    assert (tmp_path / "first.bundle").is_file()

    origin.rename(tmp_path / "origin-offline")
    second = tmp_path / "second"
    adapter._materialize_cached_repository(task, second, tmp_path / "second.bundle")
    assert (second / "solver.txt").read_text() == "release source\n"
    assert (second / "solver-link.txt").is_symlink()
    assert (tmp_path / "second.bundle").is_file()

    (cache / "untracked.txt").write_text("unexpected\n")
    with pytest.raises(ValueError, match="local modifications"):
        adapter._materialize_cached_repository(
            task, tmp_path / "third", tmp_path / "third.bundle"
        )


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
        "ortools_cp_sat_solve_v9_15",
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


def test_source_image_builder_accepts_a_task_outside_the_default_catalog(tmp_path):
    import yaml

    from pitbench.schema.task import PitBenchTask

    task = PitBenchTask.from_yaml(
        ROOT / "configs/tasks/ortools_cp_sat_solve_v9_15.yaml"
    )
    task.task_id = "research_ortools"
    config_path = tmp_path / "research_ortools.yaml"
    config_path.write_text(yaml.safe_dump(task.model_dump(mode="json")))

    def run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, "sha256:test-image\n", "")

    built = build_solver_image(
        task.task_id,
        ROOT,
        task_config_path=config_path,
        config=SolverImageBuildConfig(),
        run=run,
        progress=lambda _: None,
    )
    assert built.task_id == "research_ortools"
    assert built.image_tag == f"pitbench-ortools:{task.release.base_commit[:12]}"


def test_ortools_image_compiles_the_configurable_cp_sat_runner() -> None:
    dockerfile = DOCKERFILE.read_text()

    assert "cmake --build /workspace/repo/build --target java_package -j6" in dockerfile
    assert (
        "cd /workspace/repo \\\n"
        "    && python3 /opt/pitbench-jvm/runner.py compile --solver ortools_cp_sat"
        in dockerfile
    )
    assert dockerfile.index("COPY adapters/jvm /opt/pitbench-jvm") > dockerfile.index(
        "cmake --build /workspace/repo/build --target java_package -j6"
    )


def test_source_image_build_accepts_machine_local_base_images() -> None:
    import pytest

    command = _command(
        SolverImageBuildConfig(base_images={"JAVA_BASE_IMAGE": "maven:local-import"})
    )
    assert "JAVA_BASE_IMAGE=maven:local-import" in command
    with pytest.raises(ValueError, match="_BASE_IMAGE"):
        _command(
            SolverImageBuildConfig(base_images={"BASE_COMMIT": "different-source"})
        )


def test_publish_workflow_uses_existing_tasks_or_retained_smoke_configs():
    import yaml

    from pitbench.tasks import TaskCatalog

    root = ROOT
    workflow = yaml.safe_load(
        (root / ".github/workflows/publish-solver-images.yml").read_text()
    )
    entries = workflow["jobs"]["publish"]["strategy"]["matrix"]["include"]
    assert {row["version"] for row in entries} == {
        "0.12.2",
        "0.13.0",
        "0.13.4",
        "0.14.0",
    }
    for row in entries:
        task = TaskCatalog(root).validate_file(root / row["task_config"]).task
        assert task.task_id == row["task_id"]
        assert task.release.base_commit == row["base_commit"]
        assert task.release.version == row["version"]
    commands = "\n".join(
        step.get("run", "") for step in workflow["jobs"]["publish"]["steps"]
    )
    assert "--repository-source" not in commands
    assert "--task-config" in commands


def test_default_catalog_validates_without_external_experiment_packages(tmp_path):
    import shutil

    from pitbench.tasks import TaskCatalog

    root = ROOT
    clean = tmp_path / "clean"
    for name in ["tasks", "instance_sets"]:
        shutil.copytree(root / "configs" / name, clean / "configs" / name)
    assert not (clean / ".pitbench-runs").exists()
    assert not (clean / "dataset").exists()
    assert len(TaskCatalog(clean).validate_all()) == len(TaskCatalog(root).records())


def test_external_task_bundle_is_usable_without_changing_the_default_catalog(tmp_path):
    import shutil

    import pytest

    from pitbench.instances.storage import prepare_public_inputs
    from pitbench.tasks import TaskCatalog

    root = ROOT
    archive = root / "dataset/archives/e8-task-bundle-v1.tar.gz"
    if not archive.is_file():
        pytest.skip("external experiment task bundle is not provisioned")
    clean = tmp_path / "clean"
    for name in ["tasks", "instance_sets"]:
        shutil.copytree(root / "configs" / name, clean / "configs" / name)
    original = {record.task.task_id for record in TaskCatalog(clean).records()}
    destination = clean / "dataset/task-bundles/e8"
    assert prepare_public_inputs(str(archive), destination) == 481
    external = TaskCatalog(clean, task_config_dir=destination / "tasks").validate_all()
    assert len(external) == 40
    assert not {record.task.task_id for record in external} & original
    assert {
        record.task.task_id for record in TaskCatalog(clean).validate_all()
    } == original


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
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    cached_source: str,
    force_rebuild: bool,
    should_build: bool,
) -> None:
    base = dockerfile_base_reference(source)
    (tmp_path / "Dockerfile").write_text(
        f'FROM {base}\nLABEL {IMAGE_SOURCE_LABEL}="{source}"\n'
    )
    tags = []
    monkeypatch.setattr(
        "pitbench.cli.task_image.subprocess.run",
        lambda command, **kwargs: tags.append(command),
    )
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
    assert tags == (
        [["docker", "image", "tag", source, base]]
        if base != source and should_build
        else []
    )
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
    assert (
        "${T_BENCH_TASK_AGENT_LOGS_PATH}/development-runs:" + path in client["volumes"]
    )
    assert not any(v.endswith(":/workspace/repo/.pitbench") for v in client["volumes"])
    assert (
        "PITBENCH_RUNS"
        not in yaml.safe_load(PitBenchAdapter._compose([]))["services"]["client"][
            "environment"
        ]
    )


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
            "type": "none",
            "o": "bind",
            "device": "${T_BENCH_TASK_DEVELOPMENT_STORAGE_PATH}/" + name,
        }
    for target in ("/tmp", "/var/tmp"):
        assert any(v.endswith(":" + target) for v in client["volumes"])
    assert client["environment"]["TMPDIR"] == "/tmp"
