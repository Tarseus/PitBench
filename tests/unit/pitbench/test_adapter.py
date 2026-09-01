import subprocess
from pathlib import Path

import pytest
import yaml

from adapters.pitbench.adapter import (
    IMAGE_REVISION,
    IMAGE_REVISION_LABEL,
    IMAGE_SOURCE_LABEL,
    IMAGE_TOOLS_LABEL,
    PitBenchAdapter,
)
from adapters.pitbench.git_snapshot import GitSnapshot
from pitbench.agent_tools import AGENT_TOOLS_METADATA, AgentTool
from pitbench.tasks import TaskCatalog

ROOT = Path(__file__).resolve().parents[3]


def _git(repository: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repository, text=True).strip()


def test_materialized_release_task_has_no_hidden_assets(
    tmp_path: Path,
) -> None:
    record = next(
        item
        for item in TaskCatalog(ROOT).records()
        if item.task.task_id == "pyvrp_v0_13_4"
    )
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "--quiet")
    _git(source, "config", "user.name", "PitBench")
    _git(source, "config", "user.email", "pitbench@example.invalid")
    (source / "solver.py").write_text("BASE = True\n")
    _git(source, "add", "solver.py")
    _git(source, "commit", "--quiet", "-m", "base")
    local_base = _git(source, "rev-parse", "HEAD")

    task = record.task.model_copy(deep=True)
    task.release.base_commit = local_base
    local_task_config = tmp_path / "task-config.yaml"
    payload = yaml.safe_load(record.task_config_path.read_text())
    payload["release"]["base_commit"] = local_base
    local_task_config.write_text(yaml.safe_dump(payload, sort_keys=False))

    local_record = record.__class__(local_task_config, task)
    adapter = PitBenchAdapter(ROOT, ROOT / "private")
    adapter.catalog.validate_one = lambda task_id: local_record
    censored = tmp_path / "censored"
    GitSnapshot(clone_url=str(source), base_commit=local_base).create(censored)
    destination = tmp_path / "materialized"
    adapter.materialize(
        task.task_id,
        destination,
        repository_source=censored,
        agent_image="sha256:" + "b" * 64,
        judge_image="sha256:" + "a" * 64,
    )

    assert not (destination / "solution.sh").exists()
    assert not (destination / "private").exists()
    assert not (destination / "hidden_instances").exists()
    assert not (destination / "dev_instances").exists()
    assert not (destination / "agent_bin").exists()
    assert not (destination / "agent_config.json").exists()
    assert not (destination / "agent_tooling").exists()
    assert yaml.safe_load((destination / AGENT_TOOLS_METADATA).read_text()) == {
        "agent_tools": []
    }
    task_yaml = yaml.safe_load((destination / "task.yaml").read_text())
    assert task_yaml["parser_name"] is None
    assert task_yaml["evaluator_import_path"].endswith(":PitBenchEvaluator")
    assert task_yaml["evaluator_config"]["judge_image"] == "sha256:" + "a" * 64
    assert "read-only outside these editable paths: pyvrp" in task_yaml["instruction"]
    assert "private://" not in (destination / "Dockerfile").read_text()
    assert (
        "COPY agent_bin/pitbench /usr/local/bin/pitbench"
        not in (destination / "Dockerfile").read_text()
    )
    assert (
        "pip install --break-system-packages --no-cache-dir"
        not in (destination / "Dockerfile").read_text()
    )
    assert "COPY repo /workspace/repo" not in (destination / "Dockerfile").read_text()
    assert (
        f'LABEL {IMAGE_REVISION_LABEL}="{IMAGE_REVISION}"'
        in (destination / "Dockerfile").read_text()
    )
    assert (
        f'{IMAGE_SOURCE_LABEL}="sha256:{"b" * 64}"'
        in (destination / "Dockerfile").read_text()
    )
    assert f'{IMAGE_TOOLS_LABEL}="none"' in (destination / "Dockerfile").read_text()
    dockerfile = (destination / "Dockerfile").read_text()
    assert "USER pitbench-agent" in dockerfile
    assert "/workspace/repo/pyvrp" in dockerfile
    assert "chmod -R a=rX /workspace/repo" in dockerfile
    assert (
        (destination / "Dockerfile").read_text().startswith("FROM sha256:" + "b" * 64)
    )
    dockerignore = (destination / ".dockerignore").read_text().splitlines()
    assert dockerignore[0] == "*"
    assert "!repo" not in dockerignore
    assert "network_mode: none" in (destination / "docker-compose.yaml").read_text()
    assert "no-new-privileges:true" in (destination / "docker-compose.yaml").read_text()
    assert "PITBENCH_AGENT_UID" in (destination / "docker-compose.yaml").read_text()
    assert (
        'PITBENCH_RUNS: "/pitbench/runs"'
        not in (destination / "docker-compose.yaml").read_text()
    )

    legacy_task = task.model_copy(deep=True)
    legacy_task.repository.agent_image = None
    legacy_dockerfile = adapter._dockerfile(legacy_task)
    assert "pip install --break-system-packages --no-cache-dir" in legacy_dockerfile
    assert "COPY repo /workspace/repo" in legacy_dockerfile
    assert "USER pitbench-agent" in legacy_dockerfile

    assisted = tmp_path / "assisted"
    adapter.materialize(
        task.task_id,
        assisted,
        repository_source=censored,
        agent_image="sha256:" + "b" * 64,
        agent_tools=[AgentTool.BENCH],
    )
    assert (assisted / "dev_instances/instance_set_config.yaml").is_file()
    assert (assisted / "agent_bin/pitbench").stat().st_mode & 0o111
    tooling_config = yaml.safe_load((assisted / "agent_config.json").read_text())
    assert tooling_config["agent_tools"] == ["bench"]
    assert f'{IMAGE_TOOLS_LABEL}="bench"' in (assisted / "Dockerfile").read_text()
    assert (
        "COPY agent_bin/pitbench /usr/local/bin/pitbench"
        in (assisted / "Dockerfile").read_text()
    )
    assert (
        'PITBENCH_RUNS: "/pitbench/runs"'
        in (assisted / "docker-compose.yaml").read_text()
    )


def test_repository_source_may_match_verified_release_tree(tmp_path: Path) -> None:
    record = next(
        item
        for item in TaskCatalog(ROOT).records()
        if item.task.task_id == "pyvrp_v0_13_4"
    )
    repository = tmp_path / "archive"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.name", "PitBench")
    _git(repository, "config", "user.email", "pitbench@example.invalid")
    (repository / "solver.py").write_text("BASE = True\n")
    _git(repository, "add", "solver.py")
    _git(repository, "commit", "--quiet", "-m", "archive")
    branch = _git(repository, "branch", "--show-current")
    _git(repository, "checkout", "--quiet", "--detach")
    _git(repository, "branch", "-D", branch)

    task = record.task.model_copy(deep=True)
    task.release.tree_sha = _git(repository, "rev-parse", "HEAD^{tree}")
    PitBenchAdapter.validate_repository(task, repository)

    dirty_file = repository / "dirty.txt"
    dirty_file.write_text("not part of the release\n")
    with pytest.raises(ValueError, match="local modifications"):
        PitBenchAdapter.validate_repository(task, repository)
    dirty_file.unlink()

    task.release.tree_sha = "0" * 40
    with pytest.raises(ValueError, match="release identity mismatch"):
        PitBenchAdapter.validate_repository(task, repository)
