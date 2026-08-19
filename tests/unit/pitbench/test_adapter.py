import subprocess
from pathlib import Path

import yaml

from adapters.pitbench.adapter import PitBenchAdapter
from adapters.pitbench.git_snapshot import GitSnapshot
from pitbench.tasks import TaskCatalog

ROOT = Path(__file__).resolve().parents[3]


def _git(repository: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repository, text=True).strip()


def test_materialized_task_has_no_human_solution_or_hidden_assets(
    tmp_path: Path,
) -> None:
    record = next(
        item
        for item in TaskCatalog(ROOT).records()
        if item.task.task_id == "pyvrp_1173"
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
    task.event.base_commit = local_base
    task.event.human_commit = "f" * 40
    local_manifest = tmp_path / "manifest.yaml"
    payload = yaml.safe_load(record.manifest_path.read_text())
    payload["event"]["base_commit"] = local_base
    payload["event"]["human_commit"] = "f" * 40
    local_manifest.write_text(yaml.safe_dump(payload, sort_keys=False))

    local_record = record.__class__(local_manifest, task)
    adapter = PitBenchAdapter(ROOT, ROOT / "private")
    adapter.catalog.validate_all = lambda: [local_record]
    censored = tmp_path / "censored"
    GitSnapshot(
        clone_url=str(source),
        base_commit=local_base,
        forbidden_commits=("f" * 40,),
    ).create(censored)
    destination = tmp_path / "materialized"
    adapter.materialize(
        task.task_id,
        destination,
        repository_source=censored,
    )

    assert not (destination / "solution.sh").exists()
    assert not (destination / "private").exists()
    assert not (destination / "hidden_instances").exists()
    assert (destination / "dev_instances/population.yaml").is_file()
    task_yaml = yaml.safe_load((destination / "task.yaml").read_text())
    assert task_yaml["parser_name"] is None
    assert task_yaml["evaluator_import_path"].endswith(":PitBenchEvaluator")
    assert (destination / "agent_bin/pitbench").stat().st_mode & 0o111
    assert (destination / "agent_config.json").is_file()
    assert (destination / "agent_tooling/pitbench/agent_cli.py").is_file()
    assert "human_patches" not in (destination / "Dockerfile").read_text()
    assert "COPY agent_bin/pitbench /usr/local/bin/pitbench" in (
        destination / "Dockerfile"
    ).read_text()
    assert "network_mode: none" in (destination / "docker-compose.yaml").read_text()
