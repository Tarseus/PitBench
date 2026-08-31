import shutil
from pathlib import Path

import pytest

from pitbench.schema.task import PopulationKind
from pitbench.tasks import TaskCatalog

ROOT = Path(__file__).resolve().parents[3]


def test_catalog_contains_release_snapshots() -> None:
    records = TaskCatalog(ROOT).validate_all()
    assert {record.task.task_id for record in records} == {
        "pyvrp_v0_12_2",
        "pyvrp_v0_13_0",
        "pyvrp_v0_13_4",
        "pyvrp_v0_14_0",
        "vroom_v1_15_0",
        "highs_v1_15_1",
        "choco_v6_0_1",
        "ortools_v9_15",
    }


def test_validate_one_ignores_unrelated_invalid_manifest(tmp_path: Path) -> None:
    task_id = "pyvrp_v0_14_0"
    source = TaskCatalog(ROOT).validate_one(task_id)
    manifest = tmp_path / "manifests/tasks" / source.manifest_path.name
    manifest.parent.mkdir(parents=True)
    shutil.copy2(source.manifest_path, manifest)
    for population in source.task.populations:
        if population.kind != PopulationKind.AGENT_DEV:
            continue
        source_population = ROOT / population.manifest
        target_population = tmp_path / population.manifest
        target_population.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_population, target_population)

    (manifest.parent / "unrelated.yaml").write_text("not: a valid task\n")
    catalog = TaskCatalog(tmp_path)

    assert catalog.validate_one(task_id).task.task_id == task_id
    with pytest.raises(ValueError):
        catalog.validate_all()


def test_release_identity_and_private_judge_assets() -> None:
    for record in TaskCatalog(ROOT).validate_all():
        assert record.task.release.tag.startswith("v")
        assert len(record.task.release.base_commit) == 40
        if record.task.task_id.startswith("pyvrp_"):
            assert record.task.repository.agent_image == (
                f"ghcr.io/tarseus/pitbench-pyvrp:{record.task.release.base_commit}"
            )
        else:
            assert record.task.repository.agent_image is None
        assert record.task.repository.editable_paths
        assert all(
            path not in {"", "."} and not path.startswith(("/", ".git"))
            for path in record.task.repository.editable_paths
        )
        assert record.task.information_regime.value == "snapshot_only"
        for population in record.task.populations:
            if population.kind == PopulationKind.AGENT_DEV:
                assert not population.manifest.startswith("private://")
                assert "private://" not in (ROOT / population.manifest).read_text()
            else:
                assert population.manifest.startswith("private://")


def test_pyvrp_snapshots_use_verified_release_commits_and_common_protocol() -> None:
    expected_commits = {
        "pyvrp_v0_12_2": "ea0c4211819edac6fd920413ad7508cc9ad56e0e",
        "pyvrp_v0_13_0": "542cdd32e4b10791e15b911cfbad0691538ed987",
        "pyvrp_v0_13_4": "18815548d04a90a0e5eea2a0bed53a81ea9d2d49",
        "pyvrp_v0_14_0": "5d9776a954b810bdb3fe71d47b1e7de7cffe90d2",
    }
    expected_trees = {
        "pyvrp_v0_12_2": "d04db06800f97816a4a973a846a68562efb70ccf",
        "pyvrp_v0_13_0": "fa33d1368a11b312f97707178350ec126a5918ab",
        "pyvrp_v0_13_4": "2d24f7fd1af5b69a732d6baf2ef7d870666eb515",
        "pyvrp_v0_14_0": "6587bf9f3e09a7c8ac7807e07b23704d30b8067a",
    }
    snapshots = [
        record.task
        for record in TaskCatalog(ROOT).records()
        if record.task.task_id.startswith("pyvrp_")
    ]

    assert {task.task_id: task.release.base_commit for task in snapshots} == (
        expected_commits
    )
    assert {task.task_id: task.release.tree_sha for task in snapshots} == expected_trees
    protocols = {
        (
            tuple(task.evaluation.budgets_sec),
            tuple(task.evaluation.solver_seeds),
            task.evaluation.threads,
            tuple(
                (
                    population.name,
                    population.manifest,
                    population.size,
                    population.randomness.model_dump_json(),
                )
                for population in task.populations
            ),
        )
        for task in snapshots
    }
    assert len(protocols) == 1
