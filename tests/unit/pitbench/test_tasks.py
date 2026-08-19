from pathlib import Path

from pitbench.schema.task import PopulationKind
from pitbench.tasks import TaskCatalog

ROOT = Path(__file__).resolve().parents[3]


def test_catalog_contains_five_release_snapshots() -> None:
    records = TaskCatalog(ROOT).validate_all()
    assert {record.task.task_id for record in records} == {
        "pyvrp_v0_13_4",
        "vroom_v1_15_0",
        "highs_v1_15_1",
        "choco_v6_0_1",
        "ortools_v9_15",
    }


def test_release_identity_and_private_judge_assets() -> None:
    for record in TaskCatalog(ROOT).validate_all():
        assert record.task.release.tag.startswith("v")
        assert len(record.task.release.base_commit) == 40
        assert record.task.information_regime.value == "snapshot_only"
        for population in record.task.populations:
            if population.kind == PopulationKind.AGENT_DEV:
                assert not population.manifest.startswith("private://")
                assert "private://" not in (ROOT / population.manifest).read_text()
            else:
                assert population.manifest.startswith("private://")


def test_randomness_is_explicit_and_crn_dimensions_are_separate() -> None:
    pyvrp = next(
        record.task
        for record in TaskCatalog(ROOT).records()
        if record.task.task_id == "pyvrp_v0_13_4"
    )
    by_name = {population.name: population for population in pyvrp.populations}
    identity = by_name["judge_id"].randomness
    spatial = by_name["spatial_clustered"].randomness
    capacity = by_name["capacity_tight"].randomness
    assert spatial.instance_seed == identity.instance_seed
    assert spatial.demand_seed == identity.demand_seed
    assert spatial.coordinate_seed != identity.coordinate_seed
    assert capacity.instance_seed == identity.instance_seed
    assert capacity.coordinate_seed == identity.coordinate_seed
    assert capacity.demand_seed != identity.demand_seed
