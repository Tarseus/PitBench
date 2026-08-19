from pathlib import Path

from pitbench.schema.task import PopulationKind
from pitbench.tasks import TaskCatalog

ROOT = Path(__file__).resolve().parents[3]


def test_catalog_contains_five_verified_pr_tasks() -> None:
    records = TaskCatalog(ROOT).validate_all()
    assert {record.task.task_id for record in records} == {
        "pyvrp_1173",
        "vroom_255",
        "highs_2990",
        "choco_671",
        "ortools_2478",
    }


def test_private_references_never_enter_visible_population_manifests() -> None:
    for record in TaskCatalog(ROOT).validate_all():
        assert record.task.references.human_patch.uri.startswith("private://")
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
        if record.task.task_id == "pyvrp_1173"
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
