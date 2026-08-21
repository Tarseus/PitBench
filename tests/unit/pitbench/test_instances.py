import json
from pathlib import Path

from pitbench.families.cvrp import CVRPFamily
from pitbench.instances import (
    CVRP_QOI_V1_0_AXIS_SPECIFICATIONS,
    InterventionStatus,
    build_exact_single_qoi_panel,
    make_euclidean_cvrp_instance,
    materialize_population,
)
from pitbench.qoi.cvrp import CVRP_INSTANCE_QOI_V1_0, extract_cvrp_instance_qoi
from pitbench.tasks import TaskCatalog

ROOT = Path(__file__).resolve().parents[3]


def test_all_visible_populations_materialize(tmp_path: Path) -> None:
    for record in TaskCatalog(ROOT).validate_all():
        population = record.task.populations[0]
        paths = materialize_population(
            ROOT / population.manifest, tmp_path / record.task.task_id
        )
        assert len(paths) == population.size
        assert all(path.is_file() and path.stat().st_size > 0 for path in paths)


def test_cvrp_verifier_is_independent_and_rejects_duplicates(tmp_path: Path) -> None:
    instance = tmp_path / "instance.json"
    instance.write_text(
        json.dumps(
            {
                "depot": 0,
                "coordinates": [[0, 0], [3, 0], [0, 4]],
                "demands": [0, 1, 1],
                "capacity": 2,
            }
        )
    )
    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps({"routes": [[1, 2]]}))
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"routes": [[1, 1]]}))
    verifier = CVRPFamily()
    assert verifier.verify(instance, valid).feasible is True
    assert verifier.verify(instance, invalid).feasible is False


def test_cvrp_generator_supports_depot_and_demand_location_coupling() -> None:
    common = {
        "name": "coupled",
        "customers": 200,
        "coordinate_seed": 123,
        "demand_seed": 456,
        "capacity_ratio": 0.15,
        "depot_mode": "corner",
    }
    correlated = make_euclidean_cvrp_instance(
        **common, demand_distribution="depot_correlated"
    )
    anticorrelated = make_euclidean_cvrp_instance(
        **common, demand_distribution="depot_anticorrelated"
    )
    correlated_qoi = extract_cvrp_instance_qoi(correlated)
    anticorrelated_qoi = extract_cvrp_instance_qoi(anticorrelated)

    assert correlated_qoi.values["demand_depot_radial_pearson"] > 0.8
    assert anticorrelated_qoi.values["demand_depot_radial_pearson"] < -0.8
    assert correlated["coordinates"] == anticorrelated["coordinates"]
    assert sorted(correlated["demands"]) == sorted(anticorrelated["demands"])


def test_cvrp_generator_legacy_defaults_are_explicitly_backward_compatible() -> None:
    implicit = make_euclidean_cvrp_instance(
        name="legacy",
        customers=12,
        coordinate_seed=11,
        demand_seed=22,
        capacity_ratio=0.2,
    )
    explicit = make_euclidean_cvrp_instance(
        name="legacy",
        customers=12,
        coordinate_seed=11,
        demand_seed=22,
        capacity_ratio=0.2,
        depot_mode="center",
        demand_distribution="uniform_integer",
    )

    assert implicit == explicit


def test_cvrp_axis_catalog_covers_the_frozen_sixteen_qois() -> None:
    assert tuple(item.qoi_name for item in CVRP_QOI_V1_0_AXIS_SPECIFICATIONS) == tuple(
        axis.name for axis in CVRP_INSTANCE_QOI_V1_0.axes
    )
    assert {
        item.qoi_name
        for item in CVRP_QOI_V1_0_AXIS_SPECIFICATIONS
        if item.intervention_status == InterventionStatus.EXACT_SINGLE_QOI
    } == {"demand_cv", "pairwise_distance_median"}


def test_exact_cvrp_axis_panel_changes_only_its_declared_qoi() -> None:
    panel = build_exact_single_qoi_panel((101, 202))

    assert len(panel) == 4
    for case in panel:
        source = extract_cvrp_instance_qoi(case.source, spec_version="1.0").values
        target = extract_cvrp_instance_qoi(case.target, spec_version="1.0").values
        changed = {
            name for name in source if not abs(source[name] - target[name]) <= 1e-9
        }
        assert changed == {case.target_qoi}
        assert case.realized_qoi_delta > 0
