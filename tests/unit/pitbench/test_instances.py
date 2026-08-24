import json
import math
from pathlib import Path

import numpy as np
import pytest

from pitbench.families.cvrp import CVRPFamily
from pitbench.instances import make_euclidean_cvrp_instance, materialize_population
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


def test_cvrp_verifier_applies_declared_euc_2d_rounding(tmp_path: Path) -> None:
    payload = {
        "depot": 0,
        "coordinates": [[0, 0], [1, 1], [2, 0]],
        "demands": [0, 1, 1],
        "capacity": 2,
    }
    solution = tmp_path / "solution.json"
    solution.write_text(json.dumps({"routes": [[1, 2]]}))
    exact_instance = tmp_path / "exact.json"
    exact_instance.write_text(json.dumps(payload))
    rounded_instance = tmp_path / "rounded.json"
    rounded_instance.write_text(json.dumps({**payload, "distance_metric": "EUC_2D"}))
    verifier = CVRPFamily()

    exact = verifier.verify(exact_instance, solution)
    rounded = verifier.verify(rounded_instance, solution)

    assert exact.objective == pytest.approx(2 + 2 * math.sqrt(2))
    assert rounded.objective == 4


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
    coords = np.asarray(correlated["coordinates"], dtype=float)
    depot_dist = np.linalg.norm(coords[1:] - coords[0], axis=1)
    corr = float(np.corrcoef(correlated["demands"][1:], depot_dist)[0, 1])
    anticorr = float(np.corrcoef(anticorrelated["demands"][1:], depot_dist)[0, 1])

    assert corr > 0.8
    assert anticorr < -0.8
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
