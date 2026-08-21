from __future__ import annotations

import pytest

from pitbench.distribution.transforms import (
    reflect_cvrp,
    relabel_cvrp_customers,
    rotate_cvrp,
    translate_cvrp,
)
from pitbench.instances import make_euclidean_cvrp_instance
from pitbench.qoi.cvrp import (
    CVRP_INSTANCE_QOI,
    CVRP_INSTANCE_QOI_V1_0,
    extract_cvrp_instance_qoi,
)
from pitbench.qoi.schema import InstanceQoIObservation


def test_cvrp_instance_qoi_is_complete_and_versioned() -> None:
    instance = make_euclidean_cvrp_instance(
        name="qoi",
        customers=20,
        coordinate_seed=10,
        demand_seed=20,
        capacity_ratio=0.15,
    )
    observation = extract_cvrp_instance_qoi(instance)

    assert set(observation.values) == {axis.name for axis in CVRP_INSTANCE_QOI.axes}
    assert observation.spec_fingerprint == CVRP_INSTANCE_QOI.fingerprint()
    assert observation.values["customer_count"] == 20
    assert observation.values["capacity_volume_lower_bound"] >= 1
    assert 0 < observation.values["fleet_fill_ratio"] <= 1
    assert all(value == value for value in observation.values.values())
    assert set(observation.axis_defined) == set(observation.values)

    legacy = extract_cvrp_instance_qoi(instance, spec_version="1.0")
    assert legacy.spec_fingerprint == CVRP_INSTANCE_QOI_V1_0.fingerprint()
    assert legacy.values["vehicle_lower_bound"] >= 1


def test_cvrp_instance_qoi_has_declared_semantic_invariances() -> None:
    instance = make_euclidean_cvrp_instance(
        name="qoi-invariance",
        customers=20,
        coordinate_seed=10,
        demand_seed=20,
        capacity_ratio=0.15,
    )
    reference = extract_cvrp_instance_qoi(instance).values
    equivalents = (
        translate_cvrp(instance, dx=23, dy=-9),
        rotate_cvrp(instance, radians=0.7),
        reflect_cvrp(instance),
        relabel_cvrp_customers(instance, seed=99),
    )
    for equivalent in equivalents:
        assert extract_cvrp_instance_qoi(equivalent).values == pytest.approx(reference)

    coordinate_scaled = {**instance}
    coordinate_scaled["coordinates"] = [
        [x * 10, y * 10] for x, y in instance["coordinates"]
    ]
    coordinate_values = extract_cvrp_instance_qoi(coordinate_scaled).values
    assert coordinate_values["pairwise_distance_median"] == pytest.approx(
        10 * reference["pairwise_distance_median"]
    )
    for axis in set(reference) - {"pairwise_distance_median"}:
        assert coordinate_values[axis] == pytest.approx(reference[axis])

    demand_scaled = {**instance}
    demand_scaled["demands"] = [10 * value for value in instance["demands"]]
    demand_scaled["capacity"] = 10 * instance["capacity"]
    demand_values = extract_cvrp_instance_qoi(demand_scaled).values
    for axis in {"capacity", "total_demand"}:
        assert demand_values[axis] == pytest.approx(10 * reference[axis])
    for axis in set(reference) - {"capacity", "total_demand"}:
        assert demand_values[axis] == pytest.approx(reference[axis])


def test_instance_qoi_rejects_values_not_declared_by_spec() -> None:
    with pytest.raises(ValueError, match="missing="):
        InstanceQoIObservation.from_values("bad", CVRP_INSTANCE_QOI, {"extra": 1.0})


def test_published_v1_fingerprint_is_unchanged() -> None:
    assert CVRP_INSTANCE_QOI_V1_0.fingerprint() == (
        "f924d33c1a81754934fb8f97c9ba27b62c04312025a9cb6f33820687a87f14c0"
    )
