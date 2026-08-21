from __future__ import annotations

import math

import pytest

from pitbench.distribution.transforms import (
    reflect_cvrp,
    relabel_cvrp_customers,
    rotate_cvrp,
    translate_cvrp,
)
from pitbench.instances import make_uchoa_cvrp_instance
from pitbench.qoi.cvrp import (
    CVRP_INSTANCE_QOI_V1_0,
    CVRP_INSTANCE_QOI_V2_CANDIDATE_0,
    extract_cvrp_instance_qoi,
)
from pitbench.qoi.schema import QoIRole


def _uchoa(**updates):
    parameters = {
        "name": "v2-qoi",
        "customers": 80,
        "coordinate_seed": 101,
        "demand_seed": 202,
        "depot_positioning": "E",
        "customer_positioning": "RC",
        "demand_family": "small_large_variance",
        "route_size": 10,
    }
    parameters.update(updates)
    return make_uchoa_cvrp_instance(**parameters)


def _extract(instance):
    return extract_cvrp_instance_qoi(instance, spec_version="2.0-candidate.0")


def test_v2_candidate_has_the_audited_35_axes_and_roles() -> None:
    axes = CVRP_INSTANCE_QOI_V2_CANDIDATE_0.axes
    roles = {axis.name: axis.role for axis in axes}

    assert len(axes) == 35
    assert len(roles) == 35
    assert all(role is not None for role in roles.values())
    assert roles["capacity"] == QoIRole.RAW
    assert roles["customer_count"] == QoIRole.SCALE
    assert roles["pairwise_distance_cv"] == QoIRole.STRUCT_CORE
    assert roles["nearest_neighbor_clark_evans_ratio"] == (QoIRole.SCALE_CONDITIONED)
    assert roles["dbscan_cluster_fraction"] == QoIRole.EXPERIMENTAL
    assert "demand_depot_radial_weighted_ratio" not in roles


def test_v2_candidate_extractor_is_complete_and_keeps_v1_frozen() -> None:
    observation = _extract(_uchoa())

    assert observation.spec_version == "2.0-candidate.0"
    assert set(observation.values) == {
        axis.name for axis in CVRP_INSTANCE_QOI_V2_CANDIDATE_0.axes
    }
    assert set(observation.axis_defined) == set(observation.values)
    assert all(math.isfinite(value) for value in observation.values.values())
    assert CVRP_INSTANCE_QOI_V1_0.fingerprint() == (
        "f924d33c1a81754934fb8f97c9ba27b62c04312025a9cb6f33820687a87f14c0"
    )


def test_v2_candidate_has_declared_semantic_invariances() -> None:
    instance = _uchoa()
    reference = _extract(instance)
    equivalents = (
        translate_cvrp(instance, dx=23, dy=-9),
        rotate_cvrp(instance, radians=0.7),
        reflect_cvrp(instance),
        relabel_cvrp_customers(instance, seed=99),
    )
    for equivalent in equivalents:
        observation = _extract(equivalent)
        assert observation.values == pytest.approx(reference.values, abs=1e-10)
        assert observation.axis_defined == reference.axis_defined

    coordinate_scaled = {**instance}
    coordinate_scaled["coordinates"] = [
        [x * 10, y * 10] for x, y in instance["coordinates"]
    ]
    coordinate_values = _extract(coordinate_scaled).values
    assert coordinate_values["pairwise_distance_median"] == pytest.approx(
        10 * reference.values["pairwise_distance_median"]
    )
    for axis in set(reference.values) - {"pairwise_distance_median"}:
        assert coordinate_values[axis] == pytest.approx(reference.values[axis])

    demand_scaled = {**instance}
    demand_scaled["demands"] = [10 * value for value in instance["demands"]]
    demand_scaled["capacity"] = 10 * instance["capacity"]
    demand_values = _extract(demand_scaled).values
    for axis in {"capacity", "total_demand"}:
        assert demand_values[axis] == pytest.approx(10 * reference.values[axis])
    for axis in set(reference.values) - {"capacity", "total_demand"}:
        assert demand_values[axis] == pytest.approx(reference.values[axis])


def test_v2_candidate_marks_degenerate_axes_undefined() -> None:
    observation = _extract(
        {
            "name": "line",
            "depot": 0,
            "coordinates": [[0, 0], [1, 0], [2, 0], [3, 0]],
            "demands": [0, 1, 1, 1],
            "capacity": 3,
        }
    )

    for axis in (
        "nearest_neighbor_clark_evans_ratio",
        "nearest_neighbor_iqr_clark_evans_ratio",
        "convex_hull_area_ratio",
        "dbscan_cluster_fraction",
        "demand_depot_radial_pearson",
        "demand_spatial_quadrupole_coupling",
        "demand_local_sparsity_spearman",
    ):
        assert observation.axis_defined[axis] is False


def test_v2_quadrupole_detects_quadrant_assignment_without_demand_multiset_change() -> (
    None
):
    coupled = _uchoa(demand_family="quadrant")
    control = _uchoa(demand_family="quadrant_permuted")

    assert sorted(coupled["demands"]) == sorted(control["demands"])
    assert (
        _extract(coupled).values["demand_spatial_quadrupole_coupling"]
        > _extract(control).values["demand_spatial_quadrupole_coupling"]
    )
