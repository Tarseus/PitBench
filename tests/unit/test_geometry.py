"""Functional unit/component contracts consolidated by responsibility."""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Mapping
from typing import Any

import pytest

from pitbench.problem_families.geometry import (
    CVRP_PROBLEM_METRIC_NAME,
    CVRP_PROBLEM_METRIC_VERSION,
    CVRPMapCertificate,
    ExactMetricLimitError,
    anchored_correspondence_cost,
    as_anchored_marked_cvrp,
    certificate_from_correspondence,
    exact_cvrp_problem_metric,
    search_cvrp_upper_bound,
    signature_cvrp_map_certificate,
    verify_cvrp_map_certificate,
)


def _copy(instance: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(instance))


def translate_cvrp(
    instance: Mapping[str, Any], *, dx: float, dy: float
) -> dict[str, Any]:
    result = _copy(instance)
    result["coordinates"] = [
        [float(x) + dx, float(y) + dy] for x, y in result["coordinates"]
    ]
    return result


def rotate_cvrp(instance: Mapping[str, Any], *, radians: float) -> dict[str, Any]:
    result = _copy(instance)
    cosine, sine = math.cos(radians), math.sin(radians)
    result["coordinates"] = [
        [cosine * float(x) - sine * float(y), sine * float(x) + cosine * float(y)]
        for x, y in result["coordinates"]
    ]
    return result


def reflect_cvrp(instance: Mapping[str, Any]) -> dict[str, Any]:
    result = _copy(instance)
    result["coordinates"] = [[-float(x), float(y)] for x, y in result["coordinates"]]
    return result


def relabel_cvrp_customers(instance: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
    """Permute customer rows while preserving depot identity and all semantics."""
    result = _copy(instance)
    depot = int(result.get("depot", 0))
    customers = [index for index in range(len(result["coordinates"])) if index != depot]
    random.Random(seed).shuffle(customers)
    order = [depot, *customers]
    result["coordinates"] = [result["coordinates"][index] for index in order]
    result["demands"] = [result["demands"][index] for index in order]
    if "node_ids" in result:
        result["node_ids"] = [result["node_ids"][index] for index in order]
    result["depot"] = 0
    return result


def _instance(
    *,
    name: str = "instance",
    coordinates=None,
    demands=None,
    capacity: float = 5,
):
    return {
        "name": name,
        "depot": 0,
        "coordinates": coordinates or [[0, 0], [2, 0], [0, 1]],
        "demands": demands or [0, 1, 2],
        "capacity": capacity,
    }


def _distance(left, right) -> float:
    return exact_cvrp_problem_metric(left, right).distance


def test_metric_definition_is_explicitly_versioned() -> None:
    assert CVRP_PROBLEM_METRIC_NAME == "cvrp-anchored-marked-gromov-hausdorff"
    assert CVRP_PROBLEM_METRIC_VERSION == "1.0"


def test_metric_is_zero_under_declared_cvrp_equivalences() -> None:
    instance = _instance()
    coordinate_scaled = {
        **instance,
        "coordinates": [[7 * x, 7 * y] for x, y in instance["coordinates"]],
    }
    demand_scaled = {
        **instance,
        "demands": [11 * demand for demand in instance["demands"]],
        "capacity": 11 * instance["capacity"],
    }
    equivalents = (
        instance,
        translate_cvrp(instance, dx=13, dy=-8),
        rotate_cvrp(instance, radians=0.37),
        reflect_cvrp(instance),
        relabel_cvrp_customers(instance, seed=91),
        coordinate_scaled,
        demand_scaled,
    )

    for equivalent in equivalents:
        assert _distance(instance, equivalent) == pytest.approx(0, abs=1e-12)


def test_metric_separates_non_equivalent_problem_objects() -> None:
    reference = _instance()
    changed_demand = _instance(demands=[0, 1, 3])
    changed_geometry = _instance(coordinates=[[0, 0], [2, 0], [0, 2]])
    added_customer = _instance(
        coordinates=[[0, 0], [2, 0], [0, 1], [1, 2]],
        demands=[0, 1, 2, 1],
    )

    assert _distance(reference, changed_demand) > 0
    assert _distance(reference, changed_geometry) > 0
    assert _distance(reference, added_customer) > 0


def test_metric_is_symmetric_and_satisfies_triangle_inequality() -> None:
    first = _instance()
    second = _instance(coordinates=[[0, 0], [2, 0], [0.2, 1.1]])
    third = _instance(
        coordinates=[[0, 0], [1.5, 0], [0.3, 1.4], [1.8, 1.2]],
        demands=[0, 1, 2, 1],
    )

    first_second = _distance(first, second)
    second_first = _distance(second, first)
    second_third = _distance(second, third)
    first_third = _distance(first, third)

    assert first_second == pytest.approx(second_first)
    assert first_third <= first_second + second_third + 1e-12


def test_triangle_inequality_over_a_small_mixed_cardinality_panel() -> None:
    instances = (
        _instance(),
        _instance(coordinates=[[0, 0], [2, 0], [0.2, 1.1]]),
        _instance(demands=[0, 2, 1]),
        _instance(
            coordinates=[[0, 0], [1.5, 0], [0.3, 1.4], [1.8, 1.2]],
            demands=[0, 1, 2, 1],
        ),
    )
    distances = {
        (left, right): _distance(instances[left], instances[right])
        for left in range(len(instances))
        for right in range(len(instances))
    }

    assert all(0 <= distance <= 1 for distance in distances.values())
    for first in range(len(instances)):
        for second in range(len(instances)):
            for third in range(len(instances)):
                assert distances[first, third] <= (
                    distances[first, second] + distances[second, third] + 1e-12
                )


def test_exact_result_returns_a_valid_witness() -> None:
    left = as_anchored_marked_cvrp(_instance())
    right = as_anchored_marked_cvrp(
        _instance(coordinates=[[0, 0], [1.8, 0.1], [0.1, 1.2]])
    )

    result = exact_cvrp_problem_metric(left, right)

    assert result.configurations_evaluated == result.configurations_total
    assert anchored_correspondence_cost(
        left, right, result.correspondence
    ) == pytest.approx(result.distance)


def test_correspondence_must_be_anchored_and_surjective() -> None:
    space = as_anchored_marked_cvrp(_instance())

    with pytest.raises(ValueError, match="depot"):
        anchored_correspondence_cost(space, space, ((0, 1), (1, 0), (2, 2)))
    with pytest.raises(ValueError, match="cover every right"):
        anchored_correspondence_cost(space, space, ((0, 0), (1, 1), (2, 1)))


def test_exact_reference_refuses_unbounded_computation() -> None:
    larger = _instance(
        coordinates=[[0, 0], [1, 0], [0, 1], [2, 0], [0, 2]],
        demands=[0, 1, 1, 1, 1],
    )

    with pytest.raises(ExactMetricLimitError, match="65536"):
        exact_cvrp_problem_metric(larger, larger, max_configurations=100)


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"coordinates": [[0, 0], [0, 0], [1, 0]]}, "distinct coordinates"),
        ({"demands": [0, 6, 1]}, "demands must lie"),
        ({"demands": [1, 1, 2]}, "depot demand"),
    ],
)
def test_metric_rejects_instances_outside_its_declared_domain(updates, message) -> None:
    with pytest.raises(ValueError, match=message):
        as_anchored_marked_cvrp(_instance(**updates))


def _certificate_instance(*, coordinates=None, demands=None, capacity: float = 5):
    return {
        "depot": 0,
        "coordinates": coordinates or [[0, 0], [2, 0], [0, 1]],
        "demands": demands or [0, 1, 2],
        "capacity": capacity,
    }


def test_map_certificate_round_trips_and_matches_its_union_relation() -> None:
    left = as_anchored_marked_cvrp(_certificate_instance())
    right = as_anchored_marked_cvrp(
        _certificate_instance(
            coordinates=[[0, 0], [1.8, 0], [0, 1], [1, 1]],
            demands=[0, 1, 2, 1],
        )
    )
    certificate = CVRPMapCertificate(forward_map=(0, 1, 2), backward_map=(0, 1, 2, 2))

    restored = CVRPMapCertificate.from_mapping(certificate.to_dict())
    verified = verify_cvrp_map_certificate(left, right, restored)

    assert restored == certificate
    assert verified.upper_bound == pytest.approx(
        anchored_correspondence_cost(left, right, verified.correspondence)
    )


def test_codistortion_rejects_individually_isometric_but_incoherent_maps() -> None:
    symmetric = _certificate_instance(
        coordinates=[[0, 0], [1, 0], [-1, 0]], demands=[0, 1, 1]
    )
    incompatible = CVRPMapCertificate(forward_map=(0, 1, 2), backward_map=(0, 2, 1))

    verified = verify_cvrp_map_certificate(symmetric, symmetric, incompatible)

    assert verified.forward_distortion == pytest.approx(0)
    assert verified.backward_distortion == pytest.approx(0)
    assert verified.forward_mark_discrepancy == pytest.approx(0)
    assert verified.backward_mark_discrepancy == pytest.approx(0)
    assert verified.codistortion == pytest.approx(1)
    assert verified.upper_bound == pytest.approx(1)


def test_maps_selected_from_an_exact_correspondence_recover_the_exact_cost() -> None:
    left = _certificate_instance()
    right = _certificate_instance(
        coordinates=[[0, 0], [1.7, 0.1], [0.2, 1.1]], demands=[0, 2, 1]
    )
    exact = exact_cvrp_problem_metric(left, right)

    certificate = certificate_from_correspondence(left, right, exact.correspondence)
    verified = verify_cvrp_map_certificate(left, right, certificate)

    assert verified.upper_bound == pytest.approx(exact.distance)


def test_signature_witness_is_a_certified_upper_bound_with_mixed_cardinality() -> None:
    left = _certificate_instance()
    right = _certificate_instance(
        coordinates=[[0, 0], [1.8, 0], [0, 1], [1, 1]],
        demands=[0, 1, 2, 1],
    )
    exact = exact_cvrp_problem_metric(left, right)

    certificate = signature_cvrp_map_certificate(left, right)
    verified = verify_cvrp_map_certificate(left, right, certificate)

    assert verified.upper_bound + 1e-12 >= exact.distance
    assert len(certificate.forward_map) == 3
    assert len(certificate.backward_map) == 4


def test_bounded_local_search_returns_monotone_anytime_certificates() -> None:
    symmetric = _certificate_instance(
        coordinates=[[0, 0], [1, 0], [-1, 0]], demands=[0, 1, 1]
    )
    incompatible = CVRPMapCertificate(forward_map=(0, 1, 2), backward_map=(0, 2, 1))

    result = search_cvrp_upper_bound(
        symmetric,
        symmetric,
        initial_certificate=incompatible,
        candidate_pool_size=None,
        max_passes=2,
        max_evaluations=20,
    )

    assert result.upper_bound_history[0] == pytest.approx(1)
    assert result.best.upper_bound == pytest.approx(0)
    assert all(
        later <= earlier
        for earlier, later in zip(
            result.upper_bound_history, result.upper_bound_history[1:]
        )
    )
    assert result.verifier_evaluations <= 20


@pytest.mark.parametrize(
    "certificate, message",
    [
        (CVRPMapCertificate((0, 1), (0, 1, 2)), "one image"),
        (CVRPMapCertificate((1, 1, 2), (0, 1, 2)), "depot partition"),
        (CVRPMapCertificate((0, 1, 3), (0, 1, 2)), "out-of-range"),
    ],
)
def test_verifier_rejects_malformed_certificates(certificate, message) -> None:
    with pytest.raises(ValueError, match=message):
        verify_cvrp_map_certificate(
            _certificate_instance(), _certificate_instance(), certificate
        )


def test_serialized_certificate_does_not_coerce_non_integer_images() -> None:
    payload = {"forward_map": [0, 1.5, 2], "backward_map": [0, 1, 2]}

    with pytest.raises(ValueError, match="integers"):
        verify_cvrp_map_certificate(
            _certificate_instance(), _certificate_instance(), payload
        )
