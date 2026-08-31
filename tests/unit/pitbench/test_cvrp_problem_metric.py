from __future__ import annotations

import pytest

from pitbench.distribution.transforms import (
    reflect_cvrp,
    relabel_cvrp_customers,
    rotate_cvrp,
    translate_cvrp,
)
from pitbench.metrics.cvrp_problem import (
    CVRP_PROBLEM_METRIC_NAME,
    CVRP_PROBLEM_METRIC_VERSION,
    ExactMetricLimitError,
    anchored_correspondence_cost,
    as_anchored_marked_cvrp,
    exact_cvrp_problem_metric,
)


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
