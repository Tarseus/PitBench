from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

CVRP_PROBLEM_METRIC_NAME = "cvrp-anchored-marked-gromov-hausdorff"
CVRP_PROBLEM_METRIC_VERSION = "1.0"


class ExactMetricLimitError(ValueError):
    """Raised when the exact finite correspondence search exceeds its limit."""


@dataclass(frozen=True)
class AnchoredMarkedCVRP:
    """Canonical CVRP object used by the problem-space metric.

    Node zero is always the depot. Distances are divided by the instance diameter,
    and demand marks are divided by vehicle capacity. Customer order has no
    semantics: the metric minimizes over anchored correspondences.
    """

    distances: tuple[tuple[float, ...], ...]
    demand_fractions: tuple[float, ...]

    @property
    def node_count(self) -> int:
        return len(self.demand_fractions)

    @property
    def customer_count(self) -> int:
        return self.node_count - 1


@dataclass(frozen=True)
class ExactCVRPMetricResult:
    distance: float
    correspondence: tuple[tuple[int, int], ...]
    configurations_evaluated: int
    configurations_total: int


def as_anchored_marked_cvrp(instance: Mapping[str, Any]) -> AnchoredMarkedCVRP:
    """Map a standard metric CVRP instance to its quotient representative."""

    coordinates = np.asarray(instance["coordinates"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacity = float(instance["capacity"])
    depot = int(instance.get("depot", 0))
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 2
        or len(coordinates) != len(demands)
        or len(coordinates) < 2
        or not 0 <= depot < len(coordinates)
        or not np.all(np.isfinite(coordinates))
        or not np.all(np.isfinite(demands))
        or not math.isfinite(capacity)
        or capacity <= 0
    ):
        raise ValueError("invalid metric CVRP instance")
    if demands[depot] != 0:
        raise ValueError("the depot demand must be zero")
    customer_mask = np.arange(len(coordinates)) != depot
    customer_demands = demands[customer_mask]
    if np.any(customer_demands <= 0) or np.any(customer_demands > capacity):
        raise ValueError("customer demands must lie in (0, capacity]")

    order = np.concatenate(([depot], np.flatnonzero(customer_mask)))
    ordered_coordinates = coordinates[order]
    pairwise = np.linalg.norm(
        ordered_coordinates[:, None, :] - ordered_coordinates[None, :, :], axis=2
    )
    off_diagonal = pairwise[~np.eye(len(pairwise), dtype=bool)]
    if np.any(off_diagonal <= 0):
        raise ValueError("metric CVRP nodes must have distinct coordinates")
    diameter = float(np.max(pairwise))
    normalized = pairwise / diameter
    ordered_demands = demands[order] / capacity
    return AnchoredMarkedCVRP(
        distances=tuple(tuple(float(value) for value in row) for row in normalized),
        demand_fractions=tuple(float(value) for value in ordered_demands),
    )


def _validated_correspondence(
    left: AnchoredMarkedCVRP,
    right: AnchoredMarkedCVRP,
    correspondence: Iterable[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    relation = tuple(sorted(set(correspondence)))
    if not relation:
        raise ValueError("a correspondence must be non-empty")
    if any(
        not 0 <= left_node < left.node_count or not 0 <= right_node < right.node_count
        for left_node, right_node in relation
    ):
        raise ValueError("correspondence contains an out-of-range node")
    if any((left_node == 0) != (right_node == 0) for left_node, right_node in relation):
        raise ValueError("an anchored correspondence may only pair depot with depot")
    if (0, 0) not in relation:
        raise ValueError("an anchored correspondence must contain the depot pair")
    if {left_node for left_node, _ in relation} != set(range(left.node_count)):
        raise ValueError("correspondence must cover every left node")
    if {right_node for _, right_node in relation} != set(range(right.node_count)):
        raise ValueError("correspondence must cover every right node")
    return relation


def anchored_correspondence_cost(
    left: AnchoredMarkedCVRP,
    right: AnchoredMarkedCVRP,
    correspondence: Iterable[tuple[int, int]],
) -> float:
    """Return max relational distortion and demand-mark mismatch for one relation."""

    relation = _validated_correspondence(left, right, correspondence)
    mark_cost = max(
        abs(left.demand_fractions[left_node] - right.demand_fractions[right_node])
        for left_node, right_node in relation
    )
    distortion = max(
        abs(
            left.distances[left_first][left_second]
            - right.distances[right_first][right_second]
        )
        for left_first, right_first in relation
        for left_second, right_second in relation
    )
    return max(mark_cost, distortion)


def _configuration_count(left_customers: int, right_customers: int) -> int:
    return right_customers**left_customers * left_customers**right_customers


def exact_cvrp_problem_metric(
    left_instance: Mapping[str, Any] | AnchoredMarkedCVRP,
    right_instance: Mapping[str, Any] | AnchoredMarkedCVRP,
    *,
    max_configurations: int = 1_000_000,
) -> ExactCVRPMetricResult:
    """Compute the exact anchored marked GH metric by finite enumeration.

    The reference algorithm is exponential and intentionally refuses searches above
    ``max_configurations``. It is a definition oracle for proofs and small tests, not
    a scalable approximation whose output is mislabeled as a metric.
    """

    left = (
        left_instance
        if isinstance(left_instance, AnchoredMarkedCVRP)
        else as_anchored_marked_cvrp(left_instance)
    )
    right = (
        right_instance
        if isinstance(right_instance, AnchoredMarkedCVRP)
        else as_anchored_marked_cvrp(right_instance)
    )
    total = _configuration_count(left.customer_count, right.customer_count)
    if max_configurations < 1:
        raise ValueError("max_configurations must be positive")
    if total > max_configurations:
        raise ExactMetricLimitError(
            f"exact correspondence search requires {total} configurations; "
            f"limit is {max_configurations}"
        )

    left_nodes = tuple(range(1, left.node_count))
    right_nodes = tuple(range(1, right.node_count))
    best_cost = math.inf
    best_relation: tuple[tuple[int, int], ...] = ()
    evaluated = 0
    for left_images in itertools.product(right_nodes, repeat=left.customer_count):
        left_graph = set(zip(left_nodes, left_images, strict=True))
        for right_images in itertools.product(left_nodes, repeat=right.customer_count):
            evaluated += 1
            relation = {(0, 0), *left_graph}
            relation.update(zip(right_images, right_nodes, strict=True))
            ordered_relation = tuple(sorted(relation))
            cost = anchored_correspondence_cost(left, right, ordered_relation)
            if cost < best_cost:
                best_cost = cost
                best_relation = ordered_relation
                if cost == 0:
                    return ExactCVRPMetricResult(
                        distance=0.0,
                        correspondence=best_relation,
                        configurations_evaluated=evaluated,
                        configurations_total=total,
                    )
    return ExactCVRPMetricResult(
        distance=best_cost,
        correspondence=best_relation,
        configurations_evaluated=evaluated,
        configurations_total=total,
    )


__all__ = [
    "AnchoredMarkedCVRP",
    "CVRP_PROBLEM_METRIC_NAME",
    "CVRP_PROBLEM_METRIC_VERSION",
    "ExactCVRPMetricResult",
    "ExactMetricLimitError",
    "anchored_correspondence_cost",
    "as_anchored_marked_cvrp",
    "exact_cvrp_problem_metric",
]
