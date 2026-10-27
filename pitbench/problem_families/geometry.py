from __future__ import annotations

import itertools
import math
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

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


@dataclass(frozen=True)
class CVRPMapCertificate:
    """Two depot-preserving maps witnessing a CVRP metric upper bound."""

    forward_map: tuple[int, ...]
    backward_map: tuple[int, ...]

    def to_dict(self) -> dict[str, list[int]]:
        return {
            "forward_map": list(self.forward_map),
            "backward_map": list(self.backward_map),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> CVRPMapCertificate:
        try:
            forward = tuple(payload["forward_map"])
            backward = tuple(payload["backward_map"])
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid CVRP map certificate payload") from exc
        if any(type(node) is not int for node in (*forward, *backward)):
            raise ValueError("CVRP map certificate images must be integers")
        return cls(forward_map=forward, backward_map=backward)


@dataclass(frozen=True)
class VerifiedCVRPUpperBound:
    """Independently recomputed components of one certified upper bound."""

    certificate: CVRPMapCertificate
    upper_bound: float
    forward_distortion: float
    backward_distortion: float
    codistortion: float
    forward_mark_discrepancy: float
    backward_mark_discrepancy: float

    @property
    def correspondence(self) -> tuple[tuple[int, int], ...]:
        relation = {
            *enumerate(self.certificate.forward_map),
            *(
                (left, right)
                for right, left in enumerate(self.certificate.backward_map)
            ),
        }
        return tuple(sorted(relation))


@dataclass(frozen=True)
class CVRPUpperBoundSearchResult:
    """Best verified witness produced within a deterministic search budget."""

    best: VerifiedCVRPUpperBound
    verifier_evaluations: int
    upper_bound_history: tuple[float, ...]
    stopped_reason: Literal[
        "local_optimum", "evaluation_limit", "time_limit", "pass_limit"
    ]


def _as_space(
    instance: Mapping[str, Any] | AnchoredMarkedCVRP,
) -> AnchoredMarkedCVRP:
    return (
        instance
        if isinstance(instance, AnchoredMarkedCVRP)
        else as_anchored_marked_cvrp(instance)
    )


def _validated_map(
    mapping: tuple[int, ...],
    *,
    source_count: int,
    target_count: int,
    name: str,
) -> np.ndarray:
    if len(mapping) != source_count:
        raise ValueError(f"{name} map must contain one image per source node")
    if any(isinstance(node, bool) or not isinstance(node, int) for node in mapping):
        raise ValueError(f"{name} map images must be integers")
    if any(not 0 <= node < target_count for node in mapping):
        raise ValueError(f"{name} map contains an out-of-range node")
    if mapping[0] != 0 or any(node == 0 for node in mapping[1:]):
        raise ValueError(f"{name} map must preserve the depot partition")
    return np.asarray(mapping, dtype=int)


class _PreparedVerifier:
    def __init__(self, left: AnchoredMarkedCVRP, right: AnchoredMarkedCVRP) -> None:
        self.left = left
        self.right = right
        self.left_distances = np.asarray(left.distances, dtype=float)
        self.right_distances = np.asarray(right.distances, dtype=float)
        self.left_marks = np.asarray(left.demand_fractions, dtype=float)
        self.right_marks = np.asarray(right.demand_fractions, dtype=float)

    def verify(self, certificate: CVRPMapCertificate) -> VerifiedCVRPUpperBound:
        forward = _validated_map(
            certificate.forward_map,
            source_count=self.left.node_count,
            target_count=self.right.node_count,
            name="forward",
        )
        backward = _validated_map(
            certificate.backward_map,
            source_count=self.right.node_count,
            target_count=self.left.node_count,
            name="backward",
        )

        forward_distortion = float(
            np.max(
                np.abs(
                    self.left_distances
                    - self.right_distances[
                        forward[:, np.newaxis], forward[np.newaxis, :]
                    ]
                )
            )
        )
        backward_distortion = float(
            np.max(
                np.abs(
                    self.left_distances[
                        backward[:, np.newaxis], backward[np.newaxis, :]
                    ]
                    - self.right_distances
                )
            )
        )
        codistortion = float(
            np.max(
                np.abs(
                    self.left_distances[:, backward] - self.right_distances[forward, :]
                )
            )
        )
        forward_mark = float(
            np.max(np.abs(self.left_marks - self.right_marks[forward]))
        )
        backward_mark = float(
            np.max(np.abs(self.left_marks[backward] - self.right_marks))
        )
        upper_bound = max(
            forward_distortion,
            backward_distortion,
            codistortion,
            forward_mark,
            backward_mark,
        )
        return VerifiedCVRPUpperBound(
            certificate=certificate,
            upper_bound=upper_bound,
            forward_distortion=forward_distortion,
            backward_distortion=backward_distortion,
            codistortion=codistortion,
            forward_mark_discrepancy=forward_mark,
            backward_mark_discrepancy=backward_mark,
        )


def verify_cvrp_map_certificate(
    left_instance: Mapping[str, Any] | AnchoredMarkedCVRP,
    right_instance: Mapping[str, Any] | AnchoredMarkedCVRP,
    certificate: CVRPMapCertificate | Mapping[str, Any],
) -> VerifiedCVRPUpperBound:
    """Verify a map-pair witness in ``O(n² + m² + nm)`` time.

    CVRP metric definition 1.0 omits the conventional one-half GH factor and uses
    unit demand-mark weight. Its certified bound is therefore
    ``max(D_f, D_g, C_f,g, Q_f, Q_g)``.
    """

    left = _as_space(left_instance)
    right = _as_space(right_instance)
    parsed = (
        certificate
        if isinstance(certificate, CVRPMapCertificate)
        else CVRPMapCertificate.from_mapping(certificate)
    )
    return _PreparedVerifier(left, right).verify(parsed)


def certificate_from_correspondence(
    left_instance: Mapping[str, Any] | AnchoredMarkedCVRP,
    right_instance: Mapping[str, Any] | AnchoredMarkedCVRP,
    correspondence: Iterable[tuple[int, int]],
) -> CVRPMapCertificate:
    """Select deterministic map graphs from a valid anchored correspondence."""

    left = _as_space(left_instance)
    right = _as_space(right_instance)
    relation = tuple(sorted(set(correspondence)))
    anchored_correspondence_cost(left, right, relation)
    forward = tuple(
        next(right_node for left_node, right_node in relation if left_node == node)
        for node in range(left.node_count)
    )
    backward = tuple(
        next(left_node for left_node, right_node in relation if right_node == node)
        for node in range(right.node_count)
    )
    return CVRPMapCertificate(forward_map=forward, backward_map=backward)


def _node_signatures(space: AnchoredMarkedCVRP, quantile_count: int) -> np.ndarray:
    if quantile_count < 1:
        raise ValueError("quantile_count must be positive")
    distances = np.asarray(space.distances, dtype=float)
    probabilities = np.linspace(0, 1, quantile_count)
    quantiles = np.quantile(distances, probabilities, axis=1).T
    marks = np.asarray(space.demand_fractions, dtype=float)[:, np.newaxis]
    depot_distances = distances[:, 0, np.newaxis]
    return np.concatenate((marks, depot_distances, quantiles), axis=1)


def _target_rankings(
    source: AnchoredMarkedCVRP,
    target: AnchoredMarkedCVRP,
    quantile_count: int,
) -> tuple[tuple[int, ...], ...]:
    source_signatures = _node_signatures(source, quantile_count)
    target_signatures = _node_signatures(target, quantile_count)
    rankings: list[tuple[int, ...]] = [(0,)]
    target_nodes = np.arange(1, target.node_count)
    for signature in source_signatures[1:]:
        discrepancies = np.max(
            np.abs(target_signatures[1:] - signature[np.newaxis, :]), axis=1
        )
        order = np.lexsort((target_nodes, discrepancies))
        rankings.append(tuple(int(target_nodes[index]) for index in order))
    return tuple(rankings)


def _certificate_from_rankings(
    left: AnchoredMarkedCVRP,
    right: AnchoredMarkedCVRP,
    forward_rankings: tuple[tuple[int, ...], ...],
    backward_rankings: tuple[tuple[int, ...], ...],
    quantile_count: int,
) -> CVRPMapCertificate:
    forward = [0] * left.node_count
    backward = [0] * right.node_count

    if left.customer_count <= right.customer_count:
        signatures = _node_signatures(left, quantile_count)
        source_order = sorted(
            range(1, left.node_count), key=lambda node: tuple(signatures[node])
        )
        unused_targets = set(range(1, right.node_count))
        for left_node in source_order:
            right_node = next(
                node for node in forward_rankings[left_node] if node in unused_targets
            )
            forward[left_node] = right_node
            backward[right_node] = left_node
            unused_targets.remove(right_node)
        for right_node in unused_targets:
            backward[right_node] = backward_rankings[right_node][0]
    else:
        signatures = _node_signatures(right, quantile_count)
        source_order = sorted(
            range(1, right.node_count), key=lambda node: tuple(signatures[node])
        )
        unused_targets = set(range(1, left.node_count))
        for right_node in source_order:
            left_node = next(
                node for node in backward_rankings[right_node] if node in unused_targets
            )
            backward[right_node] = left_node
            forward[left_node] = right_node
            unused_targets.remove(left_node)
        for left_node in unused_targets:
            forward[left_node] = forward_rankings[left_node][0]

    return CVRPMapCertificate(tuple(forward), tuple(backward))


def signature_cvrp_map_certificate(
    left_instance: Mapping[str, Any] | AnchoredMarkedCVRP,
    right_instance: Mapping[str, Any] | AnchoredMarkedCVRP,
    *,
    quantile_count: int = 8,
) -> CVRPMapCertificate:
    """Build a cheap deterministic witness from solver-independent node signatures."""

    left = _as_space(left_instance)
    right = _as_space(right_instance)
    forward_rankings = _target_rankings(left, right, quantile_count)
    backward_rankings = _target_rankings(right, left, quantile_count)
    return _certificate_from_rankings(
        left, right, forward_rankings, backward_rankings, quantile_count
    )


def search_cvrp_upper_bound(
    left_instance: Mapping[str, Any] | AnchoredMarkedCVRP,
    right_instance: Mapping[str, Any] | AnchoredMarkedCVRP,
    *,
    initial_certificate: CVRPMapCertificate | None = None,
    quantile_count: int = 8,
    candidate_pool_size: int | None = 8,
    max_passes: int = 2,
    max_evaluations: int = 256,
    time_limit_sec: float | None = None,
) -> CVRPUpperBoundSearchResult:
    """Tighten a certified bound by bounded deterministic coordinate search.

    Every trial is independently verified. The returned history contains the
    initial bound followed only by strict improvements, so it is monotonically
    non-increasing and remains usable when any search budget is exhausted.
    """

    if max_passes < 0:
        raise ValueError("max_passes must be non-negative")
    if max_evaluations < 1:
        raise ValueError("max_evaluations must be positive")
    if candidate_pool_size is not None and candidate_pool_size < 1:
        raise ValueError("candidate_pool_size must be positive or None")
    if time_limit_sec is not None and (
        not math.isfinite(time_limit_sec) or time_limit_sec <= 0
    ):
        raise ValueError("time_limit_sec must be finite and positive")

    left = _as_space(left_instance)
    right = _as_space(right_instance)
    verifier = _PreparedVerifier(left, right)
    forward_rankings = _target_rankings(left, right, quantile_count)
    backward_rankings = _target_rankings(right, left, quantile_count)
    certificate = initial_certificate or _certificate_from_rankings(
        left, right, forward_rankings, backward_rankings, quantile_count
    )
    best = verifier.verify(certificate)
    evaluations = 1
    history = [best.upper_bound]
    started = time.perf_counter()

    def budget_reason() -> Literal["evaluation_limit", "time_limit"] | None:
        if evaluations >= max_evaluations:
            return "evaluation_limit"
        if (
            time_limit_sec is not None
            and time.perf_counter() - started >= time_limit_sec
        ):
            return "time_limit"
        return None

    stopped_reason: Literal[
        "local_optimum", "evaluation_limit", "time_limit", "pass_limit"
    ] = "pass_limit"
    for _ in range(max_passes):
        changed = False
        for direction, rankings in (
            ("forward", forward_rankings),
            ("backward", backward_rankings),
        ):
            for source_node in range(1, len(rankings)):
                targets = rankings[source_node]
                if candidate_pool_size is not None:
                    targets = targets[:candidate_pool_size]
                for target_node in targets:
                    reason = budget_reason()
                    if reason is not None:
                        stopped_reason = reason
                        return CVRPUpperBoundSearchResult(
                            best=best,
                            verifier_evaluations=evaluations,
                            upper_bound_history=tuple(history),
                            stopped_reason=stopped_reason,
                        )
                    forward = best.certificate.forward_map
                    backward = best.certificate.backward_map
                    current_target = (
                        forward[source_node]
                        if direction == "forward"
                        else backward[source_node]
                    )
                    if target_node == current_target:
                        continue
                    if direction == "forward":
                        updated_forward = list(forward)
                        updated_backward = list(backward)
                        previous_target = updated_forward[source_node]
                        displaced_source = next(
                            (
                                node
                                for node, image in enumerate(updated_forward[1:], 1)
                                if node != source_node and image == target_node
                            ),
                            None,
                        )
                        updated_forward[source_node] = target_node
                        updated_backward[target_node] = source_node
                        if displaced_source is not None:
                            updated_forward[displaced_source] = previous_target
                            updated_backward[previous_target] = displaced_source
                    else:
                        updated_forward = list(forward)
                        updated_backward = list(backward)
                        previous_target = updated_backward[source_node]
                        displaced_source = next(
                            (
                                node
                                for node, image in enumerate(updated_backward[1:], 1)
                                if node != source_node and image == target_node
                            ),
                            None,
                        )
                        updated_backward[source_node] = target_node
                        updated_forward[target_node] = source_node
                        if displaced_source is not None:
                            updated_backward[displaced_source] = previous_target
                            updated_forward[previous_target] = displaced_source
                    candidate = CVRPMapCertificate(
                        tuple(updated_forward), tuple(updated_backward)
                    )
                    trial = verifier.verify(candidate)
                    evaluations += 1
                    strictly_better = trial.upper_bound < best.upper_bound - 1e-15
                    if strictly_better:
                        best = trial
                        changed = True
                        history.append(best.upper_bound)
                        break
        if not changed:
            stopped_reason = "local_optimum"
            break

    return CVRPUpperBoundSearchResult(
        best=best,
        verifier_evaluations=evaluations,
        upper_bound_history=tuple(history),
        stopped_reason=stopped_reason,
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
    "CVRPMapCertificate",
    "CVRPUpperBoundSearchResult",
    "VerifiedCVRPUpperBound",
    "certificate_from_correspondence",
    "search_cvrp_upper_bound",
    "signature_cvrp_map_certificate",
    "verify_cvrp_map_certificate",
]
