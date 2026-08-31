from __future__ import annotations

import heapq
import itertools
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Generic, TypeVar

from pitbench.schema.observation import RunObservation

OutcomeT = TypeVar("OutcomeT")
GroundMetric = Callable[[OutcomeT, OutcomeT], float]

SOLVER_BEHAVIOR_METRIC_NAME = "population-conditional-empirical-wasserstein"
SOLVER_BEHAVIOR_METRIC_VERSION = "1.0"


@dataclass(frozen=True)
class EvaluationContext:
    task_id: str
    population: str
    budget_sec: float
    threads: int

    def __post_init__(self) -> None:
        if not self.task_id or not self.population:
            raise ValueError("task_id and population must be non-empty")
        if not math.isfinite(self.budget_sec) or self.budget_sec <= 0:
            raise ValueError("budget_sec must be finite and positive")
        if self.threads <= 0:
            raise ValueError("threads must be positive")


@dataclass(frozen=True)
class EmpiricalBehaviorKernel(Generic[OutcomeT]):
    """Finite observations of ``K_A(dy | x)`` under one evaluation context."""

    solver_id: str
    context: EvaluationContext
    samples_by_instance: Mapping[str, tuple[OutcomeT, ...]]

    def __post_init__(self) -> None:
        if not self.solver_id:
            raise ValueError("solver_id must be non-empty")
        if not self.samples_by_instance:
            raise ValueError("an empirical kernel requires at least one instance")
        normalized = {
            instance_id: tuple(samples)
            for instance_id, samples in sorted(self.samples_by_instance.items())
        }
        if any(not instance_id for instance_id in normalized):
            raise ValueError("instance ids must be non-empty")
        object.__setattr__(self, "samples_by_instance", MappingProxyType(normalized))


@dataclass(frozen=True)
class SolverDistanceResult:
    distance: float
    p: float
    per_instance: Mapping[str, float]
    population_weights: Mapping[str, float]


@dataclass(frozen=True)
class SolverSensitivityResult:
    lipschitz_constant: float
    witness: tuple[str, str]
    behavior_distance: float
    instance_distance: float
    per_pair_ratio: Mapping[tuple[str, str], float]


@dataclass(frozen=True)
class ResponseGeometryComparison:
    sup_distance: float
    per_pair_difference: Mapping[tuple[str, str], float]


@dataclass
class _ResidualEdge:
    to: int
    reverse: int
    capacity: Fraction
    cost: float


def empirical_kernel_from_observations(
    observations: Sequence[RunObservation],
    projector: Callable[[RunObservation], OutcomeT | None],
    *,
    solver_id: str | None = None,
) -> EmpiricalBehaviorKernel[OutcomeT]:
    """Build a seed-empirical stochastic kernel from one fixed evaluation slice.

    A projector may return ``None`` for an outcome that is mathematically undefined,
    such as conditional performance for a failed run. The instance remains in the
    support with an empty sample rather than silently changing the population.
    """

    rows = list(observations)
    if not rows:
        raise ValueError("cannot build an empirical kernel from no observations")
    contexts = {
        (row.task_id, row.population, row.budget_sec, row.threads) for row in rows
    }
    if len(contexts) != 1:
        raise ValueError(
            "observations must share task, population, budget, and threads"
        )
    code_states = {row.code_state for row in rows}
    if len(code_states) != 1:
        raise ValueError("observations must describe one solver code state")
    task_id, population, budget_sec, threads = contexts.pop()
    code_state = code_states.pop()

    samples: dict[str, list[OutcomeT]] = defaultdict(list)
    seen_runs: set[tuple[str, int]] = set()
    instance_randomness: dict[str, tuple[int, int | None, int | None]] = {}
    for row in sorted(rows, key=lambda item: (item.instance_id, item.solver_seed)):
        run_key = (row.instance_id, row.solver_seed)
        if run_key in seen_runs:
            raise ValueError(f"duplicate observation for instance/seed {run_key!r}")
        seen_runs.add(run_key)
        randomness = (row.instance_seed, row.coordinate_seed, row.demand_seed)
        previous_randomness = instance_randomness.setdefault(
            row.instance_id, randomness
        )
        if previous_randomness != randomness:
            raise ValueError(
                f"instance {row.instance_id!r} has inconsistent randomness metadata"
            )
        samples[row.instance_id]
        outcome = projector(row)
        if outcome is not None:
            samples[row.instance_id].append(outcome)

    return EmpiricalBehaviorKernel(
        solver_id=solver_id or code_state.value,
        context=EvaluationContext(
            task_id=task_id,
            population=population,
            budget_sec=budget_sec,
            threads=threads,
        ),
        samples_by_instance={key: tuple(value) for key, value in samples.items()},
    )


def _metric_cost(
    metric: GroundMetric[OutcomeT], left: OutcomeT, right: OutcomeT, p: float
) -> float:
    distance = float(metric(left, right))
    if not math.isfinite(distance) or distance < 0:
        raise ValueError("ground metric must return finite non-negative distances")
    cost = distance**p
    if not math.isfinite(cost):
        raise ValueError("ground metric cost overflowed")
    return cost


def _add_residual_edge(
    graph: list[list[_ResidualEdge]],
    source: int,
    target: int,
    capacity: Fraction,
    cost: float,
) -> None:
    forward = _ResidualEdge(target, len(graph[target]), capacity, cost)
    reverse = _ResidualEdge(source, len(graph[source]), Fraction(0), -cost)
    graph[source].append(forward)
    graph[target].append(reverse)


def _uniform_transport_cost(costs: list[list[float]]) -> float:
    """Solve the finite uniform transportation problem by min-cost flow."""

    left_count = len(costs)
    right_count = len(costs[0])
    source = 0
    left_offset = 1
    right_offset = left_offset + left_count
    sink = right_offset + right_count
    graph: list[list[_ResidualEdge]] = [[] for _ in range(sink + 1)]
    for left in range(left_count):
        _add_residual_edge(
            graph, source, left_offset + left, Fraction(1, left_count), 0.0
        )
        for right in range(right_count):
            _add_residual_edge(
                graph,
                left_offset + left,
                right_offset + right,
                Fraction(1),
                costs[left][right],
            )
    for right in range(right_count):
        _add_residual_edge(
            graph, right_offset + right, sink, Fraction(1, right_count), 0.0
        )

    flow = Fraction(0)
    total_cost = 0.0
    potentials = [0.0] * len(graph)
    while flow < 1:
        distances = [math.inf] * len(graph)
        previous: list[tuple[int, int] | None] = [None] * len(graph)
        distances[source] = 0.0
        queue = [(0.0, source)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance > distances[node] + 1e-15:
                continue
            for edge_index, edge in enumerate(graph[node]):
                if edge.capacity <= 0:
                    continue
                reduced_cost = edge.cost + potentials[node] - potentials[edge.to]
                # Floating point round-off can make a theoretically non-negative
                # reduced cost very slightly negative.
                if -1e-12 < reduced_cost < 0:
                    reduced_cost = 0.0
                candidate = distance + reduced_cost
                if candidate + 1e-15 < distances[edge.to]:
                    distances[edge.to] = candidate
                    previous[edge.to] = (node, edge_index)
                    heapq.heappush(queue, (candidate, edge.to))
        if previous[sink] is None:
            raise RuntimeError("empirical transport problem is infeasible")

        for node, distance in enumerate(distances):
            if math.isfinite(distance):
                potentials[node] += distance

        amount = 1 - flow
        node = sink
        path_cost = 0.0
        while node != source:
            previous_node, edge_index = previous[node]  # type: ignore[misc]
            edge = graph[previous_node][edge_index]
            amount = min(amount, edge.capacity)
            path_cost += edge.cost
            node = previous_node
        node = sink
        while node != source:
            previous_node, edge_index = previous[node]  # type: ignore[misc]
            edge = graph[previous_node][edge_index]
            reverse_index = edge.reverse
            edge.capacity -= amount
            graph[node][reverse_index].capacity += amount
            node = previous_node
        flow += amount
        total_cost += float(amount) * path_cost
    return max(0.0, total_cost)


def empirical_wasserstein_distance(
    left_samples: Sequence[OutcomeT],
    right_samples: Sequence[OutcomeT],
    ground_metric: GroundMetric[OutcomeT],
    *,
    p: float = 2.0,
) -> float:
    """Exact Wasserstein distance between two finite uniform empirical measures."""

    if not math.isfinite(p) or p < 1:
        raise ValueError("p must be finite and at least one")
    if not left_samples or not right_samples:
        raise ValueError("empirical Wasserstein distance requires non-empty samples")
    costs = [
        [_metric_cost(ground_metric, left, right, p) for right in right_samples]
        for left in left_samples
    ]
    return _uniform_transport_cost(costs) ** (1.0 / p)


def empirical_dispersion(
    samples: Sequence[OutcomeT],
    ground_metric: GroundMetric[OutcomeT],
    *,
    p: float = 2.0,
) -> float:
    """Return the pairwise p-dispersion of one empirical conditional kernel."""

    if not math.isfinite(p) or p < 1:
        raise ValueError("p must be finite and at least one")
    if not samples:
        raise ValueError("dispersion requires at least one sample")
    total = sum(
        _metric_cost(ground_metric, left, right, p)
        for left in samples
        for right in samples
    )
    return (total / (len(samples) ** 2)) ** (1.0 / p)


def _matching_support(
    left: EmpiricalBehaviorKernel[OutcomeT],
    right: EmpiricalBehaviorKernel[OutcomeT],
) -> tuple[str, ...]:
    if left.context != right.context:
        raise ValueError("solver kernels must share one evaluation context")
    left_support = set(left.samples_by_instance)
    right_support = set(right.samples_by_instance)
    if left_support != right_support:
        raise ValueError("solver kernels must have the same instance support")
    return tuple(sorted(left_support))


def _population_weights(
    support: tuple[str, ...], weights: Mapping[str, float] | None
) -> Mapping[str, float]:
    if weights is None:
        uniform = 1.0 / len(support)
        return MappingProxyType({instance: uniform for instance in support})
    if set(weights) != set(support):
        raise ValueError("population weights must exactly match instance support")
    normalized: dict[str, float] = {}
    for instance in support:
        weight = float(weights[instance])
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("population support weights must be finite and positive")
        normalized[instance] = weight
    total = sum(normalized.values())
    return MappingProxyType(
        {instance: weight / total for instance, weight in normalized.items()}
    )


def empirical_solver_distance(
    left: EmpiricalBehaviorKernel[OutcomeT],
    right: EmpiricalBehaviorKernel[OutcomeT],
    ground_metric: GroundMetric[OutcomeT],
    *,
    p: float = 2.0,
    population_weights: Mapping[str, float] | None = None,
) -> SolverDistanceResult:
    """Compute the population Lp aggregation of conditional Wasserstein distances."""

    support = _matching_support(left, right)
    weights = _population_weights(support, population_weights)
    per_instance: dict[str, float] = {}
    for instance in support:
        try:
            per_instance[instance] = empirical_wasserstein_distance(
                left.samples_by_instance[instance],
                right.samples_by_instance[instance],
                ground_metric,
                p=p,
            )
        except ValueError as error:
            raise ValueError(
                f"solver distance is undefined at instance {instance!r}: {error}"
            ) from error
    distance = sum(
        weights[instance] * per_instance[instance] ** p for instance in support
    ) ** (1.0 / p)
    return SolverDistanceResult(
        distance=distance,
        p=p,
        per_instance=MappingProxyType(per_instance),
        population_weights=weights,
    )


def induced_behavior_geometry(
    kernel: EmpiricalBehaviorKernel[OutcomeT],
    ground_metric: GroundMetric[OutcomeT],
    *,
    p: float = 2.0,
) -> Mapping[tuple[str, str], float]:
    """Return the empirical pseudometric ``rho_A(x, x')`` on instance support."""

    if any(not samples for samples in kernel.samples_by_instance.values()):
        raise ValueError("behavior geometry requires samples for every instance")
    distances = {
        (left, right): empirical_wasserstein_distance(
            kernel.samples_by_instance[left],
            kernel.samples_by_instance[right],
            ground_metric,
            p=p,
        )
        for left, right in itertools.combinations(kernel.samples_by_instance, 2)
    }
    if not distances:
        raise ValueError("behavior geometry requires at least two instances")
    return MappingProxyType(distances)


def solver_lipschitz_sensitivity(
    kernel: EmpiricalBehaviorKernel[OutcomeT],
    instance_metric: Callable[[str, str], float],
    ground_metric: GroundMetric[OutcomeT],
    *,
    p: float = 2.0,
) -> SolverSensitivityResult:
    """Compute the finite-support Lipschitz sensitivity and a maximizing witness."""

    geometry = induced_behavior_geometry(kernel, ground_metric, p=p)
    ratios: dict[tuple[str, str], float] = {}
    instance_distances: dict[tuple[str, str], float] = {}
    for pair, behavior_distance in geometry.items():
        instance_distance = float(instance_metric(*pair))
        if not math.isfinite(instance_distance) or instance_distance < 0:
            raise ValueError(
                "instance metric must return finite non-negative distances"
            )
        instance_distances[pair] = instance_distance
        if instance_distance == 0:
            ratios[pair] = math.inf if behavior_distance > 0 else 0.0
        else:
            ratios[pair] = behavior_distance / instance_distance
    witness = max(ratios, key=ratios.__getitem__)
    return SolverSensitivityResult(
        lipschitz_constant=ratios[witness],
        witness=witness,
        behavior_distance=geometry[witness],
        instance_distance=instance_distances[witness],
        per_pair_ratio=MappingProxyType(ratios),
    )


def compare_response_geometries(
    left: EmpiricalBehaviorKernel[OutcomeT],
    right: EmpiricalBehaviorKernel[OutcomeT],
    ground_metric: GroundMetric[OutcomeT],
    *,
    p: float = 2.0,
) -> ResponseGeometryComparison:
    """Compare the two instance pseudometrics induced by solver behavior."""

    _matching_support(left, right)
    left_geometry = induced_behavior_geometry(left, ground_metric, p=p)
    right_geometry = induced_behavior_geometry(right, ground_metric, p=p)
    differences = {
        pair: abs(left_geometry[pair] - right_geometry[pair]) for pair in left_geometry
    }
    return ResponseGeometryComparison(
        sup_distance=max(differences.values()),
        per_pair_difference=MappingProxyType(differences),
    )


__all__ = [
    "SOLVER_BEHAVIOR_METRIC_NAME",
    "SOLVER_BEHAVIOR_METRIC_VERSION",
    "EmpiricalBehaviorKernel",
    "EvaluationContext",
    "ResponseGeometryComparison",
    "SolverDistanceResult",
    "SolverSensitivityResult",
    "compare_response_geometries",
    "empirical_dispersion",
    "empirical_kernel_from_observations",
    "empirical_solver_distance",
    "empirical_wasserstein_distance",
    "induced_behavior_geometry",
    "solver_lipschitz_sensitivity",
]
