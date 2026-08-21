from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class IncumbentPoint:
    time_sec: float
    objective: float


@dataclass(frozen=True)
class AnytimeOutcomes:
    feasible: bool
    reference_gap: float | None
    primal_integral: float
    time_to_target_sec: float | None
    target_hit: bool


def _validated_trajectory(
    trajectory: Sequence[IncumbentPoint], budget_sec: float
) -> tuple[IncumbentPoint, ...]:
    if budget_sec <= 0:
        raise ValueError("budget_sec must be positive")
    previous_time = -math.inf
    previous_objective = math.inf
    result = []
    for point in trajectory:
        if point.time_sec < 0:
            raise ValueError("trajectory times must be non-negative")
        if point.time_sec <= previous_time:
            raise ValueError("trajectory times must be strictly increasing")
        if point.objective > previous_objective:
            raise ValueError("incumbent objectives must be non-increasing")
        if not math.isfinite(point.objective) or point.objective < 0:
            raise ValueError("trajectory objectives must be finite and non-negative")
        if point.time_sec <= budget_sec:
            result.append(point)
        previous_time = point.time_sec
        previous_objective = point.objective
    return tuple(result)


def anytime_outcomes(
    trajectory: Sequence[IncumbentPoint],
    *,
    reference: float,
    budget_sec: float = 10.0,
    target_gap: float = 0.01,
    gap_cap: float = 1.0,
) -> AnytimeOutcomes:
    """Compute the pre-registered two-part and anytime outcomes for one run."""

    if reference <= 0 or target_gap < 0 or gap_cap <= 0:
        raise ValueError(
            "reference and gap_cap must be positive; target_gap non-negative"
        )
    points = _validated_trajectory(trajectory, budget_sec)

    def gap(objective: float) -> float:
        return min(gap_cap, max(0.0, (objective - reference) / reference))

    area = 0.0
    cursor = 0.0
    current_gap = gap_cap
    hit_time = None
    for point in points:
        area += (point.time_sec - cursor) * current_gap
        cursor = point.time_sec
        current_gap = gap(point.objective)
        if hit_time is None and current_gap <= target_gap:
            hit_time = point.time_sec
    area += (budget_sec - cursor) * current_gap
    feasible = bool(points)
    return AnytimeOutcomes(
        feasible=feasible,
        reference_gap=gap(points[-1].objective) if feasible else None,
        primal_integral=area / budget_sec,
        time_to_target_sec=hit_time,
        target_hit=hit_time is not None,
    )


def paired_mean_difference(
    source: Mapping[str, float], target: Mapping[str, float]
) -> float:
    if set(source) != set(target) or not source:
        raise ValueError("paired outcomes require identical non-empty keys")
    return statistics.fmean(target[key] - source[key] for key in source)


def clustered_bootstrap_mean_difference(
    paired_differences: Mapping[str, Sequence[float]],
    *,
    repetitions: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap generator-seed clusters and return a percentile 95% interval."""

    if repetitions <= 0 or not paired_differences:
        raise ValueError("bootstrap requires clusters and positive repetitions")
    clusters = tuple(paired_differences)
    if any(not paired_differences[key] for key in clusters):
        raise ValueError("bootstrap clusters must be non-empty")
    rng = random.Random(seed)
    estimates = []
    for _ in range(repetitions):
        sampled = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        values = [value for key in sampled for value in paired_differences[key]]
        estimates.append(statistics.fmean(values))
    estimates.sort()
    lower = estimates[int(0.025 * (repetitions - 1))]
    upper = estimates[int(0.975 * (repetitions - 1))]
    return lower, upper
