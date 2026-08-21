from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ConfoundingRecord:
    record_id: str
    group: str
    outcome: float
    features: Mapping[str, float]


@dataclass(frozen=True)
class RecoveryMethodResult:
    estimated_effect: float
    absolute_error: float
    maximum_structural_imbalance: float


def _cost(
    left: ConfoundingRecord, right: ConfoundingRecord, axes: Sequence[str]
) -> float:
    return math.sqrt(
        sum((left.features[axis] - right.features[axis]) ** 2 for axis in axes)
    )


def greedy_pairs(
    source: Sequence[ConfoundingRecord],
    target: Sequence[ConfoundingRecord],
    *,
    axes: Sequence[str],
) -> tuple[tuple[int, int], ...]:
    if len(source) != len(target):
        raise ValueError("one-to-one matching requires equal population sizes")
    available = set(range(len(target)))
    pairs = []
    for left_index, left in enumerate(source):
        right_index = min(
            available,
            key=lambda index: (_cost(left, target[index], axes), index),
        )
        available.remove(right_index)
        pairs.append((left_index, right_index))
    return tuple(pairs)


def optimal_pairs(
    source: Sequence[ConfoundingRecord],
    target: Sequence[ConfoundingRecord],
    *,
    axes: Sequence[str],
) -> tuple[tuple[int, int], ...]:
    """Exact equal-mass OT via a deterministic Hungarian assignment."""

    if len(source) != len(target) or not source:
        raise ValueError("exact matching requires equal non-empty populations")
    size = len(source)
    costs = [[_cost(left, right, axes) for right in target] for left in source]
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    assignment = [0] * (size + 1)
    predecessor = [0] * (size + 1)
    for row in range(1, size + 1):
        assignment[0] = row
        column = 0
        minimum = [math.inf] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column] = True
            current_row = assignment[column]
            delta = math.inf
            next_column = 0
            for candidate in range(1, size + 1):
                if used[candidate]:
                    continue
                reduced = (
                    costs[current_row - 1][candidate - 1]
                    - u[current_row]
                    - v[candidate]
                )
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    predecessor[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            for candidate in range(size + 1):
                if used[candidate]:
                    u[assignment[candidate]] += delta
                    v[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if assignment[column] == 0:
                break
        while True:
            previous = predecessor[column]
            assignment[column] = assignment[previous]
            column = previous
            if column == 0:
                break
    return tuple(
        sorted((assignment[column] - 1, column - 1) for column in range(1, size + 1))
    )


def matched_effect(
    source: Sequence[ConfoundingRecord],
    target: Sequence[ConfoundingRecord],
    pairs: Sequence[tuple[int, int]],
) -> float:
    return statistics.fmean(
        target[right].outcome - source[left].outcome for left, right in pairs
    )


def maximum_standardized_imbalance(
    source: Sequence[ConfoundingRecord],
    target: Sequence[ConfoundingRecord],
    pairs: Sequence[tuple[int, int]],
    *,
    axes: Sequence[str],
) -> float:
    imbalances = []
    for axis in axes:
        left = [source[i].features[axis] for i, _ in pairs]
        right = [target[j].features[axis] for _, j in pairs]
        pooled = math.sqrt(
            (statistics.pvariance(left) + statistics.pvariance(right)) / 2
        )
        delta = abs(statistics.fmean(right) - statistics.fmean(left))
        imbalances.append(
            delta / pooled if pooled else (0.0 if delta == 0 else math.inf)
        )
    return max(imbalances, default=0.0)


def compare_recovery_methods(
    source: Sequence[ConfoundingRecord],
    target: Sequence[ConfoundingRecord],
    *,
    structural_axes: Sequence[str],
    oracle_effect: float,
) -> dict[str, RecoveryMethodResult]:
    if len(source) != len(target) or not source:
        raise ValueError("recovery comparison requires equal non-empty populations")
    all_pairs = tuple((index, index) for index in range(len(source)))
    methods = {
        "naive": all_pairs,
        "size_only": greedy_pairs(source, target, axes=()),
        "greedy": greedy_pairs(source, target, axes=structural_axes),
        "ot": optimal_pairs(source, target, axes=structural_axes),
    }
    result = {}
    for name, pairs in methods.items():
        effect = matched_effect(source, target, pairs)
        result[name] = RecoveryMethodResult(
            estimated_effect=effect,
            absolute_error=abs(effect - oracle_effect),
            maximum_structural_imbalance=maximum_standardized_imbalance(
                source, target, pairs, axes=structural_axes
            ),
        )
    return result


def ot_incremental_value(
    greedy_errors: Sequence[float],
    ot_errors: Sequence[float],
    greedy_imbalances: Sequence[float],
    ot_imbalances: Sequence[float],
    *,
    repetitions: int = 10_000,
    seed: int = 0,
) -> dict[str, float | bool]:
    lengths = {
        len(greedy_errors),
        len(ot_errors),
        len(greedy_imbalances),
        len(ot_imbalances),
    }
    if lengths == {0} or len(lengths) != 1 or repetitions <= 0:
        raise ValueError("OT comparison requires equally sized non-empty samples")
    greedy_rmse = math.sqrt(statistics.fmean(value**2 for value in greedy_errors))
    ot_rmse = math.sqrt(statistics.fmean(value**2 for value in ot_errors))
    reduction = (greedy_rmse - ot_rmse) / greedy_rmse if greedy_rmse else 0.0
    differences = [
        greedy**2 - ot**2 for greedy, ot in zip(greedy_errors, ot_errors, strict=True)
    ]
    rng = random.Random(seed)
    bootstrap = []
    for _ in range(repetitions):
        sample = [differences[rng.randrange(len(differences))] for _ in differences]
        bootstrap.append(statistics.fmean(sample))
    bootstrap.sort()
    lower = bootstrap[int(0.025 * (repetitions - 1))]
    imbalance_not_worse = statistics.fmean(ot_imbalances) <= statistics.fmean(
        greedy_imbalances
    )
    return {
        "greedy_rmse": greedy_rmse,
        "ot_rmse": ot_rmse,
        "relative_rmse_reduction": reduction,
        "bootstrap_mse_improvement_lower_95": lower,
        "imbalance_not_worse": imbalance_not_worse,
        "retain_ot": reduction >= 0.10 and lower > 0 and imbalance_not_worse,
    }
