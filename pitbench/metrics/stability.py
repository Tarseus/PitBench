from __future__ import annotations

import statistics
from collections.abc import Iterable


def instance_stability(losses: Iterable[float]) -> dict[str, float]:
    values = list(losses)
    if not values:
        raise ValueError("stability requires at least one solver seed")
    ordered = sorted(values)
    q1 = ordered[(len(ordered) - 1) // 4]
    q3 = ordered[(3 * (len(ordered) - 1)) // 4]
    return {
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "range": max(values) - min(values),
        "iqr": q3 - q1,
    }


def wasserstein_1d(first: Iterable[float], second: Iterable[float]) -> float:
    """Exact empirical Wasserstein-1 distance for one-dimensional samples."""

    left = sorted(first)
    right = sorted(second)
    if not left or not right:
        raise ValueError("Wasserstein distance requires two non-empty samples")
    left_index = right_index = 0
    left_mass = 1 / len(left)
    right_mass = 1 / len(right)
    left_remaining = left_mass
    right_remaining = right_mass
    total = 0.0
    while left_index < len(left) and right_index < len(right):
        moved = min(left_remaining, right_remaining)
        total += moved * abs(left[left_index] - right[right_index])
        left_remaining -= moved
        right_remaining -= moved
        if left_remaining <= 1e-15:
            left_index += 1
            left_remaining = left_mass
        if right_remaining <= 1e-15:
            right_index += 1
            right_remaining = right_mass
    return total


def orbit_distribution_dispersion(
    outcomes_by_transform: dict[str, Iterable[float]],
) -> dict[str, float]:
    samples = {name: list(values) for name, values in outcomes_by_transform.items()}
    if "identity" not in samples or len(samples) < 2:
        raise ValueError("orbit dispersion requires identity and another transform")
    identity_distances = [
        wasserstein_1d(samples["identity"], values)
        for name, values in samples.items()
        if name != "identity"
    ]
    names = sorted(samples)
    pairwise = [
        wasserstein_1d(samples[first], samples[second])
        for index, first in enumerate(names)
        for second in names[index + 1 :]
    ]
    return {
        "identity_wasserstein_mean": statistics.fmean(identity_distances),
        "identity_wasserstein_max": max(identity_distances),
        "orbit_pairwise_wasserstein_mean": statistics.fmean(pairwise),
        "orbit_pairwise_wasserstein_max": max(pairwise),
    }
