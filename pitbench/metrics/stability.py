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
