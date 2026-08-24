from __future__ import annotations

import math
from dataclasses import dataclass

from pitbench.schema.observation import RunObservation, RunStatus

PERFORMANCE_OUTCOME_METRIC_NAME = "conditional-normalized-gap-absolute"
RELIABILITY_OUTCOME_METRIC_NAME = "run-status-validity-discrete"
RESOURCE_OUTCOME_METRIC_NAME = "wall-time-fraction-absolute"
OUTCOME_METRIC_VERSION = "1.0"


def _finite(value: float, field: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")


@dataclass(frozen=True)
class PerformanceOutcome:
    """Oracle-relative performance for one valid, completed run."""

    normalized_gap: float

    def __post_init__(self) -> None:
        _finite(self.normalized_gap, "normalized_gap")


@dataclass(frozen=True)
class ReliabilityOutcome:
    """Categorical termination and independent-validity outcome for one run."""

    status: RunStatus
    valid: bool


@dataclass(frozen=True)
class ResourceOutcome:
    """Wall-clock consumption expressed relative to the declared run budget."""

    wall_time_fraction: float

    def __post_init__(self) -> None:
        _finite(self.wall_time_fraction, "wall_time_fraction")
        if self.wall_time_fraction < 0:
            raise ValueError("wall_time_fraction must be non-negative")


def performance_outcome(observation: RunObservation) -> PerformanceOutcome | None:
    """Project a run to conditional performance, or ``None`` when undefined.

    Performance is deliberately conditional on successful independent validation.
    Failure probability belongs to the separate reliability geometry.
    """

    if (
        observation.status is not RunStatus.COMPLETED
        or not observation.valid
        or observation.normalized_gap is None
    ):
        return None
    return PerformanceOutcome(normalized_gap=observation.normalized_gap)


def reliability_outcome(observation: RunObservation) -> ReliabilityOutcome:
    return ReliabilityOutcome(status=observation.status, valid=observation.valid)


def resource_outcome(observation: RunObservation) -> ResourceOutcome | None:
    """Project a run to the version-1 resource geometry when timing is available."""

    if observation.wall_time_sec is None:
        return None
    return ResourceOutcome(
        wall_time_fraction=observation.wall_time_sec / observation.budget_sec
    )


def performance_outcome_distance(
    left: PerformanceOutcome, right: PerformanceOutcome
) -> float:
    return abs(left.normalized_gap - right.normalized_gap)


def reliability_outcome_distance(
    left: ReliabilityOutcome, right: ReliabilityOutcome
) -> float:
    return 0.0 if left == right else 1.0


def resource_outcome_distance(left: ResourceOutcome, right: ResourceOutcome) -> float:
    return abs(left.wall_time_fraction - right.wall_time_fraction)


__all__ = [
    "OUTCOME_METRIC_VERSION",
    "PERFORMANCE_OUTCOME_METRIC_NAME",
    "RELIABILITY_OUTCOME_METRIC_NAME",
    "RESOURCE_OUTCOME_METRIC_NAME",
    "PerformanceOutcome",
    "ReliabilityOutcome",
    "ResourceOutcome",
    "performance_outcome",
    "performance_outcome_distance",
    "reliability_outcome",
    "reliability_outcome_distance",
    "resource_outcome",
    "resource_outcome_distance",
]
