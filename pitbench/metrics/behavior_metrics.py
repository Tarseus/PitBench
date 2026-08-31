from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, Field

from pitbench.metrics.outcomes import (
    OUTCOME_METRIC_VERSION,
    performance_outcome,
    performance_outcome_distance,
    reliability_outcome,
    reliability_outcome_distance,
    resource_outcome,
    resource_outcome_distance,
)
from pitbench.metrics.solver_behavior import (
    SOLVER_BEHAVIOR_METRIC_NAME,
    SOLVER_BEHAVIOR_METRIC_VERSION,
    empirical_kernel_from_observations,
    empirical_solver_distance,
)
from pitbench.schema.observation import CodeState, RunObservation


class BehaviorDistanceCoordinate(BaseModel):
    distance: float | None = None
    instances_evaluated: int = Field(default=0, ge=0)
    reason: str | None = None


class BehaviorDistanceSlice(BaseModel):
    population: str
    budget_sec: float
    threads: int
    p: float
    performance: BehaviorDistanceCoordinate
    reliability: BehaviorDistanceCoordinate
    resource: BehaviorDistanceCoordinate


class BehaviorMetricReport(BaseModel):
    metric_name: str = SOLVER_BEHAVIOR_METRIC_NAME
    metric_version: str = SOLVER_BEHAVIOR_METRIC_VERSION
    outcome_metric_version: str = OUTCOME_METRIC_VERSION
    slices: list[BehaviorDistanceSlice] = Field(default_factory=list)


def _coordinate(
    base: Sequence[RunObservation],
    agent: Sequence[RunObservation],
    projector: Callable[[RunObservation], Any | None],
    metric: Callable[[Any, Any], float],
    *,
    p: float,
) -> BehaviorDistanceCoordinate:
    try:
        base_kernel = empirical_kernel_from_observations(
            base, projector, solver_id="base"
        )
        agent_kernel = empirical_kernel_from_observations(
            agent, projector, solver_id="agent"
        )
        result = empirical_solver_distance(base_kernel, agent_kernel, metric, p=p)
    except ValueError as error:
        return BehaviorDistanceCoordinate(reason=str(error))
    return BehaviorDistanceCoordinate(
        distance=result.distance,
        instances_evaluated=len(result.per_instance),
    )


def compute_behavior_metric_report(
    observations: Sequence[RunObservation], *, p: float = 2.0
) -> BehaviorMetricReport:
    """Compute the three independent population-conditional solver distances."""

    rows = [obs for obs in observations if obs.equivalence_parent_id is None]
    contexts = sorted(
        {(obs.population, obs.budget_sec, obs.threads) for obs in rows},
        key=lambda item: (item[0], item[1], item[2]),
    )
    slices: list[BehaviorDistanceSlice] = []
    for population, budget, threads in contexts:
        context_rows = [
            obs
            for obs in rows
            if (
                obs.population == population
                and obs.budget_sec == budget
                and obs.threads == threads
            )
        ]
        base = [obs for obs in context_rows if obs.code_state == CodeState.BASE]
        agent = [obs for obs in context_rows if obs.code_state == CodeState.AGENT]
        if not base or not agent:
            continue
        slices.append(
            BehaviorDistanceSlice(
                population=population,
                budget_sec=budget,
                threads=threads,
                p=p,
                performance=_coordinate(
                    base,
                    agent,
                    performance_outcome,
                    performance_outcome_distance,
                    p=p,
                ),
                reliability=_coordinate(
                    base,
                    agent,
                    reliability_outcome,
                    reliability_outcome_distance,
                    p=p,
                ),
                resource=_coordinate(
                    base,
                    agent,
                    resource_outcome,
                    resource_outcome_distance,
                    p=p,
                ),
            )
        )
    return BehaviorMetricReport(slices=slices)


__all__ = [
    "BehaviorDistanceCoordinate",
    "BehaviorDistanceSlice",
    "BehaviorMetricReport",
    "compute_behavior_metric_report",
]
