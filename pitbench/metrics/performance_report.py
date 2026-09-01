from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from collections.abc import Sequence
from enum import Enum

from pydantic import BaseModel, Field

from pitbench.schema.observation import CodeState, RunObservation

BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 20260824
INSTANCE_BOOTSTRAP_METHOD = "instance bootstrap over per-instance seed means"


class PerformanceClassification(str, Enum):
    IMPROVED = "improved"
    REGRESSED = "regressed"
    INCONCLUSIVE = "inconclusive"
    INCOMPLETE = "incomplete"


class ConfidenceInterval(BaseModel):
    lower: float
    upper: float
    level: float = 0.95
    method: str = INSTANCE_BOOTSTRAP_METHOD
    resamples: int = BOOTSTRAP_RESAMPLES


class GapEstimate(BaseModel):
    mean_normalized_gap: float | None = None
    median_normalized_gap: float | None = None
    p95_normalized_gap: float | None = None
    mean_ci95: ConfidenceInterval | None = None


class PairedGapEvidence(BaseModel):
    paired_instances: int = Field(ge=0)
    mean_gap_reduction: float | None = None
    mean_gap_reduction_ci95: ConfidenceInterval | None = None


class BudgetPerformance(BaseModel):
    budget_sec: float = Field(gt=0)
    base: GapEstimate
    agent: GapEstimate
    paired: PairedGapEvidence


class PerformanceReport(BaseModel):
    classification: PerformanceClassification
    primary_budget_sec: float = Field(gt=0)
    budgets_sec: list[float]
    primary: BudgetPerformance
    by_budget: dict[str, BudgetPerformance]


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute percentile of empty sequence")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _cluster_mean_ci(
    values_by_instance: dict[tuple[str, str], list[float]],
    *,
    seed: int,
) -> ConfidenceInterval | None:
    cluster_means = [statistics.fmean(values) for values in values_by_instance.values()]
    if not cluster_means:
        return None
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(rng.choice(cluster_means) for _ in cluster_means)
        for _ in range(BOOTSTRAP_RESAMPLES)
    )
    return ConfidenceInterval(
        lower=means[int(0.025 * len(means))],
        upper=means[int(0.975 * len(means))],
    )


def _gap_estimate(
    observations: Sequence[RunObservation],
    *,
    seed: int,
) -> GapEstimate:
    valid = [
        item for item in observations if item.valid and item.normalized_gap is not None
    ]
    values_by_instance: dict[tuple[str, str], list[float]] = defaultdict(list)
    for item in valid:
        assert item.normalized_gap is not None
        values_by_instance[(item.instance_set, item.instance_id)].append(
            item.normalized_gap
        )
    instance_means = [
        statistics.fmean(values) for values in values_by_instance.values()
    ]
    return GapEstimate(
        mean_normalized_gap=(
            statistics.fmean(instance_means) if instance_means else None
        ),
        median_normalized_gap=(
            statistics.median(instance_means) if instance_means else None
        ),
        p95_normalized_gap=(
            _percentile(instance_means, 0.95) if instance_means else None
        ),
        mean_ci95=_cluster_mean_ci(values_by_instance, seed=seed),
    )


def _paired_gap_evidence(
    base: Sequence[RunObservation],
    agent: Sequence[RunObservation],
    *,
    seed: int,
) -> PairedGapEvidence:
    def keyed(
        observations: Sequence[RunObservation],
    ) -> dict[tuple[str, str, int], RunObservation]:
        return {
            (item.instance_set, item.instance_id, item.solver_seed): item
            for item in observations
        }

    base_by_key = keyed(base)
    agent_by_key = keyed(agent)
    deltas_by_instance: dict[tuple[str, str], list[float]] = defaultdict(list)
    for key in sorted(set(base_by_key) & set(agent_by_key)):
        base_item = base_by_key[key]
        agent_item = agent_by_key[key]
        if (
            not base_item.valid
            or not agent_item.valid
            or base_item.normalized_gap is None
            or agent_item.normalized_gap is None
        ):
            continue
        delta = base_item.normalized_gap - agent_item.normalized_gap
        instance_key = (base_item.instance_set, base_item.instance_id)
        deltas_by_instance[instance_key].append(delta)

    cluster_means = [statistics.fmean(values) for values in deltas_by_instance.values()]
    return PairedGapEvidence(
        paired_instances=len(deltas_by_instance),
        mean_gap_reduction=(statistics.fmean(cluster_means) if cluster_means else None),
        mean_gap_reduction_ci95=_cluster_mean_ci(
            deltas_by_instance,
            seed=seed,
        ),
    )


def _budget_performance(
    observations: Sequence[RunObservation],
    budget_sec: float,
    *,
    seed: int,
) -> BudgetPerformance:
    selected = [item for item in observations if item.budget_sec == budget_sec]
    base = [item for item in selected if item.code_state == CodeState.BASE]
    agent = [item for item in selected if item.code_state == CodeState.AGENT]
    return BudgetPerformance(
        budget_sec=budget_sec,
        base=_gap_estimate(base, seed=seed),
        agent=_gap_estimate(agent, seed=seed + 1),
        paired=_paired_gap_evidence(base, agent, seed=seed + 2),
    )


def compute_performance_report(
    observations: Sequence[RunObservation],
    *,
    primary_budget_sec: float,
) -> PerformanceReport:
    judge_id = [
        item
        for item in observations
        if item.equivalence_parent_id is None
        and (item.instance_set_kind or item.instance_set) == "judge_id"
    ]
    if not judge_id:
        raise ValueError("performance report requires original judge-ID observations")

    budgets = sorted({item.budget_sec for item in judge_id})
    if primary_budget_sec not in budgets:
        raise ValueError(
            f"declared primary budget {primary_budget_sec:g}s has no observations "
            "for the judge-ID instance set"
        )
    by_budget = {
        f"{budget:g}": _budget_performance(
            judge_id,
            budget,
            seed=BOOTSTRAP_SEED + budget_index * 3,
        )
        for budget_index, budget in enumerate(budgets)
    }

    primary = by_budget[f"{primary_budget_sec:g}"]
    primary_ci = primary.paired.mean_gap_reduction_ci95
    if primary_ci is None:
        classification = PerformanceClassification.INCOMPLETE
    elif primary_ci.lower > 0:
        classification = PerformanceClassification.IMPROVED
    elif primary_ci.upper < 0:
        classification = PerformanceClassification.REGRESSED
    else:
        classification = PerformanceClassification.INCONCLUSIVE

    return PerformanceReport(
        classification=classification,
        primary_budget_sec=primary_budget_sec,
        budgets_sec=budgets,
        primary=primary,
        by_budget=by_budget,
    )


def _format_gap(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.3f}%"


def format_performance_report(report: PerformanceReport) -> str:
    lines = [
        "Fixed-budget performance",
        f"Classification: {report.classification.value}",
        "",
        "Budget | Base mean gap | Agent mean gap | "
        "Base-Agent gap reduction (95% CI) | Paired instances",
        "---: | ---: | ---: | ---: | ---:",
    ]
    for cell in report.by_budget.values():
        paired = cell.paired
        delta = _format_gap(paired.mean_gap_reduction)
        if paired.mean_gap_reduction_ci95 is not None:
            interval = paired.mean_gap_reduction_ci95
            delta = (
                f"{delta} [{_format_gap(interval.lower)}, "
                f"{_format_gap(interval.upper)}]"
            )
        lines.append(
            f"{cell.budget_sec:g}s | "
            f"{_format_gap(cell.base.mean_normalized_gap)} | "
            f"{_format_gap(cell.agent.mean_normalized_gap)} | {delta} | "
            f"{paired.paired_instances}"
        )

    lines.extend(
        [
            "",
            "95% intervals use an instance bootstrap over per-instance seed means.",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "BudgetPerformance",
    "ConfidenceInterval",
    "GapEstimate",
    "PairedGapEvidence",
    "PerformanceClassification",
    "PerformanceReport",
    "compute_performance_report",
    "format_performance_report",
]
