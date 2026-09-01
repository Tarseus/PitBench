from __future__ import annotations

import math
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence

from pydantic import BaseModel, Field

from pitbench.schema.observation import CodeState, RunObservation, RunStatus

BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 20260824
INSTANCE_BOOTSTRAP_METHOD = "instance bootstrap over per-instance seed means"


class ConfidenceInterval(BaseModel):
    lower: float
    upper: float
    level: float = 0.95
    method: str = INSTANCE_BOOTSTRAP_METHOD
    resamples: int = BOOTSTRAP_RESAMPLES


class GapEstimate(BaseModel):
    total_runs: int = Field(ge=0)
    valid_runs: int = Field(ge=0)
    failure_counts: dict[str, int] = Field(default_factory=dict)
    solver_seeds: list[int] = Field(default_factory=list)
    mean_normalized_gap: float | None = None
    median_normalized_gap: float | None = None
    p95_normalized_gap: float | None = None
    mean_ci95: ConfidenceInterval | None = None


class PairedGapEvidence(BaseModel):
    paired_runs: int = Field(ge=0)
    paired_instances: int = Field(ge=0)
    solver_seeds: list[int] = Field(default_factory=list)
    agent_better: int = Field(ge=0)
    equal: int = Field(ge=0)
    agent_worse: int = Field(ge=0)
    mean_gap_reduction: float | None = None
    mean_gap_reduction_ci95: ConfidenceInterval | None = None
    mean_gap_reduction_by_seed: dict[int, float] = Field(default_factory=dict)


class BudgetPerformance(BaseModel):
    budget_sec: float = Field(gt=0)
    base: GapEstimate
    agent: GapEstimate
    paired: PairedGapEvidence


class InstanceSetPerformance(BaseModel):
    instance_set_kind: str
    by_budget: dict[str, BudgetPerformance]


class HeldOutRetention(BaseModel):
    budget_sec: float = Field(gt=0)
    judge_id_gap_reduction: float
    judge_shift_gap_reduction: float
    shift_minus_id_gap_reduction: float
    retained_on_shift: bool


class PerformanceReport(BaseModel):
    primary_instance_set_kind: str
    primary_budget_sec: float = Field(gt=0)
    solver_seeds: list[int]
    budgets_sec: list[float]
    primary: BudgetPerformance
    instance_sets: dict[str, InstanceSetPerformance]
    held_out_retention: list[HeldOutRetention] = Field(default_factory=list)


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
        item
        for item in observations
        if item.valid and item.normalized_gap is not None
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
    failure_counts = Counter(
        item.status.value
        for item in observations
        if item.status != RunStatus.COMPLETED
    )
    return GapEstimate(
        total_runs=len(observations),
        valid_runs=sum(item.valid for item in observations),
        failure_counts=dict(sorted(failure_counts.items())),
        solver_seeds=sorted({item.solver_seed for item in observations}),
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
    deltas_by_seed: dict[int, list[float]] = defaultdict(list)
    deltas: list[float] = []
    solver_seeds: set[int] = set()
    agent_better = 0
    equal = 0
    agent_worse = 0

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
        deltas.append(delta)
        instance_key = (base_item.instance_set, base_item.instance_id)
        deltas_by_instance[instance_key].append(delta)
        deltas_by_seed[base_item.solver_seed].append(delta)
        solver_seeds.add(base_item.solver_seed)
        if delta > 0:
            agent_better += 1
        elif math.isclose(delta, 0.0, rel_tol=0.0, abs_tol=1e-12):
            equal += 1
        else:
            agent_worse += 1

    cluster_means = [
        statistics.fmean(values) for values in deltas_by_instance.values()
    ]
    return PairedGapEvidence(
        paired_runs=len(deltas),
        paired_instances=len(deltas_by_instance),
        solver_seeds=sorted(solver_seeds),
        agent_better=agent_better,
        equal=equal,
        agent_worse=agent_worse,
        mean_gap_reduction=(
            statistics.fmean(cluster_means) if cluster_means else None
        ),
        mean_gap_reduction_ci95=_cluster_mean_ci(
            deltas_by_instance,
            seed=seed,
        ),
        mean_gap_reduction_by_seed={
            solver_seed: statistics.fmean(values)
            for solver_seed, values in sorted(deltas_by_seed.items())
        },
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
    originals = [
        item for item in observations if item.equivalence_parent_id is None
    ]
    if not originals:
        raise ValueError("performance report requires original observations")

    budgets = sorted({item.budget_sec for item in originals})
    solver_seeds = sorted({item.solver_seed for item in originals})
    by_kind: dict[str, list[RunObservation]] = defaultdict(list)
    for item in originals:
        by_kind[item.instance_set_kind or item.instance_set].append(item)

    instance_set_kinds = sorted(by_kind)
    primary_kind = "judge_id" if "judge_id" in by_kind else instance_set_kinds[0]
    primary_kind_budgets = {
        item.budget_sec for item in by_kind[primary_kind]
    }
    if primary_budget_sec not in primary_kind_budgets:
        raise ValueError(
            f"declared primary budget {primary_budget_sec:g}s has no observations "
            f"for primary instance set {primary_kind!r}"
        )
    instance_sets: dict[str, InstanceSetPerformance] = {}
    for kind_index, kind in enumerate(instance_set_kinds):
        by_budget = {}
        kind_budgets = sorted({item.budget_sec for item in by_kind[kind]})
        for budget_index, budget in enumerate(kind_budgets):
            by_budget[f"{budget:g}"] = _budget_performance(
                by_kind[kind],
                budget,
                seed=BOOTSTRAP_SEED + kind_index * 100 + budget_index * 3,
            )
        instance_sets[kind] = InstanceSetPerformance(
            instance_set_kind=kind,
            by_budget=by_budget,
        )

    primary = instance_sets[primary_kind].by_budget[f"{primary_budget_sec:g}"]

    held_out_retention: list[HeldOutRetention] = []
    if "judge_id" in instance_sets and "judge_shift" in instance_sets:
        id_by_budget = instance_sets["judge_id"].by_budget
        shift_by_budget = instance_sets["judge_shift"].by_budget
        for budget_key in sorted(
            set(id_by_budget) & set(shift_by_budget),
            key=float,
        ):
            id_delta = id_by_budget[budget_key].paired.mean_gap_reduction
            shift_delta = shift_by_budget[budget_key].paired.mean_gap_reduction
            if id_delta is None or shift_delta is None:
                continue
            held_out_retention.append(
                HeldOutRetention(
                    budget_sec=float(budget_key),
                    judge_id_gap_reduction=id_delta,
                    judge_shift_gap_reduction=shift_delta,
                    shift_minus_id_gap_reduction=shift_delta - id_delta,
                    retained_on_shift=shift_delta >= 0,
                )
            )

    return PerformanceReport(
        primary_instance_set_kind=primary_kind,
        primary_budget_sec=primary_budget_sec,
        solver_seeds=solver_seeds,
        budgets_sec=budgets,
        primary=primary,
        instance_sets=instance_sets,
        held_out_retention=held_out_retention,
    )


def _format_gap(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.3f}%"


def format_performance_report(report: PerformanceReport) -> str:
    lines = [
        "Independent validity and quality-time performance",
        "",
        "Instance set | Budget | Base valid | Agent valid | Base mean gap | "
        "Agent mean gap | Base-Agent gap reduction (95% CI) | "
        "Agent better/equal/worse",
        "--- | ---: | ---: | ---: | ---: | ---: | ---: | ---:",
    ]
    for kind, instance_set in report.instance_sets.items():
        for cell in instance_set.by_budget.values():
            paired = cell.paired
            delta = _format_gap(paired.mean_gap_reduction)
            if paired.mean_gap_reduction_ci95 is not None:
                interval = paired.mean_gap_reduction_ci95
                delta = (
                    f"{delta} [{_format_gap(interval.lower)}, "
                    f"{_format_gap(interval.upper)}]"
                )
            lines.append(
                f"{kind} | {cell.budget_sec:g}s | "
                f"{cell.base.valid_runs}/{cell.base.total_runs} | "
                f"{cell.agent.valid_runs}/{cell.agent.total_runs} | "
                f"{_format_gap(cell.base.mean_normalized_gap)} | "
                f"{_format_gap(cell.agent.mean_normalized_gap)} | {delta} | "
                f"{paired.agent_better}/{paired.equal}/{paired.agent_worse}"
            )

    if report.held_out_retention:
        lines.extend(
            [
                "",
                "Held-out instance-set retention",
                "",
                "Budget | Judge-ID reduction | Hidden-shift reduction | Shift - ID",
                "---: | ---: | ---: | ---:",
            ]
        )
        for item in report.held_out_retention:
            lines.append(
                f"{item.budget_sec:g}s | "
                f"{_format_gap(item.judge_id_gap_reduction)} | "
                f"{_format_gap(item.judge_shift_gap_reduction)} | "
                f"{_format_gap(item.shift_minus_id_gap_reduction)}"
            )

    lines.extend(
        [
            "",
            "Repeatability evidence",
            "",
            f"Solver seeds: {', '.join(map(str, report.solver_seeds))}",
            "95% intervals use an instance-cluster bootstrap; seeds remain paired "
            "within each resampled instance.",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "BudgetPerformance",
    "ConfidenceInterval",
    "GapEstimate",
    "HeldOutRetention",
    "PairedGapEvidence",
    "PerformanceReport",
    "InstanceSetPerformance",
    "compute_performance_report",
    "format_performance_report",
]
