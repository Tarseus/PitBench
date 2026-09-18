from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from collections.abc import Sequence
from enum import Enum

from pydantic import BaseModel, Field

from pitbench.schema.observation import (
    CodeState,
    ExpectedRun,
    ExpectedRunGrid,
    RunObservation,
    RunStatus,
    SolverTermination,
)
from pitbench.schema.task import ExactTimeBasis, PerformanceProtocol, PitBenchTask

BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 20260824
INSTANCE_BOOTSTRAP_METHOD = "instance bootstrap over per-instance seed means"


class PerformanceClassification(str, Enum):
    IMPROVED = "improved"
    REGRESSED = "regressed"
    INCONCLUSIVE = "inconclusive"
    INCOMPLETE = "incomplete"


class ExactRunOutcomeKind(str, Enum):
    VERIFIED_SOLVED = "verified_solved"
    SOLVER_ORIGIN_UNSOLVED = "solver_origin_unsolved"
    ORACLE_DISCREPANCY = "oracle_discrepancy"
    QUALIFICATION_FAILURE = "qualification_failure"
    INFRASTRUCTURE_MISSING = "infrastructure_missing"


class ExactRunOutcome(BaseModel):
    kind: ExactRunOutcomeKind
    solved: bool | None = None
    penalized_runtime_sec: float | None = None


def compute_exact_run_outcome(
    observation: RunObservation | None,
    *,
    time_basis: ExactTimeBasis = ExactTimeBasis.CPU_TIME,
) -> ExactRunOutcome:
    if observation is None or observation.status in {
        RunStatus.BUILD_FAILED,
        RunStatus.INFRASTRUCTURE_ERROR,
    }:
        return ExactRunOutcome(kind=ExactRunOutcomeKind.INFRASTRUCTURE_MISSING)

    if observation.status == RunStatus.INVALID:
        return ExactRunOutcome(kind=ExactRunOutcomeKind.QUALIFICATION_FAILURE)

    if observation.status != RunStatus.COMPLETED:
        return ExactRunOutcome(
            kind=ExactRunOutcomeKind.SOLVER_ORIGIN_UNSOLVED,
            solved=False,
            penalized_runtime_sec=2 * observation.budget_sec,
        )

    if not observation.valid:
        return ExactRunOutcome(kind=ExactRunOutcomeKind.QUALIFICATION_FAILURE)

    if observation.solver_termination != SolverTermination.OPTIMAL:
        return ExactRunOutcome(
            kind=ExactRunOutcomeKind.SOLVER_ORIGIN_UNSOLVED,
            solved=False,
            penalized_runtime_sec=2 * observation.budget_sec,
        )

    if observation.optimal_or_bks is None:
        return ExactRunOutcome(kind=ExactRunOutcomeKind.INFRASTRUCTURE_MISSING)

    if (
        observation.objective is None
        or observation.reported_objective is None
        or observation.objective != observation.reported_objective
    ):
        return ExactRunOutcome(kind=ExactRunOutcomeKind.QUALIFICATION_FAILURE)

    if observation.objective < observation.optimal_or_bks:
        return ExactRunOutcome(kind=ExactRunOutcomeKind.ORACLE_DISCREPANCY)

    if observation.objective > observation.optimal_or_bks:
        return ExactRunOutcome(
            kind=ExactRunOutcomeKind.SOLVER_ORIGIN_UNSOLVED,
            solved=False,
            penalized_runtime_sec=2 * observation.budget_sec,
        )

    measured_time_sec = (
        observation.cpu_time_sec
        if time_basis == ExactTimeBasis.CPU_TIME
        else observation.wall_time_sec
    )
    if (
        measured_time_sec is None
        or not math.isfinite(measured_time_sec)
        or measured_time_sec < 0
    ):
        return ExactRunOutcome(kind=ExactRunOutcomeKind.INFRASTRUCTURE_MISSING)

    return ExactRunOutcome(
        kind=ExactRunOutcomeKind.VERIFIED_SOLVED,
        solved=True,
        penalized_runtime_sec=measured_time_sec,
    )


class ConfidenceInterval(BaseModel):
    lower: float
    upper: float
    level: float = 0.95
    method: str = INSTANCE_BOOTSTRAP_METHOD
    resamples: int = BOOTSTRAP_RESAMPLES


class ExactStatePerformance(BaseModel):
    verified_solved_coverage: float | None = None
    penalized_average_runtime_sec: float | None = None
    penalized_average_runtime_ci95: ConfidenceInterval | None = None
    solved_run_count: int = Field(ge=0)
    expected_run_count: int = Field(ge=0)
    complete: bool
    outcome_counts: dict[str, int]


class PairedExactPerformance(BaseModel):
    paired_instances: int = Field(ge=0)
    mean_penalized_runtime_reduction_sec: float | None = None
    mean_penalized_runtime_reduction_ci95: ConfidenceInterval | None = None
    complete: bool


class ExactBudgetPerformance(BaseModel):
    budget_sec: float = Field(gt=0)
    base: ExactStatePerformance
    agent: ExactStatePerformance
    paired: PairedExactPerformance


class ExactPerformanceReport(BaseModel):
    classification: PerformanceClassification
    time_basis: ExactTimeBasis
    primary_budget_sec: float = Field(gt=0)
    budgets_sec: list[float]
    primary: ExactBudgetPerformance
    by_budget: dict[str, ExactBudgetPerformance]


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


def _run_key(
    run: ExpectedRun | RunObservation,
) -> tuple[str, str, CodeState, int, float]:
    return (
        run.instance_set,
        run.instance_id,
        run.code_state,
        run.solver_seed,
        run.budget_sec,
    )


def _is_original_judge_id(run: ExpectedRun | RunObservation) -> bool:
    return (
        run.equivalence_parent_id is None
        and run.test_suite is None
        and (run.instance_set_kind or run.instance_set) == "judge_id"
    )


def _exact_state_performance(
    expected_runs: Sequence[ExpectedRun],
    observations_by_key: dict[tuple[str, str, CodeState, int, float], RunObservation],
    *,
    seed: int,
    time_basis: ExactTimeBasis,
) -> ExactStatePerformance:
    outcomes_by_instance: dict[tuple[str, str], list[ExactRunOutcome]] = defaultdict(
        list
    )
    outcome_counts = {kind.value: 0 for kind in ExactRunOutcomeKind}
    for expected_run in expected_runs:
        outcome = compute_exact_run_outcome(
            observations_by_key.get(_run_key(expected_run)),
            time_basis=time_basis,
        )
        outcomes_by_instance[
            (expected_run.instance_set, expected_run.instance_id)
        ].append(outcome)
        outcome_counts[outcome.kind.value] += 1

    ordinary_kinds = {
        ExactRunOutcomeKind.VERIFIED_SOLVED,
        ExactRunOutcomeKind.SOLVER_ORIGIN_UNSOLVED,
    }
    complete = bool(expected_runs) and all(
        outcome.kind in ordinary_kinds
        for outcomes in outcomes_by_instance.values()
        for outcome in outcomes
    )
    solved_run_count = outcome_counts[ExactRunOutcomeKind.VERIFIED_SOLVED.value]
    if not complete:
        return ExactStatePerformance(
            solved_run_count=solved_run_count,
            expected_run_count=len(expected_runs),
            complete=False,
            outcome_counts=outcome_counts,
        )

    solved_by_instance = {
        instance: [float(outcome.solved) for outcome in outcomes]
        for instance, outcomes in outcomes_by_instance.items()
    }
    runtime_by_instance = {
        instance: [
            outcome.penalized_runtime_sec
            for outcome in outcomes
            if outcome.penalized_runtime_sec is not None
        ]
        for instance, outcomes in outcomes_by_instance.items()
    }
    instance_solved_coverage = [
        statistics.fmean(values) for values in solved_by_instance.values()
    ]
    instance_penalized_runtime = [
        statistics.fmean(values) for values in runtime_by_instance.values()
    ]
    return ExactStatePerformance(
        verified_solved_coverage=statistics.fmean(instance_solved_coverage),
        penalized_average_runtime_sec=statistics.fmean(instance_penalized_runtime),
        penalized_average_runtime_ci95=_cluster_mean_ci(
            runtime_by_instance,
            seed=seed,
        ),
        solved_run_count=solved_run_count,
        expected_run_count=len(expected_runs),
        complete=True,
        outcome_counts=outcome_counts,
    )


def _paired_exact_performance(
    expected_runs: Sequence[ExpectedRun],
    observations_by_key: dict[tuple[str, str, CodeState, int, float], RunObservation],
    *,
    seed: int,
    time_basis: ExactTimeBasis,
) -> PairedExactPerformance:
    expected_by_state = {
        state: {
            (
                run.instance_set,
                run.instance_id,
                run.solver_seed,
                run.budget_sec,
            ): run
            for run in expected_runs
            if run.code_state == state
        }
        for state in CodeState
    }
    if set(expected_by_state[CodeState.BASE]) != set(
        expected_by_state[CodeState.AGENT]
    ):
        raise ValueError("exact expected run grid does not pair Base and Agent")

    reductions_by_instance: dict[tuple[str, str], list[float]] = defaultdict(list)
    complete = True
    for pair_key in sorted(expected_by_state[CodeState.BASE]):
        base_expected = expected_by_state[CodeState.BASE][pair_key]
        agent_expected = expected_by_state[CodeState.AGENT][pair_key]
        base_outcome = compute_exact_run_outcome(
            observations_by_key.get(_run_key(base_expected)),
            time_basis=time_basis,
        )
        agent_outcome = compute_exact_run_outcome(
            observations_by_key.get(_run_key(agent_expected)),
            time_basis=time_basis,
        )
        if (
            base_outcome.penalized_runtime_sec is None
            or agent_outcome.penalized_runtime_sec is None
        ):
            complete = False
            continue
        reductions_by_instance[(pair_key[0], pair_key[1])].append(
            base_outcome.penalized_runtime_sec - agent_outcome.penalized_runtime_sec
        )

    if not complete or not reductions_by_instance:
        return PairedExactPerformance(
            paired_instances=len(reductions_by_instance),
            complete=False,
        )

    instance_reductions = [
        statistics.fmean(values) for values in reductions_by_instance.values()
    ]
    return PairedExactPerformance(
        paired_instances=len(reductions_by_instance),
        mean_penalized_runtime_reduction_sec=statistics.fmean(instance_reductions),
        mean_penalized_runtime_reduction_ci95=_cluster_mean_ci(
            reductions_by_instance,
            seed=seed,
        ),
        complete=True,
    )


def compute_exact_performance_report(
    observations: Sequence[RunObservation],
    *,
    primary_budget_sec: float,
    budgets_sec: Sequence[float],
    expected_run_grid: ExpectedRunGrid,
    time_basis: ExactTimeBasis = ExactTimeBasis.CPU_TIME,
) -> ExactPerformanceReport:
    declared_budgets = list(budgets_sec)
    if len(declared_budgets) != len(set(declared_budgets)):
        raise ValueError("exact performance budgets must be unique")
    if primary_budget_sec not in declared_budgets:
        raise ValueError("exact primary budget must belong to declared budgets")

    expected_runs = [
        run for run in expected_run_grid.runs if _is_original_judge_id(run)
    ]
    if not expected_runs:
        raise ValueError("exact performance requires an expected judge-ID run grid")
    if any(run.task_id != expected_run_grid.task_id for run in expected_runs):
        raise ValueError("expected run belongs to a different task")

    expected_by_key = {_run_key(run): run for run in expected_runs}
    if len(expected_by_key) != len(expected_runs):
        raise ValueError("duplicate run in exact expected grid")
    if {run.budget_sec for run in expected_runs} != set(declared_budgets):
        raise ValueError("exact expected grid differs from declared budgets")

    selected_observations = [
        observation
        for observation in observations
        if _is_original_judge_id(observation)
    ]
    if any(
        observation.task_id != expected_run_grid.task_id
        for observation in selected_observations
    ):
        raise ValueError("exact observation belongs to a different task")
    observations_by_key = {
        _run_key(observation): observation for observation in selected_observations
    }
    if len(observations_by_key) != len(selected_observations):
        raise ValueError("duplicate exact run observation")
    unexpected_keys = set(observations_by_key) - set(expected_by_key)
    if unexpected_keys:
        raise ValueError("exact observation is outside the expected run grid")

    by_budget = {}
    for budget_index, budget_sec in enumerate(declared_budgets):
        budget_runs = [run for run in expected_runs if run.budget_sec == budget_sec]
        base = _exact_state_performance(
            [run for run in budget_runs if run.code_state == CodeState.BASE],
            observations_by_key,
            seed=BOOTSTRAP_SEED + budget_index * 3,
            time_basis=time_basis,
        )
        agent = _exact_state_performance(
            [run for run in budget_runs if run.code_state == CodeState.AGENT],
            observations_by_key,
            seed=BOOTSTRAP_SEED + budget_index * 3 + 1,
            time_basis=time_basis,
        )
        paired = _paired_exact_performance(
            budget_runs,
            observations_by_key,
            seed=BOOTSTRAP_SEED + budget_index * 3 + 2,
            time_basis=time_basis,
        )
        by_budget[f"{budget_sec:g}"] = ExactBudgetPerformance(
            budget_sec=budget_sec,
            base=base,
            agent=agent,
            paired=paired,
        )

    primary = by_budget[f"{primary_budget_sec:g}"]
    base_coverage = primary.base.verified_solved_coverage
    agent_coverage = primary.agent.verified_solved_coverage
    paired_interval = primary.paired.mean_penalized_runtime_reduction_ci95
    if (
        not primary.base.complete
        or not primary.agent.complete
        or not primary.paired.complete
        or base_coverage is None
        or agent_coverage is None
        or paired_interval is None
    ):
        classification = PerformanceClassification.INCOMPLETE
    elif agent_coverage > base_coverage:
        classification = PerformanceClassification.IMPROVED
    elif agent_coverage < base_coverage:
        classification = PerformanceClassification.REGRESSED
    elif paired_interval.lower > 0:
        classification = PerformanceClassification.IMPROVED
    elif paired_interval.upper < 0:
        classification = PerformanceClassification.REGRESSED
    else:
        classification = PerformanceClassification.INCONCLUSIVE

    return ExactPerformanceReport(
        classification=classification,
        time_basis=time_basis,
        primary_budget_sec=primary_budget_sec,
        budgets_sec=declared_budgets,
        primary=primary,
        by_budget=by_budget,
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
        and item.test_suite is None
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


def compute_task_performance_report(
    observations: Sequence[RunObservation],
    *,
    task: PitBenchTask,
    expected_run_grid: ExpectedRunGrid | None = None,
) -> PerformanceReport | ExactPerformanceReport | None:
    protocol = task.evaluation.performance_protocol
    if protocol == PerformanceProtocol.HEURISTIC_FIXED_BUDGET:
        return compute_performance_report(
            observations,
            primary_budget_sec=task.evaluation.primary_budget_sec,
        )
    if protocol == PerformanceProtocol.EXACT_VERIFIED_SOLVE:
        if expected_run_grid is None:
            raise ValueError(
                "exact performance requires the evaluator-owned expected run grid"
            )
        if expected_run_grid.task_id != task.task_id:
            raise ValueError("expected run grid belongs to a different task")
        return compute_exact_performance_report(
            observations,
            primary_budget_sec=task.evaluation.primary_budget_sec,
            budgets_sec=task.evaluation.budgets_sec,
            expected_run_grid=expected_run_grid,
            time_basis=task.evaluation.exact_time_basis,
        )
    if protocol == PerformanceProtocol.VERIFIED_CP_SAT_MODEL_CONSTRUCTION:
        return None
    raise ValueError(f"unsupported performance protocol: {protocol}")


def _format_gap(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.3f}%"


def _format_exact_performance_report(report: ExactPerformanceReport) -> str:
    lines = [
        "Verified exact-solver performance",
        f"Classification: {report.classification.value}",
        f"Timing basis: {report.time_basis.value}",
        "",
        "Budget | Base solved | Agent solved | Base PAR-2 | Agent PAR-2 | "
        "Base-Agent PAR-2 reduction (95% CI)",
        "---: | ---: | ---: | ---: | ---: | ---:",
    ]
    for cell in report.by_budget.values():
        paired = cell.paired
        reduction = "-"
        if paired.mean_penalized_runtime_reduction_sec is not None:
            reduction = f"{paired.mean_penalized_runtime_reduction_sec:.6g}s"
        if paired.mean_penalized_runtime_reduction_ci95 is not None:
            interval = paired.mean_penalized_runtime_reduction_ci95
            reduction += f" [{interval.lower:.6g}s, {interval.upper:.6g}s]"

        def coverage(state: ExactStatePerformance) -> str:
            if state.verified_solved_coverage is None:
                return f"{state.solved_run_count}/{state.expected_run_count}"
            return (
                f"{state.verified_solved_coverage:.3%} "
                f"({state.solved_run_count}/{state.expected_run_count})"
            )

        def runtime(state: ExactStatePerformance) -> str:
            if state.penalized_average_runtime_sec is None:
                return "-"
            return f"{state.penalized_average_runtime_sec:.6g}s"

        lines.append(
            f"{cell.budget_sec:g}s | {coverage(cell.base)} | "
            f"{coverage(cell.agent)} | {runtime(cell.base)} | "
            f"{runtime(cell.agent)} | {reduction}"
        )
    lines.extend(
        [
            "",
            "Coverage is primary; equal coverage is classified by paired PAR-2.",
            "95% intervals use an instance bootstrap over per-instance seed means.",
        ]
    )
    return "\n".join(lines)


def format_performance_report(
    report: PerformanceReport | ExactPerformanceReport,
) -> str:
    if isinstance(report, ExactPerformanceReport):
        return _format_exact_performance_report(report)
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
    "ExactBudgetPerformance",
    "ExactPerformanceReport",
    "ExactRunOutcome",
    "ExactRunOutcomeKind",
    "ExactStatePerformance",
    "GapEstimate",
    "PairedGapEvidence",
    "PairedExactPerformance",
    "PerformanceClassification",
    "PerformanceReport",
    "compute_exact_run_outcome",
    "compute_exact_performance_report",
    "compute_performance_report",
    "compute_task_performance_report",
    "format_performance_report",
]
