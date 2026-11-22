from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from pitbench.evaluator.storage import ObservationStore
from pitbench.metrics.resource_report import compute_resource_reports
from pitbench.problem_families.base import objectives_equal
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


def classify_exact_performance(
    *,
    coverage_difference: float | None,
    coverage_interval: tuple[float, float] | None,
    runtime_interval: tuple[float, float] | None,
    complete: bool,
) -> PerformanceClassification:
    """Apply the common coverage-first exact decision to paired evidence."""
    if (
        not complete
        or coverage_difference is None
        or coverage_interval is None
        or runtime_interval is None
    ):
        return PerformanceClassification.INCOMPLETE
    if coverage_difference > 0:
        return (
            PerformanceClassification.IMPROVED
            if coverage_interval[0] > 0
            else PerformanceClassification.INCONCLUSIVE
        )
    if coverage_difference < 0:
        return (
            PerformanceClassification.REGRESSED
            if coverage_interval[1] < 0
            else PerformanceClassification.INCONCLUSIVE
        )
    if runtime_interval[0] > 0:
        return PerformanceClassification.IMPROVED
    if runtime_interval[1] < 0:
        return PerformanceClassification.REGRESSED
    return PerformanceClassification.INCONCLUSIVE


class ExactRunOutcomeKind(str, Enum):
    VERIFIED_SOLVED = "verified_solved"
    SOLVER_ORIGIN_UNSOLVED = "solver_origin_unsolved"
    SOLVER_FAILURE = "solver_failure"
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

    if observation.status == RunStatus.NO_SOLUTION and (
        observation.solver_termination == SolverTermination.TIME_LIMIT
    ):
        return ExactRunOutcome(
            kind=ExactRunOutcomeKind.SOLVER_ORIGIN_UNSOLVED,
            solved=False,
            penalized_runtime_sec=2 * observation.budget_sec,
        )

    if observation.status != RunStatus.COMPLETED:
        return ExactRunOutcome(
            kind=ExactRunOutcomeKind.SOLVER_FAILURE,
            solved=False,
            penalized_runtime_sec=2 * observation.budget_sec,
        )

    if not observation.valid:
        return ExactRunOutcome(kind=ExactRunOutcomeKind.QUALIFICATION_FAILURE)

    if observation.solver_termination != SolverTermination.OPTIMAL:
        return ExactRunOutcome(
            kind=(
                ExactRunOutcomeKind.SOLVER_ORIGIN_UNSOLVED
                if observation.solver_termination == SolverTermination.TIME_LIMIT
                else ExactRunOutcomeKind.SOLVER_FAILURE
            ),
            solved=False,
            penalized_runtime_sec=2 * observation.budget_sec,
        )

    if observation.optimal_or_bks is None or not math.isfinite(
        observation.optimal_or_bks
    ):
        return ExactRunOutcome(kind=ExactRunOutcomeKind.INFRASTRUCTURE_MISSING)

    if not objectives_equal(observation.objective, observation.reported_objective):
        return ExactRunOutcome(kind=ExactRunOutcomeKind.QUALIFICATION_FAILURE)

    if not objectives_equal(observation.objective, observation.optimal_or_bks):
        if observation.objective < observation.optimal_or_bks:
            return ExactRunOutcome(kind=ExactRunOutcomeKind.ORACLE_DISCREPANCY)
        return ExactRunOutcome(kind=ExactRunOutcomeKind.QUALIFICATION_FAILURE)

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
    mean_verified_solved_coverage_reduction: float | None = None
    mean_verified_solved_coverage_reduction_ci95: ConfidenceInterval | None = None
    mean_penalized_runtime_reduction_sec: float | None = None
    mean_penalized_runtime_reduction_ci95: ConfidenceInterval | None = None
    complete: bool


class ExactBudgetPerformance(BaseModel):
    budget_sec: float = Field(gt=0)
    base: ExactStatePerformance
    agent: ExactStatePerformance
    paired: PairedExactPerformance


class PerformanceConfirmation(BaseModel):
    initial_classification: PerformanceClassification
    repeat_classification: PerformanceClassification | None = None
    confirmed: bool
    repeat_observation_count: int = Field(ge=0)
    repeat_valid_observation_count: int = Field(ge=0)
    repeat_order: str = "reversed_base_agent"


class ExactPerformanceReport(BaseModel):
    classification: PerformanceClassification
    confirmation: PerformanceConfirmation | None = None
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


class PairedAvailability(BaseModel):
    paired_runs: int = Field(ge=0)
    both_valid: int = Field(ge=0)
    base_only_valid: int = Field(ge=0)
    agent_only_valid: int = Field(ge=0)
    neither_valid: int = Field(ge=0)
    base_observation_missing: int = Field(ge=0)
    agent_observation_missing: int = Field(ge=0)


class BudgetPerformance(BaseModel):
    budget_sec: float = Field(gt=0)
    base: GapEstimate
    agent: GapEstimate
    paired: PairedGapEvidence
    paired_availability: PairedAvailability | None = None


class PerformanceReport(BaseModel):
    classification: PerformanceClassification
    confirmation: PerformanceConfirmation | None = None
    primary_budget_sec: float = Field(gt=0)
    budgets_sec: list[float]
    primary: BudgetPerformance
    by_budget: dict[str, BudgetPerformance]


def adjudicate_performance_confirmation(
    initial: PerformanceReport | ExactPerformanceReport,
    repeat: PerformanceReport | ExactPerformanceReport,
    *,
    repeat_observation_count: int,
    repeat_valid_observation_count: int,
) -> PerformanceReport | ExactPerformanceReport:
    """Interpret retained historical confirmation artifacts without running solvers."""
    if type(initial) is not type(repeat):
        raise ValueError("confirmation uses a different Performance protocol")
    if (
        initial.primary_budget_sec != repeat.primary_budget_sec
        or initial.budgets_sec != repeat.budgets_sec
    ):
        raise ValueError("confirmation uses different budgets")
    direction = initial.classification
    if direction not in {
        PerformanceClassification.IMPROVED,
        PerformanceClassification.REGRESSED,
    }:
        raise ValueError("confirmation requires an initial directional result")
    confirmed = repeat.classification is direction
    final = (
        direction
        if confirmed
        else PerformanceClassification.INCOMPLETE
        if repeat.classification is PerformanceClassification.INCOMPLETE
        else PerformanceClassification.INCONCLUSIVE
    )
    return initial.model_copy(
        update={
            "classification": final,
            "confirmation": PerformanceConfirmation(
                initial_classification=direction,
                repeat_classification=repeat.classification,
                confirmed=confirmed,
                repeat_observation_count=repeat_observation_count,
                repeat_valid_observation_count=repeat_valid_observation_count,
            ),
        }
    )


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
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> ConfidenceInterval | None:
    cluster_means = [statistics.fmean(values) for values in values_by_instance.values()]
    if not cluster_means:
        return None
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(rng.choice(cluster_means) for _ in cluster_means)
        for _ in range(resamples)
    )
    return ConfidenceInterval(
        lower=means[int(0.025 * len(means))],
        upper=means[int(0.975 * len(means))],
        resamples=resamples,
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
        ExactRunOutcomeKind.SOLVER_FAILURE,
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
    coverage_reductions_by_instance: dict[tuple[str, str], list[float]] = defaultdict(
        list
    )
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
            or base_outcome.solved is None
            or agent_outcome.solved is None
        ):
            complete = False
            continue
        coverage_reductions_by_instance[(pair_key[0], pair_key[1])].append(
            float(agent_outcome.solved) - float(base_outcome.solved)
        )
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
    instance_coverage_reductions = [
        statistics.fmean(values) for values in coverage_reductions_by_instance.values()
    ]
    return PairedExactPerformance(
        paired_instances=len(reductions_by_instance),
        mean_verified_solved_coverage_reduction=statistics.fmean(
            instance_coverage_reductions
        ),
        mean_verified_solved_coverage_reduction_ci95=_cluster_mean_ci(
            coverage_reductions_by_instance,
            seed=seed,
        ),
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
    coverage_interval = primary.paired.mean_verified_solved_coverage_reduction_ci95
    paired_interval = primary.paired.mean_penalized_runtime_reduction_ci95
    classification = classify_exact_performance(
        coverage_difference=(
            agent_coverage - base_coverage
            if agent_coverage is not None and base_coverage is not None
            else None
        ),
        coverage_interval=(
            (coverage_interval.lower, coverage_interval.upper)
            if coverage_interval is not None
            else None
        ),
        runtime_interval=(
            (paired_interval.lower, paired_interval.upper)
            if paired_interval is not None
            else None
        ),
        complete=(
            primary.base.complete and primary.agent.complete and primary.paired.complete
        ),
    )

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


def paired_solution_availability(
    base: Sequence[RunObservation], agent: Sequence[RunObservation],
) -> PairedAvailability:
    def key(item):
        return item.instance_set, item.instance_id, item.solver_seed

    base_by_key = {key(item): item for item in base}
    agent_by_key = {key(item): item for item in agent}
    common = set(base_by_key) & set(agent_by_key)
    return PairedAvailability(
        paired_runs=len(common),
        both_valid=sum(base_by_key[k].valid and agent_by_key[k].valid for k in common),
        base_only_valid=sum(base_by_key[k].valid and not agent_by_key[k].valid for k in common),
        agent_only_valid=sum(not base_by_key[k].valid and agent_by_key[k].valid for k in common),
        neither_valid=sum(not base_by_key[k].valid and not agent_by_key[k].valid for k in common),
        base_observation_missing=len(set(agent_by_key) - set(base_by_key)),
        agent_observation_missing=len(set(base_by_key) - set(agent_by_key)),
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
        paired_availability=paired_solution_availability(base, agent),
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
    if protocol in {
        PerformanceProtocol.HEURISTIC_FIXED_BUDGET,
        PerformanceProtocol.BUDGET_TRUNCATED_QUALITY,
    }:
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
    raise ValueError(f"unsupported performance protocol: {protocol}")


def compute_panel_performance_reports(
    observations: Sequence[RunObservation],
    *,
    task: PitBenchTask,
    expected_run_grid: ExpectedRunGrid | None = None,
) -> dict[str, PerformanceReport | ExactPerformanceReport]:
    """Describe each hidden panel with the unchanged native estimator.

    These panel estimates do not replace the frozen judge-ID classification or
    define a controlled-retention protocol. Preserve all expected panel runs so
    a missing exact observation remains missing.
    """
    reports = {}
    for panel in task.instance_sets:
        if panel.kind.value not in {"judge_id", "judge_shift"}:
            continue
        rows = [
            row.model_copy(
                update={"instance_set": "judge_id", "instance_set_kind": "judge_id"}
            )
            for row in observations
            if row.instance_set == panel.name
            and row.test_suite is None
            and row.equivalence_parent_id is None
        ]
        if not rows:
            continue
        grid = None
        if expected_run_grid is not None:
            grid = ExpectedRunGrid(
                task_id=task.task_id,
                runs=[
                    run.model_copy(
                        update={
                            "instance_set": "judge_id",
                            "instance_set_kind": "judge_id",
                        }
                    )
                    for run in expected_run_grid.runs
                    if run.instance_set == panel.name
                    and run.test_suite is None
                    and run.equivalence_parent_id is None
                ],
            )
        panel_task = task
        if (
            panel.performance_protocol is not None
            or panel.budgets_sec is not None
            or panel.solver_seed_count is not None
        ):
            panel_task = task.model_copy(deep=True)
            if panel.performance_protocol is not None:
                panel_task.evaluation.performance_protocol = panel.performance_protocol
            if panel.budgets_sec is not None:
                panel_task.evaluation.budgets_sec = panel.budgets_sec
                panel_task.evaluation.primary_budget_sec = panel.primary_budget_sec
        reports[panel.name] = compute_task_performance_report(
            rows, task=panel_task, expected_run_grid=grid
        )
    return reports


def _format_gap(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.3f}%"


def _confirmation_text(
    report: PerformanceReport | ExactPerformanceReport,
) -> str | None:
    if report.confirmation is not None:
        evidence = report.confirmation
        repeat = (
            evidence.repeat_classification.value
            if evidence.repeat_classification is not None
            else "not_run"
        )
        return (
            "Fresh opposite-order confirmation: "
            f"initial={evidence.initial_classification.value}, "
            f"repeat={repeat}, "
            f"confirmed={evidence.confirmed}"
        )
    if report.classification in {
        PerformanceClassification.IMPROVED,
        PerformanceClassification.REGRESSED,
    }:
        return "Single-run direction; independent confirmation is required."
    return None


def _format_exact_performance_report(report: ExactPerformanceReport) -> str:
    lines = [
        "Verified exact-solver performance",
        f"Classification: {report.classification.value}",
        f"Timing basis: {report.time_basis.value}",
        "",
        "Budget | Base solved | Agent solved | Agent-Base solved coverage "
        "(95% CI) | Base PAR-2 | Agent PAR-2 | "
        "Base-Agent PAR-2 reduction (95% CI) | Base failures | Agent failures",
        "---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:",
    ]
    confirmation_text = _confirmation_text(report)
    if confirmation_text is not None:
        lines.insert(2, confirmation_text)
    for cell in report.by_budget.values():
        paired = cell.paired
        reduction = "-"
        if paired.mean_penalized_runtime_reduction_sec is not None:
            reduction = f"{paired.mean_penalized_runtime_reduction_sec:.6g}s"
        if paired.mean_penalized_runtime_reduction_ci95 is not None:
            interval = paired.mean_penalized_runtime_reduction_ci95
            reduction += f" [{interval.lower:.6g}s, {interval.upper:.6g}s]"
        coverage_reduction = "-"
        if paired.mean_verified_solved_coverage_reduction is not None:
            coverage_reduction = f"{paired.mean_verified_solved_coverage_reduction:.3%}"
        if paired.mean_verified_solved_coverage_reduction_ci95 is not None:
            interval = paired.mean_verified_solved_coverage_reduction_ci95
            coverage_reduction += f" [{interval.lower:.3%}, {interval.upper:.3%}]"

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
            f"{coverage(cell.agent)} | {coverage_reduction} | "
            f"{runtime(cell.base)} | "
            f"{runtime(cell.agent)} | {reduction} | "
            f"{cell.base.outcome_counts.get('solver_failure', '-')} | "
            f"{cell.agent.outcome_counts.get('solver_failure', '-')}"
        )
    lines.extend(
        [
            "",
            "Coverage is primary; directional coverage needs an interval excluding "
            "zero. Equal coverage is classified by paired PAR-2.",
            "Solver failures are reported separately from normal time limits; "
            "both retain the existing PAR-2 penalty.",
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
    confirmation_text = _confirmation_text(report)
    if confirmation_text is not None:
        lines.insert(2, confirmation_text)
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
    for cell in report.by_budget.values():
        availability = cell.paired_availability
        if availability is not None:
            lines.append(
                f"{cell.budget_sec:g}s paired valid solutions: both {availability.both_valid}; "
                f"Base only {availability.base_only_valid}; Agent only {availability.agent_only_valid}; "
                f"neither {availability.neither_valid}; missing observations "
                f"{availability.base_observation_missing + availability.agent_observation_missing}."
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
    "PerformanceConfirmation",
    "PerformanceReport",
    "adjudicate_performance_confirmation",
    "compute_exact_run_outcome",
    "compute_exact_performance_report",
    "compute_performance_report",
    "compute_task_performance_report",
    "ReleaseControlAnalysis",
    "format_performance_report",
]


class ReleaseControlAnalysis:
    """Predeclared control statistics; budgets, seeds and estimators are unchanged."""

    ROOT = Path(__file__).resolve().parents[2]
    BOOTSTRAP_SEED = 20261017
    BOOTSTRAP_DRAWS = 10000

    @staticmethod
    def _source_path(value: str) -> Path:
        if value.startswith("private://"):
            return (
                ReleaseControlAnalysis.ROOT
                / "private"
                / value.removeprefix("private://")
            )
        path = Path(value)
        return path if path.is_absolute() else ReleaseControlAnalysis.ROOT / path

    @staticmethod
    def _interval(values: list[float]) -> dict[str, float] | None:
        interval = _cluster_mean_ci(
            {("control", str(index)): [value] for index, value in enumerate(values)},
            seed=ReleaseControlAnalysis.BOOTSTRAP_SEED,
            resamples=ReleaseControlAnalysis.BOOTSTRAP_DRAWS,
        )
        return (
            interval.model_dump(include={"lower", "upper"})
            if interval is not None
            else None
        )

    @staticmethod
    def _sign_test(effects: list[float]) -> float | None:
        positives = sum((value > 1e-12 for value in effects))
        negatives = sum((value < -1e-12 for value in effects))
        total = positives + negatives
        if not total:
            return None
        return (
            sum((math.comb(total, index) for index in range(positives, total + 1)))
            / 2**total
        )

    @staticmethod
    def _distribution(effects: dict[str, float]) -> dict[str, Any]:
        values = list(effects.values())
        ordered = sorted(values)
        largest = max(effects, key=lambda instance: abs(effects[instance]))
        trimmed = ordered[3:-3] if len(ordered) >= 10 else ordered
        return {
            "instance_count": len(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "worst": min(values),
            "best": max(values),
            "trimmed_mean": statistics.fmean(trimmed),
            "better_instances": sum((value > 1e-12 for value in values)),
            "equal_instances": sum((abs(value) <= 1e-12 for value in values)),
            "worse_instances": sum((value < -1e-12 for value in values)),
            "ci95": ReleaseControlAnalysis._interval(values),
            "one_sided_sign_test_p": ReleaseControlAnalysis._sign_test(values),
            "largest_absolute_instance": largest,
            "mean_without_largest_absolute_instance": statistics.fmean(
                (value for instance, value in effects.items() if instance != largest)
            ),
            "per_instance": dict(sorted(effects.items())),
        }

    @staticmethod
    def _read_pairs(path: Path) -> tuple[dict[tuple[int, str, int], dict], int]:
        latest: dict[tuple[int, str, int], dict] = {}
        attempts = 0
        for line in path.read_text().splitlines():
            if not line:
                continue
            row = json.loads(line)
            key = (row["round"], row["instance_id"], row["solver_seed"])
            attempts += 1
            if key not in latest or all(
                (row[label]["status"] == "completed" for label in ("old", "new"))
            ):
                latest[key] = row
        return (latest, attempts)

    @staticmethod
    def analyze_baseline_qualification(
        manifest_path: Path, output_dir: Path
    ) -> dict[str, Any]:
        from pitbench.repositories.pr_curation import verified_exact_timeout_incumbent

        "Summarize a one-state hidden baseline without claiming a human PR gain."
        from pitbench.evaluator.artifacts import selected_output_attempt

        output_dir = selected_output_attempt(output_dir)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("run_mode") != "baseline_qualification":
            raise ValueError(
                "baseline analysis requires a baseline qualification manifest"
            )
        identity = json.loads((output_dir / "identity.json").read_text())
        task = PitBenchTask.from_yaml(
            ReleaseControlAnalysis._source_path(manifest["task_config"])
        )
        exact = (
            task.evaluation.performance_protocol
            == PerformanceProtocol.EXACT_VERIFIED_SOLVE
        )
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if identity.get("manifest_sha256") != manifest_sha:
            raise ValueError("baseline manifest differs from retained run identity")
        oracle = yaml.safe_load(
            ReleaseControlAnalysis._source_path(manifest["oracle_path"]).read_text()
        )
        expected = {
            (item["id"], seed)
            for item in oracle["instances"]
            for seed in identity["selected_seeds"]
        }
        latest = {}
        attempts = 0
        for line in (output_dir / "baseline_runs.jsonl").read_text().splitlines():
            if not line:
                continue
            row = json.loads(line)
            key = (row["instance_id"], row["solver_seed"])
            attempts += 1
            if (
                key not in latest
                or row["base"]["status"] == "completed"
                or (
                    exact
                    and row["base"]["status"] == "no_solution"
                    and (
                        row["base"].get("reported", {}).get("solver_termination")
                        == "time_limit"
                    )
                )
            ):
                latest[key] = row
        missing = sorted(expected - latest.keys())
        if exact:
            objective_sense = task.oracle.objective_sense or "minimize"
            known_optimum = task.oracle.kind == "known_optimum"
            ordinary_unsolved = {
                key
                for key, row in latest.items()
                if row["base"]["status"] == "no_solution"
                and row["base"].get("reported", {}).get("solver_termination")
                == "time_limit"
                or verified_exact_timeout_incumbent(
                    row["base"],
                    row.get("target"),
                    objective_sense=objective_sense,
                    known_optimum=known_optimum,
                )
            }
            failures = [
                {"instance_id": key[0], "seed": key[1], "status": row["base"]["status"]}
                for key, row in latest.items()
                if row["base"]["status"] != "completed" and key not in ordinary_unsolved
            ]
            invalid = []
            solved = 0
            for key, row in latest.items():
                base = row["base"]
                if base["status"] != "completed":
                    continue
                if key in ordinary_unsolved:
                    continue
                reported, verified, target = (
                    base.get("reported_objective"),
                    base.get("verified_objective"),
                    row.get("target"),
                )
                if (
                    base.get("reported_valid") is not True
                    or base.get("solver_termination") != "optimal"
                    or any(
                        (
                            type(value) not in {int, float}
                            for value in (reported, verified, target)
                        )
                    )
                    or (not math.isclose(reported, verified, rel_tol=0, abs_tol=1e-06))
                    or (not math.isclose(verified, target, rel_tol=0, abs_tol=1e-06))
                ):
                    invalid.append({"instance_id": key[0], "seed": key[1]})
                else:
                    solved += 1
            complete = not missing and (not failures) and (not invalid)
            unsolved = len(ordinary_unsolved)
            return {
                "manifest_sha256": manifest_sha,
                "source_identity": identity,
                "expected_runs": len(expected),
                "retained_run_attempts": attempts,
                "observed_unique_runs": len(latest),
                "missing_runs": missing,
                "failed_runs": failures,
                "invalid_runs": invalid,
                "base_verified_solved_runs": solved,
                "base_time_limit_runs": unsolved,
                "base_time_limit_with_incumbent_runs": sum(
                    (
                        verified_exact_timeout_incumbent(
                            row["base"],
                            row.get("target"),
                            objective_sense=objective_sense,
                            known_optimum=known_optimum,
                        )
                        for row in latest.values()
                    )
                ),
                "base_verified_solved_coverage": solved / len(expected)
                if complete and expected
                else None,
                "conclusion": "verified_unsolved_room"
                if complete and solved > 0 and (unsolved > 0)
                else "no_mixed_exact_outcomes"
                if complete
                else "incomplete",
                "claim_boundary": "This qualifies an exact-solver open challenge with both verified optimal and ordinary time-limit outcomes. It does not establish a human PR gain or Agent success.",
            }
        failures = [
            {"instance_id": key[0], "seed": key[1], "status": row["base"]["status"]}
            for key, row in latest.items()
            if row["base"]["status"] != "completed"
        ]
        gaps = []
        invalid = []
        for key, row in latest.items():
            base = row["base"]
            if base["status"] != "completed":
                continue
            gap = base.get("normalized_gap")
            reported = base.get("reported_objective")
            verified = base.get("verified_objective")
            if (
                base.get("reported_valid") is not True
                or type(gap) not in {int, float}
                or (not math.isfinite(gap))
                or (gap < 0)
                or (type(reported) not in {int, float})
                or (type(verified) not in {int, float})
                or (not math.isclose(reported, verified, rel_tol=0, abs_tol=1e-06))
            ):
                invalid.append({"instance_id": key[0], "seed": key[1]})
            else:
                gaps.append(gap)
        complete = (
            not missing
            and (not failures)
            and (not invalid)
            and (len(gaps) == len(expected))
        )
        mean_gap = statistics.fmean(gaps) if complete else None
        return {
            "manifest_sha256": manifest_sha,
            "source_identity": identity,
            "expected_runs": len(expected),
            "retained_run_attempts": attempts,
            "observed_unique_runs": len(latest),
            "missing_runs": missing,
            "failed_runs": failures,
            "invalid_runs": invalid,
            "base_mean_normalized_gap": mean_gap,
            "positive_gap_runs": sum((gap > 0 for gap in gaps)) if complete else None,
            "conclusion": "verified_improvement_room"
            if mean_gap is not None and mean_gap > 0
            else "no_verified_gap"
            if complete
            else "incomplete",
            "claim_boundary": "This qualifies an open solver optimization challenge on a pinned baseline. It does not establish a human PR gain or Agent success.",
        }

    @staticmethod
    def analyze(
        manifest_path: Path,
        output_dir: Path,
        *,
        reference_kind: str = "solver_release",
        exclude_first_seed: bool = False,
    ) -> dict[str, Any]:
        if reference_kind not in {"solver_release", "human_pr"}:
            raise ValueError("unknown reference kind")
        from pitbench.evaluator.artifacts import selected_output_attempt

        output_dir = selected_output_attempt(output_dir)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("rounds") != 2:
            raise ValueError(
                "release-control analysis requires two opposite-order rounds"
            )
        identity = json.loads((output_dir / "identity.json").read_text())
        if (
            identity["manifest_sha256"]
            != hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        ):
            raise ValueError("manifest does not match retained run identity")
        oracle = yaml.safe_load(
            ReleaseControlAnalysis._source_path(manifest["oracle_path"]).read_text()
        )
        instance_ids = [item["id"] for item in oracle["instances"]]
        seeds = identity["selected_seeds"]
        if exclude_first_seed and len(seeds) < 2:
            raise ValueError("excluding the first seed requires at least two seeds")
        expected = {
            (round_index, instance_id, seed)
            for round_index in range(manifest["rounds"])
            for instance_id in instance_ids
            for seed in seeds
        }
        pairs, attempt_count = ReleaseControlAnalysis._read_pairs(
            output_dir / "pairs.jsonl"
        )
        missing = sorted(expected - pairs.keys())
        failures = [
            {"round": key[0], "instance_id": key[1], "seed": key[2], "row": pair}
            for key, pair in pairs.items()
            if any((pair[label]["status"] != "completed" for label in ("old", "new")))
        ]
        report: dict[str, Any] = {
            "reference_kind": reference_kind,
            "manifest_sha256": identity["manifest_sha256"],
            "source_identity": identity,
            "expected_pairs": len(expected),
            "retained_pair_attempts": attempt_count,
            "observed_unique_pairs": len(pairs),
            "missing_pairs": missing,
            "failed_pairs": failures,
            "method": {
                "primary_metric": "old minus new independently verified normalized gap",
                "unit": "independent instance, after paired-seed averaging",
                "bootstrap_seed": ReleaseControlAnalysis.BOOTSTRAP_SEED,
                "bootstrap_draws": ReleaseControlAnalysis.BOOTSTRAP_DRAWS,
                "rounds_separate": True,
            },
        }
        if missing or failures:
            report["conclusion"] = "incomplete"
            return report
        rounds: dict[str, Any] = {}
        time_only: dict[str, Any] = {}
        for round_index in range(manifest["rounds"]):
            effects: dict[str, float] = {}
            wall_effects: dict[str, float] = {}
            base_wall_by_instance: dict[str, float] = {}
            time_available = True
            order_effects: dict[str, dict[str, list[float]]] = defaultdict(
                lambda: defaultdict(list)
            )
            outcomes = Counter()
            objective_mismatch = 0
            for instance_id in instance_ids:
                deltas = []
                wall_deltas = []
                base_walls = []
                for seed in seeds:
                    pair = pairs[round_index, instance_id, seed]
                    old = pair["old"]
                    new = pair["new"]
                    delta = old["normalized_gap"] - new["normalized_gap"]
                    deltas.append(delta)
                    old_wall = old.get("solver_wall_time_sec")
                    new_wall = new.get("solver_wall_time_sec")
                    if (
                        type(old_wall) not in {int, float}
                        or type(new_wall) not in {int, float}
                        or (not math.isfinite(old_wall))
                        or (not math.isfinite(new_wall))
                    ):
                        if reference_kind == "human_pr":
                            raise ValueError("paired solver wall time is unavailable")
                        time_available = False
                    else:
                        base_walls.append(old_wall)
                        wall_deltas.append(old_wall - new_wall)
                    order_effects[pair["order"][0]][instance_id].append(delta)
                    outcomes[
                        "better"
                        if delta > 1e-12
                        else "worse"
                        if delta < -1e-12
                        else "equal"
                    ] += 1
                    for run in (old, new):
                        if (
                            abs(run["reported_objective"] - run["verified_objective"])
                            > 1e-06
                        ):
                            objective_mismatch += 1
                effects[instance_id] = statistics.fmean(deltas)
                if len(wall_deltas) == len(seeds):
                    wall_effects[instance_id] = statistics.fmean(wall_deltas)
                    base_wall_by_instance[instance_id] = statistics.fmean(base_walls)
            strata = {
                order: ReleaseControlAnalysis._distribution(
                    {
                        instance: statistics.fmean(values)
                        for instance, values in by_instance.items()
                    }
                )
                for order, by_instance in order_effects.items()
            }
            rounds[str(round_index)] = {
                "effect": ReleaseControlAnalysis._distribution(effects),
                "paired_run_outcomes": dict(outcomes),
                "reported_verified_objective_mismatches": objective_mismatch,
                "execution_order_strata": strata,
            }
            if time_available and len(wall_effects) == len(instance_ids):
                wall = ReleaseControlAnalysis._distribution(wall_effects)
                time_only[str(round_index)] = {
                    "paired_base_minus_candidate_wall_sec": wall,
                    "relative_wall_reduction_percent": 100
                    * wall["mean"]
                    / statistics.fmean(base_wall_by_instance.values()),
                    "classification": "faster"
                    if wall["ci95"]["lower"] > 0
                    else "slower"
                    if wall["ci95"]["upper"] < 0
                    else "inconclusive",
                }
            else:
                time_only[str(round_index)] = {
                    "classification": "unavailable",
                    "reason": "paired solver wall time is unavailable",
                }
        report["rounds"] = rounds
        report["time_only_by_round"] = time_only
        if exclude_first_seed:
            subset = seeds[1:]
            report["unused_public_seed_subset_by_round"] = {
                str(round_index): ReleaseControlAnalysis._distribution(
                    {
                        instance_id: statistics.fmean(
                            (
                                pairs[round_index, instance_id, seed]["old"][
                                    "normalized_gap"
                                ]
                                - pairs[round_index, instance_id, seed]["new"][
                                    "normalized_gap"
                                ]
                                for seed in subset
                            )
                        )
                        for instance_id in instance_ids
                    }
                )
                for round_index in range(manifest["rounds"])
            }
        null_controls: dict[str, Any] = {}
        for label in ("old", "new"):
            effects = {
                instance_id: statistics.fmean(
                    (
                        pairs[0, instance_id, seed][label]["normalized_gap"]
                        - pairs[1, instance_id, seed][label]["normalized_gap"]
                        for seed in seeds
                    )
                )
                for instance_id in instance_ids
            }
            null_controls[label] = ReleaseControlAnalysis._distribution(effects)
        report["same_version_round_drift"] = null_controls
        if "task_config" in manifest:
            task = PitBenchTask.from_yaml(
                ReleaseControlAnalysis._source_path(manifest["task_config"])
            )
            formal_reports = []
            for round_index in range(manifest["rounds"]):
                observations = [
                    RunObservation(
                        task_id=task.task_id,
                        code_state=CodeState.BASE
                        if label == "old"
                        else CodeState.AGENT,
                        instance_set="judge_id",
                        instance_set_kind="judge_id",
                        instance_id=instance_id,
                        solver_seed=seed,
                        budget_sec=manifest["budget_sec"],
                        status=RunStatus.COMPLETED,
                        valid=True,
                        objective=pairs[round_index, instance_id, seed][label][
                            "verified_objective"
                        ],
                        reported_objective=pairs[round_index, instance_id, seed][label][
                            "reported_objective"
                        ],
                        optimal_or_bks=pairs[round_index, instance_id, seed]["target"],
                        normalized_gap=pairs[round_index, instance_id, seed][label][
                            "normalized_gap"
                        ],
                        wall_time_sec=pairs[round_index, instance_id, seed][label].get(
                            "solver_wall_time_sec"
                        ),
                        cpu_time_sec=pairs[round_index, instance_id, seed][label].get(
                            "solver_cpu_time_sec"
                        ),
                        iterations=pairs[round_index, instance_id, seed][label].get(
                            "solver_iterations"
                        ),
                    )
                    for instance_id in instance_ids
                    for seed in seeds
                    for label in ("old", "new")
                ]
                formal = compute_task_performance_report(observations, task=task)
                if formal is None:
                    raise ValueError("formal Performance report is unavailable")
                formal_reports.append(formal)
            if formal_reports[0].classification in {
                PerformanceClassification.IMPROVED,
                PerformanceClassification.REGRESSED,
            }:
                confirmed_formal = adjudicate_performance_confirmation(
                    formal_reports[0],
                    formal_reports[1],
                    repeat_observation_count=2 * len(instance_ids) * len(seeds),
                    repeat_valid_observation_count=2 * len(instance_ids) * len(seeds),
                )
                final_classification = confirmed_formal.classification.value
            else:
                final_classification = formal_reports[0].classification.value
            report["pitbench_performance"] = {
                "round_classifications": [
                    item.classification.value for item in formal_reports
                ],
                "confirmed_classification": final_classification,
                "round_mean_gap_reductions": [
                    item.primary.paired.mean_gap_reduction for item in formal_reports
                ],
                "round_gap_ci95": [
                    item.primary.paired.mean_gap_reduction_ci95.model_dump()
                    if item.primary.paired.mean_gap_reduction_ci95 is not None
                    else None
                    for item in formal_reports
                ],
                "claim_boundary": "Reduced human-PR grid, not a score on the current task"
                if reference_kind == "human_pr"
                else "Reduced release-control grid, not an Agent patch score",
            }
        means = [rounds[str(index)]["effect"]["mean"] for index in range(2)]
        null_mean = max((abs(item["mean"]) for item in null_controls.values()))
        confirmed = (
            all(
                (
                    rounds[str(index)]["effect"]["mean"] >= 0.001
                    and rounds[str(index)]["effect"]["ci95"]["lower"] > 0
                    and (
                        rounds[str(index)]["effect"][
                            "mean_without_largest_absolute_instance"
                        ]
                        > 0
                    )
                    and (
                        rounds[str(index)]["reported_verified_objective_mismatches"]
                        == 0
                    )
                    for index in range(2)
                )
            )
            and min(means) > null_mean
        )
        report["version_effect_to_max_null_mean_ratio"] = (
            min(means) / null_mean if null_mean else None
        )
        report["conclusion"] = (
            "strong_positive_human_reference"
            if confirmed and reference_kind == "human_pr"
            else "human_reference_not_confirmed_under_frozen_rule"
            if reference_kind == "human_pr"
            else "strong_positive_control_for_metric_sensitivity"
            if confirmed
            else "release_effect_not_confirmed_under_frozen_rule"
        )
        report["claim_boundary"] = (
            "This one human PR is a reference patch on a historical source commit; it does not measure Agent ability or establish general solver effects."
            if reference_kind == "human_pr"
            else "A release control tests solver-metric sensitivity. It does not show that an ordinary Agent benchmark would rank Agent patches differently."
        )
        return report

    @staticmethod
    def _reference_digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def compare_agent_reference(
        human_path: Path,
        human_manifest_path: Path,
        agent_path: Path,
        agent_task_path: Path,
        *,
        budget_sec: float,
        instance_set: str,
    ) -> dict:
        human = json.loads(human_path.read_text())
        human_manifest = json.loads(human_manifest_path.read_text())
        agent_task = yaml.safe_load(agent_task_path.read_text())
        if human["manifest_sha256"] != ReleaseControlAnalysis._reference_digest(
            human_manifest_path
        ):
            raise ValueError("human analysis and manifest identity differ")
        if human_manifest["old"]["commit"] != agent_task["release"]["base_commit"]:
            raise ValueError("human and Agent base commits differ")
        if human_manifest["budget_sec"] != budget_sec:
            raise ValueError("human and Agent budget selections differ")
        if human_manifest["oracle_path"] != agent_task["oracle"]["source"]:
            raise ValueError("human and Agent oracle sources differ")
        seeds = human["source_identity"]["selected_seeds"]
        instances = set(human["rounds"]["0"]["effect"]["per_instance"])
        expected = {(instance, seed) for instance in instances for seed in seeds}
        observations = {}
        duplicates = []
        for line in agent_path.read_text().splitlines():
            if not line:
                continue
            row = json.loads(line)
            if row["task_id"] != agent_task["task_id"]:
                raise ValueError("Agent observation belongs to another task")
            if (
                row["instance_set"] != instance_set
                or row["budget_sec"] != budget_sec
                or row["instance_id"] not in instances
                or (row["solver_seed"] not in seeds)
            ):
                continue
            key = (row["instance_id"], row["solver_seed"], row["code_state"])
            if key in observations:
                duplicates.append(key)
            observations[key] = row
        missing = [
            (instance, seed, state)
            for instance, seed in sorted(expected)
            for state in ("base", "agent")
            if (instance, seed, state) not in observations
        ]
        failed = [
            {"key": key, "status": row["status"], "valid": row["valid"]}
            for key, row in sorted(observations.items())
            if row["status"] != "completed"
            or not row["valid"]
            or row["normalized_gap"] is None
        ]
        report = {
            "human_analysis_sha256": ReleaseControlAnalysis._reference_digest(
                human_path
            ),
            "human_manifest_sha256": ReleaseControlAnalysis._reference_digest(
                human_manifest_path
            ),
            "agent_observations_sha256": ReleaseControlAnalysis._reference_digest(
                agent_path
            ),
            "agent_task_sha256": ReleaseControlAnalysis._reference_digest(
                agent_task_path
            ),
            "base_commit": agent_task["release"]["base_commit"],
            "budget_sec": budget_sec,
            "instance_set": instance_set,
            "instance_count": len(instances),
            "selected_seeds": seeds,
            "expected_pairs": len(expected),
            "observed_pairs": sum(
                (
                    all(
                        (
                            (instance, seed, state) in observations
                            for state in ("base", "agent")
                        )
                    )
                    for instance, seed in expected
                )
            ),
            "duplicate_observation_keys": duplicates,
            "missing_observation_keys": missing,
            "failed_observations": failed,
            "claim_boundary": "Exploratory matched-instance and matched-seed comparison, not an official full-grid score or an Agent success-rate estimate.",
        }
        if missing or failed or duplicates:
            report["conclusion"] = "incomplete"
            return report
        effects = {
            instance: statistics.fmean(
                (
                    observations[instance, seed, "base"]["normalized_gap"]
                    - observations[instance, seed, "agent"]["normalized_gap"]
                    for seed in seeds
                )
            )
            for instance in sorted(instances)
        }
        report["agent_effect"] = ReleaseControlAnalysis._distribution(effects)
        report["human_effect_by_round"] = {
            round_id: human["rounds"][round_id]["effect"] for round_id in ("0", "1")
        }
        report["agent_minus_human_effect_by_round"] = {
            round_id: ReleaseControlAnalysis._distribution(
                {
                    instance: effects[instance]
                    - human["rounds"][round_id]["effect"]["per_instance"][instance]
                    for instance in sorted(instances)
                }
            )
            for round_id in ("0", "1")
        }
        report["conclusion"] = "complete_exploratory_comparison"
        return report


NOISE_ROOT = Path(__file__).resolve().parents[2]
NOISE_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _noise_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else NOISE_ROOT / path


def _noise_identity(directory: Path) -> dict:
    return json.loads((directory / "judge-checkpoint.json").read_text())["identity"]


def _noise_observations(directory: Path) -> list[RunObservation]:
    return ObservationStore.read(directory / "trials.parquet")


def _noise_expected_grid(directory: Path) -> ExpectedRunGrid:
    return ExpectedRunGrid.model_validate_json(
        (directory / "expected-run-grid.json").read_text()
    )


def _noise_grid_key(run: ExpectedRun | RunObservation) -> tuple:
    return (
        run.task_id,
        run.code_state,
        run.instance_set,
        run.instance_id,
        run.solver_seed,
        run.budget_sec,
        run.condition_id,
        run.equivalence_parent_id,
        run.test_suite,
    )


def _noise_primary_keys(
    observations: list[RunObservation], task: PitBenchTask
) -> set[tuple[str, int, float]]:
    return {
        (item.instance_id, item.solver_seed, item.budget_sec)
        for item in observations
        if item.code_state is CodeState.BASE
        and (item.instance_set_kind or item.instance_set) == "judge_id"
        and (item.test_suite is None)
        and (item.equivalence_parent_id is None)
        and (item.budget_sec in task.evaluation.budgets_sec)
    }


def _noise_score(
    task: PitBenchTask, observations: list[RunObservation], expected: ExpectedRunGrid
) -> dict:
    expected_keys = [_noise_grid_key(item) for item in expected.runs]
    observed_keys = [_noise_grid_key(item) for item in observations]
    if (
        len(expected_keys) != len(set(expected_keys))
        or len(observed_keys) != len(set(observed_keys))
        or set(expected_keys) != set(observed_keys)
    ):
        raise ValueError(f"incomplete or duplicate expected grid for {task.task_id}")
    report = compute_task_performance_report(
        observations, task=task, expected_run_grid=expected
    )
    if report is None:
        raise ValueError(f"no Performance report for {task.task_id}")
    primary = report.primary
    result = {
        "classification": report.classification.value,
        "primary_budget_sec": report.primary_budget_sec,
        "observation_count": len(observations),
        "status_counts": dict(
            sorted(Counter((x.status.value for x in observations)).items())
        ),
        "paired_diagnostics": _noise_paired_diagnostics(task, observations),
    }
    if hasattr(primary.paired, "mean_gap_reduction"):
        result.update(
            kind="heuristic",
            paired_gap_reduction=primary.paired.mean_gap_reduction,
            paired_gap_ci95=primary.paired.mean_gap_reduction_ci95.model_dump()
            if primary.paired.mean_gap_reduction_ci95
            else None,
        )
    else:
        result.update(
            kind="exact",
            base_solved=primary.base.solved_run_count,
            agent_solved=primary.agent.solved_run_count,
            expected_runs_per_state=primary.base.expected_run_count,
            paired_par2_reduction_sec=primary.paired.mean_penalized_runtime_reduction_sec,
            paired_par2_ci95=primary.paired.mean_penalized_runtime_reduction_ci95.model_dump()
            if primary.paired.mean_penalized_runtime_reduction_ci95
            else None,
        )
        coverage_interval = getattr(
            primary.paired, "mean_verified_solved_coverage_reduction_ci95", None
        )
        result["paired_coverage_ci95"] = (
            coverage_interval.model_dump() if coverage_interval else None
        )
    judge_seeds = {
        item.solver_seed
        for item in observations
        if (item.instance_set_kind or item.instance_set) == "judge_id"
        and item.test_suite is None
        and (item.equivalence_parent_id is None)
    }
    resources, _ = compute_resource_reports(
        observations,
        primary_budget_sec=task.evaluation.primary_budget_sec,
        expected_seed_count=len(judge_seeds),
        budgets_sec=task.evaluation.budgets_sec,
        expected_instance_counts={item.name: item.size for item in task.instance_sets},
    )
    primary_resources = resources.by_instance_set["judge_id"][
        f"{task.evaluation.primary_budget_sec:g}"
    ]
    result["primary_resource_ratios"] = {
        "memory": primary_resources.memory.model_dump(),
        "cpu": primary_resources.cpu.model_dump(),
    }
    return result


def _noise_paired_diagnostics(
    task: PitBenchTask, observations: list[RunObservation]
) -> dict:
    """Describe noise without changing the task's public Performance metric."""
    paired: dict[tuple[str, int], dict[CodeState, RunObservation]] = defaultdict(dict)
    for item in observations:
        if (
            (item.instance_set_kind or item.instance_set) == "judge_id"
            and item.test_suite is None
            and (item.equivalence_parent_id is None)
            and (item.budget_sec == task.evaluation.primary_budget_sec)
        ):
            paired[item.instance_id, item.solver_seed][item.code_state] = item
    wall_by_instance: dict[tuple[str, str], list[float]] = defaultdict(list)
    gap_by_instance: dict[tuple[str, str], list[float]] = defaultdict(list)
    coverage_by_instance: dict[tuple[str, str], list[float]] = defaultdict(list)
    changed_coverage_by_instance: dict[str, int] = defaultdict(int)
    changed_objective_by_instance: dict[str, int] = defaultdict(int)
    missing_wall = missing_coverage = 0
    for (instance_id, _), pair in sorted(paired.items()):
        if set(pair) != set(CodeState):
            raise ValueError(f"unpaired primary observation: {instance_id}")
        base, agent = (pair[CodeState.BASE], pair[CodeState.AGENT])
        if (
            base.valid
            and agent.valid
            and (base.objective is not None)
            and (agent.objective is not None)
            and (base.objective != agent.objective)
        ):
            changed_objective_by_instance[instance_id] += 1
        if base.wall_time_sec is None or agent.wall_time_sec is None:
            missing_wall += 1
        else:
            wall_by_instance["judge_id", instance_id].append(
                base.wall_time_sec - agent.wall_time_sec
            )
        if base.normalized_gap is not None and agent.normalized_gap is not None:
            gap_by_instance["judge_id", instance_id].append(
                base.normalized_gap - agent.normalized_gap
            )
        if task.evaluation.performance_protocol.value == "exact_verified_solve":
            base_outcome = compute_exact_run_outcome(
                base, time_basis=task.evaluation.exact_time_basis
            )
            agent_outcome = compute_exact_run_outcome(
                agent, time_basis=task.evaluation.exact_time_basis
            )
            if base_outcome.solved is None or agent_outcome.solved is None:
                missing_coverage += 1
            else:
                delta = int(agent_outcome.solved) - int(base_outcome.solved)
                coverage_by_instance["judge_id", instance_id].append(float(delta))
                changed_coverage_by_instance[instance_id] += delta
    wall_ci = _cluster_mean_ci(wall_by_instance, seed=20260824)
    coverage_ci = _cluster_mean_ci(coverage_by_instance, seed=20260824)
    gap_instance_means = [
        statistics.fmean(values) for values in gap_by_instance.values()
    ]
    coverage_instance_means = [
        statistics.fmean(values) for values in coverage_by_instance.values()
    ]
    return {
        "primary_paired_runs": len(paired),
        "primary_instances": len({instance for instance, _ in paired}),
        "mean_wall_time_reduction_sec": statistics.fmean(
            (statistics.fmean(v) for v in wall_by_instance.values())
        )
        if wall_by_instance
        else None,
        "wall_time_ci95": wall_ci.model_dump() if wall_ci else None,
        "missing_wall_pairs": missing_wall,
        "paired_gap_sign_flip_p_two_sided": _noise_sign_flip_p(gap_instance_means),
        "mean_verified_solved_coverage_reduction": statistics.fmean(
            (statistics.fmean(v) for v in coverage_by_instance.values())
        )
        if coverage_by_instance
        else None,
        "coverage_ci95": coverage_ci.model_dump() if coverage_ci else None,
        "coverage_sign_flip_p_two_sided": _noise_sign_flip_p(coverage_instance_means),
        "missing_coverage_pairs": missing_coverage,
        "coverage_changes_by_instance": {
            key: value
            for key, value in sorted(changed_coverage_by_instance.items())
            if value
        },
        "objective_changed_pairs": sum(changed_objective_by_instance.values()),
        "objective_changes_by_instance": dict(
            sorted(changed_objective_by_instance.items())
        ),
    }


def _noise_sign_flip_p(instance_means: list[float]) -> float | None:
    """Exploratory paired-instance label-swap randomization diagnostic."""
    if not instance_means:
        return None
    observed = abs(statistics.fmean(instance_means))
    if observed == 0:
        return 1.0
    rng = random.Random(20261015)
    resamples = 50000
    exceedances = sum(
        (
            abs(
                statistics.fmean(
                    (
                        (1 if rng.getrandbits(1) else -1) * value
                        for value in instance_means
                    )
                )
            )
            >= observed - 1e-15
            for _ in range(resamples)
        )
    )
    return (exceedances + 1) / (resamples + 1)


def calibrate_evaluation_noise(manifest: dict) -> dict:
    task_entries = {entry["task_id"]: entry for entry in manifest["task_configs"]}
    task_models = {
        task_id: PitBenchTask.from_yaml(_noise_path(entry["task_config"]))
        for task_id, entry in task_entries.items()
    }
    controls = []
    for task_id, directories in manifest.get("current_noop_runs", {}).items():
        entry = task_entries[task_id]
        task = task_models[task_id]
        for name in directories:
            directory = _noise_path(name)
            identity = _noise_identity(directory)
            if identity["candidate_patch_sha256"] != NOISE_EMPTY_SHA256:
                raise ValueError(f"not an empty patch: {directory}")
            if identity["task_config_sha256"] != entry["task_config_sha256"]:
                raise ValueError(f"task configuration changed: {directory}")
            if identity["judge_image"] != entry["judge_image"]:
                raise ValueError(f"judge image changed: {directory}")
            if set(identity["code_states"]) != {"base", "agent"}:
                raise ValueError(f"not a same-run Base/Agent control: {directory}")
            resources = manifest["primary_resource_stratum"]
            if (
                identity["cpus"] != resources["judge_cpus"]
                or identity["parallel_runs"] != resources["judge_parallel_runs"]
            ):
                raise ValueError(f"resource stratum changed: {directory}")
            observations = _noise_observations(directory)
            score = _noise_score(task, observations, _noise_expected_grid(directory))
            controls.append(
                {
                    "task_id": task_id,
                    "directory": name,
                    "identity": identity,
                    "harness_identity_recorded": bool(identity.get("harness_sha256")),
                    "false_direction": score["classification"]
                    in {"improved", "regressed"},
                    "public_classification": score["classification"],
                    "performance": score,
                }
            )
    for task_id, repeat_name in manifest.get("confirmation_runs", {}).items():
        matching = [item for item in controls if item["task_id"] == task_id]
        if len(matching) != 1:
            raise ValueError(
                f"confirmation needs exactly one primary control: {task_id}"
            )
        control = matching[0]
        first_dir = _noise_path(control["directory"])
        repeat_dir = _noise_path(repeat_name)
        first_identity = control["identity"]
        repeat_identity = _noise_identity(repeat_dir)
        for field in (
            "candidate_patch_sha256",
            "task_config_sha256",
            "judge_image",
            "harness_sha256",
            "cpus",
            "memory",
            "parallel_runs",
        ):
            if first_identity.get(field) != repeat_identity.get(field):
                raise ValueError(f"confirmation differs in {field}: {task_id}")
        if first_identity["code_states"] != list(
            reversed(repeat_identity["code_states"])
        ):
            raise ValueError(f"confirmation did not reverse order: {task_id}")
        first_grid = _noise_expected_grid(first_dir)
        repeat_grid = _noise_expected_grid(repeat_dir)
        if first_grid != repeat_grid:
            raise ValueError(f"confirmation uses a different grid: {task_id}")
        task = task_models[task_id]
        first_report = compute_task_performance_report(
            _noise_observations(first_dir), task=task, expected_run_grid=first_grid
        )
        repeat_observations = _noise_observations(repeat_dir)
        repeat_report = compute_task_performance_report(
            repeat_observations, task=task, expected_run_grid=repeat_grid
        )
        if first_report is None or repeat_report is None:
            raise ValueError(f"confirmation report missing: {task_id}")
        if first_report.classification not in {
            PerformanceClassification.IMPROVED,
            PerformanceClassification.REGRESSED,
        }:
            raise ValueError(
                f"non-directional control does not need confirmation: {task_id}"
            )
        final = adjudicate_performance_confirmation(
            first_report,
            repeat_report,
            repeat_observation_count=len(repeat_observations),
            repeat_valid_observation_count=sum(
                (item.valid for item in repeat_observations)
            ),
        )
        control["confirmation"] = {
            "directory": repeat_name,
            "identity": repeat_identity,
            "performance": _noise_score(task, repeat_observations, repeat_grid),
            "final": final.confirmation.model_dump(mode="json"),
        }
        control["public_classification"] = final.classification.value
    baseline_pairs = []
    for task_id, directories in manifest.get("historical_base_groups", {}).items():
        task = task_models[task_id]
        for left_name, right_name in itertools.combinations(directories, 2):
            left, right = (_noise_path(left_name), _noise_path(right_name))
            left_identity, right_identity = (
                _noise_identity(left),
                _noise_identity(right),
            )
            for field in ("task_config_sha256", "judge_image", "cpus", "parallel_runs"):
                if left_identity[field] != right_identity[field]:
                    raise ValueError(f"Base repeats differ in {field}: {left}, {right}")
            if (
                left_identity.get("harness_sha256")
                and right_identity.get("harness_sha256")
                and (
                    left_identity["harness_sha256"] != right_identity["harness_sha256"]
                )
            ):
                raise ValueError(
                    f"Base repeats use different harness code: {left}, {right}"
                )
            left_base = [
                item
                for item in _noise_observations(left)
                if item.code_state is CodeState.BASE
            ]
            right_base = [
                item
                for item in _noise_observations(right)
                if item.code_state is CodeState.BASE
            ]
            if _noise_primary_keys(left_base, task) != _noise_primary_keys(
                right_base, task
            ):
                raise ValueError(f"Base primary grids differ: {left}, {right}")
            pseudo_agent = [
                item.model_copy(update={"code_state": CodeState.AGENT})
                for item in right_base
            ]
            score = _noise_score(
                task, left_base + pseudo_agent, _noise_expected_grid(left)
            )
            baseline_pairs.append(
                {
                    "task_id": task_id,
                    "base_left": left_name,
                    "base_right": right_name,
                    "resource_stratum": {
                        "judge_cpus": left_identity["cpus"],
                        "judge_parallel_runs": left_identity["parallel_runs"],
                    },
                    "harness_identity_verified": bool(
                        left_identity.get("harness_sha256")
                        and right_identity.get("harness_sha256")
                    ),
                    "primary_grid_count": len(_noise_primary_keys(left_base, task)),
                    "false_direction": score["classification"]
                    in {"improved", "regressed"},
                    "performance": score,
                }
            )
    return {
        "method": "same-code paired Base/Agent full-grid noise calibration",
        "control_count": len(controls),
        "control_false_direction_count": sum((x["false_direction"] for x in controls)),
        "public_control_false_direction_count": sum(
            (x["public_classification"] in {"improved", "regressed"} for x in controls)
        ),
        "baseline_pair_count": len(baseline_pairs),
        "baseline_pair_false_direction_count": sum(
            (x["false_direction"] for x in baseline_pairs)
        ),
        "controls": controls,
        "historical_base_pairs": baseline_pairs,
    }
