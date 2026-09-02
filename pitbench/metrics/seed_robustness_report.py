from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from pitbench.schema.observation import CodeState, RunObservation, RunStatus
from pitbench.schema.task import InstanceSetKind

BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 20260824
CROSSED_BOOTSTRAP_METHOD = "crossed instance-set and seed-list bootstrap"


class SeedSelectionMetadata(BaseModel):
    seed_min: int
    seed_max: int
    seed_count: int = Field(gt=0)


class SeedRobustnessConfidenceInterval(BaseModel):
    lower: float
    upper: float
    level: float = 0.99
    method: str = CROSSED_BOOTSTRAP_METHOD
    resamples: int = BOOTSTRAP_RESAMPLES
    bootstrap_seed: int = BOOTSTRAP_SEED


class MeanSeedIqrEstimate(BaseModel):
    mean_seed_iqr: float | None = None
    mean_seed_iqr_ci99: SeedRobustnessConfidenceInterval | None = None


class MeanSeedIqrChangeEstimate(BaseModel):
    mean_seed_iqr_change: float | None = None
    mean_seed_iqr_change_ci99: SeedRobustnessConfidenceInterval | None = None


class SeedRobustnessBudget(BaseModel):
    budget_sec: float = Field(gt=0)
    instance_count: int = Field(ge=0)
    base_complete_instance_count: int = Field(ge=0)
    agent_complete_instance_count: int = Field(ge=0)
    paired_complete_instance_count: int = Field(ge=0)
    base: MeanSeedIqrEstimate
    agent: MeanSeedIqrEstimate
    change: MeanSeedIqrChangeEstimate


class InstanceSetSeedRobustness(BaseModel):
    instance_set_kind: InstanceSetKind
    primary: SeedRobustnessBudget
    by_budget: dict[str, SeedRobustnessBudget]


class SeedRobustnessReport(BaseModel):
    task_id: str
    metric: Literal["seed_robustness"] = "seed_robustness"
    primary_budget_sec: float = Field(gt=0)
    budgets_sec: list[float]
    seed_selection: SeedSelectionMetadata
    by_instance_set: dict[str, InstanceSetSeedRobustness]


class SeedResult(BaseModel):
    seed: int
    status: RunStatus
    valid: bool
    normalized_gap: float | None = None


class SeedEcdfPoint(BaseModel):
    normalized_gap: float
    fraction_at_or_below: float = Field(ge=0, le=1)


class CodeStateSeedDetails(BaseModel):
    complete: bool
    seed_iqr: float | None = None
    seed_results: list[SeedResult]
    sorted_valid_gaps: list[float]
    ecdf: list[SeedEcdfPoint]


class InstanceSeedDetails(BaseModel):
    instance_id: str
    base: CodeStateSeedDetails
    agent: CodeStateSeedDetails


class SeedBudgetDetails(BaseModel):
    budget_sec: float = Field(gt=0)
    instances: list[InstanceSeedDetails]


class InstanceSetSeedDetails(BaseModel):
    instance_set_kind: InstanceSetKind
    by_budget: dict[str, SeedBudgetDetails]


class SeedRobustnessDetails(BaseModel):
    task_id: str
    metric: Literal["seed_robustness"] = "seed_robustness"
    development_seeds: list[int]
    evaluation_seeds: list[int]
    by_instance_set: dict[str, InstanceSetSeedDetails]


def _type_7_quantile(seed_gaps: Sequence[float], probability: float) -> float:
    ordered_seed_gaps = sorted(seed_gaps)
    if not ordered_seed_gaps:
        raise ValueError("cannot compute a quantile without seed gaps")
    position = probability * (len(ordered_seed_gaps) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered_seed_gaps[lower_index]
    upper_weight = position - lower_index
    return (
        ordered_seed_gaps[lower_index] * (1 - upper_weight)
        + ordered_seed_gaps[upper_index] * upper_weight
    )


def seed_iqr(seed_gaps: Sequence[float]) -> float:
    return _type_7_quantile(seed_gaps, 0.75) - _type_7_quantile(seed_gaps, 0.25)


def _confidence_interval(
    bootstrap_estimates: Sequence[float],
) -> SeedRobustnessConfidenceInterval:
    return SeedRobustnessConfidenceInterval(
        lower=_type_7_quantile(bootstrap_estimates, 0.005),
        upper=_type_7_quantile(bootstrap_estimates, 0.995),
    )


def _validate_declared_inputs(
    observations: Sequence[RunObservation],
    *,
    task_id: str,
    budgets_sec: Sequence[float],
    primary_budget_sec: float,
    seed_selection: SeedSelectionMetadata,
    development_seeds: Sequence[int],
    evaluation_seeds: Sequence[int],
) -> tuple[list[float], tuple[int, ...], tuple[int, ...]]:
    declared_budgets = list(budgets_sec)
    if not declared_budgets or any(budget <= 0 for budget in declared_budgets):
        raise ValueError("budgets_sec must contain positive budgets")
    if len(set(declared_budgets)) != len(declared_budgets):
        raise ValueError("budgets_sec must not contain duplicates")
    if primary_budget_sec not in declared_budgets:
        raise ValueError("primary_budget_sec must belong to budgets_sec")
    if seed_selection.seed_min > seed_selection.seed_max:
        raise ValueError("seed_min must not exceed seed_max")

    development_seed_list = tuple(development_seeds)
    evaluation_seed_list = tuple(evaluation_seeds)
    for seed_list_name, seed_list in (
        ("development_seeds", development_seed_list),
        ("evaluation_seeds", evaluation_seed_list),
    ):
        if len(seed_list) != seed_selection.seed_count:
            raise ValueError(
                f"{seed_list_name} must contain seed_count seed identifiers"
            )
        if len(set(seed_list)) != len(seed_list):
            raise ValueError(f"{seed_list_name} must not contain duplicates")
        if any(
            seed < seed_selection.seed_min or seed > seed_selection.seed_max
            for seed in seed_list
        ):
            raise ValueError(f"{seed_list_name} contains a seed outside the domain")
    if set(development_seed_list) & set(evaluation_seed_list):
        raise ValueError("development_seeds and evaluation_seeds must be disjoint")

    declared_budget_set = set(declared_budgets)
    for observation in observations:
        if observation.task_id != task_id:
            raise ValueError("all observations must match task_id")
        if observation.budget_sec not in declared_budget_set:
            raise ValueError("observation uses a budget not declared in budgets_sec")

    return declared_budgets, development_seed_list, evaluation_seed_list


def _group_observations(
    observations: Sequence[RunObservation],
    *,
    development_seeds: tuple[int, ...],
    evaluation_seeds: tuple[int, ...],
) -> tuple[
    dict[str, InstanceSetKind],
    dict[tuple[str, str, float, CodeState, int], RunObservation],
]:
    instance_set_kinds: dict[str, InstanceSetKind] = {}
    observations_by_run: dict[
        tuple[str, str, float, CodeState, int], RunObservation
    ] = {}

    for observation in observations:
        try:
            instance_set_kind = InstanceSetKind(observation.instance_set_kind)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "every observation requires a known instance_set_kind"
            ) from error

        previous_kind = instance_set_kinds.setdefault(
            observation.instance_set, instance_set_kind
        )
        if previous_kind is not instance_set_kind:
            raise ValueError("an instance_set cannot have multiple kinds")

        expected_seeds = (
            development_seeds
            if instance_set_kind is InstanceSetKind.AGENT_DEV
            else evaluation_seeds
        )
        if observation.solver_seed not in expected_seeds:
            raise ValueError("observation uses a seed not assigned to its instance set")

        run_key = (
            observation.instance_set,
            observation.instance_id,
            observation.budget_sec,
            observation.code_state,
            observation.solver_seed,
        )
        if run_key in observations_by_run:
            raise ValueError("duplicate observation for an evaluation run")
        observations_by_run[run_key] = observation

    return instance_set_kinds, observations_by_run


def _complete_seed_gaps(
    observations_by_run: dict[
        tuple[str, str, float, CodeState, int], RunObservation
    ],
    *,
    instance_set: str,
    instance_id: str,
    budget_sec: float,
    code_state: CodeState,
    expected_seeds: tuple[int, ...],
) -> list[float] | None:
    seed_gaps: list[float] = []
    for solver_seed in expected_seeds:
        observation = observations_by_run.get(
            (instance_set, instance_id, budget_sec, code_state, solver_seed)
        )
        if (
            observation is None
            or not observation.valid
            or observation.normalized_gap is None
        ):
            return None
        seed_gaps.append(observation.normalized_gap)
    return seed_gaps


def _crossed_bootstrap_intervals(
    paired_seed_gaps: Sequence[tuple[list[float], list[float]]],
) -> tuple[
    SeedRobustnessConfidenceInterval,
    SeedRobustnessConfidenceInterval,
    SeedRobustnessConfidenceInterval,
]:
    instance_count = len(paired_seed_gaps)
    seed_count = len(paired_seed_gaps[0][0])
    random_generator = random.Random(BOOTSTRAP_SEED)
    base_bootstrap_means: list[float] = []
    agent_bootstrap_means: list[float] = []
    change_bootstrap_means: list[float] = []

    for _ in range(BOOTSTRAP_RESAMPLES):
        sampled_seed_indices = [
            math.floor(seed_count * random_generator.random())
            for _ in range(seed_count)
        ]
        sampled_instance_indices = [
            math.floor(instance_count * random_generator.random())
            for _ in range(instance_count)
        ]

        sampled_base_iqrs: list[float] = []
        sampled_agent_iqrs: list[float] = []
        for instance_index in sampled_instance_indices:
            base_seed_gaps, agent_seed_gaps = paired_seed_gaps[instance_index]
            sampled_base_iqrs.append(
                seed_iqr(
                    [base_seed_gaps[seed_index] for seed_index in sampled_seed_indices]
                )
            )
            sampled_agent_iqrs.append(
                seed_iqr(
                    [agent_seed_gaps[seed_index] for seed_index in sampled_seed_indices]
                )
            )

        base_mean = statistics.fmean(sampled_base_iqrs)
        agent_mean = statistics.fmean(sampled_agent_iqrs)
        base_bootstrap_means.append(base_mean)
        agent_bootstrap_means.append(agent_mean)
        change_bootstrap_means.append(agent_mean - base_mean)

    return (
        _confidence_interval(base_bootstrap_means),
        _confidence_interval(agent_bootstrap_means),
        _confidence_interval(change_bootstrap_means),
    )


def _budget_seed_robustness(
    observations_by_run: dict[
        tuple[str, str, float, CodeState, int], RunObservation
    ],
    *,
    instance_set: str,
    budget_sec: float,
    expected_seeds: tuple[int, ...],
) -> SeedRobustnessBudget:
    instance_ids = sorted(
        {
            instance_id
            for (
                observed_instance_set,
                instance_id,
                observed_budget,
                _code_state,
                _solver_seed,
            ) in observations_by_run
            if observed_instance_set == instance_set and observed_budget == budget_sec
        }
    )

    base_complete_seed_gaps: dict[str, list[float]] = {}
    agent_complete_seed_gaps: dict[str, list[float]] = {}
    for instance_id in instance_ids:
        base_seed_gaps = _complete_seed_gaps(
            observations_by_run,
            instance_set=instance_set,
            instance_id=instance_id,
            budget_sec=budget_sec,
            code_state=CodeState.BASE,
            expected_seeds=expected_seeds,
        )
        if base_seed_gaps is not None:
            base_complete_seed_gaps[instance_id] = base_seed_gaps

        agent_seed_gaps = _complete_seed_gaps(
            observations_by_run,
            instance_set=instance_set,
            instance_id=instance_id,
            budget_sec=budget_sec,
            code_state=CodeState.AGENT,
            expected_seeds=expected_seeds,
        )
        if agent_seed_gaps is not None:
            agent_complete_seed_gaps[instance_id] = agent_seed_gaps

    paired_instance_ids = sorted(
        set(base_complete_seed_gaps) & set(agent_complete_seed_gaps)
    )
    paired_seed_gaps = [
        (
            base_complete_seed_gaps[instance_id],
            agent_complete_seed_gaps[instance_id],
        )
        for instance_id in paired_instance_ids
    ]
    base_mean_seed_iqr: float | None = None
    agent_mean_seed_iqr: float | None = None
    mean_seed_iqr_change: float | None = None
    base_interval: SeedRobustnessConfidenceInterval | None = None
    agent_interval: SeedRobustnessConfidenceInterval | None = None
    change_interval: SeedRobustnessConfidenceInterval | None = None

    if paired_seed_gaps:
        base_instance_iqrs = [
            seed_iqr(base_seed_gaps)
            for base_seed_gaps, _agent_seed_gaps in paired_seed_gaps
        ]
        agent_instance_iqrs = [
            seed_iqr(agent_seed_gaps)
            for _base_seed_gaps, agent_seed_gaps in paired_seed_gaps
        ]
        base_mean_seed_iqr = statistics.fmean(base_instance_iqrs)
        agent_mean_seed_iqr = statistics.fmean(agent_instance_iqrs)
        mean_seed_iqr_change = agent_mean_seed_iqr - base_mean_seed_iqr
        if len(paired_seed_gaps) >= 2:
            base_interval, agent_interval, change_interval = (
                _crossed_bootstrap_intervals(paired_seed_gaps)
            )

    return SeedRobustnessBudget(
        budget_sec=budget_sec,
        instance_count=len(instance_ids),
        base_complete_instance_count=len(base_complete_seed_gaps),
        agent_complete_instance_count=len(agent_complete_seed_gaps),
        paired_complete_instance_count=len(paired_seed_gaps),
        base=MeanSeedIqrEstimate(
            mean_seed_iqr=base_mean_seed_iqr,
            mean_seed_iqr_ci99=base_interval,
        ),
        agent=MeanSeedIqrEstimate(
            mean_seed_iqr=agent_mean_seed_iqr,
            mean_seed_iqr_ci99=agent_interval,
        ),
        change=MeanSeedIqrChangeEstimate(
            mean_seed_iqr_change=mean_seed_iqr_change,
            mean_seed_iqr_change_ci99=change_interval,
        ),
    )


def compute_seed_robustness_report(
    observations: Sequence[RunObservation],
    *,
    task_id: str,
    budgets_sec: Sequence[float],
    primary_budget_sec: float,
    seed_selection: SeedSelectionMetadata,
    development_seeds: Sequence[int],
    evaluation_seeds: Sequence[int],
) -> SeedRobustnessReport:
    (
        declared_budgets,
        development_seed_list,
        evaluation_seed_list,
    ) = _validate_declared_inputs(
        observations,
        task_id=task_id,
        budgets_sec=budgets_sec,
        primary_budget_sec=primary_budget_sec,
        seed_selection=seed_selection,
        development_seeds=development_seeds,
        evaluation_seeds=evaluation_seeds,
    )
    instance_set_kinds, observations_by_run = _group_observations(
        observations,
        development_seeds=development_seed_list,
        evaluation_seeds=evaluation_seed_list,
    )

    by_instance_set: dict[str, InstanceSetSeedRobustness] = {}
    for instance_set in sorted(instance_set_kinds):
        instance_set_kind = instance_set_kinds[instance_set]
        expected_seeds = (
            development_seed_list
            if instance_set_kind is InstanceSetKind.AGENT_DEV
            else evaluation_seed_list
        )
        by_budget = {
            f"{budget_sec:g}": _budget_seed_robustness(
                observations_by_run,
                instance_set=instance_set,
                budget_sec=budget_sec,
                expected_seeds=expected_seeds,
            )
            for budget_sec in declared_budgets
        }
        by_instance_set[instance_set] = InstanceSetSeedRobustness(
            instance_set_kind=instance_set_kind,
            primary=by_budget[f"{primary_budget_sec:g}"],
            by_budget=by_budget,
        )

    return SeedRobustnessReport(
        task_id=task_id,
        primary_budget_sec=primary_budget_sec,
        budgets_sec=declared_budgets,
        seed_selection=seed_selection,
        by_instance_set=by_instance_set,
    )


def _code_state_seed_details(
    observations_by_run: dict[
        tuple[str, str, float, CodeState, int], RunObservation
    ],
    *,
    instance_set: str,
    instance_id: str,
    budget_sec: float,
    code_state: CodeState,
    expected_seeds: tuple[int, ...],
) -> CodeStateSeedDetails:
    seed_results: list[SeedResult] = []
    for seed in expected_seeds:
        observation = observations_by_run.get(
            (instance_set, instance_id, budget_sec, code_state, seed)
        )
        if observation is None:
            continue
        seed_results.append(
            SeedResult(
                seed=seed,
                status=observation.status,
                valid=observation.valid,
                normalized_gap=observation.normalized_gap,
            )
        )

    complete_seed_gaps = _complete_seed_gaps(
        observations_by_run,
        instance_set=instance_set,
        instance_id=instance_id,
        budget_sec=budget_sec,
        code_state=code_state,
        expected_seeds=expected_seeds,
    )
    sorted_valid_gaps = sorted(
        result.normalized_gap
        for result in seed_results
        if result.valid and result.normalized_gap is not None
    )
    ecdf = [
        SeedEcdfPoint(
            normalized_gap=gap,
            fraction_at_or_below=(
                sum(value <= gap for value in sorted_valid_gaps)
                / len(sorted_valid_gaps)
            ),
        )
        for gap in sorted(set(sorted_valid_gaps))
    ]
    return CodeStateSeedDetails(
        complete=complete_seed_gaps is not None,
        seed_iqr=(
            seed_iqr(complete_seed_gaps)
            if complete_seed_gaps is not None
            else None
        ),
        seed_results=seed_results,
        sorted_valid_gaps=sorted_valid_gaps,
        ecdf=ecdf,
    )


def compute_seed_robustness_details(
    observations: Sequence[RunObservation],
    *,
    task_id: str,
    budgets_sec: Sequence[float],
    primary_budget_sec: float,
    seed_selection: SeedSelectionMetadata,
    development_seeds: Sequence[int],
    evaluation_seeds: Sequence[int],
) -> SeedRobustnessDetails:
    (
        declared_budgets,
        development_seed_list,
        evaluation_seed_list,
    ) = _validate_declared_inputs(
        observations,
        task_id=task_id,
        budgets_sec=budgets_sec,
        primary_budget_sec=primary_budget_sec,
        seed_selection=seed_selection,
        development_seeds=development_seeds,
        evaluation_seeds=evaluation_seeds,
    )
    instance_set_kinds, observations_by_run = _group_observations(
        observations,
        development_seeds=development_seed_list,
        evaluation_seeds=evaluation_seed_list,
    )

    by_instance_set: dict[str, InstanceSetSeedDetails] = {}
    for instance_set in sorted(instance_set_kinds):
        instance_set_kind = instance_set_kinds[instance_set]
        expected_seeds = (
            development_seed_list
            if instance_set_kind is InstanceSetKind.AGENT_DEV
            else evaluation_seed_list
        )
        instance_ids = sorted(
            {
                instance_id
                for observed_instance_set, instance_id, *_rest in observations_by_run
                if observed_instance_set == instance_set
            }
        )
        by_budget: dict[str, SeedBudgetDetails] = {}
        for budget_sec in declared_budgets:
            instances = [
                InstanceSeedDetails(
                    instance_id=instance_id,
                    base=_code_state_seed_details(
                        observations_by_run,
                        instance_set=instance_set,
                        instance_id=instance_id,
                        budget_sec=budget_sec,
                        code_state=CodeState.BASE,
                        expected_seeds=expected_seeds,
                    ),
                    agent=_code_state_seed_details(
                        observations_by_run,
                        instance_set=instance_set,
                        instance_id=instance_id,
                        budget_sec=budget_sec,
                        code_state=CodeState.AGENT,
                        expected_seeds=expected_seeds,
                    ),
                )
                for instance_id in instance_ids
            ]
            by_budget[f"{budget_sec:g}"] = SeedBudgetDetails(
                budget_sec=budget_sec,
                instances=instances,
            )
        by_instance_set[instance_set] = InstanceSetSeedDetails(
            instance_set_kind=instance_set_kind,
            by_budget=by_budget,
        )

    return SeedRobustnessDetails(
        task_id=task_id,
        development_seeds=list(development_seed_list),
        evaluation_seeds=list(evaluation_seed_list),
        by_instance_set=by_instance_set,
    )


def _format_proportion(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.3f}%"


def format_seed_robustness_report(report: SeedRobustnessReport) -> str:
    lines = [
        "Seed robustness",
        f"Primary budget: {report.primary_budget_sec:g}s",
        "",
        "Instance set | Base mean IQR | Agent mean IQR | "
        "Agent-Base IQR change (99% CI) | Paired complete instances",
        "--- | ---: | ---: | ---: | ---:",
    ]
    for instance_set, instance_set_report in report.by_instance_set.items():
        primary = instance_set_report.primary
        change = _format_proportion(primary.change.mean_seed_iqr_change)
        if primary.change.mean_seed_iqr_change_ci99 is not None:
            interval = primary.change.mean_seed_iqr_change_ci99
            change = (
                f"{change} [{_format_proportion(interval.lower)}, "
                f"{_format_proportion(interval.upper)}]"
            )
        lines.append(
            f"{instance_set} | {_format_proportion(primary.base.mean_seed_iqr)} | "
            f"{_format_proportion(primary.agent.mean_seed_iqr)} | {change} | "
            f"{primary.paired_complete_instance_count}"
        )
    return "\n".join(lines)


__all__ = [
    "CROSSED_BOOTSTRAP_METHOD",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CodeStateSeedDetails",
    "InstanceSetSeedRobustness",
    "MeanSeedIqrChangeEstimate",
    "MeanSeedIqrEstimate",
    "SeedRobustnessBudget",
    "SeedRobustnessConfidenceInterval",
    "SeedRobustnessReport",
    "SeedRobustnessDetails",
    "compute_seed_robustness_details",
    "SeedSelectionMetadata",
    "compute_seed_robustness_report",
    "format_seed_robustness_report",
    "seed_iqr",
]
