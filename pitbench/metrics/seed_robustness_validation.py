from __future__ import annotations

import random
import statistics
from collections.abc import Sequence

from pydantic import BaseModel, Field

from pitbench.metrics.seed_robustness_report import (
    SeedRobustnessConfidenceInterval,
    SeedSelectionMetadata,
    compute_seed_robustness_report,
    seed_iqr,
)
from pitbench.schema.observation import CodeState, RunObservation

VALIDATION_GENERATION_SEED = 20260902
REFERENCE_SEED_COUNT = 700
TEST_SEED_COUNT = 300
TEST_LIST_COUNT = 1000
SEEDS_PER_TEST_LIST = 30


class RealValidationSeeds(BaseModel):
    generation_seed: int
    seed_min: int
    seed_max: int
    reference_seeds: list[int]
    test_seeds: list[int]
    test_seed_lists: list[list[int]]


class RealValidationEstimate(BaseModel):
    reference_value: float
    test_list_count: int = Field(ge=0)
    estimate_count: int = Field(ge=0)
    interval_count: int = Field(ge=0)
    mean_estimate: float | None = None
    bias: float | None = None
    empirical_coverage: float | None = None
    mean_interval_width: float | None = None


class RealValidationBudget(BaseModel):
    budget_sec: float = Field(gt=0)
    base: RealValidationEstimate
    agent: RealValidationEstimate
    change: RealValidationEstimate


class RealSeedValidationSummary(BaseModel):
    task_id: str
    instance_set: str
    instance_count: int = Field(gt=0)
    budgets_sec: list[float]
    generation_seed: int
    reference_seed_count: int = Field(gt=0)
    test_seed_count: int = Field(gt=0)
    test_list_count: int = Field(gt=0)
    seeds_per_test_list: int = Field(gt=0)
    by_budget: dict[str, RealValidationBudget]


def generate_real_validation_seeds(
    *,
    seed_min: int,
    seed_max: int,
    reference_seed_count: int = REFERENCE_SEED_COUNT,
    test_seed_count: int = TEST_SEED_COUNT,
    test_list_count: int = TEST_LIST_COUNT,
    seeds_per_test_list: int = SEEDS_PER_TEST_LIST,
    generation_seed: int = VALIDATION_GENERATION_SEED,
) -> RealValidationSeeds:
    if seed_min > seed_max:
        raise ValueError("seed_min must not exceed seed_max")
    if reference_seed_count < seeds_per_test_list:
        raise ValueError("reference_seed_count must fit one test seed list")
    if test_seed_count < seeds_per_test_list:
        raise ValueError("test_seed_count must fit one test seed list")
    if test_list_count < 1 or seeds_per_test_list < 1:
        raise ValueError("test list counts must be positive")
    available_seed_count = seed_max - seed_min + 1
    selected_seed_count = reference_seed_count + test_seed_count
    if selected_seed_count > available_seed_count:
        raise ValueError("seed range does not contain enough distinct seeds")

    random_generator = random.Random(generation_seed)
    selected_seeds = random_generator.sample(
        range(seed_min, seed_max + 1), selected_seed_count
    )
    reference_seeds = selected_seeds[:reference_seed_count]
    test_seeds = selected_seeds[reference_seed_count:]
    test_seed_lists = [
        random_generator.sample(test_seeds, seeds_per_test_list)
        for _ in range(test_list_count)
    ]
    return RealValidationSeeds(
        generation_seed=generation_seed,
        seed_min=seed_min,
        seed_max=seed_max,
        reference_seeds=reference_seeds,
        test_seeds=test_seeds,
        test_seed_lists=test_seed_lists,
    )


def _observations_by_run(
    observations: Sequence[RunObservation],
    *,
    task_id: str,
) -> tuple[
    str,
    list[str],
    dict[tuple[CodeState, str, float, int], RunObservation],
]:
    instance_sets = {observation.instance_set for observation in observations}
    if len(instance_sets) != 1:
        raise ValueError("real validation requires exactly one instance set")
    if any(observation.task_id != task_id for observation in observations):
        raise ValueError("all observations must match task_id")
    if any(observation.instance_set_kind != "agent_dev" for observation in observations):
        raise ValueError("real validation requires agent_dev observations")

    observations_by_run: dict[
        tuple[CodeState, str, float, int], RunObservation
    ] = {}
    for observation in observations:
        run_key = (
            observation.code_state,
            observation.instance_id,
            observation.budget_sec,
            observation.solver_seed,
        )
        if run_key in observations_by_run:
            raise ValueError("duplicate real validation observation")
        observations_by_run[run_key] = observation

    return (
        next(iter(instance_sets)),
        sorted({observation.instance_id for observation in observations}),
        observations_by_run,
    )


def _complete_gaps(
    observations_by_run: dict[
        tuple[CodeState, str, float, int], RunObservation
    ],
    *,
    code_state: CodeState,
    instance_id: str,
    budget_sec: float,
    seeds: Sequence[int],
) -> list[float] | None:
    gaps: list[float] = []
    for seed in seeds:
        observation = observations_by_run.get(
            (code_state, instance_id, budget_sec, seed)
        )
        if (
            observation is None
            or not observation.valid
            or observation.normalized_gap is None
        ):
            return None
        gaps.append(observation.normalized_gap)
    return gaps


def _reference_value(
    observations_by_run: dict[
        tuple[CodeState, str, float, int], RunObservation
    ],
    *,
    code_state: CodeState,
    instance_ids: Sequence[str],
    budget_sec: float,
    reference_seeds: Sequence[int],
) -> float:
    instance_iqrs = []
    for instance_id in instance_ids:
        gaps = _complete_gaps(
            observations_by_run,
            code_state=code_state,
            instance_id=instance_id,
            budget_sec=budget_sec,
            seeds=reference_seeds,
        )
        if gaps is None:
            raise ValueError("reference seed results are incomplete")
        instance_iqrs.append(seed_iqr(gaps))
    return statistics.fmean(instance_iqrs)


def _estimate_summary(
    reference_value: float,
    estimates: Sequence[float],
    intervals: Sequence[SeedRobustnessConfidenceInterval],
    *,
    test_list_count: int,
) -> RealValidationEstimate:
    mean_estimate = statistics.fmean(estimates) if estimates else None
    empirical_coverage = (
        statistics.fmean(
            interval.lower <= reference_value <= interval.upper
            for interval in intervals
        )
        if intervals
        else None
    )
    mean_interval_width = (
        statistics.fmean(interval.upper - interval.lower for interval in intervals)
        if intervals
        else None
    )
    return RealValidationEstimate(
        reference_value=reference_value,
        test_list_count=test_list_count,
        estimate_count=len(estimates),
        interval_count=len(intervals),
        mean_estimate=mean_estimate,
        bias=(mean_estimate - reference_value if mean_estimate is not None else None),
        empirical_coverage=empirical_coverage,
        mean_interval_width=mean_interval_width,
    )


def analyze_real_seed_validation(
    observations: Sequence[RunObservation],
    *,
    task_id: str,
    budgets_sec: Sequence[float],
    validation_seeds: RealValidationSeeds,
) -> RealSeedValidationSummary:
    if not observations:
        raise ValueError("real validation requires observations")
    if len(validation_seeds.reference_seeds) < SEEDS_PER_TEST_LIST:
        raise ValueError("reference seeds must fit the unused evaluation seed list")
    if set(validation_seeds.reference_seeds) & set(validation_seeds.test_seeds):
        raise ValueError("reference_seeds and test_seeds must be disjoint")
    test_seed_set = set(validation_seeds.test_seeds)
    if any(
        len(test_seed_list) != SEEDS_PER_TEST_LIST
        or len(set(test_seed_list)) != SEEDS_PER_TEST_LIST
        or not set(test_seed_list) <= test_seed_set
        for test_seed_list in validation_seeds.test_seed_lists
    ):
        raise ValueError("every test seed list must contain 30 unique test seeds")
    declared_budgets = list(budgets_sec)
    if not declared_budgets or len(set(declared_budgets)) != len(declared_budgets):
        raise ValueError("budgets_sec must be non-empty and unique")

    instance_set, instance_ids, observations_by_run = _observations_by_run(
        observations,
        task_id=task_id,
    )
    estimates_by_budget = {
        budget_sec: {"base": [], "agent": [], "change": []}
        for budget_sec in declared_budgets
    }
    intervals_by_budget = {
        budget_sec: {"base": [], "agent": [], "change": []}
        for budget_sec in declared_budgets
    }
    reference_seeds_for_unused_judge_sets = validation_seeds.reference_seeds[
        :SEEDS_PER_TEST_LIST
    ]
    for test_seed_list in validation_seeds.test_seed_lists:
        selected_seeds = set(test_seed_list)
        test_observations = [
            observation
            for observation in observations
            if observation.solver_seed in selected_seeds
        ]
        report = compute_seed_robustness_report(
            test_observations,
            task_id=task_id,
            budgets_sec=declared_budgets,
            primary_budget_sec=declared_budgets[0],
            seed_selection=SeedSelectionMetadata(
                seed_min=validation_seeds.seed_min,
                seed_max=validation_seeds.seed_max,
                seed_count=SEEDS_PER_TEST_LIST,
            ),
            development_seeds=test_seed_list,
            evaluation_seeds=reference_seeds_for_unused_judge_sets,
        )
        instance_set_report = report.by_instance_set[instance_set]
        for budget_sec in declared_budgets:
            budget_report = instance_set_report.by_budget[f"{budget_sec:g}"]
            values = {
                "base": budget_report.base.mean_seed_iqr,
                "agent": budget_report.agent.mean_seed_iqr,
                "change": budget_report.change.mean_seed_iqr_change,
            }
            intervals = {
                "base": budget_report.base.mean_seed_iqr_ci99,
                "agent": budget_report.agent.mean_seed_iqr_ci99,
                "change": budget_report.change.mean_seed_iqr_change_ci99,
            }
            for estimate_name, value in values.items():
                if value is not None:
                    estimates_by_budget[budget_sec][estimate_name].append(value)
                interval = intervals[estimate_name]
                if interval is not None:
                    intervals_by_budget[budget_sec][estimate_name].append(interval)

    by_budget: dict[str, RealValidationBudget] = {}
    for budget_sec in declared_budgets:
        base_reference = _reference_value(
            observations_by_run,
            code_state=CodeState.BASE,
            instance_ids=instance_ids,
            budget_sec=budget_sec,
            reference_seeds=validation_seeds.reference_seeds,
        )
        agent_reference = _reference_value(
            observations_by_run,
            code_state=CodeState.AGENT,
            instance_ids=instance_ids,
            budget_sec=budget_sec,
            reference_seeds=validation_seeds.reference_seeds,
        )
        change_reference = agent_reference - base_reference
        summary_inputs = {
            "test_list_count": len(validation_seeds.test_seed_lists),
        }
        by_budget[f"{budget_sec:g}"] = RealValidationBudget(
            budget_sec=budget_sec,
            base=_estimate_summary(
                base_reference,
                estimates_by_budget[budget_sec]["base"],
                intervals_by_budget[budget_sec]["base"],
                **summary_inputs,
            ),
            agent=_estimate_summary(
                agent_reference,
                estimates_by_budget[budget_sec]["agent"],
                intervals_by_budget[budget_sec]["agent"],
                **summary_inputs,
            ),
            change=_estimate_summary(
                change_reference,
                estimates_by_budget[budget_sec]["change"],
                intervals_by_budget[budget_sec]["change"],
                **summary_inputs,
            ),
        )

    return RealSeedValidationSummary(
        task_id=task_id,
        instance_set=instance_set,
        instance_count=len(instance_ids),
        budgets_sec=declared_budgets,
        generation_seed=validation_seeds.generation_seed,
        reference_seed_count=len(validation_seeds.reference_seeds),
        test_seed_count=len(validation_seeds.test_seeds),
        test_list_count=len(validation_seeds.test_seed_lists),
        seeds_per_test_list=SEEDS_PER_TEST_LIST,
        by_budget=by_budget,
    )


__all__ = [
    "REFERENCE_SEED_COUNT",
    "SEEDS_PER_TEST_LIST",
    "TEST_LIST_COUNT",
    "TEST_SEED_COUNT",
    "VALIDATION_GENERATION_SEED",
    "RealSeedValidationSummary",
    "RealValidationBudget",
    "RealValidationEstimate",
    "RealValidationSeeds",
    "analyze_real_seed_validation",
    "generate_real_validation_seeds",
]
