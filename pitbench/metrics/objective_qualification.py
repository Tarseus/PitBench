"""Recompute preregistered objective decisions from complete paired Judge data."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, Field

from pitbench.metrics.performance_report import (
    BOOTSTRAP_SEED,
    _cluster_mean_ci,
    compute_exact_performance_report,
    compute_performance_report,
)
from pitbench.metrics.resource_report import compute_resource_reports
from pitbench.schema.objectives import (
    qualification_protocols,
    qualification_protocols_sha256,
)
from pitbench.schema.observation import (
    CodeState,
    ExpectedRun,
    ExpectedRunGrid,
    RunObservation,
    RunStatus,
)
from pitbench.schema.task import PerformanceProtocol, PitBenchTask


class MetricDecision(BaseModel):
    metric_id: str
    instance_set: str
    comparison: str
    # Positive means worse, in either absolute units or Agent/Base minus one.
    degradation: float | None = None
    lower: float | None = None
    upper: float | None = None
    confidence_level: float | None = None
    complete: bool = False
    classification: str = "incomplete"
    max_degradation: float | None = None
    guardrail_passed: bool | None = None
    detail: str = ""


class ObjectiveScore(BaseModel):
    classification: str
    primary: list[MetricDecision] = Field(default_factory=list)
    guardrails: list[MetricDecision] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class ObjectiveQualificationReport(BaseModel):
    protocol_version: Literal["1.0"] = "1.0"
    protocols_sha256: str
    primary_metric_id: str | None
    classification: str
    initial: ObjectiveScore | None = None
    repeat: ObjectiveScore | None = None
    eligible_for_confirmation: bool = False
    confirmed: bool = False
    performance_confirmed: bool = False
    reasons: list[str] = Field(default_factory=list)


def objective_protocol_gaps(task: PitBenchTask) -> list[str]:
    """Missing decisions are pending research inputs, never implicit tolerances."""
    profile = task.optimization_profile
    if profile is None:
        return ["No optimization profile declared."]
    issues = []
    if profile.classification_basis == "heuristic_hint":
        issues.append("Title hints cannot select a primary metric or admit a PR.")
    if profile.classification_basis == "semantic_model":
        issues.append(
            "Model labels need a confirmed objective declaration before qualification."
        )
    if not profile.primary_metric_id:
        issues.append("Primary metric not declared before qualification.")
    plan = profile.scoring
    if plan is None:
        return [*issues, "No frozen objective scoring plan."]
    if plan.guardrails is None:
        issues.append("Allowed degradation has not been declared.")
    exact = (
        task.evaluation.performance_protocol == PerformanceProtocol.EXACT_VERIFIED_SOLVE
    )
    required = (
        {"exact_solved_coverage", "exact_penalized_runtime_sec", "peak_rss_ratio"}
        if exact
        else {"fixed_budget_normalized_gap", "cpu_time_ratio", "peak_rss_ratio"}
    ) - {profile.primary_metric_id}
    for metric in sorted(required):
        covered = {
            name
            for guard in plan.guardrails or []
            if guard.metric_id == metric
            for name in guard.instance_sets
        }
        missing = set(plan.qualification_instance_sets) - covered
        if missing:
            issues.append(
                f"Declare allowed degradation for {metric}: {sorted(missing)}."
            )
    definitions = qualification_protocols()["metrics"]
    for metric, comparison in [
        (profile.primary_metric_id, None),
        *((guard.metric_id, guard.comparison) for guard in plan.guardrails or []),
    ]:
        if metric is None:
            continue
        rule = definitions[metric]
        if rule["status"] != "implemented":
            issues.append(f"{metric}: {rule['status']}; {rule['decision']}")
        elif comparison is not None and comparison != rule["comparison"]:
            issues.append(f"{metric} requires {rule['comparison']} guardrail units.")
        if metric.startswith("exact_") and not exact:
            issues.append(f"{metric} requires the exact solve protocol.")
        if metric == "fixed_budget_normalized_gap" and exact:
            issues.append(
                "Fixed-budget gap cannot replace exact correctness or coverage."
            )
    return issues


def _original(item: ExpectedRun | RunObservation) -> bool:
    return item.equivalence_parent_id is None and item.test_suite is None


def _key(item: ExpectedRun | RunObservation) -> tuple:
    return (
        item.task_id,
        item.instance_set,
        item.instance_id,
        item.code_state,
        item.solver_seed,
        item.budget_sec,
        item.condition_id,
        item.template_id,
        item.equivalence_parent_id,
        item.test_suite,
    )


def _grid_problems(
    task: PitBenchTask,
    observations: list[RunObservation],
    grid: ExpectedRunGrid | None,
    panels: set[str],
) -> list[str]:
    if grid is None:
        return ["Evaluator-owned expected run grid is missing."]
    if grid.task_id != task.task_id:
        raise ValueError("objective grid belongs to another task")
    if any(item.task_id != task.task_id for item in [*grid.runs, *observations]):
        raise ValueError("objective evidence contains another task")
    expected = [
        item for item in grid.runs if _original(item) and item.instance_set in panels
    ]
    observed = [
        item for item in observations if _original(item) and item.instance_set in panels
    ]
    expected_keys, observed_keys = (
        [_key(item) for item in expected],
        [_key(item) for item in observed],
    )
    if len(set(expected_keys)) != len(expected_keys) or len(set(observed_keys)) != len(
        observed_keys
    ):
        raise ValueError("duplicate objective grid or observation")
    if set(observed_keys) - set(expected_keys):
        raise ValueError("objective observations outside expected grid")
    problems = []
    if set(observed_keys) != set(expected_keys):
        problems.append("Missing declared paired Judge observations.")
    seed_count = (
        task.evaluation.seed_robustness.seed_selection.seed_count
        if task.evaluation.seed_robustness
        else len(task.evaluation.solver_seeds or [])
    )
    for spec in task.instance_sets:
        if spec.name not in panels:
            continue
        runs = [item for item in expected if item.instance_set == spec.name]
        cells = defaultdict(set)
        seeds_by_instance = defaultdict(set)
        for item in runs:
            cells[(item.instance_id, item.solver_seed)].add(
                (item.code_state, item.budget_sec)
            )
            seeds_by_instance[item.instance_id].add(item.solver_seed)
        combinations = {
            (state, budget)
            for state in CodeState
            for budget in task.evaluation.budgets_sec
        }
        if (
            len(seeds_by_instance) != spec.size
            or any(len(values) != seed_count for values in seeds_by_instance.values())
            or any(values != combinations for values in cells.values())
            or any(
                item.condition_id is not None or item.template_id is not None
                for item in runs
            )
        ):
            problems.append(
                f"{spec.name}: qualification grid is reduced or uses an undeclared scenario adapter."
            )
        if task.evaluation.solver_seeds is not None and any(
            values != set(task.evaluation.solver_seeds)
            for values in seeds_by_instance.values()
        ):
            problems.append(f"{spec.name}: solver seeds differ from the task.")
        if (
            seed_count
            and len({tuple(sorted(values)) for values in seeds_by_instance.values()})
            > 1
        ):
            problems.append(
                f"{spec.name}: expected seed lists differ across instances."
            )
    if any(item.threads != task.evaluation.threads for item in observed):
        problems.append("Observed thread count differs from the task.")
    if any(
        item.status in {RunStatus.BUILD_FAILED, RunStatus.INFRASTRUCTURE_ERROR}
        for item in observed
    ):
        problems.append("Infrastructure or build failures remain in the run grid.")
    return problems


def _metric_decision(
    task: PitBenchTask,
    observations: list[RunObservation],
    grid: ExpectedRunGrid,
    metric: str,
    panel: str,
) -> MetricDecision:
    rule = qualification_protocols()["metrics"][metric]
    result = MetricDecision(
        metric_id=metric, instance_set=panel, comparison=rule["comparison"]
    )
    budget = task.evaluation.primary_budget_sec
    selected = [
        item
        for item in observations
        if _original(item) and item.instance_set == panel and item.budget_sec == budget
    ]
    if rule["adapter"] == "paired_resource":
        count = next(item.size for item in task.instance_sets if item.name == panel)
        seed_count = (
            task.evaluation.seed_robustness.seed_selection.seed_count
            if task.evaluation.seed_robustness
            else len(task.evaluation.solver_seeds or [])
        )
        _, details = compute_resource_reports(
            selected,
            primary_budget_sec=budget,
            budgets_sec=[budget],
            expected_seed_count=seed_count,
            expected_instance_counts={panel: count},
        )
        instances = details.by_instance_set.get(panel, {}).get(format(budget, "g"), [])
        field = "memory" if metric == "peak_rss_ratio" else "cpu"
        values = [getattr(item, field) for item in instances]
        if (
            count < 2
            or len(values) != count
            or any(
                value.paired_seed_count != seed_count or value.agent_base_ratio is None
                for value in values
            )
            or len({value.measurement_scope for value in values}) != 1
        ):
            result.detail = "Need at least two complete instances with all seeds and one qualified resource scope. Failed runs are not savings."
            return result
        logs = {
            (panel, item.instance_id): [math.log(value.agent_base_ratio)]
            for item, value in zip(instances, values, strict=True)
        }
        ci = _cluster_mean_ci(logs, seed=BOOTSTRAP_SEED)
        assert ci is not None
        result.degradation = math.expm1(
            statistics.fmean(value[0] for value in logs.values())
        )
        result.lower, result.upper = math.expm1(ci.lower), math.expm1(ci.upper)
        result.confidence_level = ci.level
        result.detail = "Geometric mean of per-instance ratios of paired seed means; native instance bootstrap on log ratios."
    elif rule["adapter"] == "native_exact":
        # Apply the existing exact definition to each frozen qualification panel.
        native_grid = ExpectedRunGrid(
            task_id=task.task_id,
            runs=[
                item.model_copy(update={"instance_set_kind": "judge_id"})
                for item in grid.runs
                if _original(item)
                and item.instance_set == panel
                and item.budget_sec == budget
            ],
        )
        native = compute_exact_performance_report(
            [
                item.model_copy(update={"instance_set_kind": "judge_id"})
                for item in selected
            ],
            primary_budget_sec=budget,
            budgets_sec=[budget],
            expected_run_grid=native_grid,
            time_basis=task.evaluation.exact_time_basis,
        )
        paired = native.primary.paired
        coverage = metric == "exact_solved_coverage"
        value = (
            paired.mean_verified_solved_coverage_reduction
            if coverage
            else paired.mean_penalized_runtime_reduction_sec
        )
        ci = (
            paired.mean_verified_solved_coverage_reduction_ci95
            if coverage
            else paired.mean_penalized_runtime_reduction_ci95
        )
        if not paired.complete or value is None or ci is None:
            return result
        result.degradation, result.lower, result.upper = -value, -ci.upper, -ci.lower
        result.confidence_level = ci.level
        result.classification = native.classification.value
        result.detail = "Native exact coverage-first classification; signed interval describes this metric alone."
    else:
        if any(
            not item.valid
            or item.status != RunStatus.COMPLETED
            or item.normalized_gap is None
            or not math.isfinite(item.normalized_gap)
            for item in selected
        ):
            result.detail = "Every declared fixed-budget run needs an independently verified finite gap."
            return result
        native = compute_performance_report(selected, primary_budget_sec=budget)
        paired = native.primary.paired
        ci, value = paired.mean_gap_reduction_ci95, paired.mean_gap_reduction
        if ci is None or value is None:
            return result
        result.degradation, result.lower, result.upper = -value, -ci.upper, -ci.lower
        result.confidence_level = ci.level
        result.detail = (
            "Native paired normalized-gap reduction with complete-grid checks."
        )
    result.complete = True
    if rule["adapter"] != "native_exact":
        result.classification = (
            "improved"
            if result.upper < 0
            else "regressed"
            if result.lower > 0
            else "inconclusive"
        )
    return result


def _score(
    task: PitBenchTask, observations: list[RunObservation], grid: ExpectedRunGrid | None
) -> ObjectiveScore:
    profile = task.optimization_profile
    assert profile and profile.scoring and profile.primary_metric_id
    plan = profile.scoring
    panels = set(plan.qualification_instance_sets)
    panels.update(
        name for guard in plan.guardrails or [] for name in guard.instance_sets
    )
    problems = _grid_problems(task, observations, grid, panels)
    if problems:
        return ObjectiveScore(classification="incomplete", reasons=problems)
    assert grid is not None
    selected = [
        item for item in observations if _original(item) and item.instance_set in panels
    ]
    if any(
        item.status == RunStatus.INVALID
        or (item.status == RunStatus.COMPLETED and not item.valid)
        for item in selected
    ):
        return ObjectiveScore(
            classification="invalid",
            reasons=[
                "Independent correctness failed in the qualification/control panel."
            ],
        )
    primary = [
        _metric_decision(task, observations, grid, profile.primary_metric_id, name)
        for name in plan.qualification_instance_sets
    ]
    guards = []
    for guard in plan.guardrails or []:
        for name in guard.instance_sets:
            measured = _metric_decision(task, observations, grid, guard.metric_id, name)
            measured.max_degradation = guard.max_degradation
            measured.guardrail_passed = (
                measured.upper <= guard.max_degradation if measured.complete else None
            )
            guards.append(measured)
    if any(not item.complete for item in [*primary, *guards]):
        status = "incomplete"
    elif any(item.guardrail_passed is False for item in guards):
        status = "guardrail_failed"
    elif all(item.classification == "improved" for item in primary):
        status = "improved"
    elif any(item.classification == "regressed" for item in primary):
        status = "regressed"
    else:
        status = "inconclusive"
    return ObjectiveScore(classification=status, primary=primary, guardrails=guards)


def compute_objective_qualification(
    *,
    task: PitBenchTask,
    observations: list[RunObservation],
    expected_run_grid: ExpectedRunGrid | None,
    repeat_observations: list[RunObservation] | None = None,
    repeat_expected_run_grid: ExpectedRunGrid | None = None,
    repeat_identity_verified: bool = False,
    validity_accepted: bool | None = None,
) -> ObjectiveQualificationReport | None:
    """A metric decision can satisfy performance confirmation, never other gates."""
    profile = task.optimization_profile
    if profile is None:
        return None
    result = ObjectiveQualificationReport(
        protocols_sha256=qualification_protocols_sha256(),
        primary_metric_id=profile.primary_metric_id,
        classification="needs_protocol",
        reasons=objective_protocol_gaps(task),
    )
    if result.reasons:
        return result
    result.initial = _score(task, observations, expected_run_grid)
    result.classification = result.initial.classification
    result.eligible_for_confirmation = result.initial.classification == "improved"
    if validity_accepted is False:
        result.classification = "invalid"
        result.eligible_for_confirmation = False
        result.reasons.append("Evaluator correctness/build/patch validity failed.")
        return result
    if not result.eligible_for_confirmation:
        return result
    if repeat_observations is None:
        result.classification = "confirmation_required"
        return result
    if not repeat_identity_verified or expected_run_grid != repeat_expected_run_grid:
        result.classification = "incomplete"
        result.reasons.append(
            "Fresh reversed-order Judge identity and identical expected grid are required."
        )
        return result
    result.repeat = _score(task, repeat_observations, repeat_expected_run_grid)
    if result.repeat.classification != "improved":
        result.classification = (
            result.repeat.classification
            if result.repeat.classification
            in {"invalid", "incomplete", "guardrail_failed"}
            else "inconclusive"
        )
        return result
    if validity_accepted is not True:
        result.classification = "incomplete"
        result.reasons.append(
            "Independent evaluator validity has not been established."
        )
        return result
    result.classification = "improved"
    result.confirmed = result.performance_confirmed = True
    return result
