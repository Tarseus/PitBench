from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from enum import Enum

from pydantic import BaseModel, Field

from pitbench.metrics.performance_report import (
    ExactRunOutcomeKind,
    PerformanceClassification,
    classify_exact_performance,
    compute_exact_run_outcome,
)
from pitbench.schema.observation import (
    CodeState,
    ExpectedRun,
    ExpectedRunGrid,
    RunObservation,
    RunStatus,
)
from pitbench.schema.task import (
    ControlledTransferCondition,
    ControlledTransferView,
    PerformanceProtocol,
    PitBenchTask,
)

BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 20260824
RETENTION_FRACTION = 0.8
REFERENCE_BENEFIT_FLOOR = 0.0
FAMILY_ALPHA = 0.05


class ControlledTransferConditionStatus(str, Enum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    QUALIFICATION_FAILURE = "qualification_failure"
    ORACLE_DISCREPANCY = "oracle_discrepancy"


class ControlledTransferConfidenceInterval(BaseModel):
    lower: float
    upper: float
    level: float
    resamples: int = BOOTSTRAP_RESAMPLES
    bootstrap_seed: int = BOOTSTRAP_SEED


class ControlledTransferConditionReport(BaseModel):
    condition_id: str
    view: ControlledTransferView
    reference_condition_id: str | None = None
    budget_sec: float = Field(gt=0)
    status: ControlledTransferConditionStatus
    native_classification: PerformanceClassification | None = None
    instance_count: int = Field(ge=0)
    benefit: float | None = None
    benefit_ci95: ControlledTransferConfidenceInterval | None = None
    reference_benefit: float | None = None
    retention_ratio: float | None = None
    reference_improvement_established: bool = False
    retention_margin: float | None = None
    retention_margin_lower_simultaneous: float | None = None
    simultaneous_bound_count: int | None = Field(default=None, gt=0)
    supports_retention: bool = False
    base_solved_coverage: float | None = None
    agent_solved_coverage: float | None = None
    paired_coverage_ci95: ControlledTransferConfidenceInterval | None = None
    incomplete_run_count: int = Field(default=0, ge=0)
    qualification_failure_count: int = Field(default=0, ge=0)
    oracle_discrepancy_count: int = Field(default=0, ge=0)


class ControlledTransferScalePathReport(BaseModel):
    scale_path_id: str
    largest_continuously_supported_condition_id: str | None = None


class ControlledTransferReport(BaseModel):
    task_id: str
    problem_variant: str
    formulation_id: str
    retention_fraction: float = RETENTION_FRACTION
    reference_benefit_floor: float = REFERENCE_BENEFIT_FLOOR
    condition_reports: dict[str, ControlledTransferConditionReport]
    scale_paths: dict[str, ControlledTransferScalePathReport]


class _ConditionEvidence(BaseModel):
    status: ControlledTransferConditionStatus
    instance_values: dict[tuple[str, str], float] = Field(default_factory=dict)
    instance_templates: dict[tuple[str, str], str | None] = Field(default_factory=dict)
    base_solved_coverage: float | None = None
    agent_solved_coverage: float | None = None
    coverage_change_by_instance: dict[tuple[str, str], float] = Field(
        default_factory=dict
    )
    incomplete_run_count: int = 0
    qualification_failure_count: int = 0
    oracle_discrepancy_count: int = 0


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of an empty collection")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    return lower_value + (upper_value - lower_value) * (position - lower_index)


def _run_key(run: ExpectedRun | RunObservation) -> tuple[str, str, CodeState, int, float]:
    return (
        run.instance_set,
        run.instance_id,
        run.code_state,
        run.solver_seed,
        run.budget_sec,
    )


def _condition_id(
    run: ExpectedRun | RunObservation,
    config_conditions: dict[str, ControlledTransferCondition],
) -> str | None:
    if run.condition_id is not None:
        return run.condition_id
    condition = next(
        (
            item
            for item in config_conditions.values()
            if item.instance_set == run.instance_set
        ),
        None,
    )
    return condition.condition_id if condition is not None else None


def _pair_key(run: ExpectedRun) -> tuple[str, str, int, float]:
    return (
        run.instance_set,
        run.instance_id,
        run.solver_seed,
        run.budget_sec,
    )


def _instance_key(run: ExpectedRun) -> tuple[str, str]:
    return (run.instance_set, run.instance_id)


def _group_expected_runs(
    expected_run_grid: ExpectedRunGrid,
    config_conditions: dict[str, ControlledTransferCondition],
    *,
    primary_budget_sec: float,
) -> dict[str, list[ExpectedRun]]:
    grouped: dict[str, list[ExpectedRun]] = defaultdict(list)
    for run in expected_run_grid.runs:
        condition_id = _condition_id(run, config_conditions)
        if (
            condition_id is not None
            and run.budget_sec
            == (config_conditions[condition_id].budget_sec or primary_budget_sec)
        ):
            grouped[condition_id].append(run)
    return grouped


def _observations_by_key(
    observations: Iterable[RunObservation],
    config_conditions: dict[str, ControlledTransferCondition],
) -> dict[tuple[str, str, CodeState, int, float], RunObservation]:
    keyed: dict[tuple[str, str, CodeState, int, float], RunObservation] = {}
    for observation in observations:
        if _condition_id(observation, config_conditions) is None:
            continue
        key = _run_key(observation)
        if key in keyed:
            raise ValueError("duplicate controlled transfer observation")
        keyed[key] = observation
    return keyed


def _instance_templates(
    expected_runs: Sequence[ExpectedRun],
    observations_by_key: dict[tuple[str, str, CodeState, int, float], RunObservation],
) -> dict[tuple[str, str], str | None]:
    templates: dict[tuple[str, str], str | None] = {}
    for run in expected_runs:
        instance = _instance_key(run)
        observation = observations_by_key.get(_run_key(run))
        template_id = run.template_id
        if observation is not None and observation.template_id is not None:
            template_id = observation.template_id
        previous = templates.setdefault(instance, template_id)
        if previous != template_id:
            raise ValueError("controlled transfer instance has inconsistent template IDs")
    return templates


def _expected_pairs(
    expected_runs: Sequence[ExpectedRun],
) -> dict[tuple[str, str, int, float], dict[CodeState, ExpectedRun]]:
    pairs: dict[tuple[str, str, int, float], dict[CodeState, ExpectedRun]] = defaultdict(
        dict
    )
    for run in expected_runs:
        pair = pairs[_pair_key(run)]
        if run.code_state in pair:
            raise ValueError("duplicate controlled transfer expected run")
        pair[run.code_state] = run
    if any(set(pair) != set(CodeState) for pair in pairs.values()):
        raise ValueError("controlled transfer expected run grid does not pair Base and Agent")
    return pairs


def _heuristic_evidence(
    expected_runs: Sequence[ExpectedRun],
    observations_by_key: dict[tuple[str, str, CodeState, int, float], RunObservation],
) -> _ConditionEvidence:
    pairs = _expected_pairs(expected_runs)
    values_by_instance: dict[tuple[str, str], list[float]] = defaultdict(list)
    incomplete_run_count = 0
    qualification_failure_count = 0
    for pair in pairs.values():
        base = observations_by_key.get(_run_key(pair[CodeState.BASE]))
        agent = observations_by_key.get(_run_key(pair[CodeState.AGENT]))
        for observation in (base, agent):
            if observation is None or observation.status in {
                RunStatus.BUILD_FAILED,
                RunStatus.INFRASTRUCTURE_ERROR,
            }:
                incomplete_run_count += 1
            elif observation.status is RunStatus.INVALID or not observation.valid:
                qualification_failure_count += 1
            elif observation.normalized_gap is None or not math.isfinite(
                observation.normalized_gap
            ):
                incomplete_run_count += 1
        if incomplete_run_count or qualification_failure_count:
            continue
        assert base is not None and agent is not None
        assert base.normalized_gap is not None and agent.normalized_gap is not None
        values_by_instance[_instance_key(pair[CodeState.BASE])].append(
            base.normalized_gap - agent.normalized_gap
        )
    templates = _instance_templates(expected_runs, observations_by_key)
    if qualification_failure_count:
        status = ControlledTransferConditionStatus.QUALIFICATION_FAILURE
    elif incomplete_run_count:
        status = ControlledTransferConditionStatus.INCOMPLETE
    else:
        status = ControlledTransferConditionStatus.COMPLETE
    return _ConditionEvidence(
        status=status,
        instance_values={
            instance: statistics.fmean(values)
            for instance, values in values_by_instance.items()
        },
        instance_templates=templates,
        incomplete_run_count=incomplete_run_count,
        qualification_failure_count=qualification_failure_count,
    )


def _exact_evidence(
    expected_runs: Sequence[ExpectedRun],
    observations_by_key: dict[tuple[str, str, CodeState, int, float], RunObservation],
    task: PitBenchTask,
) -> _ConditionEvidence:
    pairs = _expected_pairs(expected_runs)
    values_by_instance: dict[tuple[str, str], list[float]] = defaultdict(list)
    base_solved_by_instance: dict[tuple[str, str], list[float]] = defaultdict(list)
    agent_solved_by_instance: dict[tuple[str, str], list[float]] = defaultdict(list)
    incomplete_run_count = 0
    qualification_failure_count = 0
    oracle_discrepancy_count = 0
    for pair in pairs.values():
        base_outcome = compute_exact_run_outcome(
            observations_by_key.get(_run_key(pair[CodeState.BASE])),
            time_basis=task.evaluation.exact_time_basis,
        )
        agent_outcome = compute_exact_run_outcome(
            observations_by_key.get(_run_key(pair[CodeState.AGENT])),
            time_basis=task.evaluation.exact_time_basis,
        )
        outcomes = (base_outcome, agent_outcome)
        incomplete_run_count += sum(
            outcome.kind is ExactRunOutcomeKind.INFRASTRUCTURE_MISSING
            for outcome in outcomes
        )
        qualification_failure_count += sum(
            outcome.kind is ExactRunOutcomeKind.QUALIFICATION_FAILURE
            for outcome in outcomes
        )
        oracle_discrepancy_count += sum(
            outcome.kind is ExactRunOutcomeKind.ORACLE_DISCREPANCY
            for outcome in outcomes
        )
        if any(
            outcome.kind
            not in {
                ExactRunOutcomeKind.VERIFIED_SOLVED,
                ExactRunOutcomeKind.SOLVER_ORIGIN_UNSOLVED,
            }
            for outcome in outcomes
        ):
            continue
        instance = _instance_key(pair[CodeState.BASE])
        assert base_outcome.solved is not None and agent_outcome.solved is not None
        assert base_outcome.penalized_runtime_sec is not None
        assert agent_outcome.penalized_runtime_sec is not None
        base_solved_by_instance[instance].append(float(base_outcome.solved))
        agent_solved_by_instance[instance].append(float(agent_outcome.solved))
        values_by_instance[instance].append(
            (
                base_outcome.penalized_runtime_sec
                - agent_outcome.penalized_runtime_sec
            )
            / pair[CodeState.BASE].budget_sec
        )
    templates = _instance_templates(expected_runs, observations_by_key)
    if qualification_failure_count:
        status = ControlledTransferConditionStatus.QUALIFICATION_FAILURE
    elif oracle_discrepancy_count:
        status = ControlledTransferConditionStatus.ORACLE_DISCREPANCY
    elif incomplete_run_count:
        status = ControlledTransferConditionStatus.INCOMPLETE
    else:
        status = ControlledTransferConditionStatus.COMPLETE
    instance_values = {
        instance: statistics.fmean(values)
        for instance, values in values_by_instance.items()
    }
    base_coverage = (
        statistics.fmean(statistics.fmean(values) for values in base_solved_by_instance.values())
        if base_solved_by_instance
        else None
    )
    agent_coverage = (
        statistics.fmean(statistics.fmean(values) for values in agent_solved_by_instance.values())
        if agent_solved_by_instance
        else None
    )
    if (
        status is ControlledTransferConditionStatus.COMPLETE
        and (base_coverage is None or agent_coverage is None)
    ):
        status = ControlledTransferConditionStatus.INCOMPLETE
    return _ConditionEvidence(
        status=status,
        instance_values=instance_values,
        instance_templates=templates,
        base_solved_coverage=base_coverage,
        agent_solved_coverage=agent_coverage,
        coverage_change_by_instance={
            instance: statistics.fmean(agent_solved_by_instance[instance])
            - statistics.fmean(base_solved_by_instance[instance])
            for instance in base_solved_by_instance
        },
        incomplete_run_count=incomplete_run_count,
        qualification_failure_count=qualification_failure_count,
        oracle_discrepancy_count=oracle_discrepancy_count,
    )


def _cluster_values(evidence: _ConditionEvidence) -> list[float]:
    values_by_cluster: dict[str, list[float]] = defaultdict(list)
    for instance, value in evidence.instance_values.items():
        template_id = evidence.instance_templates.get(instance)
        cluster_id = template_id if template_id is not None else f"{instance[0]}:{instance[1]}"
        values_by_cluster[cluster_id].append(value)
    return [statistics.fmean(values) for values in values_by_cluster.values()]


def _bootstrap_draws(
    evidences: dict[str, _ConditionEvidence],
) -> dict[str, list[float]]:
    clusters = {condition_id: _cluster_values(evidence) for condition_id, evidence in evidences.items()}
    if any(not values for values in clusters.values()):
        return {condition_id: [] for condition_id in evidences}
    random_generator = random.Random(BOOTSTRAP_SEED)
    draws = {condition_id: [] for condition_id in evidences}
    for _ in range(BOOTSTRAP_RESAMPLES):
        for condition_id in sorted(clusters):
            values = clusters[condition_id]
            draws[condition_id].append(
                statistics.fmean(random_generator.choice(values) for _ in values)
            )
    return draws


def _interval(draws: Sequence[float], *, tail_probability: float) -> ControlledTransferConfidenceInterval | None:
    if not draws:
        return None
    return ControlledTransferConfidenceInterval(
        lower=_percentile(draws, tail_probability),
        upper=_percentile(draws, 1 - tail_probability),
        level=1 - 2 * tail_probability,
    )


def _benefit(evidence: _ConditionEvidence) -> float | None:
    return (
        statistics.fmean(evidence.instance_values.values())
        if evidence.status is ControlledTransferConditionStatus.COMPLETE
        and evidence.instance_values
        else None
    )


def compute_controlled_transfer_report(
    observations: Sequence[RunObservation],
    *,
    task: PitBenchTask,
    expected_run_grid: ExpectedRunGrid,
) -> ControlledTransferReport | None:
    config = task.evaluation.controlled_transfer
    if config is None:
        return None
    if task.problem_variant is None or task.formulation_id is None:
        raise ValueError("controlled transfer requires task problem metadata")
    if expected_run_grid.task_id != task.task_id:
        raise ValueError("controlled transfer expected run grid belongs to another task")
    conditions = {condition.condition_id: condition for condition in config.conditions}
    for observation in observations:
        if observation.condition_id is not None and observation.condition_id not in conditions:
            raise ValueError("controlled transfer observation has an unknown condition ID")
    expected_by_condition = _group_expected_runs(
        expected_run_grid,
        conditions,
        primary_budget_sec=task.evaluation.primary_budget_sec,
    )
    observations_by_key = _observations_by_key(observations, conditions)
    for expected_run in expected_run_grid.runs:
        expected_condition_id = _condition_id(expected_run, conditions)
        observation = observations_by_key.get(_run_key(expected_run))
        if observation is None:
            continue
        observed_condition_id = _condition_id(observation, conditions)
        if expected_condition_id != observed_condition_id:
            raise ValueError(
                "controlled transfer observation condition does not match expected run"
            )
    evidences: dict[str, _ConditionEvidence] = {}
    for condition_id, condition in conditions.items():
        expected_runs = expected_by_condition.get(condition_id, [])
        if not expected_runs:
            evidences[condition_id] = _ConditionEvidence(
                status=ControlledTransferConditionStatus.INCOMPLETE,
                incomplete_run_count=1,
            )
        elif task.evaluation.performance_protocol is PerformanceProtocol.EXACT_VERIFIED_SOLVE:
            evidences[condition_id] = _exact_evidence(
                expected_runs,
                observations_by_key,
                task,
            )
        else:
            evidences[condition_id] = _heuristic_evidence(
                expected_runs,
                observations_by_key,
            )
    complete_evidences = {
        condition_id: evidence
        for condition_id, evidence in evidences.items()
        if evidence.status is ControlledTransferConditionStatus.COMPLETE
    }
    draws = _bootstrap_draws(complete_evidences)
    coverage_draws = (
        _bootstrap_draws(
            {
                condition_id: evidence.model_copy(
                    update={"instance_values": evidence.coverage_change_by_instance}
                )
                for condition_id, evidence in complete_evidences.items()
            }
        )
        if task.evaluation.performance_protocol is PerformanceProtocol.EXACT_VERIFIED_SOLVE
        else {}
    )
    reports: dict[str, ControlledTransferConditionReport] = {}
    for condition_id, condition in conditions.items():
        evidence = evidences[condition_id]
        benefit = _benefit(evidence)
        benefit_ci = _interval(draws.get(condition_id, []), tail_probability=0.025)
        coverage_ci = _interval(
            coverage_draws.get(condition_id, []), tail_probability=0.025
        )
        reports[condition_id] = ControlledTransferConditionReport(
            condition_id=condition_id,
            view=condition.view,
            reference_condition_id=condition.reference_condition_id,
            budget_sec=condition.budget_sec or task.evaluation.primary_budget_sec,
            status=evidence.status,
            instance_count=len(evidence.instance_values),
            benefit=benefit,
            benefit_ci95=benefit_ci,
            base_solved_coverage=evidence.base_solved_coverage,
            agent_solved_coverage=evidence.agent_solved_coverage,
            paired_coverage_ci95=coverage_ci,
            incomplete_run_count=evidence.incomplete_run_count,
            qualification_failure_count=evidence.qualification_failure_count,
            oracle_discrepancy_count=evidence.oracle_discrepancy_count,
        )
    if task.evaluation.performance_protocol is PerformanceProtocol.EXACT_VERIFIED_SOLVE:
        for report in reports.values():
            if report.status is not ControlledTransferConditionStatus.COMPLETE:
                continue
            report.native_classification = classify_exact_performance(
                coverage_difference=(
                    report.agent_solved_coverage - report.base_solved_coverage
                    if report.agent_solved_coverage is not None
                    and report.base_solved_coverage is not None
                    else None
                ),
                coverage_interval=(
                    (
                        report.paired_coverage_ci95.lower,
                        report.paired_coverage_ci95.upper,
                    )
                    if report.paired_coverage_ci95 is not None
                    else None
                ),
                runtime_interval=(
                    (report.benefit_ci95.lower, report.benefit_ci95.upper)
                    if report.benefit_ci95 is not None
                    else None
                ),
                complete=True,
            )
    targets_by_reference: dict[str, list[str]] = defaultdict(list)
    for condition in conditions.values():
        if condition.reference_condition_id is not None:
            targets_by_reference[condition.reference_condition_id].append(condition.condition_id)
    for reference_id, target_ids in targets_by_reference.items():
        reference = reports[reference_id]
        reference_draws = draws.get(reference_id, [])
        family_bound_count = len(target_ids) + 1
        joint_lower_probability = FAMILY_ALPHA / family_bound_count
        reference_lower = (
            _percentile(reference_draws, joint_lower_probability)
            if reference_draws
            else None
        )
        for target_id in target_ids:
            target = reports[target_id]
            if reference.benefit is not None:
                target.reference_benefit = reference.benefit
            if (
                reference.benefit is None
                or reference.benefit_ci95 is None
                or reference.benefit_ci95.lower <= REFERENCE_BENEFIT_FLOOR
                or target.benefit is None
            ):
                continue
            target.reference_improvement_established = True
            target.retention_ratio = target.benefit / reference.benefit
            target.retention_margin = target.benefit - RETENTION_FRACTION * reference.benefit
            target_draws = draws.get(target_id, [])
            if target_draws and reference_draws:
                margin_draws = [
                    target_value - RETENTION_FRACTION * reference_value
                    for target_value, reference_value in zip(target_draws, reference_draws)
                ]
                target.retention_margin_lower_simultaneous = _percentile(
                    margin_draws,
                    joint_lower_probability,
                )
                target.simultaneous_bound_count = family_bound_count
                target.supports_retention = (
                    reference_lower is not None
                    and reference_lower > REFERENCE_BENEFIT_FLOOR
                    and target.retention_margin_lower_simultaneous > 0
                    and target.status is ControlledTransferConditionStatus.COMPLETE
                    and target.native_classification
                    is not PerformanceClassification.REGRESSED
                )
    scale_paths: dict[str, ControlledTransferScalePathReport] = {}
    scale_conditions: dict[str, list[ControlledTransferCondition]] = defaultdict(list)
    for condition in config.conditions:
        if (
            condition.view in {ControlledTransferView.SCALE, ControlledTransferView.JOINT}
            and condition.scale_path_id is not None
        ):
            scale_conditions[condition.scale_path_id].append(condition)
    for scale_path_id, path_conditions in scale_conditions.items():
        path = scale_paths.setdefault(
            scale_path_id,
            ControlledTransferScalePathReport(scale_path_id=scale_path_id),
        )
        for condition in path_conditions:
            report = reports[condition.condition_id]
            if report.supports_retention:
                path.largest_continuously_supported_condition_id = condition.condition_id
            else:
                # Config order declares the ordered path. A later condition cannot bridge a failure.
                break
    return ControlledTransferReport(
        task_id=task.task_id,
        problem_variant=task.problem_variant,
        formulation_id=task.formulation_id,
        condition_reports=reports,
        scale_paths=scale_paths,
    )


def format_controlled_transfer_report(report: ControlledTransferReport) -> str:
    lines = [
        "Controlled Transfer",
        f"Task context: {report.problem_variant} / {report.formulation_id}",
        f"Retention requirement: {100 * report.retention_fraction:g}%",
    ]
    for condition in report.condition_reports.values():
        benefit = "-" if condition.benefit is None else f"{condition.benefit:.6g}"
        ratio = "-" if condition.retention_ratio is None else f"{condition.retention_ratio:.6g}"
        native = (
            condition.native_classification.value
            if condition.native_classification is not None
            else "-"
        )
        coverage = ""
        if (
            condition.base_solved_coverage is not None
            and condition.agent_solved_coverage is not None
        ):
            coverage = (
                f"; solved_coverage={condition.base_solved_coverage:.3%}"
                f"->{condition.agent_solved_coverage:.3%}"
            )
            if condition.paired_coverage_ci95 is not None:
                interval = condition.paired_coverage_ci95
                coverage += (
                    f"; coverage_delta_ci95=[{interval.lower:.3%},"
                    f" {interval.upper:.3%}]"
                )
        lines.append(
            f"{condition.condition_id}: {condition.status.value}; "
            f"benefit={benefit}; retention={ratio}; native={native}{coverage}"
        )
    for path in report.scale_paths.values():
        condition_id = path.largest_continuously_supported_condition_id or "-"
        lines.append(
            f"{path.scale_path_id}: largest continuously supported condition={condition_id}"
        )
    return "\n".join(lines)


__all__ = [
    "ControlledTransferConditionReport",
    "ControlledTransferConditionStatus",
    "ControlledTransferConfidenceInterval",
    "ControlledTransferReport",
    "ControlledTransferScalePathReport",
    "compute_controlled_transfer_report",
    "format_controlled_transfer_report",
]
