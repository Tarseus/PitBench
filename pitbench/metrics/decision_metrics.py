from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from pitbench.metrics.outcome_metrics import OutcomeReport
from pitbench.metrics.sensitivity_metrics import SensitivityReport
from pitbench.schema.observation import CodeState, RunObservation
from pitbench.schema.task import DecisionProtocol

DECISION_POLICY_NAME = "pitbench-pareto-gated-improvement"
DECISION_POLICY_VERSION = "1.0"
MODEL_BUILD_DECISION_POLICY_NAME = "pitbench-model-build-pareto-gated-improvement"


class BenchmarkDecision(BaseModel):
    policy_name: str = DECISION_POLICY_NAME
    policy_version: str = DECISION_POLICY_VERSION
    is_resolved: bool
    classification: str
    outcome_complete: bool
    resource_telemetry_complete: bool
    sensitivity_complete: bool
    improvements: list[str] = Field(default_factory=list)
    regressions: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    paired_model_runs: int = Field(default=0, ge=0)
    model_variable_ratio: float | None = None
    model_constraint_ratio: float | None = None


def _sensitivity_complete(report: SensitivityReport) -> bool:
    rows = (report.matrix.equivalence, report.matrix.scale, report.matrix.population)
    return all(
        value is not None
        for row in rows
        for value in (row.performance, row.reliability, row.resource)
    )


def compute_benchmark_decision(
    outcomes: OutcomeReport,
    sensitivity: SensitivityReport,
    protocol: DecisionProtocol,
    *,
    validity_accepted: bool,
) -> BenchmarkDecision:
    comparison = outcomes.comparison
    outcome_complete = all(
        value is not None
        for value in (
            outcomes.base.performance.mean_normalized_gap,
            outcomes.agent.performance.mean_normalized_gap,
            comparison.operational_speedup,
        )
    )
    telemetry_complete = all(
        value is not None
        for value in (
            comparison.cpu_speedup,
            comparison.peak_rss_ratio,
        )
    )
    sensitivity_complete = _sensitivity_complete(sensitivity)
    missing: list[str] = []
    if not validity_accepted:
        missing.append("candidate validity checks did not pass")
    if not outcome_complete:
        missing.append("primary outcome coordinates are incomplete")
    if not telemetry_complete:
        missing.append("CPU or peak-RSS telemetry is incomplete")
    if protocol.require_complete_sensitivity and not sensitivity_complete:
        missing.append("the 3x3 sensitivity matrix is incomplete")

    regressions: list[str] = []
    if comparison.delta_success_rate < protocol.minimum_success_rate_delta:
        regressions.append("success rate regressed")
    if comparison.gap_reduction is not None and comparison.gap_reduction < -1e-12:
        regressions.append("mean normalized gap regressed")
    if (
        comparison.operational_speedup is not None
        and comparison.operational_speedup < protocol.minimum_operational_speedup
    ):
        regressions.append("operational runtime regressed beyond tolerance")
    if (
        comparison.cpu_speedup is not None
        and comparison.cpu_speedup < protocol.minimum_cpu_speedup
    ):
        regressions.append("CPU time regressed beyond tolerance")
    if (
        comparison.peak_rss_ratio is not None
        and comparison.peak_rss_ratio > protocol.maximum_peak_rss_ratio
    ):
        regressions.append("peak RSS regressed beyond tolerance")

    improvements: list[str] = []
    if comparison.gap_reduction is not None and comparison.gap_reduction > 1e-12:
        improvements.append("mean normalized gap")
    for name, value in (
        ("operational runtime", comparison.operational_speedup),
        ("common-success runtime", comparison.common_success_speedup),
        ("CPU time", comparison.cpu_speedup),
    ):
        if value is not None and value > 1.0 + 1e-12:
            improvements.append(name)
    if (
        comparison.peak_rss_ratio is not None
        and comparison.peak_rss_ratio < 1.0 - 1e-12
    ):
        improvements.append("peak RSS")

    is_resolved = not missing and not regressions and bool(improvements)
    if missing:
        classification = "incomplete"
    elif regressions and improvements:
        classification = "tradeoff"
    elif regressions:
        classification = "regressed"
    elif improvements:
        classification = "improved"
    else:
        classification = "no_change"
    return BenchmarkDecision(
        is_resolved=is_resolved,
        classification=classification,
        outcome_complete=outcome_complete,
        resource_telemetry_complete=telemetry_complete,
        sensitivity_complete=sensitivity_complete,
        improvements=improvements,
        regressions=regressions,
        missing_evidence=missing,
    )


def _primary_original_observations(
    observations: Sequence[RunObservation],
) -> list[RunObservation]:
    originals = [
        observation
        for observation in observations
        if observation.equivalence_parent_id is None
    ]
    primary = [
        observation
        for observation in originals
        if observation.population_kind == "judge_id"
    ]
    return primary or originals


def _model_size_ratios(
    observations: Sequence[RunObservation],
) -> tuple[int, float | None, float | None]:
    selected = _primary_original_observations(observations)
    by_state: dict[CodeState, dict[tuple[str, str, int, float], RunObservation]] = {
        CodeState.BASE: {},
        CodeState.AGENT: {},
    }
    for observation in selected:
        key = (
            observation.population,
            observation.instance_id,
            observation.solver_seed,
            observation.budget_sec,
        )
        by_state[observation.code_state][key] = observation

    paired = []
    for key in sorted(set(by_state[CodeState.BASE]) & set(by_state[CodeState.AGENT])):
        base = by_state[CodeState.BASE][key]
        agent = by_state[CodeState.AGENT][key]
        if not base.valid or not agent.valid:
            continue
        if any(
            value is None
            for value in (
                base.model_variables,
                agent.model_variables,
                base.model_constraints,
                agent.model_constraints,
            )
        ):
            continue
        paired.append((base, agent))

    if not paired:
        return 0, None, None
    base_variables = sum(item[0].model_variables or 0 for item in paired)
    agent_variables = sum(item[1].model_variables or 0 for item in paired)
    base_constraints = sum(item[0].model_constraints or 0 for item in paired)
    agent_constraints = sum(item[1].model_constraints or 0 for item in paired)
    variable_ratio = agent_variables / base_variables if base_variables > 0 else None
    constraint_ratio = (
        agent_constraints / base_constraints if base_constraints > 0 else None
    )
    return len(paired), variable_ratio, constraint_ratio


def compute_model_build_decision(
    outcomes: OutcomeReport,
    sensitivity: SensitivityReport,
    observations: Sequence[RunObservation],
    protocol: DecisionProtocol,
    *,
    validity_accepted: bool,
) -> BenchmarkDecision:
    """Decide MODEL_BUILD/PRESOLVE tasks without requiring objective gaps."""

    comparison = outcomes.comparison
    paired_runs, variable_ratio, constraint_ratio = _model_size_ratios(observations)
    outcome_complete = all(
        value is not None
        for value in (
            variable_ratio,
            constraint_ratio,
            comparison.operational_speedup,
        )
    )
    telemetry_complete = all(
        value is not None
        for value in (comparison.cpu_speedup, comparison.peak_rss_ratio)
    )
    sensitivity_complete = _sensitivity_complete(sensitivity)

    missing: list[str] = []
    if not validity_accepted:
        missing.append("candidate validity checks did not pass")
    if not outcome_complete:
        missing.append("paired model-size or runtime evidence is incomplete")
    if not telemetry_complete:
        missing.append("CPU or peak-RSS telemetry is incomplete")
    if protocol.require_complete_sensitivity and not sensitivity_complete:
        missing.append("the 3x3 sensitivity matrix is incomplete")

    regressions: list[str] = []
    if comparison.delta_success_rate < protocol.minimum_success_rate_delta:
        regressions.append("success rate regressed")
    if variable_ratio is not None and variable_ratio > 1.0 + 1e-12:
        regressions.append("model variable count increased")
    if constraint_ratio is not None and constraint_ratio > 1.0 + 1e-12:
        regressions.append("model constraint count increased")
    if (
        comparison.operational_speedup is not None
        and comparison.operational_speedup < protocol.minimum_operational_speedup
    ):
        regressions.append("operational runtime regressed beyond tolerance")
    if (
        comparison.cpu_speedup is not None
        and comparison.cpu_speedup < protocol.minimum_cpu_speedup
    ):
        regressions.append("CPU time regressed beyond tolerance")
    if (
        comparison.peak_rss_ratio is not None
        and comparison.peak_rss_ratio > protocol.maximum_peak_rss_ratio
    ):
        regressions.append("peak RSS regressed beyond tolerance")

    improvements: list[str] = []
    if variable_ratio is not None and variable_ratio < 1.0 - 1e-12:
        improvements.append("model variable count")
    if constraint_ratio is not None and constraint_ratio < 1.0 - 1e-12:
        improvements.append("model constraint count")
    for name, value in (
        ("operational runtime", comparison.operational_speedup),
        ("common-success runtime", comparison.common_success_speedup),
        ("CPU time", comparison.cpu_speedup),
    ):
        if value is not None and value > 1.0 + 1e-12:
            improvements.append(name)
    if (
        comparison.peak_rss_ratio is not None
        and comparison.peak_rss_ratio < 1.0 - 1e-12
    ):
        improvements.append("peak RSS")

    is_resolved = not missing and not regressions and bool(improvements)
    if missing:
        classification = "incomplete"
    elif regressions and improvements:
        classification = "tradeoff"
    elif regressions:
        classification = "regressed"
    elif improvements:
        classification = "improved"
    else:
        classification = "no_change"
    return BenchmarkDecision(
        policy_name=MODEL_BUILD_DECISION_POLICY_NAME,
        is_resolved=is_resolved,
        classification=classification,
        outcome_complete=outcome_complete,
        resource_telemetry_complete=telemetry_complete,
        sensitivity_complete=sensitivity_complete,
        improvements=improvements,
        regressions=regressions,
        missing_evidence=missing,
        paired_model_runs=paired_runs,
        model_variable_ratio=variable_ratio,
        model_constraint_ratio=constraint_ratio,
    )


__all__ = [
    "BenchmarkDecision",
    "DECISION_POLICY_NAME",
    "DECISION_POLICY_VERSION",
    "MODEL_BUILD_DECISION_POLICY_NAME",
    "compute_benchmark_decision",
    "compute_model_build_decision",
]
