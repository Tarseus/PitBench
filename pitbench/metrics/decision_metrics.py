from __future__ import annotations

from pydantic import BaseModel, Field

from pitbench.metrics.outcome_metrics import OutcomeReport
from pitbench.metrics.sensitivity_metrics import SensitivityReport
from pitbench.schema.task import DecisionProtocol

DECISION_POLICY_NAME = "pitbench-pareto-gated-improvement"
DECISION_POLICY_VERSION = "1.0"


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


__all__ = [
    "BenchmarkDecision",
    "DECISION_POLICY_NAME",
    "DECISION_POLICY_VERSION",
    "compute_benchmark_decision",
]
