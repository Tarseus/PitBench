from __future__ import annotations

from pydantic import BaseModel, Field

from pitbench.metrics.performance_report import PerformanceReport
from pitbench.schema.task import DecisionProtocol


class PerformanceDecision(BaseModel):
    policy_name: str = "pitbench-performance-first"
    classification: str
    validity_complete: bool
    performance_complete: bool
    repeatability_complete: bool
    held_out_complete: bool
    improvements: list[str] = Field(default_factory=list)
    regressions: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)


def compute_performance_decision(
    report: PerformanceReport,
    protocol: DecisionProtocol,
    *,
    validity_accepted: bool,
) -> PerformanceDecision:
    """Decide a quality-time claim from paired multi-seed holdout evidence."""
    primary = report.primary
    paired = primary.paired
    performance_complete = all(
        value is not None
        for value in (
            primary.base.mean_normalized_gap,
            primary.agent.mean_normalized_gap,
            paired.mean_gap_reduction,
            paired.mean_gap_reduction_ci95,
        )
    )
    repeatability_complete = (
        len(paired.solver_seeds) >= 2
        and paired.paired_runs > 0
        and paired.paired_instances >= 2
    )
    retention = next(
        (
            item
            for item in report.held_out_retention
            if item.budget_sec == report.primary_budget_sec
        ),
        None,
    )
    shift_cell = report.instance_sets.get("judge_shift")
    shift_budget = (
        shift_cell.by_budget.get(f"{report.primary_budget_sec:g}")
        if shift_cell is not None
        else None
    )
    held_out_complete = bool(
        retention is not None
        and shift_budget is not None
        and shift_budget.paired.mean_gap_reduction_ci95 is not None
    )

    missing: list[str] = []
    if not validity_accepted:
        missing.append("candidate validity checks did not pass")
    if not performance_complete:
        missing.append("paired fixed-budget gap evidence is incomplete")
    if not repeatability_complete:
        missing.append("multi-seed repeatability evidence is incomplete")
    if not held_out_complete:
        missing.append("paired judge-ID/hidden-shift evidence is incomplete")

    regressions: list[str] = []
    for instance_set in report.instance_sets.values():
        cell = instance_set.by_budget.get(f"{report.primary_budget_sec:g}")
        if cell is None:
            continue
        base_success = (
            cell.base.valid_runs / cell.base.total_runs if cell.base.total_runs else 0.0
        )
        agent_success = (
            cell.agent.valid_runs / cell.agent.total_runs
            if cell.agent.total_runs
            else 0.0
        )
        if agent_success - base_success < protocol.minimum_success_rate_delta:
            regressions.append(
                f"independently verified run validity regressed on "
                f"{instance_set.instance_set_kind}"
            )
    primary_ci = paired.mean_gap_reduction_ci95
    if primary_ci is not None and primary_ci.upper < 0:
        regressions.append("judge-ID normalized gap regressed")
    if retention is not None and not retention.retained_on_shift:
        regressions.append("improvement was not retained on hidden shift")

    improvements: list[str] = []
    if (
        primary_ci is not None
        and primary_ci.lower > 0
        and retention is not None
        and retention.retained_on_shift
    ):
        improvements.append("fixed-budget normalized gap with 95% confidence")

    if missing:
        classification = "incomplete"
    elif regressions and improvements:
        classification = "tradeoff"
    elif regressions:
        classification = "regressed"
    elif improvements:
        classification = "improved"
    elif paired.mean_gap_reduction is not None and paired.mean_gap_reduction > 0:
        classification = "inconclusive"
    else:
        classification = "no_change"
    return PerformanceDecision(
        classification=classification,
        validity_complete=validity_accepted
        and not any("validity regressed" in item for item in regressions),
        performance_complete=performance_complete,
        repeatability_complete=repeatability_complete,
        held_out_complete=held_out_complete,
        improvements=improvements,
        regressions=regressions,
        missing_evidence=missing,
    )


__all__ = ["PerformanceDecision", "compute_performance_decision"]
