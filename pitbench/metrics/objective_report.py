"""Map declared PR objectives to measurements and preregistered qualification."""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, Field

from pitbench.metrics.controlled_transfer_report import ControlledTransferReport
from pitbench.metrics.objective_qualification import (
    ObjectiveQualificationReport,
    compute_objective_qualification,
)
from pitbench.metrics.performance_report import (
    ExactPerformanceReport,
    PerformanceReport,
)
from pitbench.metrics.reliability_report import ReliabilityReport
from pitbench.metrics.resource_report import ResourceReport
from pitbench.metrics.seed_robustness_report import (
    ExactSeedRobustnessReport,
    SeedRobustnessReport,
)
from pitbench.schema.objectives import objective_catalog, objective_catalog_sha256
from pitbench.schema.observation import ExpectedRunGrid, RunObservation
from pitbench.schema.task import PitBenchTask


class ObjectiveMetricEvidence(BaseModel):
    label: str
    unit: str
    direction: str
    current_support: str
    qualification: str
    status: Literal["reported", "partial", "raw_only", "unavailable", "not_applicable"]
    report_reference: str | None = None
    value: Any = None
    detail: str = ""


class ObjectiveMeasurementReport(BaseModel):
    taxonomy_version: str
    taxonomy_sha256: str
    classification_basis: str
    objectives: list[str]
    change_kinds: list[str]
    primary_metric_id: str | None
    candidate_validity_accepted: bool | None = None
    objective_metrics: dict[str, list[str]]
    scenarios: dict[str, str]
    metrics: dict[str, ObjectiveMetricEvidence]
    unavailable_metric_ids: list[str]
    reported_metric_count: int = Field(ge=0)
    partial_metric_count: int = Field(ge=0)
    qualification: ObjectiveQualificationReport | None = None
    automatic_admission: Literal[False] = False
    interpretation: str = "Qualification uses the frozen primary metric, declared guardrails, complete paired evidence and fresh confirmation. Type hints alone never admit a PR; task readiness additionally requires the existing curation gates."


def compute_objective_report(
    *,
    task: PitBenchTask,
    performance: PerformanceReport | ExactPerformanceReport | None = None,
    resources: ResourceReport | None = None,
    seeds: SeedRobustnessReport | ExactSeedRobustnessReport | None = None,
    transfer: ControlledTransferReport | None = None,
    reliability: ReliabilityReport | None = None,
    observations: list[RunObservation] | None = None,
    validity_accepted: bool | None = None,
    expected_run_grid: ExpectedRunGrid | None = None,
    repeat_observations: list[RunObservation] | None = None,
    repeat_expected_run_grid: ExpectedRunGrid | None = None,
    repeat_identity_verified: bool = False,
) -> ObjectiveMeasurementReport | None:
    """Reuse native report values; leave unsupported scenarios explicitly absent."""
    profile = task.optimization_profile
    if profile is None:
        return None
    if any(item.task_id != task.task_id for item in observations or []):
        raise ValueError("objective report contains observations from another task")
    for native in (resources, seeds, transfer):
        if native is not None and native.task_id != task.task_id:
            raise ValueError(
                "objective report contains a native report from another task"
            )
    catalog = objective_catalog()
    requested = dict.fromkeys(
        metric
        for objective in profile.objectives
        for metric in catalog["objectives"][objective]["metric_ids"]
    )
    metrics = {}
    original = [
        item
        for item in observations or []
        if item.equivalence_parent_id is None and item.test_suite is None
    ]
    for metric_id in requested:
        definition = catalog["metrics"][metric_id]
        status, reference, value, detail = (
            "unavailable",
            None,
            None,
            definition["evidence"],
        )
        if metric_id in {"exact_solved_coverage", "exact_penalized_runtime_sec"}:
            if task.evaluation.performance_protocol.value != "exact_verified_solve":
                status, detail = (
                    "not_applicable",
                    "Task uses a fixed-budget heuristic protocol.",
                )
            elif isinstance(performance, ExactPerformanceReport):
                field = (
                    "verified_solved_coverage"
                    if metric_id == "exact_solved_coverage"
                    else "penalized_average_runtime_sec"
                )
                value = {
                    "base": getattr(performance.primary.base, field),
                    "agent": getattr(performance.primary.agent, field),
                    "paired": performance.primary.paired.model_dump(mode="json"),
                    "native_classification": performance.classification.value,
                    "confirmation": performance.confirmation.model_dump(mode="json")
                    if performance.confirmation
                    else None,
                }
                status = (
                    "reported"
                    if performance.primary.base.complete
                    and performance.primary.agent.complete
                    and performance.primary.paired.complete
                    else "partial"
                )
                reference, detail = (
                    "performance.primary",
                    "Uses the existing coverage-first exact protocol and its native confirmation.",
                )
        elif metric_id == "fixed_budget_normalized_gap":
            if task.evaluation.performance_protocol.value != "heuristic_fixed_budget":
                status, detail = "not_applicable", "Task uses an exact solve protocol."
            elif isinstance(performance, PerformanceReport):
                value = performance.primary.model_dump(mode="json")
                status, reference, detail = (
                    "partial",
                    "performance.primary",
                    "Native paired-gap report; full expected-grid and validity checks remain separate.",
                )
        elif (
            metric_id in {"peak_rss_ratio", "cpu_time_ratio"} and resources is not None
        ):
            field = "memory" if metric_id == "peak_rss_ratio" else "cpu"
            value, complete = (
                {},
                set(resources.by_instance_set)
                == {item.name for item in task.instance_sets},
            )
            for instance_set, budgets in resources.by_instance_set.items():
                complete = complete and set(budgets) == {
                    format(budget, "g") for budget in task.evaluation.budgets_sec
                }
                value[instance_set] = {}
                for budget, cell in budgets.items():
                    measured = getattr(cell, field)
                    value[instance_set][budget] = {
                        **measured.model_dump(mode="json"),
                        "expected_instance_count": cell.expected_instance_count,
                        "expected_seed_count": resources.expected_seed_count,
                    }
                    complete = (
                        complete
                        and measured.complete_instance_count
                        == cell.expected_instance_count
                        and measured.paired_seed_count
                        == cell.expected_instance_count * resources.expected_seed_count
                    )
            status, reference = (
                ("reported" if complete and value else "partial"),
                "resource_usage.by_instance_set",
            )
            detail = "Existing paired ratios; preserve measurement-scope exclusions and incomplete pairs. No resource qualification threshold is inferred."
        elif metric_id == "seed_dispersion" and seeds is not None:
            value = seeds.model_dump(mode="json")
            status, reference, detail = (
                "partial",
                "nuisance_robustness",
                "Existing seed protocol; inspect complete-instance counts. Cross-seed dispersion does not establish identical-seed determinism.",
            )
        elif metric_id == "controlled_retention" and transfer is not None:
            value = transfer.model_dump(mode="json")
            status, reference, detail = (
                "reported",
                "controlled_transfer",
                "Existing controlled-transfer protocol; per-condition completeness and support decisions remain in the native report.",
            )
        elif (
            metric_id in {"boundary_pass_counts", "operational_failure_counts"}
            and reliability is not None
        ):
            value = reliability.model_dump(mode="json")
            status = "reported" if reliability.complete else "partial"
            reference, detail = (
                "operational_reliability",
                "Counts on the existing declared boundary grid; no new reliability admission threshold.",
            )
        elif metric_id == "search_work" and original:
            value = {
                "observed_runs": len(original),
                "runs_with_nodes": sum(item.nodes is not None for item in original),
                "runs_with_iterations": sum(
                    item.iterations is not None for item in original
                ),
            }
            status, reference, detail = (
                "raw_only",
                "observations.{nodes,iterations}",
                "Counters remain diagnostics pending unit and algorithm comparability checks.",
            )
        elif metric_id == "operational_failure_counts" and original:
            value = {
                state: dict(
                    Counter(
                        item.status.value
                        for item in original
                        if item.code_state.value == state
                    )
                )
                for state in ("base", "agent")
            }
            status, reference, detail = (
                "raw_only",
                "observations.status",
                "Observed status counts only; no boundary-grid completeness or failure-rate claim.",
            )
        metrics[metric_id] = ObjectiveMetricEvidence(
            **{
                key: definition[key]
                for key in (
                    "label",
                    "unit",
                    "direction",
                    "current_support",
                    "qualification",
                )
            },
            status=status,
            report_reference=reference,
            value=value,
            detail=detail,
        )
    return ObjectiveMeasurementReport(
        taxonomy_version=catalog["taxonomy_version"],
        taxonomy_sha256=objective_catalog_sha256(),
        classification_basis=profile.classification_basis,
        objectives=profile.objectives,
        change_kinds=profile.change_kinds,
        primary_metric_id=profile.primary_metric_id,
        candidate_validity_accepted=validity_accepted,
        objective_metrics={
            key: catalog["objectives"][key]["metric_ids"] for key in profile.objectives
        },
        scenarios={
            key: catalog["objectives"][key]["scenario"] for key in profile.objectives
        },
        metrics=metrics,
        unavailable_metric_ids=[
            key for key, item in metrics.items() if item.status == "unavailable"
        ],
        reported_metric_count=sum(
            item.status == "reported" for item in metrics.values()
        ),
        partial_metric_count=sum(item.status == "partial" for item in metrics.values()),
        qualification=compute_objective_qualification(
            task=task,
            observations=observations or [],
            expected_run_grid=expected_run_grid,
            repeat_observations=repeat_observations,
            repeat_expected_run_grid=repeat_expected_run_grid,
            repeat_identity_verified=repeat_identity_verified,
            validity_accepted=validity_accepted,
        ),
    )


def format_objective_report(report: ObjectiveMeasurementReport) -> str:
    lines = [
        "PR optimization objectives and measurement coverage",
        f"Classification basis: {report.classification_basis}; primary metric: {report.primary_metric_id or 'not selected'}.",
    ]
    for key, metric_ids in report.objective_metrics.items():
        evidence = ", ".join(
            f"{metric}: {report.metrics[metric].status}" for metric in metric_ids
        )
        lines.append(f"{key} / {report.scenarios[key]}: {evidence}")
    if report.qualification:
        lines.append(
            f"Objective qualification: {report.qualification.classification}; confirmed: {report.qualification.confirmed}."
        )
        lines.extend(report.qualification.reasons)
    lines.append(
        "Missing or partial evidence is not a zero gain or automatic rejection. Task readiness also requires the source, build, checker and integration gates."
    )
    return "\n".join(lines)
