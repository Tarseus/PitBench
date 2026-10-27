"""Qualification adapters retaining the existing robustness estimators."""

from __future__ import annotations

import statistics
from collections import defaultdict

from pitbench.metrics.controlled_transfer_report import (
    compute_controlled_transfer_report,
)
from pitbench.metrics.nuisance_report import representation_iqr
from pitbench.metrics.performance_report import BOOTSTRAP_SEED, _cluster_mean_ci
from pitbench.metrics.seed_robustness_report import (
    SeedSelectionMetadata,
    _exact_capped_time,
    compute_exact_seed_robustness_report,
    compute_seed_robustness_report,
)
from pitbench.schema.observation import RunStatus


def native_objective_measurement(task, observations, grid, metric, panel) -> dict:
    budget = task.evaluation.primary_budget_sec
    rows = [
        r for r in observations if r.instance_set == panel and r.budget_sec == budget
    ]
    expected = [
        r for r in grid.runs if r.instance_set == panel and r.budget_sec == budget
    ]
    if (
        metric in {"boundary_pass_counts", "operational_failure_counts"}
        and task.optimization_profile.scoring.reliability_scope
        == "declared_boundary_suite"
    ):
        rows = [
            r
            for r in observations
            if r.budget_sec == budget
            and r.test_suite == "operational_reliability"
            and r.instance_set == "boundary_cases"
        ]
        expected = [
            r
            for r in grid.runs
            if r.budget_sec == budget
            and r.test_suite == "operational_reliability"
            and r.instance_set == "boundary_cases"
        ]
    if metric == "controlled_retention":
        if task.evaluation.controlled_transfer is None:
            return {
                "detail": "The task must declare the existing controlled-transfer protocol."
            }
        report = compute_controlled_transfer_report(
            observations, task=task, expected_run_grid=grid
        )
        conditions = [
            r
            for r in report.condition_reports.values()
            if r.reference_condition_id is not None
        ]
        if not conditions or any(
            r.status.value != "complete"
            or r.retention_margin is None
            or r.retention_margin_lower_simultaneous is None
            for r in conditions
        ):
            return {
                "detail": "Every declared transfer condition needs complete native simultaneous retention evidence."
            }
        margin = min(r.retention_margin for r in conditions)
        lower_margin = min(r.retention_margin_lower_simultaneous for r in conditions)
        improved = all(r.supports_retention for r in conditions)
        return {
            "complete": True,
            "degradation": -margin,
            "upper": -lower_margin,
            "lower": None,
            "classification": "improved" if improved else "inconclusive",
            "confidence_level": 0.95,
            "detail": "Native simultaneous controlled-retention decision; no generic replacement statistic.",
        }
    if metric == "seed_dispersion":
        config = task.evaluation.seed_robustness
        measure = task.optimization_profile.scoring.seed_dispersion_measure
        if config is None or measure is None:
            return {
                "detail": "Declare the existing seed protocol and its scored dispersion coordinate."
            }
        seeds = sorted(
            {
                r.solver_seed
                for r in expected
                if r.test_suite is None and r.equivalence_parent_id is None
            }
        )
        args = dict(
            task_id=task.task_id,
            budgets_sec=[budget],
            primary_budget_sec=budget,
            seed_selection=SeedSelectionMetadata.model_validate(
                config.seed_selection.model_dump()
            ),
            development_seeds=config.development_seeds,
            evaluation_seeds=seeds,
        )
        rows = [
            r for r in rows if r.test_suite is None and r.equivalence_parent_id is None
        ]
        if measure == "normalized_gap_iqr":
            native = (
                compute_seed_robustness_report(rows, **args)
                .by_instance_set[panel]
                .primary
            )
            ci, change = (
                native.change.mean_seed_iqr_change_ci99,
                native.change.mean_seed_iqr_change,
            )
            complete = native.paired_complete_instance_count == next(
                s.size for s in task.instance_sets if s.name == panel
            )
        else:
            native = (
                compute_exact_seed_robustness_report(
                    rows, **args, time_basis=task.evaluation.exact_time_basis
                )
                .by_instance_set[panel]
                .primary
            )
            gap = measure == "exact_final_gap_iqr"
            ci = (
                native.change.final_gap_iqr_change_ci99
                if gap
                else native.change.capped_time_iqr_change_ci99
            )
            change = (
                native.change.final_gap_iqr_change
                if gap
                else native.change.capped_time_iqr_change_sec
            )
            field = (
                "complete_final_gap_instance_count"
                if gap
                else "complete_capped_time_instance_count"
            )
            count = next(s.size for s in task.instance_sets if s.name == panel)
            complete = all(
                getattr(state, field) == count for state in (native.base, native.agent)
            )
        if not complete or ci is None or change is None:
            return {
                "detail": "Seed dispersion retains missing runs and requires the full paired seed list."
            }
        return {
            "complete": True,
            "degradation": change,
            "lower": ci.lower,
            "upper": ci.upper,
            "confidence_level": ci.level,
            "classification": "improved"
            if ci.upper < 0
            else "regressed"
            if ci.lower > 0
            else "inconclusive",
            "detail": "Existing paired seed-list bootstrap and 99% interval, with the declared dispersion coordinate.",
        }
    if metric == "representation_dispersion":
        config = task.evaluation.representation_robustness
        if config is None or config.instance_set != panel:
            return {
                "detail": "Declare an equivalence panel for this qualification population."
            }
        rows = [r for r in rows if r.equivalence_parent_id is not None]
        expected = [r for r in expected if r.equivalence_parent_id is not None]
    elif metric == "boundary_pass_counts":
        rows = [r for r in rows if r.test_suite == "operational_reliability"]
        expected = [r for r in expected if r.test_suite == "operational_reliability"]
    else:
        boundary = any(r.test_suite == "operational_reliability" for r in expected)
        rows = [
            r
            for r in rows
            if r.equivalence_parent_id is None
            and (
                r.test_suite == "operational_reliability"
                if boundary
                else r.test_suite is None
            )
        ]
        expected = [
            r
            for r in expected
            if r.equivalence_parent_id is None
            and (
                r.test_suite == "operational_reliability"
                if boundary
                else r.test_suite is None
            )
        ]

    def key(row):
        return (
            row.code_state,
            row.instance_id,
            row.solver_seed,
            row.equivalence_parent_id,
            row.test_suite,
        )

    if (
        not expected
        or len({key(r) for r in rows}) != len(rows)
        or {key(r) for r in rows} != {key(r) for r in expected}
    ):
        return {"detail": "The full declared native metric grid is required."}
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        iid = (
            row.equivalence_parent_id
            if metric == "representation_dispersion"
            else row.instance_id
        )
        if metric == "representation_dispersion":
            if task.evaluation.performance_protocol.value == "exact_verified_solve":
                value = _exact_capped_time(
                    row, budget_sec=budget, time_basis=task.evaluation.exact_time_basis
                )
            else:
                value = (
                    row.normalized_gap
                    if row.valid and row.status == RunStatus.COMPLETED
                    else None
                )
            if value is None:
                return {
                    "detail": "An equivalence run lacks an independently verified native measurement."
                }
        else:
            passed = row.valid and row.status == RunStatus.COMPLETED
            value = int(passed if metric == "boundary_pass_counts" else not passed)
        groups[iid][row.code_state.value].append(value)
    differences = {}
    for iid, states in groups.items():
        if not states["base"] or len(states["base"]) != len(states["agent"]):
            return {"detail": "Incomplete native metric pairs."}
        if metric == "representation_dispersion":
            if len(states["base"]) != config.relabelings_per_instance:
                return {"detail": "The equivalence panel is reduced."}
            b, a = (
                representation_iqr(states["base"]),
                representation_iqr(states["agent"]),
            )
        else:
            b, a = sum(states["base"]), sum(states["agent"])
        differences[(panel, iid)] = [
            b - a if metric == "boundary_pass_counts" else a - b
        ]
    if len(differences) < 2:
        return {
            "detail": "Native qualification requires at least two paired instances."
        }
    ci = _cluster_mean_ci(differences, seed=BOOTSTRAP_SEED)
    return {
        "complete": True,
        "degradation": statistics.fmean(v[0] for v in differences.values()),
        "lower": ci.lower,
        "upper": ci.upper,
        "confidence_level": ci.level,
        "classification": "improved"
        if ci.upper < 0
        else "regressed"
        if ci.lower > 0
        else "inconclusive",
        "detail": "Declared native counts (shared boundary suite when explicitly selected) or Type-7 representation IQR; paired instance bootstrap.",
    }
