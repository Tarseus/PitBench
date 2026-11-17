"""Campaign reports and figures from retained native evaluation evidence.

Stored estimates remain unchanged. Supplemental hidden-panel summaries reuse
the native estimator on complete retained grids, without executing a solver.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Any

from jinja2 import Environment, select_autoescape
from pydantic import ValidationError

from pitbench.evaluator.storage import ObservationStore
from pitbench.harness.utils.agent_trace import read_trace
from pitbench.harness.utils.artifact_storage import (
    file_identity as artifact_file_identity,
)
from pitbench.metrics.objective_report import compute_objective_report
from pitbench.metrics.performance_report import (
    _run_key,
    compute_panel_performance_reports,
)
from pitbench.schema.evaluation import EvaluationSummary
from pitbench.schema.objectives import objective_catalog
from pitbench.schema.observation import ExpectedRunGrid
from pitbench.schema.task import PitBenchTask


def require_plotting() -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "Automatic report figures require the reports extra: uv sync --extra reports"
        ) from error


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _condition_index(source: Path, runs_root: Path, index: Path | None) -> list[dict]:
    snapshot_path = index if index is not None else source / "model-runs.json"
    snapshot = _read(snapshot_path) if snapshot_path.is_file() else []
    snapshot = snapshot["conditions"] if isinstance(snapshot, dict) else snapshot
    cohort_path = source / "cohort.json"
    study_statuses = sorted((source / "agent-status").glob("*/status.json"))
    if cohort_path.is_file() or study_statuses:
        planned = _read(cohort_path)["jobs"] if cohort_path.is_file() else snapshot
        if not planned:
            planned = [
                {**_read(path), "condition_id": path.parent.name}
                for path in study_statuses
            ]
        saved = {row.get("condition_id", row.get("run_id")): row for row in snapshot}
        records = []
        for job in planned:
            identity = job.get("condition_id", job.get("run_id"))
            state = {**job, **saved.get(identity, {})}
            path = (
                source
                / "agent-status"
                / str(job.get("run_id", identity))
                / "status.json"
            )
            if path.is_file():
                state.update(_read(path))
            state.setdefault("status", "not_started")
            model = state.get("model")
            records.append(
                {
                    **state,
                    "condition_id": identity,
                    "model": model.get("model") if isinstance(model, dict) else model,
                    "setting": model.get("setting", "unspecified")
                    if isinstance(model, dict)
                    else state.get("setting", "unspecified"),
                }
            )
        return records
    if index is not None:
        return snapshot
    statuses = sorted((source / "model-status").glob("*/status.json"))
    if statuses:
        return [{**_read(path), "condition_id": path.parent.name} for path in statuses]
    if (source / "model-runs.json").is_file():
        return _read(source / "model-runs.json")
    results = (
        [runs_root / "results.json"]
        if (runs_root / "results.json").is_file()
        else sorted(runs_root.glob("*/results.json"))
    )
    return [
        {
            "condition_id": f"{path.parent.name}/{trial['trial_name']}",
            "run_id": path.parent.name,
            "task_id": trial["task_id"],
            "trial_name": trial["trial_name"],
            "results_path": str(path),
        }
        for path in results
        for trial in _read(path).get("results", [])
    ]


def _result_path(source: Path, runs_root: Path, record: dict) -> Path:
    if record.get("results_path"):
        return Path(record["results_path"])
    execution = record.get("execution_run_id") or record["run_id"]
    if record.get("execution_run_id"):
        # A failed original trial must not hide its retained infrastructure retry.
        return runs_root / execution / "results.json"
    candidates = [
        source / record.get("condition_id", record["run_id"]) / "results.json",
        runs_root / record["run_id"] / "results.json",
        source / record["run_id"] / "results.json",
    ]
    return next((path for path in candidates if path.is_file()), candidates[1])


def retained_conditions(source: Path, runs_root: Path) -> list[dict]:
    """Common complete condition selection for reporting, diagnostics and export."""
    indexed = _condition_index(source, runs_root, None)
    audit_path = source / "final-model-audit-summary.json"
    if not audit_path.is_file():
        return indexed
    audit = _read(audit_path).get("conditions", [])
    explicit = (source / "cohort.json").is_file() or (
        source / "model-runs.json"
    ).is_file()
    records, additions = (list(indexed), audit) if explicit else (list(audit), indexed)
    for row in additions:
        identity = (
            row.get("execution_run_id") or row.get("run_id") or row.get("condition_id")
        )
        associated = False
        for known in records:
            names = {
                known.get("condition_id"),
                known.get("run_id"),
                known.get("execution_run_id"),
            } - {None}
            if identity in names or any(
                str(identity).startswith(str(name) + "-infra") for name in names
            ):
                associated = True
                break
        if not associated:
            records.append(row)
    return records


def _point(
    row: dict,
    *,
    metric: str,
    title: str,
    unit: str,
    estimate,
    interval: dict | None,
    reference: str,
    panel: str = "judge_id",
    scale: float = 1.0,
    reduction: bool = False,
    confirmed: bool = False,
    budget_sec: float | None = None,
) -> dict:
    lower, upper = (
        (interval.get("lower"), interval.get("upper")) if interval else (None, None)
    )
    if reduction:
        estimate = -estimate if estimate is not None else None
        lower, upper = (
            -upper if upper is not None else None,
            -lower if lower is not None else None,
        )
    values = [estimate, lower, upper]
    finite = all(
        isinstance(value, (int, float)) and math.isfinite(value) for value in values
    )
    level = interval.get("level", 0.95) if interval else None
    valid = (
        finite and lower <= upper and isinstance(level, (int, float)) and 0 < level < 1
    )
    if estimate is not None and not valid:
        row["issues"].append(
            f"{metric}/{panel}: no complete finite saved confidence interval"
        )
    return {
        "condition_id": row["condition_id"],
        "task_id": row["task_id"],
        "setting": row["setting"],
        "model": row["model"],
        "budget_sec": budget_sec
        if budget_sec is not None
        else row.get("primary_budget_sec"),
        "metric_id": metric,
        "title": title,
        "unit": unit,
        "panel": panel,
        "estimate": estimate * scale
        if isinstance(estimate, (int, float)) and math.isfinite(estimate)
        else None,
        "lower": lower * scale
        if isinstance(lower, (int, float)) and math.isfinite(lower)
        else None,
        "upper": upper * scale
        if isinstance(upper, (int, float)) and math.isfinite(upper)
        else None,
        "confidence_level": level,
        "classification": row.get("classification"),
        "confirmed": confirmed,
        "plot_ready": bool(valid),
        "source_field": reference,
    }


def _effects(row: dict, summary: dict) -> list[dict]:
    points = []
    performance = summary.get("performance")
    if performance:
        primary = performance["primary"]
        paired = primary["paired"]
        confirmed = bool((performance.get("confirmation") or {}).get("confirmed"))
        row.update(
            classification=performance["classification"],
            primary_budget_sec=performance["primary_budget_sec"],
            confirmed=confirmed,
        )
        if "time_basis" in performance:
            difference = (
                primary["agent"]["verified_solved_coverage"]
                - primary["base"]["verified_solved_coverage"]
                if primary["agent"]["verified_solved_coverage"] is not None
                and primary["base"]["verified_solved_coverage"] is not None
                else None
            )
            if difference == 0:
                metric, title, unit, scale = (
                    "exact_penalized_runtime_sec",
                    f"PAR-2 runtime reduction ({performance['time_basis']})",
                    "seconds",
                    1,
                )
                value, interval = (
                    paired["mean_penalized_runtime_reduction_sec"],
                    paired["mean_penalized_runtime_reduction_ci95"],
                )
            else:
                metric, title, unit, scale = (
                    "exact_solved_coverage",
                    "Verified solved coverage gain",
                    "percentage points",
                    100,
                )
                value, interval = (
                    paired["mean_verified_solved_coverage_reduction"],
                    paired["mean_verified_solved_coverage_reduction_ci95"],
                )
        else:
            metric, title, unit, scale = (
                "fixed_budget_normalized_gap",
                "Fixed-budget normalized gap reduction",
                "percentage points",
                100,
            )
            value, interval = (
                paired["mean_gap_reduction"],
                paired["mean_gap_reduction_ci95"],
            )
        points.append(
            _point(
                row,
                metric=metric,
                title=title,
                unit=unit,
                estimate=value,
                interval=interval,
                scale=scale,
                reference="summary.performance.primary.paired",
                confirmed=confirmed,
            )
        )
        row["primary_effect"] = points[-1]
    seed = summary.get("nuisance_robustness")
    for panel, panel_performance in row.get(
        "supplementary_panel_performance", {}
    ).items():
        pair = panel_performance["primary"]["paired"]
        if "time_basis" in panel_performance:
            coverage = pair["mean_verified_solved_coverage_reduction"]
            if coverage == 0:
                metric, title, unit, scale = (
                    "exact_penalized_runtime_sec", "PAR-2 runtime reduction", "seconds", 1
                )
                value = pair["mean_penalized_runtime_reduction_sec"]
                interval = pair["mean_penalized_runtime_reduction_ci95"]
            else:
                metric, title, unit, scale = (
                    "exact_solved_coverage", "Verified solved coverage gain", "percentage points", 100
                )
                value = coverage
                interval = pair["mean_verified_solved_coverage_reduction_ci95"]
        else:
            metric, title, unit, scale = (
                "supplementary_shift_gap", "Hidden-shift gap reduction (supplementary)", "percentage points", 100
            )
            value = pair["mean_gap_reduction"]
            interval = pair["mean_gap_reduction_ci95"]
        points.append(
            _point(
                row,
                metric=metric,
                title=title,
                unit=unit,
                estimate=value,
                interval=interval,
                scale=scale,
                panel=panel,
                budget_sec=panel_performance["primary_budget_sec"],
                reference=f"supplementary_panel_performance.{panel}.primary.paired",
            )
        )
    if seed:
        for panel, panel_report in seed["by_instance_set"].items():
            if panel_report["instance_set_kind"] == "agent_dev":
                continue
            primary = panel_report["primary"]
            exact = seed["metric"] == "exact_seed_robustness"
            key = "capped_time_iqr_change_sec" if exact else "mean_seed_iqr_change"
            ci_key = (
                "capped_time_iqr_change_ci99" if exact else "mean_seed_iqr_change_ci99"
            )
            points.append(
                _point(
                    row,
                    metric="seed_dispersion_time" if exact else "seed_dispersion_gap",
                    title="Cross-seed capped-time IQR reduction"
                    if exact
                    else "Cross-seed gap IQR reduction",
                    unit="seconds" if exact else "percentage points",
                    estimate=primary["change"][key],
                    interval=primary["change"][ci_key],
                    reduction=True,
                    scale=1 if exact else 100,
                    panel=panel,
                    budget_sec=seed["primary_budget_sec"],
                    reference=f"summary.nuisance_robustness.by_instance_set.{panel}.primary.change",
                )
            )
    transfer = summary.get("controlled_transfer")
    if transfer:
        for condition in transfer["condition_reports"].values():
            if not condition.get("reference_condition_id"):
                continue
            # Exact native benefit divides PAR-2 reduction by solve budget.
            # It is not the retention ratio or its simultaneous support bound.
            exact = performance and "time_basis" in performance
            points.append(
                _point(
                    row,
                    metric="transfer_exact_benefit"
                    if exact
                    else "transfer_gap_benefit",
                    title="Structural-shift normalized PAR-2 reduction"
                    if exact
                    else "Structural-shift gap reduction",
                    unit="fraction of solve budget" if exact else "percentage points",
                    estimate=condition["benefit"],
                    interval=condition["benefit_ci95"],
                    scale=1 if exact else 100,
                    panel=condition["condition_id"],
                    budget_sec=condition["budget_sec"],
                    reference=f"summary.controlled_transfer.condition_reports.{condition['condition_id']}",
                )
            )
    for measured in (summary.get("metric_scenarios") or {}).get("results", []):
        if measured.get("degradation") is None:
            continue
        definition = objective_catalog()["metrics"][measured["metric_id"]]
        points.append(
            _point(
                row,
                metric=measured["metric_id"],
                title=measured["metric_id"],
                unit=definition["unit"],
                estimate=measured["degradation"],
                interval={
                    "lower": measured["lower"],
                    "upper": measured["upper"],
                    "level": measured["confidence_level"],
                },
                reduction=True,
                panel=measured["instance_set"],
                budget_sec=row.get("scenario_budgets", {}).get(measured["scenario_id"]),
                reference=f"summary.metric_scenarios.{measured['scenario_id']}",
            )
        )
    return points


def _observed_scale_slices(row, task, grid, selected):
    """Describe size strata within one already complete heuristic shift panel."""
    scales_by_instance = defaultdict(set)
    for observation in selected:
        scale = observation.problem_scale
        if scale is None or not math.isfinite(scale) or scale <= 0:
            return
        scales_by_instance[observation.instance_id].add(scale)
    if any(len(scales) != 1 for scales in scales_by_instance.values()):
        row["issues"].append(
            "inconsistent recorded problem sizes; no size curve produced"
        )
        return
    scales = {next(iter(values)) for values in scales_by_instance.values()}
    if len(scales) < 2:
        return
    for scale in sorted(scales):
        observations = [r for r in selected if r.problem_scale == scale]
        identifiers = {r.instance_id for r in observations}
        subgrid = grid.model_copy(
            update={
                "runs": [
                    r
                    for r in grid.runs
                    if r.instance_set == observations[0].instance_set
                    and r.instance_id in identifiers
                ]
            }
        )
        reports = compute_panel_performance_reports(
            observations, task=task, expected_run_grid=subgrid
        )
        for panel, report in reports.items():
            for cell in report.by_budget.values():
                row.setdefault("observed_scale_slices", []).append(
                    {
                        "panel": panel,
                        "scale": scale,
                        "budget_sec": cell.budget_sec,
                        "instance_count": len(identifiers),
                        "seed_count": len({r.solver_seed for r in observations}),
                        "base_gap": cell.base.mean_normalized_gap,
                        "agent_gap": cell.agent.mean_normalized_gap,
                        "gap_reduction": cell.paired.mean_gap_reduction,
                        "base_ci95": cell.base.mean_ci95.model_dump()
                        if len(identifiers) > 1 and cell.base.mean_ci95
                        else None,
                        "agent_ci95": cell.agent.mean_ci95.model_dump()
                        if len(identifiers) > 1 and cell.agent.mean_ci95
                        else None,
                        "reduction_ci95": cell.paired.mean_gap_reduction_ci95.model_dump()
                        if len(identifiers) > 1 and cell.paired.mean_gap_reduction_ci95
                        else None,
                        "axis_label": "Customer count"
                        if task.problem_family.value == "cvrp"
                        else "Recorded problem size",
                        "interpretation": "Exploratory size strata within this fixed hidden-shift panel; no shift-severity ordering or formal scalability score. Single-instance strata have no across-instance interval.",
                    }
                )


def _inventory(row: dict, trial: dict, evaluation_dir: Path, task_path: Path) -> None:
    if not task_path.is_file():
        row["issues"].append(
            "frozen task configuration unavailable; catalog coverage cannot be verified"
        )
        return
    artifacts = (trial.get("evaluation") or {}).get("payload", {}).get("artifacts", {})
    grid_path = evaluation_dir / (artifacts.get("expected_run_grid") or {}).get(
        "path", "expected-run-grid.json"
    )
    measurement_root = grid_path.parent
    checkpoint = measurement_root / "judge-checkpoint.json"
    if checkpoint.is_file():
        expected_hash = _read(checkpoint).get("identity", {}).get("task_config_sha256")
        if (
            expected_hash
            and expected_hash != hashlib.sha256(task_path.read_bytes()).hexdigest()
        ):
            row["issues"].append(
                "task configuration differs from the saved Judge identity; coverage inventory unavailable"
            )
            return
    task = PitBenchTask.from_yaml(task_path)
    if task.task_id != row["task_id"]:
        raise ValueError("report task configuration belongs to another task")
    summary = EvaluationSummary.model_validate(row["native_summary"])
    observations_path = evaluation_dir / (artifacts.get("observations") or {}).get(
        "path", "trials.parquet"
    )
    observations = (
        ObservationStore.read(observations_path)
        if observations_path.is_file()
        else None
    )
    if (
        observations is not None
        and grid_path.is_file()
    ):
        grid = ExpectedRunGrid.model_validate_json(grid_path.read_text())
        for panel in task.instance_sets:
            if panel.kind.value != "judge_shift":
                continue
            protocol = panel.performance_protocol or task.evaluation.performance_protocol
            expected = [
                r
                for r in grid.runs
                if r.instance_set == panel.name
                and r.equivalence_parent_id is None
                and r.test_suite is None
            ]
            selected = [
                r
                for r in observations
                if r.instance_set == panel.name
                and r.equivalence_parent_id is None
                and r.test_suite is None
            ]
            seed_count = (
                panel.solver_seed_count
                if panel.solver_seed_count
                else task.evaluation.seed_robustness.seed_selection.seed_count
                if task.evaluation.seed_robustness is not None
                else len(task.evaluation.solver_seeds or [])
            )
            budgets = panel.budgets_sec or task.evaluation.budgets_sec
            full_count = panel.size * seed_count * len(budgets) * 2
            if (
                not expected
                or len(expected) != full_count
                or len({r.instance_id for r in expected}) != panel.size
                or len({r.solver_seed for r in expected}) != seed_count
                or {r.budget_sec for r in expected} != set(budgets)
                or {r.code_state.value for r in expected} != {"base", "agent"}
                or len(selected) != len(expected)
                or len({_run_key(r) for r in expected}) != len(expected)
                or {_run_key(r) for r in selected} != {_run_key(r) for r in expected}
            ):
                row["issues"].append(
                    f"{panel.name}: incomplete supplemental panel grid; no interval inferred"
                )
                continue
            unavailable = [
                r for r in selected
                if not r.valid or r.status.value != "completed"
                or r.normalized_gap is None or not math.isfinite(r.normalized_gap)
            ] if protocol.value != "exact_verified_solve" else []
            if unavailable:
                counts = Counter((r.code_state.value, r.status.value) for r in unavailable)
                row.setdefault("supplementary_panel_diagnostics", {})[panel.name] = {
                    "expected_runs": len(expected),
                    "observed_runs": len(selected),
                    "unavailable_gap_runs": len(unavailable),
                    "unavailable_by_state_and_status": [
                        {"code_state": state, "status": status, "count": count}
                        for (state, status), count in sorted(counts.items())
                    ],
                }
                row["issues"].append(
                    f"{panel.name}: observed grid is complete, but {len(unavailable)} runs lack a valid gap; no interval inferred"
                )
                continue
            row.setdefault("supplementary_panel_performance", {}).update(
                {
                    name: report.model_dump(mode="json")
                    for name, report in compute_panel_performance_reports(
                        selected, task=task, expected_run_grid=grid
                    ).items()
                }
            )
            if protocol.value != "exact_verified_solve":
                _observed_scale_slices(row, task, grid, selected)
    representation_ref = artifacts.get("representation_robustness_details")
    representation_path = (
        evaluation_dir / representation_ref["path"]
        if representation_ref
        else measurement_root / "representation/details.json"
    )
    representation = None
    if representation_path.is_file():
        details = _read(representation_path)
        patch_ref = artifacts.get("candidate_patch")
        patch_path = (
            evaluation_dir / patch_ref["path"]
            if patch_ref
            else measurement_root / "candidate.patch"
        )
        if (
            patch_path.is_file()
            and details.get("candidate_patch_sha256")
            != hashlib.sha256(patch_path.read_bytes()).hexdigest()
        ):
            raise ValueError("representation report belongs to another candidate")
        representation = details.get("representation_robustness")
    inventory = compute_objective_report(
        task=task,
        performance=summary.performance,
        resources=summary.resource_usage,
        seeds=summary.nuisance_robustness,
        transfer=summary.controlled_transfer,
        reliability=summary.operational_reliability,
        scenario_report=summary.metric_scenarios,
        observations=observations,
        representation=representation,
        include_all_metrics=True,
        include_qualification=False,
    )
    row.update(
        metric_inventory=inventory.model_dump(mode="json"),
        representation_robustness=representation,
    )
    row["source_commit"] = task.release.base_commit
    row["task_config"] = str(task_path)
    row["scenario_budgets"] = {
        scenario.id: scenario.budget_sec for scenario in task.metric_scenarios
    }
    if observations is None:
        row["issues"].append(
            "raw observations unavailable locally; saved native estimates retained"
        )


def _summary_values(row: dict) -> None:
    """Expose native tradeoffs and eligible-pair counts beside performance."""
    summary = row["native_summary"]
    baseline = summary.get("base_reuse") or {}
    row["baseline_observations_sha256"] = baseline.get("observations_sha256")
    row["shared_baseline"] = bool(baseline.get("reused"))
    resource = summary.get("resource_usage") or {}
    budget = row.get("primary_budget_sec", resource.get("primary_budget_sec"))
    row["primary_resources"] = {}
    if budget is not None:
        cell = (
            resource.get("by_instance_set", {})
            .get("judge_id", {})
            .get(f"{budget:g}", {})
        )
        for metric in ("cpu", "memory"):
            if metric in cell:
                estimate = cell[metric]
                ratio = estimate.get("geometric_mean_ratio")
                row["primary_resources"][metric] = {
                    **estimate,
                    "change_percent": (ratio - 1) * 100 if ratio is not None else None,
                    "expected_pairs": cell["expected_instance_count"]
                    * resource["expected_seed_count"],
                }
    seed = summary.get("nuisance_robustness") or {}
    seed_cell = seed.get("by_instance_set", {}).get("judge_id", {}).get("primary")
    if seed_cell:
        exact = seed["metric"] == "exact_seed_robustness"
        field = "mean_capped_time_iqr_sec" if exact else "mean_seed_iqr"
        scale = 1 if exact else 100
        row["seed_iqr"] = {
            state: seed_cell[state][field] * scale
            if seed_cell[state][field] is not None
            else None
            for state in ("base", "agent")
        }
        row["seed_iqr"]["unit"] = "seconds" if exact else "percentage points"
    representation = row.get("representation_robustness")
    if representation and budget is not None:
        cell = representation["by_budget"].get(f"{budget:g}")
        if cell:
            exact = "mean_capped_time_iqr_sec" in cell["base"]
            field, scale = (
                ("mean_capped_time_iqr_sec", 1) if exact else ("mean_iqr", 100)
            )
            row["representation_iqr"] = {
                state: cell[state][field] * scale
                if cell[state][field] is not None
                else None
                for state in ("base", "agent")
            }
            row["representation_iqr"]["unit"] = (
                "seconds" if exact else "percentage points"
            )
    transfer = summary.get("controlled_transfer")
    if transfer:
        conditions = [
            c
            for c in transfer["condition_reports"].values()
            if c.get("reference_condition_id")
        ]
        row["retention"] = {
            "supported": sum(c["supports_retention"] for c in conditions),
            "conditions": len(conditions),
            "required_fraction": transfer["retention_fraction"],
            "native_classifications": [c["native_classification"] for c in conditions],
        }


def collect_campaign(
    source: Path,
    *,
    repository_root: Path,
    index: Path | None = None,
    runs_root: Path | None = None,
    task_configs: dict[str, Path] | None = None,
) -> dict:
    source, repository_root = source.resolve(), repository_root.resolve()
    runs_root = (
        runs_root or (source / "runs" if (source / "runs").is_dir() else source)
    ).resolve()
    records = _condition_index(source, runs_root, index)
    conditions, estimates = [], []
    for record in records:
        result_path = _result_path(source, runs_root, record)
        identifier = record.get(
            "condition_id", record.get("run_id", result_path.parent.name)
        )
        row = {
            "condition_id": identifier,
            "run_id": record.get("run_id"),
            "task_id": record["task_id"],
            "model": record.get("model"),
            "setting": record.get("setting", "unspecified"),
            "execution_status": record.get("status", record.get("outcome", "unknown")),
            "evaluation_status": "missing",
            "classification": None,
            "confirmed": False,
            "error": record.get("evaluation_error")
            or record.get("error")
            or record.get("detail")
            or (record.get("audit") or {}).get("reason"),
            "issues": [],
            "result_path": str(result_path),
            "native_summary": {},
            "metric_inventory": None,
        }
        if not result_path.is_file():
            row["issues"].append("result file unavailable")
            conditions.append(row)
            continue
        try:
            payload = _read(result_path)
        except (OSError, json.JSONDecodeError) as error:
            row["error"] = f"{type(error).__name__}: {error}"
            conditions.append(row)
            continue
        trials = [
            trial
            for trial in payload.get("results", [])
            if trial.get("task_id") == row["task_id"]
            and (
                not record.get("trial_name")
                or trial.get("trial_name") == record["trial_name"]
            )
        ]
        if not trials:
            row["issues"].append("no matching trial result")
            conditions.append(row)
        for trial in trials:
            current = {**row, "issues": [], "trial_name": trial["trial_name"]}
            if len(trials) > 1:
                current["condition_id"] += "/" + trial["trial_name"]
            envelope = trial.get("evaluation") or {}
            if "evaluation_output" in record:
                attempt_output = record.get("evaluation_output")
                attempt_envelope = (
                    Path(attempt_output) / "evaluation-envelope.json"
                    if attempt_output
                    else None
                )
                envelope = (
                    _read(attempt_envelope)
                    if attempt_envelope and attempt_envelope.is_file()
                    else {}
                )
            evaluation_dir = result_path.parent / "evaluation"
            if record.get("evaluation_output"):
                evaluation_dir = Path(record["evaluation_output"])
            elif (record.get("audit") or {}).get("candidate_patch"):
                evaluation_dir = Path(record["audit"]["candidate_patch"]).parent
            elif not evaluation_dir.is_dir():
                evaluation_dir = (
                    result_path.parent
                    / row["task_id"]
                    / trial["trial_name"]
                    / "evaluation"
                )
            summary = (envelope.get("payload") or {}).get("summary") or {}
            current.update(
                evaluation_status=(
                    "completed" if envelope.get("completed") else "failed"
                )
                if envelope
                else "missing",
                native_summary=summary,
                validity_accepted=(envelope.get("payload") or {})
                .get("validity", {})
                .get("accepted"),
                error=envelope.get("error") or row["error"],
                failure_mode=trial.get("failure_mode"),
                agent_started_at=trial.get("agent_started_at"),
                agent_ended_at=trial.get("agent_ended_at"),
            )
            current["results_sha256"] = hashlib.sha256(
                result_path.read_bytes()
            ).hexdigest()
            task_path = result_path.parent / "task.yaml"
            if not task_path.is_file():
                task_path = (task_configs or {}).get(
                    row["task_id"],
                    repository_root / "configs/tasks" / f"{row['task_id']}.yaml",
                )
            if summary:
                try:
                    _inventory(current, trial, evaluation_dir, task_path)
                except (OSError, ValueError, KeyError, ValidationError) as error:
                    current["issues"].append(
                        f"native report unavailable: {type(error).__name__}: {error}"
                    )
                try:
                    estimates.extend(_effects(current, summary))
                except (ValueError, KeyError, TypeError) as error:
                    current["issues"].append(
                        f"effect estimates unavailable: {type(error).__name__}: {error}"
                    )
            else:
                current["issues"].append("native evaluation summary unavailable")
            try:
                _summary_values(current)
            except (KeyError, TypeError, ValueError) as error:
                current["issues"].append(
                    f"descriptive summaries unavailable: {type(error).__name__}: {error}"
                )
            conditions.append(current)
    return {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "planned_conditions": len(records),
        "conditions": conditions,
        "estimates": estimates,
        "summary": {
            "trial_records": len(conditions),
            "evaluated": sum(r["evaluation_status"] == "completed" for r in conditions),
            "missing_or_failed": sum(
                r["evaluation_status"] != "completed" for r in conditions
            ),
            "native_classifications": dict(
                Counter(r["classification"] or "unavailable" for r in conditions)
            ),
            "confirmed_improved": sum(
                r["classification"] == "improved"
                and r["confirmed"]
                and r.get("validity_accepted") is True
                for r in conditions
            ),
            "valid_improved": sum(
                r["classification"] == "improved" and r.get("validity_accepted") is True
                for r in conditions
            ),
        },
        "interpretation": "Saved native primary intervals remain unchanged. Supplemental hidden-shift plots reuse the native gap estimator on complete retained raw grids and are labeled separately; they do not create a retention protocol. No solver execution. Exact primary plots follow the frozen coverage-first/runtime-tie rule. Conditions are not pooled as independent repetitions. Resource ratios without saved intervals remain descriptive. Transfer benefit intervals do not replace simultaneous retention decisions.",
    }


def _save_figure(figure, output: Path, stem: str) -> dict:
    from matplotlib import rc_context

    assets = {}
    with rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
        for extension in ("svg", "png", "pdf"):
            path = output / f"{stem}.{extension}"
            figure.savefig(
                path, format=extension, dpi=180, bbox_inches="tight", facecolor="white"
            )
            assets[extension] = path.name
    return assets


def _figures(estimates: list[dict], output: Path) -> list[dict]:
    require_plotting()
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    groups = defaultdict(list)
    for point in estimates:
        if point["plot_ready"]:
            groups[
                (
                    point["metric_id"],
                    point["title"],
                    point["unit"],
                    point["panel"],
                    point["confidence_level"],
                )
            ].append(point)
    figures = []
    for (metric, title, unit, panel, level), points in sorted(groups.items()):
        points.sort(key=lambda p: (p["task_id"], p["setting"], p["condition_id"]))
        fig = Figure(figsize=(11, max(3.2, 0.42 * len(points) + 1.8)))
        FigureCanvasAgg(fig)
        axis = fig.add_subplot()
        labels = []
        models = {point["model"] for point in points if point["model"]}
        for y, point in enumerate(points):
            colour = "#235b78" if point["confirmed"] else "#66717a"
            axis.hlines(y, point["lower"], point["upper"], color=colour, linewidth=1.3)
            axis.vlines(
                [point["lower"], point["upper"]],
                y - 0.10,
                y + 0.10,
                color=colour,
                linewidth=1,
            )
            axis.scatter(
                [point["estimate"]],
                [y],
                s=32,
                facecolors=colour if point["confirmed"] else "white",
                edgecolors=colour,
                zorder=3,
            )
            axis.text(
                1.02,
                y,
                f"{point['estimate']:+.5g} [{point['lower']:+.5g}, {point['upper']:+.5g}]",
                transform=axis.get_yaxis_transform(),
                va="center",
                fontsize=8,
            )
            setting = point["setting"].split("=")[-1]
            budget = f" · {point['budget_sec']:g}s" if point["budget_sec"] else ""
            model = f" / {point['model']}" if len(models) > 1 else ""
            labels.append(
                point["task_id"].removeprefix("release_")
                + f"{model} / {setting}{budget}"
            )
        axis.axvline(0, color="#555555", linestyle="--", linewidth=0.8)
        axis.set(
            yticks=range(len(points)),
            yticklabels=labels,
            xlabel=f"Improvement ({unit}); positive = better",
            ylabel="Task / agent setting / measurement budget",
        )
        axis.invert_yaxis()
        axis.tick_params(axis="y", labelsize=8, length=0)
        axis.tick_params(axis="x", labelsize=8)
        axis.grid(axis="x", alpha=0.15)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.set_title(
            f"{title}\n{panel} · saved {level * 100:g}% confidence intervals"
            + (f" · {next(iter(models))}" if len(models) == 1 else ""),
            fontsize=11,
            loc="left",
            pad=12,
        )
        handles = [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                color="#235b78",
                label="Historical reversed-order primary repeat",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor="white",
                color="#66717a",
                label="Paired evaluation estimate",
            ),
        ]
        axis.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0, -0.14),
            frameon=False,
            fontsize=7,
        )
        caption = (
            "Source: complete retained Judge grid; native paired instance bootstrap. Supplementary, not a retention qualification."
            if metric.startswith("supplementary_")
            else "Source: retained Judge reports. Native intervals and units preserved; no pooled Agent success-rate interval."
        )
        fig.text(
            0.01,
            0.01,
            caption,
            fontsize=7,
            color="#555555",
        )
        fig.subplots_adjust(left=0.38, right=0.76, top=0.87, bottom=0.24)
        basis = hashlib.sha256(json.dumps([title, unit]).encode()).hexdigest()[:8]
        stem = re.sub(
            r"[^a-zA-Z0-9_.-]+", "-", f"{metric}-{panel}-ci{level * 100:g}-{basis}"
        )
        assets = _save_figure(fig, output, stem)
        figures.append(
            {
                "metric_id": metric,
                "title": title,
                "panel": panel,
                "unit": unit,
                "confidence_level": level,
                "point_count": len(points),
                "origin": "derived" if metric.startswith("supplementary_") else "saved",
                "files": assets,
            }
        )
    return figures


def _scale_figures(conditions, output: Path) -> list[dict]:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    groups = defaultdict(lambda: defaultdict(list))
    for row in conditions:
        for cell in row.get("observed_scale_slices", []):
            groups[(row["task_id"], cell["panel"], cell["budget_sec"])][
                row["condition_id"]
            ].append((row, cell))
    figures = []
    for (task, panel, budget), series in sorted(groups.items()):
        fig = Figure(figsize=(11, 4.8))
        FigureCanvasAgg(fig)
        quality, gain = fig.subplots(1, 2)
        for index, values in enumerate(series.values()):
            values.sort(key=lambda pair: pair[1]["scale"])
            row = values[0][0]
            cells = [pair[1] for pair in values]
            x = [cell["scale"] for cell in cells]
            label = row["setting"].split("=")[-1]
            if row.get("model"):
                label = f"{row['model']} / {label}"
            colour = f"C{index % 10}"
            for state, style in [("base", "--"), ("agent", "-")]:
                y = [cell[state + "_gap"] * 100 for cell in cells]
                quality.plot(
                    x,
                    y,
                    linestyle=style,
                    marker="o",
                    color=colour,
                    label=f"{label} / {state}",
                )
                lower = [
                    cell[state + "_ci95"]["lower"] * 100
                    if cell[state + "_ci95"]
                    else math.nan
                    for cell in cells
                ]
                upper = [
                    cell[state + "_ci95"]["upper"] * 100
                    if cell[state + "_ci95"]
                    else math.nan
                    for cell in cells
                ]
                quality.fill_between(x, lower, upper, color=colour, alpha=0.08)
            gain.plot(
                x,
                [cell["gap_reduction"] * 100 for cell in cells],
                marker="o",
                color=colour,
                label=label,
            )
            lower = [
                cell["reduction_ci95"]["lower"] * 100
                if cell["reduction_ci95"]
                else math.nan
                for cell in cells
            ]
            upper = [
                cell["reduction_ci95"]["upper"] * 100
                if cell["reduction_ci95"]
                else math.nan
                for cell in cells
            ]
            gain.fill_between(x, lower, upper, color=colour, alpha=0.15)
        first = next(iter(series.values()))
        cells = sorted([pair[1] for pair in first], key=lambda c: c["scale"])
        for axis in (quality, gain):
            axis.set_xlabel(cells[0]["axis_label"])
            axis.set_xticks([cell["scale"] for cell in cells])
            axis.grid(alpha=0.15)
            axis.spines[["top", "right"]].set_visible(False)
            axis.legend(frameon=False, fontsize=7)
        quality.set(
            title="Fixed-budget solution quality",
            ylabel="Mean normalized gap (%) — lower is better",
        )
        gain.set(title="Paired improvement", ylabel="Gap reduction (percentage points)")
        gain.axhline(0, color="#555555", linestyle="--", linewidth=0.8)
        counts = ", ".join(f"{c['scale']:g}: {c['instance_count']}" for c in cells)
        fig.suptitle(
            f"{task} / {panel} / {budget:g}s — observed size slice", fontsize=12
        )
        fig.text(
            0.01,
            0.045,
            f"Independent instances per size: {counts}. Native 95% instance-bootstrap bands; seed means within each instance.",
            fontsize=7,
        )
        fig.text(
            0.01,
            0.01,
            "Exploratory slice within this shift recipe; not a shift-severity gradient or a frozen scalability qualification.",
            fontsize=7,
        )
        fig.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.22, wspace=0.35)
        stem = re.sub(
            r"[^a-zA-Z0-9_.-]+", "-", f"observed-scale-{task}-{panel}-budget{budget:g}"
        )
        figures.append(
            {
                "metric_id": "observed_scale_slice",
                "title": f"{task}: observed size response at {budget:g}s",
                "panel": panel,
                "unit": "normalized gap (%) / paired reduction (pp)",
                "confidence_level": 0.95,
                "point_count": sum(len(v) * 3 for v in series.values()),
                "origin": "derived_scale",
                "files": _save_figure(fig, output, stem),
            }
        )
    return figures


def write_distribution_report(
    source: Path, output: Path, *, generation_specification: Path
) -> dict:
    """Plot saved per-distribution native intervals, without new solver runs."""
    require_plotting()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import yaml

    definition = yaml.safe_load(generation_specification.read_text())
    conditions = {item["id"]: item for item in definition["conditions"]}
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "kind": "distribution_effect_report",
        "scope": "evaluator_reference_headroom",
        "interpretation": "Reference-intervention calibration, not the 90 experimental Agent outcomes.",
        "estimates": [],
        "curves": [],
        "pending_or_failed": [],
        "collection": [],
        "figures": [],
    }
    records = sorted(source.rglob("headroom.json"))
    series = defaultdict(list)
    for path in records:
        saved = _read(path)
        measurement_id = path.parent.relative_to(source).as_posix()
        if saved["status"] != "measured":
            report["pending_or_failed"].append(
                {
                    "path": str(path),
                    "status": saved["status"],
                    "error": saved.get("error"),
                }
            )
            continue
        collection = saved.get("collection_by_panel")
        audit_path = path.with_name("collection-summary.json")
        if collection is None and audit_path.is_file():
            collection = _read(audit_path)["collection_by_panel"]
        failed_panels = set()
        for panel, states in (collection or {}).items():
            failed = set(states) == {"base", "agent"} and all(
                counts["observed_runs"] > 0
                and sum(
                    counts["status_counts"].get(status, 0)
                    for status in (
                        "crashed",
                        "build_failed",
                        "output_error",
                        "solver_error",
                        "infrastructure_error",
                    )
                )
                == counts["observed_runs"]
                for counts in states.values()
            )
            report["collection"].append(
                {
                    "path": str(path),
                    "task_id": saved["task_id"],
                    "measurement_id": measurement_id,
                    "panel": panel,
                    "status": "collection_failed" if failed else "recorded",
                    "by_state": states,
                }
            )
            if failed:
                failed_panels.add(panel)
                report["pending_or_failed"].append(
                    {"path": str(path), "panel": panel, "status": "collection_failed"}
                )
        for partition, panels in saved["partitions"].items():
            for panel, performance in panels.items():
                condition_id = panel
                if panel == "judge_id":
                    candidates = [
                        item
                        for item in conditions.values()
                        if item["axes"]["axis"] == "baseline"
                    ]
                    exact = "time_basis" in performance
                    condition_id = next(
                        item["id"]
                        for item in candidates
                        if (
                            definition["families"][item["family"]]["problem_family"]
                            == "mip"
                        )
                        == exact
                    )
                condition = conditions.get(condition_id)
                if condition is None:
                    report["pending_or_failed"].append(
                        {
                            "path": str(path),
                            "status": "unknown_distribution",
                            "panel": panel,
                        }
                    )
                    continue
                for budget, cell in performance["by_budget"].items():
                    metrics = [
                        (
                            "fixed_budget_normalized_gap",
                            "mean_gap_reduction",
                            "mean_gap_reduction_ci95",
                            "percentage points",
                            100,
                        )
                    ]
                    if "time_basis" in performance:
                        metrics = [
                            (
                                "exact_solved_coverage",
                                "mean_verified_solved_coverage_reduction",
                                "mean_verified_solved_coverage_reduction_ci95",
                                "percentage points",
                                100,
                            ),
                            (
                                "exact_penalized_runtime_sec",
                                "mean_penalized_runtime_reduction_sec",
                                "mean_penalized_runtime_reduction_ci95",
                                "seconds",
                                1,
                            ),
                        ]
                    for metric, key, ci_key, unit, scale in metrics:
                        row = {
                            "condition_id": f"{measurement_id}/{partition}/{condition_id}",
                            "task_id": saved["task_id"],
                            "model": "evaluator_reference",
                            "setting": f"{partition} / {measurement_id}",
                            "issues": [],
                            "classification": performance["classification"],
                        }
                        point = _point(
                            row,
                            metric=metric,
                            title=metric,
                            unit=unit,
                            estimate=cell["paired"].get(key),
                            interval=cell["paired"].get(ci_key),
                            reference=f"{path}:{partition}:{panel}:{budget}",
                            panel=condition_id,
                            scale=scale,
                            budget_sec=float(budget),
                        )
                        point.update(
                            family=condition["family"],
                            axes=condition["axes"],
                            issues=row["issues"],
                            paired_instances=cell["paired"]["paired_instances"],
                            collection_by_state=(collection or {}).get(panel),
                            measurement_id=measurement_id,
                        )
                        if panel in failed_panels:
                            point["plot_ready"] = False
                            point["issues"].append(
                                "All runs in both source states failed collection; the saved native estimate is not performance evidence."
                            )
                        report["estimates"].append(point)
                        coordinate = condition["axes"]["coordinate"]
                        if (
                            point["plot_ready"]
                            and isinstance(coordinate, (int, float))
                            and not isinstance(coordinate, bool)
                        ):
                            series[
                                (
                                    saved["task_id"],
                                    partition,
                                    condition["family"],
                                    condition["axes"]["axis"],
                                    metric,
                                    float(budget),
                                    unit,
                                    measurement_id,
                                )
                            ].append((coordinate, point))
    report["figures"] = _figures(report["estimates"], output)
    for key, points in series.items():
        if len(points) < 2:
            continue
        task, partition, family, axis, metric, budget, unit, measurement_id = key
        points.sort(key=lambda item: item[0])
        fig, ax = plt.subplots(figsize=(7, 4))
        x = [item[0] for item in points]
        mean = [item[1]["estimate"] for item in points]
        lower = [item[1]["lower"] for item in points]
        upper = [item[1]["upper"] for item in points]
        ax.plot(x, mean, "o-", label="paired reference effect")
        ax.fill_between(x, lower, upper, alpha=0.18, label="saved native interval")
        ax.axhline(0, color="#777777", linewidth=0.8)
        ax.set(
            xlabel=axis,
            ylabel=unit,
            title=f"{task} / {partition} / {budget:g}s\n{measurement_id}",
        )
        ax.legend()
        stem = re.sub(
            r"[^a-zA-Z0-9_.-]+",
            "-",
            f"distribution-{measurement_id}-{partition}-{family}-{axis}-{metric}-{budget:g}",
        )
        assets = _save_figure(fig, output, stem)
        plt.close(fig)
        curve = {
            "task_id": task,
            "measurement_id": measurement_id,
            "partition": partition,
            "family": family,
            "axis": axis,
            "metric_id": metric,
            "budget_sec": budget,
            "coordinates": x,
            "estimates": mean,
            "lower": lower,
            "upper": upper,
            "files": assets,
        }
        report["curves"].append(curve)
    _write_json(output / "distribution-report.json", report)
    environment = Environment(autoescape=select_autoescape(["html"]))
    template = environment.from_string("""<!doctype html><meta charset="utf-8"><title>Distribution effects</title>
<h1>Distribution effects</h1><p>{{ report.interpretation }}</p>
<p>{{ report.estimates|length }} saved estimates; {{ report.pending_or_failed|length }} pending or failed records.</p>
<h2>Collection audit</h2>
<p>Native estimates are retained. Panels where both source states failed every run are excluded from plots. Time limits without a solution remain solver outcomes. Missing collection summaries are not proof of successful collection.</p>
<table border="1"><tr><th>Task / panel</th><th>State</th><th>Verified solutions / observed / expected</th><th>Statuses and errors</th></tr>
{% for item in report.collection %}{% for state, counts in item.by_state.items() %}<tr><td>{{ item.task_id }} / {{ item.panel }} ({{ item.status }})<br>{{ item.measurement_id }}</td><td>{{ state }}</td><td>{{ counts.valid_solutions }} / {{ counts.observed_runs }} / {{ counts.expected_runs }}</td><td>{{ counts.status_counts|tojson }} {{ counts.error_counts|tojson }}</td></tr>{% endfor %}{% endfor %}</table>
{% for curve in report.curves %}<h2>{{ curve.task_id }}: {{ curve.axis }} / {{ curve.partition }}</h2><p>{{ curve.measurement_id }}</p><img style="max-width:100%" src="{{ curve.files.svg }}">{% endfor %}
{% for figure in report.figures %}<h2>{{ figure.title }}</h2><img style="max-width:100%" src="{{ figure.files.svg }}">{% endfor %}""")
    (output / "report.html").write_text(template.render(report=report))
    return report


def write_campaign_report(
    source: Path,
    output: Path,
    *,
    repository_root: Path,
    index: Path | None = None,
    runs_root: Path | None = None,
    task_configs: dict[str, Path] | None = None,
) -> dict:
    """Write one offline report and publication-exportable native CI figures."""
    require_plotting()
    report = collect_campaign(
        source,
        repository_root=repository_root,
        index=index,
        runs_root=runs_root,
        task_configs=task_configs,
    )
    output.mkdir(parents=True, exist_ok=True)
    report["figures"] = _figures(report["estimates"], output)
    report["figures"].extend(_scale_figures(report["conditions"], output))
    report["metric_coverage"] = {
        metric: dict(
            Counter(
                row["metric_inventory"]["metrics"][metric]["status"]
                if row["metric_inventory"]
                else "evaluation_unavailable"
                for row in report["conditions"]
            )
        )
        for metric in objective_catalog()["metrics"]
    }
    _write_json(output / "report.json", report)
    scale_fields = [
        "condition_id",
        "task_id",
        "model",
        "setting",
        "panel",
        "scale",
        "budget_sec",
        "instance_count",
        "seed_count",
        "base_gap",
        "agent_gap",
        "gap_reduction",
        "base_lower_ci95",
        "base_upper_ci95",
        "agent_lower_ci95",
        "agent_upper_ci95",
        "reduction_lower_ci95",
        "reduction_upper_ci95",
        "gap_unit",
        "confidence_level",
        "interpretation",
    ]
    with (output / "scale-slices.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=scale_fields, extrasaction="ignore")
        writer.writeheader()
        for row in report["conditions"]:
            for cell in row.get("observed_scale_slices", []):
                flat = {
                    **row,
                    **cell,
                    "gap_unit": "fraction",
                    "confidence_level": (cell["reduction_ci95"] or {}).get("level"),
                }
                for prefix in ("base", "agent", "reduction"):
                    interval = cell[prefix + "_ci95"] or {}
                    flat[prefix + "_lower_ci95"] = interval.get("lower")
                    flat[prefix + "_upper_ci95"] = interval.get("upper")
                writer.writerow(flat)
    estimate_fields = [
        "condition_id",
        "task_id",
        "model",
        "setting",
        "budget_sec",
        "metric_id",
        "panel",
        "unit",
        "estimate",
        "lower",
        "upper",
        "confidence_level",
        "classification",
        "confirmed",
        "plot_ready",
        "source_field",
    ]
    with (output / "effects.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=estimate_fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(report["estimates"])
    condition_fields = [
        "condition_id",
        "task_id",
        "model",
        "setting",
        "execution_status",
        "evaluation_status",
        "classification",
        "confirmed",
        "primary_budget_sec",
        "validity_accepted",
        "error",
        "result_path",
        "results_sha256",
        "cpu_change_percent",
        "cpu_eligible_pairs",
        "cpu_expected_pairs",
        "rss_change_percent",
        "rss_eligible_pairs",
        "rss_expected_pairs",
        "seed_base_iqr",
        "seed_agent_iqr",
        "seed_iqr_unit",
        "representation_base_iqr",
        "representation_agent_iqr",
        "representation_iqr_unit",
        "retention_supported",
        "retention_conditions",
    ]
    with (output / "conditions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=condition_fields, extrasaction="ignore"
        )
        writer.writeheader()
        for row in report["conditions"]:
            flat = dict(row)
            for metric, prefix in (("cpu", "cpu"), ("memory", "rss")):
                cell = row.get("primary_resources", {}).get(metric, {})
                flat.update(
                    {
                        prefix + "_change_percent": cell.get("change_percent"),
                        prefix + "_eligible_pairs": cell.get("paired_seed_count"),
                        prefix + "_expected_pairs": cell.get("expected_pairs"),
                    }
                )
            for key, prefix in (
                ("seed_iqr", "seed"),
                ("representation_iqr", "representation"),
            ):
                cell = row.get(key, {})
                flat.update(
                    {
                        prefix + "_base_iqr": cell.get("base"),
                        prefix + "_agent_iqr": cell.get("agent"),
                        prefix + "_iqr_unit": cell.get("unit"),
                    }
                )
            flat.update(
                retention_supported=row.get("retention", {}).get("supported"),
                retention_conditions=row.get("retention", {}).get("conditions"),
            )
            writer.writerow(flat)
    environment = Environment(autoescape=select_autoescape(default=True))
    environment.filters["number"] = lambda value: (
        f"{value:+.6g}" if isinstance(value, (int, float)) else "未取得"
    )
    environment.filters["magnitude"] = lambda value: (
        f"{value:.5g}" if isinstance(value, (int, float)) else "未取得"
    )
    template = environment.from_string(
        files("pitbench.metrics").joinpath("templates/campaign.html.j2").read_text()
    )
    (output / "report.html").write_text(
        template.render(report=report, catalog=objective_catalog())
    )
    return report


def mean(values):
    return statistics.fmean(values) if values else None


def observation_diagnostics(
    directory,
    performance,
    expected,
    outcome_function,
    observation_class,
    time_basis_class,
):
    obs_path = directory / "judge-observations.jsonl"
    if not obs_path.exists():
        return {"missing_observation_file": True}
    observations = [json.loads(line) for line in obs_path.open() if line.strip()]
    exact = "time_basis" in performance
    by_budget = defaultdict(dict)
    statuses = Counter()
    all_grid_keys = set()
    duplicate_keys = []
    for row in observations:
        fullkey = (
            row["code_state"],
            row["instance_set"],
            row["instance_id"],
            row["solver_seed"],
            row["budget_sec"],
            row.get("test_suite"),
            row.get("equivalence_parent_id"),
        )
        if fullkey in all_grid_keys:
            duplicate_keys.append(fullkey)
        all_grid_keys.add(fullkey)
        statuses[row["code_state"] + ":" + row["status"]] += 1
        if (
            row.get("instance_set_kind") != "judge_id"
            or row.get("test_suite")
            or row.get("equivalence_parent_id")
        ):
            continue
        key = (row["instance_id"], row["solver_seed"])
        by_budget[row["budget_sec"]].setdefault(key, {})[row["code_state"]] = row
    expected_keys = {
        (
            row["code_state"],
            row["instance_set"],
            row["instance_id"],
            row["solver_seed"],
            row["budget_sec"],
            row.get("test_suite"),
            row.get("equivalence_parent_id"),
        )
        for row in expected.get("runs", [])
    }
    result = {
        "observations": len(observations),
        "state_status_counts": dict(statuses),
        "expected_grid_rows": len(expected_keys),
        "missing_grid_rows": len(expected_keys - all_grid_keys),
        "unexpected_grid_rows": len(all_grid_keys - expected_keys),
        "duplicate_grid_rows": len(duplicate_keys),
        "by_budget": {},
    }
    for budget, pairs in sorted(by_budget.items()):
        per_instance = defaultdict(lambda: defaultdict(list))
        missing_pairs = 0
        progress = Counter()
        work = defaultdict(list)
        for (instance, seed), states in pairs.items():
            if set(states) != {"base", "agent"}:
                missing_pairs += 1
                continue
            base, agent = states["base"], states["agent"]
            for state, row in states.items():
                trajectory = row.get("trajectory_path")
                if trajectory:
                    path = directory / trajectory.removeprefix("/output/")
                    if path.is_file():
                        lines = sum(1 for line in path.open() if line.strip())
                        progress[state + ":files"] += 1
                        progress[state + ":multiple_records"] += lines > 1
                        progress[state + ":single_record"] += lines == 1
                    else:
                        progress[state + ":missing_file"] += 1
                for key in ["iterations", "nodes", "wall_time_sec", "cpu_time_sec"]:
                    if row.get(key) is not None:
                        work[state + ":" + key].append(row[key])
            inst = per_instance[instance]
            inst["seeds"].append(seed)
            if exact:
                outcomes = {
                    name: outcome_function(
                        observation_class.model_validate(row),
                        time_basis=time_basis_class(performance["time_basis"]),
                    )
                    for name, row in states.items()
                }
                for name, outcome in outcomes.items():
                    inst[name + "_outcomes"].append(outcome.kind.value)
                if all(
                    value.solved is not None and value.penalized_runtime_sec is not None
                    for value in outcomes.values()
                ):
                    inst["coverage_delta"].append(
                        float(outcomes["agent"].solved) - float(outcomes["base"].solved)
                    )
                    inst["runtime_reduction"].append(
                        outcomes["base"].penalized_runtime_sec
                        - outcomes["agent"].penalized_runtime_sec
                    )
                    inst["base_metric"].append(outcomes["base"].penalized_runtime_sec)
                    inst["agent_metric"].append(outcomes["agent"].penalized_runtime_sec)
            elif all(
                row.get("valid") and row.get("normalized_gap") is not None
                for row in states.values()
            ):
                inst["gap_reduction"].append(
                    base["normalized_gap"] - agent["normalized_gap"]
                )
                inst["base_metric"].append(base["normalized_gap"])
                inst["agent_metric"].append(agent["normalized_gap"])
        points = []
        coordinate = "coverage_delta" if exact else "gap_reduction"
        if (
            exact
            and mean(
                [
                    mean(v["coverage_delta"])
                    for v in per_instance.values()
                    if v["coverage_delta"]
                ]
            )
            == 0
        ):
            coordinate = "runtime_reduction"
        for instance, values in sorted(per_instance.items()):
            point = {
                "instance_id": instance,
                "paired_seeds": len(values["seeds"]),
                "valid_primary_pairs": len(values[coordinate]),
                "base_mean": mean(values["base_metric"]),
                "agent_mean": mean(values["agent_metric"]),
                "effect": mean(values[coordinate]),
            }
            if exact:
                point.update(
                    coverage_delta=mean(values["coverage_delta"]),
                    runtime_reduction=mean(values["runtime_reduction"]),
                    base_outcomes=dict(Counter(values["base_outcomes"])),
                    agent_outcomes=dict(Counter(values["agent_outcomes"])),
                )
            if len(values[coordinate]) > 1:
                point["within_instance_seed_sd"] = statistics.stdev(values[coordinate])
            points.append(point)
        deltas = [point["effect"] for point in points if point["effect"] is not None]
        leave_one_out = (
            [(sum(deltas) - v) / (len(deltas) - 1) for v in deltas]
            if len(deltas) > 1
            else []
        )
        result["by_budget"][f"{budget:g}"] = {
            "coordinate": coordinate,
            "units": "fraction"
            if coordinate in ["coverage_delta", "gap_reduction"]
            else "seconds",
            "paired_instances": len(points),
            "paired_seed_cells": len(pairs) - missing_pairs,
            "missing_pairs": missing_pairs,
            "better_instances": sum(v > 0 for v in deltas),
            "equal_instances": sum(v == 0 for v in deltas),
            "worse_instances": sum(v < 0 for v in deltas),
            "equal_instance_mean": mean(deltas),
            "median": statistics.median(deltas) if deltas else None,
            "leave_one_instance_out_min": min(leave_one_out) if leave_one_out else None,
            "leave_one_instance_out_max": max(leave_one_out) if leave_one_out else None,
            "top_absolute_contribution_share": max(map(abs, deltas))
            / sum(map(abs, deltas))
            if deltas and sum(map(abs, deltas))
            else None,
            "mean_recorded_work": {key: mean(value) for key, value in work.items()},
            "trajectory_record_counts": dict(progress),
            "instances": points,
        }
    return result


def trace_diagnostics(run, root):
    messages, commands, traces = [], {}, []
    first_change = None
    patch_states = set()
    for path in run.glob("*/*/agent-trace/*/events.jsonl"):
        coverage_path = path.parent / "coverage.json"
        coverage = _read(coverage_path) if coverage_path.exists() else {}
        finishes = []
        budget_summary = None
        for event in read_trace(path):
            kind = event["event"]
            if kind not in [
                "agent.event",
                "repository.checkpoint",
                "execution.finished",
                "research.budget_summary",
            ]:
                continue
            data = event["data"]
            source = {
                "path": str(path.relative_to(root))
                if path.is_relative_to(root)
                else str(path),
                "sequence": event["sequence"],
                "timestamp": event["timestamp"],
            }
            if kind == "repository.checkpoint":
                patch = data.get("patch", {})
                sha = patch.get("sha256")
                if sha:
                    patch_states.add(sha)
                if patch.get("bytes", 0) > 0 and (
                    first_change is None or event["timestamp"] < first_change
                ):
                    first_change = event["timestamp"]
            elif kind == "execution.finished":
                finishes.append(data)
            elif kind == "research.budget_summary":
                budget_summary = data.get("summary", data)
            else:
                native = data.get("native", {})
                item = native.get("item", {})
                if (
                    native.get("type") == "item.completed"
                    and item.get("type") == "agent_message"
                ):
                    messages.append(
                        {
                            **source,
                            "native_id": item.get("id"),
                            "text": item.get("text", ""),
                        }
                    )
                if item.get("type") == "command_execution":
                    ident = path.parent.name + ":" + item.get("id", "")
                    old = commands.get(ident, {})
                    commands[ident] = {
                        **source,
                        "started_at": old.get("started_at", event["timestamp"]),
                        "command": item.get("command", ""),
                        "exit_code": item.get("exit_code"),
                        "status": item.get("status"),
                        "output_excerpt": item.get("aggregated_output", "")[:1500],
                        "excerpt_limit": 1500,
                    }
        traces.append(
            {
                "events": artifact_file_identity(path, root),
                "execution_finished": coverage.get("execution_finished"),
                "observed_operations_paired": coverage.get(
                    "observed_operations_paired"
                ),
                "unpaired_operations": sum(
                    bool(op.get("missing")) for op in coverage.get("operations", [])
                ),
                "collection_gaps": coverage.get("collection_gaps", []),
                "event_counts": coverage.get("event_counts", {}),
                "execution_end_events": finishes,
                "budget_summary": budget_summary,
            }
        )
    for message in messages:
        message["before_first_observed_source_change"] = (
            message["timestamp"] < first_change if first_change else None
        )
    commands = sorted(commands.values(), key=lambda x: x["started_at"])
    profiler = [
        c
        for c in commands
        if re.search(
            r"\bperf\s+(?:record|stat)|\bcProfile\b|\bgprof\s+[^-]|--tool[= ]callgrind",
            c["command"],
        )
        and not re.search(r"command -v|which |--help", c["command"])
    ]
    benches = [c for c in commands if re.search(r"pitbench\s+bench\b", c["command"])]
    return {
        "traces": traces,
        "messages": messages,
        "commands": commands,
        "first_source_change_at": first_change,
        "distinct_recorded_source_patch_hashes": len(patch_states),
        "recorded_benchmark_command_count": len(benches),
        "recorded_profiler_attempts": profiler,
        "successful_profiler_command_count": sum(c["exit_code"] == 0 for c in profiler),
        "scope": "Observable event records; command-pattern counts do not establish bottleneck correctness or absence of unrecorded work.",
    }


def collect_campaign_diagnostics(
    project: Path,
    campaign_name: str,
    runs_directory: Path,
    campaign_directory: Path,
    output: Path,
) -> dict:
    """Inspect all saved conditions, retaining failures and missing executions."""
    from pitbench.metrics.performance_report import compute_exact_run_outcome
    from pitbench.schema.observation import RunObservation
    from pitbench.schema.task import ExactTimeBasis

    root = project.resolve()
    runs = (root / runs_directory).resolve()
    source = (root / campaign_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    audit_path = source / "final-model-audit-summary.json"
    audit = _read(audit_path) if audit_path.is_file() else {}
    conditions = retained_conditions(source, runs)
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project": str(root),
        "campaign": campaign_name,
        "conditions": [],
        "new_agent_calls": 0,
        "new_solver_runs": 0,
        "audit": audit,
        "analysis_scope": "Saved evidence; missing and failed conditions remain visible. No new performance thresholds.",
    }
    for row in conditions:
        execution = (
            row.get("execution_run_id") or row.get("run_id") or row["condition_id"]
        )
        run = runs / execution
        path = _result_path(
            source, runs, {**row, "run_id": row.get("run_id", execution)}
        )
        if not path.is_file():
            result["conditions"].append(
                {
                    "condition": row,
                    "missing_result": True,
                    "trace": trace_diagnostics(run, root),
                }
            )
            continue
        trials = _read(path).get("results", [])
        trial = next(
            (trial for trial in trials if trial.get("task_id") == row.get("task_id")),
            trials[0] if trials else {},
        )
        envelope = trial.get("evaluation") or {}
        payload = envelope.get("payload") or {}
        summary = payload.get("summary") or {}
        entry = {
            "condition": row,
            "result_identity": artifact_file_identity(path, root),
            "evaluation_completed": envelope.get("completed"),
            "evaluation_error": envelope.get("error"),
            "validity": payload.get("validity"),
            "summary": summary,
            "trial_count": len(trials),
            **{
                key: trial.get(key)
                for key in [
                    "failure_mode",
                    "agent_metrics",
                    "total_input_tokens",
                    "total_output_tokens",
                    "agent_started_at",
                    "agent_ended_at",
                ]
            },
        }
        directory = (
            Path(row["evaluation_output"])
            if row.get("evaluation_output")
            else next(run.glob("*/*/evaluation"), None)
        )
        if directory is not None:
            ref = (payload.get("artifacts") or {}).get("expected_run_grid") or {}
            grid = directory / ref.get("path", "expected-run-grid.json")
            if not grid.is_file():
                grid = directory / "expected-run-grid.json"
            measurement = grid.parent
            patch = measurement / "candidate.patch"
            if patch.is_file():
                entry["patch"] = {
                    **artifact_file_identity(patch, root),
                    "text": patch.read_text(errors="replace"),
                }
            performance = summary.get("performance")
            if performance and grid.is_file():
                entry["observations"] = observation_diagnostics(
                    measurement,
                    performance,
                    _read(grid),
                    compute_exact_run_outcome,
                    RunObservation,
                    ExactTimeBasis,
                )
            elif performance:
                entry["missing_expected_run_grid"] = True
        entry["trace"] = trace_diagnostics(run, root)
        result["conditions"].append(entry)
    _write_json(output / "release-diagnostics.json", result)
    return result
