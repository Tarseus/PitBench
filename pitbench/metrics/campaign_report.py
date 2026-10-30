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
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Any

from jinja2 import Environment, select_autoescape
from pydantic import ValidationError

from pitbench.evaluator.storage import ObservationStore
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
    if index is not None:
        payload = _read(index)
        return payload["conditions"] if isinstance(payload, dict) else payload
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
    candidates = [
        source / record.get("condition_id", record["run_id"]) / "results.json",
        runs_root / record["run_id"] / "results.json",
        source / record["run_id"] / "results.json",
    ]
    return next((path for path in candidates if path.is_file()), candidates[1])


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
        points.append(
            _point(
                row,
                metric="supplementary_shift_gap",
                title="Hidden-shift gap reduction (supplementary)",
                unit="percentage points",
                estimate=pair["mean_gap_reduction"],
                interval=pair["mean_gap_reduction_ci95"],
                scale=100,
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
    checkpoint = evaluation_dir / "judge-checkpoint.json"
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
    artifacts = (trial.get("evaluation") or {}).get("payload", {}).get("artifacts", {})
    observations_path = evaluation_dir / (artifacts.get("observations") or {}).get(
        "path", "trials.parquet"
    )
    observations = (
        ObservationStore.read(observations_path)
        if observations_path.is_file()
        else None
    )
    grid_path = evaluation_dir / "expected-run-grid.json"
    if (
        observations is not None
        and grid_path.is_file()
        and task.evaluation.performance_protocol.value == "heuristic_fixed_budget"
    ):
        grid = ExpectedRunGrid.model_validate_json(grid_path.read_text())
        for panel in task.instance_sets:
            if panel.kind.value != "judge_shift":
                continue
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
                task.evaluation.seed_robustness.seed_selection.seed_count
                if task.evaluation.seed_robustness is not None
                else len(task.evaluation.solver_seeds or [])
            )
            full_count = panel.size * seed_count * len(task.evaluation.budgets_sec) * 2
            if (
                not expected
                or len(expected) != full_count
                or len({r.instance_id for r in expected}) != panel.size
                or len({r.solver_seed for r in expected}) != seed_count
                or {r.budget_sec for r in expected} != set(task.evaluation.budgets_sec)
                or {r.code_state.value for r in expected} != {"base", "agent"}
                or len(selected) != len(expected)
                or len({_run_key(r) for r in expected}) != len(expected)
                or {_run_key(r) for r in selected} != {_run_key(r) for r in expected}
                or any(
                    not r.valid
                    or r.status.value != "completed"
                    or r.normalized_gap is None
                    or not math.isfinite(r.normalized_gap)
                    for r in selected
                )
            ):
                row["issues"].append(
                    f"{panel.name}: incomplete supplemental panel grid; no interval inferred"
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
            _observed_scale_slices(row, task, grid, selected)
    representation_path = evaluation_dir / "representation/details.json"
    representation = None
    if representation_path.is_file():
        details = _read(representation_path)
        patch_path = evaluation_dir / "candidate.patch"
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
            "error": record.get("error")
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
            evaluation_dir = result_path.parent / "evaluation"
            if not evaluation_dir.is_dir():
                evaluation_dir = (
                    result_path.parent
                    / row["task_id"]
                    / trial["trial_name"]
                    / "evaluation"
                )
            summary = (envelope.get("payload") or {}).get("summary") or {}
            current.update(
                evaluation_status="completed"
                if envelope.get("completed")
                else "failed",
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
                label="Reversed-order primary confirmation",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor="white",
                color="#66717a",
                label="Other saved evidence / unconfirmed",
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
