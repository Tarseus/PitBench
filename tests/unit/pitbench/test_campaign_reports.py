from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

from pitbench.cli.evaluate_config import EvaluationConfig
from pitbench.cli.main import app
from pitbench.evaluator.storage import ObservationStore
from pitbench.metrics.campaign_report import (
    _effects,
    _summary_values,
    collect_campaign,
    write_campaign_report,
)
from pitbench.metrics.performance_report import (
    BudgetPerformance,
    ConfidenceInterval,
    ExactBudgetPerformance,
    ExactPerformanceReport,
    ExactStatePerformance,
    GapEstimate,
    PairedExactPerformance,
    PairedGapEvidence,
    PerformanceConfirmation,
    PerformanceReport,
)
from pitbench.schema.observation import ExpectedRun, ExpectedRunGrid, RunObservation
from pitbench.tasks import TaskCatalog

ROOT = Path(__file__).resolve().parents[3]


def test_resource_cost_and_partial_pair_denominator_remain_visible():
    row = {
        "primary_budget_sec": 10,
        "native_summary": {
            "resource_usage": {
                "primary_budget_sec": 10,
                "expected_seed_count": 2,
                "by_instance_set": {
                    "judge_id": {
                        "10": {
                            "expected_instance_count": 1,
                            "cpu": {
                                "geometric_mean_ratio": 1.2,
                                "paired_seed_count": 1,
                            },
                            "memory": {
                                "geometric_mean_ratio": 0.9,
                                "paired_seed_count": 1,
                            },
                        }
                    }
                },
            }
        },
    }
    _summary_values(row)
    assert row["primary_resources"]["cpu"]["change_percent"] == pytest.approx(20)
    assert row["primary_resources"]["memory"]["change_percent"] == pytest.approx(-10)
    assert row["primary_resources"]["cpu"]["paired_seed_count"] == 1
    assert row["primary_resources"]["cpu"]["expected_pairs"] == 2


def test_transfer_normalization_and_scenario_interval_direction_are_preserved():
    row = {
        "condition_id": "case",
        "task_id": "case",
        "setting": "high",
        "model": "example",
        "issues": [],
        "scenario_budgets": {"setup": 2},
    }
    summary = {
        "performance": exact_report(),
        "controlled_transfer": {
            "condition_reports": {
                "shift": {
                    "condition_id": "shift",
                    "reference_condition_id": "base",
                    "budget_sec": 30,
                    "benefit": 0.1,
                    "benefit_ci95": {"lower": -0.2, "upper": 0.3, "level": 0.95},
                }
            }
        },
        "metric_scenarios": {
            "results": [
                {
                    "scenario_id": "setup",
                    "metric_id": "setup_time_sec",
                    "instance_set": "judge_id",
                    "degradation": -2,
                    "lower": -3,
                    "upper": -1,
                    "confidence_level": 0.95,
                }
            ]
        },
    }
    points = _effects(row, summary)
    shift = next(p for p in points if p["metric_id"] == "transfer_exact_benefit")
    assert (shift["estimate"], shift["lower"], shift["upper"]) == (0.1, -0.2, 0.3)
    assert shift["unit"] == "fraction of solve budget" and shift["budget_sec"] == 30
    scenario = next(p for p in points if p["metric_id"] == "setup_time_sec")
    assert (scenario["estimate"], scenario["lower"], scenario["upper"]) == (2, 1, 3)
    assert scenario["budget_sec"] == 2


def gap_report():
    cell = BudgetPerformance(
        budget_sec=10,
        base=GapEstimate(mean_normalized_gap=0.1),
        agent=GapEstimate(mean_normalized_gap=0.05),
        paired=PairedGapEvidence(
            paired_instances=2,
            mean_gap_reduction=0.05,
            mean_gap_reduction_ci95=ConfidenceInterval(lower=0.02, upper=0.08),
        ),
    )
    return PerformanceReport(
        classification="improved",
        primary_budget_sec=10,
        budgets_sec=[10],
        primary=cell,
        by_budget={"10": cell},
        confirmation=PerformanceConfirmation(
            initial_classification="improved",
            repeat_classification="improved",
            confirmed=True,
            repeat_observation_count=4,
            repeat_valid_observation_count=4,
        ),
    ).model_dump(mode="json")


def exact_report(time_basis="cpu_time"):
    base = ExactStatePerformance(
        verified_solved_coverage=0.5,
        penalized_average_runtime_sec=12,
        solved_run_count=1,
        expected_run_count=2,
        complete=True,
        outcome_counts={},
    )
    cell = ExactBudgetPerformance(
        budget_sec=10,
        base=base,
        agent=base.model_copy(update={"penalized_average_runtime_sec": 11}),
        paired=PairedExactPerformance(
            paired_instances=2,
            mean_verified_solved_coverage_reduction=0,
            mean_verified_solved_coverage_reduction_ci95=ConfidenceInterval(
                lower=0, upper=0
            ),
            mean_penalized_runtime_reduction_sec=1,
            mean_penalized_runtime_reduction_ci95=ConfidenceInterval(
                lower=-0.5, upper=2
            ),
            complete=True,
        ),
    )
    return ExactPerformanceReport(
        classification="inconclusive",
        time_basis=time_basis,
        primary_budget_sec=10,
        budgets_sec=[10],
        primary=cell,
        by_budget={"10": cell},
    ).model_dump(mode="json")


def saved_condition(source, identifier="one", performance=None):
    task_id = "pyvrp_v0_14_0"
    directory = source / identifier
    (directory / "evaluation").mkdir(parents=True)
    (directory / "task.yaml").write_text(
        (ROOT / f"configs/tasks/{task_id}.yaml").read_text()
    )
    summary = {
        "observation_count": 4,
        "valid_observation_count": 4,
        "performance": performance or gap_report(),
    }
    result = {
        "results": [
            {
                "task_id": task_id,
                "trial_name": "trial",
                "failure_mode": "unset",
                "evaluation": {
                    "completed": True,
                    "payload": {"validity": {"accepted": True}, "summary": summary},
                },
            }
        ]
    }
    (directory / "results.json").write_text(json.dumps(result))
    return {
        "condition_id": identifier,
        "run_id": identifier,
        "task_id": task_id,
        "model": "test-model",
        "setting": "reasoning_effort=high",
        "status": "command_completed",
    }


def index(source, records):
    path = source / "index.json"
    path.write_text(json.dumps({"conditions": records}))
    return path


def test_report_preserves_saved_intervals_and_keeps_failures_and_missing_rows(tmp_path):
    source = tmp_path / "campaign"
    source.mkdir()
    complete = saved_condition(source)
    failed = {
        **complete,
        "condition_id": "failed",
        "run_id": "failed",
        "status": "command_failed",
        "error": "build <failed>",
    }
    queued = {
        **complete,
        "condition_id": "queued",
        "run_id": "queued",
        "status": "queued",
    }
    source_index = index(source, [complete, failed, queued])
    original = (source / "one/results.json").read_bytes()

    report = write_campaign_report(
        source, tmp_path / "report", repository_root=ROOT, index=source_index
    )

    assert report["planned_conditions"] == 3
    assert report["summary"]["missing_or_failed"] == 2
    assert report["summary"]["confirmed_improved"] == 1
    assert [
        (p["estimate"], p["lower"], p["upper"], p["unit"]) for p in report["estimates"]
    ] == [(5, 2, 8, "percentage points")]
    assert len(report["figures"]) == 1
    figure = report["figures"][0]
    for file in figure["files"].values():
        assert (tmp_path / "report" / file).stat().st_size > 100
    assert "+5 [+2, +8]" in (tmp_path / "report" / figure["files"]["svg"]).read_text()
    assert (source / "one/results.json").read_bytes() == original
    html = (tmp_path / "report/report.html").read_text()
    assert "build &lt;failed&gt;" in html
    assert (
        html.split('id="conditions-table"')[1].split("</tbody>")[0].count("<tr>") == 4
    )
    rows = list(
        csv.DictReader((tmp_path / "report/conditions.csv").open(encoding="utf-8-sig"))
    )
    assert len(rows) == 3 and rows[1]["error"] == "build <failed>"


def test_exact_runtime_tie_and_different_time_bases_have_distinct_figures(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    records = [
        saved_condition(source, "cpu", exact_report()),
        saved_condition(source, "wall", exact_report("solve_wall_time")),
    ]
    report = write_campaign_report(
        source, tmp_path / "report", repository_root=ROOT, index=index(source, records)
    )
    assert {p["metric_id"] for p in report["estimates"]} == {
        "exact_penalized_runtime_sec"
    }
    assert {
        (p["estimate"], p["lower"], p["upper"], p["unit"]) for p in report["estimates"]
    } == {(1, -0.5, 2, "seconds")}
    assert len(report["figures"]) == 2
    assert len({p["files"]["svg"] for p in report["figures"]}) == 2
    assert report["summary"]["confirmed_improved"] == 0


def test_changed_task_config_does_not_erase_saved_native_estimates(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    record = saved_condition(source)
    (source / "one/evaluation/judge-checkpoint.json").write_text(
        json.dumps({"identity": {"task_config_sha256": "0" * 64}})
    )
    report = collect_campaign(
        source, repository_root=ROOT, index=index(source, [record])
    )
    assert report["estimates"][0]["estimate"] == 5
    assert report["conditions"][0]["metric_inventory"] is None
    assert any(
        "differs from the saved Judge identity" in s
        for s in report["conditions"][0]["issues"]
    )


def test_supplemental_shift_uses_the_native_estimator_only_for_full_valid_grids(
    tmp_path,
):
    source = tmp_path / "source"
    source.mkdir()
    record = saved_condition(source)
    directory = source / "one"
    task = yaml.safe_load((directory / "task.yaml").read_text())
    task["evaluation"].update(
        budgets_sec=[10],
        primary_budget_sec=10,
        seed_robustness=None,
        representation_robustness=None,
        solver_seeds=[0],
    )
    next(p for p in task["instance_sets"] if p["name"] == "judge_shift")["size"] = 2
    (directory / "task.yaml").write_text(yaml.safe_dump(task))
    rows = [
        RunObservation(
            task_id=record["task_id"],
            code_state=state,
            instance_set="judge_shift",
            instance_set_kind="judge_shift",
            instance_id=iid,
            solver_seed=0,
            budget_sec=10,
            status="completed",
            valid=True,
            normalized_gap=gap,
            problem_scale=100 if iid == "a" else 250,
        )
        for iid in ["a", "b"]
        for state, gap in [("base", 0.1), ("agent", 0.08)]
    ]
    grid = ExpectedRunGrid(
        task_id=record["task_id"],
        runs=[
            ExpectedRun(**{key: getattr(row, key) for key in ExpectedRun.model_fields})
            for row in rows
        ],
    )
    grid_path = directory / "evaluation/expected-run-grid.json"
    grid_path.write_text(grid.model_dump_json())
    parquet = directory / "evaluation/trials.parquet"
    ObservationStore.write(parquet, rows)
    source_index = index(source, [record])
    complete = collect_campaign(source, repository_root=ROOT, index=source_index)
    point = next(
        p for p in complete["estimates"] if p["metric_id"] == "supplementary_shift_gap"
    )
    assert point["estimate"] == pytest.approx(2)
    assert point["lower"] == pytest.approx(2) and point["upper"] == pytest.approx(2)
    assert not point["confirmed"]
    slices = complete["conditions"][0]["observed_scale_slices"]
    assert [cell["scale"] for cell in slices] == [100, 250]
    assert all(
        cell["instance_count"] == 1 and cell["reduction_ci95"] is None
        for cell in slices
    )
    rendered = write_campaign_report(
        source, tmp_path / "report", repository_root=ROOT, index=source_index
    )
    scale = next(
        figure
        for figure in rendered["figures"]
        if figure["metric_id"] == "observed_scale_slice"
    )
    assert "Customer count" in (tmp_path / "report" / scale["files"]["svg"]).read_text()
    assert (
        "not a shift-severity gradient"
        in (tmp_path / "report" / scale["files"]["svg"]).read_text()
    )

    ObservationStore.write(parquet, rows[:-1])
    missing = collect_campaign(source, repository_root=ROOT, index=source_index)
    assert len(missing["estimates"]) == 1
    assert missing["conditions"][0]["classification"] == "improved"

    ObservationStore.write(parquet, rows[:2])
    grid_path.write_text(
        grid.model_copy(update={"runs": grid.runs[:2]}).model_dump_json()
    )
    reduced = collect_campaign(source, repository_root=ROOT, index=source_index)
    assert len(reduced["estimates"]) == 1
    assert any(
        "incomplete supplemental panel" in issue
        for issue in reduced["conditions"][0]["issues"]
    )


def test_no_saved_interval_is_retained_without_a_fabricated_plot(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    performance = gap_report()
    performance["primary"]["paired"]["mean_gap_reduction_ci95"] = None
    record = saved_condition(source, performance=performance)
    report = write_campaign_report(
        source, tmp_path / "report", repository_root=ROOT, index=index(source, [record])
    )
    assert report["estimates"][0]["estimate"] == 5
    assert report["estimates"][0]["lower"] is None
    assert not report["estimates"][0]["plot_ready"]
    assert not report["figures"]
    assert len(report["conditions"]) == 1


def test_report_campaign_cli_only_reads_saved_evidence(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    record = saved_condition(source)
    source_index = index(source, [record])
    result = CliRunner().invoke(
        app,
        [
            "report-campaign",
            str(source),
            "--index",
            str(source_index),
            "--output",
            str(tmp_path / "report"),
            "--root",
            str(ROOT),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "1 confidence-interval figures" in result.output
    assert (tmp_path / "report/report.html").is_file()


def test_failed_model_campaign_still_generates_the_automatic_report(
    tmp_path, monkeypatch
):
    from scripts import run_benchmark_release as campaign

    task_id = "pyvrp_v0_14_0"
    task = TaskCatalog(ROOT).validate_one(task_id)
    campaign.write_json(
        tmp_path / "baseline.json",
        {
            task_id: {
                "status": "passed",
                "identity": {"task_config_sha256": task.task_config_sha256},
            }
        },
    )
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(EvaluationConfig().model_dump(mode="json")))
    matrix = tmp_path / "models.yaml"
    matrix.write_text(
        yaml.safe_dump(
            {
                "runs": [
                    {
                        "label": "example",
                        "agent": "codex",
                        "model": "example",
                        "setting": "reasoning_effort=high",
                    }
                ]
            }
        )
    )
    args = SimpleNamespace(
        output=tmp_path, config=config, models=matrix, workers=1, cpu_pools=[]
    )

    def fail(*_):
        raise RuntimeError("retained command failure")

    monkeypatch.setattr(campaign, "checked_run", fail)

    with pytest.raises(SystemExit):
        campaign.models([task_id], args)

    report = json.loads((tmp_path / "report/report.json").read_text())
    assert report["summary"]["missing_or_failed"] == 1
    assert report["conditions"][0]["error"] == "retained command failure"
    assert report["figures"] == []
