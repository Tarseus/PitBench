from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from pitbench.cli.main import app
from pitbench.evaluator.storage import ObservationStore
from pitbench.metrics.performance_report import (
    compute_performance_report,
    format_performance_report,
)
from pitbench.schema.observation import CodeState, RunObservation, RunStatus


def _observation(
    state: CodeState,
    instance_set_kind: str,
    instance_id: str,
    solver_seed: int,
    gap: float | None,
    *,
    budget_sec: float = 5.0,
    valid: bool = True,
    status: RunStatus = RunStatus.COMPLETED,
) -> RunObservation:
    return RunObservation(
        task_id="performance",
        code_state=state,
        instance_set=instance_set_kind,
        instance_set_kind=instance_set_kind,
        instance_id=instance_id,
        instance_seed=17,
        solver_seed=solver_seed,
        budget_sec=budget_sec,
        status=status,
        valid=valid,
        normalized_gap=gap,
    )


def _paired_panel() -> list[RunObservation]:
    observations: list[RunObservation] = []
    for instance_set, rows in {
        "judge_id": (("i1", 0.11, 0.07), ("i2", 0.21, 0.16)),
        "judge_shift": (("s1", 0.13, 0.11), ("s2", 0.18, 0.17)),
    }.items():
        for instance, base_gap, agent_gap in rows:
            for seed in (0, 1):
                observations.extend(
                    [
                        _observation(
                            CodeState.BASE,
                            instance_set,
                            instance,
                            seed,
                            base_gap,
                        ),
                        _observation(
                            CodeState.AGENT,
                            instance_set,
                            instance,
                            seed,
                            agent_gap,
                        ),
                    ]
                )
    return observations


def test_performance_report_pairs_seeds_and_bootstraps_instances() -> None:
    report = compute_performance_report(_paired_panel(), primary_budget_sec=5.0)

    assert report.primary_instance_set_kind == "judge_id"
    assert report.primary_budget_sec == 5.0
    assert report.solver_seeds == [0, 1]
    assert report.primary.base.mean_normalized_gap == pytest.approx(0.16)
    assert report.primary.agent.mean_normalized_gap == pytest.approx(0.115)
    paired = report.primary.paired
    assert paired.paired_runs == 4
    assert paired.paired_instances == 2
    assert paired.agent_better == 4
    assert paired.equal == 0
    assert paired.agent_worse == 0
    assert paired.mean_gap_reduction == pytest.approx(0.045)
    assert paired.mean_gap_reduction_by_seed == pytest.approx({0: 0.045, 1: 0.045})
    assert paired.mean_gap_reduction_ci95 is not None
    assert paired.mean_gap_reduction_ci95.lower == pytest.approx(0.04)
    assert paired.mean_gap_reduction_ci95.upper == pytest.approx(0.05)

    assert len(report.held_out_retention) == 1
    retention = report.held_out_retention[0]
    assert retention.judge_id_gap_reduction == pytest.approx(0.045)
    assert retention.judge_shift_gap_reduction == pytest.approx(0.015)
    assert retention.shift_minus_id_gap_reduction == pytest.approx(-0.03)
    assert retention.retained_on_shift


def test_performance_report_tracks_invalid_runs() -> None:
    observations = _paired_panel()
    observations.append(
        _observation(
            CodeState.AGENT,
            "judge_id",
            "failed",
            2,
            None,
            valid=False,
            status=RunStatus.TIMED_OUT,
        )
    )

    report = compute_performance_report(observations, primary_budget_sec=5.0)

    assert report.primary.agent.total_runs == 5
    assert report.primary.agent.valid_runs == 4
    assert report.primary.agent.failure_counts == {"timed_out": 1}


def test_declared_primary_budget_is_not_replaced_by_larger_budget() -> None:
    observations = _paired_panel()
    observations.extend(
        [
            _observation(
                CodeState.BASE,
                "judge_id",
                "diagnostic",
                0,
                0.20,
                budget_sec=10.0,
            ),
            _observation(
                CodeState.AGENT,
                "judge_id",
                "diagnostic",
                0,
                0.10,
                budget_sec=10.0,
            ),
        ]
    )

    report = compute_performance_report(observations, primary_budget_sec=5.0)

    assert report.budgets_sec == [5.0, 10.0]
    assert report.primary_budget_sec == 5.0
    assert report.primary.budget_sec == 5.0


def test_performance_report_rejects_missing_primary_budget() -> None:
    with pytest.raises(
        ValueError,
        match="declared primary budget 10s has no observations",
    ):
        compute_performance_report(_paired_panel(), primary_budget_sec=10.0)


def test_format_performance_report_is_performance_only() -> None:
    rendered = format_performance_report(
        compute_performance_report(_paired_panel(), primary_budget_sec=5.0)
    )

    assert "quality-time performance" in rendered
    assert "Held-out instance-set retention" in rendered
    assert "Repeatability evidence" in rendered
    assert "95% CI" in rendered
    assert "Stability" not in rendered


def test_report_command_supports_structured_output(tmp_path: Path) -> None:
    path = tmp_path / "trials.parquet"
    ObservationStore.write(path, _paired_panel())
    task_config = tmp_path / "task-config.yaml"
    source = Path(__file__).resolve().parents[3] / "configs/tasks/pyvrp_v0_14_0.yaml"
    payload = yaml.safe_load(source.read_text())
    payload["task_id"] = "performance"
    payload["evaluation"]["primary_budget_sec"] = 5.0
    task_config.write_text(yaml.safe_dump(payload, sort_keys=False))
    runner = CliRunner()

    text_result = runner.invoke(
        app,
        ["report", str(path), "--task-config", str(task_config)],
    )
    assert text_result.exit_code == 0
    assert "quality-time performance" in text_result.output

    json_result = runner.invoke(
        app,
        ["report", str(path), "--task-config", str(task_config), "--json"],
    )
    assert json_result.exit_code == 0
    assert '"performance"' in json_result.output
    assert '"mean_gap_reduction"' in json_result.output
    assert '"sensitivity"' not in json_result.output


def test_report_command_rejects_task_config_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "trials.parquet"
    ObservationStore.write(path, _paired_panel())
    task_config = (
        Path(__file__).resolve().parents[3] / "configs/tasks/pyvrp_v0_14_0.yaml"
    )

    result = CliRunner().invoke(
        app,
        ["report", str(path), "--task-config", str(task_config)],
    )

    assert result.exit_code == 2
    assert "task ID mismatch" in result.output
