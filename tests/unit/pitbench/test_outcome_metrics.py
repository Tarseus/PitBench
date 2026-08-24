from __future__ import annotations

import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pitbench.cli.main import app
from pitbench.evaluator.storage import ObservationStore
from pitbench.metrics.outcome_metrics import (
    compute_outcome_metrics,
    format_outcome_report_table,
    shifted_geometric_mean,
)
from pitbench.schema.observation import CodeState, RunObservation, RunStatus


def test_shifted_geometric_mean() -> None:
    assert shifted_geometric_mean([]) is None
    with pytest.raises(ValueError, match="shift must be non-negative"):
        shifted_geometric_mean([1.0], shift=-0.5)

    # For values [0, 1, 3] and shift=1.0:
    # (0+1) * (1+1) * (3+1) = 1 * 2 * 4 = 8.
    # geometric mean of (x+1) is (8)^(1/3) = 2.0.
    # shifted mean is 2.0 - 1.0 = 1.0.
    result = shifted_geometric_mean([0.0, 1.0, 3.0], shift=1.0)
    assert result is not None
    assert math.isclose(result, 1.0, rel_tol=1e-6)


def _make_obs(
    code_state: CodeState,
    instance_id: str,
    solver_seed: int,
    *,
    status: RunStatus = RunStatus.COMPLETED,
    valid: bool = True,
    normalized_gap: float | None = None,
    wall_time_sec: float | None = None,
    cpu_time_sec: float | None = None,
    peak_rss_bytes: int | None = None,
    population: str = "pop1",
    budget_sec: float = 10.0,
) -> RunObservation:
    return RunObservation(
        task_id="task1",
        code_state=code_state,
        population=population,
        instance_id=instance_id,
        instance_seed=100,
        solver_seed=solver_seed,
        budget_sec=budget_sec,
        status=status,
        valid=valid,
        normalized_gap=normalized_gap,
        wall_time_sec=wall_time_sec,
        cpu_time_sec=cpu_time_sec,
        peak_rss_bytes=peak_rss_bytes,
    )


def test_compute_outcome_metrics() -> None:
    mb = 1024 * 1024
    observations = [
        # Base runs (2 valid)
        _make_obs(
            CodeState.BASE,
            "inst1",
            seed,
            normalized_gap=gap,
            wall_time_sec=time_val,
            cpu_time_sec=cpu_val,
            peak_rss_bytes=rss_val,
        )
        for seed, gap, time_val, cpu_val, rss_val in [
            (0, 0.10, 4.0, 4.0, 100 * mb),
            (1, 0.20, 9.0, 9.0, 200 * mb),
        ]
    ] + [
        # Agent runs (1 valid, 1 timed out)
        _make_obs(
            CodeState.AGENT,
            "inst1",
            0,
            status=RunStatus.COMPLETED,
            valid=True,
            normalized_gap=0.05,
            wall_time_sec=2.0,
            cpu_time_sec=2.0,
            peak_rss_bytes=120 * mb,
        ),
        _make_obs(
            CodeState.AGENT,
            "inst1",
            1,
            status=RunStatus.TIMED_OUT,
            valid=False,
            normalized_gap=None,
            wall_time_sec=10.0,
            cpu_time_sec=10.0,
            peak_rss_bytes=150 * mb,
        ),
    ]

    report = compute_outcome_metrics(observations)

    # Base checks
    assert report.base.reliability.total_runs == 2
    assert report.base.reliability.valid_runs == 2
    assert math.isclose(report.base.reliability.success_rate, 1.0)
    assert report.base.performance.mean_normalized_gap == pytest.approx(0.15)
    assert report.base.resource.mean_peak_rss_mb == pytest.approx(150.0)

    # Agent checks
    assert report.agent.reliability.total_runs == 2
    assert report.agent.reliability.valid_runs == 1
    assert math.isclose(report.agent.reliability.success_rate, 0.5)
    assert report.agent.reliability.failure_counts == {"timed_out": 1}
    assert report.agent.performance.mean_normalized_gap == pytest.approx(0.05)
    # Resource metrics include all runs with telemetry (120 and 150 MB -> mean 135 MB)
    assert report.agent.resource.mean_peak_rss_mb == pytest.approx(135.0)

    # Performance: SGM checks
    # Base operational capped: times [4.0, 9.0], shift 1.0 -> sqrt(50) - 1 ≈ 6.071068
    assert report.base.performance.operational_capped_sgm_sec == pytest.approx(
        math.sqrt(50.0) - 1.0
    )
    # Agent operational capped: times [2.0, 10.0 (capped)],
    # shift 1.0 -> sqrt(33) - 1 ≈ 4.744563
    assert report.agent.performance.operational_capped_sgm_sec == pytest.approx(
        math.sqrt(33.0) - 1.0
    )

    # Common-success SGM: only seed 0 succeeded for both (times 4.0 vs 2.0)
    assert report.comparison.common_success_runs == 1
    assert report.base.performance.common_success_sgm_sec == pytest.approx(4.0)
    assert report.agent.performance.common_success_sgm_sec == pytest.approx(2.0)
    assert report.comparison.common_success_speedup == pytest.approx(2.0)
    assert report.comparison.operational_speedup == pytest.approx(
        (math.sqrt(50.0) - 1.0) / (math.sqrt(33.0) - 1.0)
    )

    # Comparison checks
    assert report.comparison.gap_reduction == pytest.approx(0.10)
    assert report.comparison.delta_success_rate == pytest.approx(-0.50)
    assert report.comparison.peak_rss_ratio == pytest.approx(135.0 / 150.0)
    assert report.comparison.paired_runs_evaluated == 2

    # Formatting test
    table_str = format_outcome_report_table(report)
    assert "[Performance]" in table_str
    assert "Operational Capped SGM" in table_str
    assert "Common-Success SGM" in table_str
    assert "[Reliability]" in table_str
    assert "[Resource Efficiency]" in table_str
    assert "Success Rate" in table_str


def test_cli_report_command(tmp_path: Path) -> None:
    parquet_path = tmp_path / "trials.parquet"
    observations = [
        _make_obs(
            CodeState.BASE,
            "inst1",
            0,
            normalized_gap=0.10,
            wall_time_sec=4.0,
            cpu_time_sec=4.0,
            peak_rss_bytes=100 * 1024 * 1024,
        ),
        _make_obs(
            CodeState.AGENT,
            "inst1",
            0,
            normalized_gap=0.08,
            wall_time_sec=2.0,
            cpu_time_sec=2.0,
            peak_rss_bytes=90 * 1024 * 1024,
        ),
    ]
    ObservationStore.write(parquet_path, observations)

    runner = CliRunner()
    result = runner.invoke(app, ["report", str(parquet_path)])
    assert result.exit_code == 0
    assert "Outcome 3D" in result.output
    assert "Performance" in result.output
    assert "Sensitivity" in result.output

    # Test JSON mode
    json_result = runner.invoke(app, ["report", str(parquet_path), "--json"])
    assert json_result.exit_code == 0
    assert '"gap_reduction"' in json_result.output
    assert '"sensitivity"' in json_result.output
