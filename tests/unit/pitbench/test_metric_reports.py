from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml
from typer.testing import CliRunner

from pitbench.cli.main import app
from pitbench.evaluator.storage import ObservationStore
from pitbench.metrics.nuisance_report import report_nuisance_results
from pitbench.metrics.performance_report import (
    ExactPerformanceReport,
    ExactRunOutcome,
    ExactRunOutcomeKind,
    PerformanceClassification,
    compute_exact_performance_report,
    compute_exact_run_outcome,
    compute_performance_report,
    compute_task_performance_report,
    format_performance_report,
)
from pitbench.metrics.resource_report import (
    compute_resource_reports,
    format_resource_report,
)
from pitbench.repositories.base import NormalizedSolverOutput
from pitbench.schema.observation import (
    CodeState,
    ExpectedRun,
    ExpectedRunGrid,
    RunObservation,
    RunStatus,
    SolverTermination,
)
from pitbench.schema.task import ExactTimeBasis, PitBenchTask
from pitbench.solver_drivers.common import process_resources, write_result
from pitbench.solver_drivers.external_runner import execute as execute_external_runner

ROOT = Path(__file__).resolve().parents[3]


def exact_observation(**changes) -> RunObservation:
    values = {
        "task_id": "exact-performance",
        "code_state": CodeState.AGENT,
        "instance_set": "judge_id",
        "instance_set_kind": "judge_id",
        "instance_id": "instance",
        "solver_seed": 0,
        "budget_sec": 10.0,
        "status": RunStatus.COMPLETED,
        "valid": True,
        "objective": 42.0,
        "reported_objective": 42.0,
        "optimal_or_bks": 42.0,
        "cpu_time_sec": 3.0,
        "wall_time_sec": 9.0,
        "solver_termination": SolverTermination.OPTIMAL,
    }
    values.update(changes)
    return RunObservation(**values)


def exact_expected_grid(
    *,
    instance_ids: tuple[str, ...] = ("i1", "i2"),
    solver_seeds: tuple[int, ...] = (0,),
    budgets_sec: tuple[float, ...] = (10.0,),
) -> ExpectedRunGrid:
    return ExpectedRunGrid(
        task_id="exact-performance",
        runs=[
            ExpectedRun(
                task_id="exact-performance",
                code_state=code_state,
                instance_set="judge_id",
                instance_set_kind="judge_id",
                instance_id=instance_id,
                solver_seed=solver_seed,
                budget_sec=budget_sec,
            )
            for instance_id in instance_ids
            for solver_seed in solver_seeds
            for budget_sec in budgets_sec
            for code_state in CodeState
        ],
    )


def exact_panel_observation(
    code_state: CodeState,
    instance_id: str,
    solver_seed: int,
    *,
    cpu_time_sec: float,
    solved: bool,
    budget_sec: float = 10.0,
) -> RunObservation:
    return exact_observation(
        code_state=code_state,
        instance_id=instance_id,
        solver_seed=solver_seed,
        budget_sec=budget_sec,
        cpu_time_sec=cpu_time_sec,
        solver_termination=(
            SolverTermination.OPTIMAL if solved else SolverTermination.TIME_LIMIT
        ),
    )


def test_verified_exact_solve_uses_cpu_time_only() -> None:
    assert compute_exact_run_outcome(exact_observation()) == ExactRunOutcome(
        kind=ExactRunOutcomeKind.VERIFIED_SOLVED,
        solved=True,
        penalized_runtime_sec=3.0,
    )


def test_parallel_exact_solve_uses_solve_wall_time() -> None:
    assert compute_exact_run_outcome(
        exact_observation(),
        time_basis=ExactTimeBasis.SOLVE_WALL_TIME,
    ) == ExactRunOutcome(
        kind=ExactRunOutcomeKind.VERIFIED_SOLVED,
        solved=True,
        penalized_runtime_sec=9.0,
    )


def test_exact_run_evidence_survives_observation_storage(tmp_path: Path) -> None:
    path = tmp_path / "exact.parquet"
    observation = exact_observation()

    ObservationStore.write(path, [observation])

    assert ObservationStore.read(path) == [observation]


def test_optimum_incumbent_without_optimal_termination_receives_par2() -> None:
    outcome = compute_exact_run_outcome(
        exact_observation(solver_termination=SolverTermination.TIME_LIMIT)
    )

    assert outcome == ExactRunOutcome(
        kind=ExactRunOutcomeKind.SOLVER_ORIGIN_UNSOLVED,
        solved=False,
        penalized_runtime_sec=20.0,
    )


@pytest.mark.parametrize(
    "status",
    [
        RunStatus.TIMED_OUT,
        RunStatus.OUT_OF_MEMORY,
        RunStatus.CRASHED,
        RunStatus.SOLVER_ERROR,
        RunStatus.NO_SOLUTION,
        RunStatus.OUTPUT_ERROR,
    ],
)
def test_exact_solver_origin_failures_receive_par2(status: RunStatus) -> None:
    assert compute_exact_run_outcome(exact_observation(status=status)) == (
        ExactRunOutcome(
            kind=ExactRunOutcomeKind.SOLVER_ORIGIN_UNSOLVED,
            solved=False,
            penalized_runtime_sec=20.0,
        )
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"status": RunStatus.INVALID},
        {"valid": False},
        {"objective": None},
        {"reported_objective": None},
        {"objective": 41.0},
    ],
)
def test_false_exact_optimality_claim_is_a_qualification_failure(changes) -> None:
    assert compute_exact_run_outcome(exact_observation(**changes)) == ExactRunOutcome(
        kind=ExactRunOutcomeKind.QUALIFICATION_FAILURE
    )


@pytest.mark.parametrize(
    "observation",
    [
        None,
        exact_observation(status=RunStatus.BUILD_FAILED),
        exact_observation(status=RunStatus.INFRASTRUCTURE_ERROR),
        exact_observation(optimal_or_bks=None),
        exact_observation(cpu_time_sec=None),
        exact_observation(cpu_time_sec=math.inf),
        exact_observation(cpu_time_sec=-1.0),
    ],
)
def test_missing_exact_run_or_evaluator_evidence_is_infrastructure_missing(
    observation: RunObservation | None,
) -> None:
    assert compute_exact_run_outcome(observation) == ExactRunOutcome(
        kind=ExactRunOutcomeKind.INFRASTRUCTURE_MISSING
    )


def test_verified_optimal_result_above_published_bks_receives_par2() -> None:
    assert compute_exact_run_outcome(
        exact_observation(objective=43.0, reported_objective=43.0)
    ) == ExactRunOutcome(
        kind=ExactRunOutcomeKind.SOLVER_ORIGIN_UNSOLVED,
        solved=False,
        penalized_runtime_sec=20.0,
    )


def test_verified_optimal_result_below_published_bks_is_an_oracle_discrepancy() -> None:
    assert compute_exact_run_outcome(
        exact_observation(objective=41.0, reported_objective=41.0)
    ) == ExactRunOutcome(kind=ExactRunOutcomeKind.ORACLE_DISCREPANCY)


def test_oracle_discrepancy_pauses_exact_aggregation() -> None:
    grid = exact_expected_grid(instance_ids=("i1",))
    observations = [
        exact_panel_observation(
            code_state,
            "i1",
            0,
            cpu_time_sec=3,
            solved=True,
        )
        for code_state in CodeState
    ]
    observations[0] = observations[0].model_copy(
        update={"objective": 41.0, "reported_objective": 41.0}
    )

    report = compute_exact_performance_report(
        observations,
        primary_budget_sec=10,
        budgets_sec=(10,),
        expected_run_grid=grid,
    )

    assert report.classification is PerformanceClassification.INCOMPLETE
    assert report.primary.base.outcome_counts["oracle_discrepancy"] == 1


def test_exact_solved_coverage_dominates_favorable_penalized_runtime() -> None:
    grid = exact_expected_grid(instance_ids=("i1",), solver_seeds=(0, 1, 2, 3))
    observations = []
    for solver_seed in (0, 1, 2, 3):
        observations.extend(
            [
                exact_panel_observation(
                    CodeState.BASE,
                    "i1",
                    solver_seed,
                    cpu_time_sec=10,
                    solved=solver_seed < 3,
                ),
                exact_panel_observation(
                    CodeState.AGENT,
                    "i1",
                    solver_seed,
                    cpu_time_sec=0,
                    solved=solver_seed < 2,
                ),
            ]
        )

    report = compute_exact_performance_report(
        observations,
        primary_budget_sec=10,
        budgets_sec=(10,),
        expected_run_grid=grid,
    )

    assert report.classification is PerformanceClassification.REGRESSED
    assert report.primary.base.verified_solved_coverage == pytest.approx(0.75)
    assert report.primary.agent.verified_solved_coverage == pytest.approx(0.5)
    assert report.primary.agent.penalized_average_runtime_sec < (
        report.primary.base.penalized_average_runtime_sec
    )


def test_equal_exact_coverage_uses_paired_penalized_runtime_interval() -> None:
    grid = exact_expected_grid()
    observations = [
        exact_panel_observation(
            code_state,
            instance_id,
            0,
            cpu_time_sec=5 if code_state == CodeState.BASE else 2,
            solved=True,
        )
        for instance_id in ("i1", "i2")
        for code_state in CodeState
    ]

    report = compute_exact_performance_report(
        observations,
        primary_budget_sec=10,
        budgets_sec=(10,),
        expected_run_grid=grid,
    )

    assert report.classification is PerformanceClassification.IMPROVED
    assert report.primary.base.verified_solved_coverage == 1
    assert report.primary.agent.verified_solved_coverage == 1
    assert report.primary.paired.mean_penalized_runtime_reduction_sec == 3
    interval = report.primary.paired.mean_penalized_runtime_reduction_ci95
    assert interval is not None
    assert (interval.lower, interval.upper) == pytest.approx((3, 3))


def test_exact_bootstrap_resamples_per_instance_seed_means() -> None:
    grid = exact_expected_grid(solver_seeds=(0, 1))
    agent_runtimes = {
        ("i1", 0): 10,
        ("i1", 1): 0,
        ("i2", 0): 0,
        ("i2", 1): 0,
    }
    observations = []
    for instance_id in ("i1", "i2"):
        for solver_seed in (0, 1):
            observations.extend(
                [
                    exact_panel_observation(
                        CodeState.BASE,
                        instance_id,
                        solver_seed,
                        cpu_time_sec=10,
                        solved=True,
                    ),
                    exact_panel_observation(
                        CodeState.AGENT,
                        instance_id,
                        solver_seed,
                        cpu_time_sec=agent_runtimes[(instance_id, solver_seed)],
                        solved=True,
                    ),
                ]
            )

    report = compute_exact_performance_report(
        observations,
        primary_budget_sec=10,
        budgets_sec=(10,),
        expected_run_grid=grid,
    )

    paired = report.primary.paired
    assert paired.mean_penalized_runtime_reduction_sec == pytest.approx(7.5)
    interval = paired.mean_penalized_runtime_reduction_ci95
    assert interval is not None
    assert (interval.lower, interval.upper) == pytest.approx((5, 10))
    assert interval.method == "instance bootstrap over per-instance seed means"
    assert interval.resamples == 5000


def test_exact_missing_run_keeps_the_declared_grid_and_is_incomplete() -> None:
    grid = exact_expected_grid()
    observations = [
        exact_panel_observation(
            code_state,
            instance_id,
            0,
            cpu_time_sec=3,
            solved=True,
        )
        for instance_id in ("i1", "i2")
        for code_state in CodeState
        if not (instance_id == "i2" and code_state == CodeState.AGENT)
    ]

    report = compute_exact_performance_report(
        observations,
        primary_budget_sec=10,
        budgets_sec=(10,),
        expected_run_grid=grid,
    )

    assert report.classification is PerformanceClassification.INCOMPLETE
    assert report.primary.agent.expected_run_count == 2
    assert report.primary.agent.complete is False
    assert report.primary.agent.outcome_counts == {
        "verified_solved": 1,
        "solver_origin_unsolved": 0,
        "oracle_discrepancy": 0,
        "qualification_failure": 0,
        "infrastructure_missing": 1,
    }
    assert report.primary.agent.penalized_average_runtime_sec is None


def test_exact_qualification_failure_does_not_enter_ordinary_aggregate() -> None:
    grid = exact_expected_grid(instance_ids=("i1",))
    observations = [
        exact_panel_observation(
            CodeState.BASE,
            "i1",
            0,
            cpu_time_sec=3,
            solved=True,
        ),
        exact_panel_observation(
            CodeState.AGENT,
            "i1",
            0,
            cpu_time_sec=1,
            solved=True,
        ).model_copy(update={"reported_objective": 41}),
    ]

    report = compute_exact_performance_report(
        observations,
        primary_budget_sec=10,
        budgets_sec=(10,),
        expected_run_grid=grid,
    )

    assert report.classification is PerformanceClassification.INCOMPLETE
    assert report.primary.agent.outcome_counts["qualification_failure"] == 1
    assert report.primary.agent.verified_solved_coverage is None
    assert report.primary.agent.penalized_average_runtime_sec is None


def test_exact_report_rejects_duplicate_and_out_of_grid_observations() -> None:
    grid = exact_expected_grid(instance_ids=("i1",))
    base = exact_panel_observation(
        CodeState.BASE,
        "i1",
        0,
        cpu_time_sec=3,
        solved=True,
    )
    with pytest.raises(ValueError, match="duplicate exact run observation"):
        compute_exact_performance_report(
            [base, base],
            primary_budget_sec=10,
            budgets_sec=(10,),
            expected_run_grid=grid,
        )

    with pytest.raises(ValueError, match="outside the expected run grid"):
        compute_exact_performance_report(
            [base.model_copy(update={"solver_seed": 99})],
            primary_budget_sec=10,
            budgets_sec=(10,),
            expected_run_grid=grid,
        )


def test_format_exact_performance_report_states_coverage_priority() -> None:
    grid = exact_expected_grid(instance_ids=("i1",))
    observations = [
        exact_panel_observation(
            code_state,
            "i1",
            0,
            cpu_time_sec=3,
            solved=True,
        )
        for code_state in CodeState
    ]
    report = compute_exact_performance_report(
        observations,
        primary_budget_sec=10,
        budgets_sec=(10,),
        expected_run_grid=grid,
    )

    assert isinstance(report, ExactPerformanceReport)
    rendered = format_performance_report(report)
    assert "Verified exact-solver performance" in rendered
    assert "Coverage is primary" in rendered
    assert "PAR-2" in rendered


def test_task_performance_dispatch_keeps_cp_sat_construction_separate() -> None:
    task = PitBenchTask.from_yaml(ROOT / "configs/tasks/ortools_v9_15.yaml")

    assert compute_task_performance_report([], task=task) is None


def performance_observation(
    code_state: CodeState,
    instance_set_kind: str,
    instance_id: str,
    normalized_gap: float | None,
    *,
    solver_seed: int = 0,
    budget_sec: float = 5,
    valid: bool = True,
    status: RunStatus = RunStatus.COMPLETED,
) -> RunObservation:
    return RunObservation(
        task_id="performance",
        code_state=code_state,
        instance_set=instance_set_kind,
        instance_set_kind=instance_set_kind,
        instance_id=instance_id,
        instance_seed=17,
        solver_seed=solver_seed,
        budget_sec=budget_sec,
        status=status,
        valid=valid,
        normalized_gap=normalized_gap,
    )


def performance_paired_panel() -> list[RunObservation]:
    observations = []
    for instance_set_kind, rows in {
        "judge_id": (("i1", 0.11, 0.07), ("i2", 0.21, 0.16)),
        "judge_shift": (("s1", 0.13, 0.11), ("s2", 0.18, 0.17)),
    }.items():
        for instance_id, base_gap, agent_gap in rows:
            for solver_seed in (0, 1):
                observations.extend(
                    [
                        performance_observation(
                            CodeState.BASE,
                            instance_set_kind,
                            instance_id,
                            base_gap,
                            solver_seed=solver_seed,
                        ),
                        performance_observation(
                            CodeState.AGENT,
                            instance_set_kind,
                            instance_id,
                            agent_gap,
                            solver_seed=solver_seed,
                        ),
                    ]
                )
    return observations


def test_performance_report_pairs_seeds_and_bootstraps_instances() -> None:
    report = compute_performance_report(
        performance_paired_panel(),
        primary_budget_sec=5,
    )

    assert report.classification is PerformanceClassification.IMPROVED
    assert report.primary_budget_sec == 5
    assert report.budgets_sec == [5]
    assert set(report.by_budget) == {"5"}
    assert report.primary.base.mean_normalized_gap == pytest.approx(0.16)
    assert report.primary.agent.mean_normalized_gap == pytest.approx(0.115)
    paired = report.primary.paired
    assert paired.paired_instances == 2
    assert paired.mean_gap_reduction == pytest.approx(0.045)
    assert paired.mean_gap_reduction_ci95 is not None
    assert paired.mean_gap_reduction_ci95.lower == pytest.approx(0.04)
    assert paired.mean_gap_reduction_ci95.upper == pytest.approx(0.05)

    payload = report.model_dump()
    assert set(payload) == {
        "classification",
        "primary_budget_sec",
        "budgets_sec",
        "primary",
        "by_budget",
    }
    assert set(payload["primary"]["base"]) == {
        "mean_normalized_gap",
        "median_normalized_gap",
        "p95_normalized_gap",
        "mean_ci95",
    }
    assert set(payload["primary"]["paired"]) == {
        "paired_instances",
        "mean_gap_reduction",
        "mean_gap_reduction_ci95",
    }


def test_invalid_runs_do_not_change_performance_estimates() -> None:
    observations = performance_paired_panel()
    original = compute_performance_report(observations, primary_budget_sec=5)
    observations.append(
        performance_observation(
            CodeState.AGENT,
            "judge_id",
            "failed",
            None,
            solver_seed=2,
            valid=False,
            status=RunStatus.TIMED_OUT,
        )
    )

    assert compute_performance_report(observations, primary_budget_sec=5) == original


def test_performance_report_excludes_generalization_observations() -> None:
    judge_id_observations = [
        performance_observation(CodeState.BASE, "judge_id", "id", 0.20),
        performance_observation(CodeState.AGENT, "judge_id", "id", 0.10),
    ]
    judge_shift_observations = [
        performance_observation(CodeState.BASE, "judge_shift", "shift", 0.10),
        performance_observation(CodeState.AGENT, "judge_shift", "shift", 0.90),
    ]

    judge_id_report = compute_performance_report(
        judge_id_observations,
        primary_budget_sec=5,
    )
    combined_report = compute_performance_report(
        [*judge_id_observations, *judge_shift_observations],
        primary_budget_sec=5,
    )

    assert combined_report == judge_id_report


@pytest.mark.parametrize(
    ("gap_reductions", "expected_classification"),
    (
        ([0.05], PerformanceClassification.IMPROVED),
        ([-0.05], PerformanceClassification.REGRESSED),
        ([-0.05, 0.05], PerformanceClassification.INCONCLUSIVE),
    ),
)
def test_performance_classification_uses_primary_gain_interval(
    gap_reductions: list[float],
    expected_classification: PerformanceClassification,
) -> None:
    observations = []
    for instance_index, gap_reduction in enumerate(gap_reductions):
        instance_id = f"i{instance_index}"
        observations.extend(
            [
                performance_observation(CodeState.BASE, "judge_id", instance_id, 0.2),
                performance_observation(
                    CodeState.AGENT,
                    "judge_id",
                    instance_id,
                    0.2 - gap_reduction,
                ),
            ]
        )

    report = compute_performance_report(observations, primary_budget_sec=5)

    assert report.classification is expected_classification


def test_performance_classification_is_incomplete_without_paired_interval() -> None:
    report = compute_performance_report(
        [performance_observation(CodeState.BASE, "judge_id", "i1", 0.2)],
        primary_budget_sec=5,
    )

    assert report.classification is PerformanceClassification.INCOMPLETE


def test_gap_summaries_weight_per_instance_seed_means_equally() -> None:
    observations = []
    for solver_seed in (0, 1, 2):
        observations.extend(
            [
                performance_observation(
                    CodeState.BASE,
                    "judge_id",
                    "many",
                    0,
                    solver_seed=solver_seed,
                ),
                performance_observation(
                    CodeState.AGENT,
                    "judge_id",
                    "many",
                    0,
                    solver_seed=solver_seed,
                ),
            ]
        )
    observations.extend(
        [
            performance_observation(CodeState.BASE, "judge_id", "one", 1),
            performance_observation(CodeState.AGENT, "judge_id", "one", 0),
        ]
    )

    report = compute_performance_report(observations, primary_budget_sec=5)

    assert report.primary.base.mean_normalized_gap == pytest.approx(0.5)
    assert report.primary.base.median_normalized_gap == pytest.approx(0.5)
    assert report.primary.base.p95_normalized_gap == pytest.approx(0.95)
    assert report.primary.agent.mean_normalized_gap == pytest.approx(0)
    assert report.primary.paired.mean_gap_reduction == pytest.approx(0.5)
    for interval in (
        report.primary.base.mean_ci95,
        report.primary.agent.mean_ci95,
        report.primary.paired.mean_gap_reduction_ci95,
    ):
        assert interval is not None
        assert interval.level == 0.95
        assert interval.resamples == 5000
        assert interval.method == "instance bootstrap over per-instance seed means"


def test_gap_summaries_exclude_instances_without_valid_seeds() -> None:
    observations = performance_paired_panel()
    original = compute_performance_report(observations, primary_budget_sec=5)
    observations.extend(
        [
            performance_observation(
                CodeState.BASE,
                "judge_id",
                "no-valid-seeds",
                None,
                valid=False,
                status=RunStatus.TIMED_OUT,
            ),
            performance_observation(
                CodeState.AGENT,
                "judge_id",
                "no-valid-seeds",
                None,
                valid=False,
                status=RunStatus.CRASHED,
            ),
        ]
    )

    report = compute_performance_report(observations, primary_budget_sec=5)

    assert report.primary.base.mean_normalized_gap == (
        original.primary.base.mean_normalized_gap
    )
    assert report.primary.agent.mean_normalized_gap == (
        original.primary.agent.mean_normalized_gap
    )
    assert report.primary.paired == original.primary.paired


def test_declared_primary_budget_is_not_replaced_by_larger_budget() -> None:
    observations = performance_paired_panel()
    observations.extend(
        [
            performance_observation(
                CodeState.BASE,
                "judge_id",
                "diagnostic",
                0.2,
                budget_sec=10,
            ),
            performance_observation(
                CodeState.AGENT,
                "judge_id",
                "diagnostic",
                0.1,
                budget_sec=10,
            ),
        ]
    )

    report = compute_performance_report(observations, primary_budget_sec=5)

    assert report.budgets_sec == [5, 10]
    assert report.primary_budget_sec == 5
    assert report.primary.budget_sec == 5


def test_performance_report_rejects_missing_primary_budget() -> None:
    with pytest.raises(
        ValueError,
        match="declared primary budget 10s has no observations",
    ):
        compute_performance_report(performance_paired_panel(), primary_budget_sec=10)


def test_format_performance_report_is_performance_only() -> None:
    rendered = format_performance_report(
        compute_performance_report(performance_paired_panel(), primary_budget_sec=5)
    )

    assert "Fixed-budget performance" in rendered
    assert "Classification: improved" in rendered
    assert "Paired instances" in rendered
    assert "instance bootstrap over per-instance seed means" in rendered
    assert "95% CI" in rendered
    assert "valid" not in rendered.lower()
    assert "hidden" not in rendered.lower()
    assert "repeatability" not in rendered.lower()


def test_report_command_supports_structured_performance_output(tmp_path: Path) -> None:
    observations_path = tmp_path / "trials.parquet"
    ObservationStore.write(observations_path, performance_paired_panel())
    task_config_path = tmp_path / "task-config.yaml"
    task_payload = yaml.safe_load(
        (ROOT / "configs/tasks/pyvrp_v0_14_0.yaml").read_text()
    )
    task_payload["task_id"] = "performance"
    task_payload["evaluation"]["primary_budget_sec"] = 5
    task_config_path.write_text(yaml.safe_dump(task_payload, sort_keys=False))
    runner = CliRunner()

    text_result = runner.invoke(
        app,
        ["report", str(observations_path), "--task-config", str(task_config_path)],
    )
    assert text_result.exit_code == 0
    assert "Fixed-budget performance" in text_result.output
    assert "Resource usage" in text_result.output

    json_result = runner.invoke(
        app,
        [
            "report",
            str(observations_path),
            "--task-config",
            str(task_config_path),
            "--json",
        ],
    )
    assert json_result.exit_code == 0
    assert '"performance"' in json_result.output
    assert '"resource_usage"' in json_result.output
    assert '"expected_seed_count": 30' in json_result.output
    assert '"mean_gap_reduction"' in json_result.output
    assert '"sensitivity"' not in json_result.output


def test_report_command_dispatches_exact_performance(tmp_path: Path) -> None:
    observations_path = tmp_path / "trials.parquet"
    observations = [
        exact_panel_observation(
            code_state,
            "i1",
            0,
            cpu_time_sec=3,
            solved=True,
        )
        for code_state in CodeState
    ]
    ObservationStore.write(observations_path, observations)
    (tmp_path / "expected-run-grid.json").write_text(
        exact_expected_grid(instance_ids=("i1",)).model_dump_json(indent=2) + "\n"
    )
    task_payload = yaml.safe_load(
        (ROOT / "configs/tasks/highs_v1_15_1.yaml").read_text()
    )
    task_payload["task_id"] = "exact-performance"
    task_payload["evaluation"].update(
        {
            "operational_reliability": False,
            "budgets_sec": [10],
            "primary_budget_sec": 10,
            "solver_seeds": [0],
        }
    )
    task_config_path = tmp_path / "task-config.yaml"
    task_config_path.write_text(yaml.safe_dump(task_payload, sort_keys=False))

    text_result = CliRunner().invoke(
        app,
        ["report", str(observations_path), "--task-config", str(task_config_path)],
    )
    assert text_result.exit_code == 0
    assert "Verified exact-solver performance" in text_result.output
    assert "Coverage is primary" in text_result.output

    json_result = CliRunner().invoke(
        app,
        [
            "report",
            str(observations_path),
            "--task-config",
            str(task_config_path),
            "--json",
        ],
    )
    assert json_result.exit_code == 0
    assert '"verified_solved_coverage": 1.0' in json_result.output
    assert '"penalized_average_runtime_sec": 3.0' in json_result.output


def test_report_command_rejects_performance_task_config_mismatch(
    tmp_path: Path,
) -> None:
    observations_path = tmp_path / "trials.parquet"
    ObservationStore.write(observations_path, performance_paired_panel())
    task_config_path = ROOT / "configs/tasks/pyvrp_v0_14_0.yaml"

    result = CliRunner().invoke(
        app,
        ["report", str(observations_path), "--task-config", str(task_config_path)],
    )

    assert result.exit_code == 2
    assert "task ID mismatch" in result.output


# Tests consolidated from tests/unit/pitbench/test_nuisance_report.py


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def directory_panel(source):
    jobs = [
        {
            "run_id": f"case/seed/{seed}/budget-{budget}",
            "instance": "case",
            "axis": "seed",
            "replicate": seed,
            "solver_seed": seed,
            "budget_sec": budget,
        }
        for budget in [5, 15]
        for seed in [0, 1]
    ]
    write(
        source / "experiment.json",
        {
            "experiment": "Another solver <release>",
            "code_state": "base",
            "instances": [{"name": "case"}],
            "budgets_sec": [5, 15],
            "jobs": jobs,
        },
    )
    write(
        source / "runs" / jobs[0]["run_id"] / "result.json",
        {
            **jobs[0],
            "execution_status": "completed",
            "verified_feasible": True,
            "verification": {"feasible": True, "objective": 0},
            "solver_runtime_sec": 3,
            "solution_value_valid": True,
        },
    )
    return jobs


def embedded(output):
    text = (output / "report.html").read_text()
    return json.loads(text.split("const DATA=", 1)[1].split(";\nconst names=", 1)[0])


def test_directory_report_uses_manifest_counts_budgets_and_identity(tmp_path):
    source, output = tmp_path / "source", tmp_path / "report"
    directory_panel(source)
    summary = report_nuisance_results(source, output)
    assert summary["expected_runs"] == 4
    assert summary["completed_runs"] == 1
    assert summary["missing_runs"] == 3
    assert not summary["complete"]
    assert [(item["expected"], item["completed"]) for item in summary["groups"]] == [
        (2, 1),
        (2, 0),
    ]
    assert summary["groups"][0]["verified_objective"] == {
        "count": 1,
        "min": 0,
        "max": 0,
    }
    assert summary["groups"][1]["verified_objective"] is None
    data = embedded(output)
    assert data["manifest"]["budgets_sec"] == [5, 15]
    assert data["records"][0]["verification"]["objective"] == 0
    page = (output / "report.html").read_text()
    assert "Another solver &lt;release&gt;" in page
    assert "HiGHS" not in page and "2026-09-09" not in page
    assert "（30 次）" not in page and "/29*" not in page


def test_directory_report_rejects_wrong_run_in_result_file(tmp_path):
    source = tmp_path / "source"
    jobs = directory_panel(source)
    path = source / "runs" / jobs[0]["run_id"] / "result.json"
    record = json.loads(path.read_text())
    record["solver_seed"] = 999
    write(path, record)
    with pytest.raises(ValueError, match="outside the declared run grid"):
        report_nuisance_results(source, tmp_path / "report")


@pytest.mark.parametrize("manifest_name", ["experiment.json", "details.json"])
def test_judge_report_separates_states_and_preserves_failure(tmp_path, manifest_name):
    source, output = tmp_path / "source", tmp_path / "report"
    write(
        source / manifest_name,
        {
            "task_id": "a_configured_task",
            "code_states": ["base", "agent"],
            "budgets_sec": [5, 10],
            "solver_seed": 9,
            "configuration": {"solver_seed": 9},
        },
    )
    write(
        source / "transformations.json",
        {
            "case__transform": {
                "original_instance_id": "case",
                "transform_id": "permutation_0",
            },
        },
    )
    records = []
    for state in CodeState:
        valid = state is CodeState.BASE
        observation = RunObservation(
            task_id="a_configured_task",
            code_state=state,
            instance_set="agent_dev",
            instance_set_kind="agent_dev",
            instance_id="case__transform",
            solver_seed=9,
            budget_sec=5,
            status=RunStatus.COMPLETED if valid else RunStatus.CRASHED,
            valid=valid,
            objective=4 if valid else None,
        )
        records.append(
            {
                "observation": observation.model_dump(mode="json"),
                "verification": {
                    "mapped_original": {
                        "feasible": valid,
                        "objective": 4 if valid else None,
                    },
                    "objective_preserved": valid,
                },
                "artifacts": {"solution": "solution.json" if valid else None},
            }
        )
    checkpoint = source / "results.jsonl"
    checkpoint.write_text("".join(json.dumps(item) + "\n" for item in records))
    original = checkpoint.read_bytes()
    summary = report_nuisance_results(source, output)
    assert summary["expected_runs"] == 4 and summary["completed_runs"] == 2
    assert summary["execution_statuses"] == {"completed": 1, "crashed": 1}
    groups = {
        (item["code_state"], item["budget_sec"]): item for item in summary["groups"]
    }
    assert groups[("base", 5)]["verified_feasible"] == 1
    assert groups[("agent", 5)]["verified_objective"] is None
    assert groups[("base", 10)]["missing"] == 1
    assert groups[("base", 5)]["solver_gap"] is None
    assert len(embedded(output)["records"]) == 2
    assert checkpoint.read_bytes() == original
    checkpoint.write_text(original.decode() + json.dumps(records[0]) + "\n")
    with pytest.raises(ValueError, match="duplicate nuisance observation"):
        report_nuisance_results(source, output)


# Tests consolidated from tests/unit/pitbench/test_resource_report.py


def observation(state, instance, seed, memory, **kwargs):
    values = dict(
        task_id="resource-test",
        code_state=state,
        instance_set="judge_id",
        instance_id=instance,
        solver_seed=seed,
        budget_sec=5,
        status=RunStatus.COMPLETED,
        valid=True,
        peak_rss_bytes=memory,
        cpu_time_sec=2.0,
        normalized_gap=0.1,
        resource_scope="worker_process_start_through_solver_return",
    )
    values.update(kwargs)
    return RunObservation(**values)


def test_ratios_use_thirty_paired_seed_means_then_equal_instance_geometric_mean():
    observations = []
    for seed in range(30):
        observations.extend(
            [
                observation(CodeState.BASE, "small", seed, 1 if seed % 2 else 9),
                observation(CodeState.AGENT, "small", seed, 1, normalized_gap=0.3),
                observation(CodeState.BASE, "large", seed, 100),
                observation(CodeState.AGENT, "large", seed, 500),
            ]
        )
    report, details = compute_resource_reports(observations[::-1], primary_budget_sec=5)
    instances = {
        item.instance_id: item for item in details.by_instance_set["judge_id"]["5"]
    }
    # Ratio of seed means is 1/5, not mean(1/9, 1/1); larger absolute
    # memory instances do not dominate the cross-instance relative ratio.
    assert instances["small"].memory.agent_base_ratio == pytest.approx(0.2)
    assert instances["large"].memory.agent_base_ratio == pytest.approx(5)
    aggregate = report.by_instance_set["judge_id"]["5"]
    assert aggregate.memory.geometric_mean_ratio == pytest.approx(1)
    assert aggregate.memory.complete_instance_count == 2
    assert aggregate.memory.paired_seed_count == 60
    assert instances["small"].mean_gap_change == pytest.approx(0.2)
    assert "small" not in report.model_dump_json()
    assert "30" in format_resource_report(report)

    swapped = [
        item.model_copy(
            update={
                "code_state": CodeState.AGENT
                if item.code_state == CodeState.BASE
                else CodeState.BASE
            }
        )
        for item in observations
    ]
    _, swapped_details = compute_resource_reports(swapped, primary_budget_sec=5)
    for item in swapped_details.by_instance_set["judge_id"]["5"]:
        assert item.memory.agent_base_ratio == pytest.approx(
            1 / instances[item.instance_id].memory.agent_base_ratio
        )


def test_missing_and_failed_runs_preserve_pairing_and_quality_regressions():
    observations = [
        observation(CodeState.BASE, "one", 0, 100),
        observation(
            CodeState.AGENT,
            "one",
            0,
            80,
            normalized_gap=0.4,
            solver_status="Time limit reached",
        ),
        observation(CodeState.BASE, "one", 1, 10000),  # unpaired, excluded
        observation(CodeState.BASE, "one", 2, 100),
        observation(
            CodeState.AGENT, "one", 2, 1, status=RunStatus.CRASHED, valid=False
        ),
        observation(CodeState.BASE, "one", 3, 100),
        observation(CodeState.AGENT, "one", 3, None),
    ]
    report, details = compute_resource_reports(
        observations, primary_budget_sec=5, budgets_sec=[5, 10]
    )
    instance = details.by_instance_set["judge_id"]["5"][0]
    assert instance.memory.base_mean == 100
    assert instance.memory.agent_base_ratio == 0.8
    assert instance.memory.paired_seed_count == 1
    assert instance.memory.excluded_pairs == {
        "missing_both": 26,
        "missing_agent": 1,
        "invalid_agent": 1,
        "missing_or_nonpositive_measurement": 1,
    }
    assert instance.agent_status_counts["crashed"] == 1
    assert instance.agent_status_counts["missing"] == 27
    assert instance.paired_quality_seed_count == 2
    assert instance.cpu.paired_seed_count == 2
    assert report.by_instance_set["judge_id"]["5"].memory.complete_instance_count == 0
    assert report.by_instance_set["judge_id"]["10"].memory.geometric_mean_ratio is None
    assert details.by_instance_set["judge_id"]["10"][0].missing_seed_count == 30


@pytest.mark.parametrize(
    "update,reason",
    [
        ({"resource_scope": None}, "unqualified_measurement_scope"),
        (
            {"resource_scope": "legacy_driver_and_children_at_output"},
            "unqualified_measurement_scope",
        ),
        (
            {"resource_scope": "single_native_child_lifetime"},
            "measurement_scope_mismatch",
        ),
        ({"threads": 2}, "thread_count_mismatch"),
        ({"peak_rss_bytes": 0}, "missing_or_nonpositive_measurement"),
        ({"peak_rss_bytes": -1}, "missing_or_nonpositive_measurement"),
    ],
)
def test_incompatible_or_invalid_resource_values_are_not_savings(update, reason):
    values = [
        observation(CodeState.BASE, "one", 0, 100),
        observation(CodeState.AGENT, "one", 0, 80, **update),
    ]
    report, details = compute_resource_reports(values, primary_budget_sec=5)
    assert report.by_instance_set["judge_id"]["5"].memory.geometric_mean_ratio is None
    assert (
        details.by_instance_set["judge_id"]["5"][0].memory.excluded_pairs[reason] == 1
    )


def test_cpu_nonfinite_values_do_not_break_memory_report():
    values = [
        observation(CodeState.BASE, "one", 0, 100),
        observation(CodeState.AGENT, "one", 0, 80, cpu_time_sec=math.nan),
    ]
    report, details = compute_resource_reports(values, primary_budget_sec=5)
    assert report.by_instance_set["judge_id"][
        "5"
    ].memory.geometric_mean_ratio == pytest.approx(0.8)
    assert details.by_instance_set["judge_id"]["5"][0].cpu.agent_base_ratio is None
    json.loads(report.model_dump_json())


def test_budgets_instance_sets_and_transformations_do_not_mix():
    values = []
    for instance_set, budget, ratio in [
        ("judge_id", 5, 0.5),
        ("judge_id", 10, 2),
        ("agent_dev", 5, 3),
    ]:
        values.extend(
            [
                observation(
                    CodeState.BASE,
                    "one",
                    0,
                    100,
                    instance_set=instance_set,
                    budget_sec=budget,
                ),
                observation(
                    CodeState.AGENT,
                    "one",
                    0,
                    int(100 * ratio),
                    instance_set=instance_set,
                    budget_sec=budget,
                ),
            ]
        )
    original, _ = compute_resource_reports(values, primary_budget_sec=5)
    values.append(
        observation(CodeState.AGENT, "transformed", 0, 1, equivalence_parent_id="one")
    )
    report, _ = compute_resource_reports(values, primary_budget_sec=5)
    assert report == original
    assert report.by_instance_set["judge_id"][
        "5"
    ].memory.geometric_mean_ratio == pytest.approx(0.5)
    assert report.by_instance_set["judge_id"][
        "10"
    ].memory.geometric_mean_ratio == pytest.approx(2)
    assert report.by_instance_set["agent_dev"][
        "5"
    ].memory.geometric_mean_ratio == pytest.approx(3)


def test_duplicate_or_mixed_task_records_are_rejected():
    record = observation(CodeState.BASE, "one", 0, 100)
    with pytest.raises(ValueError, match="duplicate"):
        compute_resource_reports([record, record], primary_budget_sec=5)
    with pytest.raises(ValueError, match="exactly one task"):
        compute_resource_reports(
            [record, record.model_copy(update={"task_id": "other"})],
            primary_budget_sec=5,
        )


def test_entire_missing_instances_and_instance_sets_remain_in_coverage():
    values = [
        observation(state, "present", seed, 100)
        for seed in range(30)
        for state in CodeState
    ]
    report, _ = compute_resource_reports(
        values,
        primary_budget_sec=5,
        expected_instance_counts={"judge_id": 3, "agent_dev": 10},
    )
    cell = report.by_instance_set["judge_id"]["5"]
    assert cell.instance_count == 1
    assert cell.expected_instance_count == 3
    assert cell.memory.complete_instance_count == 1
    assert report.by_instance_set["agent_dev"]["5"].memory.geometric_mean_ratio is None
    assert "memory pairs 30/90" in format_resource_report(report)


def test_native_driver_records_child_lifetime_peak(tmp_path):
    solver = tmp_path / "solver"
    solver.write_text(
        f"#!{sys.executable}\n"
        "import sys\nfrom pathlib import Path\n"
        "data = bytearray(32 * 1024 * 1024)\n"
        "for index in range(0, len(data), 4096): data[index] = 1\n"
        "for argument in sys.argv:\n"
        "    if argument.startswith('--solution_file='): Path(argument.split('=', 1)[1]).write_text('solution')\n"
        "print('Model status : Optimal\\nPrimal bound 1\\nDual bound 1\\nNodes 1')\n"
    )
    solver.chmod(0o755)
    output = tmp_path / "result.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pitbench.solver_drivers.run",
            "highs",
            "--solver",
            str(solver),
            "--instance",
            str(tmp_path / "instance.mps"),
            "--output",
            str(output),
            "--trajectory",
            str(tmp_path / "trajectory.jsonl"),
            "--seed",
            "0",
            "--budget",
            "1",
            "--threads",
            "1",
        ],
        check=True,
    )
    result = json.loads(output.read_text())
    assert result["resource_scope"] == "single_native_child_lifetime"
    assert result["peak_rss_bytes"] >= 32 * 1024 * 1024
    assert result["cpu_time_sec"] > 0
    assert result["solver_status"] == "Optimal"


def test_snapshots_keep_single_process_peak_and_freeze_before_output(tmp_path):
    own = SimpleNamespace(ru_utime=2, ru_stime=3, ru_maxrss=100)
    child = SimpleNamespace(ru_utime=7, ru_stime=11, ru_maxrss=500)
    with patch(
        "pitbench.solver_drivers.common.resource.getrusage", side_effect=[own, child]
    ):
        snapshot = process_resources()
        child_snapshot = process_resources(child_process=True)
    assert snapshot["peak_rss_bytes"] == 100 * 1024
    assert snapshot["cpu_time_sec"] == 5
    assert child_snapshot["peak_rss_bytes"] == 500 * 1024
    assert child_snapshot["cpu_time_sec"] == 18
    with patch("pitbench.solver_drivers.common.resource.getrusage", return_value=child):
        write_result(tmp_path / "result.json", started=0, valid=True, **snapshot)
    parsed = NormalizedSolverOutput.model_validate_json(
        (tmp_path / "result.json").read_text()
    )
    assert parsed.peak_rss_bytes == snapshot["peak_rss_bytes"]
    record = observation(
        CodeState.BASE,
        "one",
        0,
        parsed.peak_rss_bytes,
        resource_scope=parsed.resource_scope,
    )
    ObservationStore.write(tmp_path / "observations.parquet", [record])
    assert ObservationStore.read(tmp_path / "observations.parquet") == [record]


def test_external_runner_preserves_solver_cpu_and_child_peak(
    tmp_path: Path,
    monkeypatch,
) -> None:
    response = {
        "valid": True,
        "has_solution": True,
        "objective": 1,
        "cpu_time_sec": 0.25,
        "solver_status": "Optimal",
        "solver_termination": "optimal",
        "solution": {"bins": [[0]]},
    }
    monkeypatch.setenv("PITBENCH_TEST_RUNNER", "test-runner")
    monkeypatch.setattr(
        "pitbench.solver_drivers.external_runner.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(response),
            stderr="",
        ),
    )
    monkeypatch.setattr(
        "pitbench.solver_drivers.external_runner.process_resources",
        lambda **kwargs: {
            "cpu_time_sec": 9.0,
            "peak_rss_bytes": 1234,
            "resource_scope": "single_native_child_lifetime",
        },
    )
    output = tmp_path / "result.json"

    execute_external_runner(
        environment_key="PITBENCH_TEST_RUNNER",
        instance=tmp_path / "instance.json",
        output=output,
        trajectory=tmp_path / "trajectory.jsonl",
        seed=0,
        budget=1,
        threads=1,
    )

    result = json.loads(output.read_text())
    assert result["cpu_time_sec"] == 0.25
    assert result["peak_rss_bytes"] == 1234
    assert json.loads(output.with_suffix(".solution.json").read_text()) == {
        "bins": [[0]]
    }
