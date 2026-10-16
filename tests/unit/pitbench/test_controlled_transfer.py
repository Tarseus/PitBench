from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from pitbench.cli.main import app
from pitbench.evaluator.judge import JudgePlan
from pitbench.evaluator.storage import ObservationStore
from pitbench.metrics.controlled_transfer_report import (
    ControlledTransferConditionStatus,
    compute_controlled_transfer_report,
    format_controlled_transfer_report,
)
from pitbench.metrics.performance_report import PerformanceClassification
from pitbench.schema.observation import (
    CodeState,
    ExpectedRun,
    ExpectedRunGrid,
    RunObservation,
    RunStatus,
    SolverTermination,
)
from pitbench.schema.task import PitBenchTask

ROOT = Path(__file__).resolve().parents[3]


def _heuristic_task() -> PitBenchTask:
    task = PitBenchTask.from_yaml(ROOT / "configs/tasks/pyvrp_v0_14_0.yaml")
    payload = task.model_dump()
    payload["problem_variant"] = "capacitated_vehicle_routing"
    payload["formulation_id"] = "normalized_cvrp_json"
    payload["evaluation"]["controlled_transfer"] = {
        "conditions": [
            {
                "condition_id": "reference",
                "view": "reference",
                "instance_set": "judge_id",
                "budget_sec": 10.0,
                "generator_version": "test-v1",
            },
            {
                "condition_id": "scale-1",
                "view": "scale",
                "instance_set": "judge_shift",
                "reference_condition_id": "reference",
                "scale_path_id": "customers",
                "scale_vector": {"customers": 200.0},
                "generator_version": "test-v1",
            },
        ]
    }
    return PitBenchTask.model_validate(payload)


def _exact_task() -> PitBenchTask:
    task = PitBenchTask.from_yaml(ROOT / "configs/tasks/choco_v6_0_1.yaml")
    payload = task.model_dump()
    payload["problem_variant"] = "bin_packing"
    payload["formulation_id"] = "normalized_bin_packing_json"
    payload["evaluation"]["controlled_transfer"] = {
        "conditions": [
            {
                "condition_id": "reference",
                "view": "reference",
                "instance_set": "judge_id",
                "budget_sec": 20.0,
                "generator_version": "test-v1",
            },
            {
                "condition_id": "scale-1",
                "view": "scale",
                "instance_set": "agent_dev",
                "budget_sec": 20.0,
                "reference_condition_id": "reference",
                "scale_path_id": "items",
                "scale_vector": {"items": 250.0},
                "generator_version": "test-v1",
            },
        ]
    }
    return PitBenchTask.model_validate(payload)


def _expected_grid(task: PitBenchTask) -> ExpectedRunGrid:
    runs = []
    for condition_id, instance_set in (
        ("reference", "judge_id"),
        ("scale-1", "judge_shift"),
    ):
        for instance_id in ("one", "two"):
            for state in CodeState:
                runs.append(
                    ExpectedRun(
                        task_id=task.task_id,
                        code_state=state,
                        instance_set=instance_set,
                        instance_set_kind=(
                            "judge_id" if instance_set == "judge_id" else "judge_shift"
                        ),
                        instance_id=instance_id,
                        solver_seed=0,
                        budget_sec=10.0,
                        condition_id=condition_id,
                        template_id=f"template-{instance_id}",
                    )
                )
    return ExpectedRunGrid(task_id=task.task_id, runs=runs)


def _heuristic_observation(
    task: PitBenchTask,
    *,
    condition_id: str,
    instance_id: str,
    state: CodeState,
    gap: float,
) -> RunObservation:
    instance_set = "judge_id" if condition_id == "reference" else "judge_shift"
    return RunObservation(
        task_id=task.task_id,
        code_state=state,
        instance_set=instance_set,
        instance_set_kind=(
            "judge_id" if instance_set == "judge_id" else "judge_shift"
        ),
        instance_id=instance_id,
        solver_seed=0,
        budget_sec=10.0,
        status=RunStatus.COMPLETED,
        valid=True,
        normalized_gap=gap,
        condition_id=condition_id,
        template_id=f"template-{instance_id}",
    )


def _heuristic_observations(task: PitBenchTask) -> list[RunObservation]:
    observations = []
    for condition_id, agent_gap in (("reference", 0.1), ("scale-1", 0.119)):
        for instance_id in ("one", "two"):
            observations.extend(
                [
                    _heuristic_observation(
                        task,
                        condition_id=condition_id,
                        instance_id=instance_id,
                        state=CodeState.BASE,
                        gap=0.2,
                    ),
                    _heuristic_observation(
                        task,
                        condition_id=condition_id,
                        instance_id=instance_id,
                        state=CodeState.AGENT,
                        gap=agent_gap,
                    ),
                ]
            )
    return observations


def _add_heuristic_condition(
    task: PitBenchTask,
    expected: ExpectedRunGrid,
    observations: list[RunObservation],
    *,
    condition_id: str,
    instance_set: str,
    agent_gap: float,
) -> None:
    for instance_id in ("one", "two"):
        for state, gap in ((CodeState.BASE, 0.2), (CodeState.AGENT, agent_gap)):
            expected.runs.append(
                ExpectedRun(
                    task_id=task.task_id,
                    code_state=state,
                    instance_set=instance_set,
                    instance_set_kind="judge_shift",
                    instance_id=instance_id,
                    solver_seed=0,
                    budget_sec=10.0,
                    condition_id=condition_id,
                    template_id=f"{condition_id}-{instance_id}",
                )
            )
            observations.append(
                RunObservation(
                    task_id=task.task_id,
                    code_state=state,
                    instance_set=instance_set,
                    instance_set_kind="judge_shift",
                    instance_id=instance_id,
                    solver_seed=0,
                    budget_sec=10.0,
                    status=RunStatus.COMPLETED,
                    valid=True,
                    normalized_gap=gap,
                    condition_id=condition_id,
                    template_id=f"{condition_id}-{instance_id}",
                )
            )


def test_controlled_transfer_reports_task_native_retention() -> None:
    task = _heuristic_task()

    report = compute_controlled_transfer_report(
        _heuristic_observations(task),
        task=task,
        expected_run_grid=_expected_grid(task),
    )

    assert report is not None
    target = report.condition_reports["scale-1"]
    assert target.status is ControlledTransferConditionStatus.COMPLETE
    assert target.benefit == pytest.approx(0.081)
    assert target.retention_ratio == pytest.approx(0.81)
    assert target.supports_retention
    assert report.scale_paths["customers"].largest_continuously_supported_condition_id == "scale-1"


def test_controlled_transfer_noop_has_no_retention_ratio() -> None:
    task = _heuristic_task()
    observations = _heuristic_observations(task)
    for observation in observations:
        if observation.condition_id == "reference" and observation.code_state is CodeState.AGENT:
            observation.normalized_gap = 0.2

    report = compute_controlled_transfer_report(
        observations,
        task=task,
        expected_run_grid=_expected_grid(task),
    )

    assert report is not None
    target = report.condition_reports["scale-1"]
    assert not target.reference_improvement_established
    assert target.retention_ratio is None


def test_controlled_transfer_missing_run_is_incomplete() -> None:
    task = _heuristic_task()
    observations = _heuristic_observations(task)
    observations.pop()

    report = compute_controlled_transfer_report(
        observations,
        task=task,
        expected_run_grid=_expected_grid(task),
    )

    assert report is not None
    assert report.condition_reports["scale-1"].status is ControlledTransferConditionStatus.INCOMPLETE


def test_controlled_transfer_defaults_to_primary_budget() -> None:
    task = _heuristic_task()
    expected = _expected_grid(task)
    expected.runs.extend(
        run.model_copy(update={"budget_sec": 5.0}) for run in list(expected.runs)
    )
    observations = _heuristic_observations(task)
    observations.extend(
        observation.model_copy(
            update={
                "budget_sec": 5.0,
                "normalized_gap": (
                    0.3
                    if observation.condition_id == "scale-1"
                    and observation.code_state is CodeState.AGENT
                    else observation.normalized_gap
                ),
            }
        )
        for observation in list(observations)
    )

    report = compute_controlled_transfer_report(
        observations,
        task=task,
        expected_run_grid=expected,
    )

    assert report is not None
    target = report.condition_reports["scale-1"]
    assert target.budget_sec == 10.0
    assert target.benefit == pytest.approx(0.081)


def test_controlled_transfer_requires_a_strict_retention_margin() -> None:
    task = _heuristic_task()
    observations = _heuristic_observations(task)
    for observation in observations:
        if observation.condition_id == "scale-1" and observation.code_state is CodeState.AGENT:
            observation.normalized_gap = 0.12

    report = compute_controlled_transfer_report(
        observations,
        task=task,
        expected_run_grid=_expected_grid(task),
    )

    assert report is not None
    target = report.condition_reports["scale-1"]
    assert target.retention_margin == pytest.approx(0.0)
    assert target.retention_margin_lower_simultaneous == pytest.approx(0.0)
    assert not target.supports_retention


def test_controlled_transfer_retains_negative_structural_benefit() -> None:
    task = _heuristic_task()
    observations = _heuristic_observations(task)
    for observation in observations:
        if observation.condition_id == "scale-1" and observation.code_state is CodeState.AGENT:
            observation.normalized_gap = 0.3

    report = compute_controlled_transfer_report(
        observations,
        task=task,
        expected_run_grid=_expected_grid(task),
    )

    assert report is not None
    target = report.condition_reports["scale-1"]
    assert target.benefit == pytest.approx(-0.1)
    assert target.retention_ratio == pytest.approx(-1.0)
    assert not target.supports_retention


def test_controlled_transfer_stops_continuous_scale_and_uses_shared_bounds() -> None:
    base_task = _heuristic_task()
    payload = base_task.model_dump()
    payload["instance_sets"].extend(
        [
            {
                "name": "scale-two",
                "kind": "judge_shift",
                "instance_set_config": "private://scale-two.yaml",
                "size": 2,
            },
            {
                "name": "scale-three",
                "kind": "judge_shift",
                "instance_set_config": "private://scale-three.yaml",
                "size": 2,
            },
        ]
    )
    payload["evaluation"]["controlled_transfer"]["conditions"].extend(
        [
            {
                "condition_id": "scale-2",
                "view": "scale",
                "instance_set": "scale-two",
                "reference_condition_id": "reference",
                "scale_path_id": "customers",
                "scale_vector": {"customers": 400.0},
            },
            {
                "condition_id": "scale-3",
                "view": "scale",
                "instance_set": "scale-three",
                "reference_condition_id": "reference",
                "scale_path_id": "customers",
                "scale_vector": {"customers": 800.0},
            },
        ]
    )
    task = PitBenchTask.model_validate(payload)
    expected = _expected_grid(task)
    observations = _heuristic_observations(task)
    _add_heuristic_condition(
        task,
        expected,
        observations,
        condition_id="scale-2",
        instance_set="scale-two",
        agent_gap=0.13,
    )
    _add_heuristic_condition(
        task,
        expected,
        observations,
        condition_id="scale-3",
        instance_set="scale-three",
        agent_gap=0.11,
    )

    report = compute_controlled_transfer_report(
        observations,
        task=task,
        expected_run_grid=expected,
    )

    assert report is not None
    assert report.condition_reports["scale-1"].simultaneous_bound_count == 4
    assert report.condition_reports["scale-3"].supports_retention
    assert report.scale_paths["customers"].largest_continuously_supported_condition_id == "scale-1"


@pytest.mark.parametrize(
    ("lost_instances", "expected_classification"),
    [
        (("two",), PerformanceClassification.INCONCLUSIVE),
        (("one", "two"), PerformanceClassification.REGRESSED),
    ],
)
def test_controlled_transfer_exact_coverage_precedes_time(
    lost_instances: tuple[str, ...],
    expected_classification: PerformanceClassification,
) -> None:
    task = _exact_task()
    expected = _expected_grid(task)
    expected.runs = [
        run.model_copy(
                update={
                    "instance_set": "judge_id" if run.condition_id == "reference" else "agent_dev",
                    "instance_set_kind": "judge_id" if run.condition_id == "reference" else "agent_dev",
                    "budget_sec": 20.0,
                }
        )
        for run in expected.runs
    ]
    observations = []
    for run in expected.runs:
        agent_is_unsolved = (
            run.condition_id == "scale-1"
            and run.code_state is CodeState.AGENT
            and run.instance_id in lost_instances
        )
        observations.append(
            RunObservation(
                task_id=task.task_id,
                code_state=run.code_state,
                instance_set=run.instance_set,
                instance_set_kind=run.instance_set_kind,
                instance_id=run.instance_id,
                solver_seed=run.solver_seed,
                budget_sec=run.budget_sec,
                status=RunStatus.COMPLETED,
                valid=True,
                objective=1.0,
                reported_objective=1.0,
                optimal_or_bks=1.0,
                cpu_time_sec=0.1 if run.code_state is CodeState.AGENT else 4.0,
                solver_termination=(
                    SolverTermination.TIME_LIMIT
                    if agent_is_unsolved
                    else SolverTermination.OPTIMAL
                ),
                condition_id=run.condition_id,
                template_id=run.template_id,
            )
        )

    report = compute_controlled_transfer_report(
        observations,
        task=task,
        expected_run_grid=expected,
    )

    assert report is not None
    target = report.condition_reports["scale-1"]
    assert target.status is ControlledTransferConditionStatus.COMPLETE
    assert target.agent_solved_coverage == pytest.approx(
        1 - len(lost_instances) / 2
    )
    assert target.base_solved_coverage == pytest.approx(1.0)
    assert target.native_classification is expected_classification
    assert not target.supports_retention


def test_controlled_transfer_schema_metadata_reaches_expected_runs() -> None:
    task = _heuristic_task()

    grid = JudgePlan.fixture(task, instances_per_instance_set=1).expected_run_grid()

    assert {run.condition_id for run in grid.runs} >= {"reference", "scale-1"}
    assert all(run.template_id is None for run in grid.runs)


def test_controlled_transfer_report_command_includes_structured_result(
    tmp_path: Path,
) -> None:
    task = _heuristic_task()
    observations_path = tmp_path / "trials.parquet"
    ObservationStore.write(observations_path, _heuristic_observations(task))
    (tmp_path / "expected-run-grid.json").write_text(
        _expected_grid(task).model_dump_json(indent=2) + "\n"
    )
    task_path = tmp_path / "task.yaml"
    task_path.write_text(yaml.safe_dump(task.model_dump(mode="json"), sort_keys=False))

    result = CliRunner().invoke(
        app,
        ["report", str(observations_path), "--task-config", str(task_path), "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["controlled_transfer"]["condition_reports"]["scale-1"][
        "supports_retention"
    ]
    assert "Controlled Transfer" in format_controlled_transfer_report(
        compute_controlled_transfer_report(
            _heuristic_observations(task),
            task=task,
            expected_run_grid=_expected_grid(task),
        )
    )
