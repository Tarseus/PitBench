import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from pitbench.evaluator.scenarios import task_digest
from pitbench.metrics.objective_native import native_objective_measurement
from pitbench.metrics.objective_qualification import (
    compute_objective_qualification,
    objective_protocol_gaps,
)
from pitbench.metrics.scenario_report import compute_scenario_report
from pitbench.schema.objectives import (
    qualification_protocols,
    qualification_protocols_sha256,
)
from pitbench.schema.observation import (
    CodeState,
    ExpectedRun,
    ExpectedRunGrid,
    RunObservation,
    RunStatus,
    SolverTermination,
)
from pitbench.schema.scenarios import SCENARIO_METRICS
from pitbench.schema.task import PitBenchTask
from tests.unit.pitbench.test_scenario_metrics import campaign

ROOT = Path(__file__).resolve().parents[3]


def task_for(metric, protocols=(), **plan_changes):
    payload = yaml.safe_load((ROOT / "configs/tasks/highs_v1_15_1.yaml").read_text())
    payload["task_id"] = "scenario-test"
    hidden = dict(next(s for s in payload["instance_sets"] if s["name"] == "judge_id"))
    hidden.update(name="judge_shift", kind="judge_shift")
    payload["instance_sets"].append(hidden)
    payload["evaluation"].update(
        budgets_sec=[10],
        primary_budget_sec=10,
        solver_seeds=[0, 1],
        seed_robustness=None,
        representation_robustness=None,
        operational_reliability=False,
    )
    for panel in payload["instance_sets"]:
        panel["size"] = 2
    objective = next(
        k
        for k, v in qualification_protocols("1.1")["objectives"].items()
        if metric in v["metric_ids"]
    )
    plan = dict(
        protocol_version="1.1",
        protocols_sha256=qualification_protocols_sha256("1.1"),
        qualification_instance_sets=["judge_id"],
        reporting_instance_sets=["judge_shift"],
        scenario_protocols_sha256={p.id: p.digest() for p in protocols},
        guardrails=[
            dict(
                metric_id=m, comparison=c, max_degradation=0, instance_sets=["judge_id"]
            )
            for m, c in [
                ("exact_solved_coverage", "absolute"),
                ("exact_penalized_runtime_sec", "absolute"),
                ("peak_rss_ratio", "relative"),
            ]
        ],
    )
    plan.update(plan_changes)
    if metric in {"boundary_pass_counts", "operational_failure_counts"}:
        plan["reliability_scope"] = "declared_boundary_suite"
        payload["evaluation"]["operational_reliability"] = True
    payload["metric_scenarios"] = [p.model_dump(mode="json") for p in protocols]
    payload["optimization_profile"] = dict(
        objectives=[objective], primary_metric_id=metric, scoring=plan
    )
    return PitBenchTask.model_validate(payload)


def normal_evidence(task, seeds=(0, 1)):
    rows = [
        RunObservation(
            task_id=task.task_id,
            code_state=state,
            instance_set="judge_id",
            instance_set_kind="judge_id",
            instance_id=iid,
            solver_seed=seed,
            budget_sec=10,
            status=RunStatus.COMPLETED,
            valid=True,
            objective=1,
            reported_objective=1,
            optimal_or_bks=1,
            normalized_gap=0,
            cpu_time_sec=1,
            wall_time_sec=1,
            peak_rss_bytes=1000,
            resource_scope="worker_process_start_through_solver_return",
            solver_termination=SolverTermination.OPTIMAL,
        )
        for state in CodeState
        for iid in ("a", "b")
        for seed in seeds
    ]
    grid = ExpectedRunGrid(
        task_id=task.task_id,
        runs=[
            ExpectedRun(**{k: getattr(row, k) for k in ExpectedRun.model_fields})
            for row in rows
        ],
    )
    return rows, grid


def fixture(tmp_path, kind="setup", metric="setup_time_sec"):
    data = campaign(tmp_path, kind)
    task = task_for(metric, data.protocols)
    data.task_sha256 = task_digest(task)
    return task, data, *normal_evidence(task)


def test_new_protocol_has_all_adapters_while_frozen_protocol_is_unchanged():
    old, new = qualification_protocols(), qualification_protocols("1.1")
    assert old["protocol_version"] == "1.0"
    assert old["metrics"]["setup_time_sec"]["status"] == "requires_scenario_protocol"
    assert sum(v["status"] == "implemented" for v in new["metrics"].values()) == 25
    assert new["metrics"]["search_work"]["status"] == "diagnostic_only"
    assert {m for group in SCENARIO_METRICS.values() for m in group} == {
        m for m, v in new["metrics"].items() if v["adapter"] == "declared_scenario"
    }


def test_scenario_primary_requires_real_reversed_fresh_confirmation(tmp_path):
    task, data, rows, grid = fixture(tmp_path)
    assert not objective_protocol_gaps(task)
    initial = compute_scenario_report(data)
    args = dict(
        task=task,
        observations=rows,
        expected_run_grid=grid,
        scenario_report=initial,
        candidate_sha256=data.candidate_sha256,
        validity_accepted=True,
    )
    result = compute_objective_qualification(**args)
    assert result.classification == "confirmation_required"
    assert result.initial.primary[0].upper < 0
    repeat = data.model_copy(deep=True)
    repeat.collection_id = "independent-repeat"
    repeat.order = "agent_first"
    repeat.started_at, repeat.finished_at = 3, 4
    final = compute_objective_qualification(
        **args,
        repeat_observations=rows,
        repeat_expected_run_grid=grid,
        repeat_identity_verified=True,
        repeat_scenario_report=compute_scenario_report(repeat),
    )
    assert final.confirmed and final.classification == "improved"


@pytest.mark.parametrize("fault", ["same_id", "same_order", "wrong_design", "old_time"])
def test_reused_or_changed_scenario_evidence_cannot_confirm(tmp_path, fault):
    task, data, rows, grid = fixture(tmp_path)
    initial = compute_scenario_report(data)
    repeat = initial.model_copy(
        deep=True,
        update=dict(
            collection_id="new", order="agent_first", started_at=3, finished_at=4
        ),
    )
    if fault == "same_id":
        repeat.collection_id = initial.collection_id
    elif fault == "same_order":
        repeat.order = initial.order
    elif fault == "wrong_design":
        repeat.design_sha256 = "f" * 64
    else:
        repeat.started_at = 0
    result = compute_objective_qualification(
        task=task,
        observations=rows,
        expected_run_grid=grid,
        repeat_observations=rows,
        repeat_expected_run_grid=grid,
        repeat_identity_verified=True,
        validity_accepted=True,
        scenario_report=initial,
        repeat_scenario_report=repeat,
        candidate_sha256=data.candidate_sha256,
    )
    assert result.classification == "incomplete" and not result.confirmed


def test_missing_or_wrong_candidate_scenario_cannot_be_scored(tmp_path):
    task, data, rows, grid = fixture(tmp_path)
    result = compute_objective_qualification(
        task=task,
        observations=rows,
        expected_run_grid=grid,
        scenario_report=compute_scenario_report(data),
        candidate_sha256="f" * 64,
    )
    assert result.classification == "incomplete"


def test_faster_setup_with_worse_final_quality_fails_scenario_guardrail(tmp_path):
    task, data, rows, grid = fixture(tmp_path)
    next(r for r in data.runs if r.key.code_state == "agent").steps[
        0
    ].final_solution_values = [2, 0]
    report = compute_scenario_report(data)
    assert report.complete and not report.invalid_runs
    result = compute_objective_qualification(
        task=task,
        observations=rows,
        expected_run_grid=grid,
        scenario_report=report,
        candidate_sha256=data.candidate_sha256,
    )
    assert result.classification == "guardrail_failed" and not result.confirmed


def test_invalid_intermediate_solution_disqualifies_scenario_primary(tmp_path):
    task, data, rows, grid = fixture(
        tmp_path, "trajectory", "time_to_first_verified_solution_sec"
    )
    next(r for r in data.runs if r.key.code_state == "agent").steps[0].events[
        0
    ].solution_values = [0, 0]
    result = compute_objective_qualification(
        task=task,
        observations=rows,
        expected_run_grid=grid,
        scenario_report=compute_scenario_report(data),
        candidate_sha256=data.candidate_sha256,
    )
    assert result.classification == "invalid"


@pytest.mark.parametrize(
    "metric", ["boundary_pass_counts", "operational_failure_counts"]
)
def test_native_reliability_uses_declared_complete_boundary_pairs(metric):
    task = task_for(metric)
    rows, grid = normal_evidence(task)
    for state in CodeState:
        for iid in ("boundary-a", "boundary-b"):
            row = rows[0].model_copy(
                update=dict(
                    code_state=state,
                    instance_id=iid,
                    solver_seed=0,
                    test_suite="operational_reliability",
                    instance_set="boundary_cases",
                    status=RunStatus.CRASHED
                    if state == CodeState.BASE
                    else RunStatus.COMPLETED,
                    valid=state == CodeState.AGENT,
                )
            )
            rows.append(row)
            grid.runs.append(
                ExpectedRun(**{k: getattr(row, k) for k in ExpectedRun.model_fields})
            )
    value = native_objective_measurement(task, rows, grid, metric, "judge_id")
    assert value["complete"] and value["upper"] < 0
    assert not native_objective_measurement(
        task, rows[:-1], grid, metric, "judge_id"
    ).get("complete", False)


def test_native_representation_preserves_type7_iqr_and_expected_grid():
    task = task_for("representation_dispersion")
    payload = task.model_dump(mode="json")
    source = yaml.safe_load((ROOT / "configs/tasks/choco_v6_0_1.yaml").read_text())
    for key in ("release", "repository", "problem_family", "oracle"):
        payload[key] = source[key]
    payload["evaluation"]["representation_robustness"] = dict(
        kind="item_relabeling",
        instance_set="judge_id",
        solver_seed=0,
        relabeling_generation_seed=1,
        relabelings_per_instance=30,
    )
    task = PitBenchTask.model_validate(payload)
    rows, grid = normal_evidence(task)
    for state in CodeState:
        for iid in ("a", "b"):
            for variant in range(30):
                row = rows[0].model_copy(
                    update=dict(
                        code_state=state,
                        instance_id=f"{iid}-{variant}",
                        solver_seed=0,
                        equivalence_parent_id=iid,
                        equivalence_transform="row_column_permutation",
                        cpu_time_sec=1
                        + variant * (0.1 if state == CodeState.BASE else 0.05),
                        wall_time_sec=1
                        + variant * (0.1 if state == CodeState.BASE else 0.05),
                    )
                )
                rows.append(row)
                grid.runs.append(
                    ExpectedRun(
                        **{k: getattr(row, k) for k in ExpectedRun.model_fields}
                    )
                )
    value = native_objective_measurement(
        task, rows, grid, "representation_dispersion", "judge_id"
    )
    assert value["complete"] and value["degradation"] < 0
    assert not native_objective_measurement(
        task, rows[:-1], grid, "representation_dispersion", "judge_id"
    ).get("complete", False)


def test_native_seed_adapter_retains_existing_99_percent_interval():
    task = task_for("seed_dispersion", seed_dispersion_measure="exact_capped_time_iqr")
    payload = task.model_dump(mode="json")
    payload["evaluation"].update(
        solver_seeds=None,
        seed_robustness=dict(
            development_seeds=[0, 1, 2, 3],
            evaluation_seeds_file="private://fixture/seeds.yaml",
            evaluation_seeds_file_sha256="e" * 64,
            seed_selection=dict(seed_min=0, seed_max=100, seed_count=4),
        ),
    )
    task = PitBenchTask.model_validate(payload)
    rows, grid = normal_evidence(task, seeds=(4, 5, 6, 7))
    for row in rows:
        row.cpu_time_sec = (
            (row.solver_seed - 3) if row.code_state == CodeState.BASE else 2.5
        )
    value = native_objective_measurement(
        task, rows, grid, "seed_dispersion", "judge_id"
    )
    assert value["complete"] and value["confidence_level"] == 0.99
    assert value["degradation"] < 0


def test_native_controlled_retention_rejects_a_missing_condition():
    from tests.unit.pitbench.test_controlled_transfer import (
        _expected_grid,
        _heuristic_observations,
        _heuristic_task,
    )

    task = _heuristic_task()
    rows = _heuristic_observations(task)
    grid = _expected_grid(task)
    measured = native_objective_measurement(
        task, rows, grid, "controlled_retention", "judge_shift"
    )
    assert measured["complete"] and measured["classification"] == "improved"
    assert not native_objective_measurement(
        task, rows[:-1], grid, "controlled_retention", "judge_shift"
    ).get("complete", False)


def test_cli_recomputes_scenarios_and_qualification_from_raw_campaign(tmp_path):
    from typer.testing import CliRunner

    from pitbench.cli.main import app
    from pitbench.evaluator.storage import ObservationStore

    task, data, rows, grid = fixture(tmp_path)
    task_file = tmp_path / "task.yaml"
    task_file.write_text(yaml.safe_dump(task.model_dump(mode="json")))
    patch = tmp_path / "candidate.patch"
    patch.write_text("")
    data.candidate_sha256 = hashlib.sha256(patch.read_bytes()).hexdigest()
    path = tmp_path / "metric-scenarios"
    path.mkdir()
    (path / "campaign.json").write_text(data.model_dump_json())
    ObservationStore.write(tmp_path / "trials.parquet", rows)
    (tmp_path / "expected-run-grid.json").write_text(grid.model_dump_json())
    result = CliRunner().invoke(
        app, ["report", str(tmp_path), "--task-config", str(task_file), "--json"]
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["metric_scenarios"]["complete"]
    assert (
        report["objective_measurements"]["qualification"]["classification"]
        == "confirmation_required"
    )
    assert (
        report["objective_measurements"]["metrics"]["setup_time_sec"]["status"]
        == "reported"
    )


def test_evaluator_collects_declared_scenarios_and_invalidates_bad_solution(
    tmp_path, monkeypatch
):
    from pitbench.evaluator.evaluator import PitBenchEvaluator
    from pitbench.harness.evaluation import EvaluationRequest

    task, data, _, _ = fixture(tmp_path)
    task_file = tmp_path / "task.yaml"
    task_file.write_text(yaml.safe_dump(task.model_dump(mode="json")))
    patch = tmp_path / "candidate.patch"
    patch.write_text("")
    data.candidate_sha256 = hashlib.sha256(patch.read_bytes()).hexdigest()
    next(r for r in data.runs if r.key.code_state == "agent").steps[
        0
    ].final_solution_values = [0, 0]
    calls = []

    def collect(selected_task, candidate, output, config, **kwargs):
        calls.append((selected_task.task_id, candidate))
        destination = output / "metric-scenarios"
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / "campaign.json"
        path.write_text(data.model_dump_json())
        return compute_scenario_report(data), path

    monkeypatch.setattr("pitbench.evaluator.scenarios.collect_for_evaluation", collect)
    request = EvaluationRequest(
        task_id=task.task_id,
        task_path=ROOT,
        candidate_patch_path=patch,
        output_dir=tmp_path / "evaluation",
        agent_name="fixture",
        evaluator_config={
            "task_config_path": str(task_file),
            "fixture_mode": True,
            "fixture_instances_per_instance_set": 2,
            "metric_scenario_config": "configured.yaml",
        },
    )
    result = PitBenchEvaluator().evaluate(request)
    assert calls and result.summary.metric_scenarios.invalid_runs
    assert result.artifacts.metric_scenario_campaign is not None
    assert not result.validity.accepted
    assert any("metric scenarios" in check.detail for check in result.validity.checks)


def test_preflight_requires_scenario_configuration_for_declared_task(tmp_path):
    from pitbench.cli.doctor import CheckStatus, _metric_scenario_checks
    from pitbench.cli.evaluate_config import EvaluationConfig

    task, _, _, _ = fixture(tmp_path)
    context = SimpleNamespace(
        task=SimpleNamespace(task=task), config=EvaluationConfig(), repository_root=ROOT
    )
    (check,) = _metric_scenario_checks(context)
    assert check.status == CheckStatus.FAIL
