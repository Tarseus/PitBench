from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from typer.testing import CliRunner

from pitbench.cli.main import app
from pitbench.evaluator import runner
from pitbench.evaluator.evaluator import PitBenchEvaluator
from pitbench.evaluator.judge import LocalProcessJudge
from pitbench.evaluator.reliability import prepare_boundary_cases
from pitbench.evaluator.storage import ObservationStore
from pitbench.harness.evaluation import EvaluationRequest
from pitbench.instances.boundary import (
    BoundarySuite,
    LinearBoundarySuite,
    RoutingBoundarySuite,
    boundary_suite,
)
from pitbench.metrics.performance_report import compute_performance_report
from pitbench.metrics.reliability_report import compute_reliability_reports
from pitbench.metrics.resource_report import compute_resource_report
from pitbench.problem_families.verification import CVRPFamily, IntegerBoundaryFamily
from pitbench.schema.observation import CodeState, RunObservation, RunStatus
from pitbench.schema.task import PitBenchTask

# Tests consolidated from tests/unit/pitbench/test_reliability.py


ROOT = Path(__file__).resolve().parents[3]


def test_boundary_suites_register_by_family_and_reject_duplicates():
    assert boundary_suite("cvrp") is RoutingBoundarySuite
    assert boundary_suite("mip") is LinearBoundarySuite

    class FirstSuite(BoundarySuite):
        problem_family = "test-boundary-family"

        @staticmethod
        def cases():
            return []

        @staticmethod
        def write(case, directory):
            return directory

        @staticmethod
        def verifier():
            return None

    try:
        with pytest.raises(ValueError, match="duplicate boundary suite"):

            class DuplicateSuite(FirstSuite):
                problem_family = "test-boundary-family"
    finally:
        BoundarySuite._suites.pop("test-boundary-family", None)


@pytest.fixture
def task():
    value = PitBenchTask.from_yaml(ROOT / "configs/tasks/highs_v1_15_1.yaml")
    value.evaluation.solver_seeds = [0, 1]
    value.evaluation.budgets_sec = [5.0, 10.0]
    value.evaluation.primary_budget_sec = 10.0
    return value


def grid(task):
    return [
        RunObservation(
            task_id=task.task_id,
            code_state=state,
            instance_set="boundary_cases",
            instance_set_kind="agent_dev",
            instance_id=case.name,
            test_suite="operational_reliability",
            solver_seed=seed,
            budget_sec=budget,
            threads=task.evaluation.threads,
            status=RunStatus.COMPLETED,
            valid=True,
            objective=case.data["known_optimum"],
            solver_status="Time limit reached",
        )
        for case, seed, budget, state in itertools.product(
            LinearBoundarySuite.cases(),
            task.evaluation.solver_seeds,
            task.evaluation.budgets_sec,
            CodeState,
        )
    ]


def solution_text(values, objective):
    return "\n".join(
        [
            "Model status",
            "Optimal",
            "",
            "# Primal solution values",
            "Feasible",
            f"Objective {objective}",
            f"# Columns {len(values)}",
            *(f"x{index} {value}" for index, value in enumerate(values)),
            "# Rows 0",
            "# Dual solution values",
            "None",
        ]
    )


def test_boundary_references_are_feasible_and_integer_optima_are_independently_known(
    tmp_path,
):
    for case in RoutingBoundarySuite.cases():
        instance = RoutingBoundarySuite.write(case, tmp_path)
        solution = tmp_path / "solution.json"
        solution.write_text(json.dumps(case.reference_solution))
        assert CVRPFamily().verify(instance, solution).feasible
    for case in LinearBoundarySuite.cases():
        model = case.data
        feasible_costs = []
        # Exhaustive enumeration is independent of the solver and generated LP text.
        domains = [
            range(lower, upper + 1)
            for lower, upper in zip(model["lower"], model["upper"], strict=True)
        ]
        for values in itertools.product(*domains):
            for coefficients, sense, rhs in model["constraints"]:
                activity = sum(a * b for a, b in zip(coefficients, values, strict=True))
                if not {
                    "=": activity == rhs,
                    ">=": activity >= rhs,
                    "<=": activity <= rhs,
                }[sense]:
                    break
            else:
                feasible_costs.append(
                    sum(a * b for a, b in zip(model["costs"], values, strict=True))
                )
        assert min(feasible_costs) == model["known_optimum"]
        instance = LinearBoundarySuite.write(case, tmp_path)
        solution = tmp_path / "solution.json"
        solution.write_text(
            json.dumps(
                {
                    "raw_solution": solution_text(
                        case.reference_solution["values"], model["known_optimum"]
                    ),
                    "objective": model["known_optimum"],
                    "solver_status": "Optimal",
                }
            )
        )
        assert IntegerBoundaryFamily().verify(instance, solution).feasible


def test_integer_verifier_checks_feasibility_objective_and_optimality_claims(tmp_path):
    case = LinearBoundarySuite.cases()[2]
    instance = LinearBoundarySuite.write(case, tmp_path)
    path = tmp_path / "solution.json"
    for values, objective, status, expected in [
        ([0, 1], 2, "Time limit reached", True),  # Valid incumbent need not be optimal.
        ([0, 1], 2, "Optimal", False),
        ([0, 0], 0, "Time limit reached", False),
        ([1, 0], 99, "Optimal", False),
        ([0.5, 0.5], 1.5, "Optimal", False),
    ]:
        path.write_text(
            json.dumps(
                {
                    "raw_solution": solution_text(values, objective),
                    "objective": objective,
                    "solver_status": status,
                }
            )
        )
        assert IntegerBoundaryFamily().verify(instance, path).feasible is expected


def test_reports_pair_failures_without_mixing_budgets_or_hiding_invalid_solutions(task):
    observations = grid(task)
    changed = next(
        item
        for item in observations
        if item.code_state == CodeState.AGENT and item.budget_sec == 5
    )
    changed.status, changed.valid, changed.error = RunStatus.CRASHED, False, "crash"
    invalid = next(
        item
        for item in observations
        if item.code_state == CodeState.AGENT and item.budget_sec == 10
    )
    invalid.status, invalid.valid = RunStatus.INVALID, False
    report, details = compute_reliability_reports(observations, task=task)
    assert report.complete and not report.qualification_passed
    assert report.by_budget["5"].by_state["base"].pass_rate == 1
    assert report.by_budget["5"].by_state["agent"].pass_rate == 7 / 8
    assert report.by_budget["5"].paired_pass_to_fail == 1
    assert report.by_budget["10"].by_state["agent"].invalid_solutions == 1
    assert sum(len(item.failed_runs) for item in details.cases) == 2
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
    reversed_report, _ = compute_reliability_reports(swapped, task=task)
    assert reversed_report.by_budget["5"].paired_fail_to_pass == 1


def test_missing_runs_do_not_turn_into_a_perfect_pass_rate(task):
    observations = grid(task)
    removed = observations.pop()
    report, details = compute_reliability_reports(observations, task=task)
    assert not report.complete
    counts = report.by_budget[f"{removed.budget_sec:g}"].by_state[
        removed.code_state.value
    ]
    assert counts.missing_runs == 1 and counts.pass_rate is None
    assert sum(len(item.missing_runs) for item in details.cases) == 1
    with pytest.raises(ValueError, match="duplicate"):
        compute_reliability_reports([*observations, observations[0]], task=task)
    with pytest.raises(ValueError, match="outside the declared grid"):
        compute_reliability_reports(
            [removed.model_copy(update={"solver_seed": 999})], task=task
        )


def test_boundary_cases_cannot_change_performance_or_resource_reports(task):
    regular = [
        RunObservation(
            task_id=task.task_id,
            instance_set="judge_id",
            instance_id="regular",
            code_state=state,
            solver_seed=0,
            budget_sec=5,
            status=RunStatus.COMPLETED,
            valid=True,
            normalized_gap=0.1,
            peak_rss_bytes=100,
            resource_scope="worker_process_start_through_solver_return",
        )
        for state in CodeState
    ]
    boundary = [
        item.model_copy(update={"normalized_gap": 1000, "peak_rss_bytes": 1})
        for item in grid(task)
    ]
    assert compute_performance_report(
        regular, primary_budget_sec=5
    ) == compute_performance_report([*regular, *boundary], primary_budget_sec=5)
    assert compute_resource_report(
        regular, primary_budget_sec=5
    ) == compute_resource_report([*regular, *boundary], primary_budget_sec=5)


def test_preparation_retains_task_budgets_thirty_seeds_and_explicit_expected_results(
    tmp_path,
):
    for task_id in ("pyvrp_v0_14_0", "highs_v1_15_1"):
        task = PitBenchTask.from_yaml(ROOT / "configs/tasks" / f"{task_id}.yaml")
        cases = prepare_boundary_cases(task, tmp_path / task_id)
        assert len(cases) == 4
        assert all(
            len(case.solver_seeds) == 30
            and case.budgets_sec == tuple(task.evaluation.budgets_sec)
            for case in cases
        )
        manifest = json.loads(
            (tmp_path / task_id / "reliability/manifest.json").read_text()
        )
        assert all(case["expected"] for case in manifest["cases"])
        assert all(case.anchor is None and case.verifier is not None for case in cases)


def test_runner_can_execute_only_public_boundary_cases_without_private_assets(
    task, tmp_path, monkeypatch
):
    task_file = tmp_path / "task.yaml"
    task_file.write_text(yaml.safe_dump(task.model_dump(mode="json")))
    output = tmp_path / "out"
    patch_file = tmp_path / "candidate.patch"
    patch_file.touch()
    monkeypatch.setattr(LocalProcessJudge, "_workspace", lambda *args: tmp_path)
    records = {
        (item.instance_id, item.solver_seed, item.budget_sec, item.code_state): item
        for item in grid(task)
    }
    monkeypatch.setattr(
        LocalProcessJudge,
        "_run_case",
        lambda self, workspace, case, state, seed, budget, **kwargs: records[
            case.instance_id, seed, budget, state
        ],
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "runner",
            "--task-config",
            str(task_file),
            "--base-repository",
            str(tmp_path),
            "--public-root",
            str(ROOT),
            "--private-root",
            str(tmp_path / "absent"),
            "--candidate-patch",
            str(patch_file),
            "--output-dir",
            str(output),
            "--observations",
            str(output / "observations.jsonl"),
            "--reliability-only",
        ],
    )
    runner.main()
    observations = ObservationStore.read_jsonl(output / "observations.jsonl")
    assert len(observations) == 32
    assert all(item.test_suite == "operational_reliability" for item in observations)


def test_evaluator_and_report_cli_expose_reliability_separately(task, tmp_path):
    task_file = tmp_path / "task.yaml"
    task_file.write_text(yaml.safe_dump(task.model_dump(mode="json")))
    candidate = tmp_path / "candidate.patch"
    candidate.touch()
    request = EvaluationRequest(
        task_id=task.task_id,
        task_path=ROOT,
        candidate_patch_path=candidate,
        candidate_patch_sha256=hashlib.sha256(b"").hexdigest(),
        output_dir=tmp_path,
        agent_name="test",
        evaluator_config={
            "task_config_path": str(task_file),
            "base_repository": str(tmp_path),
            "private_root": str(tmp_path),
            "judge_image": "sha256:" + "a" * 64,
            "reliability_only": True,
        },
    )
    observations = grid(task)
    observations[0].status, observations[0].valid = RunStatus.CRASHED, False
    with (
        patch("pitbench.evaluator.evaluator.PitBenchAdapter.validate_repository"),
        patch("pitbench.evaluator.evaluator.DockerJudge") as docker,
    ):
        docker.return_value.run.return_value = observations
        result = PitBenchEvaluator().evaluate(request)
        assert docker.call_args.kwargs["reliability_only"]
    assert result.summary.performance is None and result.summary.resource_usage is None
    assert result.summary.operational_reliability.complete
    assert (tmp_path / result.artifacts.reliability_details.path).is_file()
    response = CliRunner().invoke(
        app, ["report", str(tmp_path), "--task-config", str(task_file), "--json"]
    )
    assert response.exit_code == 0, response.output
    payload = json.loads(response.output)
    assert payload["performance"] is None and payload["resource_usage"] is None
    assert (
        payload["operational_reliability"]["by_budget"]["5"]["by_state"]["base"][
            "failed_runs"
        ]
        == 1
    )
