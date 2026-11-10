from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pitbench.evaluator.collection import (
    OrToolsCpSatCollectionBackend,
    _run_external_configuration,
    _verify_bin_packing_solution,
)
from pitbench.evaluator.configuration import (
    ConfigurationRunner,
    NativeConfigurationRunner,
    SearchSpaceExhausted,
    SmacSearch,
    _docker_collector_command,
    _write_native_comparisons,
    attach_configuration_report,
    materialize_fidelity_stages,
    prepare,
    prepare_candidate_configuration,
    run_search,
    validate_search,
)
from pitbench.metrics.configuration_report import (
    paired_panel,
    run_feedback,
    write_configuration_report,
)
from pitbench.problem_families.base import VerificationResult
from pitbench.schema.observation import CodeState, RunObservation, RunStatus


def test_parameter_only_cli_never_starts_an_agent_or_native_grid(tmp_path, monkeypatch):
    import hashlib

    from typer.testing import CliRunner

    from pitbench.cli.evaluate_config import EvaluationConfig
    from pitbench.cli.main import app

    patch = tmp_path / "candidate.patch"
    patch.write_text("retained source patch")
    source = tmp_path / "source"
    source.mkdir()
    config = EvaluationConfig()
    config.tasks["release_highs_v1_15_1"] = config.resources_for(
        "release_highs_v1_15_1"
    ).model_copy(update={"judge_image": "sha256:" + "a" * 64})
    path = tmp_path / "evaluation.local.yaml"
    path.write_text(config.model_dump_json())
    monkeypatch.setattr(
        "adapters.pitbench.adapter.PitBenchAdapter.ensure_source_cache",
        lambda *a: source,
    )
    monkeypatch.setattr(
        "pitbench.cli.main.PitBenchEvaluator",
        lambda: pytest.fail("native grid was relaunched"),
    )
    calls = []

    def parameters(request, settings):
        assert settings.enabled
        calls.append(request)
        output = request.output_dir / "configuration-robustness"
        output.mkdir(parents=True)
        (output / "experiment.json").write_text('{"searches":[{"id":"paired"}]}')
        (output / "configuration_report.json").write_text(
            '{"searches":[{"id":"paired","status":"completed"}]}'
        )
        return output

    monkeypatch.setattr(
        "pitbench.evaluator.configuration.run_candidate_configuration", parameters
    )
    root = Path(__file__).resolve().parents[3]
    result = CliRunner().invoke(
        app,
        [
            "judge",
            "release_highs_v1_15_1",
            str(patch),
            "--expected-patch-sha256",
            hashlib.sha256(patch.read_bytes()).hexdigest(),
            "--config",
            str(path),
            "--root",
            str(root),
            "--output-dir",
            str(tmp_path / "supplement"),
            "--configuration-only",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 1 and calls[0].candidate_patch_path == patch


def test_collector_container_reuse_keeps_fresh_workers_and_data_disk_tmp(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        "pitbench.evaluator.configuration.subprocess.run",
        lambda command, **kw: calls.append(command),
    )
    search = {"collector_image": "local:test", "collector_environment": {}}
    with ConfigurationRunner(tmp_path, [min(os.sched_getaffinity(0))]) as runner:
        first = runner._worker_command(
            search, ["worker", "--job", "/a/job.json"], 0, tmp_path / "a"
        )
        second = runner._worker_command(
            search, ["worker", "--job", "/b/job.json"], 0, tmp_path / "b"
        )
        assert first[:2] == second[:2] == ["docker", "exec"]
        assert first[first.index("--env") + 1] != second[second.index("--env") + 1]
        assert first[-3:] == ["worker", "--job", "/a/job.json"]
    assert len([c for c in calls if c[:2] == ["docker", "create"]]) == 1
    create = calls[0]
    assert f"{tmp_path.resolve()}:{tmp_path.resolve()}:rw" in create
    assert "--tmpfs" not in create
    assert calls[-1][:3] == ["docker", "rm", "--force"]


def test_native_parameter_cells_reuse_nested_panels_and_isolate_patches(
    tmp_path, monkeypatch
):
    invocations = []
    fault = {"once": False}

    class Judge:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self):
            p = json.loads(self.kwargs["configuration_panel"].read_text())
            saved = [RunObservation.model_validate(r) for r in p["reused_observations"]]
            done = {(r.instance_id, r.solver_seed) for r in saved}
            for instance in p["instances"]:
                for seed in p["seeds"]:
                    panel, identity = instance["id"].split("/")
                    if (identity, seed) in done:
                        continue
                    if fault["once"] and saved:
                        from pitbench.evaluator.storage import ObservationStore

                        fault["once"] = False
                        ObservationStore.write_jsonl(
                            self.kwargs["output_dir"] / "judge-observations.jsonl",
                            saved,
                        )
                        raise RuntimeError("interrupted panel")
                    invocations.append(
                        (instance["id"], seed, self.kwargs["code_states"][0])
                    )
                    saved.append(
                        RunObservation(
                            task_id="task",
                            code_state=self.kwargs["code_states"][0],
                            instance_set=panel,
                            instance_id=identity,
                            solver_seed=seed,
                            budget_sec=p["budget_sec"],
                            status=RunStatus.COMPLETED,
                            valid=True,
                            objective=1,
                            cpu_time_sec=0.1,
                            solver_termination="optimal",
                        )
                    )
            return saved

    monkeypatch.setattr("pitbench.evaluator.docker_judge.DockerJudge", Judge)
    search = {
        "id": "task/base",
        "task_configuration": {},
        "solver": {"patch": "base"},
        "code_state": "base",
        "fixed_options": {},
        "threads": 1,
        "budget_sec": 1,
        "instances": [
            {"id": "judge/a", "sha256": "a"},
            {"id": "judge/b", "sha256": "b"},
        ],
        "native_context": {
            "judge_image": "image",
            "build_cache_path": str(tmp_path / "shared/build"),
            "task_config_path": "task.yaml",
            "base_repository": "source",
            "private_root": "private",
            "candidate_patch": "candidate.patch",
            "judge_parallel_runs": 1,
        },
    }
    runner = NativeConfigurationRunner(
        tmp_path / "first", [min(os.sched_getaffinity(0))]
    )
    runner.panel(
        search, "candidate", {"amount": 1}, search["instances"][:1], [1], "screen"
    )
    full = runner.panel(
        search, "candidate", {"amount": 1}, search["instances"], [1, 2], "full"
    )
    assert len(full) == len(invocations) == 4
    other = NativeConfigurationRunner(tmp_path / "other", runner.cpus)
    other.panel(search, "renamed", {"amount": 1}, search["instances"], [1, 2], "full")
    assert len(invocations) == 4  # Same Base source is shared across candidates.
    changed = {**search, "solver": {"patch": "new"}, "code_state": "agent"}
    other.panel(changed, "renamed", {"amount": 1}, search["instances"], [1, 2], "full")
    assert len(invocations) == 8
    assert all(row[2] == CodeState.AGENT for row in invocations[4:])
    interrupted = {**changed, "solver": {"patch": "interrupted-source"}}
    before = len(invocations)
    fault["once"] = True
    with pytest.raises(RuntimeError, match="interrupted panel"):
        other.panel(
            interrupted, "candidate", {"amount": 1}, search["instances"], [1, 2], "full"
        )
    other.panel(
        interrupted, "candidate", {"amount": 1}, search["instances"], [1, 2], "full"
    )
    assert len(invocations) - before == 4
    # Deterministic solvers reuse seed zero on holdout too. A holdout must
    # still be a fresh measurement, then its paired comparison may reuse it.
    before = len(invocations)
    other.panel(
        search,
        "retest-default",
        {"amount": 1},
        search["instances"],
        [1, 2],
        "retest_full",
    )
    assert len(invocations) - before == 4
    other.panel(
        search,
        "paired",
        {"amount": 1},
        search["instances"],
        [1, 2],
        "independent_paired",
    )
    assert len(invocations) - before == 4
    # Changed raw observations cannot be accepted as a cache hit.
    receipt_path = next(
        p
        for p in (tmp_path / "shared/configuration-measurements").glob(
            "*/identity.json"
        )
        if json.loads(p.read_text())["identity"]["solver"]["patch"] == "base"
        and json.loads(p.read_text())["identity"]["evidence_phase"] == "search"
    )
    bank = receipt_path.parent / json.loads(receipt_path.read_text())["snapshot"]
    bank.write_text(bank.read_text() + "\n")
    with pytest.raises(ValueError, match="measurements changed"):
        other.panel(search, "again", {"amount": 1}, search["instances"], [1, 2], "full")


def test_native_candidate_preparation_binds_source_and_freezes_holdout(
    tmp_path, monkeypatch
):
    import hashlib

    from pitbench.harness.evaluation import EvaluationRequest
    from pitbench.schema.task import InstanceSetKind

    root = Path(__file__).resolve().parents[3]
    monkeypatch.setattr(
        "adapters.pitbench.adapter.PitBenchAdapter.validate_repository", lambda *a: None
    )
    patch = tmp_path / "candidate.patch"
    patch.write_text("source patch")
    path = tmp_path / "input.lp"
    path.write_text("frozen input")
    case = SimpleNamespace(
        instance_set=SimpleNamespace(kind=InstanceSetKind.JUDGE_ID, name="judge"),
        instance_id="a",
        path=path,
        anchor=1,
    )
    monkeypatch.setattr(
        "pitbench.evaluator.judge.JudgePlan.from_instance_set_configs",
        lambda *a, **kw: SimpleNamespace(cases=[case]),
    )
    monkeypatch.setattr("pitbench.evaluator.judge._evaluation_seeds", lambda *a: (1, 2))
    config = tmp_path / "parameters.json"
    config.write_text(
        json.dumps(
            {
                "searcher_version": "2.4.0",
                "max_configurations": 1,
                "instance_count": 1,
                "solver_seed_count": 2,
                "watchdog_grace_sec": 60,
                "panels": [
                    {
                        "task_config": "configs/tasks/highs_v1_15_1.yaml",
                        "collector": "highs",
                        "feedback": "capped_optimal_time",
                        "seed_max": 100,
                        "parameters": {
                            "presolve": {
                                "type": "categorical",
                                "choices": ["choose", "off"],
                                "default": "choose",
                            }
                        },
                    }
                ],
            }
        )
    )
    request = EvaluationRequest(
        task_id="release_highs_v1_15_1",
        task_path=tmp_path,
        candidate_patch_path=patch,
        candidate_patch_sha256=hashlib.sha256(patch.read_bytes()).hexdigest(),
        output_dir=tmp_path,
        agent_name="test",
        evaluator_config={
            "task_config_path": str(root / "configs/tasks/release_highs_v1_15_1.yaml"),
            "private_root": str(tmp_path),
            "judge_image": "image",
            "base_repository": str(tmp_path),
            "_progress_callback": lambda text: None,
        },
    )
    output = tmp_path / "parameter-evaluation"
    args = dict(cpus=[min(os.sched_getaffinity(0))], search_seed=9)
    manifest = prepare_candidate_configuration(request, config, output, **args)
    assert len(manifest["searches"]) == 4  # Two native budgets and two sources.
    assert {s["code_state"] for s in manifest["searches"]} == {"base", "agent"}
    holdout = manifest["searches"][0]["retest_seeds"]
    assert not set(holdout) & {1, 2}
    assert all(s["retest_seeds"] == holdout for s in manifest["searches"])
    assert (
        prepare_candidate_configuration(request, config, output, **args)["searches"]
        == manifest["searches"]
    )
    native_config = json.loads(config.read_text())
    native_config["solver_seed_count"] = 30
    native_config["panels"][0].update(
        native_only=True, search_mode="multi_fidelity", seed_min=1
    )
    native_config["fidelity_stages"] = [
        {
            "name": "screen_5",
            "instance_count": 1,
            "solver_seed_count": 5,
            "configuration_count": 1,
        },
        {
            "name": "full_30",
            "instance_count": 1,
            "solver_seed_count": 30,
            "configuration_count": 1,
        },
    ]
    config.write_text(json.dumps(native_config))
    native = prepare_candidate_configuration(
        request, config, tmp_path / "native-seeds", **args
    )
    assert all(search["solver_seeds"] == [1, 2] for search in native["searches"])
    assert all(
        len(stage["solver_seeds"]) == 2
        for search in native["searches"]
        for stage in search["fidelity_stages"]
    )
    assert all(seed >= 1 for seed in native["searches"][0]["retest_seeds"])
    patch.write_text("different patch")
    with pytest.raises(ValueError, match="patch changed"):
        prepare_candidate_configuration(request, config, output, **args)


@pytest.mark.parametrize(
    "status,expected",
    [
        (RunStatus.CRASHED, None),
        (RunStatus.INFRASTRUCTURE_ERROR, None),
        (RunStatus.OUT_OF_MEMORY, None),
        (RunStatus.TIMED_OUT, None),
        (RunStatus.NO_SOLUTION, 1),
    ],
)
def test_native_failures_are_not_scored_as_solver_budget_censoring(status, expected):
    observation = RunObservation(
        task_id="task",
        code_state=CodeState.AGENT,
        instance_set="judge",
        instance_id="a",
        solver_seed=0,
        budget_sec=10,
        status=status,
        valid=False,
        solver_termination="time_limit",
    )
    value, error = run_feedback(
        NativeConfigurationRunner.feedback_record(observation),
        feedback="capped_optimal_time",
        budget=10,
        bks=1,
        objective_sense="minimize",
    )
    assert value == expected
    assert (error is None) == (expected is not None)


def test_native_comparison_cross_checks_both_selected_parameters_on_holdout(
    tmp_path, monkeypatch
):
    searches = [
        {
            "id": state,
            "code_state": state,
            "budget_sec": 10,
            "instances": [{"id": "a", "bks": 100}],
            "retest_seeds": [11, 12],
            "feedback": "normalized_gap",
            "objective_sense": "minimize",
            "parameters": {"amount": {"default": 0}},
        }
        for state in ("base", "agent")
    ]
    for index, search in enumerate(searches, 1):
        p = tmp_path / search["id"]
        p.mkdir()
        (p / "state.json").write_text(
            json.dumps(
                {"status": "completed", "selected": {"parameters": {"amount": index}}}
            )
        )
    calls = []

    def panel(self, search, name, parameters, instances, seeds, stage):
        calls.append((search["code_state"], parameters, seeds))
        return [
            record(seed=seed, objective=100 if search["code_state"] == "base" else 90)
            for seed in seeds
        ]

    monkeypatch.setattr(NativeConfigurationRunner, "panel", panel)
    _write_native_comparisons(
        {"searches": searches, "cpus": [min(os.sched_getaffinity(0))]}, tmp_path
    )
    report = json.loads((tmp_path / "candidate-parameter-comparison.json").read_text())
    assert len(report["comparisons"]) == 3
    assert {call[1]["amount"] for call in calls} == {0, 1, 2}
    assert all(call[2] == [11, 12] for call in calls)
    assert all(
        row["complete"] and row["mean_degradation"] == pytest.approx(-0.1)
        for row in report["comparisons"]
    )


def test_invalid_parameter_solution_is_attached_as_a_correctness_failure(tmp_path):
    from pitbench.cli.main import _configuration_dry_run_failures

    (tmp_path / "experiment.json").write_text('{"searches":[]}')
    directory = tmp_path / "invalid-agent-solutions"
    directory.mkdir()
    (directory / "measured-invalid.json").write_text('{"status":"invalid"}')
    report = write_configuration_report(tmp_path)
    assert report["invalid_agent_solution_count"] == 1
    payload = {
        "validity": {
            "checks": [{"code": "patch_apply", "passed": True, "detail": "captured"}],
            "accepted": True,
        }
    }
    attach_configuration_report(payload, tmp_path, output_root=tmp_path)
    assert payload["validity"]["accepted"] is False
    assert payload["validity"]["checks"][-1]["code"] == "solution"
    assert _configuration_dry_run_failures(tmp_path) == [
        "candidate has invalid solutions under parameter settings"
    ]


def test_fidelity_smoke_limits_do_not_reduce_the_default_protocol():
    search = {
        "search_mode": "multi_fidelity",
        "search_seed": 1,
        "max_configurations": 4,
        "instances": [{"id": "a"}],
        "solver_seeds": [1, 2],
        "fidelity_stage_config": [
            {
                "name": "screen",
                "instance_count": 3,
                "solver_seed_count": 5,
                "configuration_count": 4,
            },
            {
                "name": "full",
                "instance_count": 10,
                "solver_seed_count": 30,
                "configuration_count": 2,
            },
        ],
    }
    with pytest.raises(ValueError, match="nested fidelity"):
        materialize_fidelity_stages(search)
    stages = materialize_fidelity_stages(
        {**search, "instances_per_panel": 1, "seeds_per_panel": 2}
    )
    assert all(
        len(stage["instances"]) == 1 and len(stage["solver_seeds"]) == 2
        for stage in stages
    )
    assert [stage["configuration_count"] for stage in stages] == [4, 2]


# Tests consolidated from tests/unit/pitbench/test_configuration.py


def record(instance="a", seed=1, objective=100, **fields):
    return {
        "instance_id": instance,
        "solver_seed": seed,
        "budget_sec": 10,
        "execution_status": "returned",
        "model_status": "Optimal",
        "termination": "optimal",
        "solution_value_valid": True,
        "solver_runtime_sec": 2,
        "verification": {"feasible": True, "objective": objective},
        **fields,
    }


def test_docker_collector_command_uses_declared_repository_environment() -> None:
    command = _docker_collector_command(
        "pitbench-collector:test",
        ["probe"],
        environment={"PITBENCH_VROOM_SOLVER": "/workspace/repo/bin/vroom"},
    )

    assert "PITBENCH_VROOM_SOLVER=/workspace/repo/bin/vroom" in command
    assert command.count("--env") == 2
    assert "PYTHONPATH" in command[command.index("--env") + 1]


def test_cp_sat_collector_uses_independent_assignment_verifier(
    tmp_path, monkeypatch
) -> None:
    class IndependentVerifier:
        def verify(self, instance_path, solution_path):
            assert instance_path == tmp_path / "instance.pb"
            assert solution_path == tmp_path / "solution.json"
            return VerificationResult(
                feasible=False, detail="assignment violates the fixed model"
            )

    class Repository:
        def prepare_independent_verifier(self, output_dir):
            assert output_dir.name.startswith("pitbench-cp-sat-verifier-")
            return IndependentVerifier()

    monkeypatch.setattr(
        "pitbench.repositories.base.RepositoryPluginRegistry.load",
        lambda path: Repository(),
    )

    def run_external(driver, job, directory, result, parameters, verifier):
        del driver, job, directory, parameters
        solution_path = tmp_path / "solution.json"
        solution_path.write_text('{"values":[1,2]}')
        result["verification"] = verifier(tmp_path / "instance.pb", solution_path)

    monkeypatch.setattr(
        "pitbench.evaluator.collection._run_external_configuration", run_external
    )
    job = {
        "repository_plugin": "pitbench.repositories.plugins:OrToolsCpSatExactRepositoryPlugin",
        "parameters": {
            "cp_model_presolve": True,
            "cp_model_probing_level": 2,
            "linearization_level": 1,
            "symmetry_level": 2,
            "search_branching": "automatic_search",
        },
    }
    result = {}

    OrToolsCpSatCollectionBackend.run(job, tmp_path, result)

    assert result["verification"]["feasible"] is False
    assert result["verification"]["detail"] == "assignment violates the fixed model"


def test_configuration_bin_packing_uses_shared_independent_verifier(
    tmp_path: Path,
) -> None:
    instance = tmp_path / "instance.json"
    solution = tmp_path / "solution.json"
    instance.write_text(json.dumps({"weights": [2, 3, 1], "capacity": 4}))
    solution.write_text(json.dumps({"bins": [[0, 2], [1]]}))

    valid = _verify_bin_packing_solution(instance, solution)
    assert valid["feasible"] is True
    assert valid["objective"] == 2.0

    solution.write_text(json.dumps({"bins": [[0, 1], [2]]}))
    invalid = _verify_bin_packing_solution(instance, solution)
    assert invalid["feasible"] is False


def test_external_configuration_verifier_infrastructure_failure_stops_collection(
    tmp_path: Path,
) -> None:
    class Driver:
        records_trajectory = False

        @staticmethod
        def main(arguments):
            output = Path(arguments[arguments.index("--output") + 1])
            output.write_text(
                json.dumps(
                    {"valid": True, "has_solution": True, "solver_status": "OPTIMAL"}
                )
            )
            output.with_suffix(".solution.json").write_text('{"values":[0]}')

    job = {
        "instance": {"path": str(tmp_path / "instance.pb")},
        "solver_seed": 0,
        "budget_sec": 10,
        "threads": 8,
    }
    result = {}

    with pytest.raises(RuntimeError, match="independent verifier infrastructure error"):
        _run_external_configuration(
            Driver,
            job,
            tmp_path,
            result,
            {},
            lambda instance, solution: {
                "feasible": False,
                "infrastructure_error": True,
                "detail": "JNI unavailable",
            },
        )

    assert result["error_stage"] == "verification"
    assert result["verification"]["infrastructure_error"] is True


def test_pairing_uses_keys_and_retains_negative_reference_gaps():
    instances = [{"id": "a", "bks": 100}, {"id": "b", "bks": -100}]
    baseline = [record(name, seed) for name in ("a", "b") for seed in (1, 2)]
    candidates = [
        record("b", 2),
        record("a", 2, 90),
        record("a", 1, 110),
        record("b", 1),
    ]
    result = paired_panel(
        baseline,
        candidates,
        instances=instances,
        seeds=[1, 2],
        budget=10,
        feedback="normalized_gap",
        objective_sense="minimize",
    )
    assert result["complete"] and result["mean_degradation"] == pytest.approx(0)
    assert result["pairs"][0]["degradation"] == pytest.approx(0.1)
    assert result["pairs"][1]["degradation"] == pytest.approx(-0.1)


def test_missing_or_failed_pair_does_not_change_the_averaging_population():
    kwargs = dict(
        instances=[{"id": "a", "bks": 100}],
        seeds=[1, 2],
        budget=10,
        feedback="normalized_gap",
        objective_sense="minimize",
    )
    baseline = [record(seed=1), record(seed=2)]
    result = paired_panel(baseline, [record(seed=2, objective=120)], **kwargs)
    assert result["mean_degradation"] is None
    assert result["expected_pairs"] == 2 and result["available_pairs"] == 1
    assert result["pairs"][0]["candidate_unavailable"] == "missing"
    with pytest.raises(ValueError, match="duplicate run"):
        paired_panel(baseline, [record(), record()], **kwargs)
    with pytest.raises(ValueError, match="outside"):
        paired_panel(baseline, [record(budget_sec=5)], **kwargs)


@pytest.mark.parametrize(
    ("change", "value", "reason"),
    [
        (
            {
                "model_status": "Time limit",
                "termination": "time_limit",
                "solution_value_valid": False,
                "verification": None,
            },
            1,
            None,
        ),
        ({"solver_runtime_sec": 20}, 1, None),
        ({"solver_runtime_sec": 2}, 0.2, None),
        ({"execution_status": "watchdog_timeout"}, None, "watchdog_timeout"),
        ({"execution_status": "solver_error"}, None, "solver_error"),
        ({"verification": {"feasible": False}}, None, "invalid_solution"),
        (
            {
                "model_status": "Time limit",
                "termination": "time_limit",
                "verification": {"feasible": False},
            },
            None,
            "invalid_solution",
        ),
        ({"solver_runtime_sec": None}, None, "unavailable_solver_time"),
        (
            {"model_status": "Infeasible", "termination": "other"},
            None,
            "unexpected_termination",
        ),
    ],
)
def test_exact_time_feedback_distinguishes_censoring_and_failures(
    change, value, reason
):
    assert run_feedback(
        record(**change),
        feedback="capped_optimal_time",
        budget=10,
        bks=None,
        objective_sense=None,
    ) == (value, reason)


@pytest.fixture
def search():
    return {
        "id": "example/budget-10",
        "task_id": "example",
        "budget_sec": 10,
        "feedback": "normalized_gap",
        "objective_sense": "minimize",
        "instances": [{"id": "a", "bks": 100}],
        "solver_seeds": [1, 2],
        "retest_seeds": [3, 4],
        "threads": 1,
        "parameters": {"amount": {"type": "integer", "bounds": [0, 10], "default": 0}},
        "max_configurations": 2,
        "search_seed": 15,
        "searcher_version": "2.4.0",
    }


class SearchStub:
    metadata = {"version": "test"}

    def __init__(self, search, directory):
        self.proposals = iter([0, 2, 1])
        self.feedback = []

    def ask(self):
        value = next(self.proposals)
        return value, {"amount": value}

    def tell(self, info, degradation):
        self.feedback.append((info, degradation))


class ExhaustingSearchStub:
    metadata = {"version": "test"}

    def __init__(self, search, directory):
        self.proposals = iter([0, 1])
        self.feedback = []

    def ask(self):
        try:
            value = next(self.proposals)
        except StopIteration as error:
            raise SearchSpaceExhausted from error
        return value, {"amount": value}

    def tell(self, info, degradation):
        self.feedback.append((info, degradation))


class MultiFidelitySearchStub:
    metadata = {"version": "test"}

    def __init__(self, search, directory):
        self.proposals = iter([0, 4, 1, 3, 2])
        self.feedback = []

    def ask(self):
        value = next(self.proposals)
        return value, {"amount": value}

    def tell(self, info, degradation):
        self.feedback.append((info, degradation))


def test_materialized_fidelity_stages_are_nested_and_end_at_the_full_panel():
    search = {
        "search_mode": "multi_fidelity",
        "search_seed": 15,
        "instances": [{"id": name, "bks": 100} for name in ("a", "b", "c", "d")],
        "solver_seeds": [1, 2, 3, 4],
        "max_configurations": 4,
        "fidelity_stage_config": [
            {
                "name": "screen",
                "instance_count": 1,
                "solver_seed_count": 1,
                "configuration_count": 4,
            },
            {
                "name": "middle",
                "instance_count": 2,
                "solver_seed_count": 2,
                "configuration_count": 2,
            },
            {
                "name": "full",
                "instance_count": 4,
                "solver_seed_count": 4,
                "configuration_count": 1,
            },
        ],
    }

    stages = materialize_fidelity_stages(search)

    assert [stage["configuration_count"] for stage in stages] == [4, 2, 1]
    assert {item["id"] for item in stages[0]["instances"]} <= {
        item["id"] for item in stages[1]["instances"]
    }
    assert {item["id"] for item in stages[1]["instances"]} <= {
        item["id"] for item in stages[2]["instances"]
    }
    assert set(stages[0]["solver_seeds"]) <= set(stages[1]["solver_seeds"])
    assert set(stages[1]["solver_seeds"]) <= set(stages[2]["solver_seeds"])


def test_dry_run_materializes_one_instance_at_every_fidelity_level():
    search = {
        "search_mode": "multi_fidelity",
        "search_seed": 15,
        "instances_per_panel": 1,
        "instances": [{"id": "a", "bks": 100}],
        "solver_seeds": [1, 2, 3, 4],
        "max_configurations": 4,
        "fidelity_stage_config": [
            {
                "name": "screen",
                "instance_count": 3,
                "solver_seed_count": 1,
                "configuration_count": 4,
            },
            {
                "name": "full",
                "instance_count": 10,
                "solver_seed_count": 4,
                "configuration_count": 1,
            },
        ],
    }

    stages = materialize_fidelity_stages(search)

    assert [len(stage["instances"]) for stage in stages] == [1, 1]
    assert [len(stage["solver_seeds"]) for stage in stages] == [1, 4]


def test_multi_fidelity_search_promotes_only_declared_complete_candidates(
    search, tmp_path
):
    search.update(
        search_mode="multi_fidelity",
        max_configurations=4,
        instances=[{"id": name, "bks": 100} for name in ("a", "b", "c", "d")],
        solver_seeds=[1, 2, 3, 4],
        retest_seeds=[5, 6, 7, 8],
        fidelity_stages=[
            {
                "name": "screen",
                "instances": [{"id": "a", "bks": 100}],
                "solver_seeds": [1],
                "configuration_count": 4,
            },
            {
                "name": "middle",
                "instances": [{"id": name, "bks": 100} for name in ("a", "b")],
                "solver_seeds": [1, 2],
                "configuration_count": 2,
            },
            {
                "name": "full",
                "instances": [
                    {"id": name, "bks": 100} for name in ("a", "b", "c", "d")
                ],
                "solver_seeds": [1, 2, 3, 4],
                "configuration_count": 1,
            },
        ],
    )
    calls = []

    def evaluate(name, parameters, instances, seeds, fidelity_stage):
        calls.append((name, fidelity_stage, len(instances) * len(seeds)))
        return [
            record(instance=item["id"], seed=seed, objective=100 + parameters["amount"])
            for item in instances
            for seed in seeds
        ]

    state = run_search(
        search,
        tmp_path,
        evaluate,
        engine_factory=MultiFidelitySearchStub,
    )

    assert state["status"] == "completed"
    assert state["promotions"] == {
        "middle": ["candidate-0001", "candidate-0003"],
        "full": ["candidate-0001"],
    }
    assert state["selected"]["name"] == "candidate-0001"
    assert state["selection_reason"] == "largest_full_fidelity_degradation"
    assert calls == [
        ("default", "default_full", 16),
        ("candidate-0001", "screen", 1),
        ("candidate-0002", "screen", 1),
        ("candidate-0003", "screen", 1),
        ("candidate-0004", "screen", 1),
        ("candidate-0001", "middle", 4),
        ("candidate-0003", "middle", 4),
        ("candidate-0001", "full", 16),
        ("retest-default", "retest_full", 16),
        ("retest-selected", "retest_full", 16),
    ]


def test_fixed_search_count_full_panels_and_retest_without_reselection(
    search, tmp_path
):
    calls = []
    engine = SearchStub(search, tmp_path)

    def evaluate(name, parameters, instances, seeds, fidelity_stage):
        del instances, fidelity_stage
        calls.append((name, parameters, seeds))
        # The search winner improves on retest: retain the negative result.
        objective = 100 + parameters["amount"] * (
            -5 if name == "retest-selected" else 5
        )
        return [record(seed=seed, objective=objective) for seed in seeds]

    state = run_search(search, tmp_path, evaluate, engine_factory=lambda *args: engine)
    assert state["status"] == "completed"
    assert len(state["candidates"]) == 2
    assert state["selected"]["parameters"] == {"amount": 2}
    assert state["retest"]["mean_degradation"] == pytest.approx(-0.1)
    assert state["retest"]["reported_degradation"] == pytest.approx(-10)
    assert state["retest"]["reported_unit"] == "gap_percentage_points"
    assert engine.feedback == [
        (0, 0),
        (2, pytest.approx(0.1)),
        (1, pytest.approx(0.05)),
    ]
    assert [item[2] for item in calls] == [[1, 2]] * 3 + [[3, 4]] * 2
    run_search(
        search, tmp_path, lambda *args: pytest.fail("completed searches must not rerun")
    )


def test_finite_parameter_space_exhaustion_selects_and_retests_best_candidate(
    search, tmp_path
):
    search["max_configurations"] = 16
    engine = ExhaustingSearchStub(search, tmp_path)
    calls = []

    def evaluate(name, parameters, instances, seeds, fidelity_stage):
        del instances, fidelity_stage
        calls.append(name)
        return [
            record(seed=seed, objective=100 + parameters["amount"]) for seed in seeds
        ]

    state = run_search(search, tmp_path, evaluate, engine_factory=lambda *args: engine)

    assert state["status"] == "completed"
    assert state["selection_reason"] == "largest_observed_degradation"
    assert state["selected"]["parameters"] == {"amount": 1}
    assert state["search_stopped"] is False
    assert calls == ["default", "candidate-0001", "retest-default", "retest-selected"]


def test_failed_configuration_stops_without_numeric_feedback_and_is_retested(
    search, tmp_path
):
    calls = []
    engine = SearchStub(search, tmp_path)

    def evaluate(name, parameters, instances, seeds, fidelity_stage):
        del instances, fidelity_stage
        calls.append(name)
        return [
            record(
                seed=seed,
                execution_status="solver_error" if parameters["amount"] else "returned",
            )
            for seed in seeds
        ]

    state = run_search(search, tmp_path, evaluate, engine_factory=lambda *args: engine)
    assert state["status"] == "stopped_after_unavailable_feedback"
    assert len(state["candidates"]) == 1
    assert state["retest"]["mean_degradation"] is None
    assert engine.feedback == [(0, 0)]
    assert calls == ["default", "candidate-0001", "retest-default", "retest-selected"]
    assert state["selected"]["parameters"] == {"amount": 2}


def test_missing_default_does_not_start_search(search, tmp_path):
    state = run_search(
        search,
        tmp_path,
        lambda *args: [],
        engine_factory=lambda *args: pytest.fail("no baseline"),
    )
    assert state["status"] == "baseline_unavailable"


def test_collection_fault_pauses_without_claiming_a_parameter_weakness(
    search, tmp_path
):
    engine = SearchStub(search, tmp_path)

    def evaluate(name, parameters, instances, seeds, fidelity_stage):
        del instances, fidelity_stage
        return [
            record(
                seed=seed,
                execution_status="collector_error"
                if parameters["amount"]
                else "returned",
            )
            for seed in seeds
        ]

    state = run_search(search, tmp_path, evaluate, engine_factory=lambda *args: engine)
    assert state["status"] == "collection_paused"
    assert "selected" not in state and engine.feedback == [(0, 0)]
    assert state["pending"]["summary"]["unavailable"] == {"collector_error": 2}


def test_solver_seed_sets_must_be_disjoint(search):
    validate_search(search)
    search["retest_seeds"] = [2, 3]
    with pytest.raises(ValueError, match="disjoint"):
        validate_search(search)


def test_quality_feedback_requires_objective_sense_and_nonzero_anchor(search):
    search["objective_sense"] = None
    with pytest.raises(ValueError, match="requires an objective sense"):
        validate_search(search)

    search["objective_sense"] = "minimize"
    search["instances"][0]["bks"] = 0
    with pytest.raises(ValueError, match="finite nonzero BKS"):
        validate_search(search)


def test_configuration_preparation_limits_instances_without_truncating_seeds(
    tmp_path, monkeypatch
):
    config = tmp_path / "configuration.json"
    config.write_text(
        json.dumps(
            {
                "searcher_version": "2.4.0",
                "max_configurations": 1,
                "instance_count": 10,
                "solver_seed_count": 30,
                "watchdog_grace_sec": 60,
                "panels": [
                    {
                        "task_config": "configs/tasks/pyvrp_v0_14_0.yaml",
                        "collector": "pyvrp",
                        "feedback": "normalized_gap",
                        "instance_source": "configs/instance_sets/pyvrp_cvrp_v1_dev.yaml",
                        "seed_source": "configs/tasks/pyvrp_v0_14_0.yaml",
                        "seed_key": "evaluation.seed_robustness.development_seeds",
                        "seed_max": 4294967295,
                        "parameters": {
                            "ils.history_length": {
                                "type": "integer",
                                "bounds": [300, 500],
                                "default": 300,
                            }
                        },
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(
        "pitbench.evaluator.configuration.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout='{"version": "0.14.0"}\n', returncode=0
        ),
    )

    manifest = prepare(
        config,
        tmp_path / "output",
        solver_pythons={},
        cpus=[min(os.sched_getaffinity(0))],
        search_seed=20260918,
        instances_per_panel=1,
    )

    assert manifest["instances_per_panel"] == 1
    assert len(manifest["searches"]) == 2
    assert all(len(search["instances"]) == 1 for search in manifest["searches"])
    assert all(len(search["solver_seeds"]) == 30 for search in manifest["searches"])


def test_real_smac_accepts_negative_costs_and_retains_pending_trial(search, tmp_path):
    pytest.importorskip("smac")
    search["max_configurations"] = 5
    search["parameters"]["enabled"] = {
        "type": "categorical",
        "choices": [True, False],
        "default": True,
    }
    engine = SmacSearch(search, tmp_path)
    json.dumps(engine.metadata, allow_nan=False)
    info, params = engine.ask()
    json.dumps(params, allow_nan=False)
    assert params == {"amount": 0, "enabled": True}
    engine.tell(info, 0)
    info, params = engine.ask()
    json.dumps(params, allow_nan=False)
    assert params != {"amount": 0, "enabled": True}
    # Resume the exact pending panel; do not ask for another configuration.
    resumed = SmacSearch(search, tmp_path)
    resumed_info, resumed_params = resumed.ask()
    assert resumed_params == params
    resumed.tell(resumed_info, 0.25)
    assert resumed.facade.runhistory.get_cost(resumed_info.config) == -0.25
    # Exercise model training on signed observations, not just runhistory storage.
    for _ in range(3):
        next_info, next_params = resumed.ask()
        json.dumps(next_params, allow_nan=False)
        resumed.tell(next_info, next_params["amount"] / 10)


def test_run_attempts_preserve_collection_failures_and_require_explicit_retry(
    search, tmp_path, monkeypatch
):
    cpu = min(os.sched_getaffinity(0))
    search.update(
        collector="example",
        solver={},
        fixed_options={},
        solver_python=sys.executable,
        watchdog_grace_sec=60,
        task_configuration={
            "release": {"base_commit": "fixture"},
            "repository": {"plugin": "example:Plugin"},
        },
    )
    invocations = []

    def execute(command, directory, identity, **kwargs):
        job_path = Path(command[command.index("--job") + 1])
        job = json.loads(job_path.read_text())
        invocations.append(job_path)
        result = {**identity, **job, "execution_status": "collector_error"}
        (directory / "result.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr(
        "pitbench.evaluator.configuration.run_collection_process", execute
    )
    runner = ConfigurationRunner(tmp_path, [cpu])
    runner.panel(search, "default", {"amount": 0}, search["instances"], [1, 2], "full")
    runner.panel(search, "default", {"amount": 0}, search["instances"], [1, 2], "full")
    assert len(invocations) == 1  # The other planned slot is retained but not run.
    retry = ConfigurationRunner(tmp_path, [cpu], retry_collection_errors=True)
    retry.panel(search, "default", {"amount": 0}, search["instances"], [1, 2], "full")
    assert len(invocations) == 2 and all(path.exists() for path in invocations)


def test_promoted_panel_reuses_previously_completed_nested_positions(
    search, tmp_path, monkeypatch
):
    cpu = min(os.sched_getaffinity(0))
    search.update(
        collector="example",
        solver={},
        fixed_options={},
        solver_python=sys.executable,
        watchdog_grace_sec=60,
        task_configuration={
            "release": {"base_commit": "fixture"},
            "repository": {"plugin": "example:Plugin"},
        },
        instances=[{"id": "a", "bks": 100}, {"id": "b", "bks": 100}],
    )
    invocations = []

    def execute(command, directory, identity, **kwargs):
        job_path = Path(command[command.index("--job") + 1])
        job = json.loads(job_path.read_text())
        invocations.append(job)
        result = {**identity, **job, "execution_status": "returned"}
        (directory / "result.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr(
        "pitbench.evaluator.configuration.run_collection_process", execute
    )
    runner = ConfigurationRunner(tmp_path, [cpu])
    runner.panel(
        search,
        "candidate-0001",
        {"amount": 1},
        search["instances"][:1],
        [1],
        "screen",
    )
    runner.panel(
        search,
        "candidate-0001",
        {"amount": 1},
        search["instances"],
        [1, 2],
        "middle",
    )

    assert len(invocations) == 4
    assert {(item["instance_id"], item["solver_seed"]) for item in invocations} == {
        ("a", 1),
        ("a", 2),
        ("b", 1),
        ("b", 2),
    }


def test_resume_collection_fault_keeps_the_same_smac_candidate(search, tmp_path):
    pytest.importorskip("smac")
    search["max_configurations"] = 1
    healthy = False
    candidate_parameters = []

    def evaluate(name, parameters, instances, seeds, fidelity_stage):
        del instances, fidelity_stage
        if name.startswith("candidate"):
            candidate_parameters.append(parameters)
        return [
            record(
                seed=seed,
                objective=100 + parameters["amount"],
                execution_status="collector_error"
                if name.startswith("candidate") and not healthy
                else "returned",
            )
            for seed in seeds
        ]

    paused = run_search(search, tmp_path, evaluate)
    assert paused["status"] == "collection_paused"
    healthy = True
    finished = run_search(search, tmp_path, evaluate, retry_collection_errors=True)
    assert finished["status"] == "completed"
    assert len(finished["candidates"]) == 1
    assert len(candidate_parameters) == 2
    assert candidate_parameters[0] == candidate_parameters[1]
