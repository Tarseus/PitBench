from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from adapters.pitbench.adapter import PitBenchAdapter
from adapters.pitbench.agent_tooling import write_agent_tooling
from pitbench import agent_cli, agent_diagnostics
from pitbench.agent_tools import AgentTool
from pitbench.cli.evaluate_config import EvaluationConfig
from pitbench.instances import materialize_instance_set
from pitbench.schema.task import PitBenchTask
from scripts.check_feedback_ablation import compare as compare_feedback_arms

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TOOLS = frozenset({AgentTool.INSPECT, AgentTool.WORKSPACE, AgentTool.BENCH})


@pytest.fixture
def candidate_hypothesis() -> dict:
    return {
        "id": "cache_probe",
        "mechanism": "Retain repeated distance queries within an unchanged search state.",
        "applicability": ["Repeated queries exceed cache maintenance cost."],
        "failure_hypotheses": ["Low reuse increases time and retained memory."],
        "algorithm_state_features": ["Observed hit count since the last reset."],
        "correctness_invariants": ["Cached values equal uncached distances."],
        "experiments": [
            {
                "id": "feature_off",
                "controlled_change": "Disable only the cache under identical input and seed.",
                "expected_observation": "The objective trajectory is identical and CPU cost falls.",
                "falsifies_if": "Distance values differ or CPU cost does not decrease.",
                "metric_ids": ["cpu_time_ratio", "same_seed_replay_consistency"],
            }
        ],
    }


def test_bundled_hypothesis_registration_binds_real_patch_and_prevents_overwrite(
    tmp_path: Path,
    candidate_hypothesis: dict,
) -> None:
    task = PitBenchTask.from_yaml(ROOT / "configs/tasks/pyvrp_v0_14_0.yaml")
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    development = next(
        item for item in task.instance_sets if item.kind.value == "agent_dev"
    )
    materialize_instance_set(
        ROOT / development.instance_set_config, task_dir / "dev_instances"
    )
    write_agent_tooling(
        repository_root=ROOT,
        task=task,
        task_dir=task_dir,
        agent_tools=DEFAULT_TOOLS,
        candidate_hypotheses_required=True,
    )
    repository = tmp_path / "repo"
    repository.mkdir()
    for argv in [
        ["git", "init", "-q"],
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "Base",
        ],
    ]:
        subprocess.run(argv, cwd=repository, check=True, capture_output=True)
    (repository / "search.cpp").write_text("int cache_enabled = 1;\n")
    expected_hash = hashlib.sha256(agent_cli._candidate_patch(repository)).hexdigest()
    plan = tmp_path / "hypothesis.json"
    plan.write_text(json.dumps(candidate_hypothesis))
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(task_dir / "agent_tooling"),
            "PITBENCH_AGENT_CONFIG": str(task_dir / "agent_config.json"),
            "PITBENCH_REPO": str(repository),
            "PITBENCH_RUNS": str(tmp_path / "runs"),
        }
    )
    command = [
        sys.executable,
        "-m",
        "pitbench.agent_cli",
        "hypothesis",
        "--plan",
        str(plan),
    ]
    completed = subprocess.run(
        command,
        cwd=task_dir,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    printed = json.loads(completed.stdout)
    saved = Path(printed["record_path"])
    original = saved.read_bytes()
    assert printed["patch_sha256"] == expected_hash
    assert printed["hypothesis"] == candidate_hypothesis
    assert printed["record_sha256"] == hashlib.sha256(original).hexdigest()
    repeated = subprocess.run(
        command, cwd=task_dir, env=environment, capture_output=True, text=True
    )
    assert repeated.returncode != 0
    assert saved.read_bytes() == original


def test_hypothesis_rejects_empty_predictions_and_duplicate_experiments(
    candidate_hypothesis: dict,
) -> None:
    from pydantic import ValidationError

    from pitbench.schema.research import CandidateHypothesis

    candidate_hypothesis["applicability"] = ["   "]
    with pytest.raises(ValidationError):
        CandidateHypothesis.model_validate(candidate_hypothesis)
    candidate_hypothesis["applicability"] = ["Repeated queries"]
    candidate_hypothesis["experiments"] *= 2
    with pytest.raises(ValidationError, match="unique"):
        CandidateHypothesis.model_validate(candidate_hypothesis)


def test_mechanism_bench_requires_current_declaration_before_build_or_solve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate_hypothesis: dict,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = {
        "agent_tools": ["bench"],
        "task_id": "mechanism-test",
        "candidate_hypotheses_required": True,
        "development_seeds": [7],
        "budgets_sec": [5],
        "primary_budget_sec": 5,
        "threads": 1,
    }
    patch = [b"first candidate patch"]
    monkeypatch.setattr(agent_cli, "_config", lambda: config)
    monkeypatch.setattr(agent_cli, "_repository", lambda: tmp_path)
    monkeypatch.setattr(agent_cli, "_candidate_patch", lambda _: patch[0])
    monkeypatch.setenv("PITBENCH_RUNS", str(tmp_path / "runs"))
    calls = []
    monkeypatch.setattr(
        agent_cli, "_run_build_steps", lambda *a, **kw: calls.append("build") or True
    )
    monkeypatch.setattr(agent_cli, "_instances", lambda *_: [tmp_path / "dev.json"])
    monkeypatch.setattr(agent_cli, "_artifact_identity", lambda _: {})
    monkeypatch.setattr(
        agent_cli,
        "_run_one",
        lambda cfg, *a: calls.append(cfg.get("candidate_experiment")) or True,
    )
    args = agent_cli.parser().parse_args(["bench", "--build"])
    with pytest.raises(SystemExit, match="register the candidate"):
        agent_cli.bench_command(args)
    assert calls == []
    plan = tmp_path / "hypothesis.json"
    plan.write_text(json.dumps(candidate_hypothesis))
    assert agent_cli.hypothesis_command(argparse.Namespace(plan=plan)) == 0
    registered = json.loads(capsys.readouterr().out)
    args.hypothesis_id = "cache_probe"
    args.experiment_id = "not_declared"
    with pytest.raises(SystemExit, match="absent"):
        agent_cli.bench_command(args)
    assert calls == []
    args.experiment_id = "feature_off"
    assert agent_cli.bench_command(args) == 0
    assert calls[0] == "build"
    assert calls[1]["record_sha256"] == registered["record_sha256"]
    calls.clear()
    patch[0] = b"second candidate patch"
    with pytest.raises(SystemExit, match="current source patch"):
        agent_cli.bench_command(args)
    assert calls == []
    args.hypothesis_id = args.experiment_id = None
    patch[0] = b""
    args.label = "base"
    assert agent_cli.bench_command(args) == 0
    assert calls == ["build", None]
    calls.clear()
    patch[0] = b"second candidate patch"
    args.label = "agent"
    config["candidate_hypotheses_required"] = False
    assert agent_cli.bench_command(args) == 0
    assert calls == ["build", None]


@pytest.mark.parametrize("problem", ["unknown_metric", "empty_patch"])
def test_invalid_hypothesis_cannot_be_registered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate_hypothesis: dict,
    problem: str,
) -> None:
    monkeypatch.setattr(agent_cli, "_config", lambda: {"task_id": "test"})
    monkeypatch.setattr(agent_cli, "_repository", lambda: tmp_path)
    monkeypatch.setattr(
        agent_cli,
        "_candidate_patch",
        lambda _: b"" if problem == "empty_patch" else b"patch",
    )
    monkeypatch.setenv("PITBENCH_RUNS", str(tmp_path / "runs"))
    if problem == "unknown_metric":
        candidate_hypothesis["experiments"][0]["metric_ids"] = ["invented_metric"]
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(candidate_hypothesis))
    with pytest.raises(SystemExit):
        agent_cli.hypothesis_command(argparse.Namespace(plan=plan))
    assert not (tmp_path / "runs").exists()


def test_agent_command_materializes_fixed_solver_parameters(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    command = agent_cli._command(
        {
            "runner": ["python3", "solver.py", "--instance", "{instance}"],
            "fixed_solver_parameters": {"search_strategy": "dom_over_wdeg"},
        },
        tmp_path / "instance.json",
        output,
        0,
        5.0,
        1,
    )

    assert command[-2:] == ["--parameters", str(output.with_suffix(".parameters.json"))]
    assert json.loads(output.with_suffix(".parameters.json").read_text()) == {
        "search_strategy": "dom_over_wdeg"
    }


def test_default_tools_and_explicit_repository_only_condition() -> None:
    assert set(EvaluationConfig().agent_tools) == DEFAULT_TOOLS
    assert EvaluationConfig.model_validate({"agent_tools": []}).agent_tools == []
    assert EvaluationConfig().agent_feedback_mode == "solver_behavior"
    assert (
        EvaluationConfig.model_validate(
            {"agent_feedback_mode": "outcome_only"}
        ).agent_feedback_mode
        == "outcome_only"
    )
    with pytest.raises(ValueError, match="agent_feedback_mode"):
        EvaluationConfig.model_validate({"agent_feedback_mode": "unknown"})


def test_feedback_arm_configs_must_match_except_feedback_mode(tmp_path: Path) -> None:
    outcome = tmp_path / "outcome.yaml"
    behavior = tmp_path / "behavior.yaml"
    outcome.write_text(
        "agent_tools: [inspect, bench]\nagent_feedback_mode: outcome_only\n"
    )
    behavior.write_text(
        "agent_tools: [inspect, bench]\nagent_feedback_mode: solver_behavior\n"
    )

    assert compare_feedback_arms(outcome, behavior)["matched"] is True
    behavior.write_text(behavior.read_text() + "judge_cpus: 4\n")
    with pytest.raises(ValueError, match="other than feedback mode"):
        compare_feedback_arms(outcome, behavior)


@pytest.mark.parametrize(
    "task_config",
    sorted((ROOT / "configs" / "tasks").glob("*.yaml")),
    ids=lambda path: path.stem,
)
def test_instruction_matches_protocol_and_visible_files(
    task_config: Path,
) -> None:
    task = PitBenchTask.from_yaml(task_config)
    instruction = PitBenchAdapter._format_instruction(task, DEFAULT_TOOLS)
    assert "Optimization Scope" not in instruction
    assert task.optimization_scope not in instruction
    assert instruction.isascii()
    assert f"{task.evaluation.primary_budget_sec:.1f}s" in instruction
    assert "/pitbench/dev_instances" in instruction
    assert "pitbench bench --label base" in instruction
    assert "pitbench bench --label agent --build" in instruction
    assert "pitbench diagnose" in instruction
    assert "incumbent or bound progress" in instruction
    assert "git diff HEAD" in instruction
    assert "If no source changes are retained" in instruction
    if task.task_type.value == "exact_solver":
        assert "independently verified solved coverage" in instruction
        assert "paired PAR-2 runtime" in instruction
        assert "normalized gap" not in instruction
    else:
        assert "paired reduction in normalized gap" in instruction

    without_tools = PitBenchAdapter._format_instruction(task)
    assert "No PitBench development instances are mounted" in without_tools
    assert "pitbench bench" not in without_tools


@pytest.mark.parametrize(
    "task_id",
    [
        "vroom_v1_15_0",
        "pyvrp_v0_14_0",
        "pyvrp_search_20260728",
        "pyvrp_search_20260806",
        "choco_v6_0_1",
        "ortools_cp_sat_solve_v9_15",
        "scip_v9_2_4",
        "cbc_v2_10_12",
    ],
)
def test_public_tooling_has_task_protocol_and_instances(
    tmp_path: Path, task_id: str
) -> None:
    task = PitBenchTask.from_yaml(ROOT / "configs" / "tasks" / f"{task_id}.yaml")
    task_dir = tmp_path / task_id
    task_dir.mkdir()
    development = next(
        item for item in task.instance_sets if item.kind.value == "agent_dev"
    )
    materialize_instance_set(
        ROOT / development.instance_set_config, task_dir / "dev_instances"
    )
    write_agent_tooling(
        repository_root=ROOT,
        task=task,
        task_dir=task_dir,
        agent_tools=DEFAULT_TOOLS,
    )
    config = json.loads((task_dir / "agent_config.json").read_text())
    assert config["primary_budget_sec"] == task.evaluation.primary_budget_sec
    assert config["agent_feedback_mode"] == "solver_behavior"
    assert config["performance_protocol"] == task.evaluation.performance_protocol.value
    assert len(config["development_targets"]) == development.size
    if task_id == "choco_v6_0_1":
        assert all(
            target["target"] is not None and target["proven_optimal"] is True
            for target in config["development_targets"].values()
        )
    assert config["build_commands"]
    if task_id == "pyvrp_v0_14_0":
        assert "/tmp/pitbench-pyvrp-build" in config["build_commands"][0]["argv"]
    if task_id == "ortools_cp_sat_solve_v9_15":
        assert not any(
            command["argv"][:2] == ["cmake", "-S"]
            for command in config["build_commands"]
        )
    assert set(config["development_targets"]) == {
        path.name for path in (task_dir / "dev_instances").glob("dev_*.*")
    }
    assert (
        task_dir / "agent_tooling/pitbench/problem_families/public_verification.py"
    ).is_file()
    assert (task_dir / "agent_tooling/pitbench/agent_diagnostics.py").is_file()
    dockerfile = PitBenchAdapter._prepared_dockerfile(
        "local-agent-image", task, agent_tools=DEFAULT_TOOLS
    )
    assert "chmod -R a+rX /opt/pitbench-tooling /pitbench/config.json" in dockerfile
    assert "chmod -R a+rX /pitbench/dev_instances" in dockerfile
    assert "/home/pitbench-agent/.m2/repository" in dockerfile

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(task_dir / "agent_tooling"),
            "PITBENCH_AGENT_CONFIG": str(task_dir / "agent_config.json"),
            "PITBENCH_DEV_INSTANCES": str(task_dir / "dev_instances"),
            "PITBENCH_REPO": str(ROOT),
            "PITBENCH_RUNS": str(task_dir / "runs"),
        }
    )
    inspected = subprocess.run(
        [sys.executable, "-m", "pitbench.agent_cli", "inspect"],
        cwd=task_dir,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(inspected.stdout)["primary_budget_sec"] == (
        task.evaluation.primary_budget_sec
    )
    bench = subprocess.run(
        [sys.executable, "-m", "pitbench.agent_cli", "bench", "--dry-run"],
        cwd=task_dir,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    commands = [json.loads(line) for line in bench.stdout.splitlines()]
    assert len(commands) == development.size
    assert all(command["label"] == "agent" for command in commands)


def test_public_tooling_records_outcome_only_ablation_mode(tmp_path: Path) -> None:
    task = PitBenchTask.from_yaml(ROOT / "configs/tasks/pyvrp_v0_14_0.yaml")
    task_dir = tmp_path / "outcome"
    task_dir.mkdir()
    development = next(
        item for item in task.instance_sets if item.kind.value == "agent_dev"
    )
    materialize_instance_set(
        ROOT / development.instance_set_config, task_dir / "dev_instances"
    )

    write_agent_tooling(
        repository_root=ROOT,
        task=task,
        task_dir=task_dir,
        agent_tools=DEFAULT_TOOLS,
        agent_feedback_mode="outcome_only",
    )

    config = json.loads((task_dir / "agent_config.json").read_text())
    assert config["agent_feedback_mode"] == "outcome_only"
    assert (
        config["development_seeds"] == task.evaluation.seed_robustness.development_seeds
    )


def test_highs_development_feedback_uses_public_independent_verifier(
    tmp_path: Path,
) -> None:
    from pitbench.problem_families.verification import (
        _public_scheduling_processing_times,
    )

    task = PitBenchTask.from_yaml(ROOT / "configs/tasks/highs_v1_15_1.yaml")
    task_dir = tmp_path / "highs"
    task_dir.mkdir()
    instance = materialize_instance_set(
        ROOT / "configs/instance_sets/highs_v1_15_1_dev.yaml",
        task_dir / "dev_instances",
    )[0]
    write_agent_tooling(
        repository_root=ROOT,
        task=task,
        task_dir=task_dir,
        agent_tools=DEFAULT_TOOLS,
    )
    assert (
        task_dir / "agent_tooling/pitbench/problem_families/verification.py"
    ).is_file()
    config = json.loads((task_dir / "agent_config.json").read_text())
    target = config["development_targets"][instance.name]
    assert target["proven_optimal"] is True
    assert target["target"] is not None

    reference = json.loads(
        (task_dir / "dev_instances/bks_solutions/dev_0000.solution.json").read_text()
    )
    starts = reference["start_times"]
    processing = _public_scheduling_processing_times(instance)
    assert processing is not None
    values = {f"s{index}": value for index, value in enumerate(starts)}
    for first in range(len(starts)):
        for second in range(first + 1, len(starts)):
            values[f"z{first}_{second}"] = int(
                starts[first] + processing[first] <= starts[second]
            )
    output = tmp_path / "highs-result.json"
    output.with_suffix(".solution.json").write_text(
        json.dumps(
            {
                "raw_solution": "\n".join(
                    [
                        f"Objective {target['target']}",
                        f"# Columns {len(values)}",
                        *[f"{name} {value}" for name, value in values.items()],
                        "# Rows 0",
                    ]
                )
            }
        )
    )
    feedback = agent_cli._feedback(
        config,
        {
            "valid": True,
            "objective": target["target"],
            "solver_termination": "optimal",
            "cpu_time_sec": 0.5,
        },
        target["target"],
        10.0,
        instance,
        output,
    )
    assert feedback["verification"] == "independent_public"
    assert feedback["verified_valid"] is True
    assert feedback["preview_solved"] is True

    output.with_suffix(".solution.json").write_text(
        json.dumps({"raw_solution": "Objective 0\n# Columns 0\n# Rows 0"})
    )
    invalid = agent_cli._feedback(
        config,
        {"valid": True, "objective": target["target"], "solver_termination": "optimal"},
        target["target"],
        10.0,
        instance,
        output,
    )
    assert invalid["verified_valid"] is False
    assert invalid["preview_solved"] is False

    custom_instance = tmp_path / "custom.lp"
    custom_instance.write_text(instance.read_text() + "\n\\ custom constraint\n")
    unverified = agent_cli._feedback(
        config,
        {"valid": True, "objective": target["target"], "solver_termination": "optimal"},
        None,
        10.0,
        custom_instance,
        output,
    )
    assert unverified["verification"] == "unavailable"
    assert unverified["verified_valid"] is None
    assert unverified["preview_solved"] is None


def test_bench_defaults_to_primary_budget_and_allows_experimental_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = {
        "budgets_sec": [5.0, 20.0],
        "primary_budget_sec": 20.0,
        "development_seeds": [7, 8],
        "threads": 1,
    }
    calls: list[tuple[int, float, str, int]] = []
    monkeypatch.setattr(agent_cli, "_config", lambda: config)
    monkeypatch.setattr(agent_cli, "_repository", lambda: tmp_path)
    monkeypatch.setattr(agent_cli, "_candidate_patch", lambda _: b"")
    monkeypatch.setattr(
        agent_cli, "_instances", lambda _selected, _config: [tmp_path / "dev_0000.json"]
    )
    monkeypatch.setattr(
        agent_cli,
        "_run_one",
        lambda _config, _instance, seed, budget, _dry, label, _hash, threads, _artifacts, _runs_root: (
            calls.append((seed, budget, label, threads)) or True
        ),
    )
    args = argparse.Namespace(
        split="dev",
        seed=None,
        budget=None,
        threads=None,
        instance=None,
        dry_run=False,
        label="agent",
        build=False,
    )
    assert agent_cli.bench_command(args) == 0
    assert calls == [(7, 20.0, "agent", 1)]
    args.budget = 3.0
    args.threads = 4
    assert agent_cli.bench_command(args) == 0
    assert calls[-1] == (7, 3.0, "agent", 4)
    args.budget = None
    args.all_seeds = True
    args.all_budgets = True
    calls.clear()
    assert agent_cli.bench_command(args) == 0
    assert calls == [
        (7, 5.0, "agent", 4),
        (7, 20.0, "agent", 4),
        (8, 5.0, "agent", 4),
        (8, 20.0, "agent", 4),
    ]
    args.all_seeds = False
    args.all_budgets = False
    args.budget = -1.0
    with pytest.raises(SystemExit, match="positive finite"):
        agent_cli.bench_command(args)
    args.budget = 20.0
    args.build = True
    built: list[bool] = []
    monkeypatch.setattr(
        agent_cli,
        "_run_build_steps",
        lambda _config, *, dry_run, runs_root: built.append(dry_run) or True,
    )
    assert agent_cli.bench_command(args) == 0
    assert built == [False]


def test_runs_root_falls_back_when_configured_directory_is_read_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("PITBENCH_RUNS", "/proc/pitbench-read-only")
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

    run_root = agent_cli._runs_root({"task_id": "task-example"})

    assert run_root == tmp_path / "pitbench-runs" / "task-example"
    assert run_root.is_dir()
    assert "using" in capsys.readouterr().err


def test_trajectory_summary_retains_search_progress(tmp_path: Path) -> None:
    trajectory = tmp_path / "run.trajectory.jsonl"
    trajectory.write_text(
        "\n".join(
            [
                json.dumps({"time_sec": 1.0, "objective": 10, "nodes": 2}),
                "not json",
                json.dumps({"time_sec": 2.0, "objective": 8, "nodes": 5}),
                json.dumps({"time_sec": 3.0, "objective": 8, "nodes": 7}),
            ]
        )
    )

    summary = agent_diagnostics.trajectory_summary(trajectory, "minimize")

    assert summary["available"] is True
    assert summary["points"] == 3
    assert summary["best_objective"] == 8
    assert summary["best_updates"] == 2
    assert summary["observed_objective_points"] == 3
    assert summary["time_to_first_solution_sec"] == 1
    assert summary["time_to_best_sec"] == 2
    assert summary["final_nodes"] == 7


def test_pairing_does_not_treat_final_only_trajectory_as_search_progress() -> None:
    base = {
        "solver_behavior": {
            "work": {},
            "trajectory": {
                "available": True,
                "points": 1,
                "best_updates": 1,
                "time_to_best_sec": 1.0,
            },
        }
    }
    agent = {
        "solver_behavior": {
            "work": {},
            "trajectory": {
                "available": True,
                "points": 1,
                "best_updates": 1,
                "time_to_best_sec": 0.5,
            },
        }
    }

    comparison = agent_diagnostics.paired_behavior(agent, base)

    assert comparison["trajectory_artifact_available_for_both"] is True
    assert comparison["progress_trace_available_for_both"] is False
    assert comparison["time_to_best_reduction_sec"] is None


def test_bound_progress_does_not_invent_time_to_first_solution(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "base.jsonl"
    agent_path = tmp_path / "agent.jsonl"
    for path, final_time, bound_time in (
        (base_path, 1.0, 0.5),
        (agent_path, 0.8, 0.4),
    ):
        path.write_text(
            "\n".join(
                json.dumps(event)
                for event in (
                    {"event": "bound", "time_sec": 0.2, "dual_bound": 3},
                    {"event": "bound", "time_sec": bound_time, "dual_bound": 5},
                    {"event": "final", "time_sec": final_time, "objective": 5},
                )
            )
        )
    base = {
        "solver_behavior": {
            "trajectory": agent_diagnostics.trajectory_summary(base_path, "minimize")
        }
    }
    agent = {
        "solver_behavior": {
            "trajectory": agent_diagnostics.trajectory_summary(agent_path, "minimize")
        }
    }

    assert base["solver_behavior"]["trajectory"]["time_to_first_solution_sec"] is None
    assert agent["solver_behavior"]["trajectory"]["time_to_best_sec"] is None
    assert base["solver_behavior"]["trajectory"]["time_to_best_dual_bound_sec"] == 0.5

    comparison = agent_diagnostics.paired_behavior(agent, base)

    assert comparison["progress_trace_available_for_both"] is True
    assert comparison["objective_progress_available_for_both"] is False
    assert comparison["time_to_first_solution_reduction_sec"] is None
    assert comparison["time_to_best_reduction_sec"] is None
    assert comparison["bound_progress_available_for_both"] is True
    assert comparison["time_to_best_dual_bound_reduction_sec"] == pytest.approx(0.1)
    summary = agent_diagnostics.paired_behavior_summary([(base, agent)])
    assert summary["mean_time_to_best_dual_bound_reduction_sec"] == pytest.approx(0.1)


def test_dual_bound_progress_follows_objective_sense(tmp_path: Path) -> None:
    trajectory = tmp_path / "max.jsonl"
    trajectory.write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                {"time_sec": 0.1, "dual_bound": 12},
                {"time_sec": 0.2, "dual_bound": 15},
                {"time_sec": 0.3, "dual_bound": 10},
            )
        )
    )

    summary = agent_diagnostics.trajectory_summary(trajectory, "maximize")

    assert summary["best_dual_bound"] == 10
    assert summary["dual_bound_updates"] == 2
    assert summary["time_to_best_dual_bound_sec"] == 0.3


def test_contrastive_checkpoints_reveal_search_crossover_without_final_leakage(
    tmp_path: Path,
) -> None:
    base_output = tmp_path / "base.json"
    agent_output = tmp_path / "agent.json"
    for output, records in (
        (
            base_output,
            [
                {"time_sec": 1, "objective": 100, "dual_bound": 20},
                {"time_sec": 7, "objective": 80, "dual_bound": 40},
                {"event": "final", "time_sec": 10, "objective": 70},
            ],
        ),
        (
            agent_output,
            [
                {"time_sec": 1, "objective": 90, "dual_bound": 25},
                {"time_sec": 7, "objective": 85, "dual_bound": 45},
                {"event": "final", "time_sec": 10, "objective": 60},
            ],
        ),
    ):
        output.with_suffix(".trajectory.jsonl").write_text(
            "\n".join(json.dumps(record) for record in records)
        )
    config = {"objective_sense": "minimize"}
    base = {
        "instance_id": "one",
        "budget_sec": 10,
        "solver_behavior": agent_diagnostics.solver_behavior(
            config, {}, base_output, 10
        ),
    }
    agent = {
        "instance_id": "one",
        "budget_sec": 10,
        "solver_behavior": agent_diagnostics.solver_behavior(
            config, {}, agent_output, 10
        ),
    }

    checkpoints = agent_diagnostics.paired_behavior(agent, base)[
        "matched_progress_checkpoints"
    ]
    assert [point["candidate_objective_improvement"] for point in checkpoints] == [
        10,
        10,
        -5,
        -5,
    ]
    assert checkpoints[-1]["candidate_dual_bound_improvement"] == 5
    summary = agent_diagnostics.contrastive_progress_summary([(base, agent)])
    assert summary[0]["objective"]["better_instances"] == 1
    assert summary[-1]["objective"]["worse_instances"] == 1


def test_final_only_trace_is_not_anytime_progress(tmp_path: Path) -> None:
    trajectory = tmp_path / "final.jsonl"
    trajectory.write_text(
        json.dumps({"event": "final", "time_sec": 10, "objective": 1, "dual_bound": 1})
    )
    checkpoints = agent_diagnostics.trajectory_checkpoints(trajectory, "minimize", 10)
    assert all(point["best_objective"] is None for point in checkpoints)
    assert all(point["best_dual_bound"] is None for point in checkpoints)


def test_gap_integral_reports_only_observed_window(tmp_path: Path) -> None:
    trajectory = tmp_path / "search.jsonl"
    trajectory.write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                {"time_sec": 2, "primal_bound": 10, "dual_bound": 0},
                {"time_sec": 5, "primal_bound": 8, "dual_bound": 2},
                {"event": "final", "time_sec": 10, "primal_bound": 1, "dual_bound": 1},
            )
        )
    )

    summary = agent_diagnostics.observed_gap_integral(trajectory, "minimize", 10)

    assert summary["observed_duration_sec"] == 8
    assert summary["observed_fraction_of_budget"] == 0.8
    assert summary["observed_gap_area_sec"] == pytest.approx(6.75)
    assert summary["full_budget_observed"] is False


def test_gap_integral_missing_bound_remains_unavailable(tmp_path: Path) -> None:
    trajectory = tmp_path / "primal-only.jsonl"
    trajectory.write_text(json.dumps({"time_sec": 1, "objective": 5}))

    summary = agent_diagnostics.observed_gap_integral(trajectory, "minimize", 10)

    assert summary["observed_gap_area_sec"] is None
    assert summary["observed_duration_sec"] == 0


def test_gap_integral_uses_maximization_bound_direction(tmp_path: Path) -> None:
    trajectory = tmp_path / "max.jsonl"
    trajectory.write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                {"time_sec": 0, "primal_bound": 2, "dual_bound": 10},
                {"time_sec": 5, "primal_bound": 4, "dual_bound": 8},
            )
        )
    )

    summary = agent_diagnostics.observed_gap_integral(trajectory, "maximize", 10)

    assert summary["observed_gap_area_sec"] == pytest.approx(6.5)
    assert summary["full_budget_observed"] is True


def test_maximization_checkpoint_signs(tmp_path: Path) -> None:
    base_path = tmp_path / "base.jsonl"
    agent_path = tmp_path / "agent.jsonl"
    base_path.write_text(json.dumps({"time_sec": 1, "objective": 10, "dual_bound": 20}))
    agent_path.write_text(
        json.dumps({"time_sec": 1, "objective": 12, "dual_bound": 18})
    )
    base = {
        "solver_behavior": {
            "objective_sense": "maximize",
            "checkpoints": agent_diagnostics.trajectory_checkpoints(
                base_path, "maximize", 10
            ),
        }
    }
    agent = {
        "solver_behavior": {
            "objective_sense": "maximize",
            "checkpoints": agent_diagnostics.trajectory_checkpoints(
                agent_path, "maximize", 10
            ),
        }
    }
    first = agent_diagnostics.paired_behavior(agent, base)[
        "matched_progress_checkpoints"
    ][0]
    assert first["candidate_objective_improvement"] == 2
    assert first["candidate_dual_bound_improvement"] == 2


def _feedback_record(
    *,
    instance_id: str,
    instance_sha256: str,
    gap: float | None = None,
    solved: bool | None = None,
    par2: float | None = None,
    wall_time: float = 1.0,
) -> dict[str, object]:
    return {
        "instance_id": instance_id,
        "instance_sha256": instance_sha256,
        "seed": 0,
        "budget_sec": 10.0,
        "threads": 1,
        "performance_protocol": (
            "heuristic_fixed_budget" if gap is not None else "exact_verified_solve"
        ),
        "verified_valid": True,
        "verified_objective": 0.0,
        "preview_normalized_gap": gap,
        "preview_solved": solved,
        "preview_par2_sec": par2,
        "reported_wall_time_sec": wall_time,
        "reported_cpu_time_sec": wall_time,
        "solver_behavior": {
            "work": {"nodes": 10},
            "trajectory": {
                "available": True,
                "best_updates": 2,
                "time_to_first_solution_sec": 0.2,
                "time_to_best_sec": 0.8,
            },
        },
    }


def test_development_diagnosis_exposes_concentrated_heuristic_gain(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs"
    (run_root / "base").mkdir(parents=True)
    (run_root / "agent").mkdir()
    for index, (before, after) in enumerate(((0.10, 0.05), (0.20, 0.20))):
        digest = f"sha-{index}"
        base = _feedback_record(
            instance_id=f"dev_{index:04d}",
            instance_sha256=digest,
            gap=before,
        )
        agent = _feedback_record(
            instance_id=f"dev_{index:04d}",
            instance_sha256=digest,
            gap=after,
            wall_time=0.9,
        )
        agent["patch_sha256"] = "candidate"
        (run_root / "base" / f"{index}.feedback.json").write_text(json.dumps(base))
        (run_root / "agent" / f"{index}.feedback.json").write_text(json.dumps(agent))

    diagnosis = agent_cli._development_diagnosis(
        {
            "task_id": "heuristic",
            "performance_protocol": "heuristic_fixed_budget",
            "development_targets": {
                f"dev_{index:04d}.json": {"sha256": f"sha-{index}"}
                for index in range(2)
            },
            "development_seeds": [0],
            "threads": 1,
        },
        run_root,
        [10.0],
    )

    assert diagnosis["paired_runs"] == 2
    assert diagnosis["missing_declared_runs"] == 0
    assert diagnosis["instance_direction_counts"] == {
        "better": 1,
        "equal_or_unresolved": 1,
        "worse": 0,
    }
    assert (
        diagnosis["observed_pattern"] == "favorable_effect_concentrated_in_one_instance"
    )
    assert diagnosis["runtime"]["mean_wall_time_reduction_sec"] == pytest.approx(0.1)


def test_outcome_only_diagnosis_omits_structured_search_feedback(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs"
    (run_root / "base").mkdir(parents=True)
    (run_root / "agent").mkdir()
    base = _feedback_record(instance_id="one", instance_sha256="sha-one", gap=0.2)
    agent = _feedback_record(instance_id="one", instance_sha256="sha-one", gap=0.1)
    (run_root / "base/one.feedback.json").write_text(json.dumps(base))
    (run_root / "agent/one.feedback.json").write_text(json.dumps(agent))
    common = {
        "task_id": "ablation",
        "performance_protocol": "heuristic_fixed_budget",
        "development_targets": {"one.json": {"sha256": "sha-one"}},
        "development_seeds": [0],
        "threads": 1,
    }

    regular = agent_diagnostics.development_diagnosis(common, run_root, [10.0])
    limited = agent_diagnostics.development_diagnosis(
        {**common, "agent_feedback_mode": "outcome_only"}, run_root, [10.0]
    )

    assert regular["observed_pattern"] == limited["observed_pattern"]
    assert regular["instance_direction_counts"] == limited["instance_direction_counts"]
    assert "solver_behavior" in regular
    assert "solver_behavior" not in limited
    assert "behavior" not in limited["per_instance"][0]
    assert "contrastive_progress" not in limited["per_instance"][0]


def test_behavior_summary_weights_independent_instances_equally() -> None:
    pairs: list[tuple[dict[str, object], dict[str, object]]] = []
    for seed in (0, 1):
        base = _feedback_record(
            instance_id="many-seeds", instance_sha256="many", gap=0.1, wall_time=2
        )
        agent = _feedback_record(
            instance_id="many-seeds", instance_sha256="many", gap=0.1, wall_time=1
        )
        base["seed"] = seed
        agent["seed"] = seed
        pairs.append((base, agent))
    base = _feedback_record(
        instance_id="one-seed", instance_sha256="one", gap=0.1, wall_time=1
    )
    agent = _feedback_record(
        instance_id="one-seed", instance_sha256="one", gap=0.1, wall_time=2
    )
    pairs.append((base, agent))

    summary = agent_diagnostics.paired_behavior_summary(pairs)

    assert summary["mean_wall_time_reduction_sec"] == 0


def test_contrastive_summary_does_not_average_objective_scales() -> None:
    pairs = []
    for instance, base_value, candidate_value in (
        ("small", 10, 9),
        ("large", 10000, 9000),
    ):
        base = _feedback_record(instance_id=instance, instance_sha256=instance, gap=0.1)
        agent = _feedback_record(
            instance_id=instance, instance_sha256=instance, gap=0.1
        )
        for record, value in ((base, base_value), (agent, candidate_value)):
            record["solver_behavior"] = {
                "objective_sense": "minimize",
                "checkpoints": [
                    {
                        "budget_fraction": 1.0,
                        "time_sec": 10.0,
                        "best_objective": value,
                        "best_dual_bound": None,
                    }
                ],
            }
        pairs.append((base, agent))

    summary = agent_diagnostics.contrastive_progress_summary(pairs)

    assert summary[0]["objective"]["better_instances"] == 2
    assert summary[0]["objective"]["mean_candidate_improvement"] is None


def test_gap_area_summary_keeps_coverage_visible() -> None:
    base = _feedback_record(instance_id="one", instance_sha256="one", gap=0.1)
    agent = _feedback_record(instance_id="one", instance_sha256="one", gap=0.1)
    for record, area, coverage in ((base, 6.0, 0.8), (agent, 5.0, 0.5)):
        record["solver_behavior"] = {
            "observed_gap_integral": {
                "observed_gap_area_sec": area,
                "mean_gap_on_observed_window": area / (10 * coverage),
                "observed_fraction_of_budget": coverage,
            }
        }

    summary = agent_diagnostics.paired_gap_area_summary([(base, agent)])

    assert summary[0]["base_mean_observed_gap_area_sec"] == 6
    assert summary[0]["candidate_mean_observed_gap_area_sec"] == 5
    assert summary[0]["base_mean_covered_fraction"] == 0.8
    assert summary[0]["candidate_mean_covered_fraction"] == 0.5


def test_diagnosis_expands_the_grid_only_for_a_broad_shortlist(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    (run_root / "base").mkdir(parents=True)
    (run_root / "agent").mkdir()
    for index in range(2):
        digest = f"broad-{index}"
        base = _feedback_record(
            instance_id=f"dev_{index:04d}",
            instance_sha256=digest,
            gap=0.2,
        )
        agent = _feedback_record(
            instance_id=f"dev_{index:04d}",
            instance_sha256=digest,
            gap=0.1,
        )
        (run_root / "base" / f"{index}.feedback.json").write_text(json.dumps(base))
        (run_root / "agent" / f"{index}.feedback.json").write_text(json.dumps(agent))
    config = {
        "task_id": "adaptive",
        "performance_protocol": "heuristic_fixed_budget",
        "development_targets": {
            f"dev_{index:04d}.json": {"sha256": f"broad-{index}"} for index in range(2)
        },
        "development_seeds": [0, 1],
        "threads": 1,
    }

    diagnosis = agent_cli._development_diagnosis(config, run_root, [10.0])

    assert diagnosis["observed_pattern"] == "favorable_effect_across_multiple_instances"
    assert diagnosis["missing_declared_runs"] == 2
    assert any(
        "--all-seeds" in action for action in diagnosis["recommended_next_actions"]
    )

    for path in (run_root / "agent").glob("*.feedback.json"):
        unchanged = json.loads(path.read_text())
        unchanged["preview_normalized_gap"] = 0.2
        path.write_text(json.dumps(unchanged))
    no_signal = agent_cli._development_diagnosis(config, run_root, [10.0])
    assert no_signal["observed_pattern"] == "no_detected_solver_outcome_change"
    assert not any(
        "--all-seeds" in action for action in no_signal["recommended_next_actions"]
    )


def test_development_diagnosis_uses_exact_coverage_before_par2(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs"
    (run_root / "base").mkdir(parents=True)
    (run_root / "agent").mkdir()
    pairs = ((False, True, 20.0, 1.0), (True, True, 1.0, 2.0))
    for index, (base_solved, agent_solved, base_par2, agent_par2) in enumerate(pairs):
        digest = f"exact-{index}"
        base = _feedback_record(
            instance_id=f"dev_{index:04d}",
            instance_sha256=digest,
            solved=base_solved,
            par2=base_par2,
        )
        agent = _feedback_record(
            instance_id=f"dev_{index:04d}",
            instance_sha256=digest,
            solved=agent_solved,
            par2=agent_par2,
        )
        (run_root / "base" / f"{index}.feedback.json").write_text(json.dumps(base))
        (run_root / "agent" / f"{index}.feedback.json").write_text(json.dumps(agent))

    diagnosis = agent_cli._development_diagnosis(
        {
            "task_id": "exact",
            "performance_protocol": "exact_verified_solve",
            "development_targets": {
                f"dev_{index:04d}.json": {"sha256": f"exact-{index}"}
                for index in range(2)
            },
            "development_seeds": [0],
            "threads": 1,
        },
        run_root,
        [10.0],
    )

    assert diagnosis["instance_direction_counts"] == {
        "better": 1,
        "equal_or_unresolved": 0,
        "worse": 1,
    }
    assert diagnosis["observed_pattern"] == "mixed_instance_effect"


def test_diagnose_command_saves_the_printed_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_root = tmp_path / "runs"
    (run_root / "base").mkdir(parents=True)
    (run_root / "agent").mkdir()
    base = _feedback_record(instance_id="dev_0000", instance_sha256="one", gap=0.2)
    agent = _feedback_record(instance_id="dev_0000", instance_sha256="one", gap=0.1)
    (run_root / "base" / "one.feedback.json").write_text(json.dumps(base))
    (run_root / "agent" / "one.feedback.json").write_text(json.dumps(agent))
    monkeypatch.setenv("PITBENCH_RUNS", str(run_root))
    monkeypatch.setattr(
        agent_cli,
        "_config",
        lambda: {
            "task_id": "diagnose-test",
            "performance_protocol": "heuristic_fixed_budget",
            "primary_budget_sec": 10.0,
            "budgets_sec": [5.0, 10.0],
            "development_targets": {"dev_0000.json": {"sha256": "one"}},
            "development_seeds": [0],
            "threads": 1,
        },
    )

    assert (
        agent_cli.diagnose_command(argparse.Namespace(budget=None, all_budgets=False))
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    saved = json.loads((run_root / "diagnosis.json").read_text())
    assert saved == printed
    assert saved["schema_version"] == 2
    assert "possible_explanations" in saved
    assert "diagnostic_hypotheses" not in saved
    assert saved["budgets_sec"] == [10.0]
    assert saved["saved_to"] == str(run_root / "diagnosis.json")


def test_bench_full_grid_flags_exclude_single_value_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_cli, "_config", lambda: {"agent_tools": ["bench"]})
    command_line = agent_cli.parser()
    args = command_line.parse_args(["bench", "--all-seeds", "--all-budgets"])
    assert args.all_seeds is True
    assert args.all_budgets is True
    with pytest.raises(SystemExit):
        command_line.parse_args(["bench", "--seed", "7", "--all-seeds"])
    with pytest.raises(SystemExit):
        command_line.parse_args(["bench", "--budget", "5", "--all-budgets"])
    diagnosis = command_line.parse_args(["diagnose", "--all-budgets"])
    assert diagnosis.all_budgets is True


@pytest.mark.parametrize("failure", ["timeout", "start_error", "unavailable"])
def test_public_bench_records_runner_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    failure: str,
) -> None:
    instance = tmp_path / "dev_0000.json"
    instance.write_text("{}")
    monkeypatch.setenv("PITBENCH_REPO", str(tmp_path))
    monkeypatch.setenv("PITBENCH_RUNS", str(tmp_path / "runs"))
    monkeypatch.setattr(
        agent_cli,
        "_runner_available",
        lambda _config: (
            failure != "unavailable",
            "solver is missing" if failure == "unavailable" else "ready",
        ),
    )
    if failure == "timeout":
        runner_error: Exception = subprocess.TimeoutExpired(
            ["solver"], 65, output=b"partial output", stderr=b"partial error"
        )
    else:
        runner_error = FileNotFoundError("solver is missing")

    def raise_runner_error(*_args: object, **_kwargs: object) -> None:
        raise runner_error

    monkeypatch.setattr(agent_cli.subprocess, "run", raise_runner_error)
    config = {
        "runner": ["solver", "{instance}", "{output}"],
        "threads": 1,
        "budgets_sec": [5.0],
        "performance_protocol": "heuristic_fixed_budget",
    }
    assert not agent_cli._run_one(config, instance, 0, 5.0, False, "agent", "patch")
    feedback = json.loads(capsys.readouterr().out)
    assert feedback["returncode"] is None
    expected_error = {
        "timeout": "runner timed out",
        "start_error": "runner could not start",
        "unavailable": "runner unavailable",
    }
    assert feedback["result_error"].startswith(expected_error[failure])
    assert Path(feedback["result"]).with_suffix(".feedback.json").is_file()
    assert Path(feedback["result"]).with_suffix(".stdout.log").read_text() == (
        "partial output" if failure == "timeout" else ""
    )


def test_public_bench_pairs_independently_checked_cvrp_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    instance = tmp_path / "dev_0000.json"
    instance.write_text(
        json.dumps(
            {
                "coordinates": [[0, 0], [1, 0]],
                "demands": [0, 1],
                "capacity": 1,
                "distance_metric": "EUC_2D",
            }
        )
    )
    runner = tmp_path / "runner.py"
    runner.write_text(
        "import json, pathlib, sys\n"
        "output = pathlib.Path(sys.argv[1])\n"
        "output.write_text(json.dumps({'valid': True, 'objective': 999, "
        "'wall_time_sec': 0.1, 'solver_termination': 'time_limit'}))\n"
        "output.with_suffix('.solution.json').write_text("
        "json.dumps({'routes': [[1]]}))\n"
    )
    config = {
        "task_id": "paired-feedback-test",
        "runner": [sys.executable, str(runner), "{output}"],
        "threads": 1,
        "problem_family": "cvrp",
        "performance_protocol": "heuristic_fixed_budget",
        "objective_sense": "minimize",
        "budgets_sec": [10.0],
        "development_targets": {
            "dev_0000.json": {
                "id": "one",
                "sha256": hashlib.sha256(instance.read_bytes()).hexdigest(),
                "target": 2,
            }
        },
    }
    monkeypatch.setenv("PITBENCH_REPO", str(tmp_path))
    monkeypatch.setenv("PITBENCH_RUNS", "/proc/pitbench-read-only")
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    assert agent_cli._run_one(config, instance, 7, 10.0, False, "base", "clean")
    config["candidate_experiment"] = {
        "hypothesis_id": "cache_probe",
        "experiment_id": "feature_off",
        "patch_sha256": "patch",
        "record_sha256": "a" * 64,
    }
    assert agent_cli._run_one(config, instance, 7, 10.0, False, "agent", "patch")
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records[0]["verification"] == "independent_public"
    assert records[0]["verified_valid"] is True
    assert records[0]["verified_objective"] == 2
    assert records[0]["preview_normalized_gap"] == 0
    assert records[1]["paired_base"] is not None
    assert (
        str(tmp_path / "pitbench-runs" / "paired-feedback-test") in records[0]["result"]
    )
    assert records[1]["verified_objective_reduction"] == 0
    assert records[1]["preview_gap_reduction"] == 0
    assert records[1]["candidate_experiment"] == config["candidate_experiment"]
    saved_feedback = Path(records[1]["result"]).with_suffix(".feedback.json")
    assert (
        json.loads(saved_feedback.read_text())["candidate_experiment"]
        == config["candidate_experiment"]
    )

    custom_instance = tmp_path / "custom.json"
    custom_instance.write_bytes(instance.read_bytes())
    assert agent_cli._run_one(config, custom_instance, 7, 10.0, False, "agent", "patch")
    custom_feedback = json.loads(capsys.readouterr().out)
    assert custom_feedback["paired_base"] == records[1]["paired_base"]


def test_proto_feedback_labels_unverified_exact_preview(tmp_path: Path) -> None:
    instance = tmp_path / "dev_0000.pb"
    instance.write_bytes(b"proto")
    output = tmp_path / "run.json"
    output.with_suffix(".solution.json").write_text("{}")
    config = {
        "problem_family": "cp",
        "performance_protocol": "exact_verified_solve",
        "exact_time_basis": "cpu_time",
    }
    feedback = agent_cli._feedback(
        config,
        {
            "valid": True,
            "objective": 10,
            "solver_termination": "optimal",
            "cpu_time_sec": 1.5,
        },
        10,
        20.0,
        instance,
        output,
    )
    assert feedback["verification"] == "unavailable"
    assert feedback["verified_valid"] is None
    assert feedback["preview_solved"] is True
    assert feedback["preview_par2_sec"] == 1.5

    config["exact_time_basis"] = "solve_wall_time"
    feedback = agent_cli._feedback(
        config,
        {"valid": False, "wall_time_sec": 0.3},
        10,
        20.0,
        instance,
        output,
    )
    assert feedback["preview_solved"] is False
    assert feedback["preview_par2_sec"] == 40.0

    config["agent_feedback_mode"] = "outcome_only"
    limited = agent_cli._feedback(
        config,
        {"valid": False, "wall_time_sec": 0.3},
        10,
        20.0,
        instance,
        output,
    )
    assert "solver_behavior" not in limited
    assert "behavior_comparison" not in agent_diagnostics.paired_feedback(
        {**limited, "performance_protocol": "exact_verified_solve"},
        {**limited, "performance_protocol": "exact_verified_solve"},
    )
