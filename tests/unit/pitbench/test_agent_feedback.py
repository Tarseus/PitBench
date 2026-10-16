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

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TOOLS = frozenset({AgentTool.INSPECT, AgentTool.WORKSPACE, AgentTool.BENCH})


def test_default_tools_and_explicit_repository_only_condition() -> None:
    assert set(EvaluationConfig().agent_tools) == DEFAULT_TOOLS
    assert EvaluationConfig.model_validate({"agent_tools": []}).agent_tools == []


@pytest.mark.parametrize(
    ("task_id", "metric"),
    [
        ("vroom_v1_15_0", "paired reduction in normalized gap"),
        ("pyvrp_v0_14_0", "paired reduction in normalized gap"),
        ("choco_v6_0_1", "independently verified solved coverage"),
        ("highs_v1_15_1", "independently verified solved coverage"),
        ("ortools_cp_sat_solve_v9_15", "independently verified solved coverage"),
    ],
)
def test_instruction_matches_protocol_and_visible_files(
    task_id: str, metric: str
) -> None:
    task = PitBenchTask.from_yaml(ROOT / "configs" / "tasks" / f"{task_id}.yaml")
    instruction = PitBenchAdapter._format_instruction(task, DEFAULT_TOOLS)
    assert metric in instruction
    assert f"{task.evaluation.primary_budget_sec:.1f}s" in instruction
    assert "/pitbench/dev_instances" in instruction
    assert "pitbench bench --label base" in instruction
    assert "pitbench bench --label agent --build" in instruction
    assert "pitbench diagnose" in instruction
    assert "incumbent or bound progress" in instruction
    if task.task_type.value == "exact_solver":
        assert "paired PAR-2 runtime" in instruction
        assert "normalized gap" not in instruction

    without_tools = PitBenchAdapter._format_instruction(task)
    assert "No PitBench development instances are mounted" in without_tools
    assert "pitbench bench" not in without_tools


@pytest.mark.parametrize(
    "task_id",
    [
        "vroom_v1_15_0",
        "pyvrp_v0_14_0",
        "choco_v6_0_1",
        "ortools_cp_sat_solve_v9_15",
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
        diagnosis["observed_pattern"]
        == "favorable_effect_concentrated_in_one_instance"
    )
    assert diagnosis["runtime"]["mean_wall_time_reduction_sec"] == pytest.approx(
        0.1
    )


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
            f"dev_{index:04d}.json": {"sha256": f"broad-{index}"}
            for index in range(2)
        },
        "development_seeds": [0, 1],
        "threads": 1,
    }

    diagnosis = agent_cli._development_diagnosis(config, run_root, [10.0])

    assert diagnosis["observed_pattern"] == "favorable_effect_across_multiple_instances"
    assert diagnosis["missing_declared_runs"] == 2
    assert any(
        "--all-seeds" in action
        for action in diagnosis["recommended_next_actions"]
    )

    for path in (run_root / "agent").glob("*.feedback.json"):
        unchanged = json.loads(path.read_text())
        unchanged["preview_normalized_gap"] = 0.2
        path.write_text(json.dumps(unchanged))
    no_signal = agent_cli._development_diagnosis(config, run_root, [10.0])
    assert no_signal["observed_pattern"] == "no_detected_solver_outcome_change"
    assert not any(
        "--all-seeds" in action
        for action in no_signal["recommended_next_actions"]
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
    base = _feedback_record(
        instance_id="dev_0000", instance_sha256="one", gap=0.2
    )
    agent = _feedback_record(
        instance_id="dev_0000", instance_sha256="one", gap=0.1
    )
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
    assert agent_cli._run_one(config, instance, 7, 10.0, False, "agent", "patch")
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records[0]["verification"] == "independent_public"
    assert records[0]["verified_valid"] is True
    assert records[0]["verified_objective"] == 2
    assert records[0]["preview_normalized_gap"] == 0
    assert records[1]["paired_base"] is not None
    assert (
        str(tmp_path / "pitbench-runs" / "paired-feedback-test")
        in records[0]["result"]
    )
    assert records[1]["verified_objective_reduction"] == 0
    assert records[1]["preview_gap_reduction"] == 0

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
