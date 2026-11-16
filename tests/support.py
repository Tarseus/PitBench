"""Shared test roots and independently constructed scenario/transfer samples."""

from __future__ import annotations

import hashlib
import itertools
from pathlib import Path

from pitbench.metrics.scenario_report import ReportedBoundTrace, sequence_key
from pitbench.schema.observation import (
    CodeState,
    ExpectedRun,
    ExpectedRunGrid,
    RunObservation,
    RunStatus,
)
from pitbench.schema.scenarios import (
    SCENARIO_METRICS,
    ScenarioCampaign,
    ScenarioEvent,
    ScenarioProtocol,
    ScenarioRun,
    ScenarioRunKey,
    ScenarioStep,
)
from pitbench.schema.task import PitBenchTask

ROOT = Path(__file__).resolve().parents[1]


def protocol(kind, **changes):
    fields = dict(
        id=kind,
        kind=kind,
        metrics=sorted(SCENARIO_METRICS[kind]),
        instance_sets=["judge_id"],
        objective_sense="minimize",
        budget_sec=10,
        solver_seeds=[1],
        thread_counts=[1, 2] if kind == "scaling" else [1],
        repeats=2 if kind == "replay" else 1,
        modes=["cold", "warm"] if kind == "sequence" else ["warm"],
        target_by_instance={"a": 1, "b": 1},
        reference_by_instance={"a": 1, "b": 1},
        reference_solution_by_instance={"a": [1, 0], "b": [1, 0]},
        loss_scale_by_instance={"a": 1, "b": 1},
        missing_initial_loss=1,
        censoring="budget",
        reference_threads=1,
        warmup_steps=0,
        series_reduction="maximum",
        curve_reduction="mean_target_time",
        growth_estimator="ols",
        sequence_mode="warm",
        replay_fields=["objective", "termination", "solution_hash"],
        replay_absolute_tolerance=0,
        replay_relative_tolerance=0,
        memory_sample_interval_sec=0.005,
        feasibility_tolerance=1e-6,
        reset_policy="retain_state",
        memory_cleanup="none",
    )
    fields.update(changes)
    return ScenarioProtocol(**fields)


def campaign(tmp_path, kind, **changes):
    p = protocol(kind, **changes)
    expected, runs, sequences = [], [], {}
    for iid in ("a", "b"):
        path = tmp_path / (iid + ".lp")
        path.write_text(
            "Minimize\n obj: x + y\nSubject To\n c: x + y >= 1\nBounds\n 0 <= x <= 2\n 0 <= y <= 2\nEnd\n"
        )
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        count = 3 if kind in {"lifecycle", "sequence"} else 1
        sequences[sequence_key("judge_id", iid)] = [
            {"path": str(path), "sha256": sha}
        ] * count
        for state, seed, threads, repeat, mode in itertools.product(
            ("base", "agent"),
            p.solver_seeds,
            p.thread_counts,
            range(p.repeats),
            p.modes,
        ):
            key = ScenarioRunKey(
                scenario_id=p.id,
                task_id="scenario-test",
                instance_set="judge_id",
                instance_id=iid,
                code_state=state,
                solver_seed=seed,
                threads=threads,
                repeat=repeat,
                mode=mode,
            )
            expected.append(key)
            speed = 1 if state == "base" else 2
            steps = [
                ScenarioStep(
                    instance_path=str(path),
                    instance_sha256=sha,
                    process_id=100,
                    process_boot_id="boot",
                    setup_time_sec=1 / speed,
                    setup_peak_rss_bytes=400 // speed,
                    solve_time_sec=5 / speed,
                    update_time_sec=0 if index == 0 else 2 / speed,
                    retained_rss_bytes=100 + index * (20 if state == "base" else 10),
                    termination="optimal",
                    final_solution_values=[1, 0],
                    events=[
                        ScenarioEvent(
                            time_sec=2 / speed / threads,
                            solution_values=[2, 0],
                            row_multipliers=[0],
                        ),
                        ScenarioEvent(
                            time_sec=4 / speed / threads,
                            solution_values=[1, 0],
                            row_multipliers=[1],
                        ),
                    ],
                )
                for index in range(count)
            ]
            runs.append(
                ScenarioRun(
                    key=key,
                    protocol_sha256=p.digest(),
                    runtime_identity_sha256="a" * 64,
                    status="returned",
                    steps=steps,
                )
            )
    p.model_sequences_sha256 = {
        "judge_id": {
            iid: [m["sha256"] for m in sequences[sequence_key("judge_id", iid)]]
            for iid in ("a", "b")
        }
    }
    p.qualification_target_by_model_sha256 = {
        m["sha256"]: 1 for sequence in sequences.values() for m in sequence
    }
    for run in runs:
        run.protocol_sha256 = p.digest()
    return ScenarioCampaign(
        task_id="scenario-test",
        task_sha256="c" * 64,
        candidate_sha256="b" * 64,
        protocols=[p],
        expected_runs=expected,
        runs=runs,
        model_sequences=sequences,
        instances_by_set={"judge_id": ["a", "b"]},
        runtime_identities={"base": "a" * 64, "agent": "a" * 64},
        hardware_identity={"host": "fixture", "cpu_pool": [0, 1]},
        order="base_first",
        started_at=1,
        finished_at=2,
    )


def reported_traces(data):
    return {
        run.key: ReportedBoundTrace(
            events=[(2 / speed, 0), (4 / speed, 1)],
            source_path="native-progress.jsonl",
            source_sha256="d" * 64,
            observed_events=2,
        )
        for run in data.runs
        for speed in (1 if run.key.code_state == "base" else 2,)
    }


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
        instance_set_kind=("judge_id" if instance_set == "judge_id" else "judge_shift"),
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
