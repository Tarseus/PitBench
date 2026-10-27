import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

import pytest
import yaml

from pitbench.evaluator.representations import LinearModelRepresentation
from pitbench.metrics.scenario_report import compute_scenario_report, sequence_key
from pitbench.problem_families.verification import NumericModelFamily
from pitbench.schema.scenarios import (
    SCENARIO_METRICS,
    ScenarioCampaign,
    ScenarioEvent,
    ScenarioProtocol,
    ScenarioRun,
    ScenarioRunKey,
    ScenarioStep,
)


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


@pytest.mark.parametrize("kind", sorted(SCENARIO_METRICS))
def test_every_scenario_metric_has_a_complete_recomputed_result(tmp_path, kind):
    report = compute_scenario_report(campaign(tmp_path, kind))
    assert report.complete
    assert not report.invalid_runs
    assert {r.metric_id for r in report.results} == SCENARIO_METRICS[kind]
    assert all(r.complete and len(r.by_instance) == 2 for r in report.results)


def test_first_passage_and_step_integrals_use_checked_incumbents(tmp_path):
    report = compute_scenario_report(campaign(tmp_path, "trajectory"))
    values = {r.metric_id: r for r in report.results}
    assert values["time_to_first_verified_solution_sec"].base == 2
    assert values["time_to_target_sec"].base == 4
    assert values["time_to_target_sec"].agent == 2
    assert values["primal_integral"].base == pytest.approx(4)
    assert values["primal_integral"].agent == pytest.approx(2)
    assert values["dual_progress_integral"].base == pytest.approx(4)


def test_missing_bound_certificates_do_not_hide_verified_primal_metrics(tmp_path):
    data = campaign(tmp_path, "trajectory")
    for run in data.runs:
        for event in run.steps[0].events:
            event.row_multipliers = None
    values = {r.metric_id: r for r in compute_scenario_report(data).results}
    assert not values["dual_progress_integral"].complete
    assert values["primal_integral"].complete


def test_late_incumbents_are_censored_and_missing_penalty_is_explicit(tmp_path):
    data = campaign(
        tmp_path, "trajectory", metrics=["time_to_target_sec"], censoring="par2"
    )
    for run in data.runs:
        run.steps[0].solve_time_sec = 12
        run.steps[0].events[-1].time_sec = 11
    (item,) = compute_scenario_report(data).results
    assert item.base == item.agent == 20
    assert all(
        record["censored"]
        for value in item.by_instance.values()
        for state in value["records"].values()
        for record in state
    )


def test_retained_memory_is_post_warmup_current_rss_not_lifetime_peak(tmp_path):
    report = compute_scenario_report(campaign(tmp_path, "lifecycle", warmup_steps=1))
    values = {r.metric_id: r for r in report.results}
    assert values["memory_growth_per_solve"].base == 20
    assert values["memory_growth_per_solve"].agent == 10
    assert values["retained_memory_series"].base == 140


def test_scaling_divides_speedup_by_thread_ratio(tmp_path):
    values = {
        r.metric_id: r
        for r in compute_scenario_report(campaign(tmp_path, "scaling")).results
    }
    assert values["parallel_efficiency"].base == pytest.approx(1)
    assert values["parallel_efficiency"].agent == pytest.approx(1)
    assert values["thread_scaling_curve"].base == 3
    assert values["thread_scaling_curve"].agent == 1.5


def test_same_objective_different_assignments_obeys_declared_replay_fields(tmp_path):
    data = campaign(tmp_path, "replay")
    for run in data.runs:
        if run.key.code_state == "agent" and run.key.repeat == 1:
            run.steps[0].final_solution_values = [0, 1]
    (value,) = compute_scenario_report(data).results
    assert value.base == 1 and value.agent == 0


def test_invalid_solutions_are_retained_and_never_become_valid_evidence(tmp_path):
    data = campaign(tmp_path, "numerical")
    data.runs[0].steps[0].final_solution_values = [0.999, 0]
    report = compute_scenario_report(data)
    assert len(report.invalid_runs) == 1
    values = {r.metric_id: r for r in report.results}
    assert values["independent_violation"].base > 0
    assert values["numerical_failure_counts"].base > 0


@pytest.mark.parametrize(
    "change", ["missing", "process", "model", "timestamp", "cold_control"]
)
def test_malformed_or_missing_sequence_evidence_is_not_complete(tmp_path, change):
    data = campaign(tmp_path, "sequence")
    if change == "missing":
        data.runs.pop()
    elif change == "process":
        next(r for r in data.runs if r.key.mode == "warm").steps[1].process_id += 1
    elif change == "model":
        data.runs[0].steps[0].instance_sha256 = "f" * 64
    elif change == "timestamp":
        data.runs[0].steps[0].events[0].time_sec = 100
    else:
        data.runs = [r for r in data.runs if r.key.mode == "warm"]
    assert not compute_scenario_report(data).complete


def test_scenario_grid_cannot_omit_both_states_of_an_instance(tmp_path):
    data = campaign(tmp_path, "setup").model_dump(mode="json")
    data["expected_runs"] = [
        r for r in data["expected_runs"] if r["instance_id"] == "a"
    ]
    data["runs"] = [r for r in data["runs"] if r["key"]["instance_id"] == "a"]
    with pytest.raises(ValueError, match="expected grid"):
        ScenarioCampaign.model_validate(data)


def test_bound_certificate_is_independent_of_solver_search_status(tmp_path):
    data = campaign(tmp_path, "trajectory")
    path = Path(data.runs[0].steps[0].instance_path)
    model = LinearModelRepresentation.read_input(path)
    assert NumericModelFamily.certified_relaxation_bound(model, [1]) == pytest.approx(1)
    assert NumericModelFamily.certified_relaxation_bound(model, [float("nan")]) is None


def test_structured_cp_solutions_use_the_existing_family_verifier(tmp_path):
    data = campaign(tmp_path, "replay")
    p = data.protocols[0]
    p.representation = "item_relabeling"
    path = tmp_path / "packing.json"
    path.write_text(json.dumps({"capacity": 2, "weights": [1, 1]}))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    p.model_sequences_sha256 = {"judge_id": {iid: [sha] for iid in ("a", "b")}}
    for key in data.model_sequences:
        data.model_sequences[key] = [{"path": str(path), "sha256": sha}]
    for run in data.runs:
        run.protocol_sha256 = p.digest()
        step = run.steps[0]
        step.instance_path, step.instance_sha256 = str(path), sha
        step.final_solution_values = None
        step.final_solution = {"bins": [[0, 1]]}
        step.events = [ScenarioEvent(time_sec=0.1, solution=step.final_solution)]
    report = compute_scenario_report(data)
    assert report.complete and not report.invalid_runs
    assert report.results[0].base == report.results[0].agent == 1
    for run in data.runs:
        if run.key.code_state == "agent" and run.key.repeat == 1:
            run.steps[0].final_solution = {"bins": [[1, 0]]}
    assert compute_scenario_report(data).results[0].agent == 0


def test_process_timeouts_remain_terminal_until_explicit_retry(tmp_path, monkeypatch):
    from pitbench.evaluator.scenarios import run

    data = campaign(tmp_path, "setup")
    for iid in ("a", "b"):
        data.model_sequences[sequence_key("judge_id", iid, "setup")] = (
            data.model_sequences[sequence_key("judge_id", iid)]
        )
    output = tmp_path / "retained"
    output.mkdir()
    (output / "campaign.json").write_text(data.model_dump_json())
    runtime = {"python": sys.executable}
    (output / "runtime.json").write_text(
        json.dumps(
            {
                "runtimes": {"base": runtime, "agent": runtime},
                "cpus": [min(os.sched_getaffinity(0))],
                "watchdog_grace_sec": 1,
            }
        )
    )
    calls = []

    def timeout(command, directory, identity, **kwargs):
        calls.append(directory)
        return {"execution_status": "watchdog_timeout", **identity}

    monkeypatch.setattr("pitbench.evaluator.scenarios.run_collection_process", timeout)
    monkeypatch.setattr(
        "pitbench.evaluator.scenarios._hardware_identity",
        lambda cpus: data.hardware_identity,
    )
    assert not run(output)["complete"]
    assert len(calls) == 4
    assert not run(output)["complete"]
    assert len(calls) == 4
    run(output, retry_errors=True)
    assert len(calls) == 8
    assert all(path.is_dir() for path in calls)


@pytest.mark.external
def test_real_session_collects_all_seven_scenario_kinds_and_resumes(tmp_path):
    pytest.importorskip("highspy")
    from pitbench.evaluator.scenarios import prepare, run
    from pitbench.solver_drivers.run import HighsScenarioSession

    cpus = sorted(os.sched_getaffinity(0))[:2]
    if len(cpus) != 2:
        pytest.skip("the real scaling experiment requires two allocated CPUs")
    root = Path(__file__).resolve().parents[3]
    task = yaml.safe_load((root / "configs/tasks/highs_v1_15_1.yaml").read_text())
    for panel in task["instance_sets"]:
        if panel["name"] == "judge_id":
            panel["size"] = 2
    models = []
    for index in range(3):
        path = tmp_path / f"model-{index}.lp"
        path.write_text(
            f"Minimize\n obj: x + y\nSubject To\n c: x + y >= {1 + index / 4}\nBounds\n 0 <= x <= 2\n 0 <= y <= 2\nEnd\n"
        )
        models.append(str(path))
    task_path = tmp_path / "task.yaml"
    task_path.write_text(yaml.safe_dump(task))
    patch = tmp_path / "candidate.patch"
    patch.write_text("")
    binding = HighsScenarioSession.identity()
    runtime = dict(
        python=sys.executable,
        session_plugin="pitbench.solver_drivers.run:HighsScenarioSession",
        source_commit=task["release"]["base_commit"],
        patch_sha256=hashlib.sha256(b"").hexdigest(),
        expected_binding_sha256=binding["binding_sha256"],
        options={},
    )
    protocols = [protocol(kind, budget_sec=0.1) for kind in sorted(SCENARIO_METRICS)]
    for p in protocols:
        chosen = models if p.kind in {"lifecycle", "sequence"} else models[:1]
        p.model_sequences_sha256 = {
            "judge_id": {
                iid: [
                    hashlib.sha256(Path(name).read_bytes()).hexdigest()
                    for name in chosen
                ]
                for iid in ("a", "b")
            }
        }
        p.qualification_target_by_model_sha256 = {
            hashlib.sha256(Path(name).read_bytes()).hexdigest(): 1
            + models.index(name) / 4
            for name in chosen
        }
    cases = [
        {
            "id": iid,
            "models_by_scenario": {
                p.id: models if p.kind in {"lifecycle", "sequence"} else models[:1]
                for p in protocols
            },
        }
        for iid in ["a", "b"]
    ]
    config = dict(
        task_config=str(task_path),
        candidate_patch=str(patch),
        cpus=cpus,
        watchdog_grace_sec=30,
        order="base_first",
        runtimes={"base": runtime, "agent": runtime},
        instances={"judge_id": cases},
        protocols=[p.model_dump(mode="json") for p in protocols],
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    output = tmp_path / "collected"
    prepared = prepare(config_path, output)
    assert prepared["jobs"] == 40
    result = run(output)
    assert result["complete"] and result["invalid_runs"] == 0
    report = json.loads((output / "scenario-report.json").read_text())
    assert len(report["results"]) == 15
    attempts = sorted(output.glob("jobs/*/attempt-*"))
    assert len(attempts) == 40
    assert run(output)["complete"]
    assert sorted(output.glob("jobs/*/attempt-*")) == attempts
    data = ScenarioCampaign.model_validate_json((output / "campaign.json").read_text())
    lifecycle = [r for r in data.runs if r.key.scenario_id == "lifecycle"]
    assert all(
        len({s.process_id for s in r.steps}) == 1 and len(r.steps) == 3
        for r in lifecycle
    )
    assert all(
        step.memory_scope == "binding_worker_after_record_flush"
        for record in lifecycle
        for step in record.steps
    )
    stale_patch = tmp_path / "stale-candidate.patch"
    stale_patch.write_text("a distinct source change")
    stale_config = {
        **config,
        "candidate_patch": str(stale_patch),
        "runtimes": {
            "base": runtime,
            "agent": {
                **runtime,
                "patch_sha256": hashlib.sha256(stale_patch.read_bytes()).hexdigest(),
            },
        },
    }
    stale_file = tmp_path / "stale-config.yaml"
    stale_file.write_text(yaml.safe_dump(stale_config))
    with pytest.raises(ValueError, match="no distinct built API artifact"):
        prepare(stale_file, tmp_path / "stale-agent")
