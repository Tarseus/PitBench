import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

import pytest
import yaml

from pitbench.evaluator.representations import LinearModelRepresentation
from pitbench.metrics.scenario_report import (
    ReportedBoundTrace,
    compute_scenario_report,
    sequence_key,
)
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


def test_reported_dual_integral_is_computed_without_claiming_certificates(tmp_path):
    data = campaign(tmp_path, "trajectory")
    for run in data.runs:
        for event in run.steps[0].events:
            event.row_multipliers = None
    original = compute_scenario_report(data)
    report = compute_scenario_report(
        data, dual_bound_source="solver_reported", reported_bound_traces=reported_traces(data)
    )
    values = {r.metric_id: r for r in report.results}
    assert report.complete and not original.complete
    assert values["dual_progress_integral"].base == 4
    assert values["dual_progress_integral"].agent == 2
    assert values["dual_progress_integral"].dual_bound_source == "solver_reported"
    assert report.design_sha256 != original.design_sha256
    assert report.protocol_hashes == original.protocol_hashes
    assert len(report.bound_trace_sources) == len(data.runs)


def test_missing_reported_bound_trace_does_not_become_a_zero_integral(tmp_path):
    data = campaign(tmp_path, "trajectory")
    traces = reported_traces(data)
    traces.pop(data.runs[0].key)
    values = {
        r.metric_id: r
        for r in compute_scenario_report(
            data, dual_bound_source="solver_reported", reported_bound_traces=traces
        ).results
    }
    assert not values["dual_progress_integral"].complete
    assert values["primal_integral"].complete


def test_observed_no_finite_bound_uses_declared_initial_loss(tmp_path):
    data = campaign(tmp_path, "trajectory")
    traces = reported_traces(data)
    for trace in traces.values():
        trace.events = []
        trace.observed_events = 1
        trace.unavailable_bound_events = 1
        trace.native_infinity = 1e20
    values = {
        r.metric_id: r
        for r in compute_scenario_report(
            data, dual_bound_source="solver_reported", reported_bound_traces=traces
        ).results
    }
    assert values["dual_progress_integral"].complete
    assert values["dual_progress_integral"].base == 10


def native_trace_files(output, data):
    from pitbench.evaluator.scenarios import digest

    paths = {}
    for run in data.runs:
        directory = output / "jobs" / digest(run.key.model_dump(mode="json")) / "attempt-0000"
        directory.mkdir(parents=True)
        (directory / "result.json").write_text(json.dumps({"run": run.model_dump(mode="json")}))
        path = directory / "native-progress.jsonl"
        rows = [
            {"step": 0, "time_sec": 0.1, "dual_bound": -1e20},
            {"step": 0, "time_sec": 1, "dual_bound": 0},
            {"step": 0, "time_sec": 2, "dual_bound": 1},
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        paths[run.key] = path
    return paths


def test_native_bound_loader_decodes_infinity_and_retains_trace_hashes(tmp_path):
    from pitbench.evaluator.scenarios import load_reported_bound_traces

    data = campaign(tmp_path, "trajectory")
    paths = native_trace_files(tmp_path, data)
    traces = load_reported_bound_traces(tmp_path, data, native_infinity=1e20)
    for key, trace in traces.items():
        assert trace.events == [(1, 0), (2, 1)]
        assert trace.unavailable_bound_events == 1
        assert trace.source_sha256 == hashlib.sha256(paths[key].read_bytes()).hexdigest()


def test_shared_bound_trace_is_checked_against_its_measurement_manifest(tmp_path):
    from pitbench.evaluator.scenarios import load_reported_bound_traces
    from pitbench.evaluator.storage import file_manifest

    data = campaign(tmp_path, "trajectory")
    paths = native_trace_files(tmp_path, data)
    run = data.runs[0]
    directory = paths[run.key].parent
    receipt = tmp_path / "measurement.json"
    receipt.write_text(json.dumps({
        "run": run.model_dump(mode="json"),
        "raw_artifacts": str(directory),
        "files": file_manifest(directory),
    }))
    run.measurement_source = {"receipt": str(receipt)}
    assert run.key in load_reported_bound_traces(tmp_path, data, native_infinity=1e20)
    paths[run.key].write_text(paths[run.key].read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_reported_bound_traces(tmp_path, data, native_infinity=1e20)


def test_native_bound_loader_rejects_out_of_interval_timestamps(tmp_path):
    from pitbench.evaluator.scenarios import load_reported_bound_traces

    data = campaign(tmp_path, "trajectory")
    paths = native_trace_files(tmp_path, data)
    paths[data.runs[0].key].write_text(json.dumps({"step": 0, "time_sec": 100, "dual_bound": 1}))
    with pytest.raises(ValueError, match="outside the measured solve"):
        load_reported_bound_traces(tmp_path, data)


def test_reported_bound_report_can_be_saved_outside_raw_worker_directory(tmp_path):
    from pitbench.evaluator.scenarios import report

    data = campaign(tmp_path, "trajectory")
    native_trace_files(tmp_path, data)
    (tmp_path / "campaign.json").write_text(data.model_dump_json())
    destination = tmp_path / "analysis/solver-reported.json"
    summary = report(
        tmp_path,
        dual_bound_source="solver_reported",
        reported_bound_infinity=1e20,
        report_path=destination,
    )
    assert summary["complete"]
    assert summary["report"] == str(destination)
    assert json.loads(destination.read_text())["dual_bound_source"] == "solver_reported"
    assert not (tmp_path / "scenario-report.json").exists()


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


def test_budget_capped_scaling_retains_censoring_and_attainment(tmp_path):
    data = campaign(tmp_path, "scaling", censoring="par2")
    hashes = {p.id: p.digest() for p in data.protocols}
    for run in data.runs:
        if run.key.threads == 2:
            step = run.steps[0]
            step.events = []
            step.termination = "time_limit"
            step.final_solution_values = None
    original = compute_scenario_report(data)
    assert not original.complete
    report = compute_scenario_report(data, scaling_time_policy="budget_capped_v1")
    assert report.complete
    assert report.design_sha256 != original.design_sha256
    assert report.protocol_hashes == hashes
    values = {r.metric_id: r for r in report.results}
    assert values["parallel_efficiency"].base == pytest.approx(0.6)
    assert values["parallel_efficiency"].agent == pytest.approx(0.55)
    assert values["thread_scaling_curve"].base == 7
    assert values["thread_scaling_curve"].agent == 6
    item = values["parallel_efficiency"]
    assert item.scaling_time_policy == "budget_capped_v1"
    assert item.target_attainment["base"]["2"] == {
        "expected_runs": 2,
        "observed_runs": 2,
        "reached_runs": 0,
        "censored_runs": 2,
        "attainment_rate": 0,
    }
    record = item.by_instance["a"]["records"]["base"][0]
    assert record["target_times"][2] == 10
    assert record["observed_target_times"][2] is None
    assert record["censored"][2]


def test_budget_capped_scaling_retains_late_hits_as_censored(tmp_path):
    data = campaign(tmp_path, "scaling")
    for run in data.runs:
        step = run.steps[0]
        step.solve_time_sec = 12
        step.events[-1].time_sec = 11
    report = compute_scenario_report(data, scaling_time_policy="budget_capped_v1")
    values = {r.metric_id: r for r in report.results}
    assert report.complete
    assert values["thread_scaling_curve"].base == 10
    assert values["thread_scaling_curve"].agent == 10
    assert values["parallel_efficiency"].base == pytest.approx(0.75)
    for item in values.values():
        # Final quality can pass even when the target was reached after the budget.
        assert item.quality_qualified
        assert all(
            cell["attainment_rate"] == 0
            for state in item.target_attainment.values()
            for cell in state.values()
        )


def test_budget_capped_scaling_does_not_fill_missing_runs(tmp_path):
    data = campaign(tmp_path, "scaling")
    data.runs.pop()
    report = compute_scenario_report(data, scaling_time_policy="budget_capped_v1")
    assert not report.complete
    assert report.missing_run_count == 1
    assert report.results[0].target_attainment["agent"]["2"]["attainment_rate"] is None


def test_zero_target_times_only_prevent_efficiency_not_mean_time(tmp_path):
    data = campaign(tmp_path, "scaling")
    for run in data.runs:
        for event in run.steps[0].events:
            event.time_sec = 0
    values = {
        r.metric_id: r
        for r in compute_scenario_report(
            data, scaling_time_policy="budget_capped_v1"
        ).results
    }
    assert values["thread_scaling_curve"].complete
    assert values["thread_scaling_curve"].base == 0
    assert not values["parallel_efficiency"].complete


def test_unknown_scaling_time_policy_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown scaling time policy"):
        compute_scenario_report(campaign(tmp_path, "scaling"), scaling_time_policy="typo")


def test_budget_capped_report_is_saved_separately(tmp_path):
    from pitbench.evaluator.scenarios import report

    data = campaign(tmp_path, "scaling")
    (tmp_path / "campaign.json").write_text(data.model_dump_json())
    report(tmp_path)
    original = (tmp_path / "scenario-report.json").read_bytes()
    result = report(tmp_path, scaling_time_policy="budget_capped_v1")
    assert result["scaling_time_policy"] == "budget_capped_v1"
    assert (tmp_path / "scenario-report.json").read_bytes() == original
    rescored = json.loads((tmp_path / "scenario-report-budget-capped-v1.json").read_text())
    assert rescored["scaling_time_policy"] == "budget_capped_v1"


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


def test_numerical_stress_checks_every_scaled_model(tmp_path):
    data = campaign(tmp_path, "numerical")
    p = data.protocols[0]
    path = tmp_path / "scaled.lp"
    path.write_text(
        "Minimize\n obj: x + y\nSubject To\n"
        " c: 1000000 x + 1000000 y >= 1000000\n"
        "Bounds\n 0 <= x <= 2\n 0 <= y <= 2\nEnd\n"
    )
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    for iid in ("a", "b"):
        data.model_sequences[sequence_key("judge_id", iid)].append(
            {"path": str(path), "sha256": sha}
        )
        p.model_sequences_sha256["judge_id"][iid].append(sha)
    for run in data.runs:
        run.protocol_sha256 = p.digest()
        run.steps.append(
            run.steps[0].model_copy(
                deep=True,
                update={"instance_path": str(path), "instance_sha256": sha},
            )
        )
    assert compute_scenario_report(data).complete

    data.runs[0].steps[1].final_solution_values = [0.999, 0]
    report = compute_scenario_report(data)
    values = {r.metric_id: r for r in report.results}
    assert len(report.invalid_runs) == 1
    assert values["independent_violation"].base == pytest.approx(500)
    assert values["numerical_failure_counts"].base == pytest.approx(0.5)
    records = values["independent_violation"].by_instance["a"]["records"]["base"]
    assert len(records[0]["residuals_by_step"]) == 2


def test_numerical_failures_count_each_stress_solve_without_fabricating_residuals(
    tmp_path,
):
    data = campaign(tmp_path, "numerical")
    p = data.protocols[0]
    for iid in ("a", "b"):
        sequence = data.model_sequences[sequence_key("judge_id", iid)]
        sequence.append(dict(sequence[0]))
        p.model_sequences_sha256["judge_id"][iid].append(sequence[0]["sha256"])
    for run in data.runs:
        run.protocol_sha256 = p.digest()
        run.steps.append(run.steps[0].model_copy(deep=True))
        for step in run.steps:
            step.termination = "solver_error"
            step.events = []
            step.final_solution_values = None
    values = {r.metric_id: r for r in compute_scenario_report(data).results}
    assert values["numerical_failure_counts"].complete
    assert values["numerical_failure_counts"].base == 2
    assert not values["independent_violation"].complete


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


@pytest.mark.parametrize("different_options", [False, True])
@pytest.mark.parametrize("source_failed", [False, True])
def test_setup_shares_one_measurement_only_with_matching_execution(
    tmp_path, monkeypatch, different_options, source_failed
):
    from pitbench.evaluator.scenarios import run

    data = campaign(tmp_path, "trajectory")
    setup = protocol("setup")
    setup.model_sequences_sha256 = data.protocols[0].model_sequences_sha256
    originals = list(data.expected_runs)
    data.protocols.append(setup)
    data.expected_runs += [
        key.model_copy(update={"scenario_id": "setup"}) for key in originals
    ]
    measured = {record.key: record for record in data.runs}
    data.runs = []
    for iid in ("a", "b"):
        for kind in ("trajectory", "setup"):
            data.model_sequences[sequence_key("judge_id", iid, kind)] = (
                data.model_sequences[sequence_key("judge_id", iid)]
            )
    output = tmp_path / "output"
    output.mkdir()
    (output / "campaign.json").write_text(data.model_dump_json())
    runtime = {
        "python": sys.executable,
        "options_by_scenario": {"setup": {"different": True}}
        if different_options
        else {},
    }
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

    def collect(command, directory, identity, **kwargs):
        key = ScenarioRunKey.model_validate(identity["scenario_run"])
        calls.append(key)
        source_key = key.model_copy(update={"scenario_id": "trajectory"})
        record = measured[source_key].model_copy(
            update={
                "key": key,
                "protocol_sha256": next(
                    p.digest() for p in data.protocols if p.id == key.scenario_id
                ),
            }
        )
        if source_failed and key.scenario_id == "trajectory":
            record = record.model_copy(
                update={
                    "status": "process_error",
                    "steps": [],
                    "error": "retained source failure",
                }
            )
        payload = {
            "execution_status": "completed",
            "run": record.model_dump(mode="json"),
        }
        (directory / "result.json").write_text(json.dumps(payload))
        return payload

    monkeypatch.setattr("pitbench.evaluator.scenarios.run_collection_process", collect)
    monkeypatch.setattr(
        "pitbench.evaluator.scenarios._hardware_identity",
        lambda cpus: data.hardware_identity,
    )
    summary = run(output)
    assert len(calls) == (8 if different_options else 4)
    assert summary["recorded_runs"] == summary["expected_runs"] == 8
    assert summary["reused_runs"] == (0 if different_options else 4)
    if source_failed and not different_options:
        saved = ScenarioCampaign.model_validate_json(
            (output / "campaign.json").read_text()
        )
        assert all(record.status == "process_error" for record in saved.runs)
    run(output)
    assert len(calls) == (8 if different_options else 4)


def test_scenario_base_cache_is_shared_across_candidates_but_agents_are_measured(
    tmp_path, monkeypatch
):
    from pitbench.evaluator.scenarios import run

    data = campaign(tmp_path, "trajectory")
    observed = {record.key: record for record in data.runs}
    data.runs = []
    for iid in ("a", "b"):
        data.model_sequences[sequence_key("judge_id", iid, "trajectory")] = (
            data.model_sequences[sequence_key("judge_id", iid)]
        )
    runtime = {"python": sys.executable}
    calls = []

    def collect(command, directory, identity, **kwargs):
        key = ScenarioRunKey.model_validate(identity["scenario_run"])
        calls.append(key.code_state)
        payload = {
            "execution_status": "completed",
            "run": observed[key].model_dump(mode="json"),
        }
        (directory / "result.json").write_text(json.dumps(payload))
        (directory / "trajectory.jsonl").write_text("raw incumbent trace\n")
        return payload

    monkeypatch.setattr("pitbench.evaluator.scenarios.run_collection_process", collect)
    monkeypatch.setattr(
        "pitbench.evaluator.scenarios._hardware_identity",
        lambda cpus: data.hardware_identity,
    )
    summaries = []
    old_receipts = {}
    for index in range(4):
        if index == 2:
            for ready in (tmp_path / "base-cache").glob("*/ready.json"):
                saved = json.loads(ready.read_text())
                receipt = Path(saved["receipt"])
                old_receipts[receipt] = receipt.read_text()
                (Path(saved["raw_artifacts"]) / "trajectory.jsonl").write_text(
                    "damaged"
                )
        if index == 3:
            runtime = {**runtime, "environment": {"CONTROLLED_SOLVER_FLAG": "changed"}}
        output = tmp_path / f"candidate-{index}"
        output.mkdir()
        selected = data.model_copy(
            deep=True,
            update={
                "candidate_sha256": str(index + 1) * 64,
                "collection_id": f"candidate-{index}",
            },
        )
        (output / "campaign.json").write_text(selected.model_dump_json())
        (output / "runtime.json").write_text(
            json.dumps(
                {
                    "runtimes": {"base": runtime, "agent": runtime},
                    "cpus": [min(os.sched_getaffinity(0))],
                    "watchdog_grace_sec": 1,
                    "base_cache_path": str(tmp_path / "base-cache"),
                }
            )
        )
        summaries.append(run(output))
    assert calls.count("base") == 6
    assert calls.count("agent") == 8
    assert summaries[0]["shared_base_runs"] == 0
    assert summaries[1]["shared_base_runs"] == 2
    assert summaries[2]["shared_base_runs"] == 0
    assert summaries[3]["shared_base_runs"] == 0
    assert all(path.read_text() == contents for path, contents in old_receipts.items())
    for receipt in (tmp_path / "base-cache").glob("*/ready.json"):
        saved = json.loads(receipt.read_text())
        assert (
            Path(saved["raw_artifacts"]) / "trajectory.jsonl"
        ).read_text() == "raw incumbent trace\n"


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


def _frozen_task_for_campaign(data):
    from pitbench.schema.task import PitBenchTask

    root = Path(__file__).resolve().parents[3]
    task = PitBenchTask.model_validate(yaml.safe_load((root / "configs/tasks/release_scip_v10_1_0.yaml").read_text()))
    panel = next(p for p in task.instance_sets if p.name == "judge_id").model_copy(update={"size": 2})
    return task.model_copy(update={"task_id": data.task_id, "metric_scenarios": data.protocols, "instance_sets": [panel]})


def test_scenario_completeness_requires_raw_campaign_even_when_flags_claim_success(tmp_path):
    from pitbench.evaluator.scenarios import scenario_evidence_failures

    task = _frozen_task_for_campaign(campaign(tmp_path, "scaling"))
    failures = scenario_evidence_failures(task, tmp_path, {
        "summary": {"metric_scenarios": {"complete": True}, "metric_scenario_error": "missing recipe"},
        "artifacts": {},
    })
    assert any("collection failed" in item for item in failures)
    assert any("campaign is missing" in item for item in failures)


@pytest.mark.parametrize("problem", [None, "hash", "candidate", "protocol", "inventory"])
def test_raw_scenario_evidence_is_bound_to_task_patch_and_artifact_hash(tmp_path, problem):
    from pitbench.evaluator.scenarios import scenario_evidence_failures, task_digest

    data = campaign(tmp_path, "scaling")
    task = _frozen_task_for_campaign(data)
    data.task_sha256 = task_digest(task)
    patch = tmp_path / "candidate.patch"
    patch.write_bytes(b"actual patch")
    data.candidate_sha256 = hashlib.sha256(patch.read_bytes()).hexdigest()
    if problem == "candidate":
        data.candidate_sha256 = "0" * 64
    elif problem == "protocol":
        data.protocols[0].budget_sec += 1
    elif problem == "inventory":
        data.instances_by_set["judge_id"] = ["a"]
    target = tmp_path / "campaign.json"
    target.write_text(data.model_dump_json())
    payload = {"artifacts": {"metric_scenario_campaign": {
        "path": "campaign.json", "sha256": "0" * 64 if problem == "hash" else hashlib.sha256(target.read_bytes()).hexdigest(),
    }}}
    failures = scenario_evidence_failures(task, tmp_path, payload)
    assert bool(failures) == (problem is not None)


def test_scenario_inputs_resolve_relocated_private_storage_and_reject_wrong_hashes(tmp_path):
    from pitbench.evaluator.private_assets import PrivateAssetError
    from pitbench.evaluator.scenarios import resolve_model_inputs

    data = campaign(tmp_path, "setup")
    task = _frozen_task_for_campaign(data)
    logical, storage = tmp_path / "private", tmp_path / "storage"
    logical.mkdir()
    storage.mkdir()
    (logical / ".asset-root").symlink_to(storage, target_is_directory=True)
    file = storage / "model.lp"
    file.write_bytes((tmp_path / "a.lp").read_bytes())
    cases = [{"id": iid, "models": [str(logical / "model.lp")]} for iid in ("a", "b")]
    config = {"instances": {"judge_id": cases}}
    result, mappings = resolve_model_inputs(config, task, private_root=logical, config_dir=tmp_path)
    assert result["instances"]["judge_id"][0]["models_by_scenario"]["setup"] == [str(file)]
    assert mappings[0]["sha256"] == hashlib.sha256(file.read_bytes()).hexdigest()
    assert "models_by_scenario" not in config["instances"]["judge_id"][0]
    file.write_text("different model")
    with pytest.raises((ValueError, PrivateAssetError), match="hash mismatch"):
        resolve_model_inputs(config, task, private_root=logical, config_dir=tmp_path)


def test_runtime_probe_rejects_missing_representation_dependencies_before_solver_probe(monkeypatch):
    from pitbench.evaluator.scenarios import main

    def missing(name):
        if name == "numpy":
            raise ModuleNotFoundError("No module named 'numpy'")
        pytest.fail("unexpected dependency import")

    monkeypatch.setattr("pitbench.evaluator.representations.importlib.import_module", missing)
    monkeypatch.setattr(sys, "argv", ["collect", "probe", "--session-plugin", "not.loaded:Session", "--representation", "row_column_permutation"])
    with pytest.raises(ModuleNotFoundError, match="numpy"):
        main()
