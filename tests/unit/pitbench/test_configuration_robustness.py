from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pitbench.evaluator.configuration import (
    ConfigurationRunner,
    SmacSearch,
    run_search,
    validate_search,
)
from pitbench.metrics.configuration_report import paired_panel, run_feedback

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


def test_fixed_search_count_full_panels_and_retest_without_reselection(
    search, tmp_path
):
    calls = []
    engine = SearchStub(search, tmp_path)

    def evaluate(name, parameters, seeds):
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


def test_failed_configuration_stops_without_numeric_feedback_and_is_retested(
    search, tmp_path
):
    calls = []
    engine = SearchStub(search, tmp_path)

    def evaluate(name, parameters, seeds):
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

    def evaluate(name, parameters, seeds):
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

    def execute(command, **kwargs):
        job_path = Path(command[command.index("--job") + 1])
        job = json.loads(job_path.read_text())
        invocations.append(job_path)
        job_path.with_name("result.json").write_text(
            json.dumps({**job, "execution_status": "collector_error"})
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("pitbench.evaluator.configuration.subprocess.run", execute)
    runner = ConfigurationRunner(tmp_path, [cpu])
    runner.panel(search, "default", {"amount": 0}, [1, 2])
    runner.panel(search, "default", {"amount": 0}, [1, 2])
    assert len(invocations) == 1  # The other planned slot is retained but not run.
    retry = ConfigurationRunner(tmp_path, [cpu], retry_collection_errors=True)
    retry.panel(search, "default", {"amount": 0}, [1, 2])
    assert len(invocations) == 2 and all(path.exists() for path in invocations)


def test_resume_collection_fault_keeps_the_same_smac_candidate(search, tmp_path):
    pytest.importorskip("smac")
    search["max_configurations"] = 1
    healthy = False
    candidate_parameters = []

    def evaluate(name, parameters, seeds):
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
