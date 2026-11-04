"""Unit tests for the Global Planner and Hierarchical Algorithmic Transition System (HATS)."""

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from global_planner.adapters.codex import (
    CodexPlannerAdapter,
)
from global_planner.bootstrap import bootstrap_hats_model
from global_planner.controller import PlanningController
from global_planner.driver import PlanningSessionDriver, build_autonomous_worker
from global_planner.schema import (
    ContextPacket,
    DecisionType,
    ExecutionStatus,
    HATSLevel,
    HATSNode,
    MechanismEffect,
    Opportunity,
    PerformanceDelta,
    ReplanningDecision,
    SessionStatus,
    SolutionValidity,
)
from global_planner.store import PlanStore, match_source_ref
from pitbench.schema.research import PlanningStudyPlan


@pytest.mark.parametrize(
    "case,expected_exit,expected_status",
    [
        ("complete", 0, SessionStatus.COMPLETED),
        ("timeout_after_bench", 1, SessionStatus.FAILED),
        ("error_after_bench", 1, SessionStatus.FAILED),
        ("timeout_before_bench", 1, SessionStatus.FAILED),
        ("no_bench", 2, SessionStatus.INCOMPLETE),
        ("partial", 2, SessionStatus.INCOMPLETE),
        ("partial_then_complete", 2, SessionStatus.INCOMPLETE),
        ("planner_failure_after_bench", 1, SessionStatus.FAILED),
    ],
)
def test_cli_agent_boundaries_preserve_evidence_and_report_session_status(
    tmp_path, monkeypatch, capsys, case, expected_exit, expected_status
):
    from global_planner import cli

    plan_dir = tmp_path / "plan"
    task_path = tmp_path / "task.yaml"
    task_path.write_text(json.dumps({"task_id": "cli_task"}))
    instructions = tmp_path / "task.md"
    instructions.write_text("Improve the solver on public development instances.")
    monkeypatch.delenv("PITBENCH_GLOBAL_PLAN_CONTROLLED_SESSION", raising=False)

    def fake_agent(prompt):
        assert "pitbench bench" in prompt
        child_store = PlanStore(plan_dir)
        if case == "partial_then_complete":
            PlanningController(child_store).on_experiment_completed(
                experiment_id="first_partial_exp", target_nodes=[],
                metrics_summary={"runs_count": 1, "expected_runs_count": 2},
                execution_status=ExecutionStatus.PASS,
                solution_validity=SolutionValidity.VALID,
                performance_delta=PerformanceDelta.IMPROVED,
            )
        if case not in ("no_bench", "timeout_before_bench"):
            PlanningController(child_store).on_experiment_completed(
                experiment_id="child_exp",
                target_nodes=["subsystem.search_engine"],
                metrics_summary={
                    "runs_count": 1 if case == "partial" else 2,
                    "expected_runs_count": 2,
                    "raw_measurement": [10.0],
                },
                execution_status=ExecutionStatus.PASS,
                solution_validity=SolutionValidity.VALID,
                performance_delta=PerformanceDelta.IMPROVED,
                hypothesis_supported=True,
            )
        child_store.record_event("child.observation", {"case": case})
        if case.startswith("timeout"):
            raise subprocess.TimeoutExpired(["fake-agent"], 10)
        if case == "error_after_bench":
            raise RuntimeError("agent connection closed after bench")
        return "{}"

    planner_calls = 0

    def fake_planner(prompt):
        nonlocal planner_calls
        planner_calls += 1
        if planner_calls > 1:
            raise RuntimeError("planner failed after benchmark")
        return json.dumps({
            "decision": "continue", "selected_opportunity": "opp_profile_bottlenecks",
            "next_experiment": "pitbench bench",
        })

    def fake_factory(**kwargs):
        if kwargs.get("backend") == "codex":
            return fake_agent
        return fake_planner if case == "planner_failure_after_bench" else None

    monkeypatch.setattr(cli, "create_default_llm_callable", fake_factory)
    result = cli.main([
        "run", "--dir", str(plan_dir), "--task-config", str(task_path),
        "--task", str(instructions), "--repository", str(tmp_path),
        "--agent", "codex", "--planner-backend", "litellm",
        "--allow-offline-fallback", "--max-steps", "1",
    ])
    state = PlanStore(plan_dir).state
    assert result == expected_exit
    assert state.session_status == expected_status
    output = capsys.readouterr()
    assert f"Session {expected_status.value}:" in output.out + output.err
    assert state.events[-1]["payload"]["status"] == expected_status.value
    assert state.events[-1]["payload"]["executed_steps"] == 1
    assert any(event["event_type"] == "child.observation" for event in state.events)
    assert state.experiments_count == len(state.evidence) == (2 if case == "partial_then_complete" else 1)
    assert "PITBENCH_GLOBAL_PLAN_CONTROLLED_SESSION" not in cli.os.environ
    evidence = state.evidence[-1]
    if case not in ("no_bench", "timeout_before_bench"):
        assert evidence.experiment_id == "child_exp"
        assert evidence.metrics_summary["raw_measurement"] == [10.0]
    if case in ("complete", "timeout_after_bench", "error_after_bench", "planner_failure_after_bench", "partial_then_complete"):
        assert evidence.execution_status == ExecutionStatus.PASS
        assert evidence.performance_delta == PerformanceDelta.IMPROVED
    else:
        assert evidence.execution_status == ExecutionStatus.MISSING
        assert evidence.solution_validity == SolutionValidity.UNVERIFIED
        assert evidence.performance_delta == PerformanceDelta.UNKNOWN
    errors = [event for event in state.events if event["event_type"] == "agent.turn_failed"]
    assert len(errors) == (1 if case.startswith("timeout") or case == "error_after_bench" else 0)


@pytest.mark.parametrize("reported_actual", [None, 1, 2])
def test_recorded_partial_evidence_is_corrected_before_replanning(
    tmp_path, monkeypatch, reported_actual
):
    store = PlanStore(tmp_path / "plan")
    driver = PlanningSessionDriver(store, allow_offline_fallback=True, cooldown_sec=0)
    planner_inputs = []
    original_request = driver.adapter.request_decision

    def observe_decision(packet, **kwargs):
        planner_inputs.append(packet.model_dump(mode="json"))
        return original_request(packet, **kwargs)

    monkeypatch.setattr(driver.adapter, "request_decision", observe_decision)

    def legacy_worker(decision, packet):
        child = PlanStore(store.storage_dir)
        evidence, _ = PlanningController(child).on_experiment_completed(
            experiment_id="legacy_partial_exp", target_nodes=[],
            metrics_summary={"runs_count": 2, "expected_runs_count": 2},
            execution_status=ExecutionStatus.PASS,
            solution_validity=SolutionValidity.VALID,
            performance_delta=PerformanceDelta.IMPROVED,
            hypothesis_supported=True,
        )
        # Reproduce a record saved by the old writer: favorable statuses with a
        # partial run count. Keep a revision snapshot of that original record.
        evidence.metrics_summary["runs_count"] = 1
        child.save()
        result = {
            "experiment_id": evidence.experiment_id,
            "evidence_recorded": True,
            "execution_status": "pass", "solution_validity": "valid",
            "performance_delta": "improved", "hypothesis_supported": True,
        }
        if reported_actual is not None:
            result.update(actual_runs_count=reported_actual, expected_runs_count=2)
        return result

    driver.run_session({"task_id": "partial_task"}, legacy_worker, max_steps=1)
    state = PlanStore(store.storage_dir).state
    assert state.experiments_count == len(state.evidence) == 1
    evidence = state.evidence[0]
    assert evidence.execution_status == ExecutionStatus.MISSING
    assert evidence.solution_validity == SolutionValidity.UNVERIFIED
    assert evidence.performance_delta == PerformanceDelta.UNKNOWN
    assert evidence.hypothesis_supported is None
    assert evidence.status == "inconclusive"
    assert state.session_status == SessionStatus.INCOMPLETE
    correction = [event for event in state.events if event["event_type"] == "evidence.corrected"]
    assert len(correction) == 1
    assert correction[0]["payload"]["before"]["execution_status"] == "pass"
    assert correction[0]["payload"]["after"]["execution_status"] == "missing"
    assert correction[0]["payload"]["before"]["experiment_id"] == evidence.experiment_id
    assert planner_inputs[-1]["recent_evidence"][0]["execution_status"] == "missing"
    assert all(opp.status != "validated" for opp in state.opportunities.values())
    event_count = len(state.events)
    store.correct_evidence_completeness(evidence.id)
    assert len(store.state.events) == event_count


def test_schema_hats_node():
    node = HATSNode(
        id="heuristics.oneopt.knapsack_exchange",
        level=HATSLevel.LEVEL_2_MECHANISM,
        purpose="Enumerate binary exchanges on tight knapsack constraints",
        source_refs=["src/scip/heur_oneopt.c"],
        reads=["conshdlr", "solvals", "capacity"],
        writes=["bestsol"],
        preconditions=["nvars <= 128", "all_binary", "conshdlr == 'knapsack'"],
        invariants=["dual_bound_preserved", "feasibility_verified"],
        effects=[
            MechanismEffect(
                target_metric="time_to_feasible",
                predicted_direction="decrease",
                status="hypothesis",
            )
        ],
    )
    assert node.level == HATSLevel.LEVEL_2_MECHANISM
    assert len(node.preconditions) == 3
    assert node.effects[0].target_metric == "time_to_feasible"


def test_plan_store_persistence_and_invalid_opportunity_check(tmp_path: Path):
    store = PlanStore(tmp_path)
    state = store.initialize(study_id="study_01", task_id="scip_v10", solver="scip")
    assert state.revision == 1

    node = HATSNode(
        id="search.branching",
        level=HATSLevel.LEVEL_1_SUBSYSTEM,
        purpose="Variable branching rule",
        source_refs=["src/scip/branch.c"],
    )
    store.register_node(node)
    assert store.state.revision == 2
    assert "search.branching" in store.state.nodes

    # Check history snapshot
    history_files = list((tmp_path / "history").glob("*.json"))
    assert len(history_files) == 2

    # Verification: registering a decision with a non-existent opportunity MUST fail!
    invalid_decision = ReplanningDecision(
        decision=DecisionType.CONTINUE,
        selected_opportunity="non_existent_opp",
        next_experiment="pitbench bench",
    )
    with pytest.raises(ValueError, match="does not exist in state opportunities"):
        store.record_decision(invalid_decision, current_budget_spent=10.0)

    # Valid opportunity works
    store.register_opportunity(
        Opportunity(
            id="valid_opp",
            title="Valid opp",
            target_nodes=["search.branching"],
            proposal="Test",
            expected_effect="Test",
            cheapest_discriminating_experiment="bench",
        )
    )
    valid_decision = ReplanningDecision(
        decision=DecisionType.CONTINUE,
        selected_opportunity="valid_opp",
        next_experiment="pitbench bench",
    )
    store.record_decision(valid_decision, current_budget_spent=10.0)
    assert store.state.active_opportunity_id == "valid_opp"

    # Reload in a new store instance
    new_store = PlanStore(tmp_path)
    assert new_store.state.solver == "scip"
    assert "search.branching" in new_store.state.nodes


def test_event_log_persists_across_store_restarts(tmp_path: Path):
    store = PlanStore(tmp_path)
    store.initialize(study_id="study_events", task_id="task_events", solver="solver")
    store.record_event("hook.session_started", {"source": "adapter"})

    resumed = PlanStore(tmp_path)
    assert resumed.state.events[0]["event_type"] == "hook.session_started"
    assert resumed.state.events[0]["payload"] == {"source": "adapter"}


def test_create_default_callable_supports_claude_code_adapter(monkeypatch):
    from global_planner.adapters.codex import create_default_llm_callable

    commands = []

    class Result:
        returncode = 0
        stdout = "agent output"
        stderr = ""

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/claude")
    monkeypatch.setattr(
        "subprocess.run",
        lambda cmd, **kwargs: (commands.append((cmd, kwargs)) or Result()),
    )
    call = create_default_llm_callable(backend="claude-code", model_name="sonnet")
    assert call is not None
    assert call("do work") == "agent output"
    assert commands[0][0][:4] == [
        "/usr/bin/claude", "--print", "--dangerously-skip-permissions", "--model"
    ]


def test_recording_same_evidence_id_twice_is_rejected(tmp_path: Path):
    store = PlanStore(tmp_path)
    store.initialize(study_id="study_duplicate", task_id="task_duplicate", solver="solver")
    controller = PlanningController(store=store)
    controller.on_experiment_completed(
        experiment_id="exp_duplicate",
        target_nodes=[],
        metrics_summary={"runs_count": 1},
        execution_status=ExecutionStatus.MISSING,
        solution_validity=SolutionValidity.UNVERIFIED,
        performance_delta=PerformanceDelta.UNKNOWN,
    )
    with pytest.raises(ValueError, match="already been recorded"):
        store.record_evidence(store.state.evidence[0])


def test_session_driver_preserves_evidence_written_by_external_bench_process(tmp_path: Path):
    store = PlanStore(tmp_path)
    store.initialize(study_id="study_external", task_id="task_external", solver="solver")
    store.register_opportunity(
        Opportunity(
            id="opp_external",
            title="External run",
            target_nodes=["search.external"],
            proposal="Run the actual benchmark",
            expected_effect="Record measured evidence",
            cheapest_discriminating_experiment="pitbench bench",
        )
    )
    controller = PlanningController(store)
    controller.parse_and_record_decision(
        {
            "decision": "continue",
            "selected_opportunity": "opp_external",
            "next_experiment": "pitbench bench",
        },
        current_budget_spent=0,
    )
    driver = PlanningSessionDriver(store=store, allow_offline_fallback=True)

    def external_worker(decision, packet):
        external = PlanStore(tmp_path)
        PlanningController(external).on_experiment_completed(
            experiment_id="exp_external",
            target_nodes=["search.external"],
            metrics_summary={"runs_count": 2, "expected_runs_count": 2},
            execution_status=ExecutionStatus.PASS,
            solution_validity=SolutionValidity.VALID,
            performance_delta=PerformanceDelta.IMPROVED,
            hypothesis_supported=True,
        )
        return {
            "experiment_id": "exp_external",
            "execution_status": ExecutionStatus.PASS,
            "solution_validity": SolutionValidity.VALID,
            "performance_delta": PerformanceDelta.IMPROVED,
            "actual_runs_count": 2,
            "expected_runs_count": 2,
            "evidence_recorded": True,
        }

    driver.run_session(
        task_config={"task_id": "task_external", "release": {"repository": "solver"}},
        agent_worker=external_worker,
        max_steps=1,
    )
    persisted = PlanStore(tmp_path).state
    assert [e.experiment_id for e in persisted.evidence] == ["exp_external"]
    assert persisted.experiments_count == 1


def test_path_matching_glob_and_containment():
    """Verifies glob matching, prefix containment, and false positive prevention."""
    # 1. Wildcard / glob matching in reference
    assert match_source_ref("src/scip/heur_oneopt.c", "src/scip/heur_*.c")
    assert match_source_ref("src/scip/heur_rens.c", "src/scip/heur_*.c")
    assert not match_source_ref("src/scip/branch_relpscost.c", "src/scip/heur_*.c")

    # 2. Directory prefix containment
    assert match_source_ref("pyvrp/search/LocalSearch.cpp", "pyvrp/search")
    assert match_source_ref("pyvrp/search/sub/operator.cpp", "pyvrp/search")

    # 3. Exact match
    assert match_source_ref("src/main.c", "src/main.c")

    # 4. Prevention of unintended endswith match
    assert not match_source_ref("vendor/other/src/a.c", "src/a.c")
    assert not match_source_ref("external/pyvrp/search/LocalSearch.cpp", "pyvrp/search")


def test_budget_preservation_on_resume(tmp_path: Path):
    """Verifies cumulative budget is preserved across session re-instantiation."""
    store = PlanStore(tmp_path)
    driver1 = PlanningSessionDriver(store=store, allow_offline_fallback=True)
    task_config = {
        "task_id": "test_budget_task",
        "release": {"repository": "scipopt/scip"},
        "repository": {"editable_paths": ["src"]},
    }
    driver1.bootstrap(task_config=task_config)
    # Simulate spending 150 seconds and recording a decision
    dec = ReplanningDecision(
        decision=DecisionType.CONTINUE,
        selected_opportunity=driver1.store.state.active_opportunity_id or "opp_profile_bottlenecks",
        next_experiment="bench",
    )
    driver1.controller.parse_and_record_decision(dec.model_dump(), current_budget_spent=150.0)
    assert store.state.budget_spent_sec == 150.0

    # Resume with new driver instance
    driver2 = PlanningSessionDriver(store=PlanStore(tmp_path), allow_offline_fallback=True)
    assert driver2.get_current_budget_spent() >= 150.0


def test_bootstrap_context_packet_contains_subsystems(tmp_path: Path):
    """Verifies that an initial bootstrap packet contains global Level 0 & Level 1 nodes (not empty)."""
    store = PlanStore(tmp_path)
    controller = PlanningController(store=store)
    task_config = {
        "task_id": "test_bootstrap_nodes",
        "release": {"repository": "scipopt/scip"},
        "repository": {"editable_paths": ["src"]},
    }
    goal, nodes, opps = bootstrap_hats_model(task_config)
    controller.on_task_start(
        study_id="study_test",
        task_id="test_bootstrap_nodes",
        solver="scip",
        goal_statement=goal,
        initial_nodes=nodes,
        initial_opportunities=opps,
    )

    packet = controller.build_context_packet(current_budget_spent=0.0)
    assert len(packet.affected_nodes) > 0
    assert any(n.level == HATSLevel.LEVEL_0_GOAL for n in packet.affected_nodes)
    assert any(n.level == HATSLevel.LEVEL_1_SUBSYSTEM for n in packet.affected_nodes)


def test_evidence_status_disambiguation(tmp_path: Path):
    """Verifies distinct separation of execution status, solution validity,

    performance deltas, and mechanism hypothesis verification.
    """
    store = PlanStore(tmp_path)
    controller = PlanningController(store=store)
    controller.on_task_start(study_id="study_ev", task_id="task_ev", solver="solver")

    # 1. Invalid solution despite runtime improvement -> REFUTED
    ev_inv, _ = controller.on_experiment_completed(
        experiment_id="exp_inv",
        target_nodes=[],
        metrics_summary={},
        execution_status=ExecutionStatus.PASS,
        solution_validity=SolutionValidity.INVALID,
        performance_delta=PerformanceDelta.IMPROVED,
    )
    assert ev_inv.status == "refuted"
    assert ev_inv.solution_validity == SolutionValidity.INVALID

    # 2. Neutral performance change (0 delta) -> NEUTRAL (never blindly refuted!)
    ev_neut, _ = controller.on_experiment_completed(
        experiment_id="exp_neut",
        target_nodes=[],
        metrics_summary={},
        execution_status=ExecutionStatus.PASS,
        solution_validity=SolutionValidity.VALID,
        performance_delta=PerformanceDelta.NEUTRAL,
    )
    assert ev_neut.status == "neutral"

    # 3. Performance improved but mechanism support is None -> INCONCLUSIVE (not blindly supported)
    ev_inc, _ = controller.on_experiment_completed(
        experiment_id="exp_inc",
        target_nodes=[],
        metrics_summary={},
        execution_status=ExecutionStatus.PASS,
        solution_validity=SolutionValidity.VALID,
        performance_delta=PerformanceDelta.IMPROVED,
        hypothesis_supported=None,
    )
    assert ev_inc.status == "inconclusive"

    # 4. Performance improved AND mechanism hypothesis confirmed -> SUPPORTED
    ev_supp, _ = controller.on_experiment_completed(
        experiment_id="exp_supp",
        target_nodes=[],
        metrics_summary={},
        execution_status=ExecutionStatus.PASS,
        solution_validity=SolutionValidity.VALID,
        performance_delta=PerformanceDelta.IMPROVED,
        hypothesis_supported=True,
    )
    assert ev_supp.status == "supported"

    # 5. Crash -> REFUTED
    ev_fail, _ = controller.on_experiment_completed(
        experiment_id="exp_fail",
        target_nodes=[],
        metrics_summary={},
        execution_status=ExecutionStatus.FAIL,
        solution_validity=SolutionValidity.UNVERIFIED,
    )
    assert ev_fail.status == "refuted"


def test_offline_fallback_fail_fast(monkeypatch):
    """Verifies that calling decision without LLM callable fails fast in formal mode."""
    import shutil
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PORTKEY_API_KEY", raising=False)
    monkeypatch.delenv("PITBENCH_PLANNER_MODEL", raising=False)
    monkeypatch.setattr(shutil, "which", lambda cmd: None)
    monkeypatch.setenv("PITBENCH_PLANNER_BACKEND", "none")
    adapter = CodexPlannerAdapter(allow_offline_fallback=False)
    packet = ContextPacket(
        task_id="t1",
        goal_statement="Goal",
        remaining_budget_sec=100.0,
        completed_experiments_count=0,
    )
    # Must fail fast when offline fallback is disabled and no credentials exist
    with pytest.raises(RuntimeError, match="No LLM connection configured"):
        adapter.request_decision(packet)

    # Must succeed when explicitly enabled in test mode
    adapter_test = CodexPlannerAdapter(allow_offline_fallback=True)
    dec = adapter_test.request_decision(packet)
    assert dec.decision == DecisionType.STOP


def test_agy_agent_callable_configuration(monkeypatch):
    """Verifies that create_default_llm_callable invokes agy with the expected arguments."""
    import subprocess
    from unittest.mock import MagicMock

    from global_planner.adapters.codex import create_default_llm_callable

    mock_run = MagicMock()
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = '```json\n{"decision": "continue", "selected_opportunity": "opp_1", "next_experiment": "exp"}\n```'
    monkeypatch.setattr(subprocess, "run", mock_run)

    callable_fn = create_default_llm_callable(backend="antigravity", model_name="gemini-3.8-flash-low")
    assert callable_fn is not None

    output = callable_fn("Test prompt")
    assert "opp_1" in output
    assert mock_run.called
    cmd_called = mock_run.call_args[0][0]
    assert cmd_called[0].endswith("agy")
    assert "--dangerously-skip-permissions" in cmd_called
    assert "--disable-slash-commands" in cmd_called
    assert "--model" in cmd_called
    assert "gemini-3.8-flash-low" in cmd_called
    assert "--effort" in cmd_called
    assert "low" in cmd_called
    assert "--print" in cmd_called
    assert "Test prompt" in cmd_called


def test_controller_reproduction_and_boundary_triggers(tmp_path: Path):
    """Reproduces and tests the exact reviewer scenario:

    Review after 1 experiment, cooldown 0. Register initial decision, complete 1 experiment.
    Trigger result MUST return True.
    """
    store = PlanStore(tmp_path)
    controller = PlanningController(
        store=store,
        total_budget_sec=1000.0,
        cooldown_sec=0.0,
        experiments_between_reviews=1,
    )
    controller.on_task_start(
        study_id="test_study",
        task_id="scip_task",
        solver="scip",
        initial_opportunities=[
            Opportunity(
                id="opp_01",
                title="Knapsack exchange",
                target_nodes=["heuristics.oneopt"],
                proposal="Implement 6-flip swap",
                expected_effect="40x speedup on knapsack instances",
                cheapest_discriminating_experiment="Run on dev_0000 with 5s budget",
            ),
            Opportunity(
                id="opp_02",
                title="Node selection plunging",
                target_nodes=["search.nodesel"],
                proposal="Adjust plunge offset",
                expected_effect="Reduce tree depth",
                cheapest_discriminating_experiment="Run on dev instances with seed 1",
            ),
        ],
    )

    # 1. Register initial decision
    first_decision = ReplanningDecision(
        decision=DecisionType.CONTINUE,
        selected_opportunity="opp_01",
        next_experiment="pitbench bench --instance dev_0000",
    )
    controller.parse_and_record_decision(first_decision.model_dump(), current_budget_spent=0.0)
    assert controller.state.active_opportunity_id == "opp_01"

    # 2. Complete 1 experiment with supported status
    ev, should_replan = controller.on_experiment_completed(
        experiment_id="exp_01",
        target_nodes=["heuristics.oneopt"],
        metrics_summary={"gap_delta": -0.05},
        execution_status=ExecutionStatus.PASS,
        solution_validity=SolutionValidity.VALID,
        performance_delta=PerformanceDelta.IMPROVED,
        hypothesis_supported=True,
        current_budget_spent=1.0,
    )
    assert ev.status == "supported"
    # With review after 1 exp, cooldown 0, this MUST be True!
    assert should_replan is True

    # 3. Decision recorded, resetting review counter
    controller.parse_and_record_decision(first_decision.model_dump(), current_budget_spent=2.0)
    # Immediately after review, should_replan is False
    assert controller.should_trigger_replanning(current_budget_spent=2.5) is False

    # 4. Critical failure (performance regression or crash) overrides cooldown
    ev_crash, should_replan_crash = controller.on_experiment_completed(
        experiment_id="exp_02",
        target_nodes=["heuristics.oneopt"],
        metrics_summary={"error": "segfault"},
        execution_status=ExecutionStatus.FAIL,
        solution_validity=SolutionValidity.INVALID,
        current_budget_spent=3.0,
    )
    assert ev_crash.status == "refuted"
    assert should_replan_crash is True


def test_bootstrap_and_end_to_end_driver(tmp_path: Path):
    """Tests the full end-to-end automated closed-loop driver with solver-agnostic templates."""
    task_config = {
        "task_id": "release_scip_v10_1_0",
        "problem_family": "mip",
        "problem_variant": "multidimensional_knapsack",
        "release": {"repository": "scipopt/scip"},
        "repository": {"editable_paths": ["src"]},
        "evaluation": {"primary_budget_sec": 20.0, "performance_protocol": "exact_verified_solve"},
    }

    # 1. Verify bootstrap creates rich, solver-agnostic HATS model and opportunities
    goal, nodes, opps = bootstrap_hats_model(task_config)
    assert "Maximize independently verified solved coverage" in goal
    assert any(n.id == "subsystem.heuristics" for n in nodes)
    assert any("opp_explore_primal_heuristics" == o.id for o in opps)

    # 2. Driver bootstrap in test mode
    store = PlanStore(tmp_path / "driver_store")
    driver = PlanningSessionDriver(
        store=store,
        cooldown_sec=0.0,
        experiments_between_reviews=1,
        allow_offline_fallback=True,
    )
    packet, decision = driver.bootstrap(task_config=task_config)
    assert packet.task_id == "release_scip_v10_1_0"
    assert decision.decision in (DecisionType.CONTINUE, DecisionType.SWITCH)
    assert driver.store.state.active_opportunity_id == decision.selected_opportunity

    # 3. Experiment step triggering replanning
    ev, new_decision = driver.on_experiment_boundary(
        experiment_id="exp_bench_01",
        target_nodes=["subsystem.heuristics"],
        metrics_summary={"total_solved_change": 1},
        execution_status=ExecutionStatus.PASS,
        solution_validity=SolutionValidity.VALID,
        performance_delta=PerformanceDelta.IMPROVED,
        hypothesis_supported=True,
        current_budget_spent=10.0,
    )
    assert ev.status == "supported"
    assert new_decision is not None
    assert new_decision.decision == DecisionType.CONTINUE

    # 4. Critical failure triggers switch
    ev_fail, switch_decision = driver.on_experiment_boundary(
        experiment_id="exp_bench_02",
        target_nodes=["subsystem.heuristics"],
        metrics_summary={"total_solved_change": -2},
        execution_status=ExecutionStatus.PASS,
        solution_validity=SolutionValidity.VALID,
        performance_delta=PerformanceDelta.REGRESSED,
        hypothesis_supported=False,
        current_budget_spent=20.0,
    )
    assert ev_fail.status == "refuted"
    assert switch_decision is not None
    assert switch_decision.decision == DecisionType.SWITCH

    # 5. Interruption and recovery check
    resumed_store = PlanStore(tmp_path / "driver_store")
    assert resumed_store.state.experiments_count == 2
    assert len(resumed_store.state.decisions) == 3
    assert len(resumed_store.state.evidence) == 2


def test_closed_loop_agent_worker_session(tmp_path: Path):
    """Tests the full run_session loop driving a coding agent worker."""
    store = PlanStore(tmp_path / "session_store")
    driver = PlanningSessionDriver(
        store=store,
        cooldown_sec=0.0,
        experiments_between_reviews=1,
        allow_offline_fallback=True,
    )
    task_config = {
        "task_id": "test_session_task",
        "release": {"repository": "scipopt/scip"},
        "repository": {"editable_paths": ["src"]},
        "evaluation": {"primary_budget_sec": 20.0, "performance_protocol": "exact_verified_solve"},
    }

    worker_calls: list[ReplanningDecision] = []

    def mock_agent_worker(decision: ReplanningDecision, packet: ContextPacket) -> dict[str, Any]:
        worker_calls.append(decision)
        step_num = len(worker_calls)
        if step_num == 1:
            # First step: successful improvement
            return {
                "experiment_id": f"exp_worker_{step_num}",
                "modified_files": ["src/heuristic.c"],
                "candidate_patch_sha256": "abc12345",
                "execution_status": ExecutionStatus.PASS,
                "solution_validity": SolutionValidity.VALID,
                "performance_delta": PerformanceDelta.IMPROVED,
                "hypothesis_supported": True,
                "metrics_summary": {"solved_change": 2},
            }
        else:
            # Second step: regression triggering switch
            return {
                "experiment_id": f"exp_worker_{step_num}",
                "modified_files": ["src/heuristic.c"],
                "candidate_patch_sha256": "def67890",
                "execution_status": ExecutionStatus.PASS,
                "solution_validity": SolutionValidity.VALID,
                "performance_delta": PerformanceDelta.REGRESSED,
                "hypothesis_supported": False,
                "metrics_summary": {"solved_change": -1},
            }

    decisions = driver.run_session(task_config=task_config, agent_worker=mock_agent_worker, max_steps=2)
    assert len(worker_calls) == 2
    assert len(decisions) >= 2
    assert store.state.experiments_count == 2
    assert len(store.state.evidence) == 2
    assert any(n.status == "modified" for n in store.state.nodes.values())


def test_codex_adapter_format_and_extract():
    adapter = CodexPlannerAdapter(allow_offline_fallback=True)
    store = PlanStore("/tmp/test_planner_adapter_store")
    controller = PlanningController(store=store)
    controller.on_task_start(
        study_id="study_test",
        task_id="scip_task",
        solver="scip",
        initial_opportunities=[
            Opportunity(
                id="opp_01",
                title="Knapsack exchange",
                target_nodes=["heuristics.oneopt"],
                proposal="Test proposal",
                expected_effect="Test effect",
                cheapest_discriminating_experiment="Test exp",
            )
        ],
    )

    packet = controller.build_context_packet(current_budget_spent=100.0)
    prompt = adapter.format_planning_prompt(packet)
    assert "Algorithm-Space Global Replanning Request" in prompt
    assert "scip_task" in prompt

    # Mock agent reply with markdown JSON
    mock_reply = """
I have analyzed the evidence and decided to proceed with knapsack exchange.

```json
{
  "decision": "continue",
  "selected_opportunity": "opp_01",
  "evidence_refs": [],
  "alternatives_considered": [],
  "next_experiment": "pitbench bench --instance dev_0000 --budget 5.0"
}
```
"""
    extracted = adapter.extract_decision(mock_reply)
    assert extracted["decision"] == "continue"
    assert extracted["selected_opportunity"] == "opp_01"


def test_planning_study_plan_schema():
    plan_path = Path("configs/research/planning_study_scip_v1.json")
    assert plan_path.is_file(), f"Missing required plan config at {plan_path}"

    data = json.loads(plan_path.read_text(encoding="utf-8"))
    study = PlanningStudyPlan.model_validate(data)
    assert study.study_id == "algorithm_space_planning_scip_v1"
    assert len(study.arms) == 4
    assert {a.id for a in study.arms} == {
        "baseline",
        "review_only",
        "model_static",
        "model_dynamic",
    }

    # Verify validator rejects invalid arm set
    invalid_data = {
        "schema_version": "1.0",
        "study_id": "invalid_study",
        "claim_scope": "diagnostic_single_attempt",
        "release_manifest": "configs/benchmarks/0.0.1.json",
        "task_configs": ["configs/tasks/release_scip_v10_1_0.yaml"],
        "model_matrix": "configs/experiments/generalization_high_matrix.yaml",
        "population_plan": "configs/research/generalization_populations_v2.json",
        "research_budget": {
            "track": "standard",
            "wall_time_sec": 14400,
            "cpus": 8,
            "memory_gib": 32,
            "development_parallel_runs": 4,
            "checkpoints_sec": [1800, 3600, 7200, 14400],
        },
        "agent_attempts_per_condition": 1,
        "arms": [
            {
                "id": "baseline",
                "development_view": "v1",
                "planning_mode": "none",
                "instruction": "Test instruction longer than 20 characters",
            },
            {
                "id": "review_only",
                "development_view": "v2",  # Different view! Must fail validation!
                "planning_mode": "periodic_review",
                "instruction": "Test instruction longer than 20 characters",
            },
            {
                "id": "model_static",
                "development_view": "v1",
                "planning_mode": "static_model",
                "instruction": "Test instruction longer than 20 characters",
            },
            {
                "id": "model_dynamic",
                "development_view": "v1",
                "planning_mode": "dynamic_replanning",
                "instruction": "Test instruction longer than 20 characters",
            },
        ],
        "shared_agent_tools": ["bench"],
        "required_scenario_kinds": ["trajectory"],
        "inference_scope": "Inference scope description longer than 20 chars",
    }
    with pytest.raises(ValueError, match="differing from baseline"):
        PlanningStudyPlan.model_validate(invalid_data)


def test_budget_preservation_across_experiments_without_replanning(tmp_path: Path):
    """Verifies that experiment completion preserves cumulative budget even when in cooldown

    and no replanning is triggered.
    """
    store = PlanStore(tmp_path)
    controller = PlanningController(store=store, cooldown_sec=300.0, experiments_between_reviews=5)
    controller.on_task_start(
        study_id="study_budget",
        task_id="task_budget",
        solver="scip",
        initial_opportunities=[
            Opportunity(
                id="opp_01",
                title="Test",
                target_nodes=["subsystem.heuristics"],
                proposal="Test",
                expected_effect="Test",
                cheapest_discriminating_experiment="bench",
            )
        ],
    )

    # 1. Decision at t=10s
    dec = ReplanningDecision(
        decision=DecisionType.CONTINUE,
        selected_opportunity="opp_01",
        next_experiment="bench",
    )
    controller.parse_and_record_decision(dec.model_dump(), current_budget_spent=10.0)
    assert store.state.budget_spent_sec == 10.0

    # 2. Experiment completed at t=100s, but cooldown/interval prevents replanning
    ev, should_replan = controller.on_experiment_completed(
        experiment_id="exp_mid",
        target_nodes=[],
        metrics_summary={},
        execution_status=ExecutionStatus.PASS,
        solution_validity=SolutionValidity.VALID,
        performance_delta=PerformanceDelta.IMPROVED,
        hypothesis_supported=True,
        current_budget_spent=100.0,
    )
    assert should_replan is False
    # Budget MUST be updated to 100s (not stay at 10s!)
    assert store.state.budget_spent_sec == 100.0

    # 3. New driver instance recovers the full 100s
    resumed_driver = PlanningSessionDriver(store=PlanStore(tmp_path), allow_offline_fallback=True)
    assert resumed_driver.get_current_budget_spent() >= 100.0


def test_worker_missing_results_not_defaulted(tmp_path: Path):
    """Verifies that an empty worker result {} or incomplete run count is recorded as

    MISSING / UNVERIFIED, never defaulted to PASS / VALID.
    """
    store = PlanStore(tmp_path)
    driver = PlanningSessionDriver(store=store, allow_offline_fallback=True)
    task_config = {
        "task_id": "test_missing_task",
        "release": {"repository": "scipopt/scip"},
        "repository": {"editable_paths": ["src"]},
    }

    # Worker 1 returns empty dict
    def empty_worker(decision: ReplanningDecision, packet: ContextPacket) -> dict[str, Any]:
        return {}

    driver.run_session(task_config=task_config, agent_worker=empty_worker, max_steps=1)
    assert len(store.state.evidence) == 1
    ev = store.state.evidence[0]
    assert ev.execution_status == ExecutionStatus.MISSING
    assert ev.solution_validity == SolutionValidity.UNVERIFIED
    assert ev.performance_delta == PerformanceDelta.UNKNOWN

    # Worker 2 returns incomplete runs (actual < expected)
    def incomplete_worker(decision: ReplanningDecision, packet: ContextPacket) -> dict[str, Any]:
        return {
            "expected_runs_count": 4,
            "actual_runs_count": 2,
            "execution_status": ExecutionStatus.PASS,
            "solution_validity": SolutionValidity.VALID,
        }

    driver.run_session(task_config=task_config, agent_worker=incomplete_worker, max_steps=1)
    assert len(store.state.evidence) == 2
    ev2 = store.state.evidence[1]
    assert ev2.execution_status == ExecutionStatus.MISSING
    assert ev2.solution_validity == SolutionValidity.UNVERIFIED


def test_worker_receives_fresh_context_every_step_without_replanning(tmp_path: Path):
    """Verifies that even when replanning is not triggered after step 1, worker in step 2

    receives freshly updated completed_experiments_count and recent_evidence.
    """
    store = PlanStore(tmp_path)
    # Review every 10 experiments so step 1 will NOT trigger replanning
    driver = PlanningSessionDriver(
        store=store,
        experiments_between_reviews=10,
        cooldown_sec=500.0,
        allow_offline_fallback=True,
    )
    task_config = {
        "task_id": "test_fresh_context",
        "release": {"repository": "scipopt/scip"},
        "repository": {"editable_paths": ["src"]},
    }

    packets_received: list[ContextPacket] = []

    def tracking_worker(decision: ReplanningDecision, packet: ContextPacket) -> dict[str, Any]:
        packets_received.append(packet)
        return {
            "experiment_id": f"exp_{len(packets_received)}",
            "execution_status": ExecutionStatus.PASS,
            "solution_validity": SolutionValidity.VALID,
            "performance_delta": PerformanceDelta.IMPROVED,
            "hypothesis_supported": True,
        }

    driver.run_session(task_config=task_config, agent_worker=tracking_worker, max_steps=2)
    assert len(packets_received) == 2

    # Step 1 context
    assert packets_received[0].completed_experiments_count == 0
    assert len(packets_received[0].recent_evidence) == 0

    # Step 2 context MUST be fresh!
    assert packets_received[1].completed_experiments_count == 1
    assert len(packets_received[1].recent_evidence) == 1
    assert packets_received[1].recent_evidence[0].experiment_id == "exp_1"


def test_repository_source_scanning_and_concrete_nodes(tmp_path: Path):
    """Verifies that repository scanning actively discovers concrete source components

    and binds them to the Level 1 subsystems.
    """
    repo_dir = tmp_path / "repo"
    src_scip = repo_dir / "src" / "scip"
    src_scip.mkdir(parents=True, exist_ok=True)
    (src_scip / "heur_oneopt.c").write_text("// Heuristic\n")
    (src_scip / "presol_clean.c").write_text("// Presolve\n")
    (src_scip / "branch_relpscost.c").write_text("// Branching\n")
    (src_scip / "cons_linear.c").write_text("// Constraints\n")

    task_config = {
        "task_id": "scip_scan_task",
        "release": {"repository": "scipopt/scip"},
        "repository": {"editable_paths": ["src/scip"]},
    }
    goal, nodes, opps = bootstrap_hats_model(task_config, repository_root=repo_dir)

    node_ids = {n.id: n for n in nodes}
    assert "component.heuristics.heur_oneopt" in node_ids
    assert node_ids["component.heuristics.heur_oneopt"].parent_id == "subsystem.heuristics"
    assert "src/scip/heur_oneopt.c" in node_ids["component.heuristics.heur_oneopt"].source_refs

    assert "component.presolve.presol_clean" in node_ids
    assert node_ids["component.presolve.presol_clean"].parent_id == "subsystem.presolve"

    assert "component.search_engine.branch_relpscost" in node_ids
    assert node_ids["component.search_engine.branch_relpscost"].parent_id == "subsystem.search_engine"

    assert "component.constraints.cons_linear" in node_ids
    assert node_ids["component.constraints.cons_linear"].parent_id == "subsystem.constraints"


def test_dynamic_model_updating_trajectory(tmp_path: Path):
    """Verifies a trajectory where a newly discovered node and opportunity are added in step 1

    and selected by the planner in step 2.
    """
    store = PlanStore(tmp_path)
    driver = PlanningSessionDriver(
        store=store,
        experiments_between_reviews=1,
        cooldown_sec=0.0,
        allow_offline_fallback=True,
    )
    task_config = {
        "task_id": "test_trajectory_task",
        "release": {"repository": "scipopt/scip"},
        "repository": {"editable_paths": ["src"]},
    }

    step_counter = 0

    def dynamic_worker(decision: ReplanningDecision, packet: ContextPacket) -> dict[str, Any]:
        nonlocal step_counter
        step_counter += 1
        if step_counter == 1:
            # Discover a new node and opportunity
            new_node = HATSNode(
                id="mechanism.custom_relocate",
                level=HATSLevel.LEVEL_2_MECHANISM,
                parent_id="subsystem.heuristics",
                purpose="Specialized relocate neighborhood",
                source_refs=["src/custom_relocate.c"],
            )
            new_opp = Opportunity(
                id="opp_custom_relocate",
                title="Explore custom relocate neighborhood",
                target_nodes=[new_node.id],
                proposal="Test relocate proposal",
                expected_effect="Improve cost by 5%",
                cheapest_discriminating_experiment="bench --split dev",
                status="unexplored",
            )
            return {
                "experiment_id": "exp_dyn_1",
                "execution_status": ExecutionStatus.PASS,
                "solution_validity": SolutionValidity.VALID,
                "performance_delta": PerformanceDelta.REGRESSED,
                "hypothesis_supported": False,  # Triggers replanning switch!
                "discovered_nodes": [new_node],
                "discovered_opportunities": [new_opp],
            }
        else:
            return {
                "experiment_id": "exp_dyn_2",
                "execution_status": ExecutionStatus.PASS,
                "solution_validity": SolutionValidity.VALID,
                "performance_delta": PerformanceDelta.IMPROVED,
                "hypothesis_supported": True,
            }

    decisions = driver.run_session(task_config=task_config, agent_worker=dynamic_worker, max_steps=2)
    assert len(decisions) >= 2
    # Verify that the new node and opportunity were registered in store
    assert "mechanism.custom_relocate" in store.state.nodes
    assert "opp_custom_relocate" in store.state.opportunities

    # In step 1 boundary, switch decision prioritized and selected the newly discovered opportunity
    switch_decisions = [d for d in decisions if d.decision == DecisionType.SWITCH]
    assert len(switch_decisions) >= 1
    assert switch_decisions[0].selected_opportunity == "opp_custom_relocate"

    # Step 2 continued and validated the newly discovered opportunity
    assert decisions[-1].selected_opportunity == "opp_custom_relocate"


def test_autonomous_worker_session_end_to_end(tmp_path: Path):
    """Verifies that build_autonomous_worker invokes agent, parses discovered opportunities,
    and coordinates with driver for autonomous exploration."""
    repo_dir = tmp_path / "repo"
    src_dir = repo_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "solver_heur.c").write_text("// Routine\n")

    task_config = {
        "task_id": "test_auto_task",
        "release": {"repository": "scipopt/scip"},
        "repository": {"editable_paths": ["src"]},
    }

    def mock_agent_reply(prompt: str) -> str:
        return json.dumps({
            "hypothesis_id": "hyp_solver_heur",
            "hypothesis": "Identified solver_heur routine in src/solver_heur.c",
            "discovered_nodes": [
                {
                    "id": "mechanism.solver_heur",
                    "level": "level_2_mechanism",
                    "parent_id": "subsystem.heuristics",
                    "purpose": "Solver heur routine",
                    "source_refs": ["src/solver_heur.c"],
                }
            ],
            "discovered_opportunities": [
                {
                    "id": "opp_optimize_solver_heur",
                    "title": "Optimize solver heur routine",
                    "target_nodes": ["mechanism.solver_heur"],
                    "proposal": "Improve heur solve frequency",
                    "expected_effect": "Decrease solve time",
                    "cheapest_discriminating_experiment": "bench",
                }
            ],
        })

    store = PlanStore(tmp_path / "auto_store")
    driver = PlanningSessionDriver(
        store=store,
        experiments_between_reviews=1,
        cooldown_sec=0.0,
        allow_offline_fallback=True,
    )
    worker = build_autonomous_worker(
        task_config=task_config,
        repository_root=repo_dir,
        llm_callable=mock_agent_reply,
        allow_offline_fallback=True,
    )

    decisions = driver.run_session(
        task_config=task_config,
        agent_worker=worker,
        max_steps=2,
        repository_root=repo_dir,
    )
    assert len(decisions) >= 1
    assert store.state.experiments_count == 2
    # Verify discovered opportunity was parsed and registered by autonomous worker
    assert any("opp_optimize_solver_heur" in o_id for o_id in store.state.opportunities)
    # Verify evidence was recorded as MISSING / UNVERIFIED because no benchmark artifacts exist
    assert all(e.execution_status == ExecutionStatus.MISSING for e in store.state.evidence)


def test_empty_repository_autonomous_worker_no_false_success(tmp_path: Path):
    """Verifies [P0]: On a completely empty repository, autonomous worker never writes
    PASS / VALID / IMPROVED / supported without execution artifacts.
    """
    empty_repo = tmp_path / "empty_repo"
    empty_repo.mkdir()
    task_config = {
        "task_id": "test_empty_task",
        "release": {"repository": "scipopt/scip"},
        "repository": {"editable_paths": ["src"]},
    }
    store = PlanStore(tmp_path / "empty_store")
    driver = PlanningSessionDriver(store=store, allow_offline_fallback=True)

    called_count = 0

    def dummy_agent(prompt: str) -> str:
        nonlocal called_count
        called_count += 1
        return "{}"

    worker = build_autonomous_worker(
        task_config=task_config,
        repository_root=empty_repo,
        llm_callable=dummy_agent,
        allow_offline_fallback=True,
    )
    decisions = driver.run_session(
        task_config=task_config,
        agent_worker=worker,
        max_steps=2,
        repository_root=empty_repo,
    )
    assert len(decisions) >= 1
    assert called_count == 2  # Agent was actually invoked on each step!
    assert len(store.state.evidence) == 2
    for ev in store.state.evidence:
        assert ev.execution_status == ExecutionStatus.MISSING
        assert ev.solution_validity == SolutionValidity.UNVERIFIED
        assert ev.performance_delta == PerformanceDelta.UNKNOWN
        assert ev.hypothesis_supported is None
        assert ev.candidate_patch_sha256 is None


def test_zero_runs_count_marked_missing(tmp_path: Path):
    """Verifies [P1]: expected_runs_count=2, actual_runs_count=0 is marked MISSING / UNVERIFIED,
    never defaulted to PASS / VALID.
    """
    store = PlanStore(tmp_path / "zero_store")
    driver = PlanningSessionDriver(store=store, allow_offline_fallback=True)
    task_config = {
        "task_id": "test_zero_task",
        "release": {"repository": "scipopt/scip"},
        "repository": {"editable_paths": ["src"]},
    }

    def zero_runs_worker(decision: ReplanningDecision, packet: ContextPacket) -> dict[str, Any]:
        return {
            "expected_runs_count": 2,
            "actual_runs_count": 0,
            "execution_status": ExecutionStatus.PASS,
            "solution_validity": SolutionValidity.VALID,
            "performance_delta": PerformanceDelta.IMPROVED,
            "hypothesis_supported": True,
        }

    driver.run_session(task_config=task_config, agent_worker=zero_runs_worker, max_steps=1)
    assert len(store.state.evidence) == 1
    ev = store.state.evidence[0]
    assert ev.execution_status == ExecutionStatus.MISSING
    assert ev.solution_validity == SolutionValidity.UNVERIFIED
    assert ev.performance_delta == PerformanceDelta.UNKNOWN
    assert ev.hypothesis_supported is None


def test_dynamic_updates_from_raw_json_dicts(tmp_path: Path):
    """Verifies [P1]: worker returning discovered_nodes and discovered_opportunities as raw JSON
    dicts does not raise AttributeError and validates through Pydantic.
    """
    store = PlanStore(tmp_path / "json_store")
    driver = PlanningSessionDriver(store=store, allow_offline_fallback=True)
    task_config = {
        "task_id": "test_json_task",
        "release": {"repository": "scipopt/scip"},
        "repository": {"editable_paths": ["src"]},
    }

    def json_worker(decision: ReplanningDecision, packet: ContextPacket) -> dict[str, Any]:
        return {
            "experiment_id": "exp_json_1",
            "actual_runs_count": 1,
            "expected_runs_count": 1,
            "execution_status": ExecutionStatus.PASS,
            "solution_validity": SolutionValidity.VALID,
            "performance_delta": PerformanceDelta.IMPROVED,
            "hypothesis_supported": True,
            "discovered_nodes": [
                {
                    "id": "mechanism.raw_dict_node",
                    "level": "level_2_mechanism",
                    "parent_id": "subsystem.heuristics",
                    "purpose": "Discovered via raw dict",
                    "source_refs": ["src/dict.c"],
                }
            ],
            "discovered_opportunities": [
                {
                    "id": "opp_raw_dict",
                    "title": "Raw Dict Opp",
                    "target_nodes": ["mechanism.raw_dict_node"],
                    "proposal": "Proposal",
                    "expected_effect": "Effect",
                    "cheapest_discriminating_experiment": "bench",
                }
            ],
        }

    driver.run_session(task_config=task_config, agent_worker=json_worker, max_steps=1)
    assert "mechanism.raw_dict_node" in store.state.nodes
    assert "opp_raw_dict" in store.state.opportunities
    assert store.state.nodes["mechanism.raw_dict_node"].purpose == "Discovered via raw dict"


def test_codex_backend_honors_model_cwd_and_budget_timeout(monkeypatch, tmp_path: Path):
    """Verifies [P1]: Codex backend passes --model, executes in repository_root cwd,
    and constrains timeout by remaining budget.
    """
    import subprocess
    from unittest.mock import MagicMock

    from global_planner.adapters.codex import create_default_llm_callable

    mock_run = MagicMock()
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = '```json\n{"decision": "continue", "selected_opportunity": "opp_1", "next_experiment": "exp"}\n```'
    monkeypatch.setattr(subprocess, "run", mock_run)

    repo_dir = tmp_path / "repo_dir"
    repo_dir.mkdir()

    callable_fn = create_default_llm_callable(
        backend="codex",
        model_name="o3-mini",
        repository_root=repo_dir,
        remaining_budget_sec=42.5,
        timeout_sec=300.0,
    )
    assert callable_fn is not None
    callable_fn("Test prompt for codex")

    assert mock_run.called
    cmd = mock_run.call_args[0][0]
    kwargs = mock_run.call_args[1]

    # Model flag
    assert "--model" in cmd
    assert "o3-mini" in cmd
    # Cwd passed to subprocess
    assert kwargs.get("cwd") == repo_dir.resolve()
    # Timeout constrained by remaining budget
    assert kwargs.get("timeout") == 42.5

