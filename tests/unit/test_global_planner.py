"""Unit tests for the Global Planner and Hierarchical Algorithmic Transition System (HATS)."""

import json
from pathlib import Path

import pytest

from global_planner.adapters.codex import CodexPlannerAdapter
from global_planner.controller import PlanningController
from global_planner.schema import (
    DecisionType,
    HATSLevel,
    HATSNode,
    MechanismEffect,
    Opportunity,
    ReplanningDecision,
)
from global_planner.store import PlanStore
from pitbench.schema.research import PlanningStudyPlan


def test_schema_hats_node():
    node = HATSNode(
        id="heuristics.oneopt.knapsack_exchange",
        level=HATSLevel.LEVEL_2_MECHANISM,
        purpose="Enumerate 6-flip binary exchanges on tight knapsack constraints",
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


def test_plan_store_persistence(tmp_path: Path):
    store = PlanStore(tmp_path)
    state = store.initialize(study_id="study_01", task_id="scip_v10", solver="scip")
    assert state.revision == 1

    node = HATSNode(
        id="search.branching",
        level=HATSLevel.LEVEL_1_SUBSYSTEM,
        purpose="Variable branching rule",
    )
    store.register_node(node)
    assert store.state.revision == 2
    assert "search.branching" in store.state.nodes

    # Check history snapshot
    history_files = list((tmp_path / "history").glob("*.json"))
    assert len(history_files) == 2

    # Reload in a new store instance
    new_store = PlanStore(tmp_path)
    assert new_store.state.solver == "scip"
    assert "search.branching" in new_store.state.nodes


def test_controller_experiment_and_replanning_triggers(tmp_path: Path):
    store = PlanStore(tmp_path)
    controller = PlanningController(
        store=store,
        total_budget_sec=1000.0,
        cooldown_sec=50.0,
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

    # First decision should be triggered immediately (no prior decisions)
    assert controller.should_trigger_replanning(current_budget_spent=0.0) is True

    # Record first decision
    first_decision = ReplanningDecision(
        decision=DecisionType.CONTINUE,
        selected_opportunity="opp_01",
        next_experiment="pitbench bench --instance dev_0000",
    )
    controller.parse_and_record_decision(first_decision.model_dump(), current_budget_spent=10.0)
    assert controller.state.active_opportunity_id == "opp_01"

    # Immediately after decision, cooldown prevents replanning
    assert controller.should_trigger_replanning(current_budget_spent=15.0) is False

    # Experiment completes with supported status, but within cooldown (<50s)
    ev, should_replan = controller.on_experiment_completed(
        experiment_id="exp_01",
        target_nodes=["heuristics.oneopt"],
        metrics_summary={"gap_delta": -0.05},
        is_supported=True,
        current_budget_spent=20.0,
    )
    assert ev.status == "supported"
    assert should_replan is False

    # Later experiment completes with refuted status (critical evidence) -> overrides cooldown
    ev2, should_replan2 = controller.on_experiment_completed(
        experiment_id="exp_02",
        target_nodes=["heuristics.oneopt"],
        metrics_summary={"error": "timeout"},
        is_supported=False,
        current_budget_spent=30.0,
    )
    assert ev2.status == "refuted"
    assert should_replan2 is True

    # Agent decides to switch to opp_02
    switch_decision = ReplanningDecision(
        decision=DecisionType.SWITCH,
        selected_opportunity="opp_02",
        switch_justification="opp_01 refuted by timeout evidence ev2",
        next_experiment="pitbench bench --instance dev_0001",
        alternatives_considered=[{"id": "opp_01", "defer_reason": "refuted by timeout"}],
    )
    controller.parse_and_record_decision(switch_decision.model_dump(), current_budget_spent=40.0)
    assert controller.state.active_opportunity_id == "opp_02"
    assert controller.state.opportunities["opp_01"].status == "deferred"


def test_codex_adapter_format_and_extract():
    adapter = CodexPlannerAdapter()
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


def test_planning_study_plan_schema(tmp_path: Path):
    plan_path = Path("configs/research/planning_study_scip_v1.json")
    if plan_path.exists():
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
