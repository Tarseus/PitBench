"""Algorithm semantics, mechanism comparison, and real public-bench integration."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from adapters.pitbench.adapter import PitBenchAdapter
from adapters.pitbench.agent_tooling import write_agent_tooling
from pitbench import agent_cli
from pitbench.agent_tools import AgentTool, normalize_agent_tools
from pitbench.instances import materialize_instance_set
from pitbench.planning import AlgorithmPlanner
from pitbench.schema.planning import (
    AlgorithmModel,
    Evidence,
    ImplementationRecord,
    MechanismProposal,
    PlanningDecision,
    SourceVersion,
)
from pitbench.schema.task import PitBenchTask

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def source():
    return SourceVersion(base_commit="a" * 40, patch_sha256=hashlib.sha256(b"").hexdigest())


@pytest.fixture
def model(source):
    return AlgorithmModel.model_validate({
        "source": source.model_dump(),
        "nodes": [
            {"id": "solve", "level": 0, "purpose": "Improve verified solve progress under the declared budget"},
            {"id": "search", "level": 1, "parent_id": "solve", "purpose": "Explore the search space"},
            {"id": "propagation", "level": 1, "parent_id": "solve", "purpose": "Reduce variable domains"},
            {"id": "branch", "level": 2, "parent_id": "search", "purpose": "Choose a branch"},
            {"id": "filter", "level": 2, "parent_id": "propagation", "purpose": "Propagate bounds"},
            {"id": "branch_code", "level": 3, "parent_id": "branch", "purpose": "Branch implementation", "source_refs": ["src/branch.c"]},
            {"id": "filter_code", "level": 3, "parent_id": "filter", "purpose": "Propagation implementation", "source_refs": ["src/filter.c"]},
        ],
        "state": [
            {"id": "open_nodes", "node_id": "search", "description": "Unexplored nodes"},
            {"id": "bounds", "node_id": "propagation", "description": "Current variable domains"},
            {"id": "rng", "node_id": "search", "description": "Unobserved random-generator state", "hidden": True},
        ],
        "actions": [
            {"id": "select", "node_id": "branch", "description": "Select and expand a node"},
            {"id": "tighten", "node_id": "filter", "description": "Tighten implied bounds"},
        ],
        "invariants": [{"id": "safe_bounds", "description": "Do not remove any feasible optimum"}],
        "transitions": [
            {"id": "expand", "action_id": "select", "reads": ["open_nodes", "bounds", "rng"],
             "writes": ["open_nodes", "rng"], "relation": None, "stochastic": None,
             "costs": [{"resource": "cpu_time", "description": "Cost of branch evaluation is unknown"}],
             "progress": [{"metric": "verified_solved_coverage", "direction": "maximize", "description": "Global solve progress"}],
             "invariant_ids": ["safe_bounds"], "implementation_nodes": ["branch_code"],
             "unknowns": ["Tie breaking and RNG state need source/trace inspection"]},
            {"id": "reduce", "action_id": "tighten", "reads": ["bounds"], "writes": ["bounds"],
             "relation": "Intersect domains with logically implied bounds", "invariant_ids": ["safe_bounds"],
             "implementation_nodes": ["filter_code"]},
        ],
        "dependencies": [
            {"source": "propagation", "target": "search", "kind": "data", "description": "Branching consumes variable domains"},
            {"source": "propagation", "target": "search", "kind": "performance", "status": "hypothesis", "description": "More propagation might reduce future node expansion"},
        ],
        "unknowns": ["The selected variables are not guaranteed to form a Markov state"],
    })


def proposal_for(model, transition_id="expand", proposal_id="compound", revision=1):
    transition = next(item for item in model.transitions if item.id == transition_id).model_dump()
    transition.update(
        status="hypothesis", relation="Compose compatible operations before committing the next state"
    )
    return MechanismProposal.model_validate({
        "id": proposal_id, "model_revision": revision,
        "target_transitions": [transition_id], "replacement_transitions": [transition],
        "invariant_ids": ["safe_bounds"],
        "expected_global_effect": "Additional work per action may reduce total search work",
        "implementation_change": "Implement the compound operation in the referenced routine",
        "experiment": {
            "control": "Compare with only this mechanism disabled under matched instance, seed and budget",
            "measurements": ["verified_solved_coverage", "par2", "search_work"],
            "falsifies_if": "Additional per-action cost is not offset by reduced total work",
        },
    })


def decide(planner, source, selected, evidence_refs=()):
    revision = len(planner.state.models)
    planner.decide(PlanningDecision(
        model_revision=revision, selected_proposals=selected,
        global_reason="Compare the expected total search reduction with per-action overhead and the other subsystem's opportunity",
        alternatives=[
            {"proposal_id": key, "defer_reason": "The current paired experiment distinguishes the selected direction more cheaply"}
            for key, proposal in planner.state.proposals.items()
            if proposal.model_revision == revision and key not in selected
        ],
        evidence_refs=list(evidence_refs),
    ), source)


def implemented(planner, source, record_id="implemented"):
    record = ImplementationRecord(
        id=record_id, decision_index=len(planner.state.decisions), source=source,
        proposals=[
            {"proposal_id": ref, "status": "implemented"}
            for ref in planner.require_decision().selected_proposals
        ],
        source_refs=["src/branch.c"],
    )
    planner.add_implementation(record, source)
    return record


def test_partial_hats_preserves_unknowns_and_validates_semantic_connections(model):
    assert model.state[-1].hidden
    assert model.transitions[0].relation is None
    assert model.transitions[0].costs[0].status == "unknown"
    assert model.transitions[0].progress[0].direction == "maximize"
    for field, value in (("reads", ["undefined_memory"]), ("invariant_ids", ["undefined_invariant"]), ("implementation_nodes", ["search"])):
        data = model.model_dump()
        data["transitions"][0][field] = value
        with pytest.raises(ValueError):
            AlgorithmModel.model_validate(data)
    data = model.model_dump()
    data["nodes"][-1]["parent_id"] = "solve"
    with pytest.raises(ValueError, match="preceding level"):
        AlgorithmModel.model_validate(data)


def test_development_evidence_uses_the_retained_agent_log_volume():
    import yaml

    from pitbench.harness.terminal.docker_compose_manager import DockerComposeManager

    client = yaml.safe_load(PitBenchAdapter._compose(["bench", "plan"]))["services"]["client"]
    assert Path(client["environment"]["PITBENCH_RUNS"]).is_relative_to(
        DockerComposeManager.CONTAINER_AGENT_LOGS_PATH
    )
    assert "${T_BENCH_TASK_AGENT_LOGS_PATH}:${T_BENCH_CONTAINER_AGENT_LOGS_PATH}" in client["volumes"]


def test_mechanism_changes_require_invariants_and_source_realization(tmp_path, model, source):
    planner = AlgorithmPlanner(tmp_path / "plan.json", "task", "Improve search")
    planner.update_model(model, source)
    proposal = proposal_for(model)
    bad = proposal.model_copy(deep=True)
    bad.replacement_transitions[0].invariant_ids.clear()
    with pytest.raises(ValueError, match="retain.*invariants"):
        planner.add_proposal(bad)
    bad = proposal.model_copy(deep=True)
    bad.replacement_transitions[0].implementation_nodes.clear()
    with pytest.raises(ValueError, match="Level 3"):
        planner.add_proposal(bad)
    planner.add_proposal(proposal)
    other = proposal_for(model, "reduce", "propagation_change")
    planner.add_proposal(other)
    with pytest.raises(ValueError, match="other current mechanism"):
        planner.decide(PlanningDecision(
            model_revision=1, selected_proposals=[proposal.id], global_reason="Only looked at branching"
        ), source)
    decide(planner, source, [proposal.id])
    assert planner.require_decision().alternatives[0].proposal_id == other.id


def test_compound_changes_can_modify_state_and_actions_with_explicit_dependencies(tmp_path, model, source):
    planner = AlgorithmPlanner(tmp_path / "plan.json", "task", "Improve search")
    planner.update_model(model, source)
    first = proposal_for(model, "reduce", "propagation_change")
    planner.add_proposal(first)
    compound = proposal_for(model)
    compound.depends_on = [first.id]
    compound.state_updates = [model.state[0].model_copy(update={"id": "compound_cache", "hidden": True})]
    compound.action_updates = [model.actions[0].model_copy(update={"id": "compound_select"})]
    compound.replacement_transitions[0].action_id = "compound_select"
    compound.replacement_transitions[0].reads.append("compound_cache")
    planner.add_proposal(compound)
    with pytest.raises(ValueError, match="dependencies together"):
        decide(planner, source, [compound.id])
    decide(planner, source, [first.id, compound.id])
    assert len(planner.require_decision().selected_proposals) == 2


def test_combining_proposals_cannot_silently_overwrite_the_same_transition(tmp_path, model, source):
    planner = AlgorithmPlanner(tmp_path / "plan.json", "task", "Improve search")
    planner.update_model(model, source)
    planner.add_proposal(proposal_for(model, proposal_id="compound"))
    planner.add_proposal(proposal_for(model, proposal_id="branch_cache"))
    with pytest.raises(ValueError, match="overlapping transitions"):
        decide(planner, source, ["compound", "branch_cache"])
    assert not planner.state.decisions
    decide(planner, source, ["compound"])
    assert planner.require_decision().selected_proposals == ["compound"]


def test_replacement_mechanisms_expand_experiment_scope_across_subsystems(tmp_path, model, source):
    planner = AlgorithmPlanner(tmp_path / "plan.json", "task", "Improve search")
    planner.update_model(model, source)
    proposal = proposal_for(model)
    proposal.action_updates = [model.actions[1].model_copy(update={"id": "new_filter_action"})]
    proposal.replacement_transitions[0].action_id = "new_filter_action"
    proposal.replacement_transitions[0].implementation_nodes = ["filter_code"]
    planner.add_proposal(proposal)
    decide(planner, source, ["compound"])
    candidate = planner.state.decisions[-1].candidate_model
    assert next(item for item in candidate.transitions if item.id == "expand").implementation_nodes == ["filter_code"]
    implemented(planner, source)
    evidence = Evidence(
        id="cross_subsystem", kind="experiment", source=source, summary="Compound transition run",
        implementation_id="implemented", expected_runs=1, observations=[{"seed": 7}],
        execution_status="pass", solution_validity="valid",
    )
    planner.record_experiment(evidence)
    assert set(evidence.affected_nodes) == {
        "solve", "search", "branch", "branch_code", "propagation", "filter", "filter_code",
    }


def test_dependency_influence_is_kept_separate_from_direct_scope(tmp_path, model, source):
    model.dependencies.append(model.dependencies[1].model_copy(update={"source": "branch", "target": "filter"}))
    planner = AlgorithmPlanner(tmp_path / "plan.json", "task", "Improve search")
    planner.update_model(model, source)
    planner.add_proposal(proposal_for(model))
    decide(planner, source, ["compound"])
    implemented(planner, source)
    evidence = Evidence(
        id="potential", kind="experiment", source=source, summary="Observe branching",
        implementation_id="implemented", expected_runs=1, observations=[{}],
    )
    planner.record_experiment(evidence)
    assert set(evidence.affected_nodes) == {"solve", "search", "branch", "branch_code"}
    assert set(evidence.potential_affected_nodes) == {"filter", "propagation"}
    assert not set(evidence.affected_nodes) & set(evidence.potential_affected_nodes)


def test_implementation_records_expose_partial_abandoned_and_unplanned_mechanisms(tmp_path, model, source):
    planner = AlgorithmPlanner(tmp_path / "plan.json", "task", "Improve search")
    planner.update_model(model, source)
    planner.add_proposal(proposal_for(model))
    planner.add_proposal(proposal_for(model, "reduce", "propagation_instead"))
    decide(planner, source, ["compound"])
    actual = source.model_copy(update={"patch_sha256": "d" * 64})
    record = ImplementationRecord(
        id="shifted", decision_index=1, source=actual,
        proposals=[
            {"proposal_id": "compound", "status": "abandoned"},
            {"proposal_id": "propagation_instead", "status": "partial"},
        ],
        source_refs=["src/filter.c"],
    )
    with pytest.raises(ValueError, match="source-backed deviations"):
        planner.add_implementation(record, actual)
    record = ImplementationRecord.model_validate({
        **record.model_dump(), "deviations": [{
            "description": "Abandoned compound branching; only implemented one propagation shortcut",
            "source_refs": ["src/filter.c"],
        }],
    })
    with pytest.raises(ValueError, match="current source"):
        planner.add_implementation(record, source)
    with pytest.raises(ValueError, match="current decision"):
        planner.add_implementation(record.model_copy(update={"decision_index": 2}), actual)
    with pytest.raises(ValueError, match="every selected proposal"):
        planner.add_implementation(record.model_copy(update={"proposals": record.proposals[1:]}), actual)
    planner.add_implementation(record, actual)
    with pytest.raises(ValueError, match="does not match"):
        planner.require_implementation(record.id, source)
    evidence = Evidence(
        id="measure_actual", kind="experiment", source=actual, summary="Measure partial propagation",
        implementation_id=record.id, expected_runs=1, observations=[{}],
    )
    planner.record_experiment(evidence)
    assert evidence.decision_index == 1
    assert evidence.proposal_ids == ["propagation_instead"]
    assert set(evidence.affected_nodes) == {"filter_code", "filter", "propagation", "solve"}
    restored = AlgorithmPlanner(planner.path, "task", "Improve search")
    assert restored.state.decisions[0].selected_proposals == ["compound"]
    assert restored.state.implementations[record.id].source == actual
    assert restored.state.implementations[record.id].deviations[0].source_refs == ["src/filter.c"]


@pytest.mark.parametrize("failure", ["exit", "timeout", "missing_command", "no_commands"])
def test_build_attempt_is_registered_before_execution_and_failures_remain_reviewable(
    tmp_path, monkeypatch, model, source, failure
):
    runs = tmp_path / "runs"
    planner = AlgorithmPlanner(runs / "algorithm_plan.json", "task", "Improve search")
    planner.update_model(model, source)
    planner.add_proposal(proposal_for(model))
    decide(planner, source, ["compound"])
    implemented(planner, source)
    command = {"argv": [sys.executable, "-c", "import sys; print('build output'); print('build error', file=sys.stderr); sys.exit(9)"]}
    if failure == "timeout":
        command = {"argv": [sys.executable, "-c", "import time; time.sleep(1)"], "timeout_sec": 0.01}
    elif failure == "missing_command":
        command = {"argv": [str(tmp_path / "missing-compiler")]}
    config = {
        "task_id": "task", "instruction": "Improve search", "agent_tools": ["bench", "plan"],
        "development_seeds": [7], "budgets_sec": [5.0], "primary_budget_sec": 5.0,
        "threads": 1, "build_commands": [] if failure == "no_commands" else [command],
    }
    instance = tmp_path / "dev.json"
    instance.write_text("{}")
    monkeypatch.setattr(agent_cli, "_config", lambda: config)
    monkeypatch.setattr(agent_cli, "_repository", lambda: tmp_path)
    monkeypatch.setattr(agent_cli, "_runs_root", lambda _: runs)
    monkeypatch.setattr(agent_cli, "_instances", lambda *_: [instance])
    monkeypatch.setattr(agent_cli, "_candidate_patch", lambda _: b"candidate")
    monkeypatch.setattr(agent_cli, "_planning_source", lambda *_: source)
    monkeypatch.setattr(agent_cli, "_run_one", lambda *a, **kw: pytest.fail("solver ran after build failed"))
    build = agent_cli._run_build_steps

    def inspect_build(config, **kwargs):
        active = AlgorithmPlanner(planner.path, "task", "Improve search")
        state = active.state
        attempt = list(state.evidence.values())[-1]
        assert attempt.stage == "build" and attempt.attempt_status == "running"
        assert attempt.implementation_id == "implemented"
        assert attempt.proposal_ids == ["compound"]
        assert state.pending_evidence == [attempt.id]
        with pytest.raises(ValueError, match="finish the running experiment"):
            decide(active, source, ["compound"], [attempt.id])
        return build(config, **kwargs)

    monkeypatch.setattr(agent_cli, "_run_build_steps", inspect_build)
    args = agent_cli.parser().parse_args(["bench", "--build", "--implementation-id", "implemented"])
    assert agent_cli.bench_command(args) == 1
    restored = AlgorithmPlanner(planner.path, "task", "Improve search")
    assert len(restored.state.evidence) == 1
    attempt = next(iter(restored.state.evidence.values()))
    assert attempt.execution_status == "fail" and attempt.solution_validity == "unverified"
    assert attempt.attempt_status == "finished" and attempt.failure_stage == "build"
    assert attempt.source == source and attempt.decision_index == 1
    assert attempt.implementation_id == "implemented" and attempt.proposal_ids == ["compound"]
    assert attempt.artifact_paths and all(Path(ref).is_file() for ref in attempt.artifact_paths)
    report = next(Path(ref) for ref in attempt.artifact_paths if ref.endswith("report.json"))
    assert json.loads(report.read_text())["status"] == "fail"
    with pytest.raises(ValueError, match="review evidence"):
        agent_cli.bench_command(args)


@pytest.mark.parametrize("execution,validity,count", [("pass", "valid", 2), ("fail", "invalid", 2), ("pass", "valid", 1)])
def test_experiments_feed_back_to_all_algorithm_levels_without_automatic_causal_claims(
    tmp_path, model, source, execution, validity, count
):
    planner = AlgorithmPlanner(tmp_path / "plan.json", "task", "Improve search")
    planner.update_model(model, source)
    planner.add_proposal(proposal_for(model))
    decide(planner, source, ["compound"])
    implemented(planner, source)
    original_model = model.model_dump()
    evidence = Evidence(
        id="measured", kind="experiment", source=source, summary="Public paired run",
        expected_runs=2, execution_status=execution, solution_validity=validity,
        observations=[{"seed": seed, "preview_par2_reduction_sec": -0.3} for seed in range(count)],
        workload={"budget_sec": 5, "threads": 1},
        implementation_id="implemented",
    )
    planner.record_experiment(evidence)
    restored = AlgorithmPlanner(planner.path, "task", "Improve search")
    saved = restored.state.evidence["measured"]
    assert saved.affected_nodes == ["branch", "branch_code", "search", "solve"]
    assert saved.proposal_ids == ["compound"]
    assert len(saved.observations) == count
    assert saved.workload == {"budget_sec": 5, "threads": 1}
    assert saved.execution_status == ("missing" if count < 2 else execution)
    assert saved.solution_validity == ("unverified" if count < 2 else validity)
    assert restored.state.models[-1].model_dump() == original_model
    with pytest.raises(ValueError, match="review evidence"):
        restored.require_decision()
    with pytest.raises(ValueError, match="pending experiment"):
        decide(restored, source, ["compound"])
    decide(restored, source, ["compound"], [saved.id])
    assert not restored.state.pending_evidence


def test_source_dependencies_do_not_establish_performance_influence(tmp_path, model, source):
    planner = AlgorithmPlanner(tmp_path / "plan.json", "task", "Improve search")
    planner.add_evidence(Evidence(
        id="source_read", kind="source", source=source,
        summary="Bounds are read by the branching routine", artifact_paths=["src/branch.c"],
    ), source)
    model.dependencies[0].status = "observed"
    model.dependencies[0].evidence_refs = ["source_read"]
    planner.update_model(model, source)
    model.dependencies[1].status = "observed"
    model.dependencies[1].evidence_refs = ["source_read"]
    with pytest.raises(ValueError, match="experimental evidence"):
        planner.update_model(model, source)
    with pytest.raises(ValueError, match="must come from pitbench bench"):
        planner.add_evidence(Evidence(
            id="invented", kind="experiment", source=source, summary="Agent claimed speedup"
        ), source)


def test_replanning_requires_current_source_and_preserves_prior_model_versions(tmp_path, model, source):
    planner = AlgorithmPlanner(tmp_path / "plan.json", "task", "Improve search")
    planner.update_model(model, source)
    planner.add_proposal(proposal_for(model))
    changed = source.model_copy(update={"patch_sha256": "b" * 64})
    assert planner.context(changed)["model_needs_review"]
    with pytest.raises(ValueError, match="current source"):
        decide(planner, changed, ["compound"])
    revised = model.model_copy(deep=True)
    revised.source = changed
    revised.transitions[0].unknowns.append("The edited implementation needs new traces")
    planner.update_model(revised, changed)
    planner.add_proposal(proposal_for(revised, proposal_id="compound_revised", revision=2))
    decide(planner, changed, ["compound_revised"])
    restored = AlgorithmPlanner(planner.path, "task", "Improve search")
    assert [item.source for item in restored.state.models] == [source, changed]
    assert set(restored.state.proposals) == {"compound", "compound_revised"}


def test_measured_outcomes_revise_the_model_and_change_global_mechanism_selection(tmp_path, model, source):
    planner = AlgorithmPlanner(tmp_path / "plan.json", "task", "Improve search")
    planner.update_model(model, source)
    planner.add_proposal(proposal_for(model))
    planner.add_proposal(proposal_for(model, "reduce", "try_propagation"))
    decide(planner, source, ["compound"])
    changed = source.model_copy(update={"patch_sha256": "c" * 64})
    implemented(planner, changed)
    planner.record_experiment(Evidence(
        id="compound_cost", kind="experiment", source=changed,
        summary="Compound work increased total search cost under matched conditions",
        expected_runs=1, execution_status="pass", solution_validity="valid",
        observations=[{"seed": 7, "preview_par2_reduction_sec": -0.4}],
        implementation_id="implemented",
    ))
    revised = model.model_copy(deep=True)
    revised.source = changed
    revised.transitions[0].costs[0].description = "Compound branch evaluation increased CPU work"
    revised.transitions[0].costs[0].status = "observed"
    revised.transitions[0].costs[0].evidence_refs = ["compound_cost"]
    planner.update_model(revised, changed)
    planner.add_proposal(proposal_for(revised, "expand", "reduce_branch_cost", revision=2))
    planner.add_proposal(proposal_for(revised, "reduce", "try_propagation_next", revision=2))
    decide(planner, changed, ["try_propagation_next"], ["compound_cost"])
    restored = AlgorithmPlanner(planner.path, "task", "Improve search")
    assert restored.state.decisions[-1].selected_proposals == ["try_propagation_next"]
    assert restored.state.models[0].transitions[0].costs[0].status == "unknown"
    assert restored.state.models[1].transitions[0].costs[0].evidence_refs == ["compound_cost"]
    assert len(restored.state.models) == 2


def test_planning_is_bundled_into_the_existing_agent_tools(tmp_path, model):
    with pytest.raises(ValueError, match="bench tool"):
        normalize_agent_tools(["plan"])
    task = PitBenchTask.from_yaml(ROOT / "configs/tasks/pyvrp_v0_14_0.yaml")
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    development = next(item for item in task.instance_sets if item.kind.value == "agent_dev")
    materialize_instance_set(ROOT / development.instance_set_config, task_dir / "dev_instances")
    tools = frozenset({AgentTool.INSPECT, AgentTool.BENCH, AgentTool.PLAN})
    write_agent_tooling(repository_root=ROOT, task=task, task_dir=task_dir, agent_tools=tools)
    config = json.loads((task_dir / "agent_config.json").read_text())
    assert config["instruction"] == task.instruction
    assert "Algorithm-Space Global Planning" in PitBenchAdapter._format_instruction(task, tools)
    assert "Algorithm-Space Global Planning" not in PitBenchAdapter._format_instruction(task, tools - {AgentTool.PLAN})
    env = {**os.environ, "PYTHONPATH": str(task_dir / "agent_tooling"), "PITBENCH_AGENT_CONFIG": str(task_dir / "agent_config.json")}
    result = subprocess.run(
        [sys.executable, "-m", "pitbench.agent_cli", "plan", "--schema"],
        cwd=task_dir, env=env, text=True, capture_output=True, check=True,
    )
    schemas = json.loads(result.stdout)
    assert "transitions" in schemas["model"]["properties"]
    assert "replacement_transitions" in schemas["proposal"]["properties"]
    repository = tmp_path / "repo"
    (repository / "src").mkdir(parents=True)
    (repository / "src/branch.c").write_text("int branch(void) { return 1; }\n")
    (repository / "src/filter.c").write_text("int filter(void) { return 0; }\n")
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(["git", "add", "src"], cwd=repository, check=True)
    subprocess.run([
        "git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
        "commit", "-qm", "Base",
    ], cwd=repository, check=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True, capture_output=True, check=True
    ).stdout.strip()
    model.source = SourceVersion(base_commit=head, patch_sha256=hashlib.sha256(b"").hexdigest())
    env.update(PITBENCH_REPO=str(repository), PITBENCH_RUNS=str(tmp_path / "runs"))

    def submit(option, record):
        path = task_dir / f"{option}.json"
        path.write_text(record.model_dump_json())
        result = subprocess.run(
            [sys.executable, "-m", "pitbench.agent_cli", "plan", f"--{option}", str(path)],
            cwd=task_dir, env=env, capture_output=True, text=True, check=True,
        )
        return json.loads(result.stdout)

    submit("evidence", Evidence(
        id="code_read", kind="source", source=model.source,
        summary="Inspected the branch routine", artifact_paths=["src/branch.c"],
    ))
    model.nodes[-2].status = "observed"
    model.nodes[-2].evidence_refs = ["code_read"]
    context = submit("model", model)
    assert context["model_revision"] == 1
    submit("proposal", proposal_for(model))
    context = submit("decision", PlanningDecision(
        model_revision=1, selected_proposals=["compound"],
        global_reason="Test the compound mechanism while measuring global search cost",
    ))
    assert context["decision"]["selected_proposals"] == ["compound"]
    assert context["decision"]["candidate_model"]["transitions"]
    submit("implementation", ImplementationRecord(
        id="branch_change", decision_index=context["decision_index"], source=model.source,
        proposals=[{"proposal_id": "compound", "status": "implemented"}],
        source_refs=["src/branch.c"],
    ))
    persisted = AlgorithmPlanner(tmp_path / "runs/algorithm_plan.json", task.task_id, task.instruction)
    assert persisted.require_decision().model_revision == 1
    assert persisted.require_implementation("branch_change", model.source).decision_index == 1
    unbound = subprocess.run(
        [sys.executable, "-m", "pitbench.agent_cli", "bench"],
        cwd=task_dir, env=env, capture_output=True, text=True,
    )
    assert unbound.returncode != 0 and "--implementation-id" in unbound.stderr
    (repository / "src/branch.c").write_text("int branch(void) { return 2; }\n")
    stale = subprocess.run(
        [sys.executable, "-m", "pitbench.agent_cli", "bench", "--implementation-id", "branch_change"],
        cwd=task_dir, env=env, capture_output=True, text=True,
    )
    assert stale.returncode != 0 and "does not match the current source" in stale.stderr
    assert all(
        record.kind != "experiment"
        for record in AlgorithmPlanner(persisted.path, task.task_id, task.instruction).state.evidence.values()
    )


@pytest.mark.parametrize("outcome", ["complete", "built", "absent", "corrupt", "failed"])
def test_existing_bench_records_raw_evidence_and_requests_model_revision(
    tmp_path, monkeypatch, model, source, outcome
):
    instance = tmp_path / "dev_0000.json"
    instance.write_text("{}")
    config = {
        "task_id": "task", "instruction": "Improve global solver performance",
        "agent_tools": ["bench", "plan"], "budgets_sec": [5.0],
        "primary_budget_sec": 5.0, "development_seeds": [7, 8], "threads": 1,
        "build_commands": [{"argv": [sys.executable, "-c", "print('build succeeded')"]}],
    }
    runs = tmp_path / "runs"
    (runs / "agent").mkdir(parents=True)
    planner = AlgorithmPlanner(runs / "algorithm_plan.json", "task", config["instruction"])
    planner.update_model(model, source)
    planner.add_proposal(proposal_for(model))
    decide(planner, source, ["compound"])
    implemented(planner, source)
    monkeypatch.setattr(agent_cli, "_config", lambda: config)
    monkeypatch.setattr(agent_cli, "_repository", lambda: tmp_path)
    monkeypatch.setattr(agent_cli, "_candidate_patch", lambda _: b"candidate")
    monkeypatch.setattr(agent_cli, "_planning_source", lambda *_: source)
    monkeypatch.setattr(agent_cli, "_runs_root", lambda _: runs)
    monkeypatch.setattr(agent_cli, "_instances", lambda *_: [instance])
    monkeypatch.setattr(agent_cli, "_artifact_identity", lambda _: {})

    def run_one(_config, path, seed, budget, dry_run, label, patch, threads, artifacts, root):
        destination = root / label / (
            f"{path.stem}-{agent_cli._sha256(path)[:12]}.seed-{seed}."
            f"budget-{budget:g}.threads-{threads}.feedback.json"
        )
        if seed == 8 and outcome in ("absent", "corrupt"):
            if outcome == "corrupt":
                destination.write_text("invalid json")
        else:
            payload = {"seed": seed, "verified_valid": True, "preview_gap_reduction": -0.2}
            if outcome == "failed" and seed == 8:
                payload["result_error"] = "solver crashed"
            destination.write_text(json.dumps(payload))
        return outcome != "failed" or seed != 8

    monkeypatch.setattr(agent_cli, "_run_one", run_one)
    args = agent_cli.parser().parse_args(["bench", "--all-seeds", "--implementation-id", "implemented"])
    args.build = outcome == "built"
    assert agent_cli.bench_command(args) == (0 if outcome in ("complete", "built") else 1)
    restored = AlgorithmPlanner(planner.path, "task", config["instruction"])
    evidence = next(iter(restored.state.evidence.values()))
    assert evidence.execution_status == {"complete": "pass", "built": "pass", "failed": "fail"}.get(outcome, "missing")
    assert evidence.expected_runs == 2
    assert evidence.observations[0]["preview_gap_reduction"] == -0.2
    assert evidence.workload["seeds"] == [7, 8]
    assert evidence.proposal_ids == ["compound"]
    assert evidence.implementation_id == "implemented"
    assert evidence.decision_index == 1
    assert evidence.attempt_status == "finished"
    if outcome == "built":
        assert any(ref.endswith("report.json") for ref in evidence.artifact_paths)
    assert set(evidence.affected_nodes) == {"solve", "search", "branch", "branch_code"}
    assert restored.state.pending_evidence == [evidence.id]
    with pytest.raises(ValueError, match="review evidence"):
        agent_cli.bench_command(args)
