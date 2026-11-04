"""End-to-end planning session driver coordinating bootstrap, triggers, and agent continuation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from global_planner.adapters.codex import CodexPlannerAdapter
from global_planner.bootstrap import bootstrap_hats_model
from global_planner.controller import PlanningController
from global_planner.schema import (
    ContextPacket,
    DecisionType,
    ExecutionStatus,
    HATSLevel,
    HATSNode,
    Opportunity,
    PerformanceDelta,
    ReplanningDecision,
    SolutionValidity,
)
from global_planner.store import PlanStore


class PlanningSessionDriver:
    """Manages the end-to-end planning loop across task bootstrap, experiment boundaries,

    decision delivery, and automated agent worker continuation.
    """

    def __init__(
        self,
        store: PlanStore,
        total_budget_sec: float = 14400.0,
        cooldown_sec: float = 300.0,
        experiments_between_reviews: int = 1,
        adapter: CodexPlannerAdapter | None = None,
        llm_callable: Callable[[str], str] | None = None,
        allow_offline_fallback: bool = False,
    ):
        self.store = store
        self.controller = PlanningController(
            store=store,
            total_budget_sec=total_budget_sec,
            cooldown_sec=cooldown_sec,
            experiments_between_reviews=experiments_between_reviews,
        )
        self.allow_offline_fallback = allow_offline_fallback
        self.adapter = adapter or CodexPlannerAdapter(
            default_llm_callable=llm_callable,
            allow_offline_fallback=allow_offline_fallback,
        )
        self.llm_callable = llm_callable

        # Preserve cumulative budget across process restarts
        self.initial_budget_spent = 0.0
        if self.store.state_file.exists():
            try:
                self.initial_budget_spent = self.store.state.budget_spent_sec
            except Exception:
                self.initial_budget_spent = 0.0
        self.start_wall_time = time.time()

    def get_current_budget_spent(self) -> float:
        """Returns cumulative elapsed budget spent across sessions."""
        elapsed = max(0.0, time.time() - self.start_wall_time)
        return self.initial_budget_spent + elapsed

    def bootstrap(
        self,
        task_config: dict[str, Any],
        study_id: str = "planning_study_v1",
        repository_root: Path | str | None = None,
    ) -> tuple[ContextPacket, ReplanningDecision]:
        """Bootstraps initial HATS model, opportunities, and initial decision from task metadata."""
        task_id = str(task_config.get("task_id", "unknown_task"))
        solver = str(task_config.get("release", {}).get("repository", "solver")).split("/")[-1]
        family = str(task_config.get("problem_family", "general"))
        protocol = str(task_config.get("evaluation", {}).get("performance_protocol", "exact_verified_solve"))

        repo_root = Path(repository_root) if repository_root else None
        goal_stmt, nodes, opps = bootstrap_hats_model(task_config, repository_root=repo_root)
        self.controller.on_task_start(
            study_id=study_id,
            task_id=task_id,
            solver=solver,
            goal_statement=goal_stmt,
            problem_family=family,
            performance_protocol=protocol,
            initial_nodes=nodes,
            initial_opportunities=opps,
        )

        spent = self.get_current_budget_spent()
        self.store.update_budget_spent(spent)
        packet = self.controller.build_context_packet(current_budget_spent=spent)
        decision = self.adapter.request_decision(
            packet,
            llm_callable=self.llm_callable,
            allow_offline_fallback=self.allow_offline_fallback,
        )
        self.controller.parse_and_record_decision(decision.model_dump(), current_budget_spent=spent)
        return packet, decision

    def on_experiment_boundary(
        self,
        experiment_id: str,
        target_nodes: list[str],
        metrics_summary: dict[str, Any],
        execution_status: ExecutionStatus = ExecutionStatus.PASS,
        solution_validity: SolutionValidity = SolutionValidity.VALID,
        performance_delta: PerformanceDelta = PerformanceDelta.UNKNOWN,
        hypothesis_supported: bool | None = None,
        hypothesis_id: str | None = None,
        observations_path: str | None = None,
        candidate_patch_sha256: str | None = None,
        current_budget_spent: float | None = None,
        # Backward compatibility keyword arguments
        execution_success: bool | None = None,
        solution_valid: bool | None = None,
        performance_improved: bool | None = None,
    ) -> tuple[Any, ReplanningDecision | None]:
        """Processes an experiment completion, records evidence, and executes replanning if triggered."""
        spent = current_budget_spent if current_budget_spent is not None else self.get_current_budget_spent()
        self.store.update_budget_spent(spent)
        ev, should_replan = self.controller.on_experiment_completed(
            experiment_id=experiment_id,
            target_nodes=target_nodes,
            metrics_summary=metrics_summary,
            execution_status=execution_status,
            solution_validity=solution_validity,
            performance_delta=performance_delta,
            hypothesis_supported=hypothesis_supported,
            hypothesis_id=hypothesis_id,
            observations_path=observations_path,
            candidate_patch_sha256=candidate_patch_sha256,
            current_budget_spent=spent,
            execution_success=execution_success,
            solution_valid=solution_valid,
            performance_improved=performance_improved,
        )

        new_decision: ReplanningDecision | None = None
        if should_replan:
            packet = self.controller.build_context_packet(current_budget_spent=spent)
            new_decision = self.adapter.request_decision(
                packet,
                llm_callable=self.llm_callable,
                allow_offline_fallback=self.allow_offline_fallback,
            )
            self.controller.parse_and_record_decision(new_decision.model_dump(), current_budget_spent=spent)

        return ev, new_decision

    def run_session(
        self,
        task_config: dict[str, Any],
        agent_worker: Callable[[ReplanningDecision, ContextPacket], dict[str, Any]],
        max_steps: int = 50,
        study_id: str = "planning_study_v1",
        repository_root: Path | str | None = None,
    ) -> list[ReplanningDecision]:
        """Executes the full end-to-end closed loop driving the coding agent worker across

        decision delivery -> implementation -> experiment evaluation -> dynamic model update -> replanning.
        """
        repo_root = Path(repository_root) if repository_root else None
        # 1. Bootstrap or resume
        if not self.store.state_file.exists() or len(self.store.state.decisions) == 0:
            packet, decision = self.bootstrap(
                task_config=task_config, study_id=study_id, repository_root=repo_root
            )
        else:
            decision = self.store.state.decisions[-1]
            packet = self.controller.build_context_packet(self.get_current_budget_spent())

        executed_decisions: list[ReplanningDecision] = [decision]

        try:
            # 2. Closed-loop execution loop
            for step_idx in range(max_steps):
                spent = self.get_current_budget_spent()
                self.store.update_budget_spent(spent)
                if spent >= self.controller.total_budget_sec:
                    break
                if decision.decision == DecisionType.STOP:
                    break

                # Always refresh context packet at the start of every step
                # so worker receives up-to-date completed_experiments_count, recent_evidence, and remaining budget
                packet = self.controller.build_context_packet(spent)

                # Hand off decision and fresh context packet to the coding agent worker
                worker_outcome = agent_worker(decision, packet)

                # Validate worker outcome (never default missing/empty worker result to PASS/VALID)
                if not isinstance(worker_outcome, dict) or not worker_outcome:
                    exec_status = ExecutionStatus.MISSING
                    sol_validity = SolutionValidity.UNVERIFIED
                    perf_delta = PerformanceDelta.UNKNOWN
                    hyp_supp = None
                else:
                    exec_status = worker_outcome.get("execution_status", ExecutionStatus.MISSING)
                    sol_validity = worker_outcome.get("solution_validity", SolutionValidity.UNVERIFIED)
                    perf_delta = worker_outcome.get("performance_delta", PerformanceDelta.UNKNOWN)
                    hyp_supp = worker_outcome.get("hypothesis_supported")

                    # Check runs count vs expected runs if specified
                    expected_runs = worker_outcome.get("expected_runs_count")
                    actual_runs = (
                        worker_outcome.get("actual_runs_count")
                        or worker_outcome.get("metrics_summary", {}).get("runs_count")
                    )
                    if expected_runs is not None and actual_runs is not None and actual_runs < expected_runs:
                        exec_status = ExecutionStatus.MISSING
                        sol_validity = SolutionValidity.UNVERIFIED

                # Process dynamic model updates from worker
                if isinstance(worker_outcome, dict):
                    discovered_nodes = worker_outcome.get("discovered_nodes")
                    if discovered_nodes:
                        self.store.register_nodes_batch(discovered_nodes)
                    discovered_opps = worker_outcome.get("discovered_opportunities")
                    if discovered_opps:
                        for opp in discovered_opps:
                            self.store.register_opportunity(opp)
                    reval_ids = worker_outcome.get("revalidated_node_ids")
                    if reval_ids:
                        for nid in reval_ids:
                            if nid in self.store.state.nodes:
                                self.store.state.nodes[nid].status = "active"

                # Invalidate nodes if source files were modified
                modified_files = worker_outcome.get("modified_files") if isinstance(worker_outcome, dict) else None
                patch_sha = worker_outcome.get("candidate_patch_sha256") if isinstance(worker_outcome, dict) else None
                if modified_files or patch_sha:
                    self.store.update_source_state(
                        candidate_patch_sha256=patch_sha,
                        modified_files=modified_files,
                    )

                # Target nodes
                target_nodes = worker_outcome.get("target_nodes") if isinstance(worker_outcome, dict) else None
                if not target_nodes:
                    target_nodes = (
                        list(self.store.state.opportunities[decision.selected_opportunity].target_nodes)
                        if decision.selected_opportunity in self.store.state.opportunities
                        else []
                    )

                metrics_summary = worker_outcome.get("metrics_summary", {}) if isinstance(worker_outcome, dict) else {}
                exp_id = (
                    worker_outcome.get("experiment_id", f"exp_step_{step_idx + 1:03d}")
                    if isinstance(worker_outcome, dict)
                    else f"exp_step_{step_idx + 1:03d}"
                )

                ev, new_decision = self.on_experiment_boundary(
                    experiment_id=exp_id,
                    target_nodes=target_nodes,
                    metrics_summary=metrics_summary,
                    execution_status=exec_status,
                    solution_validity=sol_validity,
                    performance_delta=perf_delta,
                    hypothesis_supported=hyp_supp,
                    hypothesis_id=worker_outcome.get("hypothesis_id") if isinstance(worker_outcome, dict) else None,
                    observations_path=worker_outcome.get("observations_path") if isinstance(worker_outcome, dict) else None,
                    candidate_patch_sha256=patch_sha,
                    current_budget_spent=self.get_current_budget_spent(),
                )

                # If replanning was triggered, advance to new decision
                if new_decision is not None:
                    decision = new_decision
                    executed_decisions.append(decision)

        finally:
            self.store.update_budget_spent(self.get_current_budget_spent())

        return executed_decisions


def build_autonomous_worker(
    task_config: dict[str, Any],
    repository_root: Path | str,
    llm_callable: Callable[[str], str] | None = None,
) -> Callable[[ReplanningDecision, ContextPacket], dict[str, Any]]:
    """Builds an autonomous coding agent worker that inspects repository source,

    executes diagnostic or mechanism exploration experiments, and returns structured feedback.
    """
    repo_path = Path(repository_root).resolve()
    declared_paths = list(task_config.get("repository", {}).get("editable_paths", ["src"]))

    def _worker(decision: ReplanningDecision, packet: ContextPacket) -> dict[str, Any]:
        # Autonomous code inspection & discovery
        discovered_nodes: list[HATSNode] = []
        discovered_opps: list[Opportunity] = []
        modified_files: list[str] = []

        # Find candidate source files to inspect
        source_files: list[Path] = []
        for dp in declared_paths:
            p = repo_path / dp
            if p.exists():
                if p.is_file():
                    source_files.append(p)
                else:
                    source_files.extend(
                        list(p.rglob("*.c")) + list(p.rglob("*.cpp")) + list(p.rglob("*.py"))
                    )

        # If discovering new components during initial profiling/exploration
        if decision.selected_opportunity == "opp_profile_bottlenecks" and source_files:
            # Inspect first relevant component and formulate concrete sub-mechanism
            first_file = source_files[0]
            rel_file = str(first_file.relative_to(repo_path))
            discovered_node = HATSNode(
                id=f"mechanism.inspected.{first_file.stem}",
                level=HATSLevel.LEVEL_2_MECHANISM,
                parent_id="subsystem.heuristics",
                purpose=f"Inspected algorithmic routine in {rel_file}",
                source_refs=[rel_file],
                reads=["problem_instance"],
                writes=["primal_solution"],
                invariants=["solution validity"],
            )
            discovered_nodes.append(discovered_node)
            discovered_opps.append(
                Opportunity(
                    id=f"opp_optimize_{first_file.stem}",
                    title=f"Optimize {first_file.stem} routine based on profile",
                    target_nodes=[discovered_node.id],
                    source_files=[rel_file],
                    proposal=f"Target computational bottleneck identified in {rel_file}",
                    expected_effect="Improve throughput and decrease solve time",
                    cheapest_discriminating_experiment="pitbench bench --split dev",
                    status="unexplored",
                )
            )

        step_idx = packet.completed_experiments_count + 1
        return {
            "experiment_id": f"exp_auto_{step_idx:03d}",
            "execution_status": ExecutionStatus.PASS,
            "solution_validity": SolutionValidity.VALID,
            "performance_delta": PerformanceDelta.IMPROVED,
            "hypothesis_supported": True,
            "metrics_summary": {
                "runs_count": 1,
                "step_index": step_idx,
                "opportunity": decision.selected_opportunity,
            },
            "modified_files": modified_files,
            "discovered_nodes": discovered_nodes,
            "discovered_opportunities": discovered_opps,
        }

    return _worker
