"""End-to-end planning session driver coordinating bootstrap, triggers, and agent continuation."""

from __future__ import annotations

import time
from typing import Any, Callable

from global_planner.adapters.codex import CodexPlannerAdapter
from global_planner.bootstrap import bootstrap_hats_model
from global_planner.controller import PlanningController
from global_planner.schema import (
    ContextPacket,
    DecisionType,
    ExecutionStatus,
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
    ) -> tuple[ContextPacket, ReplanningDecision]:
        """Bootstraps initial HATS model, opportunities, and initial decision from task metadata."""
        task_id = str(task_config.get("task_id", "unknown_task"))
        solver = str(task_config.get("release", {}).get("repository", "solver")).split("/")[-1]
        family = str(task_config.get("problem_family", "general"))
        protocol = str(task_config.get("evaluation", {}).get("performance_protocol", "exact_verified_solve"))

        goal_stmt, nodes, opps = bootstrap_hats_model(task_config)
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
    ) -> list[ReplanningDecision]:
        """Executes the full end-to-end closed loop driving the coding agent worker across

        decision delivery -> implementation -> experiment evaluation -> replanning triggers.
        """
        # 1. Bootstrap or resume
        if not self.store.state_file.exists() or len(self.store.state.decisions) == 0:
            packet, decision = self.bootstrap(task_config=task_config, study_id=study_id)
        else:
            decision = self.store.state.decisions[-1]
            packet = self.controller.build_context_packet(self.get_current_budget_spent())

        executed_decisions: list[ReplanningDecision] = [decision]

        # 2. Closed-loop execution loop
        for step_idx in range(max_steps):
            spent = self.get_current_budget_spent()
            if spent >= self.controller.total_budget_sec:
                break
            if decision.decision == DecisionType.STOP:
                break

            # Hand off decision to the coding agent worker
            worker_outcome = agent_worker(decision, packet)

            # Invalidate nodes if source files were modified
            modified_files = worker_outcome.get("modified_files")
            patch_sha = worker_outcome.get("candidate_patch_sha256")
            if modified_files or patch_sha:
                self.store.update_source_state(
                    candidate_patch_sha256=patch_sha,
                    modified_files=modified_files,
                )

            # Target nodes
            target_nodes = worker_outcome.get("target_nodes")
            if not target_nodes:
                target_nodes = (
                    list(self.store.state.opportunities[decision.selected_opportunity].target_nodes)
                    if decision.selected_opportunity in self.store.state.opportunities
                    else []
                )

            ev, new_decision = self.on_experiment_boundary(
                experiment_id=worker_outcome.get("experiment_id", f"exp_step_{step_idx + 1:03d}"),
                target_nodes=target_nodes,
                metrics_summary=worker_outcome.get("metrics_summary", {}),
                execution_status=worker_outcome.get("execution_status", ExecutionStatus.PASS),
                solution_validity=worker_outcome.get("solution_validity", SolutionValidity.VALID),
                performance_delta=worker_outcome.get("performance_delta", PerformanceDelta.UNKNOWN),
                hypothesis_supported=worker_outcome.get("hypothesis_supported"),
                hypothesis_id=worker_outcome.get("hypothesis_id"),
                observations_path=worker_outcome.get("observations_path"),
                candidate_patch_sha256=patch_sha,
                current_budget_spent=self.get_current_budget_spent(),
            )

            # If replanning was triggered, advance to new decision
            if new_decision is not None:
                decision = new_decision
                packet = self.controller.build_context_packet(self.get_current_budget_spent())
                executed_decisions.append(decision)

        return executed_decisions
