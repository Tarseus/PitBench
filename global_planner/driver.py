"""End-to-end planning session driver coordinating bootstrap, triggers, and agent continuation."""

from __future__ import annotations

import time
from typing import Any, Callable

from global_planner.adapters.codex import CodexPlannerAdapter
from global_planner.bootstrap import bootstrap_hats_model
from global_planner.controller import PlanningController
from global_planner.schema import (
    ContextPacket,
    ReplanningDecision,
)
from global_planner.store import PlanStore


class PlanningSessionDriver:
    """Manages the end-to-end planning loop across task bootstrap, experiment boundaries,

    and decision delivery.
    """

    def __init__(
        self,
        store: PlanStore,
        total_budget_sec: float = 14400.0,
        cooldown_sec: float = 300.0,
        experiments_between_reviews: int = 1,
        adapter: CodexPlannerAdapter | None = None,
        llm_callable: Callable[[str], str] | None = None,
    ):
        self.store = store
        self.controller = PlanningController(
            store=store,
            total_budget_sec=total_budget_sec,
            cooldown_sec=cooldown_sec,
            experiments_between_reviews=experiments_between_reviews,
        )
        self.adapter = adapter or CodexPlannerAdapter()
        self.llm_callable = llm_callable
        self.start_wall_time = time.time()

    def get_current_budget_spent(self) -> float:
        """Returns elapsed wall time since driver start, or records in state."""
        return max(0.0, time.time() - self.start_wall_time)

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

        packet = self.controller.build_context_packet(current_budget_spent=0.0)
        decision = self.adapter.request_decision(packet, llm_callable=self.llm_callable)
        self.controller.parse_and_record_decision(decision.model_dump(), current_budget_spent=0.0)
        return packet, decision

    def on_experiment_boundary(
        self,
        experiment_id: str,
        target_nodes: list[str],
        metrics_summary: dict[str, Any],
        execution_success: bool = True,
        solution_valid: bool = True,
        performance_improved: bool | None = None,
        hypothesis_supported: bool | None = None,
        hypothesis_id: str | None = None,
        observations_path: str | None = None,
        candidate_patch_sha256: str | None = None,
        current_budget_spent: float | None = None,
    ) -> tuple[Any, ReplanningDecision | None]:
        """Processes an experiment completion, records evidence, and executes replanning if triggered."""
        spent = current_budget_spent if current_budget_spent is not None else self.get_current_budget_spent()
        ev, should_replan = self.controller.on_experiment_completed(
            experiment_id=experiment_id,
            target_nodes=target_nodes,
            metrics_summary=metrics_summary,
            execution_success=execution_success,
            solution_valid=solution_valid,
            performance_improved=performance_improved,
            hypothesis_supported=hypothesis_supported,
            hypothesis_id=hypothesis_id,
            observations_path=observations_path,
            candidate_patch_sha256=candidate_patch_sha256,
            current_budget_spent=spent,
        )

        new_decision: ReplanningDecision | None = None
        if should_replan:
            packet = self.controller.build_context_packet(current_budget_spent=spent)
            new_decision = self.adapter.request_decision(packet, llm_callable=self.llm_callable)
            self.controller.parse_and_record_decision(new_decision.model_dump(), current_budget_spent=spent)

        return ev, new_decision
