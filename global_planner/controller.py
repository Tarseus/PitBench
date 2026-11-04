"""Event-driven Planning Controller with persistent state and experiment-boundary gating."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from global_planner.schema import (
    ContextPacket,
    EvidenceRecord,
    ExecutionStatus,
    GlobalPlanState,
    HATSLevel,
    HATSNode,
    Opportunity,
    PerformanceDelta,
    ReplanningDecision,
    SolutionValidity,
)
from global_planner.store import PlanStore


class PlanningController:
    """Coordinates algorithmic planning events, triggers replanning at controlled boundaries,

    and enforces cooldowns and budget tracking.
    """

    def __init__(
        self,
        store: PlanStore,
        total_budget_sec: float = 14400.0,
        cooldown_sec: float = 300.0,
        experiments_between_reviews: int = 1,
    ):
        self.store = store
        self.total_budget_sec = total_budget_sec
        self.cooldown_sec = cooldown_sec
        self.experiments_between_reviews = experiments_between_reviews

    @property
    def state(self) -> GlobalPlanState:
        return self.store.state

    def on_task_start(
        self,
        study_id: str,
        task_id: str,
        solver: str,
        goal_statement: str = "Improve solver performance within budget",
        problem_family: str | None = None,
        performance_protocol: str | None = None,
        initial_nodes: list[HATSNode] | None = None,
        initial_opportunities: list[Opportunity] | None = None,
    ) -> bool:
        """Called when a development task starts.

        Initializes store if needed, registers initial skeletal nodes,
        and returns True if an initial planning bootstrap is required.
        """
        if not self.store.state_file.exists():
            self.store.initialize(
                study_id=study_id,
                task_id=task_id,
                solver=solver,
                goal_statement=goal_statement,
                problem_family=problem_family,
                performance_protocol=performance_protocol,
            )
            if initial_nodes:
                self.store.register_nodes_batch(initial_nodes)
            if initial_opportunities:
                for opp in initial_opportunities:
                    self.store.register_opportunity(opp)
            return True
        return len(self.state.decisions) == 0

    def on_experiment_completed(
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
        current_budget_spent: float = 0.0,
        # Backward compatibility keyword arguments
        execution_success: bool | None = None,
        solution_valid: bool | None = None,
        performance_improved: bool | None = None,
    ) -> tuple[EvidenceRecord, bool]:
        """Called immediately after an empirical experiment completes.

        Distinguishes execution success, solution correctness, performance delta vs baseline,
        and whether the mechanism hypothesis was supported.
        Returns (evidence, should_replan).
        """
        if execution_success is not None:
            execution_status = ExecutionStatus.PASS if execution_success else ExecutionStatus.FAIL
        if solution_valid is not None:
            solution_validity = SolutionValidity.VALID if solution_valid else SolutionValidity.INVALID
        if performance_improved is not None:
            if performance_improved is True:
                performance_delta = PerformanceDelta.IMPROVED
            elif performance_improved is False:
                performance_delta = PerformanceDelta.REGRESSED
            else:
                performance_delta = PerformanceDelta.NEUTRAL

        # Disambiguate empirical statuses without conflating validity, performance, and mechanism
        if execution_status == ExecutionStatus.FAIL or solution_validity == SolutionValidity.INVALID:
            status_str = "refuted"
        elif hypothesis_supported is False or performance_delta == PerformanceDelta.REGRESSED:
            status_str = "refuted"
        elif performance_delta == PerformanceDelta.NEUTRAL:
            status_str = "neutral"
        elif performance_delta == PerformanceDelta.IMPROVED and hypothesis_supported is True:
            status_str = "supported"
        else:
            status_str = "inconclusive"

        evidence = EvidenceRecord(
            id=f"ev_{self.state.experiments_count + 1:03d}",
            experiment_id=experiment_id,
            hypothesis_id=hypothesis_id,
            target_nodes=target_nodes,
            execution_status=execution_status,
            solution_validity=solution_validity,
            performance_delta=performance_delta,
            hypothesis_supported=hypothesis_supported,
            status=status_str,
            metrics_summary=metrics_summary,
            candidate_patch_sha256=candidate_patch_sha256,
            observations_path=observations_path,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.store.record_evidence(evidence, current_budget_spent=current_budget_spent)

        # Critical evidence (hypothesis refuted, crash, or regression) overrides cooldown
        is_critical = (status_str == "refuted")
        should_replan = self.should_trigger_replanning(
            current_budget_spent=current_budget_spent,
            critical_evidence=is_critical,
        )
        return evidence, should_replan

    def should_trigger_replanning(
        self,
        current_budget_spent: float,
        critical_evidence: bool = False,
    ) -> bool:
        """Determines whether to trigger replanning, respecting cooldowns and review intervals."""
        st = self.state

        # If budget exhausted, do not trigger
        if current_budget_spent >= self.total_budget_sec:
            return False

        # First decision is always triggered if no decisions exist
        if not st.decisions:
            return True

        # Critical evidence (e.g. hypothesis refuted) triggers immediately
        if critical_evidence:
            return True

        # Check experiment count since last review
        exps_since_last_review = st.experiments_count - st.experiments_count_at_last_review
        if exps_since_last_review >= self.experiments_between_reviews:
            # Check elapsed time cooldown
            elapsed_since_last_review = current_budget_spent - st.last_replanned_at_sec
            if elapsed_since_last_review >= self.cooldown_sec:
                return True

        return False

    def build_context_packet(self, current_budget_spent: float) -> ContextPacket:
        """Constructs a compact context packet for the agent when replanning."""
        st = self.state
        remaining = max(0.0, self.total_budget_sec - current_budget_spent)

        active_opp = None
        if st.active_opportunity_id:
            active_opp = st.opportunities.get(st.active_opportunity_id)

        # Recent evidence (last 3 records)
        recent_ev = st.evidence[-3:] if st.evidence else []

        # Affected nodes
        affected_node_ids = set()
        for ev in recent_ev:
            affected_node_ids.update(ev.target_nodes)
        if active_opp:
            affected_node_ids.update(active_opp.target_nodes)
        affected_nodes = [st.nodes[nid] for nid in affected_node_ids if nid in st.nodes]

        # If no specific nodes singled out (e.g. bootstrap or initial replan),
        # supply global structural view of Level 0 & Level 1 nodes so model never plans blind
        if not affected_nodes:
            affected_nodes = [
                n for n in st.nodes.values()
                if n.level in (HATSLevel.LEVEL_0_GOAL, HATSLevel.LEVEL_1_SUBSYSTEM)
            ]
            if not affected_nodes:
                affected_nodes = list(st.nodes.values())

        # Unexplored opportunities
        unexplored = [
            opp for opp in st.opportunities.values()
            if opp.status in ("unexplored", "deferred")
        ]

        return ContextPacket(
            task_id=st.task_id,
            goal_statement=st.goal_statement,
            remaining_budget_sec=remaining,
            completed_experiments_count=st.experiments_count,
            active_opportunity=active_opp,
            recent_evidence=recent_ev,
            affected_nodes=affected_nodes,
            unexplored_opportunities=unexplored,
        )

    def parse_and_record_decision(
        self,
        decision_payload: dict[str, Any] | str,
        current_budget_spent: float,
    ) -> ReplanningDecision:
        """Validates and persists a structured replanning decision."""
        if isinstance(decision_payload, str):
            data = json.loads(decision_payload)
        else:
            data = dict(decision_payload)

        if not data.get("timestamp"):
            data["timestamp"] = datetime.now(timezone.utc).isoformat()

        decision = ReplanningDecision.model_validate(data)
        self.store.record_decision(decision, current_budget_spent=current_budget_spent)
        return decision
