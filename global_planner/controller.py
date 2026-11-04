"""Event-driven Planning Controller with persistent state and experiment-boundary gating."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from global_planner.schema import (
    ContextPacket,
    DecisionType,
    EvidenceRecord,
    GlobalPlanState,
    HATSNode,
    Opportunity,
    ReplanningDecision,
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
        initial_nodes: list[HATSNode] | None = None,
        initial_opportunities: list[Opportunity] | None = None,
    ) -> bool:
        """Called when a development task starts.

        Initializes store if needed, registers initial skeletal nodes,
        and returns True if an initial planning bootstrap is required.
        """
        if not self.store.state_file.exists():
            self.store.initialize(study_id=study_id, task_id=task_id, solver=solver)
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
        is_supported: bool | None,
        hypothesis_id: str | None = None,
        observations_path: str | None = None,
        code_patch_sha256: str | None = None,
        current_budget_spent: float = 0.0,
    ) -> tuple[EvidenceRecord, bool]:
        """Called immediately after an empirical experiment (e.g. `pitbench bench`) completes.

        Records evidence and evaluates whether an algorithm-space replanning event is triggered.
        Returns (evidence, should_replan).
        """
        status_str = "inconclusive"
        if is_supported is True:
            status_str = "supported"
        elif is_supported is False:
            status_str = "refuted"

        evidence = EvidenceRecord(
            id=f"ev_{self.state.experiments_count + 1:03d}",
            experiment_id=experiment_id,
            hypothesis_id=hypothesis_id,
            target_nodes=target_nodes,
            status=status_str,
            metrics_summary=metrics_summary,
            code_patch_sha256=code_patch_sha256,
            observations_path=observations_path,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.store.record_evidence(evidence)

        # Evaluate trigger
        should_replan = self.should_trigger_replanning(
            current_budget_spent=current_budget_spent,
            critical_evidence=(status_str == "refuted"),
        )
        return evidence, should_replan

    def should_trigger_replanning(
        self,
        current_budget_spent: float,
        critical_evidence: bool = False,
    ) -> bool:
        """Determines whether to trigger replanning, respecting cooldowns and budget limits."""
        st = self.state

        # If budget exhausted, do not trigger
        if current_budget_spent >= self.total_budget_sec:
            return False

        # First decision is always triggered if no decisions exist
        if not st.decisions:
            return True

        # Critical evidence (e.g. a core hypothesis is explicitly refuted) triggers faster
        if critical_evidence:
            return True

        # Check experiment count since last replan
        exps_since_replan = st.experiments_count - (
            len([d for d in st.decisions if d.decision != DecisionType.STOP])
        )
        if exps_since_replan >= self.experiments_between_reviews:
            # Check time cooldown
            elapsed_since_replan = current_budget_spent - st.last_replanned_at_sec
            if elapsed_since_replan >= self.cooldown_sec:
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

        # Unexplored opportunities
        unexplored = [
            opp for opp in st.opportunities.values()
            if opp.status in ("unexplored", "deferred")
        ]

        return ContextPacket(
            task_id=st.task_id,
            goal_statement=f"Improve verified solved coverage on {st.solver} under research budget",
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
