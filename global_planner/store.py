"""Persistent, versioned storage for Global Planning state and evidence."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from global_planner.schema import (
    EvidenceRecord,
    GlobalPlanState,
    HATSNode,
    Opportunity,
    ReplanningDecision,
)


class PlanStore:
    """Manages the versioned persistence of an algorithm-space plan."""

    def __init__(self, storage_dir: Path | str):
        self.storage_dir = Path(storage_dir).resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.storage_dir / "plan_state.json"
        self.history_dir = self.storage_dir / "history"
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self._state: GlobalPlanState | None = None

    @property
    def state(self) -> GlobalPlanState:
        if self._state is None:
            if self.state_file.exists():
                self._state = self.load()
            else:
                raise FileNotFoundError(f"No plan state found at {self.state_file}")
        return self._state

    def initialize(self, study_id: str, task_id: str, solver: str) -> GlobalPlanState:
        """Create or re-initialize a new plan state."""
        self._state = GlobalPlanState(
            study_id=study_id,
            task_id=task_id,
            solver=solver,
            revision=0,
        )
        self.save()
        return self._state

    def load(self) -> GlobalPlanState:
        """Load state from disk."""
        data = json.loads(self.state_file.read_text(encoding="utf-8"))
        self._state = GlobalPlanState.model_validate(data)
        return self._state

    def save(self) -> None:
        """Atomically persist state and record historical revision snapshot."""
        if self._state is None:
            return

        self._state.revision += 1
        serialized = self._state.model_dump_json(indent=2)

        # Atomic file write
        temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            dir=self.storage_dir,
            delete=False,
            encoding="utf-8",
        )
        try:
            temp_file.write(serialized)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_file.close()
            os.replace(temp_file.name, self.state_file)
        finally:
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)

        # Snapshot in history
        snapshot_file = self.history_dir / f"plan_rev_{self._state.revision:04d}.json"
        snapshot_file.write_text(serialized, encoding="utf-8")

    def register_node(self, node: HATSNode) -> None:
        """Register or update a HATS transition node."""
        st = self.state
        st.nodes[node.id] = node
        self.save()

    def register_nodes_batch(self, nodes: list[HATSNode]) -> None:
        st = self.state
        for n in nodes:
            st.nodes[n.id] = n
        self.save()

    def register_opportunity(self, opportunity: Opportunity) -> None:
        """Register a new optimization opportunity."""
        st = self.state
        st.opportunities[opportunity.id] = opportunity
        self.save()

    def record_evidence(self, evidence: EvidenceRecord) -> None:
        """Record empirical experiment results and update affected node effects."""
        st = self.state
        st.evidence.append(evidence)
        st.experiments_count += 1

        # Link to active opportunity if present
        if st.active_opportunity_id and st.active_opportunity_id in st.opportunities:
            opp = st.opportunities[st.active_opportunity_id]
            if evidence.id not in opp.evidence_refs:
                opp.evidence_refs.append(evidence.id)
            if evidence.status == "refuted":
                opp.status = "refuted"
            elif evidence.status == "supported":
                opp.status = "validated"

        self.save()

    def record_decision(self, decision: ReplanningDecision, current_budget_spent: float) -> None:
        """Record a structured replanning decision."""
        st = self.state
        st.decisions.append(decision)
        st.active_opportunity_id = decision.selected_opportunity
        st.last_replanned_at_sec = current_budget_spent
        st.budget_spent_sec = current_budget_spent

        # Update opportunity status
        if decision.selected_opportunity in st.opportunities:
            st.opportunities[decision.selected_opportunity].status = "in_progress"

        # Record deferred reasons for alternatives
        for alt in decision.alternatives_considered:
            alt_id = alt.get("id")
            if alt_id and alt_id in st.opportunities:
                st.opportunities[alt_id].status = "deferred"
                st.opportunities[alt_id].defer_reason = alt.get("defer_reason")

        self.save()
