"""Persistent, versioned storage for Global Planning state and evidence."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from global_planner.schema import (
    EvidenceRecord,
    GlobalPlanState,
    HATSNode,
    Opportunity,
    ReplanningDecision,
)


def match_source_ref(file_path: str, ref: str) -> bool:
    """Matches a modified file path against a node source reference.

    Handles:
    - Path equality (e.g. 'src/solver/main.c')
    - Directory / prefix containment (e.g. ref='pyvrp/search' matches 'pyvrp/search/LocalSearch.cpp')
    - Wildcards and globs (e.g. ref='src/scip/heur_*.c' matches 'src/scip/heur_oneopt.c')
    """
    p = PurePosixPath(file_path.strip().lstrip("/"))
    r = PurePosixPath(ref.strip().lstrip("/"))

    # Glob pattern in reference
    if any(c in ref for c in ("*", "?", "[", "]")):
        if fnmatch.fnmatch(str(p), str(r)):
            return True
        if fnmatch.fnmatch(p.name, str(r)):
            return True

    # Exact match
    if p == r:
        return True

    # Prefix directory containment (file is inside ref directory)
    try:
        p.relative_to(r)
        return True
    except ValueError:
        pass

    return False


class PlanStore:
    """Manages versioned persistence of an algorithm-space plan."""

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

    def initialize(
        self,
        study_id: str,
        task_id: str,
        solver: str,
        goal_statement: str = "Improve solver performance within budget",
        problem_family: str | None = None,
        performance_protocol: str | None = None,
        budget_spent_sec: float = 0.0,
    ) -> GlobalPlanState:
        """Create or re-initialize a new plan state, preserving accumulated budget on restarts."""
        accumulated_budget = budget_spent_sec
        if self._state is not None:
            accumulated_budget = max(accumulated_budget, self._state.budget_spent_sec)
        elif self.state_file.exists():
            try:
                existing = self.load()
                accumulated_budget = max(accumulated_budget, existing.budget_spent_sec)
            except Exception:
                pass

        self._state = GlobalPlanState(
            study_id=study_id,
            task_id=task_id,
            solver=solver,
            goal_statement=goal_statement,
            problem_family=problem_family,
            performance_protocol=performance_protocol,
            revision=0,
            budget_spent_sec=accumulated_budget,
            budget_clock_started_at_utc=datetime.now(timezone.utc).isoformat(),
            last_replanned_at_sec=accumulated_budget,
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

    def register_node(self, node: HATSNode | dict[str, Any]) -> None:
        """Register or update a HATS transition node."""
        st = self.state
        node_obj = HATSNode.model_validate(node) if isinstance(node, dict) else node
        st.nodes[node_obj.id] = node_obj
        self.save()

    def register_nodes_batch(self, nodes: list[HATSNode | dict[str, Any]]) -> None:
        st = self.state
        for n in nodes:
            node_obj = HATSNode.model_validate(n) if isinstance(n, dict) else n
            st.nodes[node_obj.id] = node_obj
        self.save()

    def register_opportunity(self, opportunity: Opportunity | dict[str, Any]) -> None:
        """Register a new optimization opportunity."""
        st = self.state
        opp = Opportunity.model_validate(opportunity) if isinstance(opportunity, dict) else opportunity
        st.opportunities[opp.id] = opp
        self.save()

    def update_source_state(
        self,
        candidate_patch_sha256: str | None,
        modified_files: list[str] | None = None,
    ) -> list[str]:
        """Checks for patch modifications and marks overlapping model nodes as modified/stale."""
        st = self.state
        st.current_patch_sha256 = candidate_patch_sha256
        invalidated_node_ids = []
        if modified_files:
            for node in st.nodes.values():
                if any(any(match_source_ref(mf, ref) for ref in node.source_refs) for mf in modified_files):
                    node.status = "modified"
                    invalidated_node_ids.append(node.id)
            if invalidated_node_ids:
                self.save()
        return invalidated_node_ids

    def update_budget_spent(self, current_budget_spent: float) -> None:
        """Persist latest elapsed budget spent without waiting for a decision or experiment."""
        st = self.state
        if current_budget_spent > st.budget_spent_sec:
            st.budget_spent_sec = float(current_budget_spent)
            self.save()

    def record_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Append an auditable controller event to the versioned state."""
        from datetime import datetime, timezone

        st = self.state
        st.events.append({
            "event_id": f"event_{len(st.events) + 1:06d}",
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        })
        self.save()

    def bind_source_revision(self, repository_root: Path | str) -> str | None:
        """Bind the active model to the candidate patch currently in the repository."""
        root = Path(repository_root)
        try:
            import subprocess

            tracked = subprocess.run(
                ["git", "diff", "--binary", "--no-ext-diff", "HEAD", "--", ".", ":(exclude).pitbench", ":(exclude).pitbench/**"], cwd=root,
                check=True, capture_output=True,
            )
            patch = bytearray(tracked.stdout)
            untracked = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard"],
                cwd=root, check=True, capture_output=True, text=True,
            )
            untracked_paths = [
                path for path in untracked.stdout.splitlines()
                if path != ".pitbench" and not path.startswith(".pitbench/")
            ]
            for rel_path in untracked_paths:
                result = subprocess.run(
                    ["git", "diff", "--binary", "--no-ext-diff", "--no-index", "--", "/dev/null", rel_path],
                    cwd=root, check=False, capture_output=True,
                )
                if result.returncode not in (0, 1):
                    raise subprocess.CalledProcessError(
                        result.returncode, result.args,
                        output=result.stdout, stderr=result.stderr,
                    )
                patch.extend(result.stdout)
        except (OSError, subprocess.CalledProcessError):
            return None
        digest = hashlib.sha256(patch).hexdigest() if patch else None
        changed = subprocess.run(
            ["git", "diff", "--name-only", "HEAD", "--", ".", ":(exclude).pitbench", ":(exclude).pitbench/**"],
            cwd=root, check=False, capture_output=True, text=True,
        )
        modified_files = changed.stdout.splitlines() if changed.returncode == 0 else []
        modified_files.extend(untracked_paths)
        self.update_source_state(digest, modified_files=modified_files)
        return digest

    def record_evidence(
        self,
        evidence: EvidenceRecord,
        current_budget_spent: float | None = None,
    ) -> None:
        """Record empirical experiment results and update affected node effects and cumulative budget."""
        st = self.state
        if any(existing.id == evidence.id for existing in st.evidence):
            raise ValueError(f"evidence '{evidence.id}' has already been recorded")
        st.evidence.append(evidence)
        st.experiments_count += 1
        if current_budget_spent is not None:
            st.budget_spent_sec = max(st.budget_spent_sec, float(current_budget_spent))

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
        """Record a structured replanning decision, register discovered nodes/opps, and reset review counters."""
        st = self.state

        # Register dynamic model updates introduced in this decision
        for n in decision.discovered_nodes:
            st.nodes[n.id] = n
        for opp in decision.discovered_opportunities:
            st.opportunities[opp.id] = opp
        for nid in decision.revalidated_node_ids:
            if nid in st.nodes:
                st.nodes[nid].status = "active"

        # Validate that selected opportunity exists
        if decision.decision != "stop" and decision.selected_opportunity not in st.opportunities:
            raise ValueError(
                f"Selected opportunity '{decision.selected_opportunity}' does not exist in state opportunities "
                f"({sorted(st.opportunities.keys())})"
            )

        st.decisions.append(decision)
        st.active_opportunity_id = decision.selected_opportunity
        st.last_replanned_at_sec = current_budget_spent
        st.budget_spent_sec = max(st.budget_spent_sec, current_budget_spent)
        st.experiments_count_at_last_review = st.experiments_count

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
