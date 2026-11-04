"""Algorithm-space planning inside the existing PitBench Agent session.

The Agent supplies partial models and mechanism changes. This module checks their
connections and retains the models, decisions, and public experiment observations;
it does not invent solver semantics or choose optimization strategies.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from pitbench.schema.planning import (
    AlgorithmModel,
    Claim,
    Evidence,
    MechanismProposal,
    PlanningDecision,
    PlanningState,
    SourceVersion,
)


class AlgorithmPlanner:
    def __init__(self, path: Path, task_id: str, objective: str):
        self.path = path
        self.state = (
            PlanningState.model_validate_json(path.read_text())
            if path.exists() else PlanningState(task_id=task_id, objective=objective)
        )
        if self.state.task_id != task_id:
            raise ValueError("planning state belongs to a different task")

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=self.path.parent, delete=False) as output:
            temporary = Path(output.name)
            try:
                output.write(self.state.model_dump_json(indent=2) + "\n")
                output.flush()
                os.fsync(output.fileno())
                output.close()
                os.replace(temporary, self.path)
            finally:
                temporary.unlink(missing_ok=True)

    def _validate_claim(self, claim: Claim, source: SourceVersion) -> None:
        if set(claim.evidence_refs) - self.state.evidence.keys():
            raise ValueError("claim references unknown evidence")
        if claim.status in ("observed", "refuted"):
            if not claim.evidence_refs:
                raise ValueError("observed/refuted claims require evidence")
            if any(self.state.evidence[ref].source != source for ref in claim.evidence_refs):
                raise ValueError("claim evidence must be reviewed for the model's source version")

    def update_model(self, model: AlgorithmModel, source: SourceVersion) -> None:
        if model.source != source:
            raise ValueError("model must describe the current source version")
        claims = [*model.nodes, *model.state, *model.actions, *model.invariants, *model.dependencies]
        for transition in model.transitions:
            claims.extend([transition, *transition.costs, *transition.progress])
        for claim in claims:
            self._validate_claim(claim, source)
        for dependency in model.dependencies:
            if dependency.kind == "performance" and dependency.status in ("observed", "refuted"):
                records = [self.state.evidence[ref] for ref in dependency.evidence_refs]
                if not records or any(
                    record.kind != "experiment" or record.execution_status != "pass"
                    or record.solution_validity != "valid" for record in records
                ):
                    raise ValueError("performance influence needs valid experimental evidence")
        self.state.models.append(model.model_copy(deep=True))
        self._save()

    def add_evidence(self, evidence: Evidence, source: SourceVersion) -> None:
        if evidence.kind == "experiment":
            raise ValueError("experimental evidence must come from pitbench bench")
        if evidence.id in self.state.evidence:
            raise ValueError("evidence id already exists")
        if evidence.source != source:
            raise ValueError("evidence must be bound to the current source")
        self.state.evidence[evidence.id] = evidence.model_copy(deep=True)
        self._save()

    @staticmethod
    def _changed_model(model: AlgorithmModel, proposals: list[MechanismProposal]) -> AlgorithmModel:
        targets = [ref for proposal in proposals for ref in proposal.target_transitions]
        if len(set(targets)) != len(targets):
            raise ValueError("combine overlapping transitions in one explicit mechanism proposal")
        candidate = model.model_dump()
        for name, update_field in (
            ("state", "state_updates"), ("actions", "action_updates"),
            ("transitions", "replacement_transitions"),
        ):
            items = {item["id"]: item for item in candidate[name]}
            if name == "transitions":
                items = {key: value for key, value in items.items() if key not in targets}
            changes = {}
            for proposal in proposals:
                for item in getattr(proposal, update_field):
                    value = item.model_dump()
                    if item.id in changes and changes[item.id] != value:
                        raise ValueError("selected mechanisms contain conflicting state/action/transition changes")
                    changes[item.id] = value
            items.update(changes)
            candidate[name] = list(items.values())
        return AlgorithmModel.model_validate(candidate)

    def add_proposal(self, proposal: MechanismProposal) -> None:
        if not self.state.models or proposal.model_revision != len(self.state.models):
            raise ValueError("proposal must reference the current model revision")
        if proposal.id in self.state.proposals:
            raise ValueError("proposal id already exists; retain earlier proposals as history")
        for ref in proposal.depends_on:
            dependency = self.state.proposals.get(ref)
            if dependency is None or dependency.model_revision != proposal.model_revision:
                raise ValueError("proposal dependency must exist in the same model revision")
        model = self.state.models[-1]
        transitions = {transition.id: transition for transition in model.transitions}
        if set(proposal.target_transitions) - transitions.keys():
            raise ValueError("proposal must change declared algorithm transitions")
        replacements = [transition.id for transition in proposal.replacement_transitions]
        if len(set(replacements)) != len(replacements):
            raise ValueError("replacement transition ids must be unique")
        if set(replacements) & (transitions.keys() - set(proposal.target_transitions)):
            raise ValueError("replacement would overwrite an untargeted transition")
        invariant_ids = set(proposal.invariant_ids)
        for ref in proposal.target_transitions:
            invariant_ids.update(transitions[ref].invariant_ids)
        for replacement in proposal.replacement_transitions:
            if not invariant_ids.issubset(replacement.invariant_ids):
                raise ValueError("mechanism change must retain the correctness invariants")
            if not replacement.implementation_nodes:
                raise ValueError("mechanism change needs a concrete Level 3 implementation")
            if replacement.status not in ("unknown", "hypothesis"):
                raise ValueError("proposed transitions are hypotheses, not observed results")
        # Validate the proposed A' with the same structural rules as A.
        self._changed_model(model, [proposal])
        claims = [*proposal.state_updates, *proposal.action_updates]
        for transition in proposal.replacement_transitions:
            claims.extend([transition, *transition.costs, *transition.progress])
        for claim in claims:
            self._validate_claim(claim, model.source)
        self.state.proposals[proposal.id] = proposal.model_copy(deep=True)
        self._save()

    def decide(self, decision: PlanningDecision, source: SourceVersion) -> None:
        if not self.state.models or decision.model_revision != len(self.state.models):
            raise ValueError("decision must use the current model revision")
        if self.state.models[-1].source != source:
            raise ValueError("review and update the model for the current source before replanning")
        proposals = {
            key: value for key, value in self.state.proposals.items()
            if value.model_revision == decision.model_revision
        }
        selected = set(decision.selected_proposals)
        if len(selected) != len(decision.selected_proposals) or selected - proposals.keys():
            raise ValueError("selected proposals must be distinct and in the current model")
        alternatives = {alternative.proposal_id for alternative in decision.alternatives}
        if alternatives != proposals.keys() - selected:
            raise ValueError("compare and explain the other current mechanism proposals")
        for key in selected:
            if set(proposals[key].depends_on) - selected:
                raise ValueError("select required mechanism dependencies together")
        self._changed_model(self.state.models[-1], [proposals[key] for key in decision.selected_proposals])
        if set(decision.evidence_refs) - self.state.evidence.keys():
            raise ValueError("decision references unknown evidence")
        if set(self.state.pending_evidence) - set(decision.evidence_refs):
            raise ValueError("replanning must address the pending experiment observations")
        self.state.decisions.append(decision.model_copy(deep=True))
        self.state.pending_evidence.clear()
        self._save()

    def require_decision(self) -> PlanningDecision:
        if not self.state.decisions or self.state.pending_evidence:
            raise ValueError("use pitbench plan to compare mechanisms and review evidence before the next candidate experiment")
        decision = self.state.decisions[-1]
        if decision.model_revision != len(self.state.models) or not decision.selected_proposals:
            raise ValueError("select a mechanism change in the current algorithm model first")
        return decision

    def record_experiment(
        self, evidence: Evidence, *, baseline: bool = False
    ) -> Evidence:
        if evidence.kind != "experiment" or evidence.id in self.state.evidence:
            raise ValueError("experiment evidence needs a new id")
        if not baseline:
            decision = self.require_decision()
            evidence.proposal_ids = list(decision.selected_proposals)
        affected = set()
        if self.state.models:
            model = self.state.models[-1]
            nodes = {node.id: node for node in model.nodes}
            actions = {action.id: action for action in model.actions}
            transitions = {transition.id: transition for transition in model.transitions}
            for key in evidence.proposal_ids:
                for ref in self.state.proposals[key].target_transitions:
                    transition = transitions[ref]
                    affected.add(actions[transition.action_id].node_id)
                    affected.update(transition.implementation_nodes)
            for key in list(affected):
                while nodes[key].parent_id is not None:
                    key = nodes[key].parent_id
                    affected.add(key)
        evidence.affected_nodes = sorted(affected)
        self.state.evidence[evidence.id] = evidence.model_copy(deep=True)
        self.state.pending_evidence.append(evidence.id)
        self._save()
        return evidence

    def context(self, source: SourceVersion) -> dict:
        model = self.state.models[-1] if self.state.models else None
        return {
            "task_id": self.state.task_id,
            "objective": self.state.objective,
            "source": source.model_dump(),
            "model_revision": len(self.state.models),
            "model_needs_review": model is not None and model.source != source,
            "model": model.model_dump() if model else None,
            "proposals": [
                proposal.model_dump() for proposal in self.state.proposals.values()
                if proposal.model_revision == len(self.state.models)
            ],
            "decision": self.state.decisions[-1].model_dump() if self.state.decisions else None,
            "evidence": [
                record.model_dump(exclude={"observations"}) for record in self.state.evidence.values()
            ],
            "observations_to_review": [
                self.state.evidence[key].model_dump() for key in self.state.pending_evidence
            ],
        }


PLANNING_INSTRUCTION = """### Algorithm-Space Global Planning
Use the same Agent and development budget to plan changes to algorithm mechanisms.
Keep planning JSON inputs in the public runs directory, outside implementation source.
After the public baseline, run `pitbench plan --schema` and construct a partial
model from source inspection, development traces, and domain knowledge. The model
must connect Level 0 objectives, Level 1 subsystems, Level 2 mechanisms, and Level 3
implementation references. Declare abstract state S, actions U, transitions T,
resource costs C, progress signals R, and correctness invariants I. Record hidden
state and unknown transitions explicitly. Source/control dependencies and predicted
performance influence are different claims; generated models are not proofs.
Register source/trace evidence with `pitbench plan --evidence FILE`, submit the model
with `--model FILE`, and submit alternative mechanism changes with `--proposal FILE`.
Proposals describe transition replacements (including state/action changes), their
global effects and dependencies, concrete source changes, and falsifiable paired
experiments. Compare proposals across the known subsystems using `--decision FILE`.
An empty selection records a reasoned stop. Staying with the same mechanism is valid.
Implement the selected change and use the existing `pitbench bench` for measurement
and independent verification. Bench preserves raw public results and maps affected
mechanisms back to their subsystem and global ancestors. Then inspect `pitbench plan`,
revise model claims using those observations, and record a new global decision before
another candidate experiment. Performance improvement does not automatically prove
a mechanism hypothesis. Keep all failures and missing observations in the record.
"""
