"""Partial algorithm models and concrete mechanism changes used by the Agent."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceVersion(Record):
    base_commit: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    patch_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Claim(Record):
    status: Literal["unknown", "hypothesis", "observed", "refuted"] = "unknown"
    evidence_refs: list[str] = Field(default_factory=list)


class AlgorithmNode(Claim):
    id: Text
    level: Literal[0, 1, 2, 3]
    parent_id: str | None = None
    purpose: Text
    source_refs: list[str] = Field(default_factory=list)


class StateVariable(Claim):
    id: Text
    node_id: Text
    description: Text
    domain: str | None = None
    hidden: bool = False


class Action(Claim):
    id: Text
    node_id: Text
    description: Text
    preconditions: list[str] = Field(default_factory=list)


class Invariant(Claim):
    id: Text
    description: Text


class ResourceCost(Claim):
    resource: Text
    description: Text


class ProgressSignal(Claim):
    metric: Text
    direction: Literal["minimize", "maximize"]
    description: Text


class Transition(Claim):
    id: Text
    action_id: Text
    reads: list[str] = Field(default_factory=list)
    writes: list[str] = Field(default_factory=list)
    relation: str | None = None
    stochastic: bool | None = None
    costs: list[ResourceCost] = Field(default_factory=list)
    progress: list[ProgressSignal] = Field(default_factory=list)
    invariant_ids: list[str] = Field(default_factory=list)
    implementation_nodes: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)


class Dependency(Claim):
    source: Text
    target: Text
    kind: Literal["control", "data", "performance"]
    description: Text


class AlgorithmModel(Record):
    source: SourceVersion
    nodes: list[AlgorithmNode] = Field(min_length=1)
    state: list[StateVariable] = Field(default_factory=list)
    actions: list[Action] = Field(default_factory=list)
    transitions: list[Transition] = Field(default_factory=list)
    invariants: list[Invariant] = Field(default_factory=list)
    dependencies: list[Dependency] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_structure(self):
        for name in ("nodes", "state", "actions", "transitions", "invariants"):
            values = getattr(self, name)
            if len({value.id for value in values}) != len(values):
                raise ValueError(f"duplicate ids in {name}")
        nodes = {node.id: node for node in self.nodes}
        roots = [node for node in self.nodes if node.level == 0]
        if len(roots) != 1 or roots[0].parent_id is not None:
            raise ValueError("model requires one Level 0 root")
        for node in self.nodes:
            if node.level > 0:
                parent = nodes.get(node.parent_id)
                if parent is None or parent.level != node.level - 1:
                    raise ValueError(f"{node.id}: parent must be in the preceding level")
            if node.level == 3 and not node.source_refs:
                raise ValueError(f"{node.id}: Level 3 requires concrete source references")
        for variable in self.state:
            if variable.node_id not in nodes:
                raise ValueError(f"unknown state owner: {variable.node_id}")
        for action in self.actions:
            if action.node_id not in nodes or nodes[action.node_id].level != 2:
                raise ValueError(f"{action.id}: action must belong to a Level 2 mechanism")
        actions = {action.id for action in self.actions}
        variables = {variable.id for variable in self.state}
        invariants = {invariant.id for invariant in self.invariants}
        for transition in self.transitions:
            if transition.action_id not in actions:
                raise ValueError(f"unknown action: {transition.action_id}")
            if set(transition.reads + transition.writes) - variables:
                raise ValueError(f"{transition.id}: undeclared state variables")
            if set(transition.invariant_ids) - invariants:
                raise ValueError(f"{transition.id}: undeclared correctness invariant")
            if any(ref not in nodes or nodes[ref].level != 3 for ref in transition.implementation_nodes):
                raise ValueError(f"{transition.id}: implementation references must be Level 3 nodes")
        for dependency in self.dependencies:
            if dependency.source not in nodes or dependency.target not in nodes:
                raise ValueError("dependency references an unknown algorithm node")
        return self


class ExperimentPlan(Record):
    control: Text
    measurements: list[Text] = Field(min_length=1)
    falsifies_if: Text


class MechanismProposal(Record):
    id: Text
    model_revision: int = Field(ge=1)
    target_transitions: list[Text] = Field(min_length=1)
    replacement_transitions: list[Transition] = Field(min_length=1)
    state_updates: list[StateVariable] = Field(default_factory=list)
    action_updates: list[Action] = Field(default_factory=list)
    invariant_ids: list[Text] = Field(min_length=1)
    expected_global_effect: Text
    implementation_change: Text
    experiment: ExperimentPlan
    depends_on: list[str] = Field(default_factory=list)


class Alternative(Record):
    proposal_id: Text
    defer_reason: Text


class PlanningDecision(Record):
    model_revision: int = Field(ge=1)
    selected_proposals: list[str]
    global_reason: Text
    alternatives: list[Alternative] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)


class Evidence(Record):
    id: Text
    kind: Literal["source", "trace", "experiment"]
    source: SourceVersion
    summary: Text
    artifact_paths: list[str] = Field(default_factory=list)
    workload: dict[str, Any] = Field(default_factory=dict)
    observations: list[dict[str, Any]] = Field(default_factory=list)
    expected_runs: int | None = Field(default=None, ge=1)
    execution_status: Literal["pass", "fail", "missing"] = "missing"
    solution_validity: Literal["valid", "invalid", "unverified"] = "unverified"
    proposal_ids: list[str] = Field(default_factory=list)
    affected_nodes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_measurement(self):
        if self.kind == "experiment" and (
            self.expected_runs is None or len(self.observations) < self.expected_runs
        ):
            self.execution_status = "missing"
            self.solution_validity = "unverified"
        return self


class PlanningState(Record):
    task_id: Text
    objective: Text
    models: list[AlgorithmModel] = Field(default_factory=list)
    proposals: dict[str, MechanismProposal] = Field(default_factory=dict)
    evidence: dict[str, Evidence] = Field(default_factory=dict)
    decisions: list[PlanningDecision] = Field(default_factory=list)
    pending_evidence: list[str] = Field(default_factory=list)
