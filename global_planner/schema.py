"""Hierarchical Algorithmic Transition System (HATS) and Global Planning Schemas.

Provides the formal models for representing solvers as layered transition systems,
tracking opportunities and hypotheses, and recording structured replanning decisions.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

NonemptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class HATSLevel(str, Enum):
    """Hierarchy of abstraction for algorithmic transition systems."""
    LEVEL_0_GOAL = "level_0_goal"            # Overall computational goal & budget
    LEVEL_1_SUBSYSTEM = "level_1_subsystem"  # Algorithmic subsystem (Presolve, Search, Heuristic)
    LEVEL_2_MECHANISM = "level_2_mechanism"  # Mechanism & transition dynamics (e.g. 6-flip, cut rule)
    LEVEL_3_CODE = "level_3_code"            # Source-level files, functions, data structures


class MechanismEffect(BaseModel):
    """Predicted or observed impact of a mechanism on performance or search dynamics."""
    model_config = ConfigDict(extra="forbid")

    target_metric: NonemptyText  # e.g., "search_tree_size", "time_to_feasible", "solved_coverage"
    predicted_direction: Literal["decrease", "increase", "neutral"]
    status: Literal["hypothesis", "supported", "refuted", "inconclusive"] = "hypothesis"
    evidence_refs: list[str] = Field(default_factory=list)


class HATSNode(BaseModel):
    """A node in the Hierarchical Algorithmic Transition System."""
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-zA-Z0-9_\-\.]{2,80}$")
    level: HATSLevel
    parent_id: str | None = None
    purpose: NonemptyText
    source_refs: list[str] = Field(default_factory=list)
    reads: list[str] = Field(default_factory=list)          # Algorithm-state features read
    writes: list[str] = Field(default_factory=list)         # State/variables updated
    preconditions: list[str] = Field(default_factory=list)  # Mathematical & structural preconditions
    invariants: list[str] = Field(default_factory=list)     # Invariants that must be strictly preserved
    effects: list[MechanismEffect] = Field(default_factory=list)
    status: Literal["active", "proposed", "deprecated", "unknown"] = "active"


class Opportunity(BaseModel):
    """A prioritized optimization opportunity within the algorithm space."""
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-zA-Z0-9_\-]{2,64}$")
    title: NonemptyText
    target_nodes: list[str] = Field(min_length=1)
    proposal: NonemptyText
    expected_effect: NonemptyText
    cheapest_discriminating_experiment: NonemptyText
    status: Literal["unexplored", "in_progress", "validated", "refuted", "deferred"] = "unexplored"
    evidence_refs: list[str] = Field(default_factory=list)
    defer_reason: str | None = None


class DecisionType(str, Enum):
    CONTINUE = "continue"  # Refine or advance current mechanism
    SWITCH = "switch"      # Pivot to alternate subsystem/opportunity after falsification/saturation
    COMBINE = "combine"    # Compose multiple synergistic mechanisms
    STOP = "stop"          # Budget exhausted or goal achieved


class ReplanningDecision(BaseModel):
    """Structured decision output produced during an algorithm-space replanning event."""
    model_config = ConfigDict(extra="forbid")

    decision: DecisionType
    selected_opportunity: str
    evidence_refs: list[str] = Field(default_factory=list)
    alternatives_considered: list[dict[str, str]] = Field(default_factory=list)
    next_experiment: NonemptyText
    switch_justification: str | None = None
    revisit_condition: str | None = None
    timestamp: str | None = None


class EvidenceRecord(BaseModel):
    """Empirical result from a controlled experiment mapped back into the planning model."""
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-zA-Z0-9_\-]{2,64}$")
    experiment_id: str
    hypothesis_id: str | None = None
    target_nodes: list[str] = Field(default_factory=list)
    status: Literal["supported", "refuted", "inconclusive"]
    metrics_summary: dict[str, Any] = Field(default_factory=dict)
    code_patch_sha256: str | None = None
    observations_path: str | None = None
    timestamp: str | None = None


class ContextPacket(BaseModel):
    """Compact context delivered to the Agent during an active replanning step."""
    model_config = ConfigDict(extra="forbid")

    task_id: str
    goal_statement: str
    remaining_budget_sec: float
    completed_experiments_count: int
    active_opportunity: Opportunity | None = None
    recent_evidence: list[EvidenceRecord] = Field(default_factory=list)
    affected_nodes: list[HATSNode] = Field(default_factory=list)
    unexplored_opportunities: list[Opportunity] = Field(default_factory=list)


class GlobalPlanState(BaseModel):
    """Persistent, versioned algorithm-space model and planning trajectory."""
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    study_id: str
    task_id: str
    solver: str
    revision: int = 0
    nodes: dict[str, HATSNode] = Field(default_factory=dict)
    opportunities: dict[str, Opportunity] = Field(default_factory=dict)
    active_opportunity_id: str | None = None
    decisions: list[ReplanningDecision] = Field(default_factory=list)
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    budget_spent_sec: float = 0.0
    experiments_count: int = 0
    last_replanned_at_sec: float = 0.0
