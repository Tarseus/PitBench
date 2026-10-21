"""Research task inputs, kept separate from solver scoring and PR references."""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


class ResearchBudget(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    track: Literal["lite", "standard", "research", "custom"] = "custom"
    wall_time_sec: float = Field(gt=0)
    # Total sandbox CPU and development-experiment CPU are different scopes.
    cpu_time_sec: float | None = Field(default=None, gt=0)
    development_cpu_time_sec: float | None = Field(default=None, gt=0)
    cpus: float | None = Field(default=None, gt=0)
    memory_gib: float | None = Field(default=None, gt=0)
    max_parallel_runs: int | None = Field(default=None, gt=0)
    model_token_limit: int | None = Field(default=None, gt=0)
    checkpoints_sec: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_checkpoints(self):
        if any(not 0 < point <= self.wall_time_sec for point in self.checkpoints_sec):
            raise ValueError(
                "checkpoints must be positive and within the research budget"
            )
        if self.checkpoints_sec != sorted(set(self.checkpoints_sec)):
            raise ValueError("research checkpoints must be unique and increasing")
        return self


class GeneratorCapability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    parameters: list[str] = Field(min_length=1)
    description: str


class GuidedContextEvidence(BaseModel):
    """Evaluator review bound to the exact released text and source snapshot."""

    model_config = ConfigDict(extra="forbid")

    context_index: int = Field(ge=0)
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_url: str
    snapshot_path: str
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    available_at: AwareDatetime
    solution_cutoff: AwareDatetime
    reviewer: str = Field(min_length=1)
    review_note: str = Field(min_length=1)


class ResearchCasePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    task_config: str
    historical_request: str
    historical_bundle_root: str
    open_question: str = Field(min_length=20)
    guided_question: str = Field(min_length=20)
    # Only independently reviewed pre-solution evidence may populate this input.
    guided_context: list[str] = Field(default_factory=list)
    guided_context_evidence: list[GuidedContextEvidence] = Field(default_factory=list)
    generators: list[GeneratorCapability] = Field(min_length=1)
    experimental_access: Literal[
        "fixed_workload", "select_instances", "construct_instances"
    ] = "construct_instances"
    research_budget: ResearchBudget | None = None
    # Map hidden panels to scientific roles only after checking their generation.
    hidden_view_roles: dict[str, Literal["id", "structure", "scale", "joint"]] = Field(
        default_factory=dict
    )


class ResearchPilotPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    protocol: str = "configs/research/protocol.json"
    cases: list[ResearchCasePlan] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_cases(self):
        if len({item.case_id for item in self.cases}) != len(self.cases):
            raise ValueError("research case IDs must be unique")
        return self
