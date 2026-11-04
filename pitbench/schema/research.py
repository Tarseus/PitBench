"""Research task inputs, kept separate from solver scoring and PR references."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

NonemptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


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
    # Scheduler preference, not an independently enforced resource quota.
    development_parallel_runs: int = Field(default=4, gt=0)
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


class CandidateExperiment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    controlled_change: NonemptyText
    expected_observation: NonemptyText
    falsifies_if: NonemptyText
    metric_ids: list[NonemptyText] = Field(min_length=1)


class CandidateHypothesis(BaseModel):
    """Short, observable predictions, not a request for private reasoning."""

    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    mechanism: NonemptyText
    applicability: list[NonemptyText] = Field(min_length=1)
    failure_hypotheses: list[NonemptyText] = Field(min_length=1)
    algorithm_state_features: list[NonemptyText] = Field(min_length=1)
    correctness_invariants: list[NonemptyText] = Field(min_length=1)
    experiments: list[CandidateExperiment] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_experiments(self):
        if len({e.id for e in self.experiments}) != len(self.experiments):
            raise ValueError("candidate experiment IDs must be unique")
        return self


class StudyArm(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: Literal["original", "diverse", "mechanism_aware"]
    development_view: str
    require_candidate_hypotheses: bool = False
    instruction: str = Field(min_length=20)


class StudySeedPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed_count: int = Field(gt=0)
    seed_min: int = Field(default=1, ge=1)
    seed_max: int = Field(default=2147483646, ge=1)
    preserve_existing: bool = True


class GeneralizationStudyPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    study_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    claim_scope: Literal["diagnostic_single_attempt", "replicated_study"]
    release_manifest: str
    task_configs: list[str] = Field(min_length=1)
    model_matrix: str
    population_plan: str
    research_budget: ResearchBudget
    agent_attempts_per_condition: int = Field(gt=0)
    arms: list[StudyArm] = Field(min_length=3, max_length=3)
    shared_agent_tools: list[str] = Field(min_length=1)
    required_scenario_kinds: list[
        Literal["trajectory", "lifecycle", "scaling", "sequence", "numerical", "replay"]
    ] = Field(min_length=1)
    inference_scope: str = Field(min_length=20)
    evaluation_seed_policy: StudySeedPolicy | None = None

    @model_validator(mode="after")
    def validate_controls(self):
        arms = {arm.id: arm for arm in self.arms}
        if set(arms) != {"original", "diverse", "mechanism_aware"}:
            raise ValueError("study needs original, diverse and mechanism-aware arms")
        if arms["diverse"].development_view != arms["mechanism_aware"].development_view:
            raise ValueError(
                "diverse and mechanism-aware arms must receive identical data"
            )
        if arms["original"].development_view == arms["diverse"].development_view:
            raise ValueError(
                "information contrast needs a distinct original development view"
            )
        if (
            any(
                arms[key].require_candidate_hypotheses
                for key in ("original", "diverse")
            )
            or not arms["mechanism_aware"].require_candidate_hypotheses
        ):
            raise ValueError(
                "only the mechanism-aware intervention requires declarations"
            )
        if (
            self.agent_attempts_per_condition == 1
            and self.claim_scope != "diagnostic_single_attempt"
        ):
            raise ValueError("one Agent attempt only supports the diagnostic scope")
        if len(set(self.task_configs)) != len(self.task_configs):
            raise ValueError("study task configurations must be unique")
        return self


class PlanningStudyArm(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: Literal["baseline", "review_only", "model_static", "model_dynamic"]
    development_view: str
    planning_mode: Literal["none", "periodic_review", "static_model", "dynamic_replanning"]
    instruction: str = Field(min_length=20)


class PlanningStudyPlan(BaseModel):
    """Four-arm controlled study validating algorithmic transition models and dynamic replanning."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    study_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    claim_scope: Literal["diagnostic_single_attempt", "replicated_study"]
    release_manifest: str
    task_configs: list[str] = Field(min_length=1)
    model_matrix: str
    population_plan: str
    research_budget: ResearchBudget
    agent_attempts_per_condition: int = Field(gt=0)
    arms: list[PlanningStudyArm] = Field(min_length=4, max_length=4)
    shared_agent_tools: list[str] = Field(min_length=1)
    required_scenario_kinds: list[
        Literal["trajectory", "lifecycle", "scaling", "sequence", "numerical", "replay"]
    ] = Field(min_length=1)
    inference_scope: str = Field(min_length=20)
    evaluation_seed_policy: StudySeedPolicy | None = None

    @model_validator(mode="after")
    def validate_planning_controls(self):
        arm_map = {arm.id: arm for arm in self.arms}
        expected_ids = {"baseline", "review_only", "model_static", "model_dynamic"}
        if set(arm_map) != expected_ids:
            raise ValueError(
                f"planning study requires exactly these arms: {sorted(expected_ids)}"
            )

        # All arms must share the same data population to isolate the planning intervention
        common_view = arm_map["baseline"].development_view
        for arm_id, arm in arm_map.items():
            if arm.development_view != common_view:
                raise ValueError(
                    f"arm '{arm_id}' has development_view='{arm.development_view}' "
                    f"differing from baseline='{common_view}'; planning studies require identical data across arms"
                )

        mode_expectations = {
            "baseline": "none",
            "review_only": "periodic_review",
            "model_static": "static_model",
            "model_dynamic": "dynamic_replanning",
        }
        for arm_id, expected_mode in mode_expectations.items():
            if arm_map[arm_id].planning_mode != expected_mode:
                raise ValueError(
                    f"arm '{arm_id}' must have planning_mode='{expected_mode}', got '{arm_map[arm_id].planning_mode}'"
                )

        if (
            self.agent_attempts_per_condition == 1
            and self.claim_scope != "diagnostic_single_attempt"
        ):
            raise ValueError("one Agent attempt only supports the diagnostic scope")
        if len(set(self.task_configs)) != len(self.task_configs):
            raise ValueError("study task configurations must be unique")
        return self

