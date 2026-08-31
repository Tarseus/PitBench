from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, computed_field

from pitbench.metrics.behavior_metrics import BehaviorMetricReport
from pitbench.metrics.decision_metrics import BenchmarkDecision
from pitbench.metrics.outcome_metrics import OutcomeReport
from pitbench.metrics.sensitivity_metrics import SensitivityReport
from pitbench.schema.observation import CodeState, RunObservation


class ValidityCode(str, Enum):
    PATCH_APPLY = "patch_apply"
    VALIDATION_BUILD = "validation_build"
    SANITIZER = "sanitizer"
    SOLUTION = "solution"
    OBJECTIVE = "objective"
    CRASH = "crash"
    TIMEOUT = "timeout"
    EVALUATOR = "evaluator"


class ValidityCheck(BaseModel):
    code: ValidityCode
    passed: bool
    detail: str


class ValidityResult(BaseModel):
    checks: list[ValidityCheck]

    @computed_field
    @property
    def accepted(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)


class ArtifactRef(BaseModel):
    path: str
    sha256: str
    size_bytes: int = Field(ge=0)
    media_type: str | None = None
    private: bool = False


class ArtifactManifest(BaseModel):
    candidate_patch: ArtifactRef | None = None
    observations: ArtifactRef | None = None
    trajectories: list[ArtifactRef] = Field(default_factory=list)
    solutions: list[ArtifactRef] = Field(default_factory=list)
    logs: list[ArtifactRef] = Field(default_factory=list)


class EvaluationSummary(BaseModel):
    observation_count: int = Field(ge=0)
    valid_observation_count: int = Field(ge=0)
    counts_by_state: dict[CodeState, int] = Field(default_factory=dict)
    outcomes: OutcomeReport | None = None
    sensitivity: SensitivityReport | None = None
    behavior: BehaviorMetricReport | None = None
    decision: BenchmarkDecision | None = None


class EvaluationResult(BaseModel):
    task_id: str
    validity: ValidityResult
    observations: list[RunObservation]
    artifacts: ArtifactManifest
    summary: EvaluationSummary

    @computed_field
    @property
    def is_resolved(self) -> bool:
        return bool(self.summary.decision and self.summary.decision.is_resolved)
