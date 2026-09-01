from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Literal, Self

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class TaskType(str, Enum):
    HEURISTIC_SOLVER = "heuristic_solver"
    EXACT_SOLVER = "exact_solver"


class ProblemFamily(str, Enum):
    CVRP = "cvrp"
    MIP = "mip"
    CP = "cp"


class InformationRegime(str, Enum):
    SNAPSHOT_ONLY = "snapshot_only"


class InstanceSetKind(str, Enum):
    AGENT_DEV = "agent_dev"
    JUDGE_ID = "judge_id"
    JUDGE_SHIFT = "judge_shift"


class ReleaseSnapshot(BaseModel):
    repository: str
    version: str
    tag: str
    base_commit: str = Field(min_length=7)
    tree_sha: str | None = Field(default=None, min_length=40, max_length=40)
    released_at: datetime
    url: str


class RepositorySpec(BaseModel):
    clone_url: str
    language: str
    plugin: str
    editable_paths: list[str] = Field(min_length=1)
    agent_image: str | None = None
    judge_image: str | None = None

    @field_validator("editable_paths")
    @classmethod
    def validate_editable_paths(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("repository editable paths must be unique")
        for value in values:
            path = PurePosixPath(value)
            if value in {"", "."} or path.is_absolute() or ".." in path.parts:
                raise ValueError(
                    "repository editable paths must be relative subdirectories"
                )
            if path.parts[0] == ".git":
                raise ValueError("repository .git metadata cannot be editable")
        return values


class OracleReference(BaseModel):
    kind: Literal[
        "known_optimum",
        "best_known_solution",
        "solver_certificate",
        "independent_measurement",
    ]
    source: str
    objective_sense: Literal["minimize", "maximize"] | None = None


class RandomnessSpec(BaseModel):
    instance_seed: int
    coordinate_seed: int | None = None
    demand_seed: int | None = None


class InstanceSetSpec(BaseModel):
    name: str
    kind: InstanceSetKind
    instance_set_config: str
    instance_set_config_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    size: int = Field(gt=0)
    randomness: RandomnessSpec
    shift: str | None = None


class DecisionProtocol(BaseModel):
    """Acceptance rule for performance-first solver-improvement tasks."""

    minimum_success_rate_delta: float = 0.0


class EvaluationProtocol(BaseModel):
    budgets_sec: list[float]
    primary_budget_sec: float = Field(gt=0)
    solver_seeds: list[int]
    threads: int = Field(default=1, gt=0)
    verifier: str
    decision: DecisionProtocol = Field(default_factory=DecisionProtocol)

    @model_validator(mode="after")
    def validate_grid(self) -> Self:
        if not self.budgets_sec or any(value <= 0 for value in self.budgets_sec):
            raise ValueError("evaluation budgets must be positive")
        if self.primary_budget_sec not in self.budgets_sec:
            raise ValueError("primary budget must belong to evaluation budgets")
        if not self.solver_seeds:
            raise ValueError("at least one solver seed is required")
        if len(set(self.solver_seeds)) != len(self.solver_seeds):
            raise ValueError("solver seeds must be unique")
        return self


class PitBenchTask(BaseModel):
    schema_version: Literal["3.0"] = "3.0"
    task_id: str
    release: ReleaseSnapshot
    task_type: TaskType
    problem_family: ProblemFamily
    optimization_scope: str
    instruction: str
    information_regime: InformationRegime
    repository: RepositorySpec
    oracle: OracleReference
    evaluation: EvaluationProtocol
    instance_sets: list[InstanceSetSpec]

    @model_validator(mode="after")
    def validate_instance_set_roles(self) -> Self:
        kinds = {instance_set.kind for instance_set in self.instance_sets}
        required = {InstanceSetKind.AGENT_DEV, InstanceSetKind.JUDGE_ID}
        missing = required - kinds
        if missing:
            missing_values = sorted(kind.value for kind in missing)
            raise ValueError(f"missing required instance-set kinds: {missing_values}")
        names = [instance_set.name for instance_set in self.instance_sets]
        if len(names) != len(set(names)):
            raise ValueError("instance-set names must be unique")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> "PitBenchTask":
        return cls.model_validate(yaml.safe_load(path.read_text()))
