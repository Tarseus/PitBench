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


class PopulationKind(str, Enum):
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


class PopulationSpec(BaseModel):
    name: str
    kind: PopulationKind
    manifest: str
    manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    size: int = Field(gt=0)
    randomness: RandomnessSpec
    shift: str | None = None


class SensitivityProtocol(BaseModel):
    """Frozen panels used to populate the three input-sensitivity dimensions."""

    scale_descriptor: Literal["customer_count", "model_variables"] | None = None
    equivalence_transform: Literal["customer_relabel"] | None = None
    equivalence_instances_per_population: int = Field(default=0, ge=0)
    equivalence_budgets_sec: list[float] = Field(default_factory=list)
    equivalence_solver_seeds: list[int] = Field(default_factory=list)


class DecisionProtocol(BaseModel):
    """Versioned Pareto-style acceptance rule for solver-improvement tasks."""

    minimum_success_rate_delta: float = 0.0
    minimum_operational_speedup: float = Field(default=0.95, gt=0)
    minimum_cpu_speedup: float = Field(default=0.95, gt=0)
    maximum_peak_rss_ratio: float = Field(default=1.05, gt=0)
    require_complete_sensitivity: bool = False


class EvaluationProtocol(BaseModel):
    budgets_sec: list[float]
    solver_seeds: list[int]
    threads: int = Field(default=1, gt=0)
    verifier: str
    sensitivity: SensitivityProtocol = Field(default_factory=SensitivityProtocol)
    decision: DecisionProtocol = Field(default_factory=DecisionProtocol)

    @model_validator(mode="after")
    def validate_grid(self) -> Self:
        if not self.budgets_sec or any(value <= 0 for value in self.budgets_sec):
            raise ValueError("evaluation budgets must be positive")
        if not self.solver_seeds:
            raise ValueError("at least one solver seed is required")
        if len(set(self.solver_seeds)) != len(self.solver_seeds):
            raise ValueError("solver seeds must be unique")
        sensitivity = self.sensitivity
        if sensitivity.equivalence_transform is not None:
            if sensitivity.equivalence_instances_per_population == 0:
                raise ValueError(
                    "equivalence panel requires at least one instance per population"
                )
            unknown_budgets = set(sensitivity.equivalence_budgets_sec) - set(
                self.budgets_sec
            )
            unknown_seeds = set(sensitivity.equivalence_solver_seeds) - set(
                self.solver_seeds
            )
            if unknown_budgets or unknown_seeds:
                raise ValueError(
                    "equivalence budgets and seeds must belong to the evaluation grid"
                )
        return self


class PitBenchTask(BaseModel):
    schema_version: Literal["2.0"] = "2.0"
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
    populations: list[PopulationSpec]

    @model_validator(mode="after")
    def validate_population_roles(self) -> Self:
        kinds = {population.kind for population in self.populations}
        required = {PopulationKind.AGENT_DEV, PopulationKind.JUDGE_ID}
        missing = required - kinds
        if missing:
            missing_values = sorted(kind.value for kind in missing)
            raise ValueError(f"missing required population kinds: {missing_values}")
        names = [population.name for population in self.populations]
        if len(names) != len(set(names)):
            raise ValueError("population names must be unique")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> "PitBenchTask":
        return cls.model_validate(yaml.safe_load(path.read_text()))
