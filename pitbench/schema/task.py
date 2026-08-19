from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, Field, model_validator


class TaskType(str, Enum):
    SOLVER_SEARCH = "solver_search"
    HEURISTIC_SOLVER = "heuristic_solver"
    EXACT_SOLVER = "exact_solver"
    MODEL_BUILD = "model_build"
    PRESOLVE = "presolve"


class ProblemFamily(str, Enum):
    CVRP = "cvrp"
    MIP = "mip"
    CP = "cp"


class InformationRegime(str, Enum):
    SNAPSHOT_ONLY = "snapshot_only"
    ISSUE_CONTEXT = "issue_context"
    ISSUE_PLUS_PRE_PR_HISTORY = "issue_plus_pre_pr_history"
    RICH_DEVELOPER_CONTEXT = "rich_developer_context"


class TaskSplit(str, Enum):
    METRIC_DEV = "metric_dev_patches"
    METRIC_VALIDATION = "metric_validation_patches"
    BENCHMARK_TEST = "benchmark_test_patches"


class PopulationKind(str, Enum):
    AGENT_DEV = "agent_dev"
    JUDGE_ID = "judge_id"
    JUDGE_SHIFT = "judge_shift"


class HistoricalEvent(BaseModel):
    repository: str
    pr_number: int = Field(gt=0)
    base_commit: str = Field(min_length=7)
    human_commit: str = Field(min_length=7)
    pr_created_at: datetime
    pr_merged_at: datetime
    title: str
    url: str


class RepositorySpec(BaseModel):
    clone_url: str
    language: str
    plugin: str
    agent_image: str | None = None
    judge_image: str | None = None


class HumanReference(BaseModel):
    uri: str
    sha256: str | None = None

    @model_validator(mode="after")
    def require_private_uri(self) -> Self:
        if not self.uri.startswith("private://"):
            raise ValueError("human patch must use a private:// URI")
        return self


class OracleReference(BaseModel):
    kind: Literal[
        "known_optimum",
        "best_known_solution",
        "solver_certificate",
        "independent_measurement",
    ]
    source: str
    objective_sense: Literal["minimize", "maximize"] | None = None


class References(BaseModel):
    human_patch: HumanReference
    oracle: OracleReference


class RandomnessSpec(BaseModel):
    instance_seed: int
    coordinate_seed: int | None = None
    demand_seed: int | None = None


class PopulationSpec(BaseModel):
    name: str
    kind: PopulationKind
    manifest: str
    size: int = Field(gt=0)
    randomness: RandomnessSpec
    shift: str | None = None


class BuildSpec(BaseModel):
    compiler: str | None = None
    compiler_version: str | None = None
    flags: list[str] = Field(default_factory=list)
    sanitizers: list[Literal["address", "undefined"]] = Field(default_factory=list)


class EvaluationProtocol(BaseModel):
    family_plugin: str
    budgets_sec: list[float]
    solver_seeds: list[int]
    threads: int = Field(default=1, gt=0)
    verifier: str
    validation_build: BuildSpec
    performance_build: BuildSpec

    @model_validator(mode="after")
    def validate_grid(self) -> Self:
        if not self.budgets_sec or any(value <= 0 for value in self.budgets_sec):
            raise ValueError("evaluation budgets must be positive")
        if not self.solver_seeds:
            raise ValueError("at least one solver seed is required")
        if len(set(self.solver_seeds)) != len(self.solver_seeds):
            raise ValueError("solver seeds must be unique")
        return self


class PitBenchTask(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    task_id: str
    split: TaskSplit
    event: HistoricalEvent
    task_type: TaskType
    problem_family: ProblemFamily
    optimization_scope: str
    instruction: str
    information_regime: InformationRegime
    repository: RepositorySpec
    references: References
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
