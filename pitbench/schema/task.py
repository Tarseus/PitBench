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


class PerformanceProtocol(str, Enum):
    HEURISTIC_FIXED_BUDGET = "heuristic_fixed_budget"
    EXACT_VERIFIED_SOLVE = "exact_verified_solve"


class ExactTimeBasis(str, Enum):
    CPU_TIME = "cpu_time"
    SOLVE_WALL_TIME = "solve_wall_time"


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
    source_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
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
    randomness: RandomnessSpec | None = None
    shift: str | None = None


class SeedSelectionConfig(BaseModel):
    seed_min: int
    seed_max: int
    seed_count: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_seed_range(self) -> Self:
        if self.seed_min > self.seed_max:
            raise ValueError("seed_min must not exceed seed_max")
        available_seed_count = self.seed_max - self.seed_min + 1
        if available_seed_count < 2 * self.seed_count:
            raise ValueError(
                "seed range must fit disjoint development and evaluation seeds"
            )
        return self


class SeedRobustnessConfig(BaseModel):
    development_seeds: list[int]
    evaluation_seeds_file: str
    evaluation_seeds_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed_selection: SeedSelectionConfig

    @field_validator("evaluation_seeds_file")
    @classmethod
    def validate_evaluation_seeds_file(cls, value: str) -> str:
        if not value.startswith("private://"):
            raise ValueError("evaluation_seeds_file must use private:// storage")
        return value

    @model_validator(mode="after")
    def validate_development_seeds(self) -> Self:
        if len(self.development_seeds) != self.seed_selection.seed_count:
            raise ValueError("development_seeds must contain seed_count values")
        if len(set(self.development_seeds)) != len(self.development_seeds):
            raise ValueError("development_seeds must be unique")
        if any(
            seed < self.seed_selection.seed_min or seed > self.seed_selection.seed_max
            for seed in self.development_seeds
        ):
            raise ValueError("development_seeds must belong to the seed range")
        return self


class RepresentationRobustnessConfig(BaseModel):
    kind: str = "customer_relabeling"
    verification_tolerance: float | None = Field(default=None, gt=0)
    instance_set: str = "agent_dev"
    solver_seed: int = Field(default=0, ge=0, le=4294967295)
    relabeling_generation_seed: int = 20260907
    relabelings_per_instance: int = Field(default=30, gt=0)


class EvaluationProtocol(BaseModel):
    performance_protocol: PerformanceProtocol
    budgets_sec: list[float]
    primary_budget_sec: float = Field(gt=0)
    solver_seeds: list[int] | None = None
    seed_robustness: SeedRobustnessConfig | None = None
    representation_robustness: RepresentationRobustnessConfig | None = None
    operational_reliability: bool = False
    threads: int = Field(default=1, gt=0)
    exact_time_basis: ExactTimeBasis = ExactTimeBasis.CPU_TIME
    verifier: str

    @model_validator(mode="before")
    @classmethod
    def reject_removed_decision(cls, data: object) -> object:
        if isinstance(data, dict) and "decision" in data:
            raise ValueError("evaluation.decision has been removed")
        return data

    @model_validator(mode="after")
    def validate_grid(self) -> Self:
        if not self.budgets_sec or any(value <= 0 for value in self.budgets_sec):
            raise ValueError("evaluation budgets must be positive")
        if self.primary_budget_sec not in self.budgets_sec:
            raise ValueError("primary budget must belong to evaluation budgets")
        if (self.solver_seeds is None) == (self.seed_robustness is None):
            raise ValueError(
                "evaluation requires either solver_seeds or seed_robustness, not both"
            )
        if self.solver_seeds is not None:
            if not self.solver_seeds:
                raise ValueError("at least one solver seed is required")
            if len(set(self.solver_seeds)) != len(self.solver_seeds):
                raise ValueError("solver seeds must be unique")
        if (
            self.performance_protocol != PerformanceProtocol.EXACT_VERIFIED_SOLVE
            and self.exact_time_basis != ExactTimeBasis.CPU_TIME
        ):
            raise ValueError(
                "non-exact performance protocols cannot select an exact time basis"
            )
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
        performance_protocol = self.evaluation.performance_protocol
        if (
            self.task_type == TaskType.HEURISTIC_SOLVER
            and performance_protocol != PerformanceProtocol.HEURISTIC_FIXED_BUDGET
        ):
            raise ValueError("heuristic task requires heuristic performance protocol")
        if (
            self.task_type == TaskType.EXACT_SOLVER
            and performance_protocol == PerformanceProtocol.HEURISTIC_FIXED_BUDGET
        ):
            raise ValueError("exact task requires an exact performance protocol")
        if performance_protocol == PerformanceProtocol.EXACT_VERIFIED_SOLVE:
            if (
                self.oracle.kind not in {"known_optimum", "best_known_solution"}
                or not self.oracle.source.startswith("private://")
                or self.oracle.source_sha256 is None
                or self.oracle.objective_sense is None
            ):
                raise ValueError(
                    "exact verified solve requires a hash-pinned private "
                    "known-optimum or best-known-solution oracle"
                )
            expected_verifier = (
                "trusted_optimum"
                if self.oracle.kind == "known_optimum"
                else "exact_target"
            )
            if self.evaluation.verifier != expected_verifier:
                raise ValueError(
                    f"exact verified solve with {self.oracle.kind} requires the "
                    f"{expected_verifier} verifier"
                )
            if self.evaluation.exact_time_basis == ExactTimeBasis.SOLVE_WALL_TIME:
                if self.problem_family != ProblemFamily.CP:
                    raise ValueError(
                        "solve wall time is reserved for the parallel CP-SAT task"
                    )
                if self.evaluation.threads != 8:
                    raise ValueError(
                        "parallel CP-SAT exact solving requires exactly eight threads"
                    )
        kinds = {instance_set.kind for instance_set in self.instance_sets}
        required = {InstanceSetKind.AGENT_DEV, InstanceSetKind.JUDGE_ID}
        missing = required - kinds
        if missing:
            missing_values = sorted(kind.value for kind in missing)
            raise ValueError(f"missing required instance-set kinds: {missing_values}")
        names = [instance_set.name for instance_set in self.instance_sets]
        if len(names) != len(set(names)):
            raise ValueError("instance-set names must be unique")
        representation = self.evaluation.representation_robustness
        if representation is not None:
            from pitbench.evaluator.representations import representation_type
            from pitbench.repositories.base import RepositoryPluginRegistry

            transform = representation_type(representation.kind)
            repository = RepositoryPluginRegistry.load(self.repository.plugin)
            if (
                transform.family != self.problem_family
                or "judge"
                not in repository.representations.get(representation.kind, ())
            ):
                raise ValueError(
                    "representation is not supported by the configured family and repository"
                )
            if (
                transform.requires_tolerance
                and representation.verification_tolerance is None
            ):
                raise ValueError(
                    "numeric representation requires a verification tolerance"
                )
            if not any(
                item.name == representation.instance_set
                and item.kind == InstanceSetKind.AGENT_DEV
                for item in self.instance_sets
            ):
                raise ValueError(
                    "representation robustness requires a declared agent_dev instance set"
                )
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> "PitBenchTask":
        return cls.model_validate(yaml.safe_load(path.read_text()))
