"""Explicit experimental contracts for metrics requiring more than one solve."""

from __future__ import annotations

import hashlib
import itertools
import json
import uuid
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCENARIO_METRICS = {
    "trajectory": {
        "time_to_first_verified_solution_sec",
        "time_to_target_sec",
        "primal_integral",
        "dual_progress_integral",
    },
    "lifecycle": {"retained_memory_series", "memory_growth_per_solve"},
    "scaling": {"thread_scaling_curve", "parallel_efficiency"},
    "sequence": {"sequence_solve_time_sec", "update_overhead_sec"},
    "numerical": {"independent_violation", "numerical_failure_counts"},
    "replay": {"same_seed_replay_consistency"},
    "setup": {"setup_time_sec", "setup_peak_memory"},
}

SCENARIO_QUALITY_GUARDS = {
    "retained_memory_series",
    "memory_growth_per_solve",
    "sequence_solve_time_sec",
    "update_overhead_sec",
    "setup_time_sec",
    "setup_peak_memory",
    "same_seed_replay_consistency",
}


class ScenarioProtocol(BaseModel):
    """All choices affecting an estimator are hashed before collection.

    There are no implicit targets, missing-incumbent penalties, replay tolerances,
    warm-up conventions, or reductions of curves to qualification scalars.
    """

    model_config = ConfigDict(extra="forbid")
    protocol_version: Literal["1.0"] = "1.0"
    id: str = Field(min_length=1)
    kind: Literal[
        "trajectory", "lifecycle", "scaling", "sequence", "numerical", "replay", "setup"
    ]
    metrics: list[str] = Field(min_length=1)
    representation: str = "row_column_permutation"
    instance_sets: list[str] = Field(min_length=1)
    objective_sense: Literal["minimize", "maximize"]
    budget_sec: float = Field(gt=0, allow_inf_nan=False)
    solver_seeds: list[int] = Field(min_length=1)
    thread_counts: list[int] = Field(min_length=1)
    repeats: int = Field(gt=0)
    modes: list[Literal["cold", "warm"]] = Field(min_length=1)
    target_by_instance: dict[str, float] = Field(default_factory=dict)
    reference_by_instance: dict[str, float] = Field(default_factory=dict)
    model_sequences_sha256: dict[str, dict[str, list[str]]] = Field(
        default_factory=dict
    )
    reference_solution_by_instance: dict[str, list[float] | dict] = Field(
        default_factory=dict
    )
    qualification_target_by_model_sha256: dict[str, float] = Field(default_factory=dict)
    loss_scale_by_instance: dict[str, float] = Field(default_factory=dict)
    missing_initial_loss: float | None = Field(default=None, allow_inf_nan=False)
    censoring: Literal["budget", "par2"] | None = None
    reference_threads: int | None = Field(default=None, gt=0)
    warmup_steps: int | None = Field(default=None, ge=0)
    series_reduction: Literal["mean", "maximum", "last"] | None = None
    curve_reduction: Literal["mean_target_time", "mean_efficiency"] | None = None
    growth_estimator: Literal["ols"] | None = None
    sequence_mode: Literal["cold", "warm"] | None = None
    replay_fields: list[Literal["objective", "termination", "solution_hash"]] = Field(
        default_factory=list
    )
    replay_absolute_tolerance: float | None = Field(
        default=None, ge=0, allow_inf_nan=False
    )
    replay_relative_tolerance: float | None = Field(
        default=None, ge=0, allow_inf_nan=False
    )
    memory_sample_interval_sec: float = Field(gt=0, allow_inf_nan=False)
    feasibility_tolerance: float = Field(ge=0, allow_inf_nan=False)
    reset_policy: Literal["retain_state", "clear_search"]
    memory_cleanup: Literal["none", "gc_collect"]
    memory_scope: Literal["binding_worker_after_record_flush"] = (
        "binding_worker_after_record_flush"
    )

    @model_validator(mode="after")
    def validate_experiment(self) -> Self:
        for values in (
            self.metrics,
            self.instance_sets,
            self.solver_seeds,
            self.thread_counts,
            self.modes,
            self.replay_fields,
        ):
            if len(set(values)) != len(values):
                raise ValueError("scenario grids and field lists must be unique")
        if any(type(n) is not int or n < 1 for n in self.thread_counts):
            raise ValueError("thread counts must be positive integers")
        if set(self.metrics) - SCENARIO_METRICS[self.kind]:
            raise ValueError("metric does not belong to the declared scenario")
        timed = {"time_to_first_verified_solution_sec", "time_to_target_sec"} & set(
            self.metrics
        )
        if timed and self.censoring is None:
            raise ValueError("first-passage metrics require declared censoring")
        if self.kind == "trajectory" and any("integral" in m for m in self.metrics):
            if (
                self.missing_initial_loss is None
                or not self.reference_by_instance
                or not self.loss_scale_by_instance
            ):
                raise ValueError(
                    "integrals require reference, loss scale and initial missing loss"
                )
        if "time_to_target_sec" in self.metrics or self.kind == "scaling":
            if not self.target_by_instance:
                raise ValueError(
                    "the common quality target must be fixed before collection"
                )
        if self.kind == "scaling":
            if (
                len(self.thread_counts) < 2
                or self.reference_threads not in self.thread_counts
                or self.curve_reduction is None
            ):
                raise ValueError(
                    "scaling needs a reference, at least two thread counts and a curve reduction"
                )
        if self.kind == "lifecycle":
            if (
                self.warmup_steps is None
                or self.series_reduction is None
                or self.growth_estimator is None
                or self.modes != ["warm"]
            ):
                raise ValueError(
                    "lifecycle needs one warm process, warm-up, series reduction and growth estimator"
                )
        if self.kind == "sequence" and (
            set(self.modes) != {"cold", "warm"} or self.sequence_mode is None
        ):
            raise ValueError(
                "incremental experiments require matched cold/warm sequences and a declared scored mode"
            )
        if self.kind == "replay":
            if (
                self.repeats < 2
                or not self.replay_fields
                or self.replay_absolute_tolerance is None
                or self.replay_relative_tolerance is None
            ):
                raise ValueError(
                    "replay requires repeated runs, compared fields and explicit tolerances"
                )
        if self.kind == "numerical" and not self.reference_solution_by_instance:
            raise ValueError(
                "numerical stress scenarios require independently checkable feasible witnesses"
            )
        for mapping in (
            self.target_by_instance,
            self.reference_by_instance,
            self.loss_scale_by_instance,
            self.qualification_target_by_model_sha256,
        ):
            import math

            if any(not math.isfinite(value) for value in mapping.values()):
                raise ValueError("scenario targets and scales must be finite")
        if any(value <= 0 for value in self.loss_scale_by_instance.values()):
            raise ValueError("loss scales must be strictly positive")
        return self

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()


class ScenarioRunKey(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    scenario_id: str
    task_id: str
    instance_set: str
    instance_id: str
    code_state: Literal["base", "agent"]
    solver_seed: int
    threads: int = Field(gt=0)
    repeat: int = Field(ge=0)
    mode: Literal["cold", "warm"]


class ScenarioEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    time_sec: float = Field(ge=0, allow_inf_nan=False)
    solution_values: list[float] | None = None
    solution: dict | None = None
    row_multipliers: list[float] | None = None


class ScenarioStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instance_path: str
    instance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    process_id: int = Field(gt=0)
    process_boot_id: str
    setup_time_sec: float = Field(ge=0, allow_inf_nan=False)
    setup_peak_rss_bytes: int = Field(ge=0)
    solve_time_sec: float = Field(ge=0, allow_inf_nan=False)
    update_time_sec: float = Field(ge=0, allow_inf_nan=False)
    retained_rss_bytes: int = Field(ge=0)
    memory_scope: Literal["binding_worker_after_record_flush"] = (
        "binding_worker_after_record_flush"
    )
    termination: str
    events: list[ScenarioEvent]
    final_solution_values: list[float] | None = None
    final_solution: dict | None = None


class ScenarioRun(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: ScenarioRunKey
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal[
        "returned",
        "collector_error",
        "watchdog_timeout",
        "process_error",
        "unsupported",
    ]
    steps: list[ScenarioStep] = Field(default_factory=list)
    error: str | None = None


class ScenarioCampaign(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str
    task_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocols: list[ScenarioProtocol]
    expected_runs: list[ScenarioRunKey]
    runs: list[ScenarioRun]
    # Changes in the required model sequence are experimental-state changes.
    model_sequences: dict[str, list[dict[str, str]]]
    instances_by_set: dict[str, list[str]]
    runtime_identities: dict[str, str]
    hardware_identity: dict = Field(default_factory=dict)
    order: Literal["base_first", "agent_first"]
    collection_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    started_at: float | None = Field(default=None, allow_inf_nan=False)
    finished_at: float | None = Field(default=None, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if (
            self.started_at is not None
            and self.finished_at is not None
            and self.finished_at < self.started_at
        ):
            raise ValueError("scenario collection ended before it started")
        if len({p.id for p in self.protocols}) != len(self.protocols):
            raise ValueError("scenario IDs must be unique")
        if len(set(self.expected_runs)) != len(self.expected_runs) or len(
            {r.key for r in self.runs}
        ) != len(self.runs):
            raise ValueError("duplicate scenario run")
        if any(r.key not in self.expected_runs for r in self.runs):
            raise ValueError("scenario evidence is outside the expected grid")
        if any(k.task_id != self.task_id for k in self.expected_runs):
            raise ValueError("scenario grid contains another task")
        protocols = {p.id: p for p in self.protocols}
        complete_grid = {
            ScenarioRunKey(
                scenario_id=p.id,
                task_id=self.task_id,
                instance_set=panel,
                instance_id=instance,
                code_state=state,
                solver_seed=seed,
                threads=threads,
                repeat=repeat,
                mode=mode,
            )
            for p in self.protocols
            for panel in p.instance_sets
            for instance, state, seed, threads, repeat, mode in itertools.product(
                self.instances_by_set.get(panel, []),
                ("base", "agent"),
                p.solver_seeds,
                p.thread_counts,
                range(p.repeats),
                p.modes,
            )
        }
        if not complete_grid or complete_grid != set(self.expected_runs):
            raise ValueError(
                "scenario expected grid differs from the declared protocol"
            )
        if any(
            not ids or len(ids) != len(set(ids))
            for ids in self.instances_by_set.values()
        ):
            raise ValueError(
                "scenario instance inventories must be nonempty and unique"
            )
        if set(self.runtime_identities) != {"base", "agent"}:
            raise ValueError("both runtime identities are required")
        for run in self.runs:
            if (
                run.key.scenario_id not in protocols
                or run.protocol_sha256 != protocols[run.key.scenario_id].digest()
            ):
                raise ValueError("scenario protocol identity mismatch")
            if run.runtime_identity_sha256 != self.runtime_identities.get(
                run.key.code_state
            ):
                raise ValueError("scenario runtime identity mismatch")
        return self
