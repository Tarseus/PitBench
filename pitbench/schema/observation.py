from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class CodeState(str, Enum):
    BASE = "base"
    AGENT = "agent"


class RunStatus(str, Enum):
    COMPLETED = "completed"
    INVALID = "invalid"
    CRASHED = "crashed"
    TIMED_OUT = "timed_out"
    BUILD_FAILED = "build_failed"


class RunObservation(BaseModel):
    task_id: str
    code_state: CodeState
    population: str
    instance_id: str
    instance_seed: int
    coordinate_seed: int | None = None
    demand_seed: int | None = None
    solver_seed: int
    budget_sec: float = Field(gt=0)
    threads: int = Field(default=1, gt=0)
    status: RunStatus
    valid: bool
    objective: float | None = None
    optimal_or_bks: float | None = None
    normalized_gap: float | None = None
    primal_bound: float | None = None
    dual_bound: float | None = None
    wall_time_sec: float | None = None
    cpu_time_sec: float | None = None
    iterations: int | None = None
    nodes: int | None = None
    model_variables: int | None = None
    model_constraints: int | None = None
    peak_rss_bytes: int | None = None
    trajectory_path: str | None = None
    solution_path: str | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
    error: str | None = None
