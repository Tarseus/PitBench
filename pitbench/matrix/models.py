from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class MatrixAgentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    agent: str
    model: str
    agent_kwargs: dict[str, Any] = Field(default_factory=dict)


class MatrixSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    experiment_id: str
    tasks: list[str]
    agents: list[MatrixAgentSpec]
    repeats: int = Field(default=3, ge=1)
    schedule_seed: int = 20260827
    base_cache_scope: Literal["task_repeat"] = "task_repeat"

    @model_validator(mode="after")
    def validate_unique_entries(self) -> MatrixSpec:
        if not self.tasks:
            raise ValueError("matrix tasks cannot be empty")
        if not self.agents:
            raise ValueError("matrix agents cannot be empty")
        if len(set(self.tasks)) != len(self.tasks):
            raise ValueError("matrix task IDs must be unique")
        agent_ids = [agent.id for agent in self.agents]
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError("matrix agent IDs must be unique")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> MatrixSpec:
        payload = yaml.safe_load(path.read_text())
        return cls.model_validate(payload or {})


class PipelineStatus(str, Enum):
    PENDING = "pending"
    GENERATING = "generating"
    CANDIDATE_READY = "candidate_ready"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"


class MatrixTrialState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    agent_id: str
    repeat: int = Field(ge=1)
    status: PipelineStatus = PipelineStatus.PENDING
    generation_attempts: int = 0
    evaluation_attempts: int = 0
    candidate_patch: str | None = None
    candidate_sha256: str | None = None
    agent_result: str | None = None
    evaluation_result: str | None = None
    agent_duration_sec: float | None = None
    evaluation_duration_sec: float | None = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    infrastructure_failures: list[str] = Field(default_factory=list)
    error: str | None = None
    updated_at: str
