from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import yaml
from pydantic import BaseModel, ConfigDict, Field

ConfigValue: TypeAlias = str | int | float | bool | None


class EvaluationPaths(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_path: Path = Path("runs")
    workspace_path: Path = Path(".pitbench/tasks")
    private_root: Path = Path("private")


class TaskResources(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repository_source: Path | None = None
    agent_image: str | None = None
    judge_image: str | None = None


class PipelineResources(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_workers: int = Field(default=3, ge=1)
    agent_cpus_per_worker: int = Field(default=8, ge=1)
    agent_cpu_pool: str | None = None
    judge_workers: int = Field(default=4, ge=1)
    judge_parallel_runs: int = Field(default=8, ge=1)
    judge_cpu_pool: str | None = None
    infrastructure_retries: int = Field(default=2, ge=0)


class EvaluationConfig(BaseModel):
    """Machine-local settings for the unified evaluation command."""

    model_config = ConfigDict(extra="forbid")

    paths: EvaluationPaths = Field(default_factory=EvaluationPaths)
    tasks: dict[str, TaskResources] = Field(default_factory=dict)
    agents: dict[str, dict[str, ConfigValue]] = Field(default_factory=dict)
    pipeline: PipelineResources = Field(default_factory=PipelineResources)

    @classmethod
    def from_yaml(cls, path: Path) -> EvaluationConfig:
        payload = yaml.safe_load(path.read_text())
        return cls.model_validate(payload or {})

    def kwargs_for(self, agent: str) -> dict[str, ConfigValue]:
        return self.agents.get(agent, {}).copy()

    def resources_for(self, task_id: str) -> TaskResources:
        return self.tasks.get(task_id, TaskResources())


def resolve_config_path(repository_root: Path, config_path: Path | None) -> Path | None:
    if config_path is None:
        local_path = repository_root / "config/evaluate.local.yaml"
        return local_path if local_path.is_file() else None
    if config_path.is_absolute():
        return config_path
    return repository_root / config_path


def resolve_repository_path(repository_root: Path, value: Path) -> Path:
    expanded = value.expanduser()
    if expanded.is_absolute():
        return expanded
    return repository_root / expanded


def encode_agent_kwargs(values: dict[str, ConfigValue]) -> list[str]:
    return [f"{key}={value}" for key, value in values.items() if value is not None]
