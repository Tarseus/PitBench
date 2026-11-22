from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, TypeAlias

import yaml
from pydantic import BaseModel, ConfigDict, Field

from pitbench.agent_tools import AgentTool
from pitbench.schema.research import ResearchBudget
from pitbench.schema.task import PitBenchTask

ConfigValue: TypeAlias = str | int | float | bool | None


class EvaluationPaths(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_path: Path = Path("gushengda/experiments/ad-hoc/runs")
    workspace_path: Path = Path("gushengda/experiments/ad-hoc/workspaces")
    private_root: Path = Path("private")
    base_cache_path: Path = Path(".pitbench/cache/base")


class TaskResources(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_image: str | None = None
    judge_image: str | None = None
    judge_cpus: float | None = None
    judge_memory: str | None = None
    judge_tmpfs_size: str | None = None
    judge_parallel_runs: int | None = None
    judge_cpuset_cpus: str | None = None
    base_observations_path: Path | None = None
    research_budget: ResearchBudget | None = None
    task_config_path: Path | None = None
    agent_task_config_path: Path | None = None


class SolverImageBuildConfig(BaseModel):
    """Machine-local network settings for source solver image builds."""

    model_config = ConfigDict(extra="forbid")

    proxy_url: str | None = None
    docker_network: str | None = None
    no_proxy: str | None = None
    base_images: dict[str, str] = Field(default_factory=dict)
    no_cache: bool = False


class ConfigurationRobustnessConfig(BaseModel):
    """Parameter evaluation enabled by default for captured candidates."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    scope: Literal["candidate", "release"] = "candidate"
    experiment_config: Path = Path(
        "configs/research/configuration_robustness_native_v2.yaml"
    )
    output_path: Path = Path("configuration-robustness")
    cpus: list[int] | None = None
    search_seed: int = 20260918


class DryRunConfig(BaseModel):
    """Short solver budgets for empty-patch checks and real-Agent smoke tests."""

    model_config = ConfigDict(extra="forbid")

    solver_budget_sec: float = Field(default=0.1, gt=0, allow_inf_nan=False)

    def task_view(self, task: PitBenchTask, *, solver_seed_limit: int | None = None) -> PitBenchTask:
        payload = task.model_dump(mode="json")
        evaluation = payload["evaluation"]
        evaluation["budgets_sec"] = [self.solver_budget_sec]
        evaluation["primary_budget_sec"] = self.solver_budget_sec
        for panel in payload["instance_sets"]:
            if solver_seed_limit is not None and panel.get("solver_seed_count") is not None:
                panel["solver_seed_count"] = min(panel["solver_seed_count"], solver_seed_limit)
            if panel.get("budgets_sec") is not None:
                panel["budgets_sec"] = [self.solver_budget_sec]
                panel["primary_budget_sec"] = self.solver_budget_sec
        if evaluation.get("controlled_transfer"):
            for condition in evaluation["controlled_transfer"]["conditions"]:
                if condition.get("budget_sec") is not None:
                    condition["budget_sec"] = self.solver_budget_sec
        for scenario in payload["metric_scenarios"]:
            scenario["budget_sec"] = self.solver_budget_sec
        return PitBenchTask.model_validate(payload)


class EvaluationConfig(BaseModel):
    """Machine-local settings for the unified evaluation command."""

    model_config = ConfigDict(extra="forbid")

    paths: EvaluationPaths = Field(default_factory=EvaluationPaths)
    dry_run: DryRunConfig = Field(default_factory=DryRunConfig)
    solver_image_build: SolverImageBuildConfig = Field(
        default_factory=SolverImageBuildConfig
    )
    configuration_robustness: ConfigurationRobustnessConfig = Field(
        default_factory=ConfigurationRobustnessConfig
    )
    agent_tools: list[AgentTool] = Field(
        default_factory=lambda: [
            AgentTool.INSPECT,
            AgentTool.WORKSPACE,
            AgentTool.BENCH,
        ]
    )
    agent_feedback_mode: Literal["outcome_only", "solver_behavior"] = "solver_behavior"
    candidate_hypotheses_required: bool = False
    agent_cpuset_cpus: str | None = None
    task_ids: list[str] = Field(default_factory=list)
    tasks: dict[str, TaskResources] = Field(default_factory=dict)
    agents: dict[str, dict[str, ConfigValue]] = Field(default_factory=dict)
    judge_cpus: float | None = None
    judge_memory: str | None = None
    judge_tmpfs_size: str | None = None
    judge_parallel_runs: int | None = None
    base_cache: bool = False
    metric_scenario_configs: dict[str, Path] = Field(default_factory=dict)
    research_track: Literal["lite", "standard", "research"] | None = None
    research_budget: ResearchBudget | None = None

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


def compare_feedback_configs(
    outcome_config: Path, behavior_config: Path
) -> dict[str, object]:
    outcome = EvaluationConfig.from_yaml(outcome_config).model_dump(mode="json")
    behavior = EvaluationConfig.from_yaml(behavior_config).model_dump(mode="json")
    if outcome.pop("agent_feedback_mode") != "outcome_only":
        raise ValueError("outcome arm must use outcome_only feedback")
    if behavior.pop("agent_feedback_mode") != "solver_behavior":
        raise ValueError("behavior arm must use solver_behavior feedback")
    if outcome != behavior:
        raise ValueError("feedback arms differ in settings other than feedback mode")
    if "bench" not in outcome["agent_tools"]:
        raise ValueError("matched feedback arms require the same public bench tool")
    digest = hashlib.sha256(json.dumps(outcome, sort_keys=True).encode()).hexdigest()
    return {"matched": True, "common_config_sha256": digest}
