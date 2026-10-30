from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterable
from pathlib import Path

import yaml

from pitbench.agent_tools import AgentTool, normalize_agent_tools
from pitbench.repositories.base import BuildKind, RepositoryPluginRegistry
from pitbench.schema.research import ResearchBudget
from pitbench.schema.task import PitBenchTask


def write_agent_tooling(
    *,
    repository_root: Path,
    task: PitBenchTask,
    task_dir: Path,
    agent_tools: Iterable[AgentTool | str],
    agent_feedback_mode: str = "solver_behavior",
    research_budget: ResearchBudget | None = None,
    candidate_hypotheses_required: bool = False,
) -> None:
    """Bundle the explicitly enabled public developer tooling into a task."""
    tools = normalize_agent_tools(agent_tools)
    if agent_feedback_mode not in {"outcome_only", "solver_behavior"}:
        raise ValueError("unknown agent feedback mode")
    if not tools:
        return

    tooling = task_dir / "agent_tooling" / "pitbench"
    tooling.mkdir(parents=True)
    for filename in ("__init__.py", "agent_cli.py", "agent_diagnostics.py"):
        shutil.copy2(repository_root / "pitbench" / filename, tooling)
    if AgentTool.BENCH in tools:
        schema = tooling / "schema"
        schema.mkdir()
        (schema / "__init__.py").write_text("")
        for name in ("research.py", "objective_catalog.json"):
            shutil.copy2(repository_root / "pitbench/schema" / name, schema / name)
        shutil.copytree(
            repository_root / "pitbench" / "solver_drivers",
            tooling / "solver_drivers",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        checks = tooling / "problem_families"
        checks.mkdir()
        (checks / "__init__.py").write_text("")
        for filename in ("base.py", "public_verification.py", "verification.py"):
            shutil.copy2(
                repository_root / "pitbench" / "problem_families" / filename,
                checks,
            )

    agent_bin = task_dir / "agent_bin"
    agent_bin.mkdir()
    executable = agent_bin / "pitbench"
    executable.write_text('#!/bin/sh\nexec python3 -m pitbench.agent_cli "$@"\n')
    executable.chmod(0o755)

    config: dict[str, object] = {
        "agent_tools": sorted(tool.value for tool in tools),
        "agent_feedback_mode": agent_feedback_mode,
        "candidate_hypotheses_required": candidate_hypotheses_required,
        "development_parallel_runs": research_budget.development_parallel_runs
        if research_budget
        else 1,
        "research_cpus": research_budget.cpus if research_budget else None,
    }
    repository = RepositoryPluginRegistry.load(task.repository.plugin)
    if tools & {AgentTool.INSPECT, AgentTool.BENCH}:
        seed_robustness = task.evaluation.seed_robustness
        development_seeds = (
            seed_robustness.development_seeds
            if seed_robustness is not None
            else task.evaluation.solver_seeds
        )
        if development_seeds is None:
            raise ValueError("task does not provide development seeds")
        config.update(
            {
                "task_id": task.task_id,
                "task_type": task.task_type.value,
                "problem_family": task.problem_family.value,
                "budgets_sec": task.evaluation.budgets_sec,
                "primary_budget_sec": task.evaluation.primary_budget_sec,
                "performance_protocol": task.evaluation.performance_protocol.value,
                "exact_time_basis": task.evaluation.exact_time_basis.value,
                "objective_sense": task.oracle.objective_sense,
                "development_seeds": development_seeds,
                "threads": task.evaluation.threads,
                "runner": repository.agent_run_template(),
                "runner_requirement": repository.agent_requirement,
                "artifact_globs": repository.agent_artifact_globs,
                "build_output_paths": repository.agent_build_paths,
                "build_commands": [
                    command.model_dump(mode="json")
                    for command in repository.agent_build_commands()
                ],
            }
        )
        fixed_parameters = task.evaluation.fixed_solver_parameters
        if fixed_parameters is not None:
            if not repository.supports_fixed_solver_parameters:
                raise ValueError(
                    "repository driver cannot apply fixed solver parameters"
                )
            config["fixed_solver_parameters"] = fixed_parameters
        instance_index = yaml.safe_load(
            (task_dir / "dev_instances" / "instance_set_config.yaml").read_text()
        )
        config["development_targets"] = {
            item["path"]: {
                "id": item["id"],
                "sha256": hashlib.sha256(
                    (task_dir / "dev_instances" / item["path"]).read_bytes()
                ).hexdigest(),
                "target": item.get("bks"),
                "proven_optimal": item.get("proven_optimal"),
            }
            for item in instance_index["instances"]
        }
    if tools & {
        AgentTool.INSPECT,
        AgentTool.WORKSPACE,
        AgentTool.VALIDATE,
    }:
        config["editable_paths"] = task.repository.editable_paths
    if AgentTool.VALIDATE in tools:
        config["validation_commands"] = [
            command.model_dump(mode="json")
            for command in repository.build_commands(BuildKind.VALIDATION)
        ]
    (task_dir / "agent_config.json").write_text(json.dumps(config, indent=2) + "\n")
