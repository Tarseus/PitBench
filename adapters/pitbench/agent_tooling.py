from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from pathlib import Path

from pitbench.agent_tools import AgentTool, normalize_agent_tools
from pitbench.repositories.base import BuildKind, RepositoryPluginRegistry
from pitbench.schema.task import PitBenchTask


def write_agent_tooling(
    *,
    repository_root: Path,
    task: PitBenchTask,
    task_dir: Path,
    agent_tools: Iterable[AgentTool | str],
) -> None:
    """Bundle the explicitly enabled public developer tooling into a task."""
    tools = normalize_agent_tools(agent_tools)
    if not tools:
        return

    tooling = task_dir / "agent_tooling" / "pitbench"
    tooling.mkdir(parents=True)
    for filename in ("__init__.py", "agent_cli.py"):
        shutil.copy2(repository_root / "pitbench" / filename, tooling)
    if AgentTool.BENCH in tools:
        shutil.copytree(
            repository_root / "pitbench" / "solver_drivers", tooling / "solver_drivers"
        )

    agent_bin = task_dir / "agent_bin"
    agent_bin.mkdir()
    executable = agent_bin / "pitbench"
    executable.write_text('#!/bin/sh\nexec python3 -m pitbench.agent_cli "$@"\n')
    executable.chmod(0o755)

    config: dict[str, object] = {
        "agent_tools": sorted(tool.value for tool in tools),
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
                "development_seeds": development_seeds,
                "threads": task.evaluation.threads,
                "runner": repository.agent_run_template(),
                "runner_requirement": repository.agent_requirement,
            }
        )
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
