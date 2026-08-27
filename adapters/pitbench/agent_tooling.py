from __future__ import annotations

import json
import shutil
from pathlib import Path

from pitbench.evaluator.patch_policy import PatchPolicy
from pitbench.repositories.base import BuildKind, RepositoryPluginRegistry
from pitbench.schema.task import PitBenchTask


def _runner(
    module: str, *, solver: str | None = None, trajectory: bool = True
) -> list[str]:
    command = ["python3", "-m", module]
    if solver is not None:
        command.extend(["--solver", solver])
    command.extend(["--instance", "{instance}", "--output", "{output}"])
    if trajectory:
        command.extend(["--trajectory", "{trajectory}"])
    command.extend(
        [
            "--seed",
            "{seed}",
            "--budget",
            "{budget}",
            "--threads",
            "{threads}",
        ]
    )
    return command


_RUNNERS = {
    "pitbench.repositories.pyvrp:PyVRPRepositoryPlugin": _runner(
        "pitbench.drivers.pyvrp"
    ),
    "pitbench.repositories.vroom:VroomRepositoryPlugin": _runner(
        "pitbench.drivers.vroom", solver="./bin/vroom"
    ),
    "pitbench.repositories.highs:HighsRepositoryPlugin": _runner(
        "pitbench.drivers.highs", solver="./build/bin/highs"
    ),
    "pitbench.repositories.choco:ChocoRepositoryPlugin": _runner(
        "pitbench.drivers.choco"
    ),
    "pitbench.repositories.ortools:OrToolsRepositoryPlugin": _runner(
        "pitbench.drivers.ortools_model_build", trajectory=False
    ),
}

_REQUIREMENTS = {
    "pitbench.repositories.pyvrp:PyVRPRepositoryPlugin": "pyvrp_import",
    "pitbench.repositories.vroom:VroomRepositoryPlugin": "file:bin/vroom",
    "pitbench.repositories.highs:HighsRepositoryPlugin": "file:build/bin/highs",
    "pitbench.repositories.choco:ChocoRepositoryPlugin": "env:PITBENCH_CHOCO_RUNNER",
    "pitbench.repositories.ortools:OrToolsRepositoryPlugin": (
        "env:PITBENCH_ORTOOLS_JAVA_RUNNER"
    ),
}


def write_agent_tooling(
    *, repository_root: Path, task: PitBenchTask, task_dir: Path
) -> None:
    """Bundle only public developer tooling into the agent task."""
    tooling = task_dir / "agent_tooling" / "pitbench"
    tooling.mkdir(parents=True)
    for filename in ("__init__.py", "agent_cli.py"):
        shutil.copy2(repository_root / "pitbench" / filename, tooling)
    shutil.copytree(repository_root / "pitbench" / "drivers", tooling / "drivers")

    agent_bin = task_dir / "agent_bin"
    agent_bin.mkdir()
    executable = agent_bin / "pitbench"
    executable.write_text('#!/bin/sh\nexec python3 -m pitbench.agent_cli "$@"\n')
    executable.chmod(0o755)

    repository = RepositoryPluginRegistry.load(task.repository.plugin)
    config = {
        "task_id": task.task_id,
        "task_type": task.task_type.value,
        "problem_family": task.problem_family.value,
        "budgets_sec": task.evaluation.budgets_sec,
        "solver_seeds": task.evaluation.solver_seeds,
        "threads": task.evaluation.threads,
        "runner": _RUNNERS[task.repository.plugin],
        "runner_requirement": _REQUIREMENTS[task.repository.plugin],
        "protected_paths": list(PatchPolicy.DEFAULT_PROTECTED),
        "validation_commands": [
            command.model_dump(mode="json")
            for command in repository.build_commands(BuildKind.VALIDATION)
        ],
    }
    (task_dir / "agent_config.json").write_text(json.dumps(config, indent=2) + "\n")
