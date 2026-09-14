from __future__ import annotations

import importlib
import json
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class BuildKind(str, Enum):
    VALIDATION = "validation"
    PERFORMANCE = "performance"


class SolverTermination(str, Enum):
    OPTIMAL = "optimal"
    TIME_LIMIT = "time_limit"
    ERROR = "solver_error"
    OTHER = "other"


class CommandSpec(BaseModel):
    argv: list[str]
    cwd: str = "."
    env: dict[str, str] = Field(default_factory=dict)
    timeout_sec: float = Field(default=1800, gt=0)


class SolverRunSpec(BaseModel):
    instance_path: Path
    output_path: Path
    trajectory_path: Path
    solver_seed: int
    budget_sec: float
    threads: int


class NormalizedSolverOutput(BaseModel):
    valid: bool
    has_solution: bool | None = None
    failure_reason: str | None = None
    objective: float | None = None
    primal_bound: float | None = None
    dual_bound: float | None = None
    iterations: int | None = None
    nodes: int | None = None
    model_variables: int | None = None
    model_constraints: int | None = None
    wall_time_sec: float | None = None
    cpu_time_sec: float | None = None
    peak_rss_bytes: int | None = None
    resource_scope: str | None = None
    solver_status: str | None = None
    error: str | None = None


class AgentEnvironment(BaseModel):
    image: str
    system_packages: str = ""
    python_packages: str = ""
    prebuild: str = ""


class RepositoryPlugin(ABC):
    name: str
    driver_name: str
    driver_python = "python"
    driver_solver: str | None = None
    driver_records_trajectory = True
    deterministic = False
    agent_environment: AgentEnvironment | None = None
    agent_requirement: str | None = None
    agent_python: str | None = None
    collection_backend: str | None = None
    representations: dict[str, tuple[str, ...]] = {}

    @abstractmethod
    def build_commands(self, kind: BuildKind) -> list[CommandSpec]:
        """Commands executed in a fresh code-state workspace."""

    def run_command(self, run: SolverRunSpec) -> CommandSpec:
        """Return one normalized invocation through the shared driver entry."""
        argv = [
            self.driver_python,
            "-m",
            "pitbench.solver_drivers.run",
            self.driver_name,
        ]
        if self.driver_solver is not None:
            argv.extend(["--solver", self.driver_solver])
        argv.extend(
            ["--instance", str(run.instance_path), "--output", str(run.output_path)]
        )
        if self.driver_records_trajectory:
            argv.extend(["--trajectory", str(run.trajectory_path)])
        argv.extend(
            [
                "--seed",
                str(run.solver_seed),
                "--budget",
                str(run.budget_sec),
                "--threads",
                str(run.threads),
            ]
        )
        return CommandSpec(argv=argv, timeout_sec=run.budget_sec + 60)

    def parse_output(self, path: Path) -> NormalizedSolverOutput:
        return NormalizedSolverOutput.model_validate(json.loads(path.read_text()))

    def agent_run_template(self) -> list[str]:
        """Use the same invocation contract for public development runs."""
        command = self.run_command(
            SolverRunSpec(
                instance_path=Path("{instance}"),
                output_path=Path("{output}"),
                trajectory_path=Path("{trajectory}"),
                solver_seed=0,
                budget_sec=1,
                threads=1,
            )
        )
        argv = list(command.argv)
        if self.agent_python is not None:
            argv[0] = self.agent_python
        for option, field in (
            ("--seed", "seed"),
            ("--budget", "budget"),
            ("--threads", "threads"),
        ):
            if option in argv:
                argv[argv.index(option) + 1] = "{" + field + "}"
        return argv

    def load_collection_backend(self):
        if self.collection_backend is None:
            raise ValueError(f"{self.name} does not provide isolated collection")
        module, name = self.collection_backend.split(":", 1)
        backend = getattr(importlib.import_module(module), name)
        from pitbench.evaluator.collection import CollectionBackend

        if not isinstance(backend, type) or not issubclass(backend, CollectionBackend):
            raise TypeError(f"{self.collection_backend} is not a CollectionBackend")
        return backend


class RepositoryPluginRegistry:
    @staticmethod
    def load(import_path: str) -> RepositoryPlugin:
        if ":" not in import_path:
            raise ValueError("repository plugin must be 'module:Class'")
        module_name, class_name = import_path.split(":", 1)
        plugin_class: Any = getattr(importlib.import_module(module_name), class_name)
        plugin = plugin_class()
        if not isinstance(plugin, RepositoryPlugin):
            raise TypeError(f"{import_path} is not a RepositoryPlugin")
        return plugin
