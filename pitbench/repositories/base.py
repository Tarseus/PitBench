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
    error: str | None = None


class RepositoryPlugin(ABC):
    name: str
    deterministic = False

    @abstractmethod
    def build_commands(self, kind: BuildKind) -> list[CommandSpec]:
        """Commands executed in a fresh code-state workspace."""

    @abstractmethod
    def run_command(self, run: SolverRunSpec) -> CommandSpec:
        """Return one normalized solver invocation."""

    def parse_output(self, path: Path) -> NormalizedSolverOutput:
        return NormalizedSolverOutput.model_validate(json.loads(path.read_text()))


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
