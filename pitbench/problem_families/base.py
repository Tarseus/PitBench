from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class VerificationResult(BaseModel):
    feasible: bool
    objective: float | None = None
    detail: str = ""


class ProblemFamilyPlugin(ABC):
    name: str

    @abstractmethod
    def verify(self, instance_path: Path, solution_path: Path) -> VerificationResult:
        """Verify independently of the solver implementation."""

    @staticmethod
    def normalized_gap(
        objective: float | None,
        anchor: float | None,
        *,
        epsilon: float = 1e-12,
    ) -> float | None:
        if objective is None or anchor is None:
            return None
        return (objective - anchor) / (abs(anchor) + epsilon)


class ProblemFamilyRegistry:
    _PLUGIN_PATHS = {
        "cvrp": "pitbench.problem_families.cvrp:CVRPFamily",
        "mip": "pitbench.problem_families.external:MIPFamily",
        "cp": "pitbench.problem_families.external:CPFamily",
    }

    @staticmethod
    def load(family: str) -> ProblemFamilyPlugin:
        try:
            import_path = ProblemFamilyRegistry._PLUGIN_PATHS[family]
        except KeyError as error:
            raise ValueError(f"unsupported problem family: {family}") from error
        if ":" not in import_path:
            raise ValueError("family plugin must be 'module:Class'")
        module_name, class_name = import_path.split(":", 1)
        plugin_class: Any = getattr(importlib.import_module(module_name), class_name)
        plugin = plugin_class()
        if not isinstance(plugin, ProblemFamilyPlugin):
            raise TypeError(f"{import_path} is not a ProblemFamilyPlugin")
        return plugin
