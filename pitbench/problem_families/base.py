from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel


class VerificationResult(BaseModel):
    feasible: bool
    objective: float | None = None
    detail: str = ""


class ProblemFamilyPlugin(ABC):
    name: str
    _plugins: ClassVar[dict[str, type[ProblemFamilyPlugin]]] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        name = getattr(cls, "name", None)
        if name:
            existing = cls._plugins.get(name)
            if existing is not None and existing is not cls:
                raise ValueError(f"duplicate problem family plugin: {name}")
            cls._plugins[name] = cls

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
    @staticmethod
    def load(family: str) -> ProblemFamilyPlugin:
        from pitbench.problem_families import verification  # noqa: F401

        family_name = getattr(family, "value", family)
        try:
            plugin_class = ProblemFamilyPlugin._plugins[str(family_name)]
        except KeyError as error:
            raise ValueError(f"unsupported problem family: {family}") from error
        plugin = plugin_class()
        if not isinstance(plugin, ProblemFamilyPlugin):
            raise TypeError(f"{plugin_class.__name__} is not a ProblemFamilyPlugin")
        return plugin
