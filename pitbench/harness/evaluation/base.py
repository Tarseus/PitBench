from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class EvaluationRequest(BaseModel):
    """Data transferred from the generic harness to a task evaluator."""

    task_id: str
    task_path: Path
    candidate_patch_path: Path
    candidate_patch_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    output_dir: Path
    agent_name: str
    model_name: str | None = None
    evaluator_config: dict[str, Any] = Field(default_factory=dict)
    telemetry: dict[str, Any] = Field(default_factory=dict)


class EvaluationEnvelope(BaseModel):
    """Opaque evaluator payload plus generic execution metadata."""

    evaluator: str
    evaluator_version: str
    completed: bool
    payload: dict[str, Any]
    error: str | None = None
    is_resolved: bool | None = None


class Evaluator(ABC):
    """Plugin boundary. Harness code must not interpret evaluator payloads."""

    name = "evaluator"
    version = "1"

    @abstractmethod
    def evaluate(self, request: EvaluationRequest) -> BaseModel:
        """Evaluate a captured candidate in an evaluator-owned environment."""

    def envelope(self, request: EvaluationRequest) -> EvaluationEnvelope:
        try:
            result = self.evaluate(request)
        except Exception as exc:
            return EvaluationEnvelope(
                evaluator=self.name,
                evaluator_version=self.version,
                completed=False,
                payload={},
                error=f"{type(exc).__name__}: {exc}",
                is_resolved=False,
            )
        return EvaluationEnvelope(
            evaluator=self.name,
            evaluator_version=self.version,
            completed=True,
            payload=result.model_dump(mode="json"),
            is_resolved=getattr(result, "is_resolved", None),
        )


class EvaluatorFactory:
    @staticmethod
    def from_import_path(import_path: str, **kwargs: Any) -> Evaluator:
        if ":" not in import_path:
            raise ValueError("Evaluator import path must be 'module:Class'")
        module_name, class_name = import_path.split(":", 1)
        module = importlib.import_module(module_name)
        evaluator_class = getattr(module, class_name)
        if not issubclass(evaluator_class, Evaluator):
            raise TypeError(f"{import_path} is not an Evaluator")
        return evaluator_class(**kwargs)
