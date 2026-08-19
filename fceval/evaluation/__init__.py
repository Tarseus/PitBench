"""Evaluator-agnostic boundary used by the execution harness."""

from fceval.evaluation.base import (
    EvaluationEnvelope,
    EvaluationRequest,
    Evaluator,
    EvaluatorFactory,
)

__all__ = [
    "EvaluationEnvelope",
    "EvaluationRequest",
    "Evaluator",
    "EvaluatorFactory",
]
