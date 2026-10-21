"""Versioned optimization objectives and their measurement catalog."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from importlib.resources import files
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


@lru_cache(maxsize=1)
def objective_catalog() -> dict:
    return json.loads(
        files("pitbench.schema").joinpath("objective_catalog.json").read_text()
    )


def objective_catalog_sha256() -> str:
    return hashlib.sha256(
        files("pitbench.schema").joinpath("objective_catalog.json").read_bytes()
    ).hexdigest()


@lru_cache(maxsize=1)
def qualification_protocols() -> dict:
    return json.loads(
        files("pitbench.schema").joinpath("qualification_protocols.json").read_text()
    )


def qualification_protocols_sha256() -> str:
    return hashlib.sha256(
        files("pitbench.schema").joinpath("qualification_protocols.json").read_bytes()
    ).hexdigest()


class MetricGuardrail(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metric_id: str
    comparison: Literal["absolute", "relative"]
    max_degradation: float = Field(ge=0, allow_inf_nan=False)
    instance_sets: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_metric(self) -> Self:
        if self.metric_id not in objective_catalog()["metrics"]:
            raise ValueError("unknown guardrail metric")
        if len(self.instance_sets) != len(set(self.instance_sets)):
            raise ValueError("guardrail instance sets must be unique")
        return self


class ObjectiveScoringPlan(BaseModel):
    """Qualification settings frozen in the task before collecting outcomes.

    None means the tradeoff decision has not been made. An empty list is an
    explicit declaration, but still fails the required cross-objective checks.
    """

    model_config = ConfigDict(extra="forbid")
    protocol_version: Literal["1.0"] = "1.0"
    protocols_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    qualification_instance_sets: list[str] = Field(min_length=1)
    reporting_instance_sets: list[str] = Field(min_length=1)
    guardrails: list[MetricGuardrail] | None = None

    @model_validator(mode="after")
    def validate_panels(self) -> Self:
        if self.protocols_sha256 != qualification_protocols_sha256():
            raise ValueError("objective qualification protocol hash mismatch")
        qualification = self.qualification_instance_sets
        reporting = self.reporting_instance_sets
        if any(
            len(values) != len(set(values)) for values in (qualification, reporting)
        ):
            raise ValueError("qualification/reporting panels must be unique")
        if set(qualification) & set(reporting):
            raise ValueError("qualification and final reporting panels must differ")
        keys = [
            (item.metric_id, name)
            for item in self.guardrails or []
            for name in item.instance_sets
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate guardrail for the same metric and panel")
        if any(
            set(item.instance_sets) & set(reporting) for item in self.guardrails or []
        ):
            raise ValueError(
                "final reporting panels cannot select tasks through guardrails"
            )
        return self


class OptimizationProfile(BaseModel):
    taxonomy_version: str = "1.0"
    taxonomy_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    objectives: list[str] = Field(min_length=1)
    change_kinds: list[str] = Field(default_factory=list)
    solver_phases: list[str] = Field(default_factory=list)
    classification_basis: Literal[
        "declared", "source_evidence", "heuristic_hint", "semantic_model"
    ] = "declared"
    evidence: list[str] = Field(default_factory=list)
    primary_metric_id: str | None = None
    scoring: ObjectiveScoringPlan | None = None

    @model_validator(mode="after")
    def validate_catalog_membership(self) -> Self:
        catalog = objective_catalog()
        if self.taxonomy_version != catalog["taxonomy_version"]:
            raise ValueError("unsupported optimization taxonomy version")
        if (
            self.taxonomy_sha256 is not None
            and self.taxonomy_sha256 != objective_catalog_sha256()
        ):
            raise ValueError("optimization taxonomy hash mismatch")
        for field, available in (
            ("objectives", catalog["objectives"]),
            ("change_kinds", catalog["change_kinds"]),
            ("solver_phases", catalog["solver_phases"]),
        ):
            values = getattr(self, field)
            if len(values) != len(set(values)) or set(values) - set(available):
                raise ValueError(f"invalid or duplicate optimization {field}")
        metrics = {
            metric
            for objective in self.objectives
            for metric in catalog["objectives"][objective]["metric_ids"]
        }
        if self.primary_metric_id is not None and self.primary_metric_id not in metrics:
            raise ValueError("primary metric must belong to a declared objective")
        if (
            self.primary_metric_id is not None
            and catalog["metrics"][self.primary_metric_id]["qualification"]
            == "diagnostic_only"
        ):
            raise ValueError(
                "diagnostic counters cannot be a qualification primary metric"
            )
        if (
            self.classification_basis in {"heuristic_hint", "semantic_model"}
            and self.primary_metric_id is not None
        ):
            raise ValueError(
                "unconfirmed classification hints cannot select a qualification primary metric"
            )
        return self
