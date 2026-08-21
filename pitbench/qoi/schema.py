from __future__ import annotations

import hashlib
import json
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class QoIKind(str, Enum):
    STATIC = "static"
    STRUCTURAL = "structural"
    RELAXATION = "relaxation"
    LANDMARK = "landmark"
    FEASIBILITY = "feasibility"
    QUALITY = "quality"
    PROOF = "proof"
    WORK = "work"
    RESOURCE = "resource"
    TERMINATION = "termination"


class QoIShape(str, Enum):
    SCALAR = "scalar"
    BUDGET_RESPONSE = "budget_response"
    TRAJECTORY = "trajectory"
    EVENT = "event"


class QoIRole(str, Enum):
    RAW = "raw"
    SCALE = "scale"
    STRUCT_CORE = "struct_core"
    SCALE_CONDITIONED = "scale_conditioned"
    EXPERIMENTAL = "experimental"


class QoIAxis(BaseModel):
    name: str
    description: str
    unit: str
    kind: QoIKind
    shape: QoIShape = QoIShape.SCALAR
    solver_independent: bool
    role: QoIRole | None = None


class _QoISpec(BaseModel):
    name: str
    version: str
    problem_family: str
    axes: tuple[QoIAxis, ...]

    @model_validator(mode="after")
    def unique_axes(self) -> "_QoISpec":
        names = [axis.name for axis in self.axes]
        if not names or len(names) != len(set(names)):
            raise ValueError("QoI axes must be non-empty and unique")
        if self.version != "1.0" and any(axis.role is None for axis in self.axes):
            raise ValueError("post-v1.0 QoI schema axes must declare a role")
        return self

    def fingerprint(self) -> str:
        # ``role`` was added after v1.0. Excluding absent optional fields preserves
        # the already-published v1.0 fingerprint byte-for-byte.
        payload = self.model_dump(mode="json", exclude_none=True)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


class InstanceQoISpec(_QoISpec):
    pass


class SolverQoISpec(_QoISpec):
    pass


class InstanceQoIObservation(BaseModel):
    instance_id: str
    spec_name: str
    spec_version: str
    spec_fingerprint: str
    values: dict[str, float] = Field(min_length=1)
    axis_defined: dict[str, bool] = Field(default_factory=dict)

    @classmethod
    def from_values(
        cls,
        instance_id: str,
        spec: InstanceQoISpec,
        values: dict[str, float],
        *,
        axis_defined: dict[str, bool] | None = None,
    ) -> "InstanceQoIObservation":
        expected = {axis.name for axis in spec.axes}
        if set(values) != expected:
            missing = sorted(expected - set(values))
            extra = sorted(set(values) - expected)
            raise ValueError(
                f"QoI values disagree with spec: missing={missing}, extra={extra}"
            )
        defined = axis_defined or {name: True for name in expected}
        if set(defined) != expected:
            missing = sorted(expected - set(defined))
            extra = sorted(set(defined) - expected)
            raise ValueError(
                "QoI defined flags disagree with spec: "
                f"missing={missing}, extra={extra}"
            )
        return cls(
            instance_id=instance_id,
            spec_name=spec.name,
            spec_version=spec.version,
            spec_fingerprint=spec.fingerprint(),
            values=values,
            axis_defined=defined,
        )
