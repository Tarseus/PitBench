"""Versioned quantities of interest for instances and solver runs."""

from pitbench.qoi.cvrp import (
    CVRP_INSTANCE_QOI,
    CVRP_INSTANCE_QOI_V1_0,
    CVRP_INSTANCE_QOI_V1_1,
    CVRP_INSTANCE_QOI_V2_CANDIDATE_0,
    CVRP_SOLVER_QOI,
    extract_cvrp_instance_qoi,
)
from pitbench.qoi.schema import (
    InstanceQoIObservation,
    InstanceQoISpec,
    QoIAxis,
    QoIKind,
    QoIRole,
    QoIShape,
    SolverQoISpec,
)

__all__ = [
    "CVRP_INSTANCE_QOI",
    "CVRP_INSTANCE_QOI_V1_0",
    "CVRP_INSTANCE_QOI_V1_1",
    "CVRP_INSTANCE_QOI_V2_CANDIDATE_0",
    "CVRP_SOLVER_QOI",
    "InstanceQoIObservation",
    "InstanceQoISpec",
    "QoIAxis",
    "QoIKind",
    "QoIRole",
    "QoIShape",
    "SolverQoISpec",
    "extract_cvrp_instance_qoi",
]
