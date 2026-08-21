from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from pitbench.qoi.schema import (
    InstanceQoIObservation,
    InstanceQoISpec,
    QoIAxis,
    QoIKind,
    QoIShape,
    SolverQoISpec,
)


def _axis(
    name: str,
    description: str,
    unit: str,
    kind: QoIKind,
    *,
    shape: QoIShape = QoIShape.SCALAR,
    solver_independent: bool = True,
) -> QoIAxis:
    return QoIAxis(
        name=name,
        description=description,
        unit=unit,
        kind=kind,
        shape=shape,
        solver_independent=solver_independent,
    )


CVRP_INSTANCE_QOI = InstanceQoISpec(
    name="cvrp-instance-qoi",
    version="1.0",
    problem_family="cvrp",
    axes=(
        _axis("customer_count", "Number of customers", "customers", QoIKind.STATIC),
        _axis("capacity", "Vehicle capacity", "demand units", QoIKind.STATIC),
        _axis("total_demand", "Sum of customer demand", "demand units", QoIKind.STATIC),
        _axis(
            "vehicle_lower_bound",
            "Capacity-only lower bound on required vehicles",
            "vehicles",
            QoIKind.STRUCTURAL,
        ),
        _axis(
            "fleet_fill_ratio",
            "Total demand divided by capacity of the lower-bound fleet",
            "ratio",
            QoIKind.STRUCTURAL,
        ),
        _axis(
            "demand_mean_fraction",
            "Mean customer demand divided by capacity",
            "ratio",
            QoIKind.STRUCTURAL,
        ),
        _axis(
            "demand_cv",
            "Coefficient of variation of customer demands",
            "ratio",
            QoIKind.STRUCTURAL,
        ),
        _axis(
            "pairwise_distance_median",
            "Median positive customer-to-customer distance",
            "coordinate units",
            QoIKind.STRUCTURAL,
        ),
        _axis(
            "depot_distance_mean_normalized",
            "Mean depot distance divided by median pairwise distance",
            "ratio",
            QoIKind.STRUCTURAL,
        ),
        _axis(
            "depot_distance_iqr_normalized",
            "IQR of depot distances divided by median pairwise distance",
            "ratio",
            QoIKind.STRUCTURAL,
        ),
        _axis(
            "nearest_distance_mean_normalized",
            "Mean nearest-customer distance divided by median pairwise distance",
            "ratio",
            QoIKind.STRUCTURAL,
        ),
        _axis(
            "nearest_distance_iqr_normalized",
            "IQR of nearest-customer distances divided by median pairwise distance",
            "ratio",
            QoIKind.STRUCTURAL,
        ),
        _axis(
            "mst_edge_mean_normalized",
            "Mean customer MST edge divided by median pairwise distance",
            "ratio",
            QoIKind.STRUCTURAL,
        ),
        _axis(
            "convex_hull_fraction",
            "Fraction of customers on the convex hull",
            "ratio",
            QoIKind.STRUCTURAL,
        ),
        _axis(
            "demand_depot_correlation",
            "Pearson correlation between demand and depot distance",
            "correlation",
            QoIKind.STRUCTURAL,
        ),
        _axis(
            "demand_weighted_depot_ratio",
            "Demand-weighted mean depot distance divided by unweighted mean",
            "ratio",
            QoIKind.STRUCTURAL,
        ),
    ),
)

# A versioned specification is immutable.  This independently pinned value makes
# an axis or metadata change fail validation until the version is deliberately
# bumped and its new fingerprint is reviewed and recorded.
CVRP_INSTANCE_QOI_PINNED_FINGERPRINTS: Mapping[str, str] = {
    "1.0": "f924d33c1a81754934fb8f97c9ba27b62c04312025a9cb6f33820687a87f14c0",
}


CVRP_SOLVER_QOI = SolverQoISpec(
    name="cvrp-solver-qoi",
    version="1.0",
    problem_family="cvrp",
    axes=(
        _axis(
            "feasible_rate",
            "Probability of producing an independently verified solution",
            "probability",
            QoIKind.FEASIBILITY,
            shape=QoIShape.BUDGET_RESPONSE,
            solver_independent=False,
        ),
        _axis(
            "conditional_gap",
            "Mean oracle-relative gap conditional on feasibility",
            "relative gap",
            QoIKind.QUALITY,
            shape=QoIShape.BUDGET_RESPONSE,
            solver_independent=False,
        ),
        _axis(
            "wall_time_fraction",
            "Wall time divided by declared budget",
            "ratio",
            QoIKind.RESOURCE,
            shape=QoIShape.BUDGET_RESPONSE,
            solver_independent=False,
        ),
        _axis(
            "iterations",
            "Reported solver iterations",
            "iterations",
            QoIKind.WORK,
            shape=QoIShape.BUDGET_RESPONSE,
            solver_independent=False,
        ),
    ),
)


def _iqr(values: np.ndarray) -> float:
    return float(np.quantile(values, 0.75) - np.quantile(values, 0.25))


def _mst_mean_edge(pairwise: np.ndarray) -> float:
    size = len(pairwise)
    if size <= 1:
        return 0.0
    selected = np.zeros(size, dtype=bool)
    selected[0] = True
    best = pairwise[0].copy()
    best[0] = math.inf
    total = 0.0
    for _ in range(size - 1):
        node = int(np.argmin(np.where(selected, math.inf, best)))
        total += float(best[node])
        selected[node] = True
        best = np.minimum(best, pairwise[node])
    return total / (size - 1)


def _cross(
    first: tuple[float, float], second: tuple[float, float], point: tuple[float, float]
) -> float:
    return (second[0] - first[0]) * (point[1] - first[1]) - (second[1] - first[1]) * (
        point[0] - first[0]
    )


def _convex_hull_fraction(points: np.ndarray) -> float:
    unique = sorted({(float(x), float(y)) for x, y in points})
    if len(unique) <= 2:
        return len(unique) / len(points)
    lower: list[tuple[float, float]] = []
    upper: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(unique):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return len(lower[:-1] + upper[:-1]) / len(points)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def extract_cvrp_instance_qoi(
    instance: Mapping[str, Any],
    *,
    instance_id: str | None = None,
) -> InstanceQoIObservation:
    coordinates = np.asarray(instance["coordinates"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacity = float(instance["capacity"])
    depot = int(instance.get("depot", 0))
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 2
        or len(coordinates) != len(demands)
        or not 0 <= depot < len(coordinates)
        or capacity <= 0
    ):
        raise ValueError("invalid normalized CVRP instance")
    mask = np.arange(len(coordinates)) != depot
    points = coordinates[mask]
    customer_demands = demands[mask]
    if not len(points) or np.any(customer_demands < 0):
        raise ValueError("CVRP requires customers with non-negative demands")

    pairwise = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    upper = pairwise[np.triu_indices(len(points), k=1)]
    positive = upper[upper > 0]
    characteristic = float(np.median(positive)) if positive.size else 1.0
    depot_distances = np.linalg.norm(points - coordinates[depot], axis=1)
    if len(points) > 1:
        nearest = np.partition(pairwise, kth=1, axis=1)[:, 1]
    else:
        nearest = np.zeros(1, dtype=float)
    total_demand = float(np.sum(customer_demands))
    vehicles = math.ceil(total_demand / capacity)
    mean_demand = float(np.mean(customer_demands))
    mean_depot = float(np.mean(depot_distances))
    weighted_depot = (
        float(np.average(depot_distances, weights=customer_demands))
        if total_demand > 0
        else mean_depot
    )
    values = {
        "customer_count": float(len(points)),
        "capacity": capacity,
        "total_demand": total_demand,
        "vehicle_lower_bound": float(vehicles),
        "fleet_fill_ratio": (total_demand / (vehicles * capacity) if vehicles else 0.0),
        "demand_mean_fraction": mean_demand / capacity,
        "demand_cv": (
            float(np.std(customer_demands)) / mean_demand if mean_demand else 0.0
        ),
        "pairwise_distance_median": characteristic,
        "depot_distance_mean_normalized": mean_depot / characteristic,
        "depot_distance_iqr_normalized": _iqr(depot_distances) / characteristic,
        "nearest_distance_mean_normalized": float(np.mean(nearest)) / characteristic,
        "nearest_distance_iqr_normalized": _iqr(nearest) / characteristic,
        "mst_edge_mean_normalized": _mst_mean_edge(pairwise) / characteristic,
        "convex_hull_fraction": _convex_hull_fraction(points),
        "demand_depot_correlation": _correlation(customer_demands, depot_distances),
        "demand_weighted_depot_ratio": (
            weighted_depot / mean_depot if mean_depot else 1.0
        ),
    }
    return InstanceQoIObservation.from_values(
        instance_id or str(instance.get("name", "unnamed")),
        CVRP_INSTANCE_QOI,
        values,
    )
