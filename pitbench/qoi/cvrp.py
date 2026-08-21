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
    QoIRole,
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
    role: QoIRole | None = None,
) -> QoIAxis:
    return QoIAxis(
        name=name,
        description=description,
        unit=unit,
        kind=kind,
        shape=shape,
        solver_independent=solver_independent,
        role=role,
    )


CVRP_INSTANCE_QOI_V1_0 = InstanceQoISpec(
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


def _instance_axis(
    name: str,
    description: str,
    unit: str,
    role: QoIRole,
) -> QoIAxis:
    return _axis(
        name,
        description,
        unit,
        QoIKind.STATIC if role in {QoIRole.RAW, QoIRole.SCALE} else QoIKind.STRUCTURAL,
        role=role,
    )


CVRP_INSTANCE_QOI_V1_1 = InstanceQoISpec(
    name="cvrp-instance-qoi",
    version="1.1",
    problem_family="cvrp",
    axes=(
        _instance_axis(
            "customer_count", "Number of customers", "customers", QoIRole.SCALE
        ),
        _instance_axis("capacity", "Vehicle capacity", "demand units", QoIRole.RAW),
        _instance_axis(
            "total_demand", "Sum of customer demand", "demand units", QoIRole.RAW
        ),
        _instance_axis(
            "capacity_volume_lower_bound",
            "Ceiling of total demand divided by capacity",
            "routes",
            QoIRole.SCALE,
        ),
        _instance_axis(
            "fleet_fill_ratio",
            "Total demand divided by capacity of the volume-lower-bound fleet",
            "ratio",
            QoIRole.STRUCT_CORE,
        ),
        _instance_axis(
            "volume_lb_customers_per_route",
            "Customers divided by the capacity-volume route lower bound",
            "customers per route",
            QoIRole.STRUCT_CORE,
        ),
        _instance_axis(
            "demand_mean_fraction",
            "Mean customer demand divided by capacity",
            "ratio",
            QoIRole.SCALE_CONDITIONED,
        ),
        _instance_axis(
            "max_demand_fraction",
            "Maximum customer demand divided by capacity",
            "ratio",
            QoIRole.STRUCT_CORE,
        ),
        _instance_axis(
            "demand_cv",
            "Coefficient of variation of customer demands",
            "ratio",
            QoIRole.SCALE_CONDITIONED,
        ),
        _instance_axis(
            "pairwise_distance_median",
            "Median positive customer-to-customer distance",
            "coordinate units",
            QoIRole.RAW,
        ),
        _instance_axis(
            "depot_distance_mean_normalized",
            "Mean depot distance divided by median pairwise distance",
            "ratio",
            QoIRole.STRUCT_CORE,
        ),
        _instance_axis(
            "depot_distance_iqr_normalized",
            "IQR of depot distances divided by median pairwise distance",
            "ratio",
            QoIRole.STRUCT_CORE,
        ),
        _instance_axis(
            "nearest_neighbor_clark_evans_ratio",
            "Mean nearest-neighbor distance divided by the hull-area CSR reference",
            "ratio",
            QoIRole.SCALE_CONDITIONED,
        ),
        _instance_axis(
            "nearest_neighbor_iqr_clark_evans_ratio",
            "Nearest-neighbor IQR divided by the hull-area CSR reference",
            "ratio",
            QoIRole.SCALE_CONDITIONED,
        ),
        _instance_axis(
            "mst_edge_mean_n_corrected",
            "Mean MST edge times sqrt(n), divided by median pairwise distance",
            "ratio",
            QoIRole.SCALE_CONDITIONED,
        ),
        _instance_axis(
            "convex_hull_area_ratio",
            "Customer convex-hull area divided by squared median pairwise distance",
            "ratio",
            QoIRole.SCALE_CONDITIONED,
        ),
        _instance_axis(
            "demand_depot_radial_pearson",
            "Pearson correlation between demand and depot distance",
            "correlation",
            QoIRole.STRUCT_CORE,
        ),
        _instance_axis(
            "demand_depot_radial_weighted_ratio",
            "Demand-weighted mean depot radius divided by its unweighted mean",
            "ratio",
            QoIRole.STRUCT_CORE,
        ),
        _instance_axis(
            "demand_spatial_quadrupole_coupling",
            "Rotation-invariant traceless second-order demand-location coupling",
            "ratio",
            QoIRole.EXPERIMENTAL,
        ),
    ),
)


CVRP_INSTANCE_QOI_V2_CANDIDATE_0 = InstanceQoISpec(
    name="cvrp-static-qoi-basis",
    version="2.0-candidate.0",
    problem_family="cvrp",
    axes=(
        _instance_axis(
            "customer_count", "Number of customers", "customers", QoIRole.SCALE
        ),
        _instance_axis("capacity", "Vehicle capacity", "demand units", QoIRole.RAW),
        _instance_axis(
            "total_demand", "Sum of customer demand", "demand units", QoIRole.RAW
        ),
        _instance_axis(
            "capacity_volume_lower_bound",
            "Ceiling of total demand divided by capacity",
            "routes",
            QoIRole.SCALE,
        ),
        _instance_axis(
            "volume_lb_customers_per_route",
            "Customers divided by the capacity-volume route lower bound",
            "customers per route",
            QoIRole.STRUCT_CORE,
        ),
        _instance_axis(
            "pairwise_distance_median",
            "Median positive customer-to-customer distance",
            "coordinate units",
            QoIRole.RAW,
        ),
        _instance_axis(
            "fleet_fill_ratio",
            "Total demand divided by capacity of the volume-lower-bound fleet",
            "ratio",
            QoIRole.STRUCT_CORE,
        ),
        _instance_axis(
            "demand_mean_fraction",
            "Mean customer demand divided by capacity",
            "ratio",
            QoIRole.SCALE_CONDITIONED,
        ),
        _instance_axis(
            "max_demand_fraction",
            "Maximum customer demand divided by capacity",
            "ratio",
            QoIRole.STRUCT_CORE,
        ),
        _instance_axis(
            "demand_cv",
            "Coefficient of variation of customer demands",
            "ratio",
            QoIRole.STRUCT_CORE,
        ),
        _instance_axis(
            "pairwise_distance_cv",
            "Coefficient of variation of customer pairwise distances",
            "ratio",
            QoIRole.STRUCT_CORE,
        ),
        _instance_axis(
            "distinct_distance_fraction_3dp",
            "Fraction of normalized pairwise distances distinct at three decimals",
            "ratio",
            QoIRole.EXPERIMENTAL,
        ),
        _instance_axis(
            "depot_centroid_distance_normalized",
            "Depot-to-customer-centroid distance divided by median pairwise distance",
            "ratio",
            QoIRole.STRUCT_CORE,
        ),
        _instance_axis(
            "depot_distance_mean_normalized",
            "Mean depot distance divided by median pairwise distance",
            "ratio",
            QoIRole.STRUCT_CORE,
        ),
        _instance_axis(
            "depot_distance_iqr_normalized",
            "IQR of depot distances divided by median pairwise distance",
            "ratio",
            QoIRole.STRUCT_CORE,
        ),
        _instance_axis(
            "nearest_neighbor_clark_evans_ratio",
            "Mean nearest-neighbor distance divided by the hull-area CSR reference",
            "ratio",
            QoIRole.SCALE_CONDITIONED,
        ),
        _instance_axis(
            "nearest_neighbor_iqr_clark_evans_ratio",
            "Nearest-neighbor IQR divided by the hull-area CSR reference",
            "ratio",
            QoIRole.SCALE_CONDITIONED,
        ),
        _instance_axis(
            "two_nearest_neighbor_angle_median",
            "Median first-two-neighbor angle divided by pi",
            "ratio",
            QoIRole.STRUCT_CORE,
        ),
        _instance_axis(
            "depot_as_nearest_neighbor_fraction",
            "Fraction of customers whose nearest node is the depot",
            "ratio",
            QoIRole.STRUCT_CORE,
        ),
        _instance_axis(
            "mst_total_length_n_corrected",
            "Customer MST total length times sqrt(n), divided by (n-1)s",
            "ratio",
            QoIRole.SCALE_CONDITIONED,
        ),
        _instance_axis(
            "mst_edge_cv",
            "Coefficient of variation of customer MST edge lengths",
            "ratio",
            QoIRole.SCALE_CONDITIONED,
        ),
        _instance_axis(
            "mst_leaf_fraction",
            "Fraction of customer MST nodes with degree one",
            "ratio",
            QoIRole.SCALE_CONDITIONED,
        ),
        _instance_axis(
            "mst_depth_mean_n_corrected",
            "Mean depth in the depot-rooted MST divided by sqrt(n)",
            "ratio",
            QoIRole.EXPERIMENTAL,
        ),
        _instance_axis(
            "convex_hull_area_ratio",
            "Customer convex-hull area divided by squared median pairwise distance",
            "ratio",
            QoIRole.SCALE_CONDITIONED,
        ),
        _instance_axis(
            "convex_hull_perimeter_ratio",
            "Customer convex-hull perimeter divided by median pairwise distance",
            "ratio",
            QoIRole.SCALE_CONDITIONED,
        ),
        _instance_axis(
            "convex_hull_fraction",
            "Fraction of customers on the convex hull",
            "ratio",
            QoIRole.SCALE_CONDITIONED,
        ),
        _instance_axis(
            "dbscan_cluster_fraction",
            "DBSCAN non-noise cluster count divided by customer count",
            "ratio",
            QoIRole.EXPERIMENTAL,
        ),
        _instance_axis(
            "dbscan_cluster_size_cv",
            "Coefficient of variation of non-noise DBSCAN cluster sizes",
            "ratio",
            QoIRole.EXPERIMENTAL,
        ),
        _instance_axis(
            "dbscan_outlier_fraction",
            "Fraction of customers labeled as DBSCAN noise",
            "ratio",
            QoIRole.EXPERIMENTAL,
        ),
        _instance_axis(
            "dbscan_core_fraction",
            "Fraction of customers labeled as DBSCAN core points",
            "ratio",
            QoIRole.EXPERIMENTAL,
        ),
        _instance_axis(
            "dbscan_within_cluster_distance_cv",
            "CV of distances from clustered customers to their cluster centroids",
            "ratio",
            QoIRole.EXPERIMENTAL,
        ),
        _instance_axis(
            "dbscan_max_cluster_demand_fraction",
            "Largest DBSCAN cluster demand divided by capacity",
            "ratio",
            QoIRole.EXPERIMENTAL,
        ),
        _instance_axis(
            "demand_depot_radial_pearson",
            "Pearson correlation between demand and depot distance",
            "correlation",
            QoIRole.STRUCT_CORE,
        ),
        _instance_axis(
            "demand_spatial_quadrupole_coupling",
            "Rotation-invariant traceless second-order demand-location coupling",
            "ratio",
            QoIRole.EXPERIMENTAL,
        ),
        _instance_axis(
            "demand_local_sparsity_spearman",
            "Spearman correlation between demand and fourth-neighbor distance",
            "correlation",
            QoIRole.EXPERIMENTAL,
        ),
    ),
)

# The unqualified name intentionally follows the latest schema. Published v1.0
# artifacts and callers can select the immutable v1.0 object explicitly.
CVRP_INSTANCE_QOI = CVRP_INSTANCE_QOI_V1_1

# A versioned specification is immutable.  This independently pinned value makes
# an axis or metadata change fail validation until the version is deliberately
# bumped and its new fingerprint is reviewed and recorded.
CVRP_INSTANCE_QOI_PINNED_FINGERPRINTS: Mapping[str, str] = {
    "1.0": "f924d33c1a81754934fb8f97c9ba27b62c04312025a9cb6f33820687a87f14c0",
    "1.1": "5c099976c936e502f42ea13aadb0b611ee2927a26f34ec0028a10a924bdfe549",
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


def _convex_hull(points: np.ndarray) -> list[tuple[float, float]]:
    unique = sorted({(float(x), float(y)) for x, y in points})
    if len(unique) <= 2:
        return unique
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
    return lower[:-1] + upper[:-1]


def _convex_hull_fraction(points: np.ndarray) -> float:
    return len(_convex_hull(points)) / len(points)


def _convex_hull_area(points: np.ndarray) -> float:
    hull = _convex_hull(points)
    if len(hull) < 3:
        return 0.0
    return (
        abs(
            sum(
                first[0] * second[1] - second[0] * first[1]
                for first, second in zip(hull, [*hull[1:], hull[0]], strict=True)
            )
        )
        / 2
    )


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _quadrupole_coupling(points: np.ndarray, demands: np.ndarray) -> tuple[float, bool]:
    demand_std = float(np.std(demands))
    centered = points - np.mean(points, axis=0)
    mean_radius_squared = float(np.mean(np.sum(centered**2, axis=1)))
    if demand_std == 0 or mean_radius_squared == 0:
        return 0.0, False
    standardized_demands = (demands - np.mean(demands)) / demand_std
    moment = np.zeros((2, 2), dtype=float)
    identity = np.eye(2)
    for demand, point in zip(standardized_demands, centered, strict=True):
        outer = np.outer(point, point)
        traceless = outer - np.dot(point, point) * identity / 2
        moment += demand * traceless
    moment /= len(points) * mean_radius_squared
    return float(np.linalg.norm(moment, ord="fro")), True


def _correlation_with_defined(
    left: np.ndarray, right: np.ndarray
) -> tuple[float, bool]:
    defined = len(left) >= 2 and float(np.std(left)) > 0 and float(np.std(right)) > 0
    return (_correlation(left, right), defined)


def _distance_signature_keys(
    pairwise: np.ndarray, *, depot_index: int | None = None
) -> tuple[tuple[float | int, ...], ...]:
    positive = pairwise[pairwise > 0]
    scale = float(np.median(positive)) if positive.size else 1.0
    keys = []
    for node, distances in enumerate(pairwise):
        role = 0 if node == depot_index else 1
        signature = tuple(
            float(value) for value in np.round(np.sort(distances) / scale, 10)
        )
        keys.append((role, *signature))
    return tuple(keys)


def _mst_edges_and_parents(
    pairwise: np.ndarray, *, depot_index: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    size = len(pairwise)
    if size <= 1:
        return np.asarray([], dtype=float), np.full(size, -1, dtype=int)
    selected = np.zeros(size, dtype=bool)
    selected[0] = True
    best = pairwise[0].copy()
    parents = np.zeros(size, dtype=int)
    parents[0] = -1
    edges = []
    positive = pairwise[pairwise > 0]
    scale = float(np.median(positive)) if positive.size else 1.0
    keys = _distance_signature_keys(pairwise, depot_index=depot_index)
    for _ in range(size - 1):
        candidates = [
            (round(float(best[node]) / scale, 10), keys[node], node)
            for node in range(size)
            if not selected[node]
        ]
        _, _, node = min(candidates)
        edges.append(float(best[node]))
        selected[node] = True
        for other in range(size):
            if selected[other]:
                continue
            weight = float(pairwise[node, other])
            weight_key = round(weight / scale, 10)
            best_key = round(float(best[other]) / scale, 10)
            if weight_key < best_key or (
                weight_key == best_key and keys[node] < keys[parents[other]]
            ):
                best[other] = weight
                parents[other] = node
    return np.asarray(edges, dtype=float), parents


def _mst_is_unique(pairwise: np.ndarray) -> bool:
    size = len(pairwise)
    if size <= 1:
        return True
    positive = pairwise[pairwise > 0]
    scale = float(np.median(positive)) if positive.size else 1.0
    edges = sorted(
        (
            round(float(pairwise[left, right]) / scale, 10),
            left,
            right,
        )
        for left in range(size)
        for right in range(left + 1, size)
    )
    parent = list(range(size))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    start = 0
    while start < len(edges):
        end = start + 1
        while end < len(edges) and edges[end][0] == edges[start][0]:
            end += 1
        component_parent: dict[int, int] = {}

        def component_find(component: int) -> int:
            component_parent.setdefault(component, component)
            while component_parent[component] != component:
                component_parent[component] = component_parent[
                    component_parent[component]
                ]
                component = component_parent[component]
            return component

        candidates = []
        for _, left, right in edges[start:end]:
            left_root = find(left)
            right_root = find(right)
            if left_root == right_root:
                continue
            candidates.append((left_root, right_root))
            left_group = component_find(left_root)
            right_group = component_find(right_root)
            if left_group == right_group:
                return False
            component_parent[right_group] = left_group
        for left_root, right_root in candidates:
            left_root = find(left_root)
            right_root = find(right_root)
            if left_root != right_root:
                parent[right_root] = left_root
        start = end
    return True


def _tree_depths(parents: np.ndarray, *, root: int = 0) -> np.ndarray:
    depths = np.full(len(parents), math.nan, dtype=float)
    depths[root] = 0.0
    for node in range(len(parents)):
        if not math.isnan(float(depths[node])):
            continue
        path = []
        cursor = node
        while math.isnan(float(depths[cursor])):
            path.append(cursor)
            cursor = int(parents[cursor])
        depth = float(depths[cursor])
        for member in reversed(path):
            depth += 1
            depths[member] = depth
    return depths


def _convex_hull_perimeter(points: np.ndarray) -> float:
    hull = _convex_hull(points)
    if len(hull) < 2:
        return 0.0
    if len(hull) == 2:
        return 2 * math.dist(hull[0], hull[1])
    return sum(
        math.dist(first, second)
        for first, second in zip(hull, [*hull[1:], hull[0]], strict=True)
    )


def _minimum_oriented_rectangle_area(points: np.ndarray) -> float:
    hull = np.asarray(_convex_hull(points), dtype=float)
    if len(hull) < 3:
        return 0.0
    minimum = math.inf
    for first, second in zip(hull, np.vstack((hull[1:], hull[0])), strict=True):
        edge = second - first
        angle = math.atan2(float(edge[1]), float(edge[0]))
        cosine = math.cos(angle)
        sine = math.sin(angle)
        rotation = np.asarray(((cosine, sine), (-sine, cosine)))
        rotated = hull @ rotation.T
        spans = np.max(rotated, axis=0) - np.min(rotated, axis=0)
        minimum = min(minimum, float(spans[0] * spans[1]))
    return minimum


def _two_nearest_neighbor_angle(points: np.ndarray) -> tuple[float, bool]:
    if len(points) < 3:
        return 0.0, False
    pairwise = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    angles = []
    for node in range(len(points)):
        neighbors = sorted(
            (float(pairwise[node, other]), other)
            for other in range(len(points))
            if other != node
        )[:2]
        vectors = [points[other] - points[node] for _, other in neighbors]
        norms = [float(np.linalg.norm(vector)) for vector in vectors]
        if min(norms) <= np.finfo(float).eps:
            return 0.0, False
        cosine = float(np.dot(vectors[0], vectors[1]) / (norms[0] * norms[1]))
        angles.append(math.acos(max(-1.0, min(1.0, cosine))) / math.pi)
    return float(np.median(angles)), True


def _depot_as_nearest_fraction(
    points: np.ndarray, depot_point: np.ndarray
) -> tuple[float, bool]:
    if not len(points):
        return 0.0, False
    all_points = np.vstack((depot_point, points))
    pairwise = np.linalg.norm(points[:, None, :] - all_points[None, :, :], axis=2)
    for customer in range(len(points)):
        pairwise[customer, customer + 1] = math.inf
    nearest = np.argmin(pairwise, axis=1)
    return float(np.mean(nearest == 0)), True


def _dbscan_labels(
    points: np.ndarray, *, epsilon: float, min_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    pairwise = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    neighborhoods = [
        np.flatnonzero(pairwise[node] <= epsilon + np.finfo(float).eps)
        for node in range(len(points))
    ]
    core = np.asarray(
        [len(neighbors) >= min_samples for neighbors in neighborhoods], dtype=bool
    )
    labels = np.full(len(points), -1, dtype=int)
    cluster = 0
    for seed in range(len(points)):
        if not core[seed] or labels[seed] >= 0:
            continue
        labels[seed] = cluster
        queue = [seed]
        cursor = 0
        while cursor < len(queue):
            node = queue[cursor]
            cursor += 1
            if not core[node]:
                continue
            for neighbor in neighborhoods[node]:
                neighbor = int(neighbor)
                if labels[neighbor] < 0:
                    labels[neighbor] = cluster
                    if core[neighbor]:
                        queue.append(neighbor)
        cluster += 1
    return labels, core


def _dbscan_features(
    points: np.ndarray, demands: np.ndarray, capacity: float
) -> tuple[dict[str, float], dict[str, bool]]:
    names = (
        "dbscan_cluster_fraction",
        "dbscan_cluster_size_cv",
        "dbscan_outlier_fraction",
        "dbscan_core_fraction",
        "dbscan_within_cluster_distance_cv",
        "dbscan_max_cluster_demand_fraction",
    )
    values = {name: 0.0 for name in names}
    defined = {name: False for name in names}
    area = _minimum_oriented_rectangle_area(points)
    if len(points) <= 1 or area <= 0:
        return values, defined
    epsilon = math.sqrt(area) / (math.sqrt(len(points)) - 1)
    labels, core = _dbscan_labels(points, epsilon=epsilon, min_samples=4)
    clusters = sorted(set(int(label) for label in labels if label >= 0))
    values["dbscan_cluster_fraction"] = len(clusters) / len(points)
    values["dbscan_outlier_fraction"] = float(np.mean(labels < 0))
    values["dbscan_core_fraction"] = float(np.mean(core))
    for name in (
        "dbscan_cluster_fraction",
        "dbscan_outlier_fraction",
        "dbscan_core_fraction",
    ):
        defined[name] = True
    if not clusters:
        return values, defined
    sizes = np.asarray([np.sum(labels == cluster) for cluster in clusters], dtype=float)
    if len(sizes) >= 2 and float(np.mean(sizes)) > 0:
        values["dbscan_cluster_size_cv"] = float(np.std(sizes) / np.mean(sizes))
        defined["dbscan_cluster_size_cv"] = True
    centroid_distances = []
    cluster_demands = []
    for cluster in clusters:
        members = points[labels == cluster]
        centroid = np.mean(members, axis=0)
        centroid_distances.extend(np.linalg.norm(members - centroid, axis=1))
        cluster_demands.append(float(np.sum(demands[labels == cluster])))
    distance_array = np.asarray(centroid_distances, dtype=float)
    if len(distance_array) and float(np.mean(distance_array)) > 0:
        values["dbscan_within_cluster_distance_cv"] = float(
            np.std(distance_array) / np.mean(distance_array)
        )
        defined["dbscan_within_cluster_distance_cv"] = True
    values["dbscan_max_cluster_demand_fraction"] = max(cluster_demands) / capacity
    defined["dbscan_max_cluster_demand_fraction"] = True
    return values, defined


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def _local_sparsity_spearman(
    pairwise: np.ndarray, demands: np.ndarray
) -> tuple[float, bool]:
    if len(pairwise) <= 4:
        return 0.0, False
    distances = pairwise.copy()
    np.fill_diagonal(distances, math.inf)
    fourth_neighbor = np.sort(distances, axis=1)[:, 3]
    return _correlation_with_defined(
        _average_ranks(demands), _average_ranks(fourth_neighbor)
    )


def _extract_cvrp_v2_candidate(
    *,
    instance: Mapping[str, Any],
    instance_id: str | None,
    points: np.ndarray,
    customer_demands: np.ndarray,
    depot_point: np.ndarray,
    pairwise: np.ndarray,
    positive_pairwise: np.ndarray,
    characteristic: float,
    depot_distances: np.ndarray,
    nearest: np.ndarray,
    capacity: float,
    total_demand: float,
    vehicles: int,
) -> InstanceQoIObservation:
    customer_count = len(points)
    mean_demand = float(np.mean(customer_demands))
    hull_area = _convex_hull_area(points)
    hull_perimeter = _convex_hull_perimeter(points)
    clark_evans_defined = customer_count >= 3 and hull_area > 0
    csr_distance = (
        0.5 * math.sqrt(hull_area / customer_count) if clark_evans_defined else 1.0
    )
    pairwise_mean = float(np.mean(positive_pairwise)) if positive_pairwise.size else 0.0
    pairwise_cv_defined = pairwise_mean > 0
    customer_mst_unique = _mst_is_unique(pairwise)
    mst_edges, _ = _mst_edges_and_parents(pairwise)
    mst_mean = float(np.mean(mst_edges)) if len(mst_edges) else 0.0
    mst_edge_cv_defined = mst_mean > 0
    customer_degrees = np.zeros(customer_count, dtype=int)
    _, customer_parents = _mst_edges_and_parents(pairwise)
    for node in range(1, customer_count):
        parent = int(customer_parents[node])
        customer_degrees[node] += 1
        customer_degrees[parent] += 1
    all_points = np.vstack((depot_point, points))
    all_pairwise = np.linalg.norm(
        all_points[:, None, :] - all_points[None, :, :], axis=2
    )
    rooted_mst_unique = _mst_is_unique(all_pairwise)
    _, rooted_parents = _mst_edges_and_parents(all_pairwise, depot_index=0)
    depths = _tree_depths(rooted_parents)
    angle, angle_defined = _two_nearest_neighbor_angle(points)
    depot_nearest, depot_nearest_defined = _depot_as_nearest_fraction(
        points, depot_point
    )
    radial, radial_defined = _correlation_with_defined(
        customer_demands, depot_distances
    )
    quadrupole, quadrupole_defined = _quadrupole_coupling(points, customer_demands)
    sparsity, sparsity_defined = _local_sparsity_spearman(pairwise, customer_demands)
    dbscan_values, dbscan_defined = _dbscan_features(points, customer_demands, capacity)
    values = {
        "customer_count": float(customer_count),
        "capacity": capacity,
        "total_demand": total_demand,
        "capacity_volume_lower_bound": float(vehicles),
        "volume_lb_customers_per_route": customer_count / vehicles if vehicles else 0.0,
        "pairwise_distance_median": characteristic,
        "fleet_fill_ratio": total_demand / (vehicles * capacity) if vehicles else 0.0,
        "demand_mean_fraction": mean_demand / capacity,
        "max_demand_fraction": float(np.max(customer_demands)) / capacity,
        "demand_cv": float(np.std(customer_demands)) / mean_demand
        if mean_demand
        else 0.0,
        "pairwise_distance_cv": (
            float(np.std(positive_pairwise)) / pairwise_mean
            if pairwise_cv_defined
            else 0.0
        ),
        "distinct_distance_fraction_3dp": (
            len(np.unique(np.round(positive_pairwise / characteristic, 3)))
            / len(positive_pairwise)
            if positive_pairwise.size
            else 0.0
        ),
        "depot_centroid_distance_normalized": float(
            np.linalg.norm(depot_point - np.mean(points, axis=0)) / characteristic
        ),
        "depot_distance_mean_normalized": float(np.mean(depot_distances))
        / characteristic,
        "depot_distance_iqr_normalized": _iqr(depot_distances) / characteristic,
        "nearest_neighbor_clark_evans_ratio": (
            float(np.mean(nearest)) / csr_distance if clark_evans_defined else 0.0
        ),
        "nearest_neighbor_iqr_clark_evans_ratio": (
            _iqr(nearest) / csr_distance if clark_evans_defined else 0.0
        ),
        "two_nearest_neighbor_angle_median": angle,
        "depot_as_nearest_neighbor_fraction": depot_nearest,
        "mst_total_length_n_corrected": (
            math.sqrt(customer_count)
            * float(np.sum(mst_edges))
            / ((customer_count - 1) * characteristic)
            if customer_count > 1
            else 0.0
        ),
        "mst_edge_cv": (
            float(np.std(mst_edges)) / mst_mean if mst_edge_cv_defined else 0.0
        ),
        "mst_leaf_fraction": (
            float(np.mean(customer_degrees == 1))
            if customer_count > 1 and customer_mst_unique
            else 0.0
        ),
        "mst_depth_mean_n_corrected": (
            float(np.mean(depths[1:])) / math.sqrt(customer_count)
            if rooted_mst_unique
            else 0.0
        ),
        "convex_hull_area_ratio": hull_area / characteristic**2,
        "convex_hull_perimeter_ratio": hull_perimeter / characteristic,
        "convex_hull_fraction": _convex_hull_fraction(points),
        **dbscan_values,
        "demand_depot_radial_pearson": radial,
        "demand_spatial_quadrupole_coupling": quadrupole,
        "demand_local_sparsity_spearman": sparsity,
    }
    axis_defined = {name: True for name in values}
    axis_defined.update(dbscan_defined)
    axis_defined.update(
        {
            "pairwise_distance_cv": pairwise_cv_defined,
            "distinct_distance_fraction_3dp": bool(positive_pairwise.size),
            "nearest_neighbor_clark_evans_ratio": clark_evans_defined,
            "nearest_neighbor_iqr_clark_evans_ratio": clark_evans_defined,
            "two_nearest_neighbor_angle_median": angle_defined,
            "depot_as_nearest_neighbor_fraction": depot_nearest_defined,
            "mst_total_length_n_corrected": customer_count > 1,
            "mst_edge_cv": mst_edge_cv_defined,
            "mst_leaf_fraction": customer_count > 1 and customer_mst_unique,
            "mst_depth_mean_n_corrected": rooted_mst_unique,
            "convex_hull_area_ratio": hull_area > 0,
            "convex_hull_perimeter_ratio": hull_perimeter > 0,
            "convex_hull_fraction": customer_count > 0,
            "demand_depot_radial_pearson": radial_defined,
            "demand_spatial_quadrupole_coupling": quadrupole_defined,
            "demand_local_sparsity_spearman": sparsity_defined,
        }
    )
    return InstanceQoIObservation.from_values(
        instance_id or str(instance.get("name", "unnamed")),
        CVRP_INSTANCE_QOI_V2_CANDIDATE_0,
        values,
        axis_defined=axis_defined,
    )


def extract_cvrp_instance_qoi(
    instance: Mapping[str, Any],
    *,
    instance_id: str | None = None,
    spec_version: str = "1.1",
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
    common = {
        "customer_count": float(len(points)),
        "capacity": capacity,
        "total_demand": total_demand,
        "fleet_fill_ratio": (total_demand / (vehicles * capacity) if vehicles else 0.0),
        "demand_mean_fraction": mean_demand / capacity,
        "demand_cv": (
            float(np.std(customer_demands)) / mean_demand if mean_demand else 0.0
        ),
        "pairwise_distance_median": characteristic,
        "depot_distance_mean_normalized": mean_depot / characteristic,
        "depot_distance_iqr_normalized": _iqr(depot_distances) / characteristic,
    }
    if spec_version == "2.0-candidate.0":
        return _extract_cvrp_v2_candidate(
            instance=instance,
            instance_id=instance_id,
            points=points,
            customer_demands=customer_demands,
            depot_point=coordinates[depot],
            pairwise=pairwise,
            positive_pairwise=positive,
            characteristic=characteristic,
            depot_distances=depot_distances,
            nearest=nearest,
            capacity=capacity,
            total_demand=total_demand,
            vehicles=vehicles,
        )
    if spec_version == "1.0":
        values = {
            **common,
            "vehicle_lower_bound": float(vehicles),
            "nearest_distance_mean_normalized": float(np.mean(nearest))
            / characteristic,
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
            CVRP_INSTANCE_QOI_V1_0,
            values,
        )
    if spec_version != "1.1":
        raise ValueError(f"unsupported CVRP QoI spec version: {spec_version}")

    hull_area = _convex_hull_area(points)
    clark_evans_defined = len(points) >= 3 and hull_area > 0
    csr_distance = 0.5 * math.sqrt(hull_area / len(points)) if hull_area > 0 else 1.0
    quadrupole, quadrupole_defined = _quadrupole_coupling(points, customer_demands)
    values = {
        **common,
        "capacity_volume_lower_bound": float(vehicles),
        "volume_lb_customers_per_route": len(points) / vehicles if vehicles else 0.0,
        "max_demand_fraction": float(np.max(customer_demands)) / capacity,
        "nearest_neighbor_clark_evans_ratio": (
            float(np.mean(nearest)) / csr_distance if clark_evans_defined else 0.0
        ),
        "nearest_neighbor_iqr_clark_evans_ratio": (
            _iqr(nearest) / csr_distance if clark_evans_defined else 0.0
        ),
        "mst_edge_mean_n_corrected": (
            math.sqrt(len(points)) * _mst_mean_edge(pairwise) / characteristic
        ),
        "convex_hull_area_ratio": hull_area / characteristic**2,
        "demand_depot_radial_pearson": _correlation(customer_demands, depot_distances),
        "demand_depot_radial_weighted_ratio": (
            weighted_depot / mean_depot if mean_depot else 1.0
        ),
        "demand_spatial_quadrupole_coupling": quadrupole,
    }
    axis_defined = {name: True for name in values}
    axis_defined["nearest_neighbor_clark_evans_ratio"] = clark_evans_defined
    axis_defined["nearest_neighbor_iqr_clark_evans_ratio"] = clark_evans_defined
    axis_defined["demand_spatial_quadrupole_coupling"] = quadrupole_defined
    return InstanceQoIObservation.from_values(
        instance_id or str(instance.get("name", "unnamed")),
        CVRP_INSTANCE_QOI_V1_1,
        values,
        axis_defined=axis_defined,
    )
