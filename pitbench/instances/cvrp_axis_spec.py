from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pitbench.instances.generate import make_euclidean_cvrp_instance
from pitbench.qoi.cvrp import CVRP_INSTANCE_QOI_V1_0, extract_cvrp_instance_qoi


class LiteratureStatus(StrEnum):
    CVRP_PRIMITIVE = "cvrp_primitive"
    CVRP_ESTABLISHED = "cvrp_established"
    CROSS_DOMAIN_ADAPTED = "cross_domain_adapted"
    PROJECT_SPECIFIC = "project_specific"


class InterventionStatus(StrEnum):
    EXACT_SINGLE_QOI = "exact_single_qoi"
    COMPOUND = "compound"
    DESCRIPTOR_ONLY = "descriptor_only"


@dataclass(frozen=True)
class CVRPAxisSpecification:
    qoi_name: str
    definition: str
    literature_status: LiteratureStatus
    intervention_status: InterventionStatus
    treatment_axis: str | None
    unavoidable_collateral_qois: tuple[str, ...]
    audit_verdict: str


@dataclass(frozen=True)
class SingleQoIInterventionCase:
    target_qoi: str
    pair_id: str
    generator_seed: int
    source_level: str
    target_level: str
    source: dict[str, Any]
    target: dict[str, Any]
    realized_qoi_delta: float


def _spec(
    qoi_name: str,
    definition: str,
    literature_status: LiteratureStatus,
    intervention_status: InterventionStatus,
    *,
    treatment_axis: str | None = None,
    collateral: tuple[str, ...] = (),
    verdict: str,
) -> CVRPAxisSpecification:
    return CVRPAxisSpecification(
        qoi_name=qoi_name,
        definition=definition,
        literature_status=literature_status,
        intervention_status=intervention_status,
        treatment_axis=treatment_axis,
        unavoidable_collateral_qois=collateral,
        audit_verdict=verdict,
    )


CVRP_QOI_V1_0_AXIS_SPECIFICATIONS = (
    _spec(
        "customer_count",
        "n",
        LiteratureStatus.CVRP_PRIMITIVE,
        InterventionStatus.COMPOUND,
        treatment_axis="scale",
        collateral=("total_demand", "vehicle_lower_bound", "demand_mean_fraction"),
        verdict=(
            "Standard primitive, but adding positive-demand customers is not "
            "single-QoI."
        ),
    ),
    _spec(
        "capacity",
        "Q",
        LiteratureStatus.CVRP_PRIMITIVE,
        InterventionStatus.COMPOUND,
        treatment_axis="capacity_pressure",
        collateral=(
            "vehicle_lower_bound",
            "fleet_fill_ratio",
            "demand_mean_fraction",
        ),
        verdict="Standard primitive with deterministic downstream QoI changes.",
    ),
    _spec(
        "total_demand",
        "D = sum_i q_i",
        LiteratureStatus.CVRP_ESTABLISHED,
        InterventionStatus.COMPOUND,
        treatment_axis="demand_scale",
        collateral=(
            "vehicle_lower_bound",
            "fleet_fill_ratio",
            "demand_mean_fraction",
        ),
        verdict="Standard aggregate, not an independently manipulable construct.",
    ),
    _spec(
        "vehicle_lower_bound",
        "ceil(D / Q)",
        LiteratureStatus.CVRP_ESTABLISHED,
        InterventionStatus.DESCRIPTOR_ONLY,
        treatment_axis="capacity_pressure",
        collateral=("capacity", "total_demand", "fleet_fill_ratio"),
        verdict="Valid volume lower bound, but not the bin-packing fleet minimum.",
    ),
    _spec(
        "fleet_fill_ratio",
        "D / (ceil(D / Q) Q)",
        LiteratureStatus.PROJECT_SPECIFIC,
        InterventionStatus.DESCRIPTOR_ONLY,
        treatment_axis="capacity_pressure",
        collateral=("capacity", "total_demand", "vehicle_lower_bound"),
        verdict=(
            "Related to loading, but uses a volume lower bound rather than an "
            "actual fleet."
        ),
    ),
    _spec(
        "demand_mean_fraction",
        "mean(q) / Q = D / (n Q)",
        LiteratureStatus.CVRP_ESTABLISHED,
        InterventionStatus.DESCRIPTOR_ONLY,
        treatment_axis="capacity_pressure",
        collateral=("customer_count", "capacity", "total_demand"),
        verdict=(
            "Established demand-to-capacity feature with an exact algebraic dependency."
        ),
    ),
    _spec(
        "demand_cv",
        "std(q) / mean(q)",
        LiteratureStatus.CVRP_ESTABLISHED,
        InterventionStatus.EXACT_SINGLE_QOI,
        treatment_axis="demand_dispersion",
        verdict=(
            "Exact intervention is possible while holding demand sum and radial "
            "moments fixed."
        ),
    ),
    _spec(
        "pairwise_distance_median",
        "median_{i<j,d_ij>0}(d_ij)",
        LiteratureStatus.CROSS_DOMAIN_ADAPTED,
        InterventionStatus.EXACT_SINGLE_QOI,
        treatment_axis="coordinate_scale",
        verdict=(
            "Distance summaries are established; this robust median choice is "
            "project-specific."
        ),
    ),
    _spec(
        "depot_distance_mean_normalized",
        "mean_i d(0,i) / median_{i<j}(d_ij)",
        LiteratureStatus.CROSS_DOMAIN_ADAPTED,
        InterventionStatus.COMPOUND,
        treatment_axis="depot_position",
        collateral=(
            "depot_distance_iqr_normalized",
            "demand_depot_correlation",
            "demand_weighted_depot_ratio",
        ),
        verdict=(
            "Depot-distance means are established; median normalization is "
            "project-specific."
        ),
    ),
    _spec(
        "depot_distance_iqr_normalized",
        "IQR_i d(0,i) / median_{i<j}(d_ij)",
        LiteratureStatus.PROJECT_SPECIFIC,
        InterventionStatus.COMPOUND,
        treatment_axis="depot_position",
        collateral=("depot_distance_mean_normalized",),
        verdict=(
            "Robust spread statistic, but no direct CVRP precedent for this exact "
            "definition."
        ),
    ),
    _spec(
        "nearest_distance_mean_normalized",
        "mean_i min_{j != i} d(i,j) / median_{i<j}(d_ij)",
        LiteratureStatus.CROSS_DOMAIN_ADAPTED,
        InterventionStatus.COMPOUND,
        treatment_axis="customer_structure",
        collateral=("mst_edge_mean_normalized", "convex_hull_fraction"),
        verdict=(
            "Nearest-neighbour summaries are established; normalization is "
            "project-specific."
        ),
    ),
    _spec(
        "nearest_distance_iqr_normalized",
        "IQR_i min_{j != i} d(i,j) / median_{i<j}(d_ij)",
        LiteratureStatus.PROJECT_SPECIFIC,
        InterventionStatus.COMPOUND,
        treatment_axis="customer_structure",
        collateral=("nearest_distance_mean_normalized",),
        verdict=(
            "IQR variant is reasonable but lacks direct validation as a standalone "
            "treatment."
        ),
    ),
    _spec(
        "mst_edge_mean_normalized",
        "mean(MST edge length) / median_{i<j}(d_ij)",
        LiteratureStatus.CROSS_DOMAIN_ADAPTED,
        InterventionStatus.COMPOUND,
        treatment_axis="customer_structure",
        collateral=("nearest_distance_mean_normalized", "convex_hull_fraction"),
        verdict=(
            "MST summaries are established; the current normalization leaks "
            "finite-sample size."
        ),
    ),
    _spec(
        "convex_hull_fraction",
        "number of hull customers / n",
        LiteratureStatus.CVRP_ESTABLISHED,
        InterventionStatus.COMPOUND,
        treatment_axis="customer_structure",
        collateral=("nearest_distance_mean_normalized", "mst_edge_mean_normalized"),
        verdict=(
            "Established geometric feature, but point mutations also alter other "
            "geometry QoIs."
        ),
    ),
    _spec(
        "demand_depot_correlation",
        "corr(q_i, d(0,i))",
        LiteratureStatus.PROJECT_SPECIFIC,
        InterventionStatus.COMPOUND,
        treatment_axis="radial_demand_coupling",
        collateral=("demand_weighted_depot_ratio",),
        verdict=(
            "Interpretable radial coupling, but not independent of the weighted ratio."
        ),
    ),
    _spec(
        "demand_weighted_depot_ratio",
        "weighted_mean_q(d(0,i)) / mean_i d(0,i)",
        LiteratureStatus.PROJECT_SPECIFIC,
        InterventionStatus.COMPOUND,
        treatment_axis="radial_demand_coupling",
        collateral=("demand_depot_correlation",),
        verdict="A normalized radial first moment, algebraically linked to covariance.",
    ),
)


def _changed_qois(
    source: dict[str, Any], target: dict[str, Any], *, tolerance: float = 1e-9
) -> tuple[str, ...]:
    source_qoi = extract_cvrp_instance_qoi(source, spec_version="1.0").values
    target_qoi = extract_cvrp_instance_qoi(target, spec_version="1.0").values
    return tuple(
        name
        for name in source_qoi
        if not math.isclose(
            source_qoi[name], target_qoi[name], rel_tol=tolerance, abs_tol=tolerance
        )
    )


def _distance_scale_pair(seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    source = make_euclidean_cvrp_instance(
        name=f"pairwise-distance-source-{seed}",
        customers=80,
        coordinate_seed=seed,
        demand_seed=seed + 100_000,
        capacity_ratio=0.08,
    )
    target = copy.deepcopy(source)
    target["name"] = f"pairwise-distance-target-{seed}"
    target["coordinates"] = [
        [2 * coordinate[0], 2 * coordinate[1]] for coordinate in source["coordinates"]
    ]
    return source, target


def _integer_circle_points(radius: int = 65) -> list[list[float]]:
    points = set()
    for x_coord in range(-radius, radius + 1):
        y_squared = radius * radius - x_coord * x_coord
        y_coord = math.isqrt(y_squared)
        if y_coord * y_coord == y_squared:
            points.add((x_coord, y_coord))
            points.add((x_coord, -y_coord))
    return [[float(x_coord), float(y_coord)] for x_coord, y_coord in sorted(points)]


def _demand_cv_pair(seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    points = _integer_circle_points()
    if len(points) % 2:
        raise AssertionError("circle construction must have even cardinality")
    random.Random(seed).shuffle(points)
    source_demands = [10] * len(points)
    target_demands = [5] * (len(points) // 2) + [15] * (len(points) // 2)
    random.Random(seed + 100_000).shuffle(target_demands)
    common = {
        "depot": 0,
        "coordinates": [[0.0, 0.0], *points],
        "capacity": 60,
    }
    source = {
        **common,
        "name": f"demand-cv-source-{seed}",
        "demands": [0, *source_demands],
    }
    target = {
        **common,
        "name": f"demand-cv-target-{seed}",
        "demands": [0, *target_demands],
    }
    return source, target


def build_exact_single_qoi_panel(
    seeds: tuple[int, ...],
) -> tuple[SingleQoIInterventionCase, ...]:
    """Build paired interventions proven to change one v1.0 QoI within tolerance."""

    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("single-QoI panel seeds must be non-empty and unique")
    definitions = (
        (
            "pairwise_distance_median",
            "coordinate_scale=1",
            "coordinate_scale=2",
            _distance_scale_pair,
        ),
        ("demand_cv", "CV=0", "CV=0.5", _demand_cv_pair),
    )
    cases = []
    for target_qoi, source_level, target_level, builder in definitions:
        for seed in seeds:
            source, target = builder(seed)
            changed = _changed_qois(source, target)
            if changed != (target_qoi,):
                raise AssertionError(
                    f"{target_qoi} intervention changed unexpected QoIs: {changed}"
                )
            source_qoi = extract_cvrp_instance_qoi(source, spec_version="1.0").values[
                target_qoi
            ]
            target_qoi_value = extract_cvrp_instance_qoi(
                target, spec_version="1.0"
            ).values[target_qoi]
            cases.append(
                SingleQoIInterventionCase(
                    target_qoi=target_qoi,
                    pair_id=f"{target_qoi}-{seed}",
                    generator_seed=seed,
                    source_level=source_level,
                    target_level=target_level,
                    source=source,
                    target=target,
                    realized_qoi_delta=target_qoi_value - source_qoi,
                )
            )
    return tuple(cases)


def validate_axis_specification_catalog() -> None:
    names = tuple(item.qoi_name for item in CVRP_QOI_V1_0_AXIS_SPECIFICATIONS)
    expected = tuple(axis.name for axis in CVRP_INSTANCE_QOI_V1_0.axes)
    if names != expected:
        raise AssertionError("axis specification catalog does not match QoI v1.0")


validate_axis_specification_catalog()


__all__ = [
    "CVRP_QOI_V1_0_AXIS_SPECIFICATIONS",
    "CVRPAxisSpecification",
    "InterventionStatus",
    "LiteratureStatus",
    "SingleQoIInterventionCase",
    "build_exact_single_qoi_panel",
    "validate_axis_specification_catalog",
]
