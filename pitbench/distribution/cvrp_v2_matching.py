from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from pitbench.distribution.confounding_recovery import (
    ConfoundingRecord,
    greedy_pairs,
    optimal_pairs,
)
from pitbench.instances.generate import make_uchoa_cvrp_instance
from pitbench.qoi.cvrp import (
    CVRP_INSTANCE_QOI_V2_CANDIDATE_0,
    extract_cvrp_instance_qoi,
)
from pitbench.qoi.schema import InstanceQoIObservation, QoIRole

CVRP_V2_MATCHING_BLOCKS: Mapping[str, tuple[str, ...]] = {
    "scale": ("customer_count",),
    "capacity_route": (
        "capacity_volume_lower_bound",
        "volume_lb_customers_per_route",
        "fleet_fill_ratio",
        "demand_mean_fraction",
        "max_demand_fraction",
    ),
    "demand_shape": ("demand_cv",),
    "global_geometry": (
        "pairwise_distance_cv",
        "distinct_distance_fraction_3dp",
    ),
    "depot_geometry": (
        "depot_centroid_distance_normalized",
        "depot_distance_mean_normalized",
        "depot_distance_iqr_normalized",
        "depot_as_nearest_neighbor_fraction",
    ),
    "local_geometry": (
        "nearest_neighbor_clark_evans_ratio",
        "nearest_neighbor_iqr_clark_evans_ratio",
        "two_nearest_neighbor_angle_median",
    ),
    "mst_topology": (
        "mst_total_length_n_corrected",
        "mst_edge_cv",
        "mst_leaf_fraction",
        "mst_depth_mean_n_corrected",
    ),
    "shape": (
        "convex_hull_area_ratio",
        "convex_hull_perimeter_ratio",
        "convex_hull_fraction",
    ),
    "clustering": (
        "dbscan_cluster_fraction",
        "dbscan_cluster_size_cv",
        "dbscan_outlier_fraction",
        "dbscan_core_fraction",
        "dbscan_within_cluster_distance_cv",
        "dbscan_max_cluster_demand_fraction",
    ),
    "coupling": (
        "demand_depot_radial_pearson",
        "demand_spatial_quadrupole_coupling",
        "demand_local_sparsity_spearman",
    ),
}

_MATCHING_ROLES = frozenset(
    {QoIRole.SCALE, QoIRole.STRUCT_CORE, QoIRole.SCALE_CONDITIONED}
)
_TREATMENT_PROFILE_ROLES = frozenset(
    {
        QoIRole.SCALE,
        QoIRole.STRUCT_CORE,
        QoIRole.SCALE_CONDITIONED,
        QoIRole.EXPERIMENTAL,
    }
)
_AXIS_ROLES = {axis.name: axis.role for axis in CVRP_INSTANCE_QOI_V2_CANDIDATE_0.axes}


@dataclass(frozen=True)
class MatchingTreatment:
    name: str
    source_level: str
    target_level: str
    affected_blocks: tuple[str, ...]


@dataclass(frozen=True)
class MatchingMethodEvidence:
    recovered_pair_count: int
    pair_recovery_rate: float
    mean_confounder_distance: float
    p95_confounder_distance: float
    mean_treatment_distance: float
    treatment_distance_retention: float


@dataclass(frozen=True)
class CVRPV2MatchingReport:
    schema_version: str
    kind: str
    conclusion: str
    claim_scope: str
    qoi_spec_status: str
    qoi_spec_version: str
    qoi_spec_fingerprint: str
    pair_count_per_treatment: int
    generator_seed: int
    matching_axis_policy: dict[str, Any]
    matching_blocks: dict[str, list[str]]
    treatments: dict[str, dict[str, Any]]
    summary: dict[str, Any]
    panel_fingerprint: str
    solver_runs_used: int
    solver_runs_created: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _AxisScale:
    center: float
    scale: float


_TREATMENTS = (
    MatchingTreatment(
        "customer_count",
        "n=80",
        "n=160",
        ("scale", "local_geometry", "mst_topology", "shape", "clustering"),
    ),
    MatchingTreatment(
        "depot_positioning",
        "C",
        "E",
        ("depot_geometry", "coupling"),
    ),
    MatchingTreatment(
        "customer_positioning",
        "R",
        "C",
        (
            "global_geometry",
            "depot_geometry",
            "local_geometry",
            "mst_topology",
            "shape",
            "clustering",
            "coupling",
        ),
    ),
    MatchingTreatment(
        "demand_dispersion",
        "small_small_variance",
        "small_large_variance",
        ("capacity_route", "demand_shape", "clustering", "coupling"),
    ),
    MatchingTreatment(
        "route_size",
        "5",
        "20",
        ("capacity_route",),
    ),
    MatchingTreatment(
        "quadrant_coupling",
        "quadrant_permuted",
        "quadrant",
        ("clustering", "coupling"),
    ),
)


def _seed(namespace: str, value: int) -> int:
    digest = hashlib.sha256(f"{namespace}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _base_parameters(pair_index: int, generator_seed: int) -> dict[str, Any]:
    rng = random.Random(_seed("v2-matching-nuisance", generator_seed + pair_index))
    return {
        "customers": rng.choice((80, 100, 120, 160)),
        "depot_positioning": rng.choice(("C", "E", "R")),
        "customer_positioning": rng.choice(("R", "C", "RC")),
        "demand_family": rng.choice(
            (
                "small_large_variance",
                "small_small_variance",
                "large_large_variance",
                "large_small_variance",
                "many_small_few_large",
                "quadrant_permuted",
            )
        ),
        "route_size": rng.choice((5.0, 10.0, 20.0, 40.0)),
        "coordinate_seed": _seed("v2-matching-coordinate", generator_seed + pair_index),
        "demand_seed": _seed("v2-matching-demand", generator_seed + pair_index),
    }


def _treatment_parameters(
    treatment: MatchingTreatment,
    base: Mapping[str, Any],
    *,
    target: bool,
) -> dict[str, Any]:
    result = dict(base)
    level = treatment.target_level if target else treatment.source_level
    if treatment.name == "customer_count":
        result["customers"] = int(level.split("=")[1])
    elif treatment.name == "depot_positioning":
        result["depot_positioning"] = level
    elif treatment.name == "customer_positioning":
        result["customer_positioning"] = level
    elif treatment.name == "demand_dispersion":
        result["demand_family"] = level
    elif treatment.name == "route_size":
        result["route_size"] = float(level)
    elif treatment.name == "quadrant_coupling":
        result["demand_family"] = level
    else:  # pragma: no cover
        raise ValueError(f"unknown matching treatment: {treatment.name}")
    return result


def _observations(
    treatment: MatchingTreatment,
    *,
    pair_count: int,
    generator_seed: int,
) -> tuple[list[InstanceQoIObservation], list[InstanceQoIObservation]]:
    source = []
    target = []
    for pair_index in range(pair_count):
        pair_id = f"{treatment.name}-{pair_index:04d}"
        base = _base_parameters(pair_index, generator_seed)
        levels = []
        for is_target in (False, True):
            parameters = _treatment_parameters(treatment, base, target=is_target)
            instance = make_uchoa_cvrp_instance(
                name=f"{pair_id}-{'target' if is_target else 'source'}",
                **parameters,
            )
            levels.append(
                extract_cvrp_instance_qoi(
                    instance,
                    instance_id=pair_id,
                    spec_version="2.0-candidate.0",
                )
            )
        source.append(levels[0])
        target.append(levels[1])
    order = list(range(pair_count))
    random.Random(
        _seed(f"v2-matching-shuffle-{treatment.name}", generator_seed)
    ).shuffle(order)
    return source, [target[index] for index in order]


def _fit_axis_scales(
    observations: Sequence[InstanceQoIObservation],
) -> dict[str, _AxisScale]:
    result = {}
    for axes in CVRP_V2_MATCHING_BLOCKS.values():
        for axis in axes:
            if not all(observation.axis_defined[axis] for observation in observations):
                continue
            values = np.asarray(
                [observation.values[axis] for observation in observations], dtype=float
            )
            center = float(np.median(values))
            scale = float(np.median(np.abs(values - center)))
            if scale <= np.finfo(float).eps:
                scale = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
            if scale <= np.finfo(float).eps:
                continue
            result[axis] = _AxisScale(center=center, scale=scale)
    return result


def _records(
    observations: Sequence[InstanceQoIObservation],
    *,
    blocks: Sequence[str],
    scales: Mapping[str, _AxisScale],
    allowed_roles: frozenset[QoIRole],
) -> tuple[list[ConfoundingRecord], tuple[str, ...]]:
    unknown = set(blocks) - set(CVRP_V2_MATCHING_BLOCKS)
    if unknown:
        raise ValueError(f"unknown matching blocks: {sorted(unknown)}")
    active = {
        block: tuple(
            axis
            for axis in CVRP_V2_MATCHING_BLOCKS[block]
            if axis in scales and _AXIS_ROLES[axis] in allowed_roles
        )
        for block in blocks
    }
    active = {block: axes for block, axes in active.items() if axes}
    if not active:
        raise ValueError("matching profile has no defined, varying QoI axes")
    block_count = len(active)
    axes = tuple(axis for block_axes in active.values() for axis in block_axes)
    records = []
    for observation in observations:
        features = {}
        for block_axes in active.values():
            weight = 1 / math.sqrt(block_count * len(block_axes))
            for axis in block_axes:
                axis_scale = scales[axis]
                features[axis] = (
                    (observation.values[axis] - axis_scale.center)
                    / axis_scale.scale
                    * weight
                )
        records.append(
            ConfoundingRecord(
                record_id=observation.instance_id,
                group="instance",
                outcome=0.0,
                features=features,
            )
        )
    return records, axes


def _distance(
    left: ConfoundingRecord,
    right: ConfoundingRecord,
    axes: Sequence[str],
) -> float:
    return math.sqrt(
        sum((left.features[axis] - right.features[axis]) ** 2 for axis in axes)
    )


def _evaluate(
    pairs: Sequence[tuple[int, int]],
    *,
    source_ids: Sequence[str],
    target_ids: Sequence[str],
    source_confounders: Sequence[ConfoundingRecord],
    target_confounders: Sequence[ConfoundingRecord],
    confounder_axes: Sequence[str],
    source_treatment: Sequence[ConfoundingRecord],
    target_treatment: Sequence[ConfoundingRecord],
    treatment_axes: Sequence[str],
    oracle_treatment_distance: float,
) -> MatchingMethodEvidence:
    confounder_distances = [
        _distance(source_confounders[left], target_confounders[right], confounder_axes)
        for left, right in pairs
    ]
    treatment_distances = [
        _distance(source_treatment[left], target_treatment[right], treatment_axes)
        for left, right in pairs
    ]
    recovered = sum(source_ids[left] == target_ids[right] for left, right in pairs)
    mean_treatment = statistics.fmean(treatment_distances)
    return MatchingMethodEvidence(
        recovered_pair_count=recovered,
        pair_recovery_rate=recovered / len(pairs),
        mean_confounder_distance=statistics.fmean(confounder_distances),
        p95_confounder_distance=float(np.quantile(confounder_distances, 0.95)),
        mean_treatment_distance=mean_treatment,
        treatment_distance_retention=(
            mean_treatment / oracle_treatment_distance
            if oracle_treatment_distance
            else 1.0
        ),
    )


def _run_treatment(
    treatment: MatchingTreatment,
    *,
    pair_count: int,
    generator_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source, target = _observations(
        treatment,
        pair_count=pair_count,
        generator_seed=generator_seed,
    )
    all_observations = [*source, *target]
    scales = _fit_axis_scales(all_observations)
    all_blocks = tuple(CVRP_V2_MATCHING_BLOCKS)
    confounder_blocks = tuple(
        block for block in all_blocks if block not in treatment.affected_blocks
    )

    source_all, all_axes = _records(
        source,
        blocks=all_blocks,
        scales=scales,
        allowed_roles=_MATCHING_ROLES,
    )
    target_all, _ = _records(
        target,
        blocks=all_blocks,
        scales=scales,
        allowed_roles=_MATCHING_ROLES,
    )
    source_confounders, confounder_axes = _records(
        source,
        blocks=confounder_blocks,
        scales=scales,
        allowed_roles=_MATCHING_ROLES,
    )
    target_confounders, _ = _records(
        target,
        blocks=confounder_blocks,
        scales=scales,
        allowed_roles=_MATCHING_ROLES,
    )
    source_treatment, treatment_axes = _records(
        source,
        blocks=treatment.affected_blocks,
        scales=scales,
        allowed_roles=_TREATMENT_PROFILE_ROLES,
    )
    target_treatment, _ = _records(
        target,
        blocks=treatment.affected_blocks,
        scales=scales,
        allowed_roles=_TREATMENT_PROFILE_ROLES,
    )

    source_ids = [observation.instance_id for observation in source]
    target_ids = [observation.instance_id for observation in target]
    target_by_id = {record_id: index for index, record_id in enumerate(target_ids)}
    oracle = tuple(
        (index, target_by_id[record_id]) for index, record_id in enumerate(source_ids)
    )
    oracle_treatment_distance = statistics.fmean(
        _distance(source_treatment[left], target_treatment[right], treatment_axes)
        for left, right in oracle
    )
    methods = {
        "oracle_crn": oracle,
        "unconditioned_greedy": greedy_pairs(source_all, target_all, axes=all_axes),
        "unconditioned_ot": optimal_pairs(source_all, target_all, axes=all_axes),
        "conditioned_greedy": greedy_pairs(
            source_confounders, target_confounders, axes=confounder_axes
        ),
        "conditioned_ot": optimal_pairs(
            source_confounders, target_confounders, axes=confounder_axes
        ),
    }
    evidence = {
        name: asdict(
            _evaluate(
                pairs,
                source_ids=source_ids,
                target_ids=target_ids,
                source_confounders=source_confounders,
                target_confounders=target_confounders,
                confounder_axes=confounder_axes,
                source_treatment=source_treatment,
                target_treatment=target_treatment,
                treatment_axes=treatment_axes,
                oracle_treatment_distance=oracle_treatment_distance,
            )
        )
        for name, pairs in methods.items()
    }
    payload = {
        "source_level": treatment.source_level,
        "target_level": treatment.target_level,
        "affected_blocks": list(treatment.affected_blocks),
        "confounder_blocks": list(confounder_blocks),
        "active_unconditioned_axes": list(all_axes),
        "active_confounder_axes": list(confounder_axes),
        "active_treatment_profile_axes": list(treatment_axes),
        "methods": evidence,
    }
    panel = {
        "treatment": treatment.name,
        "source": [observation.model_dump(mode="json") for observation in source],
        "target": [observation.model_dump(mode="json") for observation in target],
    }
    return payload, panel


def run_cvrp_v2_matching_validation(
    *,
    pair_count: int = 48,
    generator_seed: int = 20260821,
) -> CVRPV2MatchingReport:
    """Validate treatment-conditioned matching on the implemented v2 candidate."""

    if pair_count < 4:
        raise ValueError("v2 matching validation requires at least four pairs")
    evidence = {}
    panels = []
    for treatment in _TREATMENTS:
        treatment_evidence, panel = _run_treatment(
            treatment,
            pair_count=pair_count,
            generator_seed=generator_seed,
        )
        evidence[treatment.name] = treatment_evidence
        panels.append(panel)

    unconditioned = [item["methods"]["unconditioned_ot"] for item in evidence.values()]
    conditioned = [item["methods"]["conditioned_ot"] for item in evidence.values()]
    recovery_gains = [
        right["pair_recovery_rate"] - left["pair_recovery_rate"]
        for left, right in zip(unconditioned, conditioned, strict=True)
    ]
    confounder_gains = [
        left["mean_confounder_distance"] - right["mean_confounder_distance"]
        for left, right in zip(unconditioned, conditioned, strict=True)
    ]
    summary = {
        "treatment_count": len(evidence),
        "mean_unconditioned_ot_pair_recovery": statistics.fmean(
            item["pair_recovery_rate"] for item in unconditioned
        ),
        "mean_conditioned_ot_pair_recovery": statistics.fmean(
            item["pair_recovery_rate"] for item in conditioned
        ),
        "mean_pair_recovery_gain": statistics.fmean(recovery_gains),
        "mean_unconditioned_ot_confounder_distance": statistics.fmean(
            item["mean_confounder_distance"] for item in unconditioned
        ),
        "mean_conditioned_ot_confounder_distance": statistics.fmean(
            item["mean_confounder_distance"] for item in conditioned
        ),
        "mean_confounder_distance_reduction": statistics.fmean(confounder_gains),
        "treatments_with_strict_recovery_gain_and_no_worse_confounding": sum(
            recovery > 0 and imbalance >= -1e-12
            for recovery, imbalance in zip(
                recovery_gains, confounder_gains, strict=True
            )
        ),
        "treatments_with_no_pair_recovery_loss": sum(
            recovery >= 0 for recovery in recovery_gains
        ),
        "treatments_with_no_worse_confounding": sum(
            imbalance >= -1e-12 for imbalance in confounder_gains
        ),
    }
    conclusion = (
        "Supported with treatment-specific incremental value for v2-candidate.0"
        if summary["mean_pair_recovery_gain"] > 0
        and summary["mean_confounder_distance_reduction"] >= 0
        else "Not supported for v2-candidate.0 on this synthetic hidden-CRN panel"
    )
    return CVRPV2MatchingReport(
        schema_version="2.0",
        kind="cvrp_v2_experiment_conditioned_matching_validation",
        conclusion=conclusion,
        claim_scope=(
            "Solver-free definition and hidden-pair recovery for v2-candidate.0 only; "
            "this does not freeze v2.0 or validate solver-response effects."
        ),
        qoi_spec_status="candidate.0_not_frozen",
        qoi_spec_version=CVRP_INSTANCE_QOI_V2_CANDIDATE_0.version,
        qoi_spec_fingerprint=CVRP_INSTANCE_QOI_V2_CANDIDATE_0.fingerprint(),
        pair_count_per_treatment=pair_count,
        generator_seed=generator_seed,
        matching_axis_policy={
            "excluded_roles": [QoIRole.RAW.value, QoIRole.EXPERIMENTAL.value],
            "conditional_role": QoIRole.SCALE_CONDITIONED.value,
            "affected_blocks_excluded_from_conditioned_cost": True,
            "block_weighting": "equal_blocks_equal_axes_within_block",
            "normalization": "pooled_median_mad_with_iqr_fallback",
        },
        matching_blocks={
            name: list(axes) for name, axes in CVRP_V2_MATCHING_BLOCKS.items()
        },
        treatments=evidence,
        summary=summary,
        panel_fingerprint=_hash(panels),
        solver_runs_used=0,
        solver_runs_created=0,
    )


__all__ = [
    "CVRP_V2_MATCHING_BLOCKS",
    "CVRPV2MatchingReport",
    "MatchingMethodEvidence",
    "MatchingTreatment",
    "run_cvrp_v2_matching_validation",
]
