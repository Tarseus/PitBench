"""Solver-free, axis-level validation of the CVRP instance QoI specification.

This module tests the ground-geometry axioms from the methodology without first
combining the QoI axes into a universal distance.  The result therefore says
which coordinates are safe invariants, which respond to controlled structural
directions, and which leak computational scale.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Literal

import numpy as np

from pitbench.distribution.transforms import (
    reflect_cvrp,
    relabel_cvrp_customers,
    rotate_cvrp,
    translate_cvrp,
)
from pitbench.instances import make_euclidean_cvrp_instance
from pitbench.qoi.cvrp import (
    CVRP_INSTANCE_QOI,
    CVRP_INSTANCE_QOI_PINNED_FINGERPRINTS,
    extract_cvrp_instance_qoi,
)
from pitbench.qoi.schema import InstanceQoIObservation, InstanceQoISpec

RAW_UNIT_AXES = (
    "capacity",
    "total_demand",
    "pairwise_distance_median",
)
SCALE_AXES = ("customer_count",)
STRUCTURAL_AXES = tuple(
    axis.name
    for axis in CVRP_INSTANCE_QOI.axes
    if axis.name not in {*RAW_UNIT_AXES, *SCALE_AXES}
)
UNIT_ROBUST_AXES = (*SCALE_AXES, *STRUCTURAL_AXES)


@dataclass(frozen=True)
class QoIAxiomThresholds:
    """Pre-registered gates; values are not adapted after seeing the panel."""

    numerical_standardized_tolerance: float = 1e-9
    equivalence_unrelated_ratio_max: float = 0.05
    scale_monotonicity_spearman_min: float = 0.99
    scale_leakage_median_iqr_max: float = 0.25
    scale_stable_axis_fraction_min: float = 0.75
    controlled_effect_median_iqr_min: float = 0.50
    controlled_direction_fraction_min: float = 0.75
    controlled_direction_pass_fraction_min: float = 0.80
    unrelated_distance_median_iqr_min: float = 0.50


@dataclass(frozen=True)
class ControlledDirectionResult:
    name: str
    source_level: str
    target_level: str
    target_axes: tuple[str, ...]
    oriented_median_effect_iqr: dict[str, float]
    correct_direction_fraction: dict[str, float]
    responsive_axes: tuple[str, ...]
    scale_axis_max_delta: float
    passed: bool


@dataclass(frozen=True)
class CVRPQoIAxiomReport:
    schema_version: str
    kind: str
    conclusion: str
    qoi_spec_name: str
    qoi_spec_version: str
    qoi_spec_fingerprint: str
    panel_fingerprint: str
    preregistration: dict[str, Any]
    axioms: dict[str, dict[str, Any]]
    axis_diagnostics: dict[str, dict[str, Any]]
    controlled_directions: tuple[ControlledDirectionResult, ...]
    solver_runs_used: int
    solver_runs_created: int
    production_geometry_changed: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Pair:
    kind: Literal["equivalence", "unit", "scale", "unrelated"]
    pair_id: str
    left: InstanceQoIObservation
    right: InstanceQoIObservation
    requested_source_customers: int | None = None
    requested_target_customers: int | None = None


@dataclass(frozen=True)
class _DirectionSpec:
    name: str
    source_level: str
    target_level: str
    target_axes: tuple[str, ...]
    signs: tuple[int, ...]


def _version_freezing_evidence(
    spec: InstanceQoISpec,
    pinned_fingerprints: Mapping[str, str],
    observed_fingerprints: set[str],
) -> dict[str, Any]:
    """Compare a spec and its observations with an independent version pin."""

    current_fingerprint = spec.fingerprint()
    pinned_fingerprint = pinned_fingerprints.get(spec.version)
    version_has_pin = pinned_fingerprint is not None
    spec_matches_pin = version_has_pin and current_fingerprint == pinned_fingerprint
    observations_match_pin = version_has_pin and observed_fingerprints == {
        pinned_fingerprint
    }
    return {
        "pinned_spec_fingerprint": pinned_fingerprint,
        "current_spec_fingerprint": current_fingerprint,
        "version_has_pinned_fingerprint": version_has_pin,
        "spec_matches_pinned_fingerprint": spec_matches_pin,
        "observations_match_pinned_fingerprint": observations_match_pin,
        "qoi_spec_fingerprint_frozen": spec_matches_pin and observations_match_pin,
    }


_DIRECTIONS = (
    _DirectionSpec(
        name="capacity_pressure",
        source_level="capacity_ratio=0.30",
        target_level="capacity_ratio=0.08",
        target_axes=("vehicle_lower_bound", "demand_mean_fraction"),
        signs=(1, 1),
    ),
    _DirectionSpec(
        name="cluster_spread",
        source_level="cluster_spread=3",
        target_level="cluster_spread=24",
        target_axes=(
            "nearest_distance_mean_normalized",
            "mst_edge_mean_normalized",
        ),
        signs=(1, 1),
    ),
    _DirectionSpec(
        name="demand_dispersion",
        source_level="uniform_integer",
        target_level="bimodal",
        target_axes=("demand_cv",),
        signs=(1,),
    ),
    _DirectionSpec(
        name="demand_location_coupling",
        source_level="depot_anticorrelated",
        target_level="depot_correlated",
        target_axes=(
            "demand_depot_correlation",
            "demand_weighted_depot_ratio",
        ),
        signs=(1, 1),
    ),
    _DirectionSpec(
        name="depot_position",
        source_level="center",
        target_level="corner",
        target_axes=("depot_distance_mean_normalized",),
        signs=(1,),
    ),
)


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _seed(namespace: str, dimension: str, index: int) -> int:
    digest = hashlib.sha256(f"{namespace}:{dimension}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot take median of empty values")
    return float(np.median(np.asarray(values, dtype=float)))


def _ranks(values: Sequence[float]) -> np.ndarray:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = np.zeros(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2
        start = end
    return result


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman correlation requires paired observations")
    left_ranks, right_ranks = _ranks(left), _ranks(right)
    if np.std(left_ranks) == 0 or np.std(right_ranks) == 0:
        return 0.0
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def _observe(instance: Mapping[str, Any], instance_id: str) -> InstanceQoIObservation:
    return extract_cvrp_instance_qoi(instance, instance_id=instance_id)


def _instance(
    *,
    seed: int,
    name: str,
    customers: int = 100,
    capacity_ratio: float = 0.15,
    coordinate_distribution: str = "uniform",
    cluster_spread: float = 9.0,
    depot_mode: str = "center",
    demand_distribution: str = "uniform_integer",
) -> dict[str, Any]:
    return make_euclidean_cvrp_instance(
        name=name,
        customers=customers,
        coordinate_seed=seed,
        demand_seed=seed + 10_000,
        capacity_ratio=capacity_ratio,
        coordinate_distribution=coordinate_distribution,
        cluster_count=4,
        cluster_spread=cluster_spread,
        depot_mode=depot_mode,
        demand_distribution=demand_distribution,
    )


def _unit_scaled(instance: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(instance)
    result["coordinates"] = [
        [float(x) * 10.0, float(y) * 10.0]
        for x, y in instance["coordinates"]
    ]
    result["demands"] = [float(value) * 10.0 for value in instance["demands"]]
    result["capacity"] = float(instance["capacity"]) * 10.0
    return result


def _robust_scales(
    *, calibration_size: int, calibration_namespace: str
) -> tuple[dict[str, float], str]:
    customer_counts = (50, 100, 200)
    capacity_ratios = (0.08, 0.15, 0.30)
    coordinate_distributions = ("uniform", "clustered")
    cluster_spreads = (3.0, 9.0, 24.0)
    depot_modes = ("center", "corner", "random")
    demand_distributions = (
        "uniform_integer",
        "bimodal",
        "depot_correlated",
        "depot_anticorrelated",
    )
    design = []
    observations = []
    for index in range(calibration_size):
        parameters = {
            "name": f"{calibration_namespace}-{index:04d}",
            "customers": customer_counts[index % len(customer_counts)],
            "coordinate_seed": _seed(
                calibration_namespace, "coordinates", index
            ),
            "demand_seed": _seed(calibration_namespace, "demands", index),
            "capacity_ratio": capacity_ratios[
                (index // len(customer_counts)) % len(capacity_ratios)
            ],
            "coordinate_distribution": coordinate_distributions[
                index % len(coordinate_distributions)
            ],
            "cluster_count": 4,
            "cluster_spread": cluster_spreads[
                (index * 2) % len(cluster_spreads)
            ],
            "depot_mode": depot_modes[(index * 2) % len(depot_modes)],
            "demand_distribution": demand_distributions[
                (index * 3) % len(demand_distributions)
            ],
        }
        design.append(parameters)
        observations.append(
            _observe(
                make_euclidean_cvrp_instance(**parameters),
                str(parameters["name"]),
            )
        )
    scales: dict[str, float] = {}
    for axis in (item.name for item in CVRP_INSTANCE_QOI.axes):
        values = np.asarray([item.values[axis] for item in observations], dtype=float)
        iqr = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
        scales[axis] = iqr if iqr > np.finfo(float).eps else 1.0
    fingerprint = _canonical_hash(
        {
            "namespace": calibration_namespace,
            "design": design,
            "qoi": [item.model_dump(mode="json") for item in observations],
            "scales": scales,
        }
    )
    return scales, fingerprint


def _build_pairs(
    seeds: Sequence[int], sizes: Sequence[int]
) -> tuple[
    list[_Pair],
    dict[str, list[tuple[InstanceQoIObservation, InstanceQoIObservation]]],
]:
    pairs: list[_Pair] = []
    directions: dict[
        str, list[tuple[InstanceQoIObservation, InstanceQoIObservation]]
    ] = {item.name: [] for item in _DIRECTIONS}
    for seed in seeds:
        base_instance = _instance(seed=seed, name=f"base-{seed}")
        base = _observe(base_instance, f"base-{seed}")
        transformed = {
            "translate": translate_cvrp(base_instance, dx=37.0, dy=-19.0),
            "rotate": rotate_cvrp(base_instance, radians=0.731),
            "reflect": reflect_cvrp(base_instance),
            "relabel": relabel_cvrp_customers(base_instance, seed=seed + 99),
        }
        for transform_name, instance in transformed.items():
            pairs.append(
                _Pair(
                    kind="equivalence",
                    pair_id=f"equivalence-{seed}-{transform_name}",
                    left=base,
                    right=_observe(instance, f"equivalence-{seed}-{transform_name}"),
                )
            )
        pairs.append(
            _Pair(
                kind="unit",
                pair_id=f"unit-{seed}",
                left=base,
                right=_observe(_unit_scaled(base_instance), f"unit-{seed}"),
            )
        )
        scale_observations = {
            size: _observe(
                _instance(seed=seed, name=f"scale-{seed}-{size}", customers=size),
                f"scale-{seed}-{size}",
            )
            for size in sizes
        }
        for low, high in combinations(sizes, 2):
            pairs.append(
                _Pair(
                    kind="scale",
                    pair_id=f"scale-{seed}-{low}-{high}",
                    left=scale_observations[low],
                    right=scale_observations[high],
                    requested_source_customers=low,
                    requested_target_customers=high,
                )
            )
        unrelated = _observe(
            _instance(
                seed=seed + 50_000,
                name=f"unrelated-{seed}",
                capacity_ratio=0.08,
                coordinate_distribution="clustered",
                cluster_spread=3.0,
                depot_mode="corner",
                demand_distribution="bimodal",
            ),
            f"unrelated-{seed}",
        )
        pairs.append(
            _Pair(
                kind="unrelated",
                pair_id=f"unrelated-{seed}",
                left=base,
                right=unrelated,
            )
        )

        direction_instances = {
            "capacity_pressure": (
                _instance(
                    seed=seed,
                    name=f"capacity-loose-{seed}",
                    capacity_ratio=0.30,
                ),
                _instance(
                    seed=seed,
                    name=f"capacity-tight-{seed}",
                    capacity_ratio=0.08,
                ),
            ),
            "cluster_spread": (
                _instance(
                    seed=seed,
                    name=f"cluster-compact-{seed}",
                    coordinate_distribution="clustered",
                    cluster_spread=3.0,
                ),
                _instance(
                    seed=seed,
                    name=f"cluster-diffuse-{seed}",
                    coordinate_distribution="clustered",
                    cluster_spread=24.0,
                ),
            ),
            "demand_dispersion": (
                _instance(seed=seed, name=f"demand-uniform-{seed}"),
                _instance(
                    seed=seed,
                    name=f"demand-bimodal-{seed}",
                    demand_distribution="bimodal",
                ),
            ),
            "demand_location_coupling": (
                _instance(
                    seed=seed,
                    name=f"demand-anticorrelated-{seed}",
                    demand_distribution="depot_anticorrelated",
                ),
                _instance(
                    seed=seed,
                    name=f"demand-correlated-{seed}",
                    demand_distribution="depot_correlated",
                ),
            ),
            "depot_position": (
                _instance(seed=seed, name=f"depot-center-{seed}"),
                _instance(
                    seed=seed,
                    name=f"depot-corner-{seed}",
                    depot_mode="corner",
                ),
            ),
        }
        for name, (source, target) in direction_instances.items():
            directions[name].append(
                (
                    _observe(source, f"{name}-source-{seed}"),
                    _observe(target, f"{name}-target-{seed}"),
                )
            )
    return pairs, directions


def _standardized_distance(
    left: InstanceQoIObservation,
    right: InstanceQoIObservation,
    *,
    axes: Sequence[str],
    scales: Mapping[str, float],
) -> float:
    return float(
        np.mean(
            [
                abs(right.values[axis] - left.values[axis]) / scales[axis]
                for axis in axes
            ]
        )
    )


def _controlled_results(
    directions: Mapping[
        str, Sequence[tuple[InstanceQoIObservation, InstanceQoIObservation]]
    ],
    *,
    scales: Mapping[str, float],
    thresholds: QoIAxiomThresholds,
) -> tuple[ControlledDirectionResult, ...]:
    results = []
    for spec in _DIRECTIONS:
        pairs = directions[spec.name]
        effects: dict[str, float] = {}
        fractions: dict[str, float] = {}
        responsive = []
        for axis, sign in zip(spec.target_axes, spec.signs, strict=True):
            oriented = [
                sign * (right.values[axis] - left.values[axis]) / scales[axis]
                for left, right in pairs
            ]
            effects[axis] = _median(oriented)
            fractions[axis] = sum(value > 0 for value in oriented) / len(oriented)
            if (
                effects[axis] >= thresholds.controlled_effect_median_iqr_min
                and fractions[axis] >= thresholds.controlled_direction_fraction_min
            ):
                responsive.append(axis)
        scale_axis_max_delta = max(
            abs(right.values["customer_count"] - left.values["customer_count"])
            for left, right in pairs
        )
        results.append(
            ControlledDirectionResult(
                name=spec.name,
                source_level=spec.source_level,
                target_level=spec.target_level,
                target_axes=spec.target_axes,
                oriented_median_effect_iqr=effects,
                correct_direction_fraction=fractions,
                responsive_axes=tuple(responsive),
                scale_axis_max_delta=scale_axis_max_delta,
                passed=bool(responsive) and scale_axis_max_delta == 0.0,
            )
        )
    return tuple(results)


def run_cvrp_qoi_axiom_validation(
    *,
    seeds: Sequence[int] = (
        101,
        211,
        307,
        401,
        503,
        601,
        701,
        809,
        907,
        1009,
        1103,
        1201,
        1301,
        1409,
        1511,
        1601,
    ),
    sizes: Sequence[int] = (50, 100, 200, 500),
    calibration_size: int = 128,
    calibration_namespace: str = "cvrp-qoi-axiom-calibration-v1",
    thresholds: QoIAxiomThresholds = QoIAxiomThresholds(),
) -> CVRPQoIAxiomReport:
    """Validate methodology axioms 1--8 against the frozen CVRP QoI spec."""
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("at least two unique panel seeds are required")
    if len(sizes) < 3 or tuple(sorted(set(sizes))) != tuple(sizes):
        raise ValueError("sizes must contain at least three increasing unique values")
    if calibration_size < 8:
        raise ValueError("calibration_size must be at least eight")

    scales, calibration_fingerprint = _robust_scales(
        calibration_size=calibration_size,
        calibration_namespace=calibration_namespace,
    )
    pairs, directions = _build_pairs(seeds, sizes)
    by_kind = {
        kind: [pair for pair in pairs if pair.kind == kind]
        for kind in ("equivalence", "unit", "scale", "unrelated")
    }

    axis_diagnostics: dict[str, dict[str, Any]] = {}
    for axis_spec in CVRP_INSTANCE_QOI.axes:
        axis = axis_spec.name
        equivalence_errors = [
            abs(pair.right.values[axis] - pair.left.values[axis]) / scales[axis]
            for pair in by_kind["equivalence"]
        ]
        unit_errors = [
            abs(pair.right.values[axis] - pair.left.values[axis]) / scales[axis]
            for pair in by_kind["unit"]
        ]
        scale_changes = [
            abs(pair.right.values[axis] - pair.left.values[axis]) / scales[axis]
            for pair in by_kind["scale"]
        ]
        unrelated_changes = [
            abs(pair.right.values[axis] - pair.left.values[axis]) / scales[axis]
            for pair in by_kind["unrelated"]
        ]
        axis_diagnostics[axis] = {
            "unit": axis_spec.unit,
            "solver_independent_declared": axis_spec.solver_independent,
            "role": (
                "raw_unit"
                if axis in RAW_UNIT_AXES
                else "scale"
                if axis in SCALE_AXES
                else "structure"
            ),
            "equivalence_max_error_iqr": max(equivalence_errors),
            "unit_max_error_iqr": max(unit_errors),
            "scale_leakage_median_iqr": _median(scale_changes),
            "unrelated_change_median_iqr": _median(unrelated_changes),
            "equivalence_invariant": max(equivalence_errors)
            <= thresholds.numerical_standardized_tolerance,
            "unit_robust": (
                None
                if axis in RAW_UNIT_AXES
                else max(unit_errors) <= thresholds.numerical_standardized_tolerance
            ),
            "scale_stable": (
                None
                if axis not in STRUCTURAL_AXES
                else _median(scale_changes)
                <= thresholds.scale_leakage_median_iqr_max
            ),
        }

    equivalence_distances = [
        _standardized_distance(
            pair.left, pair.right, axes=STRUCTURAL_AXES, scales=scales
        )
        for pair in by_kind["equivalence"]
    ]
    unrelated_distances = [
        _standardized_distance(
            pair.left, pair.right, axes=STRUCTURAL_AXES, scales=scales
        )
        for pair in by_kind["unrelated"]
    ]
    equivalence_unrelated_ratio = _median(equivalence_distances) / max(
        _median(unrelated_distances), np.finfo(float).eps
    )

    annotated = _instance(seed=seeds[0], name="solver-annotation-control")
    annotated_with_outcomes = {
        **annotated,
        "gap": 0.75,
        "wall_time": 999.0,
        "iterations": 123_456,
        "solver_status": "synthetic-annotation",
    }
    annotation_left = _observe(annotated, "annotation-left")
    annotation_right = _observe(annotated_with_outcomes, "annotation-right")
    annotation_max_delta = max(
        abs(annotation_left.values[axis] - annotation_right.values[axis])
        for axis in annotation_left.values
    )

    scale_magnitudes = [
        abs(
            math.log(
                pair.right.values["customer_count"]
                / pair.left.values["customer_count"]
            )
        )
        for pair in by_kind["scale"]
    ]
    requested_magnitudes = [
        abs(
            math.log(
                float(pair.requested_target_customers)
                / float(pair.requested_source_customers)
            )
        )
        for pair in by_kind["scale"]
    ]
    scale_spearman = _spearman(requested_magnitudes, scale_magnitudes)
    count_exact = all(
        pair.left.values["customer_count"] == pair.requested_source_customers
        and pair.right.values["customer_count"] == pair.requested_target_customers
        for pair in by_kind["scale"]
    )

    stable_structural_axes = tuple(
        axis
        for axis in STRUCTURAL_AXES
        if axis_diagnostics[axis]["scale_stable"] is True
    )
    scale_stable_fraction = len(stable_structural_axes) / len(STRUCTURAL_AXES)
    controlled = _controlled_results(
        directions, scales=scales, thresholds=thresholds
    )
    controlled_pass_fraction = sum(item.passed for item in controlled) / len(controlled)
    structure_scale_max_delta = max(item.scale_axis_max_delta for item in controlled)

    unit_max = max(
        float(axis_diagnostics[axis]["unit_max_error_iqr"])
        for axis in UNIT_ROBUST_AXES
    )
    all_solver_independent = all(
        axis.solver_independent for axis in CVRP_INSTANCE_QOI.axes
    )
    repeated = _observe(annotated, "annotation-left")
    deterministic_observation = repeated == annotation_left
    all_fingerprints = {
        observation.spec_fingerprint
        for pair in pairs
        for observation in (pair.left, pair.right)
    }
    version_evidence = _version_freezing_evidence(
        CVRP_INSTANCE_QOI,
        CVRP_INSTANCE_QOI_PINNED_FINGERPRINTS,
        all_fingerprints,
    )

    axiom_pass = {
        "1_solver_independence": all_solver_independent and annotation_max_delta == 0.0,
        "2_semantic_invariance": (
            max(
                float(axis_diagnostics[axis]["equivalence_max_error_iqr"])
                for axis in axis_diagnostics
            )
            <= thresholds.numerical_standardized_tolerance
            and equivalence_unrelated_ratio
            <= thresholds.equivalence_unrelated_ratio_max
        ),
        "3_scale_sensitivity": count_exact
        and scale_spearman >= thresholds.scale_monotonicity_spearman_min,
        "4_structure_scale_separability": (
            scale_stable_fraction >= thresholds.scale_stable_axis_fraction_min
            and controlled_pass_fraction
            >= thresholds.controlled_direction_pass_fraction_min
            and structure_scale_max_delta == 0.0
        ),
        "5_unit_representation_robustness": unit_max
        <= thresholds.numerical_standardized_tolerance,
        "6_no_circular_learned_semantics": all_solver_independent
        and annotation_max_delta == 0.0,
        "7_version_freezing": (
            version_evidence["qoi_spec_fingerprint_frozen"]
            and deterministic_observation
        ),
        "8_falsifiability": (
            _median(unrelated_distances)
            >= thresholds.unrelated_distance_median_iqr_min
            and equivalence_unrelated_ratio
            <= thresholds.equivalence_unrelated_ratio_max
            and len(controlled) == len(_DIRECTIONS)
        ),
    }

    axioms = {
        "1_solver_independence": {
            "passed": axiom_pass["1_solver_independence"],
            "declared_solver_independent_axis_fraction": sum(
                axis.solver_independent for axis in CVRP_INSTANCE_QOI.axes
            )
            / len(CVRP_INSTANCE_QOI.axes),
            "solver_annotation_max_delta": annotation_max_delta,
        },
        "2_semantic_invariance": {
            "passed": axiom_pass["2_semantic_invariance"],
            "transformations": ["translation", "rotation", "reflection", "relabeling"],
            "equivalence_distance_median_iqr": _median(equivalence_distances),
            "unrelated_distance_median_iqr": _median(unrelated_distances),
            "equivalence_unrelated_ratio": equivalence_unrelated_ratio,
        },
        "3_scale_sensitivity": {
            "passed": axiom_pass["3_scale_sensitivity"],
            "scale_descriptor": "customer_count",
            "sizes": list(sizes),
            "log_ratio_spearman": scale_spearman,
            "customer_count_exact": count_exact,
        },
        "4_structure_scale_separability": {
            "passed": axiom_pass["4_structure_scale_separability"],
            "stable_structural_axes": list(stable_structural_axes),
            "leaking_structural_axes": sorted(
                set(STRUCTURAL_AXES) - set(stable_structural_axes)
            ),
            "scale_stable_axis_fraction": scale_stable_fraction,
            "controlled_direction_pass_fraction": controlled_pass_fraction,
            "structure_shift_customer_count_max_delta": structure_scale_max_delta,
        },
        "5_unit_representation_robustness": {
            "passed": axiom_pass["5_unit_representation_robustness"],
            "unit_robust_axes": list(UNIT_ROBUST_AXES),
            "excluded_raw_unit_axes": list(RAW_UNIT_AXES),
            "unit_robust_axis_max_error_iqr": unit_max,
        },
        "6_no_circular_learned_semantics": {
            "passed": axiom_pass["6_no_circular_learned_semantics"],
            "learned_from_target_solver": False,
            "solver_runs_used": 0,
            "solver_annotation_max_delta": annotation_max_delta,
        },
        "7_version_freezing": {
            "passed": axiom_pass["7_version_freezing"],
            **version_evidence,
            "deterministic_repeat_identical": deterministic_observation,
            "calibration_fingerprint": calibration_fingerprint,
        },
        "8_falsifiability": {
            "passed": axiom_pass["8_falsifiability"],
            "negative_controls": ["equivalence", "unit", "unrelated"],
            "controlled_direction_count": len(controlled),
            "thresholds_frozen_in_artifact": True,
            "cost_weight_sensitivity": "not_applicable_no_combined_weighted_cost",
        },
    }

    core = (
        "2_semantic_invariance",
        "3_scale_sensitivity",
        "4_structure_scale_separability",
        "5_unit_representation_robustness",
        "8_falsifiability",
    )
    conclusion = "Validated" if all(axiom_pass.values()) else "Falsified"
    if all(axiom_pass[name] for name in core) and not all(axiom_pass.values()):
        conclusion = "Promising"

    panel_payload = {
        "seeds": list(seeds),
        "sizes": list(sizes),
        "calibration_namespace": calibration_namespace,
        "calibration_size": calibration_size,
        "calibration_fingerprint": calibration_fingerprint,
        "pairs": [
            {
                "kind": pair.kind,
                "pair_id": pair.pair_id,
                "left": pair.left.model_dump(mode="json"),
                "right": pair.right.model_dump(mode="json"),
            }
            for pair in pairs
        ],
    }
    return CVRPQoIAxiomReport(
        schema_version="1.0",
        kind="cvrp_instance_qoi_axiom_validation",
        conclusion=conclusion,
        qoi_spec_name=CVRP_INSTANCE_QOI.name,
        qoi_spec_version=CVRP_INSTANCE_QOI.version,
        qoi_spec_fingerprint=CVRP_INSTANCE_QOI.fingerprint(),
        panel_fingerprint=_canonical_hash(panel_payload),
        preregistration={
            "methodology_axioms": list(axioms),
            "thresholds": asdict(thresholds),
            "panel": {
                "seeds": list(seeds),
                "sizes": list(sizes),
                "calibration_namespace": calibration_namespace,
                "calibration_size": calibration_size,
                "calibration_fingerprint": calibration_fingerprint,
            },
            "unit_policy": {
                "raw_unit_axes_reported_but_excluded_from_ground_geometry": list(
                    RAW_UNIT_AXES
                ),
                "unit_robust_axes": list(UNIT_ROBUST_AXES),
            },
        },
        axioms=axioms,
        axis_diagnostics=axis_diagnostics,
        controlled_directions=controlled,
        solver_runs_used=0,
        solver_runs_created=0,
        production_geometry_changed=False,
    )


__all__ = [
    "CVRPQoIAxiomReport",
    "QoIAxiomThresholds",
    "RAW_UNIT_AXES",
    "SCALE_AXES",
    "STRUCTURAL_AXES",
    "UNIT_ROBUST_AXES",
    "run_cvrp_qoi_axiom_validation",
]
