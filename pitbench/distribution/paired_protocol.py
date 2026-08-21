from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pitbench.instances import make_uchoa_cvrp_instance


@dataclass(frozen=True)
class PairedTreatmentCase:
    treatment: str
    pair_id: str
    source_level: str
    target_level: str
    source: Mapping[str, Any]
    target: Mapping[str, Any]


PRIMARY_TREATMENTS = (
    "scale",
    "depot",
    "customer_structure",
    "demand_dispersion",
    "non_radial_coupling",
    "route_size",
)


def _instance(seed: int, name: str, **updates: Any) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "name": name,
        "customers": 200,
        "coordinate_seed": seed,
        "demand_seed": seed + 100_000,
        "depot_positioning": "C",
        "customer_positioning": "R",
        "demand_family": "small_large_variance",
        "route_size": 10,
    }
    parameters.update(updates)
    return make_uchoa_cvrp_instance(**parameters)


def build_paired_cvrp_panel(seeds: Sequence[int]) -> tuple[PairedTreatmentCase, ...]:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("paired panel seeds must be non-empty and unique")
    pairs = []
    for seed in seeds:
        definitions = (
            (
                "scale",
                "n=100",
                "n=500",
                _instance(seed, f"scale-source-{seed}", customers=100),
                _instance(seed, f"scale-target-{seed}", customers=500),
            ),
            (
                "depot",
                "C",
                "E",
                _instance(seed, f"depot-source-{seed}"),
                _instance(seed, f"depot-target-{seed}", depot_positioning="E"),
            ),
            (
                "customer_structure",
                "R",
                "C",
                _instance(seed, f"customer-source-{seed}"),
                _instance(seed, f"customer-target-{seed}", customer_positioning="C"),
            ),
            (
                "demand_dispersion",
                "UD[5,10]",
                "UD[1,100]",
                _instance(
                    seed,
                    f"demand-source-{seed}",
                    demand_family="small_small_variance",
                ),
                _instance(
                    seed,
                    f"demand-target-{seed}",
                    demand_family="large_large_variance",
                ),
            ),
            (
                "non_radial_coupling",
                "location_permuted",
                "quadrant",
                _instance(
                    seed,
                    f"coupling-source-{seed}",
                    demand_family="quadrant_permuted",
                ),
                _instance(
                    seed,
                    f"coupling-target-{seed}",
                    demand_family="quadrant",
                ),
            ),
            (
                "route_size",
                "r=5",
                "r=20",
                _instance(seed, f"route-source-{seed}", route_size=5),
                _instance(seed, f"route-target-{seed}", route_size=20),
            ),
        )
        pairs.extend(
            PairedTreatmentCase(
                treatment=treatment,
                pair_id=f"{treatment}-{seed}",
                source_level=source_level,
                target_level=target_level,
                source=source,
                target=target,
            )
            for treatment, source_level, target_level, source, target in definitions
        )
    return tuple(pairs)


_TREATMENT_EVIDENCE = {
    "scale": ("axiom", "3_scale_sensitivity"),
    "depot": ("direction", "depot_position"),
    "customer_structure": ("direction", "cluster_spread"),
    "demand_dispersion": ("direction", "demand_dispersion"),
    "non_radial_coupling": ("direction", "non_radial_coupling"),
    "route_size": ("direction", "route_size"),
}


def solver_experiment_gate(
    axiom_artifact: Mapping[str, Any], treatment: str | None = None
) -> tuple[bool, str]:
    """Check schema integrity plus evidence for the requested treatment axis."""

    if axiom_artifact.get("qoi_spec_version") != "1.1":
        return False, "paired experiment requires the frozen QoI schema v1.1"
    axioms = axiom_artifact.get("axioms", {})
    schema_axioms = (
        "1_solver_independence",
        "2_semantic_invariance",
        "5_unit_representation_robustness",
        "7_version_freezing",
        "8_falsifiability",
    )
    failed_schema = [
        name for name in schema_axioms if not axioms.get(name, {}).get("passed", False)
    ]
    if failed_schema:
        return False, f"v1.1 schema checks did not pass: {', '.join(failed_schema)}"

    treatments = PRIMARY_TREATMENTS if treatment is None else (treatment,)
    unknown = [name for name in treatments if name not in _TREATMENT_EVIDENCE]
    if unknown:
        return False, f"unknown paired treatment: {', '.join(unknown)}"
    directions = {
        item.get("name"): item
        for item in axiom_artifact.get("controlled_directions", ())
        if isinstance(item, Mapping)
    }
    missing = []
    for name in treatments:
        evidence_kind, evidence_name = _TREATMENT_EVIDENCE[name]
        if evidence_kind == "axiom":
            passed = axioms.get(evidence_name, {}).get("passed", False)
        else:
            passed = directions.get(evidence_name, {}).get("passed", False)
        if not passed:
            missing.append(name)
    if missing:
        return False, f"treatment-specific evidence did not pass: {', '.join(missing)}"
    return True, "v1.1 schema and treatment-specific evidence passed"
