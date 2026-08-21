from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from pitbench.qoi.cvrp import CVRP_INSTANCE_QOI_V1_1, extract_cvrp_instance_qoi


@dataclass(frozen=True)
class UchoaConstructCase:
    instance_id: str
    instance: Mapping[str, Any]
    depot_positioning: str
    customer_positioning: str
    demand_family: str
    route_size: float


@dataclass(frozen=True)
class UchoaConstructReport:
    schema_version: str
    kind: str
    conclusion: str
    qoi_spec_fingerprint: str
    panel_fingerprint: str
    evidence: dict[str, dict[str, Any]]
    solver_runs_used: int
    solver_runs_created: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _iqr(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    result = float(np.quantile(array, 0.75) - np.quantile(array, 0.25))
    return result if result > np.finfo(float).eps else 1.0


def _contrast(
    values: Sequence[float],
    labels: Sequence[bool],
    *,
    nuisance: np.ndarray,
    sign: int,
    permutations: int,
    seed: int,
) -> tuple[float, float]:
    if not any(labels) or all(labels):
        return 0.0, 1.0
    response = np.asarray(values, dtype=float)
    treatment = np.asarray(labels, dtype=float)
    if nuisance.shape[0] != len(response):
        raise ValueError("nuisance design and construct outcomes disagree")
    scale = _iqr(values)
    full = np.column_stack((nuisance, treatment))
    observed = sign * float(np.linalg.lstsq(full, response, rcond=None)[0][-1]) / scale
    reduced_coefficients = np.linalg.lstsq(nuisance, response, rcond=None)[0]
    fitted = nuisance @ reduced_coefficients
    residuals = response - fitted
    rng = random.Random(seed)
    exceedances = 0
    for _ in range(permutations):
        shuffled = residuals.tolist()
        rng.shuffle(shuffled)
        permuted = fitted + np.asarray(shuffled)
        effect = (
            sign * float(np.linalg.lstsq(full, permuted, rcond=None)[0][-1]) / scale
        )
        exceedances += effect >= observed
    return observed, (exceedances + 1) / (permutations + 1)


def _nuisance_design(
    cases: Sequence[UchoaConstructCase],
    observations: Sequence[Any],
    *,
    excluded_attribute: str,
) -> np.ndarray:
    columns: list[list[float]] = [
        [1.0] * len(cases),
        [
            math.log(observation.values["customer_count"])
            for observation in observations
        ],
        [case.route_size for case in cases],
    ]
    attributes = {
        "depot_positioning": [case.depot_positioning for case in cases],
        "customer_positioning": [case.customer_positioning for case in cases],
        "demand_family": [case.demand_family for case in cases],
    }
    for attribute, values in attributes.items():
        if attribute == excluded_attribute:
            continue
        levels = sorted(set(values))
        for level in levels[1:]:
            columns.append([float(value == level) for value in values])
    return np.asarray(columns, dtype=float).T


def _bh_adjust(pvalues: Sequence[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted = [1.0] * len(pvalues)
    running = 1.0
    for rank_index in range(len(order) - 1, -1, -1):
        index = order[rank_index]
        rank = rank_index + 1
        running = min(running, pvalues[index] * len(pvalues) / rank)
        adjusted[index] = running
    return adjusted


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 2 or len(left) != len(right):
        return 0.0

    def ranks(values: Sequence[float]) -> np.ndarray:
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

    left_ranks = ranks(left)
    right_ranks = ranks(right)
    if np.std(left_ranks) == 0 or np.std(right_ranks) == 0:
        return 0.0
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def run_uchoa_construct_validation(
    cases: Sequence[UchoaConstructCase],
    *,
    permutations: int = 4_999,
) -> UchoaConstructReport:
    if not cases or permutations <= 0:
        raise ValueError("construct validation requires cases and permutations")
    observations = [
        extract_cvrp_instance_qoi(
            case.instance, instance_id=case.instance_id, spec_version="1.1"
        )
        for case in cases
    ]
    definitions = (
        (
            "depot_c_to_e",
            "depot_distance_mean_normalized",
            [case.depot_positioning == "E" for case in cases],
            [case.depot_positioning in {"C", "E"} for case in cases],
            1,
            "depot_positioning",
        ),
        (
            "customer_r_to_c",
            "nearest_neighbor_clark_evans_ratio",
            [case.customer_positioning == "C" for case in cases],
            [case.customer_positioning in {"R", "C"} for case in cases],
            -1,
            "customer_positioning",
        ),
        (
            "unitary_to_variable_demand",
            "demand_cv",
            [case.demand_family != "unitary" for case in cases],
            [True for _ in cases],
            1,
            "demand_family",
        ),
        (
            "nonquadrant_to_quadrant",
            "demand_spatial_quadrupole_coupling",
            [case.demand_family == "quadrant" for case in cases],
            [True for _ in cases],
            1,
            "demand_family",
        ),
    )
    evidence: dict[str, dict[str, Any]] = {}
    pvalues = []
    names = []
    for index, (
        name,
        axis,
        labels,
        include,
        sign,
        excluded_attribute,
    ) in enumerate(definitions):
        selection = [
            keep and observation.axis_defined[axis]
            for observation, keep in zip(observations, include, strict=True)
        ]
        selected_cases = [
            case for case, keep in zip(cases, selection, strict=True) if keep
        ]
        selected_observations = [
            observation
            for observation, keep in zip(observations, selection, strict=True)
            if keep
        ]
        selected_values = [
            observation.values[axis] for observation in selected_observations
        ]
        selected_labels = [
            label for label, keep in zip(labels, selection, strict=True) if keep
        ]
        effect, pvalue = _contrast(
            selected_values,
            selected_labels,
            nuisance=_nuisance_design(
                selected_cases,
                selected_observations,
                excluded_attribute=excluded_attribute,
            ),
            sign=sign,
            permutations=permutations,
            seed=10_000 + index,
        )
        evidence[name] = {
            "axis": axis,
            "oriented_effect_iqr": effect,
            "permutation_pvalue": pvalue,
            "sample_size": len(selected_values),
        }
        names.append(name)
        pvalues.append(pvalue)
    for name, qvalue in zip(names, _bh_adjust(pvalues), strict=True):
        evidence[name]["bh_qvalue"] = qvalue
        evidence[name]["passed"] = (
            evidence[name]["oriented_effect_iqr"] >= 0.5 and qvalue <= 0.05
        )

    requested = [case.route_size for case in cases]
    observed_route_size = [
        observation.values["volume_lb_customers_per_route"]
        for observation in observations
    ]
    route_spearman = _spearman(requested, observed_route_size)
    evidence["route_size"] = {
        "axis": "volume_lb_customers_per_route",
        "spearman": route_spearman,
        "passed": route_spearman >= 0.90,
        "sample_size": len(cases),
    }
    passed = [item["passed"] for item in evidence.values()]
    conclusion = (
        "Externally supported"
        if all(passed)
        else "Partially supported"
        if any(passed)
        else "Falsified"
    )
    panel = [
        {
            "instance_id": case.instance_id,
            "depot_positioning": case.depot_positioning,
            "customer_positioning": case.customer_positioning,
            "demand_family": case.demand_family,
            "route_size": case.route_size,
            "observation": observation.model_dump(mode="json"),
        }
        for case, observation in zip(cases, observations, strict=True)
    ]
    return UchoaConstructReport(
        schema_version="1.1",
        kind="cvrp_uchoa_external_construct_validation",
        conclusion=conclusion,
        qoi_spec_fingerprint=CVRP_INSTANCE_QOI_V1_1.fingerprint(),
        panel_fingerprint=_hash(panel),
        evidence=evidence,
        solver_runs_used=0,
        solver_runs_created=0,
    )
