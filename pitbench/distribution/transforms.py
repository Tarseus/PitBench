"""Certified semantic-preserving transforms for normalized CVRP instances."""

from __future__ import annotations

import copy
import math
import random
from typing import Any, Mapping


def _copy(instance: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(instance))


def translate_cvrp(
    instance: Mapping[str, Any], *, dx: float, dy: float
) -> dict[str, Any]:
    result = _copy(instance)
    result["coordinates"] = [
        [float(x) + dx, float(y) + dy] for x, y in result["coordinates"]
    ]
    return result


def rotate_cvrp(instance: Mapping[str, Any], *, radians: float) -> dict[str, Any]:
    result = _copy(instance)
    cosine, sine = math.cos(radians), math.sin(radians)
    result["coordinates"] = [
        [cosine * float(x) - sine * float(y), sine * float(x) + cosine * float(y)]
        for x, y in result["coordinates"]
    ]
    return result


def reflect_cvrp(instance: Mapping[str, Any]) -> dict[str, Any]:
    result = _copy(instance)
    result["coordinates"] = [[-float(x), float(y)] for x, y in result["coordinates"]]
    return result


def relabel_cvrp_customers(instance: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
    """Permute customer rows while preserving depot identity and all semantics."""
    result = _copy(instance)
    depot = int(result.get("depot", 0))
    customers = [index for index in range(len(result["coordinates"])) if index != depot]
    random.Random(seed).shuffle(customers)
    order = [depot, *customers]
    result["coordinates"] = [result["coordinates"][index] for index in order]
    result["demands"] = [result["demands"][index] for index in order]
    if "node_ids" in result:
        result["node_ids"] = [result["node_ids"][index] for index in order]
    result["depot"] = 0
    return result
