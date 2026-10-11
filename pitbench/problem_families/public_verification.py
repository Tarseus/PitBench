"""Dependency-free checks shared by the judge and visible development bench."""

from __future__ import annotations

import math


def edge_cost(
    coordinates: list[list[float]],
    first: int,
    second: int,
    distance_metric: str | None,
) -> float:
    x1, y1 = coordinates[first]
    x2, y2 = coordinates[second]
    distance = math.hypot(x2 - x1, y2 - y1)
    if distance_metric is None or distance_metric == "EXACT_2D":
        return distance
    if distance_metric == "EUC_2D":
        return float(math.floor(distance + 0.5))
    raise ValueError(f"unsupported CVRP distance metric: {distance_metric}")


def check_cvrp(instance: dict, solution: dict) -> dict:
    coordinates = instance["coordinates"]
    demands = instance["demands"]
    capacity = instance["capacity"]
    depot = int(instance.get("depot", 0))
    distance_metric = instance.get("distance_metric")
    expected = set(range(len(coordinates))) - {depot}
    visited: list[int] = []
    objective = 0.0

    for route in solution["routes"]:
        if any(type(node) is not int or node not in expected for node in route):
            return {"feasible": False, "detail": "invalid customer index"}
        load = sum(demands[node] for node in route)
        if load > capacity:
            return {
                "feasible": False,
                "detail": f"route capacity {load} exceeds {capacity}",
            }
        path = [depot, *route, depot]
        for first, second in zip(path, path[1:]):
            objective += edge_cost(coordinates, first, second, distance_metric)
        visited.extend(route)

    if len(visited) != len(set(visited)):
        return {"feasible": False, "detail": "duplicate customer"}
    if set(visited) != expected:
        return {"feasible": False, "detail": "customer set mismatch"}
    return {
        "feasible": True,
        "objective": objective,
        "detail": "independent CVRP verification passed",
    }


def check_bin_packing(solution: dict, *, weights: list[int], capacity: int) -> dict:
    bins = solution.get("bins")
    if not isinstance(bins, list) or any(
        not isinstance(bin_items, list) or not bin_items for bin_items in bins
    ):
        return {
            "feasible": False,
            "detail": "bin packing solution requires nonempty bins",
        }
    item_indices = [item for bin_items in bins for item in bin_items]
    if any(type(item) is not int for item in item_indices):
        return {"feasible": False, "detail": "bin packing item index is not an integer"}
    if len(item_indices) != len(weights) or set(item_indices) != set(
        range(len(weights))
    ):
        return {
            "feasible": False,
            "detail": "bin packing items must appear exactly once",
        }
    for bin_index, bin_items in enumerate(bins):
        load = sum(weights[item] for item in bin_items)
        if load > capacity:
            return {"feasible": False, "detail": f"bin {bin_index} exceeds capacity"}
    return {
        "feasible": True,
        "objective": float(len(bins)),
        "detail": "exact integer bin packing verification passed",
    }
