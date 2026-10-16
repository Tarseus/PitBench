"""Dependency-free checks shared by the judge and visible development bench."""

from __future__ import annotations

import math


def format_multidimensional_knapsack_lp(
    weights: list[list[int]], capacities: list[int]
) -> str:
    """Canonical LP representation shared by generation and public checking."""

    if (
        not weights
        or len(weights) != len(capacities)
        or any(len(row) != len(weights[0]) for row in weights)
        or any(value <= 0 for row in weights for value in row)
        or any(capacity <= 0 for capacity in capacities)
    ):
        raise ValueError("invalid multidimensional knapsack model")
    item_count = len(weights[0])
    profits = [sum(row[item] for row in weights) for item in range(item_count)]
    lines = [
        "Minimize",
        " obj: "
        + " ".join(f"- {profit} x{item}" for item, profit in enumerate(profits)),
        "Subject To",
    ]
    for dimension, (row, capacity) in enumerate(zip(weights, capacities, strict=True)):
        lines.append(
            f" c{dimension}: "
            + " + ".join(f"{weight} x{item}" for item, weight in enumerate(row))
            + f" <= {capacity}"
        )
    lines.extend(["Binary", *(f" x{item}" for item in range(item_count)), "End"])
    return "\n".join(lines) + "\n"


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


def check_graph_coloring(instance: dict, solution: dict) -> dict:
    vertex_count = instance.get("vertex_count")
    endpoints = instance.get("edge_endpoints")
    colors = solution.get("colors")
    if (
        type(vertex_count) is not int
        or vertex_count < 2
        or not isinstance(endpoints, list)
        or len(endpoints) % 2
        or not isinstance(colors, list)
        or len(colors) != vertex_count
        or any(
            type(color) is not int or not 0 <= color < vertex_count for color in colors
        )
    ):
        return {"feasible": False, "detail": "invalid graph coloring input or solution"}
    for first, second in zip(endpoints[::2], endpoints[1::2], strict=True):
        if (
            type(first) is not int
            or type(second) is not int
            or not 0 <= first < second < vertex_count
            or colors[first] == colors[second]
        ):
            return {
                "feasible": False,
                "detail": "graph edge is invalid or monochromatic",
            }
    return {
        "feasible": True,
        "objective": float(max(colors) + 1),
        "detail": "independent graph coloring verification passed",
    }
