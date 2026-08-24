from __future__ import annotations

import json
import math
from pathlib import Path

from pitbench.families.base import ProblemFamilyPlugin, VerificationResult


def _edge_cost(
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


class CVRPFamily(ProblemFamilyPlugin):
    """Independent verifier for PitBench's normalized CVRP JSON format."""

    name = "cvrp"

    def verify(self, instance_path: Path, solution_path: Path) -> VerificationResult:
        instance = json.loads(instance_path.read_text())
        solution = json.loads(solution_path.read_text())
        coordinates = instance["coordinates"]
        demands = instance["demands"]
        capacity = instance["capacity"]
        depot = int(instance.get("depot", 0))
        distance_metric = instance.get("distance_metric")
        expected = set(range(len(coordinates))) - {depot}
        visited: list[int] = []
        objective = 0.0

        for route in solution["routes"]:
            load = sum(demands[node] for node in route)
            if load > capacity:
                return VerificationResult(
                    feasible=False,
                    detail=f"route capacity {load} exceeds {capacity}",
                )
            path = [depot, *route, depot]
            for first, second in zip(path, path[1:]):
                objective += _edge_cost(coordinates, first, second, distance_metric)
            visited.extend(route)

        if len(visited) != len(set(visited)):
            return VerificationResult(feasible=False, detail="duplicate customer")
        if set(visited) != expected:
            return VerificationResult(feasible=False, detail="customer set mismatch")
        return VerificationResult(
            feasible=True,
            objective=objective,
            detail="independent CVRP verification passed",
        )
