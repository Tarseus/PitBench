from __future__ import annotations

import json
import math
from pathlib import Path

from pitbench.families.base import ProblemFamilyPlugin, VerificationResult


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
                x1, y1 = coordinates[first]
                x2, y2 = coordinates[second]
                objective += math.hypot(x2 - x1, y2 - y1)
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
