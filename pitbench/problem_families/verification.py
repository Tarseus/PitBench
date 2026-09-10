"""Independent solution verification by problem family."""

from __future__ import annotations

import json
import math
import subprocess
from decimal import Decimal, InvalidOperation
from pathlib import Path

from pitbench.problem_families.base import ProblemFamilyPlugin, VerificationResult


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
            if any(type(node) is not int or node not in expected for node in route):
                return VerificationResult(
                    feasible=False, detail="invalid customer index"
                )
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


class ExternalVerifierFamily(ProblemFamilyPlugin):
    """Runs an evaluator-owned verifier executable with a fixed JSON contract."""

    name = "external"

    def __init__(self, verifier: Path | None = None) -> None:
        self.verifier = verifier

    def verify(self, instance_path: Path, solution_path: Path) -> VerificationResult:
        if self.verifier is None or not self.verifier.is_file():
            return VerificationResult(
                feasible=False,
                detail="private independent verifier is unavailable",
            )
        completed = subprocess.run(
            [str(self.verifier), str(instance_path), str(solution_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            return VerificationResult(
                feasible=False,
                detail=completed.stderr.strip() or "verifier failed",
            )
        return VerificationResult.model_validate(json.loads(completed.stdout))


class MIPFamily(ExternalVerifierFamily):
    name = "mip"


class IntegerBoundaryFamily(ProblemFamilyPlugin):
    """Verify the small, integral boundary models against evaluator-owned coefficients.

    These test cases use integer coefficients and assignments, so checks are exact.
    This is not a replacement for the verifier of arbitrary floating-point MIPs.
    """

    name = "integer_boundary"

    def verify(self, instance_path: Path, solution_path: Path) -> VerificationResult:
        model = json.loads(instance_path.with_suffix(".model.json").read_text())
        solution = json.loads(solution_path.read_text())
        lines = solution["raw_solution"].splitlines()
        values: dict[str, Decimal] = {}
        objective = None
        reading_columns = False
        for line in lines:
            fields = line.split()
            if fields[:2] == ["#", "Columns"]:
                if reading_columns:
                    break  # Do not read dual solution values as primal assignments.
                reading_columns = True
                continue
            if reading_columns and line.startswith("#"):
                break
            try:
                if fields[:1] == ["Objective"] and len(fields) == 2:
                    objective = Decimal(fields[1])
                if reading_columns and len(fields) == 2:
                    if fields[0] in values:
                        raise ValueError("duplicate variable in primal solution")
                    values[fields[0]] = Decimal(fields[1])
            except InvalidOperation as error:
                raise ValueError("non-numeric primal solution") from error
        expected = {f"x{index}" for index in range(len(model["costs"]))}
        if set(values) != expected or objective is None:
            raise ValueError("incomplete primal solution output")
        ordered = [values[f"x{index}"] for index in range(len(expected))]
        if not objective.is_finite() or any(not value.is_finite() for value in ordered):
            return VerificationResult(feasible=False, detail="non-finite solution")
        for value, lower, upper in zip(
            ordered, model["lower"], model["upper"], strict=True
        ):
            if value != value.to_integral_value() or not lower <= value <= upper:
                return VerificationResult(
                    feasible=False, detail="integer variable or bound violated"
                )
        for coefficients, sense, rhs in model["constraints"]:
            activity = sum(
                coefficient * value
                for coefficient, value in zip(coefficients, ordered, strict=True)
            )
            satisfied = {
                "=": activity == rhs,
                ">=": activity >= rhs,
                "<=": activity <= rhs,
            }[sense]
            if not satisfied:
                return VerificationResult(
                    feasible=False, detail="linear constraint violated"
                )
        evaluated = sum(
            coefficient * value
            for coefficient, value in zip(model["costs"], ordered, strict=True)
        )
        reported = solution.get("objective")
        if (
            objective != evaluated
            or reported is None
            or Decimal(str(reported)) != evaluated
        ):
            return VerificationResult(
                feasible=False, detail="objective does not match assignment"
            )
        if (
            (solution.get("solver_status") or "").lower() == "optimal"
            and evaluated != model["known_optimum"]
        ):
            return VerificationResult(
                feasible=False, detail="false optimality claim on known boundary model"
            )
        return VerificationResult(
            feasible=True,
            objective=float(evaluated),
            detail="exact integer boundary verification passed",
        )


class CPFamily(ExternalVerifierFamily):
    name = "cp"
