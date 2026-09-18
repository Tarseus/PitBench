"""Independent solution verification by problem family."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from pitbench.problem_families.base import ProblemFamilyPlugin, VerificationResult


class TrustedOptimumRecord(BaseModel):
    instance_set: str
    instance_id: str
    instance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_kind: Literal["trusted_optimum", "published_bks"] = "trusted_optimum"
    problem_kind: Literal[
        "single_machine_scheduling",
        "job_shop_scheduling",
        "bin_packing",
    ]
    objective_sense: Literal["minimize", "maximize"]
    optimal_objective: int
    reference_solution_uri: str | None = None
    reference_solution_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    optimality_basis: dict
    generation_provenance: dict
    verification_provenance: str

    @model_validator(mode="after")
    def validate_target_kind(self) -> "TrustedOptimumRecord":
        has_reference = self.reference_solution_uri is not None
        has_reference_hash = self.reference_solution_sha256 is not None
        if has_reference != has_reference_hash:
            raise ValueError(
                "reference solution URI and hash must be supplied together"
            )
        if self.target_kind == "trusted_optimum":
            if not has_reference:
                raise ValueError("trusted optimum records require a reference solution")
            return self
        if has_reference:
            raise ValueError(
                "published BKS records must not retain a reference solution"
            )
        source = self.generation_provenance.get("published_bks_source")
        if (
            not isinstance(source, dict)
            or not isinstance(source.get("repository"), str)
            or not isinstance(source.get("artifact_path"), str)
            or not isinstance(source.get("instance_path"), str)
            or not isinstance(source.get("artifact_sha256"), str)
            or len(source["artifact_sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in source["artifact_sha256"]
            )
        ):
            raise ValueError(
                "published BKS records require a hash-bound source artifact"
            )
        return self


class TrustedOptimumOracle(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    protocol: Literal["exact_verified_solve"] = "exact_verified_solve"
    task_id: str
    records: list[TrustedOptimumRecord]

    @model_validator(mode="after")
    def validate_unique_records(self) -> "TrustedOptimumOracle":
        identities = [
            (record.instance_set, record.instance_id) for record in self.records
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("trusted optimum records must have unique identities")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> "TrustedOptimumOracle":
        return cls.model_validate(yaml.safe_load(path.read_text()))


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
                infrastructure_error=True,
            )
        try:
            completed = subprocess.run(
                [str(self.verifier), str(instance_path), str(solution_path)],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as error:
            return VerificationResult(
                feasible=False,
                detail=str(error),
                infrastructure_error=True,
            )
        if completed.returncode != 0:
            return VerificationResult(
                feasible=False,
                detail=completed.stderr.strip() or "verifier failed",
                infrastructure_error=True,
            )
        try:
            return VerificationResult.model_validate(json.loads(completed.stdout))
        except (json.JSONDecodeError, ValueError, TypeError) as error:
            return VerificationResult(
                feasible=False,
                detail=f"invalid verifier output: {error}",
                infrastructure_error=True,
            )


def _parse_highs_primal_solution(
    solution: dict,
) -> tuple[dict[str, Decimal], Decimal]:
    raw_solution = solution.get("raw_solution")
    if not isinstance(raw_solution, str):
        raise ValueError("missing raw HiGHS solution")
    values: dict[str, Decimal] = {}
    objective = None
    reading_columns = False
    for line in raw_solution.splitlines():
        fields = line.split()
        if fields[:2] == ["#", "Columns"]:
            if reading_columns:
                break
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
    if objective is None or not objective.is_finite():
        raise ValueError("missing or nonfinite primal objective")
    if any(not value.is_finite() for value in values.values()):
        raise ValueError("nonfinite primal variable")
    return values, objective


def _check_scheduling_solution(
    solution: dict,
    processing_times: list[int],
) -> VerificationResult:
    try:
        values, reported_objective = _parse_highs_primal_solution(solution)
    except (TypeError, ValueError) as error:
        return VerificationResult(feasible=False, detail=str(error))
    job_count = len(processing_times)
    start_names = {f"s{index}" for index in range(job_count)}
    order_names = {
        f"z{first}_{second}"
        for first in range(job_count)
        for second in range(first + 1, job_count)
    }
    if set(values) != start_names | order_names:
        return VerificationResult(
            feasible=False,
            detail="scheduling solution variable set mismatch",
        )
    if any(value != value.to_integral_value() for value in values.values()):
        return VerificationResult(
            feasible=False,
            detail="scheduling solution requires integer variables",
        )
    horizon = sum(processing_times)
    starts = [values[f"s{index}"] for index in range(job_count)]
    if any(start < 0 or start > horizon for start in starts):
        return VerificationResult(
            feasible=False,
            detail="scheduling start time outside model bounds",
        )
    for first in range(job_count):
        for second in range(first + 1, job_count):
            order = values[f"z{first}_{second}"]
            if order == 1:
                nonoverlap = starts[first] + processing_times[first] <= starts[second]
            elif order == 0:
                nonoverlap = starts[second] + processing_times[second] <= starts[first]
            else:
                return VerificationResult(
                    feasible=False,
                    detail="scheduling order variable is not binary",
                )
            if not nonoverlap:
                return VerificationResult(
                    feasible=False,
                    detail=f"scheduling jobs {first} and {second} overlap",
                )
    objective = sum(starts, Decimal(0))
    if reported_objective != objective:
        return VerificationResult(
            feasible=False,
            detail="scheduling objective does not match start times",
        )
    return VerificationResult(
        feasible=True,
        objective=float(objective),
        detail="exact integer scheduling verification passed",
    )


def _check_bin_packing_solution(
    solution: dict,
    *,
    weights: list[int],
    capacity: int,
) -> VerificationResult:
    bins = solution.get("bins")
    if not isinstance(bins, list) or any(
        not isinstance(bin_items, list) or not bin_items for bin_items in bins
    ):
        return VerificationResult(
            feasible=False,
            detail="bin packing solution requires nonempty bins",
        )
    item_indices = [item for bin_items in bins for item in bin_items]
    if any(type(item) is not int for item in item_indices):
        return VerificationResult(
            feasible=False,
            detail="bin packing item index is not an integer",
        )
    if len(item_indices) != len(weights) or set(item_indices) != set(
        range(len(weights))
    ):
        return VerificationResult(
            feasible=False,
            detail="bin packing items must appear exactly once",
        )
    for bin_index, bin_items in enumerate(bins):
        load = sum(weights[item] for item in bin_items)
        if load > capacity:
            return VerificationResult(
                feasible=False,
                detail=f"bin {bin_index} exceeds capacity",
            )
    return VerificationResult(
        feasible=True,
        objective=float(len(bins)),
        detail="exact integer bin packing verification passed",
    )


def _job_shop_basis(
    basis: dict,
) -> tuple[list[list[tuple[int, int]]], list[int]]:
    jobs = basis.get("jobs")
    start_variable_indices = basis.get("start_variable_indices")
    if (
        basis.get("kind")
        not in {"published_job_shop_optimum", "published_job_shop_bks"}
        or not isinstance(jobs, list)
        or not isinstance(start_variable_indices, list)
    ):
        raise ValueError("invalid job-shop optimality basis")
    operations = []
    for job in jobs:
        if not isinstance(job, list) or not job:
            raise ValueError("job-shop basis contains an invalid job")
        parsed_job = []
        for operation in job:
            if (
                not isinstance(operation, list)
                or len(operation) != 2
                or type(operation[0]) is not int
                or type(operation[1]) is not int
                or operation[0] < 0
                or operation[1] <= 0
            ):
                raise ValueError("job-shop basis contains an invalid operation")
            parsed_job.append((operation[0], operation[1]))
        operations.append(parsed_job)
    operation_count = sum(len(job) for job in operations)
    if (
        len(start_variable_indices) != operation_count
        or any(type(index) is not int or index < 0 for index in start_variable_indices)
        or len(set(start_variable_indices)) != len(start_variable_indices)
    ):
        raise ValueError("job-shop start-variable indices are invalid")
    return operations, start_variable_indices


def _check_job_shop_start_times(
    start_times: list[int],
    jobs: list[list[tuple[int, int]]],
) -> VerificationResult:
    operation_count = sum(len(job) for job in jobs)
    if len(start_times) != operation_count or any(
        type(value) is not int or value < 0 for value in start_times
    ):
        return VerificationResult(
            feasible=False,
            detail="job-shop start times are incomplete or invalid",
        )
    intervals_by_machine: dict[int, list[tuple[int, int]]] = {}
    cursor = 0
    makespan = 0
    for job in jobs:
        previous_end = 0
        for machine, duration in job:
            start = start_times[cursor]
            cursor += 1
            end = start + duration
            if start < previous_end:
                return VerificationResult(
                    feasible=False,
                    detail="job-shop precedence constraint violated",
                )
            previous_end = end
            makespan = max(makespan, end)
            intervals_by_machine.setdefault(machine, []).append((start, end))
    for intervals in intervals_by_machine.values():
        intervals.sort()
        if any(first[1] > second[0] for first, second in zip(intervals, intervals[1:])):
            return VerificationResult(
                feasible=False,
                detail="job-shop machine non-overlap constraint violated",
            )
    return VerificationResult(
        feasible=True,
        objective=float(makespan),
        detail="independent job-shop verification passed",
    )


def _check_job_shop_solution(solution: dict, basis: dict) -> VerificationResult:
    try:
        jobs, start_variable_indices = _job_shop_basis(basis)
    except (TypeError, ValueError) as error:
        return VerificationResult(feasible=False, detail=str(error))
    values = solution.get("values")
    if not isinstance(values, list) or any(type(value) is not int for value in values):
        return VerificationResult(
            feasible=False,
            detail="job-shop solution values are missing or non-integral",
        )
    if max(start_variable_indices, default=-1) >= len(values):
        return VerificationResult(
            feasible=False,
            detail="job-shop solution omits a start variable",
        )
    return _check_job_shop_start_times(
        [values[index] for index in start_variable_indices],
        jobs,
    )


class TrustedOptimumFamily(ProblemFamilyPlugin):
    """Verify one hash-bound instance against an evaluator-owned optimum record."""

    def __init__(
        self,
        record: TrustedOptimumRecord,
        reference_solution_path: Path | None,
    ) -> None:
        self.record = record
        self.reference_solution_path = reference_solution_path

    def _oracle_error(self, detail: str) -> VerificationResult:
        return VerificationResult(
            feasible=False,
            detail=detail,
            infrastructure_error=True,
        )

    def _check_oracle(self, instance_path: Path) -> VerificationResult | None:
        if self.record.objective_sense != "minimize":
            return self._oracle_error("unsupported trusted optimum objective sense")
        if self.record.verification_provenance != (
            "pitbench.problem_families.verification:TrustedOptimumFamily"
        ):
            return self._oracle_error("trusted optimum verifier provenance mismatch")
        if hashlib.sha256(instance_path.read_bytes()).hexdigest() != (
            self.record.instance_sha256
        ):
            return self._oracle_error("trusted optimum instance hash mismatch")
        basis = self.record.optimality_basis
        if self.record.target_kind == "published_bks":
            if self.record.problem_kind != "job_shop_scheduling":
                return self._oracle_error(
                    "published BKS record has an unsupported problem kind"
                )
            try:
                _job_shop_basis(basis)
            except (TypeError, ValueError) as error:
                return self._oracle_error(str(error))
            if basis.get("kind") != "published_job_shop_bks":
                return self._oracle_error("invalid published job-shop BKS basis")
            return None

        if self.reference_solution_path is None:
            return self._oracle_error("trusted optimum reference solution is missing")
        if hashlib.sha256(self.reference_solution_path.read_bytes()).hexdigest() != (
            self.record.reference_solution_sha256
        ):
            return self._oracle_error(
                "trusted optimum reference solution hash mismatch"
            )
        try:
            reference_solution = json.loads(self.reference_solution_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            return self._oracle_error(f"invalid reference solution: {error}")
        if self.record.problem_kind == "single_machine_scheduling":
            processing_times = basis.get("processing_times")
            if (
                basis.get("kind") != "shortest_processing_time"
                or not isinstance(processing_times, list)
                or any(
                    type(value) is not int or value <= 0 for value in processing_times
                )
            ):
                return self._oracle_error("invalid scheduling optimality basis")
            expected_order = sorted(
                range(len(processing_times)),
                key=lambda index: (processing_times[index], index),
            )
            expected_starts = [0] * len(processing_times)
            elapsed = 0
            for index in expected_order:
                expected_starts[index] = elapsed
                elapsed += processing_times[index]
            expected_objective = sum(expected_starts)
            if (
                reference_solution
                != {
                    "order": expected_order,
                    "start_times": expected_starts,
                }
                or expected_objective != self.record.optimal_objective
            ):
                return self._oracle_error("invalid scheduling reference optimum")
            return None

        if self.record.problem_kind == "job_shop_scheduling":
            try:
                jobs, _ = _job_shop_basis(basis)
            except (TypeError, ValueError) as error:
                return self._oracle_error(str(error))
            reference_start_times = reference_solution.get("start_times")
            if not isinstance(reference_start_times, list):
                return self._oracle_error("job-shop reference start times are missing")
            reference_check = _check_job_shop_start_times(reference_start_times, jobs)
            if (
                not reference_check.feasible
                or reference_check.objective != self.record.optimal_objective
            ):
                return self._oracle_error("invalid job-shop reference optimum")
            return None

        try:
            instance = json.loads(instance_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            return self._oracle_error(f"invalid bin packing instance: {error}")
        weights = instance.get("weights")
        capacity = instance.get("capacity")
        if (
            basis.get("kind") != "capacity_lower_bound"
            or not isinstance(weights, list)
            or any(type(weight) is not int or weight <= 0 for weight in weights)
            or type(capacity) is not int
            or capacity <= 0
        ):
            return self._oracle_error("invalid bin packing optimality basis")
        lower_bound = math.ceil(sum(weights) / capacity)
        if basis != {
            "kind": "capacity_lower_bound",
            "total_weight": sum(weights),
            "capacity": capacity,
            "lower_bound": lower_bound,
        }:
            return self._oracle_error("bin packing lower-bound record mismatch")
        reference_check = _check_bin_packing_solution(
            reference_solution,
            weights=weights,
            capacity=capacity,
        )
        if (
            not reference_check.feasible
            or reference_check.objective != lower_bound
            or self.record.optimal_objective != lower_bound
        ):
            return self._oracle_error("invalid bin packing reference optimum")
        return None

    def verify(self, instance_path: Path, solution_path: Path) -> VerificationResult:
        try:
            oracle_error = self._check_oracle(instance_path)
        except (OSError, TypeError, ValueError) as error:
            return self._oracle_error(f"trusted optimum verification error: {error}")
        if oracle_error is not None:
            return oracle_error
        try:
            solution = json.loads(solution_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            return VerificationResult(
                feasible=False, detail=f"invalid solution: {error}"
            )
        if self.record.problem_kind == "single_machine_scheduling":
            return _check_scheduling_solution(
                solution,
                self.record.optimality_basis["processing_times"],
            )
        if self.record.problem_kind == "job_shop_scheduling":
            return _check_job_shop_solution(solution, self.record.optimality_basis)
        instance = json.loads(instance_path.read_text())
        return _check_bin_packing_solution(
            solution,
            weights=instance["weights"],
            capacity=instance["capacity"],
        )


class MIPFamily(ExternalVerifierFamily):
    name = "mip"


class NumericModelFamily(ProblemFamilyPlugin):
    """Independently verify preserved sparse models using the declared tolerance."""

    name = "numeric_model"

    def __init__(self, tolerance: float):
        self.tolerance = tolerance

    def verify(self, instance_path: Path, solution_path: Path) -> VerificationResult:
        from pitbench.evaluator.representations import LinearModelRepresentation

        model = LinearModelRepresentation.read_input(instance_path)
        values = json.loads(solution_path.read_text())["values"]
        checked = self.check_model(model, values, self.tolerance)
        return VerificationResult(
            feasible=checked["feasible"],
            objective=checked.get("objective"),
            detail=json.dumps(checked),
        )

    @staticmethod
    def check_model(model: dict, values, tolerance: float) -> dict:
        import numpy as np

        values = np.asarray(values, dtype=float)
        if values.shape != model["cost"].shape or not np.isfinite(values).all():
            return {
                "feasible": False,
                "reason": "missing, nonfinite or wrong-sized solution",
            }
        activity = model["matrix"] @ values
        if not np.isfinite(activity).all():
            return {"feasible": False, "reason": "nonfinite constraint activity"}
        bound_error = float(
            max(
                np.max(model["col_lower"] - values, initial=0),
                np.max(values - model["col_upper"], initial=0),
            )
        )
        row_error = float(
            max(
                np.max(model["row_lower"] - activity, initial=0),
                np.max(activity - model["row_upper"], initial=0),
            )
        )
        integer_values = values[model["integrality"] == 1]
        integer_error = float(
            np.max(np.abs(integer_values - np.rint(integer_values)), initial=0)
        )
        objective = (
            math.fsum(float(a) * float(b) for a, b in zip(model["cost"], values))
            + model["offset"]
        )
        return {
            "feasible": max(bound_error, row_error, integer_error) <= tolerance,
            "tolerance": tolerance,
            "max_bound_violation": bound_error,
            "max_row_violation": row_error,
            "max_integrality_violation": integer_error,
            "objective": objective,
        }


class IntegerBoundaryFamily(ProblemFamilyPlugin):
    """Verify the small, integral boundary models against evaluator-owned coefficients.

    These test cases use integer coefficients and assignments, so checks are exact.
    This is not a replacement for the verifier of arbitrary floating-point MIPs.
    """

    name = "integer_boundary"

    def verify(self, instance_path: Path, solution_path: Path) -> VerificationResult:
        model = json.loads(instance_path.with_suffix(".model.json").read_text())
        solution = json.loads(solution_path.read_text())
        values, objective = _parse_highs_primal_solution(solution)
        expected = {f"x{index}" for index in range(len(model["costs"]))}
        if set(values) != expected:
            raise ValueError("incomplete primal solution output")
        ordered = [values[f"x{index}"] for index in range(len(expected))]
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
            solution.get("solver_status") or ""
        ).lower() == "optimal" and evaluated != model["known_optimum"]:
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
