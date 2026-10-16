"""Independent solution verification by problem family."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from pitbench.problem_families.base import ProblemFamilyPlugin, VerificationResult
from pitbench.problem_families.public_verification import (
    check_bin_packing,
    check_cvrp,
    check_graph_coloring,
    edge_cost,
    format_multidimensional_knapsack_lp,
)

_edge_cost = edge_cost
_SCHEDULING_INTEGRALITY_TOLERANCE = Decimal("0.000001")


def validate_bin_packing_optimum_record(
    proof: dict,
    *,
    instance_sha256: str,
    weights: list[int],
    capacity: int,
    reference_bins: list[list[int]],
) -> int:
    """Check a fixed external solver result and its independently checked packing.

    The external solver's optimal bound is retained as provenance; this check
    does not reconstruct that solver's search proof.
    """
    if not isinstance(proof, dict):
        raise ValueError("bin packing optimum record is not an object")
    if not isinstance(reference_bins, list) or any(
        not isinstance(bin_items, list) for bin_items in reference_bins
    ):
        raise ValueError("bin packing reference bins are missing")
    solver_commit = proof.get("solver_commit")
    adapter_source_sha256 = proof.get("adapter_source_sha256")
    budget = proof.get("budget_sec")
    objective = proof.get("objective")
    best_bound = proof.get("best_bound")
    if (
        proof.get("instance_sha256") != instance_sha256
        or not isinstance(solver_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", solver_commit) is None
        or not isinstance(adapter_source_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", adapter_source_sha256) is None
        or proof.get("solver_threads") != 1
        or proof.get("solver_seed") != 0
        or type(budget) not in {int, float}
        or not math.isfinite(budget)
        or budget <= 0
        or proof.get("status") != "OPTIMAL"
        or type(objective) is not int
        or objective <= 0
        or type(best_bound) not in {int, float}
        or not math.isfinite(best_bound)
        or best_bound != objective
        or proof.get("bins") != reference_bins
    ):
        raise ValueError("bin packing optimum record is inconsistent")
    checked = check_bin_packing(
        {"bins": reference_bins}, weights=weights, capacity=capacity
    )
    if not checked["feasible"] or checked["objective"] != objective:
        raise ValueError("bin packing reference does not match optimum record")
    return objective


class TrustedOptimumRecord(BaseModel):
    instance_set: str
    instance_id: str
    instance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_kind: Literal["trusted_optimum", "published_bks"] = "trusted_optimum"
    problem_kind: Literal[
        "single_machine_scheduling",
        "job_shop_scheduling",
        "bin_packing",
        "multidimensional_knapsack",
        "graph_coloring",
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


class CVRPFamily(ProblemFamilyPlugin):
    """Independent verifier for PitBench's normalized CVRP JSON format."""

    name = "cvrp"

    def verify(self, instance_path: Path, solution_path: Path) -> VerificationResult:
        instance = json.loads(instance_path.read_text())
        solution = json.loads(solution_path.read_text())
        return VerificationResult.model_validate(check_cvrp(instance, solution))


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


class ProcessVerifierFamily(ProblemFamilyPlugin):
    """Run an evaluator-owned verifier command with instance and solution inputs."""

    name = ""

    def __init__(self, command_prefix: list[str]) -> None:
        self.command_prefix = command_prefix

    def verify(self, instance_path: Path, solution_path: Path) -> VerificationResult:
        try:
            completed = subprocess.run(
                [*self.command_prefix, str(instance_path), str(solution_path)],
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
    rounded_values = {name: value.to_integral_value() for name, value in values.items()}
    if any(
        abs(value - rounded_values[name]) > _SCHEDULING_INTEGRALITY_TOLERANCE
        for name, value in values.items()
    ):
        return VerificationResult(
            feasible=False,
            detail="scheduling solution exceeds integrality tolerance",
        )
    values = rounded_values
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
    if abs(reported_objective - objective) > _SCHEDULING_INTEGRALITY_TOLERANCE:
        return VerificationResult(
            feasible=False,
            detail="scheduling objective does not match start times",
        )
    return VerificationResult(
        feasible=True,
        objective=float(objective),
        detail="exact integer scheduling verification passed",
    )


def _public_scheduling_processing_times(instance_path: Path) -> list[int] | None:
    """Recover the declared processing times from PitBench's public LP form."""
    try:
        lines = instance_path.read_text().splitlines()
    except OSError:
        return None
    processing: dict[int, int] = {}
    horizon = None
    first_pattern = re.compile(
        r"^\s*c\d+:\s+s(\d+)\s+-\s+s\d+\s+\+\s+(\d+)\s+z\d+_\d+\s+<=\s+(-?\d+)"
    )
    second_pattern = re.compile(
        r"^\s*c\d+:\s+s(\d+)\s+-\s+s\d+\s+-\s+(\d+)\s+z\d+_\d+\s+<=\s+(-?\d+)"
    )
    for line in lines:
        match = first_pattern.match(line)
        if match is None:
            match = second_pattern.match(line)
            if match is None:
                continue
            job, candidate_horizon, rhs = map(int, match.groups())
            duration = -rhs
        else:
            job, candidate_horizon, rhs = map(int, match.groups())
            duration = candidate_horizon - rhs
        if horizon is None:
            horizon = candidate_horizon
        if candidate_horizon != horizon:
            return None
        if duration <= 0 or (job in processing and processing[job] != duration):
            return None
        processing[job] = duration
    if horizon is None or not processing:
        return None
    job_count = max(processing) + 1
    if set(processing) != set(range(job_count)):
        return None
    if sum(processing.values()) != horizon:
        return None
    return [processing[index] for index in range(job_count)]


def _check_bin_packing_solution(
    solution: dict,
    *,
    weights: list[int],
    capacity: int,
) -> VerificationResult:
    return VerificationResult.model_validate(
        check_bin_packing(solution, weights=weights, capacity=capacity)
    )


def _public_multidimensional_knapsack_data(
    instance_path: Path,
) -> tuple[list[list[int]], list[int]] | None:
    """Accept only the canonical LP emitted by the shared instance generator."""

    try:
        source = instance_path.read_text()
    except OSError:
        return None
    lines = source.splitlines()
    if len(lines) < 8 or lines[:1] != ["Minimize"] or "Subject To" not in lines:
        return None
    subject = lines.index("Subject To")
    if subject != 2 or "Binary" not in lines:
        return None
    binary = lines.index("Binary")
    weights: list[list[int]] = []
    capacities: list[int] = []
    for dimension, line in enumerate(lines[subject + 1 : binary]):
        match = re.fullmatch(r" c(\d+): (.+) <= (\d+)", line)
        if match is None or int(match.group(1)) != dimension:
            return None
        terms = match.group(2).split(" + ")
        row: list[int] = []
        for item, term in enumerate(terms):
            parsed = re.fullmatch(r"(\d+) x(\d+)", term)
            if parsed is None or int(parsed.group(2)) != item:
                return None
            row.append(int(parsed.group(1)))
        weights.append(row)
        capacities.append(int(match.group(3)))
    try:
        canonical = format_multidimensional_knapsack_lp(weights, capacities)
    except ValueError:
        return None
    return (weights, capacities) if source == canonical else None


def _check_multidimensional_knapsack_solution(
    solution: dict,
    weights: list[list[int]],
    capacities: list[int],
) -> VerificationResult:
    try:
        values, reported_objective = _parse_highs_primal_solution(solution)
    except (TypeError, ValueError) as error:
        return VerificationResult(feasible=False, detail=str(error))
    item_count = len(weights[0])
    if set(values) != {f"x{item}" for item in range(item_count)}:
        return VerificationResult(
            feasible=False, detail="knapsack variable set mismatch"
        )
    assignment: list[int] = []
    for item in range(item_count):
        value = values[f"x{item}"]
        rounded = value.to_integral_value()
        if abs(value - rounded) > _SCHEDULING_INTEGRALITY_TOLERANCE or rounded not in {
            0,
            1,
        }:
            return VerificationResult(
                feasible=False, detail="nonbinary knapsack assignment"
            )
        assignment.append(int(rounded))
    if any(
        sum(weight * assignment[item] for item, weight in enumerate(row)) > capacity
        for row, capacity in zip(weights, capacities, strict=True)
    ):
        return VerificationResult(feasible=False, detail="knapsack capacity exceeded")
    objective = -sum(
        sum(row[item] for row in weights) * assignment[item]
        for item in range(item_count)
    )
    if abs(reported_objective - objective) > _SCHEDULING_INTEGRALITY_TOLERANCE:
        return VerificationResult(feasible=False, detail="knapsack objective mismatch")
    return VerificationResult(
        feasible=True,
        objective=float(objective),
        detail="independent multidimensional knapsack verification passed",
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
        if not isinstance(reference_solution, dict):
            return self._oracle_error("trusted optimum reference is not an object")
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

        if self.record.problem_kind == "multidimensional_knapsack":
            model = _public_multidimensional_knapsack_data(instance_path)
            if model is None:
                return self._oracle_error("invalid knapsack instance")
            weights, capacities = model
            selected = reference_solution.get("selected_items")
            if (
                not isinstance(selected, list)
                or any(type(item) is not int for item in selected)
                or len(set(selected)) != len(selected)
                or any(not 0 <= item < len(weights[0]) for item in selected)
                or any(
                    sum(row[item] for item in selected) != capacity
                    for row, capacity in zip(weights, capacities, strict=True)
                )
                or basis
                != {
                    "kind": "capacity_dual_bound",
                    "dimensions": len(weights),
                    "items": len(weights[0]),
                }
                or self.record.optimal_objective != -sum(capacities)
            ):
                return self._oracle_error("invalid knapsack reference optimum")
            return None

        if self.record.problem_kind == "graph_coloring":
            from pitbench.instances.generate import mycielski_graph

            try:
                instance = json.loads(instance_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                return self._oracle_error(f"invalid graph instance: {error}")
            chromatic_number = basis.get("chromatic_number")
            permutation = basis.get("permutation")
            edges = instance.get("edge_endpoints")
            if (
                basis.get("kind") != "mycielski_lower_bound"
                or type(chromatic_number) is not int
                or not 2 <= chromatic_number <= 9
                or not isinstance(permutation, list)
                or not isinstance(edges, list)
                or any(type(vertex) is not int for vertex in permutation)
                or chromatic_number != self.record.optimal_objective
            ):
                return self._oracle_error("invalid Mycielski graph basis")
            canonical_edges, canonical_colors = mycielski_graph(chromatic_number)
            vertex_count = len(canonical_colors)
            if (
                instance.get("problem_kind") != "graph_coloring"
                or instance.get("vertex_count") != vertex_count
                or len(permutation) != vertex_count
                or set(permutation) != set(range(vertex_count))
                or len(edges) % 2
            ):
                return self._oracle_error("graph coloring instance differs from basis")
            old_to_new = {old: new for new, old in enumerate(permutation)}
            edge_set = set(zip(edges[::2], edges[1::2], strict=True))
            if any(
                (
                    min(old_to_new[first], old_to_new[second]),
                    max(old_to_new[first], old_to_new[second]),
                )
                not in edge_set
                for first, second in canonical_edges
            ):
                return self._oracle_error("Mycielski lower-bound subgraph is absent")
            checked = check_graph_coloring(instance, reference_solution)
            if not checked["feasible"] or checked["objective"] != chromatic_number:
                return self._oracle_error("invalid graph coloring reference optimum")
            return None

        try:
            instance = json.loads(instance_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            return self._oracle_error(f"invalid bin packing instance: {error}")
        weights = instance.get("weights")
        capacity = instance.get("capacity")
        if (
            not isinstance(weights, list)
            or any(type(weight) is not int or weight <= 0 for weight in weights)
            or type(capacity) is not int
            or capacity <= 0
        ):
            return self._oracle_error("invalid bin packing optimality basis")
        if basis.get("kind") == "independent_solver_optimal":
            if basis.get("solver") != "ortools_cp_sat":
                return self._oracle_error("unsupported bin packing proof solver")
            try:
                optimum = validate_bin_packing_optimum_record(
                    basis.get("proof"),
                    instance_sha256=self.record.instance_sha256,
                    weights=weights,
                    capacity=capacity,
                    reference_bins=reference_solution.get("bins"),
                )
            except (TypeError, ValueError) as error:
                return self._oracle_error(str(error))
            if self.record.optimal_objective != optimum:
                return self._oracle_error("bin packing optimum record target mismatch")
            return None
        if basis.get("kind") != "capacity_lower_bound":
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
        if self.record.problem_kind == "multidimensional_knapsack":
            model = _public_multidimensional_knapsack_data(instance_path)
            if model is None:
                return self._oracle_error("invalid knapsack instance")
            return _check_multidimensional_knapsack_solution(solution, *model)
        instance = json.loads(instance_path.read_text())
        if self.record.problem_kind == "graph_coloring":
            return VerificationResult.model_validate(
                check_graph_coloring(instance, solution)
            )
        return _check_bin_packing_solution(
            solution,
            weights=instance["weights"],
            capacity=instance["capacity"],
        )


class MIPFamily(ExternalVerifierFamily):
    name = "mip"
    fixture_reports_search_nodes = True

    def verify(self, instance_path: Path, solution_path: Path) -> VerificationResult:
        if self.verifier is not None:
            return super().verify(instance_path, solution_path)
        processing_times = _public_scheduling_processing_times(instance_path)
        if processing_times is not None:
            try:
                solution = json.loads(solution_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                return VerificationResult(feasible=False, detail=str(error))
            return _check_scheduling_solution(solution, processing_times)
        knapsack = _public_multidimensional_knapsack_data(instance_path)
        if knapsack is not None:
            try:
                solution = json.loads(solution_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                return VerificationResult(feasible=False, detail=str(error))
            return _check_multidimensional_knapsack_solution(solution, *knapsack)
        return super().verify(instance_path, solution_path)


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
    reliability_uses_task_development_instances = True

    def verify(self, instance_path: Path, solution_path: Path) -> VerificationResult:
        """Verify public normalized bin-packing CP instances without a private oracle.

        Exact judge instances attach their hash-bound trusted-optimum verifier to
        each case. Visible development instances have no private oracle record,
        but use the same public normalized bin-packing format and can be
        independently verified here. An explicitly configured external verifier
        remains authoritative.
        """
        if self.verifier is not None:
            return super().verify(instance_path, solution_path)
        if instance_path.suffix.lower() != ".json":
            return super().verify(instance_path, solution_path)
        try:
            instance = json.loads(instance_path.read_text())
            solution = json.loads(solution_path.read_text())
            if instance.get("problem_kind") == "graph_coloring":
                return VerificationResult.model_validate(
                    check_graph_coloring(instance, solution)
                )
            weights = instance["weights"]
            capacity = instance["capacity"]
        except (
            KeyError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            return VerificationResult(
                feasible=False,
                detail=f"unsupported normalized CP instance or solution: {error}",
            )
        if (
            not isinstance(weights, list)
            or any(type(weight) is not int or weight <= 0 for weight in weights)
            or type(capacity) is not int
            or capacity <= 0
        ):
            return VerificationResult(
                feasible=False,
                detail="invalid normalized bin packing instance",
            )
        return _check_bin_packing_solution(
            solution,
            weights=weights,
            capacity=capacity,
        )
