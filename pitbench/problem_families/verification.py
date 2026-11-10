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

from pitbench.problem_families.base import (
    OBJECTIVE_ABSOLUTE_TOLERANCE,
    ProblemFamilyPlugin,
    VerificationResult,
    objectives_equal,
)
from pitbench.problem_families.public_verification import (
    check_bin_packing,
    check_cvrp,
    check_graph_coloring,
    check_resource_constrained_project_schedule,
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


def _check_balanced_subset_sum_solution(
    instance: dict, solution: dict
) -> VerificationResult:
    weights = instance.get("weights")
    cardinality = instance.get("cardinality")
    target = instance.get("target")
    selected = solution.get("selected_items")
    if (
        instance.get("problem_kind") != "balanced_subset_sum"
        or not isinstance(weights, list)
        or len(weights) < 2
        or any(type(weight) is not int or weight <= 0 for weight in weights)
        or type(cardinality) is not int
        or not 0 < cardinality < len(weights)
        or type(target) is not int
        or not 0 <= target <= sum(weights)
        or instance.get("domain_mode") not in {"bounded", "enumerated"}
        or not isinstance(selected, list)
        or len(selected) != cardinality
        or any(
            type(item) is not int or not 0 <= item < len(weights) for item in selected
        )
        or len(set(selected)) != len(selected)
    ):
        return VerificationResult(
            feasible=False, detail="invalid balanced subset-sum instance or solution"
        )
    objective = abs(sum(weights[item] for item in selected) - target)
    return VerificationResult(
        feasible=True,
        objective=objective,
        detail="original balanced subset-sum constraints verified",
    )


class TrustedOptimumRecord(BaseModel):
    instance_set: str
    instance_id: str
    instance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_kind: Literal["trusted_optimum", "published_bks", "best_found"] = (
        "trusted_optimum"
    )
    problem_kind: Literal[
        "single_machine_scheduling",
        "job_shop_scheduling",
        "bin_packing",
        "multidimensional_knapsack",
        "set_covering",
        "binary_clique_cover",
        "binary_clique_cover_fme",
        "graph_coloring",
        "balanced_subset_sum",
        "resource_constrained_project_scheduling",
        "linear_mip",
    ]
    objective_sense: Literal["minimize", "maximize"]
    optimal_objective: int | float = Field(allow_inf_nan=False)
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
        if self.target_kind in {"trusted_optimum", "best_found"}:
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


def _parse_mip_primal_solution(
    solution: dict,
    variable_names: list[str] | None = None,
) -> tuple[dict[str, Decimal], Decimal]:
    structured = solution.get("values")
    if structured is not None:
        if not isinstance(structured, dict) or not structured:
            raise ValueError("invalid MIP variable assignment")
        try:
            values = {
                name: Decimal(str(value))
                for name, value in structured.items()
                if isinstance(name, str) and type(value) in {int, float, str}
            }
            objective = Decimal(str(solution["objective"]))
        except (KeyError, InvalidOperation) as error:
            raise ValueError("invalid MIP objective or assignment") from error
        if (
            len(values) != len(structured)
            or not objective.is_finite()
            or any(not value.is_finite() for value in values.values())
        ):
            raise ValueError("nonfinite or invalid MIP solution")
        return values, objective
    raw_solution = solution.get("raw_solution")
    if not isinstance(raw_solution, str):
        raise ValueError("missing raw MIP solution")
    values: dict[str, Decimal] = {}
    objective = solution.get("objective")
    reading_columns = False
    legacy_column_values = []
    legacy_solution_available = False
    for line in raw_solution.splitlines():
        fields = line.split()
        if fields[:2] == ["#", "Columns"]:
            if reading_columns:
                break
            reading_columns = True
            continue
        if fields[:1] == ["Columns"]:
            reading_columns = True
            continue
        if reading_columns and line.startswith("#"):
            break
        if reading_columns and fields[:1] == ["Rows"]:
            break
        try:
            if fields[:1] == ["Objective"] and len(fields) == 2:
                objective = Decimal(fields[1])
            if reading_columns and len(fields) == 2:
                if fields[0] in values:
                    raise ValueError("duplicate variable in primal solution")
                try:
                    values[fields[0]] = Decimal(fields[1])
                except InvalidOperation:
                    if variable_names is None:
                        raise ValueError(
                            "legacy column solution needs the original variable order"
                        )
                    values.clear()
                    raise
            elif reading_columns and len(fields) == 1:
                legacy_column_values.append(Decimal(fields[0]))
        except InvalidOperation as error:
            raise ValueError("non-numeric primal solution") from error
    if not values and legacy_column_values:
        if variable_names is None or len(legacy_column_values) != len(variable_names):
            raise ValueError("legacy primal column count differs from the model")
        values = dict(zip(variable_names, legacy_column_values, strict=True))
        legacy_solution_available = True
    if legacy_solution_available and "T Primal solution" not in raw_solution:
        raise ValueError("legacy solution does not contain a primal assignment")
    try:
        objective = Decimal(str(objective))
    except InvalidOperation as error:
        raise ValueError("missing or nonfinite primal objective") from error
    if not objective.is_finite():
        raise ValueError("missing or nonfinite primal objective")
    if any(not value.is_finite() for value in values.values()):
        raise ValueError("nonfinite primal variable")
    return values, objective


def _check_scheduling_solution(
    solution: dict,
    processing_times: list[int],
) -> VerificationResult:
    try:
        variable_names = [f"s{index}" for index in range(len(processing_times))]
        variable_names.extend(
            f"z{first}_{second}"
            for first in range(len(processing_times))
            for second in range(first + 1, len(processing_times))
        )
        values, reported_objective = _parse_mip_primal_solution(
            solution, variable_names
        )
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


def _read_mps_numeric_model(instance_path: Path) -> dict | None:
    """Read the free-format MPS subset used by HiGHS without solver bindings."""
    try:
        lines = instance_path.read_text().splitlines()
    except OSError:
        return None

    section = None
    objective_sense = "minimize"
    row_types: dict[str, str] = {}
    columns: list[str] = []
    seen_columns: set[str] = set()
    coefficients: dict[str, dict[str, float]] = {}
    cost: dict[str, float] = {}
    rhs: dict[str, float] = {}
    column_lower: dict[str, float] = {}
    column_upper: dict[str, float] = {}
    integer_columns: set[str] = set()
    integer_marker = False
    rhs_set = None
    bound_set = None
    unsupported = False

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("*"):
            continue
        fields = line.split()
        token = fields[0].upper()
        if token in {"ROWS", "COLUMNS", "RHS", "RANGES", "BOUNDS", "ENDATA"}:
            section = token
            if token == "ENDATA":
                break
            if token == "RANGES":
                unsupported = True
            continue
        if token in {"OBJSENSE", "OBJNAME"}:
            section = token
            continue

        if section == "OBJSENSE":
            if token in {"MIN", "MINIMIZE"}:
                objective_sense = "minimize"
            elif token in {"MAX", "MAXIMIZE"}:
                objective_sense = "maximize"
            else:
                unsupported = True
            continue
        if section == "OBJNAME":
            continue
        if section == "ROWS":
            if len(fields) != 2 or fields[0].upper() not in {"N", "L", "G", "E"}:
                return None
            row_types[fields[1]] = fields[0].upper()
            continue
        if section == "COLUMNS":
            if len(fields) >= 3 and fields[1].strip("'").upper() == "MARKER":
                marker = fields[-1].strip("'").upper()
                if marker == "INTORG":
                    integer_marker = True
                elif marker == "INTEND":
                    integer_marker = False
                else:
                    return None
                continue
            if len(fields) < 3 or (len(fields) - 1) % 2:
                return None
            column = fields[0]
            if column not in seen_columns:
                seen_columns.add(column)
                columns.append(column)
                if integer_marker:
                    integer_columns.add(column)
            for index in range(1, len(fields), 2):
                row, raw_value = fields[index], fields[index + 1]
                value = float(raw_value)
                if row not in row_types:
                    return None
                if row_types[row] == "N":
                    cost[column] = cost.get(column, 0.0) + value
                else:
                    row_coefficients = coefficients.setdefault(row, {})
                    row_coefficients[column] = row_coefficients.get(column, 0.0) + value
            continue
        if section == "RHS":
            if len(fields) < 3 or (len(fields) - 1) % 2:
                return None
            if rhs_set is None:
                rhs_set = fields[0]
            if fields[0] != rhs_set:
                continue
            for index in range(1, len(fields), 2):
                row, value = fields[index], float(fields[index + 1])
                if row in row_types and row_types[row] == "N":
                    unsupported = True
                elif row in row_types:
                    rhs[row] = value
            continue
        if section == "BOUNDS":
            if len(fields) not in {3, 4}:
                return None
            bound_type, current_set, column = fields[:3]
            bound_type = bound_type.upper()
            if bound_set is None:
                bound_set = current_set
            if current_set != bound_set:
                continue
            value = float(fields[3]) if len(fields) == 4 else None
            lower = column_lower.get(column, 0.0)
            upper = column_upper.get(column, math.inf)
            if bound_type == "BV":
                lower, upper = 0.0, 1.0
                integer_columns.add(column)
            elif bound_type == "LI" and value is not None:
                lower = value
                integer_columns.add(column)
            elif bound_type == "UI" and value is not None:
                upper = value
                integer_columns.add(column)
            elif bound_type == "LO" and value is not None:
                lower = value
            elif bound_type == "UP" and value is not None:
                upper = value
            elif bound_type == "FX" and value is not None:
                lower = upper = value
            elif bound_type == "FR":
                lower, upper = -math.inf, math.inf
            elif bound_type == "MI":
                lower = -math.inf
            elif bound_type == "PL":
                upper = math.inf
            else:
                return None
            column_lower[column], column_upper[column] = lower, upper

    if unsupported or not columns:
        return None
    objective_rows = [name for name, kind in row_types.items() if kind == "N"]
    if len(objective_rows) != 1:
        return None
    rows = [name for name, kind in row_types.items() if kind != "N"]
    row_lower, row_upper = {}, {}
    for row in rows:
        sense = row_types[row]
        value = rhs.get(row, 0.0)
        if sense == "L":
            row_lower[row], row_upper[row] = -math.inf, value
        elif sense == "G":
            row_lower[row], row_upper[row] = value, math.inf
        else:
            row_lower[row] = row_upper[row] = value
    for column in columns:
        column_lower.setdefault(column, 0.0)
        column_upper.setdefault(column, math.inf)
        cost.setdefault(column, 0.0)
    return {
        "columns": columns,
        "rows": rows,
        "coefficients": coefficients,
        "cost": cost,
        "offset": 0.0,
        "row_lower": row_lower,
        "row_upper": row_upper,
        "column_lower": column_lower,
        "column_upper": column_upper,
        "integer_columns": integer_columns,
        "objective_sense": objective_sense,
    }


def _mps_metadata_payload(instance_path: Path, marker: str) -> dict | None:
    prefix = f"* {marker} "
    parts = []
    count = None
    for line in instance_path.read_text().splitlines():
        if not line.startswith(prefix):
            continue
        sequence, separator, payload = line[len(prefix) :].partition(" ")
        index_text, slash, count_text = sequence.partition("/")
        if not separator or not slash:
            raise ValueError(f"invalid {marker} MPS metadata")
        index, current_count = int(index_text), int(count_text)
        if count is None:
            count = current_count
        if current_count != count or index != len(parts) + 1:
            raise ValueError(f"nonsequential {marker} MPS metadata")
        parts.append(payload)
    if count is None:
        return None
    if len(parts) != count:
        raise ValueError(f"incomplete {marker} MPS metadata")
    payload = json.loads("".join(parts))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid {marker} MPS metadata payload")
    return payload


def _mip_graph_coloring_source_check(
    instance_path: Path, solution: dict, result: VerificationResult
) -> VerificationResult:
    from pitbench.problem_families.public_verification import check_graph_coloring

    graph = _mps_metadata_payload(instance_path, "PITBENCH_GRAPH_DATA")
    if graph is None:
        return result
    model = _read_mps_numeric_model(instance_path)
    if model is None:
        return VerificationResult(feasible=False, detail="invalid graph-coloring MPS")
    vertex_count = graph.get("vertex_count")
    endpoints = graph.get("edge_endpoints")
    if (
        type(vertex_count) is not int
        or vertex_count < 2
        or not isinstance(endpoints, list)
        or len(endpoints) % 2
        or any(type(value) is not int for value in endpoints)
    ):
        return VerificationResult(feasible=False, detail="invalid graph metadata")
    formulation = next(
        (
            line
            for line in instance_path.read_text().splitlines()
            if line.startswith("* PITBENCH_FORMULATION graph_coloring_mip ")
        ),
        None,
    )
    if formulation is None:
        return VerificationResult(
            feasible=False, detail="graph-coloring MPS formulation marker is missing"
        )
    color_count = int(formulation.split("colors=", 1)[1].split()[0])
    expected_columns = {
        *(f"x{vertex}_{color}" for vertex in range(vertex_count) for color in range(color_count)),
        *(f"y{color}" for color in range(color_count)),
    }
    edges = list(zip(endpoints[::2], endpoints[1::2], strict=True))
    expected_rows = {
        *(f"v{vertex}" for vertex in range(vertex_count)),
        *(f"e{edge}_{color}" for edge in range(len(edges)) for color in range(color_count)),
        *(f"l{vertex}_{color}" for vertex in range(vertex_count) for color in range(color_count)),
        *(f"s{color}" for color in range(color_count - 1)),
    }
    if set(model["columns"]) != expected_columns or set(model["rows"]) != expected_rows:
        return VerificationResult(
            feasible=False, detail="graph-coloring MPS structure differs from source graph"
        )
    expected_costs = {f"y{color}": 1 for color in range(color_count)}
    if model["cost"] != {
        column: expected_costs.get(column, 0) for column in model["columns"]
    }:
        return VerificationResult(
            feasible=False, detail="graph-coloring MPS objective differs from source model"
        )

    def matches_row(name: str, sense: str, rhs: float, terms: dict[str, float]) -> bool:
        expected_lower = rhs if sense in {"E", "G"} else -math.inf
        expected_upper = rhs if sense in {"E", "L"} else math.inf
        actual_terms = model["coefficients"].get(name, {})
        return (
            math.isclose(model["row_lower"][name], expected_lower)
            and math.isclose(model["row_upper"][name], expected_upper)
            and actual_terms == terms
        )

    for vertex in range(vertex_count):
        if not matches_row(
            f"v{vertex}",
            "E",
            1,
            {f"x{vertex}_{color}": 1 for color in range(color_count)},
        ):
            return VerificationResult(
                feasible=False, detail="graph-coloring assignment row is malformed"
            )
        for color in range(color_count):
            if not matches_row(
                f"l{vertex}_{color}",
                "L",
                0,
                {f"x{vertex}_{color}": 1, f"y{color}": -1},
            ):
                return VerificationResult(
                    feasible=False, detail="graph-coloring color-use row is malformed"
                )
    for edge_index, (first, second) in enumerate(edges):
        for color in range(color_count):
            if not matches_row(
                f"e{edge_index}_{color}",
                "L",
                1,
                {f"x{first}_{color}": 1, f"x{second}_{color}": 1},
            ):
                return VerificationResult(
                    feasible=False, detail="graph edge is absent from MIP formulation"
                )
    for color in range(color_count - 1):
        if not matches_row(
            f"s{color}",
            "G",
            0,
            {f"y{color}": 1, f"y{color + 1}": -1},
        ):
            return VerificationResult(
                feasible=False, detail="graph-coloring symmetry row is malformed"
            )
    values, _ = _parse_mip_primal_solution(solution, model["columns"])
    colors = []
    for vertex in range(vertex_count):
        selected = [
            color
            for color in range(color_count)
            if values[f"x{vertex}_{color}"] == 1
        ]
        if len(selected) != 1:
            return VerificationResult(
                feasible=False, detail=f"vertex {vertex} has no unique selected color"
            )
        colors.append(selected[0])
    checked = check_graph_coloring(graph, {"colors": colors})
    if (
        not checked["feasible"]
        or checked["objective"] != result.objective
        or sum(values[f"y{color}"] for color in range(color_count))
        != checked["objective"]
    ):
        return VerificationResult(
            feasible=False, detail="MIP assignment fails original graph coloring semantics"
        )
    return VerificationResult(
        feasible=True,
        objective=checked["objective"],
        detail="independent graph coloring semantics and MPS assignment verified",
    )


def _mip_weighted_tardiness_source_check(
    instance_path: Path, solution: dict, result: VerificationResult
) -> VerificationResult:
    data = _mps_metadata_payload(instance_path, "PITBENCH_SCHEDULING_DATA")
    if data is None:
        return result
    processing = data.get("processing_times")
    releases = data.get("release_dates")
    dues = data.get("due_dates")
    weights = data.get("weights")
    setups = data.get("setup_times")
    count = len(processing) if isinstance(processing, list) else 0
    if (
        count < 2
        or any(
            not isinstance(values, list) or len(values) != count
            for values in (releases, dues, weights)
        )
        or not isinstance(setups, list)
        or len(setups) != count
        or any(not isinstance(row, list) or len(row) != count for row in setups)
    ):
        return VerificationResult(feasible=False, detail="invalid scheduling metadata")
    values, _ = _parse_mip_primal_solution(solution)
    # The generic MPS verifier already checks integrality within its frozen
    # tolerance. Round near-integral solver output before exact source checks.
    start_times = [round(float(values[f"s{job}"])) for job in range(count)]
    tardiness = [round(float(values[f"t{job}"])) for job in range(count)]
    for job, start in enumerate(start_times):
        if start < releases[job]:
            return VerificationResult(
                feasible=False, detail=f"job {job} violates its release date"
            )
        actual = max(0, start + processing[job] - dues[job])
        if tardiness[job] < actual:
            return VerificationResult(
                feasible=False, detail=f"job {job} tardiness variable understates lateness"
            )
    for first in range(count):
        for second in range(first + 1, count):
            before = round(float(values[f"y{first}_{second}"]))
            if before == 1:
                valid = (
                    start_times[second]
                    >= start_times[first]
                    + processing[first]
                    + setups[first][second]
                )
            else:
                valid = (
                    start_times[first]
                    >= start_times[second]
                    + processing[second]
                    + setups[second][first]
                )
            if not valid:
                return VerificationResult(
                    feasible=False,
                    detail=f"sequence-dependent setup violated for {first},{second}",
                )
    objective = sum(
        weight * max(0, start + processing[job] - dues[job])
        for job, (weight, start) in enumerate(zip(weights, start_times, strict=True))
    )
    return VerificationResult(
        feasible=True,
        objective=float(objective),
        detail="independent weighted-tardiness schedule and source objective verified",
    )


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
    """Legacy capacity-dual models only; unrelated profits need another proof."""
    model = _public_multidimensional_knapsack_model(instance_path)
    if model is None:
        return None
    weights, capacities, profits = model
    return (
        (weights, capacities)
        if profits == [sum(column) for column in zip(*weights, strict=True)]
        else None
    )


def _public_multidimensional_knapsack_model(instance_path: Path):
    """Parse all coefficients from the original binary-knapsack LP or MPS."""

    if instance_path.suffix.lower() == ".mps":
        model = _read_mps_numeric_model(instance_path)
        if model is None:
            return None
        row_count, item_count = len(model["rows"]), len(model["columns"])
        expected_columns = [f"x{item}" for item in range(item_count)]
        if (
            row_count < 1
            or item_count < 1
            or model["columns"] != expected_columns
            or model["rows"] != [f"c{row}" for row in range(row_count)]
            or model["objective_sense"] != "minimize"
            or model["offset"] != 0
            or any(model["column_lower"][name] != 0 for name in expected_columns)
            or any(model["column_upper"][name] != 1 for name in expected_columns)
            or model["integer_columns"] != set(expected_columns)
            or any(
                not math.isinf(model["row_lower"][name]) or model["row_lower"][name] > 0
                for name in model["rows"]
            )
            or any(
                value < 0 or not float(value).is_integer()
                for value in model["row_upper"].values()
            )
        ):
            return None
        weights = []
        for row in model["rows"]:
            values = model["coefficients"].get(row, {})
            if set(values) - set(expected_columns) or any(
                value < 0 or not float(value).is_integer() for value in values.values()
            ):
                return None
            weights.append([int(values.get(column, 0)) for column in expected_columns])
        capacities = [int(model["row_upper"][row]) for row in model["rows"]]
        costs = [model["cost"].get(name, 0) for name in expected_columns]
        if any(value > 0 or not float(value).is_integer() for value in costs):
            return None
        return weights, capacities, [-int(value) for value in costs]

    try:
        source = instance_path.read_text()
    except OSError:
        return None
    lines = source.splitlines()
    if len(lines) < 8 or lines[:1] != ["Minimize"] or "Subject To" not in lines:
        return None
    subject = lines.index("Subject To")
    if subject < 2 or "Binary" not in lines:
        return None
    binary = lines.index("Binary")
    objective = " ".join(lines[1:subject])
    terms = list(re.finditer(r"-\s*(\d+)\s+x(\d+)", objective))
    if not terms or any(
        int(term.group(2)) != index for index, term in enumerate(terms)
    ):
        return None
    profits = [int(term.group(1)) for term in terms]
    weights: list[list[int]] = []
    capacities: list[int] = []
    constraint_text = " ".join(lines[subject + 1 : binary])
    row_matches = list(
        re.finditer(
            r"(?:^|\s)c(\d+):\s*(.*?)(?=\s+c\d+:|$)",
            constraint_text,
        )
    )
    if not row_matches:
        return None
    for dimension, row_match in enumerate(row_matches):
        match = re.fullmatch(r"\s*(.*?)\s*<=\s*(\d+)\s*", row_match.group(2))
        if match is None or int(row_match.group(1)) != dimension:
            return None
        terms = match.group(1).split(" + ")
        row: list[int] = []
        for item, term in enumerate(terms):
            normalized_term = " ".join(term.split())
            parsed = re.fullmatch(r"(\d+) x(\d+)", normalized_term)
            if parsed is None or int(parsed.group(2)) != item:
                return None
            row.append(int(parsed.group(1)))
        weights.append(row)
        capacities.append(int(match.group(2)))
    try:
        canonical = format_multidimensional_knapsack_lp(
            weights, capacities, profits=profits
        )
    except ValueError:
        return None

    def normalize_whitespace(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    return (
        (weights, capacities, profits)
        if normalize_whitespace(source) == normalize_whitespace(canonical)
        else None
    )


def _check_multidimensional_knapsack_solution(
    solution: dict,
    weights: list[list[int]],
    capacities: list[int],
    profits: list[int] | None = None,
) -> VerificationResult:
    try:
        values, reported_objective = _parse_mip_primal_solution(
            solution, [f"x{item}" for item in range(len(weights[0]))]
        )
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
    if profits is None:
        profits = [sum(column) for column in zip(*weights, strict=True)]
    objective = -sum(profit * assignment[item] for item, profit in enumerate(profits))
    if abs(reported_objective - objective) > _SCHEDULING_INTEGRALITY_TOLERANCE:
        return VerificationResult(feasible=False, detail="knapsack objective mismatch")
    return VerificationResult(
        feasible=True,
        objective=float(objective),
        detail="independent multidimensional knapsack verification passed",
    )


def expected_mip_variable_names(instance_path: Path) -> set[str]:
    """Return assignment keys from supported original MPS or canonical LP models."""
    if instance_path.suffix.lower() == ".mps":
        model = _read_mps_numeric_model(instance_path)
        if model is not None:
            return set(model["columns"])
    processing = _public_scheduling_processing_times(instance_path)
    if processing is not None:
        count = len(processing)
        return {f"s{index}" for index in range(count)} | {
            f"z{first}_{second}"
            for first in range(count)
            for second in range(first + 1, count)
        }
    knapsack = _public_multidimensional_knapsack_model(instance_path)
    if knapsack is not None:
        return {f"x{index}" for index in range(len(knapsack[0][0]))}
    raise ValueError("unsupported or noncanonical MIP instance")


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

    def _check_best_found_solution(
        self, instance_path: Path, solution: dict
    ) -> VerificationResult:
        """Check a BKS witness against source semantics without claiming optimality."""
        kind = self.record.problem_kind
        if kind == "linear_mip":
            checked = MIPFamily._verify_numeric_model_solution(instance_path, solution)
            if checked.feasible and _mps_metadata_payload(
                instance_path, "PITBENCH_SCHEDULING_DATA"
            ) is not None:
                return _mip_weighted_tardiness_source_check(
                    instance_path, solution, checked
                )
            if checked.feasible and _mps_metadata_payload(
                instance_path, "PITBENCH_GRAPH_DATA"
            ) is not None:
                return _mip_graph_coloring_source_check(
                    instance_path, solution, checked
                )
            return checked

        if kind == "job_shop_scheduling":
            try:
                lines = instance_path.read_text().splitlines()
                job_count, machine_count = map(int, lines[0].split())
                rows = [list(map(int, line.split())) for line in lines[1:]]
                if (
                    len(rows) != job_count
                    or any(len(row) != 2 * machine_count for row in rows)
                ):
                    raise ValueError("job-shop operation dimensions do not match")
                jobs = [
                    [
                        (row[index], row[index + 1])
                        for index in range(0, len(row), 2)
                    ]
                    for row in rows
                ]
                start_times = solution.get("start_times")
                if not isinstance(start_times, list):
                    raise ValueError("job-shop BKS has no operation start times")
                return _check_job_shop_start_times(start_times, jobs)
            except (OSError, IndexError, TypeError, ValueError) as error:
                return VerificationResult(
                    feasible=False, detail=f"invalid job-shop BKS: {error}"
                )

        try:
            instance = json.loads(instance_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            return VerificationResult(feasible=False, detail=f"invalid instance: {error}")

        if kind == "bin_packing":
            return _check_bin_packing_solution(
                solution,
                weights=instance.get("weights", []),
                capacity=instance.get("capacity"),
            )
        if kind == "graph_coloring":
            return VerificationResult.model_validate(
                check_graph_coloring(instance, solution)
            )
        if kind == "balanced_subset_sum":
            return _check_balanced_subset_sum_solution(instance, solution)
        if kind == "resource_constrained_project_scheduling":
            return VerificationResult.model_validate(
                check_resource_constrained_project_schedule(instance, solution)
            )
        if kind == "single_machine_scheduling":
            processing = instance.get("processing_times")
            if not isinstance(processing, list):
                return VerificationResult(
                    feasible=False, detail="single-machine BKS lacks processing times"
                )
            return _check_scheduling_solution(solution, processing)
        return VerificationResult(
            feasible=False,
            detail=f"best-found references are unsupported for {kind}",
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
        if self.record.target_kind == "best_found":
            checked = self._check_best_found_solution(instance_path, reference_solution)
            if (
                not checked.feasible
                or checked.objective is None
                or not math.isclose(
                    float(checked.objective),
                    float(self.record.optimal_objective),
                    rel_tol=0,
                    abs_tol=1e-6,
                )
            ):
                return self._oracle_error(
                    "best-found reference fails original-problem feasibility/objective checks"
                )
            return None
        if self.record.problem_kind == "linear_mip":
            checked = MIPFamily._verify_numeric_model_solution(
                instance_path, reference_solution
            )
            if checked.feasible and _mps_metadata_payload(
                instance_path, "PITBENCH_SCHEDULING_DATA"
            ) is not None:
                checked = _mip_weighted_tardiness_source_check(
                    instance_path, reference_solution, checked
                )
            if (
                not checked.feasible
                or checked.objective is None
                or not math.isclose(
                    checked.objective,
                    float(self.record.optimal_objective),
                    rel_tol=0,
                    abs_tol=1e-6,
                )
            ):
                return self._oracle_error(
                    "linear reference fails original-model feasibility/objective checks"
                )
            if self.record.target_kind == "best_found":
                if basis.get("kind") != "feasible_search_reference":
                    return self._oracle_error(
                        "best-found reference cannot claim a proven bound"
                    )
                return None
            if (
                basis.get("kind") != "independent_mip_search"
                or basis.get("reported_optimal") is not True
                or basis.get("reported_primal") != self.record.optimal_objective
                or basis.get("reported_dual") != self.record.optimal_objective
                or not isinstance(basis.get("solver_identity"), dict)
                or not re.fullmatch(
                    r"[0-9a-f]{64}", str(basis.get("reference_result_sha256", ""))
                )
            ):
                return self._oracle_error("inconsistent external MIP optimum record")
            # Trust the independently run solver's optimal return, as with other
            # external-search targets. This does not replay its branch-and-bound proof.
            return None
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

        if (
            self.record.problem_kind
            == "resource_constrained_project_scheduling"
        ):
            try:
                instance = json.loads(instance_path.read_text())
                checked = check_resource_constrained_project_schedule(
                    instance, reference_solution
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                return self._oracle_error(f"invalid RCPSP reference: {error}")
            if (
                not checked.get("feasible")
                or checked.get("objective") != self.record.optimal_objective
            ):
                return self._oracle_error("RCPSP reference violates original model")
            if self.record.target_kind == "best_found":
                if basis.get("kind") not in {
                    "feasible_search_reference",
                    "serial_feasible_schedule",
                    "cp_sat_search",
                }:
                    return self._oracle_error("invalid feasible RCPSP reference basis")
                return None
            if (
                basis.get("kind") != "cp_sat_search"
                or basis.get("solver_status") != "OPTIMAL"
                or not math.isclose(
                    float(basis.get("dual_bound", math.nan)),
                    float(self.record.optimal_objective),
                    rel_tol=0,
                    abs_tol=1e-6,
                )
            ):
                return self._oracle_error("RCPSP optimum lacks matching CP-SAT bound")
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
            model = _public_multidimensional_knapsack_model(instance_path)
            if model is None:
                return self._oracle_error("invalid knapsack instance")
            weights, capacities, profits = model
            multipliers = (
                basis.get("row_multipliers")
                if basis.get("kind") == "weighted_capacity_dual_bound"
                else [1] * len(weights)
            )
            if (
                not isinstance(multipliers, list)
                or len(multipliers) != len(weights)
                or any(type(value) is not int or value < 0 for value in multipliers)
            ):
                return self._oracle_error("invalid knapsack dual multipliers")
            expected_basis = {
                "kind": "capacity_dual_bound",
                "dimensions": len(weights),
                "items": len(weights[0]),
            }
            if basis.get("kind") == "weighted_capacity_dual_bound":
                expected_basis.update(
                    kind="weighted_capacity_dual_bound", row_multipliers=multipliers
                )
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
                or basis != expected_basis
                or profits
                != [
                    sum(
                        multiplier * value
                        for multiplier, value in zip(multipliers, column, strict=True)
                    )
                    for column in zip(*weights, strict=True)
                ]
                or self.record.optimal_objective
                != -sum(
                    multiplier * capacity
                    for multiplier, capacity in zip(
                        multipliers, capacities, strict=True
                    )
                )
            ):
                return self._oracle_error("invalid knapsack reference optimum")
            return None

        if self.record.problem_kind == "set_covering":
            model = _read_mps_numeric_model(instance_path)
            groups, partition_rows = basis.get("groups"), basis.get("partition_rows")
            if (
                model is None
                or basis.get("kind") != "disjoint_covering_rows"
                or not isinstance(groups, list)
                or not groups
                or any(
                    not isinstance(group, list)
                    or not group
                    or any(not isinstance(name, str) for name in group)
                    for group in groups
                )
                or not isinstance(partition_rows, list)
                or len(partition_rows) != len(groups)
                or any(not isinstance(row, str) for row in partition_rows)
                or len(set(partition_rows)) != len(partition_rows)
            ):
                return self._oracle_error("invalid set-covering partition certificate")
            flattened = [name for group in groups for name in group]
            if (
                len(flattened) != len(set(flattened))
                or set(flattened) != set(model["columns"])
                or model["integer_columns"] != set(flattened)
                or model["objective_sense"] != "minimize"
                or model["offset"] != 0
                or any(
                    model["column_lower"][name] != 0
                    or model["column_upper"][name] != 1
                    or model["cost"].get(name) != 1
                    for name in flattened
                )
                or self.record.optimal_objective != len(groups)
            ):
                return self._oracle_error(
                    "set-covering lower bound does not match the original objective or variables"
                )
            for row in model["rows"]:
                values = model["coefficients"].get(row, {})
                if (
                    not values
                    or set(values) - set(flattened)
                    or any(value != 1 for value in values.values())
                    or model["row_lower"][row] != 1
                    or model["row_upper"][row] != math.inf
                ):
                    return self._oracle_error(
                        "original model is not a unit-cost binary covering instance"
                    )
            for row, group in zip(partition_rows, groups, strict=True):
                if model["coefficients"].get(row) != {name: 1 for name in group}:
                    return self._oracle_error(
                        "set-covering partition row is absent from the original model"
                    )
            checked = MIPFamily._verify_numeric_model_solution(
                instance_path, reference_solution
            )
            if (
                not checked.feasible
                or checked.objective != self.record.optimal_objective
            ):
                return self._oracle_error(
                    "set-covering witness does not attain the independent lower bound"
                )
            return None

        if self.record.problem_kind in {
            "binary_clique_cover",
            "binary_clique_cover_fme",
        }:
            model = _read_mps_numeric_model(instance_path)
            groups = basis.get("groups")
            has_fme_paper_block = self.record.problem_kind == "binary_clique_cover_fme"
            fme_paper_block = basis.get("fme_paper_block")
            if (
                model is None
                or basis.get("kind")
                != (
                    "binary_clique_cover_fme"
                    if has_fme_paper_block
                    else "binary_clique_cover"
                )
                or has_fme_paper_block != (fme_paper_block is not None)
                or not isinstance(groups, list)
                or not groups
                or any(
                    not isinstance(group, list)
                    or not group
                    or any(not isinstance(name, str) for name in group)
                    for group in groups
                )
            ):
                return self._oracle_error("invalid binary clique-cover proof basis")
            columns = model["columns"]
            flattened = [name for group in groups for name in group]
            fme_extensions = basis.get("fme_extensions", [])
            if not isinstance(fme_extensions, list):
                return self._oracle_error("invalid FME extension proof list")
            binary_columns = set(flattened)
            auxiliary_columns = {
                extension.get("column")
                for extension in fme_extensions
                if isinstance(extension, dict)
            }
            if len(auxiliary_columns) != len(fme_extensions) or any(
                not isinstance(extension, dict)
                or not isinstance(extension.get("column"), str)
                or not extension.get("column")
                or not isinstance(extension.get("positive_rows"), list)
                or not isinstance(extension.get("negative_rows"), list)
                or len(extension["positive_rows"]) < 2
                or len(extension["negative_rows"]) < 2
                for extension in fme_extensions
            ):
                return self._oracle_error("invalid FME extension proof entries")
            if (
                len(flattened) != len(set(flattened))
                or binary_columns.intersection(auxiliary_columns)
                or binary_columns.union(auxiliary_columns) != set(columns)
                or binary_columns != set(model["integer_columns"])
                or any(
                    model["column_lower"][name] != 0 or model["column_upper"][name] != 1
                    for name in binary_columns
                )
                or any(model["cost"].get(name, 0.0) > 1e-9 for name in columns)
            ):
                return self._oracle_error(
                    "binary clique-cover basis does not partition binary columns"
                )
            extension_rows: set[str] = set()
            for extension in fme_extensions:
                column = extension["column"]
                if (
                    model["column_lower"].get(column) != -math.inf
                    or model["column_upper"].get(column) != math.inf
                    or not math.isclose(
                        model["cost"].get(column, 0.0),
                        0.0,
                        rel_tol=0,
                        abs_tol=1e-12,
                    )
                ):
                    return self._oracle_error(
                        "invalid FME auxiliary-column bounds or rows"
                    )
                positive_support: set[str] = set()
                negative_support: set[str] = set()
                for direction, entries, sign, support in (
                    ("positive", extension["positive_rows"], 1.0, positive_support),
                    ("negative", extension["negative_rows"], -1.0, negative_support),
                ):
                    for entry in entries:
                        if (
                            not isinstance(entry, dict)
                            or not isinstance(entry.get("row"), str)
                            or not entry["row"]
                            or entry["row"] in extension_rows
                            or entry["row"] not in model["coefficients"]
                            or not isinstance(entry.get("binary_coefficients"), dict)
                            or not entry["binary_coefficients"]
                            or type(entry.get("rhs")) not in {int, float}
                            or not math.isfinite(entry["rhs"])
                        ):
                            return self._oracle_error(
                                f"invalid FME {direction}-row declaration"
                            )
                        row_name = entry["row"]
                        binary_coefficients = entry["binary_coefficients"]
                        if any(
                            name not in binary_columns
                            or type(value) not in {int, float}
                            or not math.isfinite(value)
                            or value == 0
                            for name, value in binary_coefficients.items()
                        ):
                            return self._oracle_error(
                                "FME row has an invalid binary coefficient"
                            )
                        expected_coefficients = {column: sign}
                        expected_coefficients.update(binary_coefficients)
                        # The zero-RHS extension can restrict other binary
                        # assignments, but the trusted witness sets every
                        # supported binary to zero. Combined with the clique
                        # lower bound, that witness certifies the same optimum.
                        if not math.isclose(entry["rhs"], 0, rel_tol=0, abs_tol=1e-9):
                            return self._oracle_error(
                                "FME extension row has a nonzero RHS"
                            )
                        actual_coefficients = {
                            name: value
                            for name, value in model["coefficients"][row_name].items()
                            if abs(value) > 1e-12
                        }
                        if (
                            actual_coefficients != expected_coefficients
                            or not math.isinf(
                                model["row_lower"].get(row_name, math.nan)
                            )
                            or model["row_lower"].get(row_name, math.nan) > 0
                            or not math.isclose(
                                model["row_upper"].get(row_name, math.nan),
                                0,
                                rel_tol=0,
                                abs_tol=1e-9,
                            )
                        ):
                            return self._oracle_error(
                                "FME extension row differs from its proof record"
                            )
                        support.update(binary_coefficients)
                        extension_rows.add(row_name)
                if len(positive_support.intersection(negative_support)) < 5:
                    return self._oracle_error(
                        "FME extension lacks a nonzero-reduction shared support"
                    )
                reference_values = reference_solution.get("values", {})
                if any(
                    reference_values.get(name) != 0
                    for name in positive_support.union(negative_support)
                ):
                    return self._oracle_error(
                        "FME extension does not preserve the clique-cover witness"
                    )
            for row_name, coefficients in model["coefficients"].items():
                if row_name in extension_rows:
                    continue
                if set(coefficients).intersection(auxiliary_columns):
                    return self._oracle_error(
                        "FME auxiliary column occurs outside its extension rows"
                    )
            clique_rows_by_support: dict[
                frozenset[str], list[tuple[dict[str, float], float]]
            ] = {}
            for row_name, coefficients in model["coefficients"].items():
                if row_name in extension_rows:
                    continue
                row_upper = model["row_upper"][row_name]
                if not math.isfinite(row_upper) or any(
                    value < -1e-9 for value in coefficients.values()
                ):
                    continue
                support = frozenset(
                    name for name, value in coefficients.items() if abs(value) > 1e-12
                )
                clique_rows_by_support.setdefault(support, []).append(
                    (coefficients, row_upper)
                )
            for group in groups:
                witnesses = clique_rows_by_support.get(frozenset(group), [])
                if not any(
                    all(
                        coefficients.get(first, 0.0) + coefficients.get(second, 0.0)
                        > row_upper + 1e-9
                        for first_index, first in enumerate(group)
                        for second in group[first_index + 1 :]
                    )
                    for coefficients, row_upper in witnesses
                ):
                    return self._oracle_error(
                        "clique-cover group has an unproved variable pair"
                    )
            upper_bound = model["offset"] - sum(
                max([0.0] + [max(-model["cost"].get(name, 0.0), 0.0) for name in group])
                for group in groups
            )
            try:
                _, reference_objective = _parse_mip_primal_solution(
                    reference_solution, model["columns"]
                )
            except (InvalidOperation, TypeError, ValueError) as error:
                return self._oracle_error(f"invalid clique-cover reference: {error}")
            reference_check = MIPFamily._verify_numeric_model_solution(
                instance_path, reference_solution
            )
            if (
                not math.isclose(
                    upper_bound,
                    self.record.optimal_objective,
                    rel_tol=0,
                    abs_tol=1e-6,
                )
                or not reference_check.feasible
                or reference_check.objective is None
                or not math.isclose(
                    reference_objective,
                    self.record.optimal_objective,
                    rel_tol=0,
                    abs_tol=1e-6,
                )
                or not math.isclose(
                    reference_check.objective,
                    self.record.optimal_objective,
                    rel_tol=0,
                    abs_tol=1e-6,
                )
            ):
                return self._oracle_error(
                    "binary clique-cover bound and feasible reference differ"
                )
            return None

        if self.record.problem_kind == "balanced_subset_sum":
            try:
                instance = json.loads(instance_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                return self._oracle_error(f"invalid subset-sum instance: {error}")
            weights = instance.get("weights")
            cardinality = instance.get("cardinality")
            target = instance.get("target")
            checked = _check_balanced_subset_sum_solution(instance, reference_solution)
            if not checked.feasible:
                return self._oracle_error(checked.detail)
            layers = [1] + [0] * cardinality
            for position, weight in enumerate(weights):
                for count in range(min(cardinality, position + 1), 0, -1):
                    layers[count] |= layers[count - 1] << weight
            total = sum(weights)
            reachable = layers[cardinality]
            optimum = min(
                abs(value - target)
                for value in range(total + 1)
                if (reachable >> value) & 1
            )
            expected_basis = {
                "kind": "fixed_cardinality_subset_sum_dp",
                "total_weight": total,
                "cardinality": cardinality,
                "target": target,
                "reachable_bitset_sha256": hashlib.sha256(
                    reachable.to_bytes((total + 8) // 8, "little")
                ).hexdigest(),
            }
            if (
                basis != expected_basis
                or checked.objective != optimum
                or self.record.optimal_objective != optimum
            ):
                return self._oracle_error("invalid subset-sum DP optimum record")
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
        if self.record.target_kind == "best_found":
            return self._check_best_found_solution(instance_path, solution)
        if self.record.problem_kind == "single_machine_scheduling":
            return _check_scheduling_solution(
                solution,
                self.record.optimality_basis["processing_times"],
            )
        if (
            self.record.problem_kind
            == "resource_constrained_project_scheduling"
        ):
            try:
                return VerificationResult.model_validate(
                    check_resource_constrained_project_schedule(
                        json.loads(instance_path.read_text()), solution
                    )
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                return VerificationResult(
                    feasible=False,
                    detail=f"invalid RCPSP solution input: {error}",
                )
        if self.record.problem_kind == "job_shop_scheduling":
            return _check_job_shop_solution(solution, self.record.optimality_basis)
        if self.record.problem_kind == "multidimensional_knapsack":
            model = _public_multidimensional_knapsack_model(instance_path)
            if model is None:
                return self._oracle_error("invalid knapsack instance")
            return _check_multidimensional_knapsack_solution(solution, *model)
        if self.record.problem_kind in {
            "binary_clique_cover",
            "set_covering",
            "linear_mip",
        }:
            return MIPFamily._verify_numeric_model_solution(instance_path, solution)
        instance = json.loads(instance_path.read_text())
        if self.record.problem_kind == "graph_coloring":
            return VerificationResult.model_validate(
                check_graph_coloring(instance, solution)
            )
        if self.record.problem_kind == "balanced_subset_sum":
            return _check_balanced_subset_sum_solution(instance, solution)
        return _check_bin_packing_solution(
            solution,
            weights=instance["weights"],
            capacity=instance["capacity"],
        )


class MIPFamily(ExternalVerifierFamily):
    name = "mip"
    fixture_reports_search_nodes = True

    @staticmethod
    def _verify_numeric_model_solution(
        instance_path: Path, solution: dict
    ) -> VerificationResult:
        try:
            model = _read_mps_numeric_model(instance_path)
            if model is None:
                raise ValueError("unsupported or invalid MPS model")
            variable_names = model["columns"]
            values, objective = _parse_mip_primal_solution(solution, variable_names)
            if set(values) != set(variable_names):
                raise ValueError("MPS primal solution variable set mismatch")
            tolerance = OBJECTIVE_ABSOLUTE_TOLERANCE
            bound_violation = max(
                [0.0]
                + [
                    model["column_lower"][name] - float(values[name])
                    for name in variable_names
                ]
                + [
                    float(values[name]) - model["column_upper"][name]
                    for name in variable_names
                ]
            )
            integer_violation = max(
                [0.0]
                + [
                    abs(float(values[name]) - round(float(values[name])))
                    for name in model["integer_columns"]
                ]
            )
            row_violation = 0.0
            for row_name in model["rows"]:
                activity = math.fsum(
                    coefficient * float(values[column])
                    for column, coefficient in model["coefficients"][row_name].items()
                )
                row_violation = max(
                    row_violation,
                    model["row_lower"][row_name] - activity,
                    activity - model["row_upper"][row_name],
                )
            checked_objective = (
                math.fsum(
                    coefficient * float(values[column])
                    for column, coefficient in model["cost"].items()
                )
                + model["offset"]
            )
        except (KeyError, OSError, TypeError, ValueError, InvalidOperation) as error:
            return VerificationResult(feasible=False, detail=str(error))
        feasible = max(bound_violation, integer_violation, row_violation) <= tolerance
        detail = {
            "max_bound_violation": bound_violation,
            "max_integrality_violation": integer_violation,
            "max_row_violation": row_violation,
            "objective": checked_objective,
            "tolerance": tolerance,
        }
        if not feasible:
            return VerificationResult(
                feasible=False,
                objective=checked_objective,
                detail=json.dumps(detail, sort_keys=True),
            )
        if not objectives_equal(float(objective), checked_objective):
            return VerificationResult(
                feasible=False,
                objective=checked_objective,
                detail="reported MIP objective differs from independent MPS arithmetic",
            )
        return VerificationResult(
            feasible=True,
            objective=float(checked_objective),
            detail="independent sparse MPS feasibility and objective check passed",
        )

    def verify(self, instance_path: Path, solution_path: Path) -> VerificationResult:
        if self.verifier is not None:
            return super().verify(instance_path, solution_path)
        if instance_path.suffix.lower() == ".mps":
            try:
                solution = json.loads(solution_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                return VerificationResult(feasible=False, detail=str(error))
            if not isinstance(solution, dict):
                return VerificationResult(
                    feasible=False, detail="invalid MIP solution container"
                )
            result = self._verify_numeric_model_solution(instance_path, solution)
            if not result.feasible:
                return result
            try:
                graph = _mps_metadata_payload(instance_path, "PITBENCH_GRAPH_DATA")
                if graph is not None:
                    return _mip_graph_coloring_source_check(
                        instance_path, solution, result
                    )
                scheduling = _mps_metadata_payload(
                    instance_path, "PITBENCH_SCHEDULING_DATA"
                )
                if scheduling is not None:
                    return _mip_weighted_tardiness_source_check(
                        instance_path, solution, result
                    )
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                return VerificationResult(
                    feasible=False, detail=f"invalid MIP family metadata: {error}"
                )
            return result
        processing_times = _public_scheduling_processing_times(instance_path)
        if processing_times is not None:
            try:
                solution = json.loads(solution_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                return VerificationResult(feasible=False, detail=str(error))
            return _check_scheduling_solution(solution, processing_times)
        knapsack = _public_multidimensional_knapsack_model(instance_path)
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
        try:
            objective = (
                math.fsum(float(a) * float(b) for a, b in zip(model["cost"], values))
                + model["offset"]
            )
        except OverflowError:
            return {
                "feasible": False,
                "infrastructure_error": True,
                "reason": "independent objective arithmetic overflow",
            }
        if not math.isfinite(objective):
            return {
                "feasible": False,
                "infrastructure_error": True,
                "reason": "nonfinite independent objective arithmetic",
            }
        return {
            "feasible": max(bound_error, row_error, integer_error) <= tolerance,
            "tolerance": tolerance,
            "max_bound_violation": bound_error,
            "max_row_violation": row_error,
            "max_integrality_violation": integer_error,
            "objective": objective,
        }

    @staticmethod
    def certified_relaxation_bound(model: dict, row_multipliers) -> float | None:
        """Independently evaluate a weak-duality certificate, including bounds.

        Exact rational arithmetic on the stored numeric model avoids accepting a
        solver-reported search bound. A certificate for the LP relaxation is also
        valid for its integer restriction. Missing finite support yields no bound.
        """
        from fractions import Fraction

        import numpy as np

        y = np.asarray(row_multipliers, dtype=float)
        if y.shape != model["row_lower"].shape or not np.isfinite(y).all():
            return None

        def q(value):
            return Fraction.from_float(float(value))

        sign = 1 if model["sense"] in {1, "minimize"} else -1
        multipliers = [q(value) * sign for value in y]
        reduced = [q(value) * sign for value in model["cost"]]
        total = q(model["offset"]) * sign
        matrix = model["matrix"].tocoo()
        for i, j, value in zip(matrix.row, matrix.col, matrix.data, strict=True):
            reduced[j] -= multipliers[i] * q(value)
        for coefficients, lower, upper in (
            (multipliers, model["row_lower"], model["row_upper"]),
            (reduced, model["col_lower"], model["col_upper"]),
        ):
            for coefficient, low, high in zip(coefficients, lower, upper, strict=True):
                if not coefficient:
                    continue
                bound = low if coefficient > 0 else high
                if not math.isfinite(float(bound)):
                    return None
                total += coefficient * q(bound)
        try:
            lower_bound = math.nextafter(float(total), -math.inf)
        except OverflowError:
            return None
        if not math.isfinite(lower_bound):
            return None
        return lower_bound * sign


class IntegerBoundaryFamily(ProblemFamilyPlugin):
    """Verify the small, integral boundary models against evaluator-owned coefficients.

    These test cases use integer coefficients and assignments, so checks are exact.
    This is not a replacement for the verifier of arbitrary floating-point MIPs.
    """

    name = "integer_boundary"

    def verify(self, instance_path: Path, solution_path: Path) -> VerificationResult:
        model = json.loads(instance_path.with_suffix(".model.json").read_text())
        solution = json.loads(solution_path.read_text())
        values, objective = _parse_mip_primal_solution(solution)
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
            if instance.get("problem_kind") == "balanced_subset_sum":
                return _check_balanced_subset_sum_solution(instance, solution)
            if (
                instance.get("problem_kind")
                == "resource_constrained_project_scheduling"
            ):
                return VerificationResult.model_validate(
                    check_resource_constrained_project_schedule(instance, solution)
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
