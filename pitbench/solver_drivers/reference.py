"""Offline reference search on generated grids; separate from scored solver runs.

Small instances can be exhausted. A feasible incumbent after a time limit is
only a best-found reference, never a published BKS or an optimality claim.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pitbench.instances.generate import (
    _job_shop_operations,
    balanced_subset_sum_data,
    binary_clique_cover_data,
    binary_clique_cover_fme_extensions,
    capacity_partition_routes,
    first_fit_decreasing_packing,
    multidimensional_knapsack_data,
    multidimensional_knapsack_dual_weights,
)
from pitbench.problem_families.public_verification import (
    check_bin_packing,
    check_cvrp,
    check_graph_coloring,
    check_resource_constrained_project_schedule,
    check_routing_variant,
    edge_cost,
)
from pitbench.problem_families.verification import (
    CPFamily,
    CVRPFamily,
    MIPFamily,
    VerificationResult,
    _check_balanced_subset_sum_solution,
    _check_job_shop_start_times,
    _check_multidimensional_knapsack_solution,
    _check_scheduling_solution,
    _mip_weighted_tardiness_source_check,
    _mps_metadata_payload,
    _public_multidimensional_knapsack_model,
    _read_mps_numeric_model,
    _public_scheduling_processing_times,
)


def generated_reference_instances(manifest_path: Path) -> list[dict]:
    """Read grid/suite manifests into the common offline-reference input contract."""
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("kind") == "generalization_population_construction":
        return [
            entry
            for split in manifest["splits"].values()
            for entry in generated_reference_instances(Path(split["manifest"]))
        ]
    if isinstance(manifest.get("instances"), list):
        return [
            {
                **entry,
                "path": str(Path(entry["path"]).resolve()),
                "role": manifest.get("role"),
            }
            for entry in manifest["instances"]
        ]
    if "panels" in manifest:
        return [
            entry
            for panel in manifest["panels"].values()
            for entry in generated_reference_instances(
                manifest_path.parent / panel["manifest"]
            )
        ]
    return [
        {
            "id": f"{manifest_path.parent.name}/{cell['condition_id']}/{entry['id']}",
            "path": str((manifest_path.parent / entry["path"]).resolve()),
            "sha256": entry["sha256"],
            "generator": cell["generator"],
            "replicate_index": entry["replicate_index"],
        }
        for cell in manifest["conditions"]
        for entry in cell["instances"]
    ]


def verify_reference_file(instance: Path, solution: Path, kind: str) -> dict:
    if kind in {"euclidean_cvrp", "heterogeneous_fleet_vrp", "multi_depot_vrp", "open_vrp"}:
        checked = CVRPFamily().verify(instance, solution)
    elif kind in {"bin_packing", "graph_coloring", "balanced_subset_sum"}:
        checked = CPFamily().verify(instance, solution)
    elif kind == "job_shop_scheduling":
        checked = _check_job_shop_start_times(
            json.loads(solution.read_text())["start_times"],
            _job_shop_operations(instance),
        )
    elif kind == "resource_constrained_project_scheduling":
        checked = check_resource_constrained_project_schedule(
            json.loads(instance.read_text()), json.loads(solution.read_text())
        )
    else:
        checked = MIPFamily().verify(instance, solution)
    return checked.model_dump() if hasattr(checked, "model_dump") else checked


@dataclass
class ReferenceResult:
    solution: dict | None
    proven: bool
    basis: dict
    search_status: str = "completed"


class SearchBudget:
    def __init__(self, seconds: float):
        self.deadline = time.monotonic() + seconds
        self.nodes = 0

    def tick(self) -> None:
        self.nodes += 1
        if time.monotonic() >= self.deadline:
            raise TimeoutError("offline reference search time limit")


def _bin_packing(instance: dict, budget: SearchBudget) -> ReferenceResult:
    weights, capacity = instance["weights"], instance["capacity"]
    best = first_fit_decreasing_packing(weights, capacity)
    lower = max(
        math.ceil(sum(weights) / capacity), sum(2 * w > capacity for w in weights)
    )
    order = sorted(range(len(weights)), key=lambda i: (-weights[i], i))
    bins: list[list[int]] = []
    loads: list[int] = []

    def search(position: int) -> None:
        nonlocal best
        budget.tick()
        if len(best) == lower or len(bins) >= len(best):
            return
        if position == len(order):
            best = [items.copy() for items in bins]
            return
        item = order[position]
        seen = set()
        for b, load in enumerate(loads):
            if load in seen or load + weights[item] > capacity:
                continue
            seen.add(load)
            bins[b].append(item)
            loads[b] += weights[item]
            search(position + 1)
            loads[b] -= weights[item]
            bins[b].pop()
        bins.append([item])
        loads.append(weights[item])
        search(position + 1)
        bins.pop()
        loads.pop()

    status = "completed"
    try:
        search(0)
    except TimeoutError:
        status = "time_limit"
    return ReferenceResult(
        {"bins": best},
        status == "completed" or len(best) == lower,
        {
            "kind": "bin_packing_branch_and_bound",
            "lower_bound": lower,
            "nodes": budget.nodes,
        },
        status,
    )


def _graph_coloring(instance: dict, budget: SearchBudget) -> ReferenceResult:
    n = instance["vertex_count"]
    adjacent = [set() for _ in range(n)]
    endpoints = instance["edge_endpoints"]
    for a, b in zip(endpoints[::2], endpoints[1::2], strict=True):
        adjacent[a].add(b)
        adjacent[b].add(a)
    colors = [-1] * n

    def vertex() -> int:
        return max(
            (v for v in range(n) if colors[v] < 0),
            key=lambda v: (
                len({colors[w] for w in adjacent[v] if colors[w] >= 0}),
                len(adjacent[v]),
                -v,
            ),
        )

    for _ in range(n):
        v = vertex()
        forbidden = {colors[w] for w in adjacent[v]}
        colors[v] = next(c for c in range(n) if c not in forbidden)
    best = colors.copy()
    upper = max(best) + 1
    clique: list[int] = []
    for v in sorted(range(n), key=lambda v: -len(adjacent[v])):
        if all(v in adjacent[w] for w in clique):
            clique.append(v)
    lower = len(clique)
    colors = [-1] * n

    def search(remaining: int, used: int) -> None:
        nonlocal best, upper
        budget.tick()
        if upper == lower or used >= upper:
            return
        if not remaining:
            upper, best = used, colors.copy()
            return
        v = vertex()
        forbidden = {colors[w] for w in adjacent[v]}
        for c in range(min(used + 1, upper - 1)):
            if c not in forbidden:
                colors[v] = c
                search(remaining - 1, max(used, c + 1))
                colors[v] = -1

    status = "completed"
    try:
        search(n, 0)
    except TimeoutError:
        status = "time_limit"
    return ReferenceResult(
        {"colors": best},
        status == "completed" or upper == lower,
        {
            "kind": "dsatur_exhaustive_coloring",
            "clique_lower_bound": clique,
            "nodes": budget.nodes,
        },
        status,
    )


def _knapsack(
    path: Path, config: dict, index: int, budget: SearchBudget
) -> ReferenceResult:
    parsed = _public_multidimensional_knapsack_model(path)
    if parsed is None:
        raise ValueError("unsupported knapsack LP")
    weights, capacities, profits = parsed
    n, dimensions = len(weights[0]), len(weights)
    try:
        dual = multidimensional_knapsack_dual_weights(config, dimensions)
    except ValueError:
        dual = [1] * dimensions
    capacity_dual = profits == [
        sum(dual[d] * weights[d][i] for d in range(dimensions)) for i in range(n)
    ]
    order = sorted(range(n), key=lambda i: -profits[i])
    remaining = [0] * (n + 1)
    for pos in range(n - 1, -1, -1):
        remaining[pos] = remaining[pos + 1] + profits[order[pos]]
    best: list[int] = []
    used = [0] * dimensions
    for i in order:
        if all(used[d] + weights[d][i] <= capacities[d] for d in range(dimensions)):
            best.append(i)
            for d in range(dimensions):
                used[d] += weights[d][i]
    best_value = sum(profits[i] for i in best)
    if config.get("capacity_mode", "planted") == "planted":
        generated_weights, generated_caps, planted = multidimensional_knapsack_data(
            config, index
        )
        if (generated_weights, generated_caps) != (weights, capacities):
            raise ValueError("knapsack provenance does not match input")
        planted_value = sum(profits[i] for i in planted)
        if planted_value > best_value:
            best, best_value = planted, planted_value
    chosen: list[int] = []
    used = [0] * dimensions
    bound = (
        sum(dual[d] * capacities[d] for d in range(dimensions))
        if capacity_dual
        else sum(profits)
    )

    def search(pos: int, value: int) -> None:
        nonlocal best, best_value
        budget.tick()
        if value > best_value:
            best, best_value = chosen.copy(), value
        if (
            pos == n
            or best_value == bound
            or value
            + (
                min(
                    remaining[pos],
                    sum(dual[d] * (capacities[d] - used[d]) for d in range(dimensions)),
                )
                if capacity_dual
                else remaining[pos]
            )
            <= best_value
        ):
            return
        i = order[pos]
        if all(used[d] + weights[d][i] <= capacities[d] for d in range(dimensions)):
            chosen.append(i)
            for d in range(dimensions):
                used[d] += weights[d][i]
            search(pos + 1, value + profits[i])
            for d in range(dimensions):
                used[d] -= weights[d][i]
            chosen.pop()
        search(pos + 1, value)

    status = "completed"
    try:
        search(0, 0)
    except TimeoutError:
        status = "time_limit"
    solution = {
        "values": {f"x{i}": int(i in best) for i in range(n)},
        "objective": -best_value,
    }
    return ReferenceResult(
        solution,
        status == "completed" or best_value == bound,
        {
            "kind": "knapsack_branch_and_bound",
            "objective_lower_bound": -bound,
            "nodes": budget.nodes,
        },
        status,
    )


def _job_shop(jobs: list[list[list[int]]], budget: SearchBudget) -> ReferenceResult:
    lengths = [len(job) for job in jobs]
    offsets = [sum(lengths[:j]) for j in range(len(jobs))]
    count = sum(lengths)
    machine_count = 1 + max(op[0] for job in jobs for op in job)
    lower = max(
        max(sum(duration for _, duration in job) for job in jobs),
        max(
            sum(duration for job in jobs for machine, duration in job if machine == m)
            for m in range(machine_count)
        ),
    )
    positions = [0] * len(jobs)
    job_ends = [0] * len(jobs)
    machine_ends = [0] * machine_count
    starts = [0] * count
    # Serial feasible incumbent, independent of search completion.
    serial = 0
    for j, job in enumerate(jobs):
        for op, (_, duration) in enumerate(job):
            starts[offsets[j] + op] = serial
            serial += duration
    best, upper = starts.copy(), serial

    def search(remaining: int) -> None:
        nonlocal best, upper
        budget.tick()
        if upper == lower:
            return
        if remaining == 0:
            value = max(job_ends)
            if value < upper:
                best, upper = starts.copy(), value
            return
        for j, job in enumerate(jobs):
            op = positions[j]
            if op == len(job):
                continue
            machine, duration = job[op]
            start = max(job_ends[j], machine_ends[machine])
            if start + sum(d for _, d in job[op:]) >= upper:
                continue
            previous_job, previous_machine = job_ends[j], machine_ends[machine]
            starts[offsets[j] + op] = start
            job_ends[j] = machine_ends[machine] = start + duration
            positions[j] += 1
            search(remaining - 1)
            positions[j] -= 1
            job_ends[j], machine_ends[machine] = previous_job, previous_machine

    status = "completed"
    try:
        search(count)
    except TimeoutError:
        status = "time_limit"
    return ReferenceResult(
        {"start_times": best},
        status == "completed" or upper == lower,
        {
            "kind": "exhaustive_job_dispatch",
            "lower_bound": lower,
            "nodes": budget.nodes,
        },
        status,
    )


def _weighted_tardiness_schedule_reference(path: Path) -> ReferenceResult:
    from pitbench.problem_families.verification import _mps_metadata_payload

    data = _mps_metadata_payload(path, "PITBENCH_SCHEDULING_DATA")
    if data is None:
        raise ValueError("weighted-tardiness MPS has no original scheduling data")
    processing = data["processing_times"]
    releases = data["release_dates"]
    dues = data["due_dates"]
    weights = data["weights"]
    setup = data["setup_times"]
    count = len(processing)
    order = sorted(
        range(count),
        key=lambda job: (
            dues[job],
            processing[job] / weights[job],
            job,
        ),
    )
    starts = [0] * count
    finish, previous = 0, None
    for job in order:
        ready = releases[job] if previous is None else finish + setup[previous][job]
        starts[job] = max(releases[job], ready)
        finish = starts[job] + processing[job]
        previous = job
    tardiness = [
        max(0, starts[job] + processing[job] - dues[job])
        for job in range(count)
    ]
    values = {f"s{job}": starts[job] for job in range(count)}
    values.update({f"t{job}": tardiness[job] for job in range(count)})
    values.update(
        {
            f"y{first}_{second}": int(starts[first] < starts[second])
            for first in range(count)
            for second in range(first + 1, count)
        }
    )
    objective = sum(
        weight * late for weight, late in zip(weights, tardiness, strict=True)
    )
    return ReferenceResult(
        {"values": values, "objective": objective},
        False,
        {"kind": "deadline_order_feasible_schedule", "order": order},
        "feasible_reference",
    )


def _constructive_mps_reference(
    path: Path, config: dict[str, Any], index: int
) -> ReferenceResult:
    """Construct a feasible MPS witness when no analytic proof is available."""
    model = _read_mps_numeric_model(path)
    if model is None:
        raise ValueError("constructive MPS reference cannot parse input")
    kind = config["kind"]
    values = dict.fromkeys(model["columns"], 0)
    if kind == "multidimensional_knapsack_mps":
        pass  # The empty selection is source-feasible.
    elif kind == "random_set_covering_mip":
        values = dict.fromkeys(model["columns"], 1)
    elif kind == "facility_location_mip":
        facilities = sorted(
            int(name[1:]) for name in model["columns"] if name.startswith("y")
        )
        customers = sorted(
            {
                int(name.rsplit("_", 1)[1])
                for name in model["columns"]
                if name.startswith("x") and "_" in name
            }
        )
        remaining = {}
        for facility in facilities:
            row = model["coefficients"].get(f"c{facility}", {})
            remaining[facility] = int(-row[f"y{facility}"])
            values[f"y{facility}"] = 1
        for customer in customers:
            demand = int(model["row_lower"][f"d{customer}"])
            for facility in facilities:
                amount = min(demand, remaining[facility])
                if amount:
                    values[f"x{facility}_{customer}"] = amount
                    remaining[facility] -= amount
                    demand -= amount
                if demand == 0:
                    break
            if demand:
                raise ValueError("facility-location witness cannot satisfy demand")
    elif kind == "graph_coloring_mip":
        from pitbench.instances.generate import graph_coloring_data

        graph, planted, _ = graph_coloring_data(config, index)
        source = _mps_metadata_payload(path, "PITBENCH_GRAPH_DATA")
        if (
            source is None
            or graph["vertex_count"] != source["vertex_count"]
            or graph["edge_endpoints"] != source["edge_endpoints"]
        ):
            raise ValueError("graph-coloring source graph differs from MPS metadata")
        for vertex, color in enumerate(planted):
            values[f"x{vertex}_{color}"] = 1
            values[f"y{color}"] = 1
    else:
        raise ValueError(f"no constructive MPS witness for {kind}")
    objective = sum(
        coefficient * values[column]
        for column, coefficient in model["cost"].items()
    )
    return ReferenceResult(
        {"values": values, "objective": objective},
        False,
        {"kind": f"constructive_{kind}_feasible_witness"},
        "feasible_reference",
    )


def _cvrp(
    instance: dict,
    budget: SearchBudget,
    max_customers: int,
    max_exact_time_window_customers: int = 0,
) -> ReferenceResult:
    n = len(instance["demands"]) - 1
    if instance.get("depot", 0) != 0:
        raise ValueError("reference CVRP expects depot zero")
    if "time_windows" in instance:
        solution = {"routes": capacity_partition_routes(instance)}
        if not check_cvrp(instance, solution)["feasible"]:
            raise ValueError(
                "native reference cannot construct a feasible time-window witness"
            )
        if n <= max_exact_time_window_customers:
            best = solution
            best_cost = check_cvrp(instance, best)["objective"]
            completed = True
            try:
                for order in itertools.permutations(range(1, n + 1)):
                    for mask in range(1 << max(0, n - 1)):
                        budget.tick()
                        routes, current = [], []
                        for i, node in enumerate(order):
                            current.append(node)
                            if i == n - 1 or mask & (1 << i):
                                routes.append(current)
                                current = []
                        checked = check_cvrp(instance, {"routes": routes})
                        if checked["feasible"] and checked["objective"] < best_cost:
                            best, best_cost = {"routes": routes}, checked["objective"]
            except TimeoutError:
                completed = False
            return ReferenceResult(
                best,
                completed,
                {
                    "kind": "exhaustive_time_window_route_partitions",
                    "enumerated_candidates": budget.nodes,
                },
                "completed" if completed else "time_limit",
            )
        return ReferenceResult(
            solution,
            False,
            {"kind": "time_window_feasibility_witness"},
            "variant_witness_only",
        )
    if "num_vehicles" in instance:
        solution = {"routes": capacity_partition_routes(instance)}
        checked = check_cvrp(instance, solution)
        if not checked["feasible"]:
            raise ValueError(
                "native reference cannot construct a witness within the vehicle limit"
            )
        return ReferenceResult(
            solution,
            False,
            {"kind": "capacity_partition_with_vehicle_limit"},
            "feasible_reference_only",
        )
    solution = {"routes": [[i] for i in range(1, n + 1)]}
    if n > max_customers:
        return ReferenceResult(
            solution, False, {"kind": "singleton_routes"}, "exact_size_limit"
        )
    if instance.get("distance_metric") != "EUC_2D":
        raise ValueError("exact CVRP reference requires integer EUC_2D costs")
    size = 1 << n
    cost = [
        [int(edge_cost(instance["coordinates"], a, b, "EUC_2D")) for b in range(n + 1)]
        for a in range(n + 1)
    ]
    loads = [0] * size
    paths: dict[tuple[int, int], tuple[int, int]] = {}
    route_cost = [math.inf] * size
    route_end = [-1] * size
    status = "completed"
    try:
        for mask in range(1, size):
            budget.tick()
            bit = mask & -mask
            loads[mask] = loads[mask ^ bit] + instance["demands"][bit.bit_length()]
            if loads[mask] > instance["capacity"]:
                continue
            for end in range(n):
                if not mask & (1 << end):
                    continue
                previous = mask ^ (1 << end)
                if not previous:
                    value, parent = cost[0][end + 1], -1
                else:
                    value, parent = min(
                        (paths[previous, p][0] + cost[p + 1][end + 1], p)
                        for p in range(n)
                        if previous & (1 << p)
                    )
                paths[mask, end] = value, parent
                if value + cost[end + 1][0] < route_cost[mask]:
                    route_cost[mask], route_end[mask] = value + cost[end + 1][0], end
        values = [math.inf] * size
        selected = [0] * size
        values[0] = 0
        for mask in range(1, size):
            budget.tick()
            first = mask & -mask
            sub = mask
            while sub:
                budget.tick()
                if sub & first and route_cost[sub] + values[mask ^ sub] < values[mask]:
                    values[mask], selected[mask] = (
                        route_cost[sub] + values[mask ^ sub],
                        sub,
                    )
                sub = (sub - 1) & mask
        routes = []
        mask = size - 1
        while mask:
            sub = selected[mask]
            end, route, remaining = route_end[sub], [], sub
            while remaining:
                route.append(end + 1)
                parent = paths[remaining, end][1]
                remaining ^= 1 << end
                end = parent
            routes.append(route[::-1])
            mask ^= sub
        solution = {"routes": routes}
    except TimeoutError:
        status = "time_limit"
    return ReferenceResult(
        solution,
        status == "completed",
        {"kind": "held_karp_routes_and_subset_partition", "nodes": budget.nodes},
        status,
    )


def _hgs(
    instance: dict, binary: Path, seconds: float, seed: int, artifact_dir: Path
) -> ReferenceResult:
    vrp, sol = artifact_dir / "input.vrp", artifact_dir / "hgs.sol"
    _write_reference_vrplib(instance, vrp)
    command = [
        str(binary.resolve()),
        str(vrp.resolve()),
        str(sol.resolve()),
        "-t",
        str(seconds),
        "-seed",
        str(seed),
        "-round",
        "1",
        "-log",
        "1",
        "-veh",
        str(len(instance["demands"]) - 1),
    ]
    (artifact_dir / "command.json").write_text(json.dumps(command, indent=2))
    with (
        (artifact_dir / "stdout.log").open("w") as stdout,
        (artifact_dir / "stderr.log").open("w") as stderr,
    ):
        run = subprocess.run(
            command,
            stdout=stdout,
            stderr=stderr,
            timeout=max(10.0, seconds * 5 + 5),
            check=False,
        )
    if run.returncode != 0:
        raise RuntimeError(f"HGS exited with {run.returncode}; see raw logs")
    return ReferenceResult(
        _read_routing_solution(sol),
        False,
        {"kind": "hgs_cvrp_incumbent", "seed": seed, "cpu_budget_sec": seconds},
    )


def _write_reference_vrplib(instance: dict, vrp: Path) -> None:
    from pitbench.solver_drivers.run import PyVRPDriver

    PyVRPDriver._vrplib(instance, vrp)
    # This HGS reader skips the first three header lines (NAME, COMMENT, TYPE).
    lines = vrp.read_text().splitlines()
    lines.insert(1, "COMMENT : PitBench generated reference instance")
    vrp.write_text("\n".join(lines) + "\n")


def _read_routing_solution(sol: Path) -> dict:
    routes = [
        [int(node) for node in match.group(1).split()]
        for match in re.finditer(r"^Route #\d+:\s*(.*)$", sol.read_text(), re.MULTILINE)
    ]
    if not routes:
        raise ValueError("routing solver produced no routes")
    return {"routes": routes}


def _filo(
    instance: dict, binary: Path, seconds: float, seed: int, directory: Path
) -> ReferenceResult:
    vrp = directory / "input.vrp"
    _write_reference_vrplib(instance, vrp)
    command = [
        str(binary),
        str(vrp),
        "--optimization-seconds",
        str(seconds),
        "--seed",
        str(seed),
        "--outpath",
        str(directory),
    ]
    (directory / "command.json").write_text(json.dumps(command, indent=2))
    with (
        (directory / "stdout.log").open("w") as stdout,
        (directory / "stderr.log").open("w") as stderr,
    ):
        run = subprocess.run(
            command,
            stdout=stdout,
            stderr=stderr,
            timeout=max(30.0, seconds + 240),
            check=False,
        )
    if run.returncode != 0:
        raise RuntimeError(f"FILO2 exited with {run.returncode}")
    sol = next(directory.glob("*.vrp.sol"))
    return ReferenceResult(
        _read_routing_solution(sol),
        False,
        {"kind": "filo2_incumbent", "seed": seed, "wall_budget_sec": seconds},
    )


def solve_reference(
    path: Path,
    config: dict,
    index: int,
    *,
    seconds: float,
    max_exact_customers: int = 12,
    max_exact_time_window_customers: int = 0,
) -> tuple[ReferenceResult, dict]:
    """Return a reference and an independent original-instance feasibility check."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("reference budget must be finite and positive")
    kind, budget = config["kind"], SearchBudget(seconds)
    if kind == "euclidean_cvrp":
        instance = json.loads(path.read_text())
        result = _cvrp(
            instance, budget, max_exact_customers, max_exact_time_window_customers
        )
        checked = check_cvrp(instance, result.solution)
    elif kind in {"heterogeneous_fleet_vrp", "multi_depot_vrp", "open_vrp"}:
        instance = json.loads(path.read_text())
        capacities = [vehicle["capacity"] for vehicle in instance["vehicle_types"]]
        type_id = len(capacities) - 1 if kind == "heterogeneous_fleet_vrp" else 0
        capacity = capacities[type_id]
        routes, current, load = [], [], 0
        for node in instance["customers"]:
            demand = instance["demands"][node]
            if current and load + demand > capacity:
                routes.append(current)
                current, load = [], 0
            current.append(node)
            load += demand
        if current:
            routes.append(current)
        depot_count = len(instance["depots"])
        type_ids = (
            [route % depot_count for route in range(len(routes))]
            if kind == "multi_depot_vrp"
            else [type_id] * len(routes)
        )
        result = ReferenceResult(
            {"routes": routes, "vehicle_type_ids": type_ids},
            False,
            {"kind": "sequential_capacity_partition"},
        )
        checked = check_routing_variant(instance, result.solution)
    elif kind == "bin_packing":
        instance = json.loads(path.read_text())
        result = _bin_packing(instance, budget)
        checked = check_bin_packing(
            result.solution, weights=instance["weights"], capacity=instance["capacity"]
        )
    elif kind == "graph_coloring":
        instance = json.loads(path.read_text())
        result = _graph_coloring(instance, budget)
        checked = check_graph_coloring(instance, result.solution)
    elif kind == "multidimensional_knapsack_mip":
        result = _knapsack(path, config, index, budget)
        checked = _check_multidimensional_knapsack_solution(
            result.solution, *_public_multidimensional_knapsack_model(path)
        ).model_dump()
    elif kind == "job_shop_scheduling":
        jobs = _job_shop_operations(path)
        result = _job_shop(jobs, budget)
        checked = _check_job_shop_start_times(
            result.solution["start_times"], jobs
        ).model_dump()
    elif kind == "resource_constrained_project_scheduling":
        instance = json.loads(path.read_text())
        starts = [0] * len(instance["activities"])
        elapsed = 0
        # Generated precedence arcs are acyclic and ordered by activity id.
        for activity in range(len(starts)):
            starts[activity] = elapsed
            elapsed += instance["activities"][activity]["duration"]
        result = ReferenceResult(
            {"start_times": starts},
            False,
            {"kind": "serial_feasible_schedule"},
        )
        checked = check_resource_constrained_project_schedule(
            instance, result.solution
        )
        checked = VerificationResult.model_validate(checked).model_dump()
    elif kind == "single_machine_scheduling_mip":
        processing = _public_scheduling_processing_times(path)
        if processing is None:
            raise ValueError("unsupported scheduling LP")
        order = sorted(range(len(processing)), key=lambda j: (processing[j], j))
        starts, elapsed = [0] * len(order), 0
        for j in order:
            starts[j], elapsed = elapsed, elapsed + processing[j]
        values = {f"s{j}": start for j, start in enumerate(starts)}
        values.update(
            {
                f"z{a}_{b}": int(starts[a] < starts[b])
                for a in range(len(order))
                for b in range(a + 1, len(order))
            }
        )
        result = ReferenceResult(
            {"values": values, "objective": sum(starts)},
            True,
            {"kind": "shortest_processing_time", "order": order},
        )
        checked = _check_scheduling_solution(result.solution, processing).model_dump()
    elif kind == "single_machine_weighted_tardiness_mip":
        result = _weighted_tardiness_schedule_reference(path)
        numeric = MIPFamily._verify_numeric_model_solution(path, result.solution)
        checked = _mip_weighted_tardiness_source_check(
            path, result.solution, numeric
        ).model_dump()
    elif kind in {
        "multidimensional_knapsack_mps",
        "random_set_covering_mip",
        "facility_location_mip",
        "graph_coloring_mip",
    }:
        result = _constructive_mps_reference(path, config, index)
        with tempfile.TemporaryDirectory(prefix="pitbench-mps-reference-") as tmp:
            candidate = Path(tmp) / "candidate.solution.json"
            candidate.write_text(json.dumps(result.solution))
            checked = MIPFamily().verify(path, candidate).model_dump()
    elif kind == "balanced_subset_sum":
        instance = json.loads(path.read_text())
        generated, selected, basis = balanced_subset_sum_data(config, index)
        generated["name"] = path.stem
        if generated != instance:
            raise ValueError("subset-sum provenance does not match input")
        result = ReferenceResult({"selected_items": selected}, True, basis)
        checked = _check_balanced_subset_sum_solution(
            instance, result.solution
        ).model_dump()
    elif kind == "binary_clique_cover_mip":
        groups, planted, _ = binary_clique_cover_data(config, index)
        extensions = binary_clique_cover_fme_extensions(config, index, groups, planted)
        values = {
            name: int(i == planted[g])
            for g, group in enumerate(groups)
            for i, name in enumerate(group)
        }
        values.update({item["column"]: 0 for item in extensions})
        objective = -len(groups)
        if config.get("fme_paper_block", False):
            values.update(fme_i=12, fme_x=0, fme_y=14, fme_z=18)
            objective += 12
        result = ReferenceResult(
            {"values": values, "objective": objective},
            True,
            {
                "kind": "planted_clique_cover_bound",
                "groups": groups,
                "fme_paper_block": config.get("fme_paper_block", False),
            },
        )
        checked = MIPFamily._verify_numeric_model_solution(
            path, result.solution
        ).model_dump()
    else:
        raise ValueError(f"unsupported reference problem kind: {kind}")
    if not checked["feasible"]:
        raise ValueError(f"reference failed independent verification: {checked}")
    return result, checked


def collect_reference_solutions(
    manifest_path: Path,
    output_dir: Path,
    *,
    seconds: float,
    max_exact_customers: int = 12,
    hgs_binary: Path | None = None,
    seeds: tuple[int, ...] = (0,),
) -> dict[str, Any]:
    """Retain every attempt, solution, proof basis, failure and input hash.

    Output is research evidence. This does not install or change benchmark
    oracles, scoring anchors, budgets, task configurations or dataset splits.
    """
    if not math.isfinite(seconds) or seconds <= 0 or not seeds:
        raise ValueError("positive finite seconds and at least one seed required")
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    if "panels" in manifest:
        manifests = [
            manifest_path.parent / p["manifest"] for p in manifest["panels"].values()
        ]
    else:
        manifests = [manifest_path]
    output_dir.mkdir(parents=True, exist_ok=False)
    attempts, references = [], []
    provenance = {
        "reference_module_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "seconds_per_search": seconds,
        "max_exact_customers": max_exact_customers,
        "seeds": list(seeds),
        "hgs": None,
    }
    if hgs_binary is not None:
        hgs_binary = hgs_binary.resolve()
        commit = subprocess.run(
            ["git", "-C", str(hgs_binary.parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        provenance["hgs"] = {
            "source_commit": commit,
            "binary": str(hgs_binary),
            "binary_sha256": hashlib.sha256(hgs_binary.read_bytes()).hexdigest(),
        }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    for panel_manifest in manifests:
        panel = json.loads(panel_manifest.read_text())
        for cell in panel["conditions"]:
            for entry in cell["instances"]:
                path = panel_manifest.parent / entry["path"]
                identity = (
                    f"{panel_manifest.parent.name}/{cell['condition_id']}/{entry['id']}"
                )
                if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                    raise ValueError(f"generated input hash mismatch: {identity}")
                candidates = []
                methods = [("native", None)]
                if (
                    hgs_binary is not None
                    and cell["generator"]["kind"] == "euclidean_cvrp"
                ):
                    methods += [("hgs", seed) for seed in seeds]
                for method, seed in methods:
                    artifact_dir = (
                        output_dir / "attempts" / identity / f"{method}-{seed}"
                    )
                    artifact_dir.mkdir(parents=True)
                    record = {
                        "instance": identity,
                        "instance_path": str(path),
                        "instance_sha256": entry["sha256"],
                        "method": method,
                        "seed": seed,
                        "objective_sense": "minimize",
                        "status": "started",
                    }
                    started = time.monotonic()
                    try:
                        if method == "hgs":
                            instance = json.loads(path.read_text())
                            result = _hgs(
                                instance, hgs_binary, seconds, seed, artifact_dir
                            )
                            checked = check_cvrp(instance, result.solution)
                            if not checked["feasible"]:
                                raise ValueError(f"HGS verification failed: {checked}")
                        else:
                            result, checked = solve_reference(
                                path,
                                cell["generator"],
                                entry["replicate_index"],
                                seconds=seconds,
                                max_exact_customers=max_exact_customers,
                            )
                        solution_path = artifact_dir / "solution.json"
                        solution_path.write_text(
                            json.dumps(result.solution, indent=2) + "\n"
                        )
                        record.update(
                            status=result.search_status,
                            objective=checked["objective"],
                            proven_optimal=result.proven,
                            optimality_basis=result.basis,
                            verification=checked,
                            solution=str(solution_path.relative_to(output_dir)),
                            solution_sha256=hashlib.sha256(
                                solution_path.read_bytes()
                            ).hexdigest(),
                        )
                        candidates.append(record)
                    except Exception as error:
                        record.update(
                            status="failed",
                            error={"type": type(error).__name__, "message": str(error)},
                        )
                    record["wall_time_sec"] = time.monotonic() - started
                    attempts.append(record)
                    (artifact_dir / "record.json").write_text(
                        json.dumps(record, indent=2) + "\n"
                    )
                    with (output_dir / "attempts.jsonl").open("a") as stream:
                        stream.write(json.dumps(record) + "\n")
                if candidates:
                    best = min(
                        candidates,
                        key=lambda r: (r["objective"], not r["proven_optimal"]),
                    )
                    references.append(
                        {
                            **best,
                            "reference_kind": "proven_optimal"
                            if best["proven_optimal"]
                            else "best_found",
                        }
                    )
                else:
                    references.append(
                        {
                            "instance": identity,
                            "instance_sha256": entry["sha256"],
                            "reference_kind": "missing",
                        }
                    )
    summary = {
        "schema_version": "1.0",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "counts": dict(Counter(r["reference_kind"] for r in references)),
        "attempts": len(attempts),
        "failed_attempts": sum(r["status"] == "failed" for r in attempts),
        "references": references,
    }
    (output_dir / "references.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _packing_flow_model(path: Path):
    """Integral capacity-network formulation, aggregated by equal item weights."""
    from pyscipopt import Model, quicksum

    instance = json.loads(path.read_text())
    weights, capacity = instance["weights"], instance["capacity"]
    demand = Counter(weights)
    model = Model("packing_capacity_flow")
    incoming = [[] for _ in range(capacity + 1)]
    outgoing = [[] for _ in range(capacity + 1)]
    arcs = []
    by_weight = {weight: [] for weight in demand}
    for start in range(capacity):
        for weight in sorted(demand):
            if start + weight <= capacity:
                variable = model.addVar(
                    f"a{start}_{weight}", vtype="I", ub=demand[weight]
                )
                arcs.append((start, start + weight, weight, variable.name))
                outgoing[start].append(variable)
                incoming[start + weight].append(variable)
                by_weight[weight].append(variable)
        loss = model.addVar(f"loss{start}", vtype="I", ub=len(weights))
        arcs.append((start, start + 1, None, loss.name))
        outgoing[start].append(loss)
        incoming[start + 1].append(loss)
    bins = model.addVar(
        "bin_count", vtype="I", lb=math.ceil(sum(weights) / capacity), ub=len(weights)
    )
    incoming[0].append(bins)
    outgoing[capacity].append(bins)
    for vertex in range(capacity + 1):
        model.addCons(quicksum(outgoing[vertex]) == quicksum(incoming[vertex]))
    for weight, count in demand.items():
        model.addCons(quicksum(by_weight[weight]) == count)
    model.setObjective(bins, "minimize")

    def decode(values: dict) -> dict:
        counts = {name: round(values[name]) for _, _, _, name in arcs}
        items = {
            w: [i for i, value in enumerate(weights) if value == w] for w in demand
        }
        outgoing_arcs = [[] for _ in range(capacity + 1)]
        for arc in arcs:
            outgoing_arcs[arc[0]].append(arc)
        packing = []
        for _ in range(round(values["bin_count"])):
            vertex, selected = 0, []
            while vertex < capacity:
                arc = next(a for a in outgoing_arcs[vertex] if counts[a[3]] > 0)
                counts[arc[3]] -= 1
                if arc[2] is not None:
                    selected.append(items[arc[2]].pop())
                vertex = arc[1]
            if selected:
                packing.append(selected)
        return {"bins": packing}

    return model, decode


def solve_highs_original_model(path, seconds, seed, threads, directory, emit):
    """Search the original linear input without a second solver's model conversion."""
    import highspy

    solver = highspy.Highs()
    settings = {
        "time_limit": seconds,
        "threads": threads,
        "random_seed": seed,
        "mip_rel_gap": 0.0,
        "mip_abs_gap": 0.0,
        "log_file": str(directory / "solver.log"),
    }
    for name, value in settings.items():
        if solver.setOptionValue(name, value) != highspy.HighsStatus.kOk:
            raise ValueError(f"reference solver rejected {name}")
    if solver.readModel(str(path)) != highspy.HighsStatus.kOk:
        raise ValueError("reference solver rejected original input")
    lp = solver.getLp()
    names = [solver.getColName(i)[1] for i in range(solver.getNumCol())]

    def decode(values):
        return {
            "values": dict(zip(names, values, strict=True)),
            "objective": float(
                sum(
                    cost * value
                    for cost, value in zip(lp.col_cost_, values, strict=True)
                )
                + lp.offset_
            ),
        }

    def improved(event):
        values = list(event.data_out.mip_solution)
        if len(values) == len(names):
            emit(decode(values))

    solver.cbMipImprovingSolution += improved
    solver.run()
    info = solver.getInfo()
    proven = solver.getModelStatus() == highspy.HighsModelStatus.kOptimal
    basis = {
        "kind": "highs_mip_search",
        "solver_version": solver.version(),
        "solver_status": solver.modelStatusToString(solver.getModelStatus()),
        "dual_bound": info.mip_dual_bound
        if math.isfinite(info.mip_dual_bound) and abs(info.mip_dual_bound) < 1e19
        else None,
        "settings": settings,
        "formulation": "original_model",
        "nodes": info.mip_node_count,
        "solver_time_sec": solver.getRunTime(),
    }
    if not solver.getSolution().value_valid:
        if solver.getModelStatus() == highspy.HighsModelStatus.kTimeLimit:
            return ReferenceResult(None, False, basis, "time_limit")
        raise RuntimeError(
            f"HiGHS found no feasible solution: {solver.getModelStatus()}"
        )
    return ReferenceResult(
        decode(solver.getSolution().col_value), proven, basis, basis["solver_status"]
    )


def _solve_native_mip_reference(path, backend, seconds, seed, threads, directory):
    from pitbench.solver_drivers.run import DRIVERS

    output = directory / "native-result.json"
    DRIVERS[backend["solver_driver"]].main(
        [
            "--solver",
            backend["binary"],
            "--instance",
            str(path),
            "--output",
            str(output),
            "--trajectory",
            str(directory / "native-trajectory.jsonl"),
            "--seed",
            str(seed),
            "--budget",
            str(seconds),
            "--threads",
            str(threads),
        ]
    )
    result = json.loads(output.read_text())
    solution = (
        json.loads(output.with_suffix(".solution.json").read_text())
        if result.get("has_solution")
        else None
    )
    basis = {
        "kind": "native_mip_search",
        "source_commit": backend["source_commit"],
        "binary_sha256": backend["binary_sha256"],
        "dual_bound": result.get("dual_bound"),
        "solver_status": result.get("solver_status"),
        "cpu_time_sec": result.get("cpu_time_sec"),
        "solver_time_sec": result.get("wall_time_sec"),
        "nodes": result.get("nodes"),
    }
    return ReferenceResult(
        solution,
        result.get("solver_termination") == "optimal"
        and "gap tolerance" not in str(result.get("solver_status", "")).lower(),
        basis,
        result.get("solver_termination", "other"),
    )


def _solve_highs_docker_reference(
    path: Path,
    backend: dict,
    seconds: float,
    seed: int,
    threads: int,
    directory: Path,
    cpus: list[int],
    warm: dict | None = None,
) -> ReferenceResult:
    """Run the pinned Highs image when host highspy ABI differs from the image."""
    temporary = directory / "tmp"
    temporary.mkdir(exist_ok=True)
    options_path = directory / "highs.options"
    options_path.write_text(
        "mip_rel_gap = 0\nmip_abs_gap = 0\nlog_file = /output/solver.log\n"
    )
    warm_path = None
    if warm is not None:
        warm_path = directory / "warm.solution"
        raw = warm.get("raw_solution")
        if raw is None:
            values = warm.get("values")
            model = _read_mps_numeric_model(path)
            if (
                not isinstance(values, dict)
                or model is None
                or set(values) != set(model["columns"])
            ):
                raise ValueError("Highs MIP warm start has no complete MPS assignment")
            lines = [
                "Model status",
                "Time limit reached",
                "",
                "# Primal solution values",
                "Feasible",
                f"Objective {warm['objective']}",
                f"# Columns {len(model['columns'])}",
                *(f"{name} {values[name]}" for name in model["columns"]),
                "None",
                "",
                "# Basis",
                "None",
            ]
            raw = "\n".join(lines) + "\n"
        warm_path.write_text(raw)
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--cpus",
        str(threads),
        "--cpuset-cpus",
        ",".join(map(str, cpus)),
        "--memory",
        backend.get("memory", "4g"),
        "--pids-limit",
        "512",
        "--volume",
        f"{path.resolve()}:/input/model.mps:ro",
        "--volume",
        f"{options_path.resolve()}:/input/highs.options:ro",
        "--volume",
        f"{directory.resolve()}:/output:rw",
        "--volume",
        f"{temporary.resolve()}:/tmp:rw",
        "--env",
        "TMPDIR=/tmp",
        backend["image"],
        "/workspace/repo/build/bin/highs",
        "--model_file",
        "/input/model.mps",
        "--time_limit",
        str(seconds),
        "--threads",
        str(threads),
        "--random_seed",
        str(seed),
        "--options_file",
        "/input/highs.options",
        "--solution_file",
        "/output/native.solution",
    ]
    if warm_path is not None:
        command.extend(["--read_solution_file", "/output/warm.solution"])
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    (directory / "container.stdout.log").write_text(completed.stdout)
    (directory / "container.stderr.log").write_text(completed.stderr)
    if completed.returncode:
        raise RuntimeError(
            completed.stderr.strip() or "Highs reference container failed"
        )
    solution_path = directory / "native.solution"
    raw_solution = solution_path.read_text() if solution_path.is_file() else None
    status_match = re.search(r"^\s*Status\s+(.+)$", completed.stdout, re.MULTILINE)
    dual_match = re.search(r"^\s*Dual bound\s+(-?[0-9.eE+]+)", completed.stdout, re.MULTILINE)
    nodes_match = re.search(r"^\s*Nodes\s+([0-9]+)", completed.stdout, re.MULTILINE)
    solver_status = status_match.group(1).strip() if status_match else "unknown"
    objective_match = (
        re.search(r"^Objective\s+(-?[0-9.eE+]+)$", raw_solution, re.MULTILINE)
        if raw_solution
        else None
    )
    has_solution = objective_match is not None
    solution = (
        {
            "raw_solution": raw_solution,
            "objective": float(objective_match.group(1)),
        }
        if has_solution
        else None
    )
    basis = {
        "kind": "highs_mip_search",
        "solver_status": solver_status,
        "dual_bound": float(dual_match.group(1)) if dual_match else None,
        "solver_time_sec": seconds,
        "nodes": int(nodes_match.group(1)) if nodes_match else None,
        "image": backend["image"],
    }
    return ReferenceResult(
        solution,
        solver_status.lower() == "optimal",
        basis,
        solver_status,
    )


def _solve_mip_backend(
    path: Path,
    kind: str,
    backend: str,
    seconds: float,
    seed: int,
    threads: int,
    directory: Path,
    emit,
) -> ReferenceResult:
    if backend == "highs" and kind not in {"bin_packing", "balanced_subset_sum"}:
        return solve_highs_original_model(path, seconds, seed, threads, directory, emit)
    from pyscipopt import SCIP_EVENTTYPE, Eventhdlr, Model

    if kind == "bin_packing":
        model, decode = _packing_flow_model(path)
    elif kind == "balanced_subset_sum":
        from pyscipopt import quicksum

        instance = json.loads(path.read_text())
        model = Model("fixed_cardinality_subset_sum")
        selected = [
            model.addVar(f"x{i}", vtype="B") for i in range(len(instance["weights"]))
        ]
        model.addCons(quicksum(selected) == instance["cardinality"])
        delta = (
            quicksum(w * v for w, v in zip(instance["weights"], selected, strict=True))
            - instance["target"]
        )
        deviation = model.addVar("deviation", vtype="I", ub=sum(instance["weights"]))
        model.addCons(deviation >= delta)
        model.addCons(deviation >= -delta)
        model.setObjective(deviation, "minimize")

        def decode(values):
            return {
                "selected_items": [
                    i for i in range(len(selected)) if values[f"x{i}"] > 0.5
                ]
            }

        emit({"selected_items": list(range(instance["cardinality"]))})
    else:
        model = Model()
        model.readProblem(str(path))

        def decode(values):
            return {
                "values": values,
                "objective": sum(v.getObj() * values[v.name] for v in model.getVars()),
            }

    model.writeProblem(str(directory / "model.mps"))
    variables = model.getVars()
    if backend == "scip":
        model.setRealParam("limits/time", seconds)
        model.setRealParam("limits/memory", 8192)
        model.setRealParam("numerics/feastol", 1e-6)
        model.setIntParam("randomization/randomseedshift", seed)
        model.setIntParam("parallel/maxnthreads", threads)
        model.setLogfile(str(directory / "solver.log"))

        class Incumbents(Eventhdlr):
            def eventinit(self):
                self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

            def eventexit(self):
                self.model.dropEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

            def eventexec(self, event):
                sol = self.model.getBestSol()
                if sol is not None:
                    emit(
                        decode(
                            {v.name: self.model.getSolVal(sol, v) for v in variables}
                        )
                    )

        model.includeEventhdlr(
            Incumbents(), "reference_incumbents", "persist verified incumbents"
        )
        model.optimize()
        if model.getNSols() == 0:
            raise RuntimeError(
                f"SCIP finished without a feasible solution: {model.getStatus()}"
            )
        sol = model.getBestSol()
        solution = decode({v.name: model.getSolVal(sol, v) for v in variables})
        status, bound = str(model.getStatus()), model.getDualbound()
        proven = status == "optimal"
    else:
        import highspy

        solver = highspy.Highs()
        solver.readModel(str(directory / "model.mps"))
        for name, value in {
            "time_limit": seconds,
            "threads": threads,
            "random_seed": seed,
            "mip_rel_gap": 0.0,
            "log_file": str(directory / "solver.log"),
        }.items():
            solver.setOptionValue(name, value)
        names = [solver.getColName(i)[1] for i in range(solver.getNumCol())]

        def improved(event):
            values = list(event.data_out.mip_solution)
            if len(values) == len(names):
                emit(decode(dict(zip(names, values, strict=True))))

        solver.cbMipImprovingSolution += improved
        solver.run()
        if not solver.getSolution().value_valid:
            raise RuntimeError(
                f"HiGHS finished without a feasible solution: {solver.getModelStatus()}"
            )
        solution = decode(dict(zip(names, solver.getSolution().col_value, strict=True)))
        status = solver.modelStatusToString(solver.getModelStatus())
        bound = solver.getInfo().mip_dual_bound
        proven = solver.getModelStatus() == highspy.HighsModelStatus.kOptimal
    basis = {
        "kind": f"{backend}_mip_search",
        "solver_status": status,
        "dual_bound": bound if math.isfinite(bound) and abs(bound) < 1e19 else None,
        "formulation": "capacity_arc_flow"
        if kind == "bin_packing"
        else "original_model",
    }
    return ReferenceResult(solution, proven, basis, status)


def _solve_cp_backend(
    path: Path,
    kind: str,
    seconds: float,
    seed: int,
    threads: int,
    directory: Path,
    emit,
    warm: dict | None,
    *,
    build_only: bool = False,
) -> ReferenceResult | dict:
    from ortools.sat.python import cp_model

    model = cp_model.CpModel()
    if kind == "bin_packing":
        instance = json.loads(path.read_text())
        weights = instance["weights"]
        capacity = int(instance["capacity"])
        if not weights or any(type(weight) is not int or not 0 < weight <= capacity for weight in weights):
            raise ValueError("invalid bin-packing weights or capacity")
        total = sum(weights)
        lower_bound = math.ceil(total / capacity)
        order = sorted(range(len(weights)), key=lambda index: (-weights[index], index))
        initial_bins: list[list[int]] = []
        initial_loads: list[int] = []
        for item in order:
            choices = [
                index
                for index, load in enumerate(initial_loads)
                if load + weights[item] <= capacity
            ]
            if choices:
                bin_index = min(choices, key=lambda index: (capacity - initial_loads[index], index))
                initial_bins[bin_index].append(item)
                initial_loads[bin_index] += weights[item]
            else:
                initial_bins.append([item])
                initial_loads.append(weights[item])
        emit({"bins": initial_bins})
        bin_count = len(initial_bins)
        assignments = [
            [model.new_bool_var(f"item_{item}_bin_{bin_index}") for bin_index in range(bin_count)]
            for item in range(len(weights))
        ]
        used = [model.new_bool_var(f"used_{bin_index}") for bin_index in range(bin_count)]
        for item in range(len(weights)):
            model.add_exactly_one(assignments[item])
        for bin_index in range(bin_count):
            model.add(
                sum(weights[item] * assignments[item][bin_index] for item in range(len(weights)))
                <= capacity * used[bin_index]
            )
            model.add(
                sum(assignments[item][bin_index] for item in range(len(weights)))
                >= used[bin_index]
            )
            if bin_index:
                model.add(used[bin_index - 1] >= used[bin_index])
        objective = model.new_int_var(lower_bound, bin_count, "bin_count")
        model.add(objective == sum(used))

        def decode(value):
            return {
                "bins": [
                    [item for item in range(len(weights)) if value(assignments[item][bin_index])]
                    for bin_index in range(bin_count)
                    if value(used[bin_index])
                ]
            }

        if warm:
            warm_bins = warm.get("bins", [])
            bin_of = {item: bin_index for bin_index, items in enumerate(warm_bins) for item in items}
            if len(bin_of) == len(weights):
                for item, bin_index in bin_of.items():
                    if bin_index < bin_count:
                        model.add_hint(assignments[item][bin_index], 1)
                for bin_index in range(min(len(warm_bins), bin_count)):
                    model.add_hint(used[bin_index], 1)
    elif kind == "job_shop_scheduling":
        jobs = _job_shop_operations(path)
        horizon = sum(d for job in jobs for _, d in job)
        starts, ends, machines = [], [], {}
        for j, job in enumerate(jobs):
            previous = None
            for op, (machine, duration) in enumerate(job):
                start = model.new_int_var(0, horizon, f"s{j}_{op}")
                end = model.new_int_var(0, horizon, f"e{j}_{op}")
                interval = model.new_interval_var(start, duration, end, f"i{j}_{op}")
                machines.setdefault(machine, []).append(interval)
                if previous is not None:
                    model.add(start >= previous)
                previous = end
                starts.append(start)
                ends.append(end)
            # Only job completion times are needed for the makespan.
        for intervals in machines.values():
            model.add_no_overlap(intervals)
        objective = model.new_int_var(0, horizon, "makespan")
        model.add_max_equality(objective, ends)

        def decode(value):
            return {"start_times": [value(v) for v in starts]}

        if warm:
            for variable, value in zip(starts, warm["start_times"], strict=True):
                model.add_hint(variable, value)
    elif kind == "resource_constrained_project_scheduling":
        instance = json.loads(path.read_text())
        activities = instance["activities"]
        horizon = sum(int(activity["duration"]) for activity in activities)
        starts, by_resource = [], [
            ([], []) for _ in instance["resource_capacities"]
        ]
        serial_starts, elapsed = [], 0
        for activity in activities:
            serial_starts.append(elapsed)
            elapsed += int(activity["duration"])
        emit({"start_times": serial_starts})
        for index, activity in enumerate(activities):
            start = model.new_int_var(0, horizon, f"start_{index}")
            end = model.new_int_var(0, horizon, f"end_{index}")
            interval = model.new_interval_var(
                start, int(activity["duration"]), end, f"activity_{index}"
            )
            starts.append(start)
            for resource, demand in enumerate(activity["resource_demands"]):
                if demand:
                    by_resource[resource][0].append(interval)
                    by_resource[resource][1].append(demand)
        for predecessor, successor in instance["precedence_arcs"]:
            model.add(
                starts[successor]
                >= starts[predecessor]
                + int(activities[predecessor]["duration"])
            )
        for resource, (resource_intervals, demands) in enumerate(by_resource):
            if resource_intervals:
                model.add_cumulative(
                    resource_intervals,
                    demands,
                    instance["resource_capacities"][resource],
                )
        objective = model.new_int_var(0, horizon, "makespan")
        model.add_max_equality(
            objective,
            [
                starts[index] + int(activity["duration"])
                for index, activity in enumerate(activities)
            ],
        )

        def decode(value):
            return {"start_times": [value(variable) for variable in starts]}

        if warm:
            for variable, value in zip(starts, warm["start_times"], strict=True):
                model.add_hint(variable, value)
    elif kind == "graph_coloring":
        instance = json.loads(path.read_text())
        n = instance["vertex_count"]
        adjacent = [set() for _ in range(n)]
        endpoints = instance["edge_endpoints"]
        for a, b in zip(endpoints[::2], endpoints[1::2], strict=True):
            adjacent[a].add(b)
            adjacent[b].add(a)
        coloring = [-1] * n
        clique = []
        for v in sorted(range(n), key=lambda v: -len(adjacent[v])):
            forbidden = {coloring[w] for w in adjacent[v]}
            coloring[v] = next(c for c in range(n) if c not in forbidden)
            if all(v in adjacent[w] for w in clique):
                clique.append(v)
        emit({"colors": coloring})
        upper = len(set(coloring))
        if warm:
            upper = min(upper, len(set(warm["colors"])))
        colors = [model.new_int_var(1, upper, f"c{v}") for v in range(n)]
        for a, b in zip(endpoints[::2], endpoints[1::2], strict=True):
            model.add(colors[a] != colors[b])
        for color, v in enumerate(clique, 1):
            model.add(colors[v] == color)
        objective = model.new_int_var(len(clique), upper, "color_count")
        model.add_max_equality(objective, colors)

        def decode(value):
            return {"colors": [value(v) - 1 for v in colors]}
    elif kind == "balanced_subset_sum":
        instance = json.loads(path.read_text())
        weights, target = instance["weights"], instance["target"]
        emit({"selected_items": list(range(instance["cardinality"]))})
        selected = [model.new_bool_var(f"x{i}") for i in range(len(weights))]
        model.add(sum(selected) == instance["cardinality"])
        objective = model.new_int_var(0, sum(weights), "deviation")
        model.add_abs_equality(
            objective,
            sum(w * v for w, v in zip(weights, selected, strict=True)) - target,
        )

        def decode(value):
            return {"selected_items": [i for i, v in enumerate(selected) if value(v)]}

        if warm:
            for i, variable in enumerate(selected):
                model.add_hint(variable, int(i in warm["selected_items"]))
    elif kind == "multidimensional_knapsack_mip":
        weights, capacities, profits = _public_multidimensional_knapsack_model(path)
        n = len(weights[0])
        variables = [model.new_bool_var(f"x{i}") for i in range(n)]
        for row, capacity in zip(weights, capacities, strict=True):
            model.add(
                sum(w * v for w, v in zip(row, variables, strict=True)) <= capacity
            )
        objective = -sum(p * v for p, v in zip(profits, variables, strict=True))

        def decode(value):
            return {
                "values": {f"x{i}": value(v) for i, v in enumerate(variables)},
                "objective": -sum(
                    p * value(v) for p, v in zip(profits, variables, strict=True)
                ),
            }
    else:
        raise ValueError(f"CP-SAT reference backend does not support {kind}")
    model.minimize(objective)
    model.export_to_file(str(directory / "model.pbtxt"))
    if build_only:
        return {"model": model, "decode": decode}
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_search_workers = threads
    solver.parameters.random_seed = seed
    solver.parameters.log_search_progress = True
    if kind == "balanced_subset_sum":
        # High-precision subset instances caused very long domain computations
        # inside presolve, before the solver could honor its search deadline.
        solver.parameters.cp_model_presolve = False

    class Incumbents(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            emit(decode(self.value))

    status = solver.solve(model, Incumbents())
    if status == cp_model.UNKNOWN:
        incumbent_path = directory / "incumbent.json"
        incumbent = (
            json.loads(incumbent_path.read_text())
            if incumbent_path.exists()
            else None
        )
        return ReferenceResult(
            json.loads(Path(incumbent["solution"]).read_text())
            if incumbent
            else None,
            False,
            {
                "kind": "cp_sat_search",
                "solver_status": "UNKNOWN",
                "solver_has_incumbent": bool(incumbent),
                "dual_bound": solver.best_objective_bound,
                "threads": threads,
            },
            "time_limit",
        )
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(
            f"CP-SAT returned {solver.status_name(status)} without a solution"
        )
    return ReferenceResult(
        decode(solver.value),
        status == cp_model.OPTIMAL,
        {
            "kind": "cp_sat_search",
            "solver_status": solver.status_name(status),
            "dual_bound": solver.best_objective_bound,
            "threads": threads,
        },
        solver.status_name(status),
    )


def _solve_pyvrp_backend(
    instance: dict,
    seconds: float,
    seed: int,
    directory: Path,
    emit,
    warm: dict | None,
    checkpoint_sec: float,
) -> ReferenceResult:
    import pyvrp
    from pyvrp.stop import MaxRuntime

    from pitbench.solver_drivers.run import PyVRPDriver

    if instance.get("problem_kind") in {
        "heterogeneous_fleet_vrp",
        "multi_depot_vrp",
        "open_vrp",
    }:
        model, customer_nodes = PyVRPDriver._routing_variant_model(pyvrp, instance)
        result = model.solve(
            MaxRuntime(seconds),
            seed=seed,
            display=False,
        )
        if not result.best.is_feasible():
            return ReferenceResult(
                None,
                False,
                {
                    "kind": "pyvrp_routing_variant_search",
                    "solver_runtime_sec": result.runtime,
                    "iterations": result.num_iterations,
                },
                "time_limit",
            )
        routes, vehicle_type_ids = [], []
        for route in result.best.routes():
            routes.append(
                [
                    customer_nodes[int(activity.idx)]
                    for activity in route
                    if activity.is_client()
                ]
            )
            vehicle_type = getattr(route, "vehicle_type")
            vehicle_type_ids.append(
                int(vehicle_type() if callable(vehicle_type) else vehicle_type)
            )
        solution = {"routes": routes, "vehicle_type_ids": vehicle_type_ids}
        emit(
            solution,
            False,
            {
                "kind": "pyvrp_routing_variant_search",
                "solver_runtime_sec": result.runtime,
                "iterations": result.num_iterations,
            },
            "incumbent",
        )
        return ReferenceResult(
            solution,
            False,
            {
                "kind": "pyvrp_routing_variant_search",
                "solver_runtime_sec": result.runtime,
                "iterations": result.num_iterations,
            },
            "time_limit",
        )

    vrp = directory / "input.vrp"
    PyVRPDriver._vrplib(instance, vrp)
    data = pyvrp.read(vrp, round_func=PyVRPDriver.round_euc_2d)
    initial = (
        pyvrp.Solution(data, [[node - 1 for node in route] for route in warm["routes"]])
        if warm
        else None
    )
    deadline = time.monotonic() + seconds
    rounds = 0
    best = None
    while time.monotonic() < deadline:
        result = pyvrp.solve(
            data,
            MaxRuntime(min(checkpoint_sec, deadline - time.monotonic())),
            seed=seed + rounds,
            display=True,
            initial_solution=initial,
        )
        if result.best.is_feasible():
            solution = {
                "routes": [
                    PyVRPDriver._route_visits(route) for route in result.best.routes()
                ]
            }
            emit(solution)
            initial = result.best
            best = solution
        rounds += 1
    return ReferenceResult(
        best,
        False,
        {"kind": "pyvrp_incumbent", "rounds": rounds, "checkpoint_sec": checkpoint_sec},
        "time_limit",
    )


def run_reference_worker(job_path: Path) -> dict:
    """One isolated solver process with independently checked live incumbents."""
    from pitbench.evaluator.artifacts import write_json

    job = json.loads(job_path.read_text())
    directory = Path(job["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    cpus = job["cpus"]
    os.sched_setaffinity(0, cpus)
    path = Path(job["instance"]["path"])
    kind = job["instance"]["generator"]["kind"]
    backend = job["backend"]["name"]
    implementation = job["backend"].get("driver", backend)
    seconds, seed = job["seconds"], job["seed"]
    started = time.monotonic()
    best = None
    improvements = 0
    record = {
        "job_id": job["id"],
        "instance": job["instance"]["id"],
        "instance_path": str(path),
        "instance_sha256": job["instance"]["sha256"],
        "backend": backend,
        "seed": seed,
        "budget_sec": seconds,
        "cpus": cpus,
        "pid": os.getpid(),
        "worker_boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "host_id": job.get("host_id"),
        "started_at": time.time(),
        "worker_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "code_revision": job.get("code_revision", "initial_snapshot"),
        "execution_status": "running",
        "objective_sense": "minimize",
    }
    write_json(directory / "running.json", record)

    def emit(solution, proven=False, basis=None, solver_status="incumbent"):
        nonlocal best, improvements
        candidate = directory / "candidate.solution.json"
        write_json(candidate, solution)
        checked = verify_reference_file(path, candidate, kind)
        if not checked["feasible"]:
            write_json(
                directory / "invalid_incumbent.json",
                {"solution": solution, "verification": checked},
            )
            raise ValueError(
                f"reference incumbent violates original problem: {checked}"
            )
        if best is not None and checked["objective"] > best["objective"]:
            return
        if (
            best is not None
            and checked["objective"] == best["objective"]
            and not proven
            and basis is None
        ):
            return
        solution_path = directory / "incumbents" / f"{improvements:06d}.solution.json"
        write_json(solution_path, solution)
        improvements += 1
        best = {
            **record,
            "objective": checked["objective"],
            "verification": checked,
            "verified_feasible": True,
            "proven_optimal": proven,
            "reference_kind": "proven_optimal" if proven else "best_found",
            "solution": str(solution_path.resolve()),
            "solution_sha256": hashlib.sha256(solution_path.read_bytes()).hexdigest(),
            "optimality_basis": basis or {},
            "solver_status": solver_status,
            "wall_time_sec": time.monotonic() - started,
        }
        write_json(directory / "incumbent.json", best)
        with (directory / "trajectory.jsonl").open("a") as stream:
            stream.write(json.dumps(best) + "\n")

    try:
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["instance_sha256"]:
            raise ValueError("reference input hash mismatch")
        warm = None
        if job.get("warm_start"):
            warm_path = Path(job["warm_start"])
            if verify_reference_file(path, warm_path, kind)["feasible"]:
                warm = json.loads(warm_path.read_text())
                emit(warm)
        if implementation == "native":
            result, _ = solve_reference(
                path,
                job["instance"]["generator"],
                job["instance"]["replicate_index"],
                seconds=seconds,
                max_exact_customers=job["backend"].get("max_exact_customers", 12),
                max_exact_time_window_customers=job["backend"].get(
                    "max_exact_time_window_customers", 0
                ),
            )
        elif implementation == "hgs":
            result = _hgs(
                json.loads(path.read_text()),
                Path(job["backend"]["binary"]),
                seconds,
                seed,
                directory,
            )
        elif implementation == "filo2":
            result = _filo(
                json.loads(path.read_text()),
                Path(job["backend"]["binary"]),
                seconds,
                seed,
                directory,
            )
        elif implementation == "pyvrp":
            result = _solve_pyvrp_backend(
                json.loads(path.read_text()),
                seconds,
                seed,
                directory,
                emit,
                warm,
                job["backend"].get("checkpoint_sec", 300),
            )
        elif implementation == "native_mip_cli":
            result = _solve_native_mip_reference(
                path, job["backend"], seconds, seed, len(cpus), directory
            )
        elif implementation == "highs_docker":
            result = _solve_highs_docker_reference(
                path,
                job["backend"],
                seconds,
                seed,
                len(cpus),
                directory,
                cpus,
                warm,
            )
        elif implementation in {"scip", "highs"}:
            result = _solve_mip_backend(
                path, kind, implementation, seconds, seed, len(cpus), directory, emit
            )
        elif implementation == "cp_sat":
            result = _solve_cp_backend(
                path, kind, seconds, seed, len(cpus), directory, emit, warm
            )
        else:
            raise ValueError(f"unknown reference backend: {backend}")
        if result.solution is not None:
            emit(result.solution, result.proven, result.basis, result.search_status)
        record.update(
            execution_status="completed",
            solver_status=result.search_status,
            has_solution=result.solution is not None,
            optimality_basis=result.basis,
            proven_optimal=result.proven,
        )
    except Exception as error:
        import traceback

        traceback.print_exc()
        record.update(
            execution_status="solver_error",
            error={"type": type(error).__name__, "message": str(error)},
        )
    if best:
        record.update(
            {
                key: value
                for key, value in best.items()
                if key not in {"execution_status", "wall_time_sec"}
            }
        )
    record["process_wall_sec"] = time.monotonic() - started
    record["incumbents"] = improvements
    write_json(directory / "result.json", record)
    return record


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Isolated offline reference worker")
    parser.add_argument("--worker-job", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(run_reference_worker(arguments.worker_job)))
