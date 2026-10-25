"""Offline reference search on generated grids; separate from scored solver runs.

Small instances can be exhausted. A feasible incumbent after a time limit is
only a best-found reference, never a published BKS or an optimality claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
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
    first_fit_decreasing_packing,
    multidimensional_knapsack_data,
)
from pitbench.problem_families.public_verification import (
    check_bin_packing,
    check_cvrp,
    check_graph_coloring,
    edge_cost,
)
from pitbench.problem_families.verification import (
    CPFamily,
    CVRPFamily,
    MIPFamily,
    _check_balanced_subset_sum_solution,
    _check_job_shop_start_times,
    _check_multidimensional_knapsack_solution,
    _check_scheduling_solution,
    _public_multidimensional_knapsack_data,
    _public_scheduling_processing_times,
)


def generated_reference_instances(manifest_path: Path) -> list[dict]:
    """Read grid/suite manifests into the common offline-reference input contract."""
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
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
    if kind == "euclidean_cvrp":
        checked = CVRPFamily().verify(instance, solution)
    elif kind in {"bin_packing", "graph_coloring", "balanced_subset_sum"}:
        checked = CPFamily().verify(instance, solution)
    elif kind == "job_shop_scheduling":
        checked = _check_job_shop_start_times(
            json.loads(solution.read_text())["start_times"],
            _job_shop_operations(instance),
        )
    else:
        checked = MIPFamily().verify(instance, solution)
    return checked.model_dump()


@dataclass
class ReferenceResult:
    solution: dict
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
    parsed = _public_multidimensional_knapsack_data(path)
    if parsed is None:
        raise ValueError("unsupported knapsack LP")
    weights, capacities = parsed
    n, dimensions = len(weights[0]), len(weights)
    profits = [sum(row[i] for row in weights) for i in range(n)]
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
        if (generated_weights, generated_caps) != parsed:
            raise ValueError("knapsack provenance does not match input")
        best = planted
        best_value = sum(profits[i] for i in best)
    chosen: list[int] = []
    used = [0] * dimensions
    bound = sum(capacities)

    def search(pos: int, value: int) -> None:
        nonlocal best, best_value
        budget.tick()
        if value > best_value:
            best, best_value = chosen.copy(), value
        if (
            pos == n
            or best_value == bound
            or value
            + min(
                remaining[pos], sum(capacities[d] - used[d] for d in range(dimensions))
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


def _cvrp(instance: dict, budget: SearchBudget, max_customers: int) -> ReferenceResult:
    n = len(instance["demands"]) - 1
    if instance.get("depot", 0) != 0:
        raise ValueError("reference CVRP expects depot zero")
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
) -> tuple[ReferenceResult, dict]:
    """Return a reference and an independent original-instance feasibility check."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("reference budget must be finite and positive")
    kind, budget = config["kind"], SearchBudget(seconds)
    if kind == "euclidean_cvrp":
        instance = json.loads(path.read_text())
        result = _cvrp(instance, budget, max_exact_customers)
        checked = check_cvrp(instance, result.solution)
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
            result.solution, *_public_multidimensional_knapsack_data(path)
        ).model_dump()
    elif kind == "job_shop_scheduling":
        jobs = _job_shop_operations(path)
        result = _job_shop(jobs, budget)
        checked = _check_job_shop_start_times(
            result.solution["start_times"], jobs
        ).model_dump()
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
) -> ReferenceResult:
    from ortools.sat.python import cp_model

    model = cp_model.CpModel()
    if kind == "job_shop_scheduling":
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
        weights, capacities = _public_multidimensional_knapsack_data(path)
        n = len(weights[0])
        variables = [model.new_bool_var(f"x{i}") for i in range(n)]
        for row, capacity in zip(weights, capacities, strict=True):
            model.add(
                sum(w * v for w, v in zip(row, variables, strict=True)) <= capacity
            )
        profits = [sum(row[i] for row in weights) for i in range(n)]
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
    if status == cp_model.UNKNOWN and (directory / "incumbent.json").exists():
        incumbent = json.loads((directory / "incumbent.json").read_text())
        return ReferenceResult(
            json.loads(Path(incumbent["solution"]).read_text()),
            False,
            {
                "kind": "cp_sat_search",
                "solver_status": "UNKNOWN",
                "solver_has_incumbent": False,
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

    vrp = directory / "input.vrp"
    PyVRPDriver._vrplib(instance, vrp)
    data = pyvrp.read(vrp, round_func="round")
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
    if best is None:
        raise RuntimeError("PyVRP produced no feasible solution")
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
        if backend == "native":
            result, _ = solve_reference(
                path,
                job["instance"]["generator"],
                job["instance"]["replicate_index"],
                seconds=seconds,
            )
        elif backend == "hgs":
            result = _hgs(
                json.loads(path.read_text()),
                Path(job["backend"]["binary"]),
                seconds,
                seed,
                directory,
            )
        elif backend == "filo2":
            result = _filo(
                json.loads(path.read_text()),
                Path(job["backend"]["binary"]),
                seconds,
                seed,
                directory,
            )
        elif backend == "pyvrp":
            result = _solve_pyvrp_backend(
                json.loads(path.read_text()),
                seconds,
                seed,
                directory,
                emit,
                warm,
                job["backend"].get("checkpoint_sec", 300),
            )
        elif backend in {"scip", "highs"}:
            result = _solve_mip_backend(
                path, kind, backend, seconds, seed, len(cpus), directory, emit
            )
        elif backend == "cp_sat":
            result = _solve_cp_backend(
                path, kind, seconds, seed, len(cpus), directory, emit, warm
            )
        else:
            raise ValueError(f"unknown reference backend: {backend}")
        emit(result.solution, result.proven, result.basis, result.search_status)
        record.update(execution_status="completed", solver_status=result.search_status)
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
