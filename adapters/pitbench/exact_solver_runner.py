"""JSON bridge for the shared native exact solver adapter."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def solve(request: dict, *, engine: str) -> dict:
    if engine != "gecode":
        raise ValueError(f"unsupported exact solver engine: {engine}")
    if request.get("threads") != 1:
        raise ValueError("Gecode exact tasks require one thread")
    instance_path = Path(request["instance_path"])
    if instance_path.suffix.lower() == ".txt":
        tokens = instance_path.read_text().split()
        if any(not token.isdecimal() for token in tokens) or len(tokens) < 2:
            raise ValueError("invalid job-shop text instance")
        values = [int(token) for token in tokens]
        jobs, machines = values[:2]
        if jobs < 1 or machines < 1 or len(values) != 2 + 2 * jobs * machines:
            raise ValueError("job-shop dimensions do not match input")
        kind = "job_shop_scheduling"
        instance = None
    else:
        instance = json.loads(instance_path.read_text())
        kind = instance.get("problem_kind", "bin_packing")
    parameters = request.get("parameters") or {}
    executable = os.environ.get("PITBENCH_EXACT_SOLVER", "build/pitbench-exact")
    budget_ms = str(max(1, round(float(request["budget_sec"]) * 1000)))
    if kind == "job_shop_scheduling":
        argv = [
            executable,
            kind,
            budget_ms,
            str(jobs),
            str(machines),
            *(str(value) for value in values[2:]),
        ]
    elif kind == "bin_packing":
        weights = instance.get("weights")
        capacity = instance.get("capacity")
        if (
            not isinstance(weights, list)
            or not weights
            or any(type(weight) is not int or weight <= 0 for weight in weights)
            or type(capacity) is not int
            or capacity <= 0
            or any(weight > capacity for weight in weights)
        ):
            raise ValueError("invalid normalized bin-packing instance")
        level = parameters.get("gecode.bin_packing_propagation_level", "full")
        if level not in {"full", "advanced", "basic", "default"}:
            raise ValueError(f"unsupported Gecode propagation level: {level!r}")
        argv = [
            executable,
            budget_ms,
            level,
            str(capacity),
            str(len(weights)),
            *(str(weight) for weight in weights),
        ]
    elif kind == "balanced_subset_sum":
        weights = instance.get("weights")
        cardinality, target = instance.get("cardinality"), instance.get("target")
        if (
            not isinstance(weights, list)
            or not weights
            or any(type(value) is not int or value <= 0 for value in weights)
            or type(cardinality) is not int
            or type(target) is not int
        ):
            raise ValueError("invalid balanced subset-sum instance")
        argv = [executable, kind, budget_ms, str(cardinality), str(target), str(len(weights)), *(str(v) for v in weights)]
    elif kind == "graph_coloring":
        vertices, edges = instance.get("vertex_count"), instance.get("edge_endpoints")
        if (
            type(vertices) is not int
            or vertices < 2
            or not isinstance(edges, list)
            or len(edges) % 2
            or any(type(value) is not int for value in edges)
        ):
            raise ValueError("invalid graph-coloring instance")
        argv = [executable, kind, budget_ms, str(vertices), str(len(edges) // 2), *(str(v) for v in edges)]
    elif kind == "resource_constrained_project_scheduling":
        activities = instance.get("activities")
        capacities = instance.get("resource_capacities")
        arcs = instance.get("precedence_arcs")
        if not isinstance(activities, list) or not isinstance(capacities, list) or not isinstance(arcs, list):
            raise ValueError("invalid RCPSP instance")
        durations = [activity.get("duration") for activity in activities]
        demands = [value for activity in activities for value in activity.get("resource_demands", [])]
        endpoints = [value for arc in arcs for value in arc]
        if (
            any(type(value) is not int or value <= 0 for value in durations)
            or any(type(value) is not int or value < 0 for value in demands)
            or any(type(value) is not int or value <= 0 for value in capacities)
            or any(len(activity.get("resource_demands", [])) != len(capacities) for activity in activities)
            or any(type(value) is not int for value in endpoints)
        ):
            raise ValueError("invalid flattened RCPSP data")
        argv = [
            executable,
            kind,
            budget_ms,
            str(len(activities)),
            str(len(capacities)),
            str(len(arcs)),
            *(str(v) for v in capacities),
            *(str(v) for v in durations),
            *(str(v) for v in demands),
            *(str(v) for v in endpoints),
        ]
    else:
        raise ValueError(f"Gecode adapter does not support {kind}")
    completed = subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        timeout=float(request["budget_sec"]) + 10,
    )
    if completed.returncode:
        diagnostics = "\n".join(
            value.strip()
            for value in (completed.stderr, completed.stdout)
            if value and value.strip()
        )
        raise RuntimeError(
            f"native exact solver exited with {completed.returncode}"
            + (f": {diagnostics}" if diagnostics else " without diagnostics")
        )
    result = json.loads(completed.stdout)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    args = parser.parse_args()
    try:
        request = json.load(sys.stdin)
        json.dump(solve(request, engine=args.engine), sys.stdout, separators=(",", ":"))
        sys.stdout.write("\n")
    except Exception as error:
        print(f"exact solver runner failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
