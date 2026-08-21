from __future__ import annotations

import json
import math
import statistics
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from pitbench.families.cvrp import CVRPFamily
from pitbench.instances.cvrp_axis_spec import build_exact_single_qoi_panel
from pitbench.metrics.paired_response import (
    IncumbentPoint,
    anytime_outcomes,
    clustered_bootstrap_mean_difference,
)
from pitbench.qoi.cvrp import extract_cvrp_instance_qoi


@dataclass(frozen=True)
class SingleQoIPilotReport:
    schema_version: str
    kind: str
    conclusion_scope: str
    solver: dict[str, Any]
    design: dict[str, Any]
    axis_effects: dict[str, dict[str, Any]]
    reference_runs: tuple[dict[str, Any], ...]
    runs: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _write_vrplib(instance: dict[str, Any], path: Path) -> None:
    coordinates = instance["coordinates"]
    demands = instance["demands"]
    lines = [
        f"NAME : {instance.get('name', path.stem)}",
        "TYPE : CVRP",
        f"DIMENSION : {len(coordinates)}",
        "EDGE_WEIGHT_TYPE : EUC_2D",
        f"CAPACITY : {instance['capacity']}",
        "NODE_COORD_SECTION",
    ]
    lines.extend(
        f"{index + 1} {coordinate[0]} {coordinate[1]}"
        for index, coordinate in enumerate(coordinates)
    )
    lines.append("DEMAND_SECTION")
    lines.extend(f"{index + 1} {demand}" for index, demand in enumerate(demands))
    lines.extend(("DEPOT_SECTION", "1", "-1", "EOF"))
    path.write_text("\n".join(lines) + "\n")


def _trajectory(result: Any) -> tuple[IncumbentPoint, ...]:
    elapsed = 0.0
    incumbent = math.inf
    points = []
    for runtime, datum in zip(result.stats.runtimes, result.stats, strict=True):
        elapsed += runtime
        if datum.best_feas and datum.best_cost < incumbent:
            incumbent = float(datum.best_cost)
            if not points or elapsed > points[-1].time_sec:
                points.append(IncumbentPoint(elapsed, incumbent))
    final = float(result.cost())
    final_time = float(result.runtime)
    if final < incumbent and (not points or final_time > points[-1].time_sec):
        points.append(IncumbentPoint(final_time, final))
    return tuple(points)


def _solve(
    instance: dict[str, Any], *, solver_seed: int, budget_sec: float
) -> dict[str, Any]:
    from pyvrp import Model, read
    from pyvrp.stop import MaxRuntime

    started = time.perf_counter()
    cpu_started = time.process_time()
    with tempfile.TemporaryDirectory(prefix="pitbench-single-qoi-") as temporary:
        root = Path(temporary)
        vrp_path = root / "instance.vrp"
        instance_path = root / "instance.json"
        solution_path = root / "solution.json"
        _write_vrplib(instance, vrp_path)
        data = read(vrp_path, round_func="round")
        result = Model.from_data(data).solve(
            stop=MaxRuntime(budget_sec), seed=solver_seed, display=False
        )
        routes = [list(map(int, route.visits())) for route in result.best.routes()]
        instance_path.write_text(json.dumps(instance))
        solution_path.write_text(json.dumps({"routes": routes}))
        verification = CVRPFamily().verify(instance_path, solution_path)
    wall_time = time.perf_counter() - started
    cpu_time = time.process_time() - cpu_started
    return {
        "feasible": verification.feasible,
        "verification_detail": verification.detail,
        "solver_objective": float(result.cost()) if verification.feasible else None,
        "verified_objective": verification.objective,
        "wall_time_sec": wall_time,
        "cpu_time_sec": cpu_time,
        "iterations": int(result.num_iterations),
        "trajectory": [asdict(point) for point in _trajectory(result)],
    }


def _mean_ci(
    differences: dict[str, list[float]], *, bootstrap_repetitions: int
) -> dict[str, float | int]:
    values = [value for cluster in differences.values() for value in cluster]
    lower, upper = clustered_bootstrap_mean_difference(
        differences, repetitions=bootstrap_repetitions, seed=0
    )
    return {
        "paired_run_count": len(values),
        "generator_seed_clusters": len(differences),
        "target_minus_source_mean": statistics.fmean(values),
        "cluster_bootstrap_95_lower": lower,
        "cluster_bootstrap_95_upper": upper,
    }


def run_pyvrp_single_qoi_pilot(
    *,
    generator_seeds: tuple[int, ...] = (101, 211, 307, 401, 503, 601),
    solver_seeds: tuple[int, ...] = (0, 1, 2),
    budget_sec: float = 0.5,
    reference_budget_sec: float = 2.0,
    reference_solver_seed: int = 97,
    bootstrap_repetitions: int = 2_000,
) -> SingleQoIPilotReport:
    if (
        budget_sec <= 0
        or reference_budget_sec <= budget_sec
        or not solver_seeds
        or len(set(solver_seeds)) != len(solver_seeds)
    ):
        raise ValueError(
            "reference budget must exceed a positive evaluation budget; solver "
            "seeds must be unique and non-empty"
        )
    cases = build_exact_single_qoi_panel(generator_seeds)
    reference_runs = []
    references = {}
    for case in cases:
        for level, instance in (("source", case.source), ("target", case.target)):
            solved = _solve(
                instance,
                solver_seed=reference_solver_seed,
                budget_sec=reference_budget_sec,
            )
            reference_runs.append(
                {
                    "target_qoi": case.target_qoi,
                    "pair_id": case.pair_id,
                    "generator_seed": case.generator_seed,
                    "solver_seed": reference_solver_seed,
                    "level": level,
                    "budget_sec": reference_budget_sec,
                    **solved,
                }
            )
            if solved["feasible"]:
                references[(case.pair_id, level)] = solved["solver_objective"]

    raw_runs = []
    for case in cases:
        for level, instance in (("source", case.source), ("target", case.target)):
            qoi = extract_cvrp_instance_qoi(instance, spec_version="1.0").values
            for solver_seed in solver_seeds:
                raw_runs.append(
                    {
                        "target_qoi": case.target_qoi,
                        "pair_id": case.pair_id,
                        "generator_seed": case.generator_seed,
                        "solver_seed": solver_seed,
                        "level": level,
                        "qoi_value": qoi[case.target_qoi],
                        "pairwise_distance_median": qoi["pairwise_distance_median"],
                        **_solve(
                            instance,
                            solver_seed=solver_seed,
                            budget_sec=budget_sec,
                        ),
                    }
                )

    for run in raw_runs:
        if run["feasible"]:
            key = (run["pair_id"], run["level"])
            objective = float(run["solver_objective"])
            references[key] = min(references.get(key, math.inf), objective)

    runs = []
    for run in raw_runs:
        reference = references.get((run["pair_id"], run["level"]))
        trajectory = tuple(IncumbentPoint(**point) for point in run["trajectory"])
        outcomes = (
            anytime_outcomes(
                trajectory,
                reference=reference,
                budget_sec=budget_sec,
                target_gap=0.01,
            )
            if reference is not None
            else None
        )
        runs.append(
            {
                **run,
                "best_observed_reference": reference,
                "best_observed_gap": (
                    outcomes.reference_gap if outcomes is not None else None
                ),
                "primal_integral": (
                    outcomes.primal_integral if outcomes is not None else None
                ),
                "time_to_target_or_budget_sec": (
                    outcomes.time_to_target_sec
                    if outcomes is not None and outcomes.time_to_target_sec is not None
                    else budget_sec
                ),
                "target_hit": outcomes.target_hit if outcomes is not None else False,
                "normalized_solver_objective": (
                    float(run["solver_objective"])
                    / float(run["pairwise_distance_median"])
                    if run["feasible"]
                    else None
                ),
            }
        )

    by_key = {(run["pair_id"], run["solver_seed"], run["level"]): run for run in runs}
    metrics = (
        "feasible",
        "best_observed_gap",
        "primal_integral",
        "time_to_target_or_budget_sec",
        "target_hit",
        "normalized_solver_objective",
        "wall_time_sec",
        "cpu_time_sec",
        "iterations",
    )
    effects = {}
    for axis in sorted({case.target_qoi for case in cases}):
        axis_cases = [case for case in cases if case.target_qoi == axis]
        metric_effects = {}
        for metric in metrics:
            differences: dict[str, list[float]] = defaultdict(list)
            for case in axis_cases:
                for solver_seed in solver_seeds:
                    source = by_key[(case.pair_id, solver_seed, "source")][metric]
                    target = by_key[(case.pair_id, solver_seed, "target")][metric]
                    if source is not None and target is not None:
                        differences[str(case.generator_seed)].append(
                            float(target) - float(source)
                        )
            metric_effects[metric] = _mean_ci(
                differences, bootstrap_repetitions=bootstrap_repetitions
            )
        effects[axis] = {
            "estimand": "matched target-minus-source mean",
            "qoi_delta_by_generator_seed": {
                str(case.generator_seed): case.realized_qoi_delta for case in axis_cases
            },
            "metrics": metric_effects,
        }

    return SingleQoIPilotReport(
        schema_version="1.0",
        kind="cvrp_exact_single_qoi_solver_pilot",
        conclusion_scope=(
            "Identification pilot on exact synthetic fixtures; not evidence for a "
            "unified geometry or for compound QoIs."
        ),
        solver={"name": "PyVRP", "version": version("pyvrp"), "threads": 1},
        design={
            "qoi_spec_version": "1.0",
            "generator_seeds": list(generator_seeds),
            "solver_seeds": list(solver_seeds),
            "budget_sec": budget_sec,
            "reference_budget_sec": reference_budget_sec,
            "reference_solver_seed": reference_solver_seed,
            "common_random_numbers": True,
            "matched_solver_seeds": True,
            "reference_policy": (
                "minimum of the independent longer-budget reference and fixed-budget "
                "evaluation runs, per pair level"
            ),
            "bootstrap_repetitions": bootstrap_repetitions,
            "peak_rss": "not available from the in-process Windows pilot",
        },
        axis_effects=effects,
        reference_runs=tuple(reference_runs),
        runs=tuple(runs),
    )


__all__ = ["SingleQoIPilotReport", "run_pyvrp_single_qoi_pilot"]
