from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from pitbench.solver_drivers.common import (
    append_trajectory,
    parser,
    write_result,
    write_solution,
)


def _route_visits(route: object) -> list[int]:
    """Return zero-based CVRP node ids across PyVRP route API generations."""

    legacy_visits = getattr(route, "visits", None)
    if callable(legacy_visits):
        return list(map(int, legacy_visits()))

    # PyVRP v0.14 represents routes as scheduled activities. For the
    # single-depot VRPLIB files emitted below, client activity index zero maps
    # to normalized CVRP node one.
    return [
        int(activity.idx) + 1
        for activity in route  # type: ignore[union-attr]
        if activity.is_client()
    ]


def _statistics_rows(stats: object):
    """Yield runtime, feasibility, and incumbent across PyVRP statistics APIs."""

    runtimes = stats.runtimes  # type: ignore[attr-defined]
    legacy_feasible = getattr(stats, "feas_stats", None)
    if legacy_feasible is not None:
        for runtime, datum in zip(runtimes, legacy_feasible, strict=True):
            yield runtime, datum.size > 0, datum.best_cost
        return

    for runtime, datum in zip(runtimes, stats, strict=True):
        yield runtime, datum.best_feas, datum.best_cost


def _vrplib(instance: dict, path: Path) -> None:
    coordinates = instance["coordinates"]
    demands = instance["demands"]
    distance_metric = instance.get("distance_metric", "EUC_2D")
    if distance_metric != "EUC_2D":
        raise ValueError(
            "PyVRP driver only supports normalized instances with EUC_2D "
            f"distance semantics, got {distance_metric!r}"
        )
    lines = [
        f"NAME : {instance.get('name', path.stem)}",
        "TYPE : CVRP",
        f"DIMENSION : {len(coordinates)}",
        f"EDGE_WEIGHT_TYPE : {distance_metric}",
        f"CAPACITY : {instance['capacity']}",
        "NODE_COORD_SECTION",
    ]
    lines.extend(
        f"{index + 1} {coordinate[0]} {coordinate[1]}"
        for index, coordinate in enumerate(coordinates)
    )
    lines.append("DEMAND_SECTION")
    lines.extend(f"{index + 1} {demand}" for index, demand in enumerate(demands))
    lines.extend(["DEPOT_SECTION", "1", "-1", "EOF"])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parser().parse_args()
    started = time.perf_counter()
    instance = json.loads(args.instance.read_text())
    args.trajectory.parent.mkdir(parents=True, exist_ok=True)
    args.trajectory.write_text("")
    try:
        from pyvrp import Model, read
        from pyvrp.stop import MaxRuntime

        with tempfile.TemporaryDirectory(prefix="pitbench-pyvrp-") as temporary:
            vrp = Path(temporary) / "instance.vrp"
            _vrplib(instance, vrp)
            data = read(vrp, round_func="round")
            result = Model.from_data(data).solve(
                stop=MaxRuntime(args.budget), seed=args.seed, display=False
            )
        routes = [_route_visits(route) for route in result.best.routes()]
        objective = float(result.cost())
        write_solution(args.output, {"routes": routes})
        elapsed = 0.0
        incumbent = float("inf")
        for runtime, feasible, best_cost in _statistics_rows(result.stats):
            elapsed += runtime
            if feasible and best_cost < incumbent:
                incumbent = float(best_cost)
                append_trajectory(
                    args.trajectory,
                    {"time_sec": elapsed, "objective": incumbent},
                )
        if incumbent != objective:
            append_trajectory(
                args.trajectory,
                {"time_sec": result.runtime, "objective": objective},
            )
        write_result(
            args.output,
            started=started,
            valid=True,
            objective=objective,
            iterations=result.num_iterations,
            solver_runtime_sec=result.runtime,
        )
    except Exception as exc:
        write_result(args.output, started=started, valid=False, error=str(exc))
        raise


if __name__ == "__main__":
    main()
