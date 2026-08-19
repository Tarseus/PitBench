from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from pitbench.drivers.common import (
    append_trajectory,
    parser,
    write_result,
    write_solution,
)


def _vrplib(instance: dict, path: Path) -> None:
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
    lines.extend(["DEPOT_SECTION", "1", "-1", "EOF"])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parser().parse_args()
    started = time.perf_counter()
    instance = json.loads(args.instance.read_text())
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
        routes = [list(map(int, route.visits())) for route in result.best.routes()]
        objective = float(result.cost())
        write_solution(args.output, {"routes": routes})
        append_trajectory(
            args.trajectory,
            {"time_sec": time.perf_counter() - started, "objective": objective},
        )
        write_result(args.output, started=started, valid=True, objective=objective)
    except Exception as exc:
        write_result(args.output, started=started, valid=False, error=str(exc))
        raise


if __name__ == "__main__":
    main()
