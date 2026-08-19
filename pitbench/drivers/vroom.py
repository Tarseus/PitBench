from __future__ import annotations

import json
import math
import subprocess
import tempfile
import time
from pathlib import Path

from pitbench.drivers.common import (
    append_trajectory,
    parser,
    write_result,
    write_solution,
)


def _request(instance: dict) -> dict:
    coordinates = instance["coordinates"]
    matrix = [
        [round(math.hypot(a[0] - b[0], a[1] - b[1]) * 1000) for b in coordinates]
        for a in coordinates
    ]
    total = sum(instance["demands"])
    capacity = int(instance["capacity"])
    vehicles = max(1, math.ceil(total / capacity) + 1)
    return {
        "matrix": matrix,
        "jobs": [
            {"id": index, "location_index": index, "amount": [int(demand)]}
            for index, demand in enumerate(instance["demands"])
            if index != int(instance.get("depot", 0))
        ],
        "vehicles": [
            {
                "id": index,
                "start_index": int(instance.get("depot", 0)),
                "end_index": int(instance.get("depot", 0)),
                "capacity": [capacity],
            }
            for index in range(vehicles)
        ],
    }


def main() -> None:
    args = parser(solver=True).parse_args()
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="pitbench-vroom-") as temporary:
            request = Path(temporary) / "request.json"
            request.write_text(
                json.dumps(_request(json.loads(args.instance.read_text())))
            )
            completed = subprocess.run(
                [args.solver, "-i", str(request), "-t", str(args.threads)],
                check=False,
                capture_output=True,
                text=True,
                timeout=args.budget + 30,
            )
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip() or "vroom failed")
        response = json.loads(completed.stdout)
        routes = [
            [step["id"] for step in route["steps"] if step.get("type") == "job"]
            for route in response.get("routes", [])
        ]
        objective = float(response.get("summary", {}).get("cost", 0))
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
