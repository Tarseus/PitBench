from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from pitbench.solver_drivers.common import (
    append_trajectory,
    process_resources,
    write_result,
    write_solution,
)


def execute(
    *,
    environment_key: str,
    instance: Path,
    output: Path,
    trajectory: Path | None,
    seed: int,
    budget: float,
    threads: int,
    parameters: dict[str, Any] | None = None,
) -> None:
    """Run an immutable judge-image adapter with a JSON stdin/stdout contract."""
    started = time.perf_counter()
    configured = os.environ.get(environment_key)
    try:
        if trajectory is not None:
            trajectory.unlink(missing_ok=True)
        if not configured:
            raise RuntimeError(
                f"judge image does not provide required {environment_key} runner"
            )
        request: dict[str, Any] = {
            "instance_path": str(instance),
            "solver_seed": seed,
            "budget_sec": budget,
            "threads": threads,
            "parameters": parameters,
        }
        completed = subprocess.run(
            shlex.split(configured),
            input=json.dumps(request),
            check=False,
            capture_output=True,
            text=True,
            timeout=budget + 60,
        )
        resources = process_resources(child_process=True)
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip() or "external runner failed")
        response = json.loads(completed.stdout)
        solution = response.pop("solution", {})
        events = response.pop("trajectory", None)
        solution_path = output.with_suffix(".solution.json")
        if response.get("has_solution") is False:
            solution_path.unlink(missing_ok=True)
        else:
            write_solution(output, solution)
        if trajectory is not None:
            if events is None:
                events = [
                    {
                        "time_sec": time.perf_counter() - started,
                        "objective": response.get("objective"),
                    }
                ]
            for event in events:
                append_trajectory(trajectory, event)
        write_result(
            output,
            started=started,
            **{**resources, **response},
        )
    except Exception as exc:
        write_result(output, started=started, valid=False, error=str(exc))
        raise
