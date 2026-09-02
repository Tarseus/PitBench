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
) -> None:
    """Run an immutable judge-image adapter with a JSON stdin/stdout contract."""
    started = time.perf_counter()
    configured = os.environ.get(environment_key)
    try:
        if not configured:
            raise RuntimeError(
                f"judge image does not provide required {environment_key} runner"
            )
        request: dict[str, Any] = {
            "instance_path": str(instance),
            "solver_seed": seed,
            "budget_sec": budget,
            "threads": threads,
        }
        completed = subprocess.run(
            shlex.split(configured),
            input=json.dumps(request),
            check=False,
            capture_output=True,
            text=True,
            timeout=budget + 60,
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip() or "external runner failed")
        response = json.loads(completed.stdout)
        solution = response.pop("solution", {})
        write_solution(output, solution)
        if trajectory is not None:
            append_trajectory(
                trajectory,
                {
                    "time_sec": time.perf_counter() - started,
                    "objective": response.get("objective"),
                },
            )
        write_result(output, started=started, **response)
    except Exception as exc:
        write_result(output, started=started, valid=False, error=str(exc))
        raise
