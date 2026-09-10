from __future__ import annotations

import argparse
import json
import resource
import subprocess
import time
from pathlib import Path
from typing import Any


def process_resources(*, child_process: bool = False) -> dict[str, Any]:
    """Snapshot one solver process, before result parsing or verification.

    The default measures the fresh Python worker through solver return. The
    child mode is for a fresh driver that has run exactly one native solver
    subprocess: its RSS is that subprocess's lifetime peak, not the maximum
    of unrelated driver/child peaks or a process-tree memory measurement.
    """
    usage = resource.getrusage(
        resource.RUSAGE_CHILDREN if child_process else resource.RUSAGE_SELF
    )
    return {
        "cpu_time_sec": usage.ru_utime + usage.ru_stime,
        "peak_rss_bytes": int(usage.ru_maxrss * 1024),
        "resource_scope": (
            "single_native_child_lifetime"
            if child_process
            else "worker_process_start_through_solver_return"
        ),
    }


def parser(*, solver: bool = False, trajectory: bool = True) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    if solver:
        result.add_argument("--solver", required=True)
    result.add_argument("--instance", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    if trajectory:
        result.add_argument("--trajectory", type=Path, required=True)
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--budget", type=float, required=True)
    result.add_argument("--threads", type=int, required=True)
    return result


def write_result(
    path: Path,
    *,
    started: float,
    valid: bool,
    objective: float | None = None,
    error: str | None = None,
    **metrics: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "valid": valid,
                "objective": objective,
                "wall_time_sec": time.perf_counter() - started,
                "error": error,
                **metrics,
            },
            indent=2,
        )
    )


def write_solution(output: Path, payload: dict[str, Any]) -> Path:
    path = output.with_suffix(".solution.json")
    path.write_text(json.dumps(payload, indent=2))
    return path


def failure_reason(error: Exception) -> str | None:
    """Preserve explicit failures across a driver subprocess boundary."""
    if isinstance(error, subprocess.TimeoutExpired):
        return "timed_out"
    if isinstance(error, MemoryError):
        return "out_of_memory"
    return None


def append_trajectory(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload))
        handle.write("\n")
