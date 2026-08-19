from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


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


def append_trajectory(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload))
        handle.write("\n")
