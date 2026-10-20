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
        raise ValueError("Gecode bin-packing tasks require one thread")
    instance = json.loads(Path(request["instance_path"]).read_text())
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
    parameters = request.get("parameters") or {}
    level = parameters.get("gecode.bin_packing_propagation_level", "full")
    if level not in {"full", "advanced", "basic", "default"}:
        raise ValueError(f"unsupported Gecode propagation level: {level!r}")
    executable = os.environ.get("PITBENCH_EXACT_SOLVER", "build/pitbench-exact")
    argv = [
        executable,
        str(max(1, round(float(request["budget_sec"]) * 1000))),
        level,
        str(capacity),
        str(len(weights)),
        *(str(weight) for weight in weights),
    ]
    completed = subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        timeout=float(request["budget_sec"]) + 10,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "native exact solver failed")
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
