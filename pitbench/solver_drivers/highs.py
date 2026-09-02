from __future__ import annotations

import re
import subprocess
import tempfile
import time
from pathlib import Path

from pitbench.solver_drivers.common import (
    append_trajectory,
    parser,
    write_result,
    write_solution,
)

_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _match(pattern: str, text: str) -> float | None:
    found = re.search(pattern, text, re.IGNORECASE)
    return float(found.group(1)) if found else None


def main() -> None:
    args = parser(solver=True).parse_args()
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="pitbench-highs-") as temporary:
            raw_solution = Path(temporary) / "solution.txt"
            completed = subprocess.run(
                [
                    args.solver,
                    str(args.instance),
                    f"--time_limit={args.budget}",
                    f"--random_seed={args.seed}",
                    f"--threads={args.threads}",
                    f"--solution_file={raw_solution}",
                    "--write_solution_to_file=true",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=args.budget + 60,
            )
            raw = (
                raw_solution.read_text(errors="replace")
                if raw_solution.exists()
                else ""
            )
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip() or "HiGHS failed")
        log = completed.stdout + "\n" + completed.stderr
        objective = _match(rf"Primal bound\s+({_FLOAT})", log)
        dual = _match(rf"Dual bound\s+({_FLOAT})", log)
        nodes = _match(r"Nodes\s+(\d+)", log)
        write_solution(args.output, {"raw_solution": raw, "objective": objective})
        append_trajectory(
            args.trajectory,
            {"time_sec": time.perf_counter() - started, "objective": objective},
        )
        write_result(
            args.output,
            started=started,
            valid=objective is not None,
            objective=objective,
            primal_bound=objective,
            dual_bound=dual,
            nodes=int(nodes) if nodes is not None else None,
        )
    except Exception as exc:
        write_result(args.output, started=started, valid=False, error=str(exc))
        raise


if __name__ == "__main__":
    main()
