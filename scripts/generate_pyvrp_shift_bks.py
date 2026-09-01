from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from pitbench.families.cvrp import CVRPFamily
from pitbench.instances.generate import materialize_generated_instance_set


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _solver_version(python: Path) -> str:
    completed = subprocess.run(
        [
            str(python),
            "-c",
            "import importlib.metadata; print(importlib.metadata.version('pyvrp'))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _solve(
    solver_python: Path,
    instance: Path,
    run_root: Path,
    seed: int,
    budget_sec: float,
) -> tuple[int, Path, Path]:
    run_dir = run_root / instance.stem / f"seed-{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / "result.json"
    trajectory = run_dir / "trajectory.jsonl"
    subprocess.run(
        [
            str(solver_python),
            "-m",
            "pitbench.drivers.pyvrp",
            "--instance",
            str(instance),
            "--output",
            str(output),
            "--trajectory",
            str(trajectory),
            "--seed",
            str(seed),
            "--budget",
            str(budget_sec),
            "--threads",
            "1",
        ],
        check=True,
        timeout=budget_sec + 120,
    )
    return seed, output, output.with_suffix(".solution.json")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate independently verified empirical BKS anchors."
    )
    parser.add_argument("--instance-set-config", type=Path, required=True)
    parser.add_argument("--solver-python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget-sec", type=float, default=30.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--workers", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.budget_sec <= 0 or args.workers <= 0:
        raise ValueError("budget and worker count must be positive")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be non-empty and unique")

    payload = yaml.safe_load(args.instance_set_config.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    verifier = CVRPFamily()
    solver_version = _solver_version(args.solver_python)

    with tempfile.TemporaryDirectory(prefix="pitbench-shift-bks-") as temporary:
        root = Path(temporary)
        instances = materialize_generated_instance_set(
            payload,
            root / "instances",
            expected_visibility="judge",
            stem_prefix="judge_shift",
        )
        futures = {}
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for instance in instances:
                for seed in args.seeds:
                    future = executor.submit(
                        _solve,
                        args.solver_python,
                        instance,
                        root / "runs",
                        seed,
                        args.budget_sec,
                    )
                    futures[future] = (instance, seed)

            results: dict[str, list[dict[str, Any]]] = {
                instance.stem: [] for instance in instances
            }
            for future in as_completed(futures):
                instance, expected_seed = futures[future]
                seed, output, solution = future.result()
                if seed != expected_seed:
                    raise RuntimeError("solver result seed mismatch")
                run_payload = json.loads(output.read_text())
                verified = verifier.verify(instance, solution)
                if not verified.feasible or verified.objective is None:
                    raise RuntimeError(
                        f"invalid BKS candidate for {instance.stem} seed {seed}: "
                        f"{verified.detail}"
                    )
                if verified.objective != run_payload["objective"]:
                    raise RuntimeError(
                        f"objective mismatch for {instance.stem} seed {seed}"
                    )
                results[instance.stem].append(
                    {
                        "seed": seed,
                        "objective": verified.objective,
                        "solution": solution,
                    }
                )

        anchors = []
        for instance in instances:
            candidates = sorted(
                results[instance.stem], key=lambda row: (row["objective"], row["seed"])
            )
            best = candidates[0]
            destination = args.output_dir / f"{instance.stem}.bks.solution.json"
            destination.write_bytes(best["solution"].read_bytes())
            anchors.append(
                {
                    "id": instance.stem,
                    "bks": best["objective"],
                    "bks_solution_uri": (
                        "private://oracles/pyvrp_cvrp_shift_v1/"
                        f"{destination.name}"
                    ),
                    "instance_sha256": _sha256(instance),
                    "bks_solution_sha256": _sha256(destination),
                    "winning_seed": best["seed"],
                    "verified_objectives": {
                        str(row["seed"]): row["objective"] for row in candidates
                    },
                }
            )

    oracle = {
        "schema_version": "1.0",
        "kind": "empirical_best_known_solution",
        "problem_family": "cvrp",
        "objective_sense": "minimize",
        "distance_metric": "EUC_2D",
        "generated_at": datetime.now(UTC).isoformat(),
        "instance_set_config_sha256": _sha256(args.instance_set_config),
        "solver": {"name": "PyVRP", "version": solver_version},
        "protocol": {
            "budget_sec": args.budget_sec,
            "seeds": args.seeds,
            "workers": args.workers,
            "independent_verifier": "pitbench.families.cvrp:CVRPFamily",
        },
        "anchors": anchors,
    }
    oracle_path = args.output_dir / "oracle.yaml"
    oracle_path.write_text(yaml.safe_dump(oracle, sort_keys=False))
    print(oracle_path)


if __name__ == "__main__":
    main()
