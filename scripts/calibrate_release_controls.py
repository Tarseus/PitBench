"""Run paired solver-image controls through a task's shared driver and verifier.

This is a calibration experiment, not a new PitBench task type. The manifest
selects releases, data, and budget; solver behavior stays in the repository
plugin and shared driver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

from pitbench.problem_families.base import ProblemFamilyPlugin, ProblemFamilyRegistry
from pitbench.repositories.base import RepositoryPluginRegistry
from pitbench.schema.task import PitBenchTask

ROOT = Path(__file__).resolve().parents[1]


def _source_path(value: str) -> Path:
    if value.startswith("private://"):
        return ROOT / "private" / value.removeprefix("private://")
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _image_id(reference: str) -> str:
    completed = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", reference],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _inputs(manifest: dict[str, Any]) -> tuple[PitBenchTask, list[dict], list[int]]:
    task = PitBenchTask.from_yaml(_source_path(manifest["task_config"]))
    oracle = yaml.safe_load(_source_path(manifest["oracle_path"]).read_text())
    instances = []
    for entry in oracle["instances"]:
        path = _source_path(entry["instance_uri"])
        if not path.is_file():
            raise FileNotFoundError(path)
        instances.append(
            {"id": entry["id"], "path": path, "target": float(entry["bks"])}
        )
    if len(instances) != len({item["id"] for item in instances}):
        raise ValueError("duplicate instance IDs in oracle")
    seed_data = yaml.safe_load(_source_path(manifest["seed_file"]).read_text())
    seeds = list(map(int, seed_data["evaluation_seeds"][: manifest["seed_count"]]))
    if len(seeds) != manifest["seed_count"] or len(seeds) != len(set(seeds)):
        raise ValueError("seed selection is incomplete or duplicated")
    if task.evaluation.threads != 1 or manifest["cpus_per_solver"] != 1:
        raise ValueError("release control requires one thread and one CPU per run")
    if manifest["budget_sec"] not in task.evaluation.budgets_sec:
        raise ValueError("release-control budget is outside the task protocol")
    for label in ("old", "new"):
        reference = manifest[label]["image"]
        if _image_id(reference) != reference:
            raise ValueError(f"{label} image identity does not match manifest")
    return task, instances, seeds


def _run_solver(
    *,
    manifest: dict[str, Any],
    task: PitBenchTask,
    instance: dict[str, Any],
    seed: int,
    round_index: int,
    attempt: int,
    label: str,
    output_dir: Path,
    runner_template: list[str],
) -> dict[str, Any]:
    artifact_dir = (
        output_dir
        / "artifacts"
        / f"round-{round_index}"
        / instance["id"]
        / f"seed-{seed}"
        / f"attempt-{attempt}"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output = artifact_dir / f"{label}.json"
    trajectory = output.with_suffix(".trajectory.jsonl")
    container_output = "/output/" + str(output.relative_to(output_dir))
    container_trajectory = "/output/" + str(trajectory.relative_to(output_dir))
    container_instance = "/instances/" + instance["path"].name
    container_name = (
        f"pitbench-release-{os.getpid()}-{round_index}-{seed}-{label}-{attempt}-"
        + hashlib.sha256(instance["id"].encode()).hexdigest()[:10]
    )
    arguments = [
        token.format_map(
            {
                "instance": container_instance,
                "output": container_output,
                "trajectory": container_trajectory,
                "seed": seed,
                "budget": manifest["budget_sec"],
                "threads": 1,
            }
        )
        for token in runner_template
    ]
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--network",
        "none",
        "--cpus",
        "1",
        "--memory",
        manifest["memory_limit"],
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--mount",
        f"type=bind,src={output_dir / 'harness'},dst=/opt,readonly",
        "--mount",
        f"type=bind,src={instance['path'].parent},dst=/instances,readonly",
        "--mount",
        f"type=bind,src={output_dir},dst=/output",
        "-e",
        "PYTHONPATH=/opt",
        manifest[label]["image"],
        *arguments,
    ]
    started = time.perf_counter()
    try:
        process = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=manifest["budget_sec"] + 120,
        )
        exit_code = process.returncode
        stdout = process.stdout
        stderr = process.stderr
    except subprocess.TimeoutExpired as error:
        exit_code = None
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        subprocess.run(
            ["docker", "container", "rm", "-f", container_name],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        exit_code = None
        stdout = ""
        stderr = str(error)
    if isinstance(stdout, bytes):
        stdout = stdout.decode(errors="replace")
    if isinstance(stderr, bytes):
        stderr = stderr.decode(errors="replace")
    stdout_path = artifact_dir / f"{label}.stdout.log"
    stderr_path = artifact_dir / f"{label}.stderr.log"
    stdout_path.write_text(stdout)
    stderr_path.write_text(stderr)

    record: dict[str, Any] = {
        "version": manifest[label]["version"],
        "commit": manifest[label]["commit"],
        "image": manifest[label]["image"],
        "container_exit_code": exit_code,
        "host_wall_time_sec": time.perf_counter() - started,
        "output_path": str(output),
        "solution_path": str(output.with_suffix(".solution.json")),
        "trajectory_path": str(trajectory),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    if exit_code != 0 or not output.is_file():
        record.update(status="execution_failure", error=stderr[-2000:])
        return record
    try:
        solver = json.loads(output.read_text())
        solution = output.with_suffix(".solution.json")
        if not solution.is_file():
            record.update(status="no_solution", reported=solver)
            return record
        verified = ProblemFamilyRegistry.load(task.problem_family).verify(
            instance["path"], solution
        )
        record.update(
            status="completed" if verified.feasible else "invalid",
            reported_valid=solver.get("valid"),
            reported_objective=solver.get("objective"),
            verified_objective=verified.objective,
            verification_detail=verified.detail,
            normalized_gap=ProblemFamilyPlugin.normalized_gap(
                verified.objective if verified.feasible else None,
                instance["target"],
                objective_sense=task.oracle.objective_sense,
            ),
            solver_cpu_time_sec=solver.get("cpu_time_sec"),
            solver_wall_time_sec=solver.get("wall_time_sec"),
            solver_iterations=solver.get("iterations"),
        )
        if solver.get("valid") is not True:
            record["status"] = "invalid"
    except (OSError, ValueError, KeyError, TypeError) as error:
        record.update(status="output_error", error=str(error))
    return record


def _run_pair(
    *,
    manifest: dict[str, Any],
    task: PitBenchTask,
    instance: dict[str, Any],
    seed: int,
    round_index: int,
    pair_index: int,
    attempt: int,
    output_dir: Path,
    runner_template: list[str],
) -> dict[str, Any]:
    order = ("old", "new") if (pair_index + round_index) % 2 == 0 else ("new", "old")
    runs = {
        label: _run_solver(
            manifest=manifest,
            task=task,
            instance=instance,
            seed=seed,
            round_index=round_index,
            attempt=attempt,
            label=label,
            output_dir=output_dir,
            runner_template=runner_template,
        )
        for label in order
    }
    return {
        "round": round_index,
        "instance_id": instance["id"],
        "instance_path": str(instance["path"]),
        "target": instance["target"],
        "solver_seed": seed,
        "budget_sec": manifest["budget_sec"],
        "order": list(order),
        "attempt": attempt,
        "old": runs["old"],
        "new": runs["new"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="validate inputs only")
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("rounds", 0) < 1:
        raise ValueError("unsupported or empty release-control manifest")
    task, instances, seeds = _inputs(manifest)
    runner = RepositoryPluginRegistry.load(task.repository.plugin).agent_run_template()
    if args.check:
        print(
            json.dumps(
                {
                    "task_id": task.task_id,
                    "instance_count": len(instances),
                    "selected_seeds": seeds,
                    "expected_pairs": len(instances) * len(seeds) * manifest["rounds"],
                    "runner_template": runner,
                },
                indent=2,
            )
        )
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    harness_root = output_dir / "harness" / "pitbench"
    if not harness_root.is_dir():
        shutil.copytree(
            ROOT / "pitbench",
            harness_root,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    if _sha256(harness_root / "solver_drivers/run.py") != _sha256(
        ROOT / "pitbench/solver_drivers/run.py"
    ):
        raise ValueError("frozen driver differs from current run identity")
    identity = {
        "manifest_sha256": _sha256(manifest_path),
        "task_sha256": _sha256(_source_path(manifest["task_config"])),
        "oracle_sha256": _sha256(_source_path(manifest["oracle_path"])),
        "seed_file_sha256": _sha256(_source_path(manifest["seed_file"])),
        "driver_sha256": _sha256(ROOT / "pitbench/solver_drivers/run.py"),
        "verifier_sha256": _sha256(ROOT / "pitbench/problem_families/verification.py"),
        "runner_script_sha256": _sha256(Path(__file__)),
        "instance_count": len(instances),
        "selected_seeds": seeds,
        "expected_pairs": len(instances) * len(seeds) * manifest["rounds"],
    }
    identity_path = output_dir / "identity.json"
    if identity_path.is_file():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("output directory belongs to another protocol identity")
    else:
        identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    record_path = output_dir / "pairs.jsonl"
    previous: dict[tuple[int, str, int], list[dict[str, Any]]] = {}
    if record_path.is_file():
        for line in record_path.read_text().splitlines():
            row = json.loads(line)
            key = (row["round"], row["instance_id"], row["solver_seed"])
            previous.setdefault(key, []).append(row)
    jobs = []
    for round_index in range(manifest["rounds"]):
        for instance_index, instance in enumerate(instances):
            for seed_index, seed in enumerate(seeds):
                key = (round_index, instance["id"], seed)
                if any(
                    all(row[label]["status"] == "completed" for label in ("old", "new"))
                    for row in previous.get(key, [])
                ):
                    continue
                jobs.append(
                    {
                        "manifest": manifest,
                        "task": task,
                        "instance": instance,
                        "seed": seed,
                        "round_index": round_index,
                        "pair_index": instance_index * len(seeds) + seed_index,
                        "attempt": len(previous.get(key, [])) + 1,
                        "output_dir": output_dir,
                        "runner_template": runner,
                    }
                )
    print(
        json.dumps(
            {
                "expected_pairs": identity["expected_pairs"],
                "completed_prior_pairs": identity["expected_pairs"] - len(jobs),
                "remaining_pairs": len(jobs),
                "parallel_pairs": manifest["parallel_pairs"],
            }
        ),
        flush=True,
    )
    finished = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=manifest["parallel_pairs"]) as pool:
        futures = {pool.submit(_run_pair, **job): job for job in jobs}
        with record_path.open("a") as stream:
            for future in as_completed(futures):
                try:
                    row = future.result()
                except Exception as error:
                    job = futures[future]
                    row = {
                        "round": job["round_index"],
                        "instance_id": job["instance"]["id"],
                        "solver_seed": job["seed"],
                        "budget_sec": manifest["budget_sec"],
                        "attempt": job["attempt"],
                        "old": {"status": "orchestration_failure", "error": str(error)},
                        "new": {"status": "orchestration_failure", "error": str(error)},
                    }
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                finished += 1
                if any(row[label]["status"] != "completed" for label in ("old", "new")):
                    failed += 1
                if finished % 10 == 0 or finished == len(jobs):
                    print(
                        json.dumps(
                            {
                                "finished_pairs": finished,
                                "remaining_pairs": len(jobs) - finished,
                                "failed_pairs": failed,
                            }
                        ),
                        flush=True,
                    )
    if failed:
        raise SystemExit(f"{failed} release-control pairs failed; raw records retained")


if __name__ == "__main__":
    main()
