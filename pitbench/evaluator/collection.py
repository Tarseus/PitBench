"""Approved raw-result collection protocols, separated by solver backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pitbench.schema.observation import RunObservation

ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


class HighsCollection:
    """Collect the fixed HiGHS seed, row/column permutation and control panel."""

    SOURCE_COMMIT = "04024d701f79feb8e2f18bc3df0dffc04ef05088"
    SOLVER_VERSION = "1.15.1"

    @staticmethod
    def solver_identity() -> dict:
        import highspy

        solver = highspy.Highs()
        if (
            solver.version() != HighsCollection.SOLVER_VERSION
            or not HighsCollection.SOURCE_COMMIT.startswith(solver.githash())
        ):
            raise ValueError(
                "experiment requires the pinned HiGHS release and source revision"
            )
        return {
            "version": solver.version(),
            "binding_git_hash": solver.githash(),
            "source_commit": HighsCollection.SOURCE_COMMIT,
        }

    @staticmethod
    def prepare(output: Path, names: list[str]) -> None:
        import highspy
        import numpy as np
        import scipy

        from pitbench.evaluator.representations import LinearModelRepresentation

        manifest_path = output / "experiment.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if names != [item["name"] for item in manifest["instances"]]:
                raise ValueError("existing experiment has a different instance panel")
            if manifest["solver"] != HighsCollection.solver_identity():
                raise ValueError("solver changed")
            print("Existing fixed experiment retained.", flush=True)
            return
        if len(names) != 10 or len(set(names)) != 10:
            raise ValueError("this experiment requires ten distinct instances")
        generator = np.random.default_rng(20260909)
        seeds = random.SystemRandom().sample(range(0, 2**31), 30)
        defaults = highspy.Highs()
        options = {
            "threads": 1,
            "parallel": "off",
            "presolve": defaults.getOptionValue("presolve")[1],
            "mip_feasibility_tolerance": defaults.getOptionValue(
                "mip_feasibility_tolerance"
            )[1],
            "mip_rel_gap": defaults.getOptionValue("mip_rel_gap")[1],
            "mip_abs_gap": defaults.getOptionValue("mip_abs_gap")[1],
        }
        instances = []
        jobs = []
        for name in names:
            source = output / "sources" / f"{name}.mps.gz"
            model = LinearModelRepresentation.read(source)
            directory = output / "models" / name
            directory.mkdir(parents=True, exist_ok=True)
            LinearModelRepresentation.save(directory / "original.npz", model)
            instances.append(
                {
                    "name": name,
                    "source": str(source.relative_to(output)),
                    "source_url": f"https://miplib.zib.de/WebData/instances/{name}.mps.gz",
                    "details_url": f"https://miplib.zib.de/instance_details_{name}.html",
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "rows": model["matrix"].shape[0],
                    "columns": model["matrix"].shape[1],
                    "nonzeros": model["matrix"].nnz,
                    "integer_columns": int(np.count_nonzero(model["integrality"])),
                    "objective_sense": model["sense"],
                    "objective_offset": model["offset"],
                }
            )
            seen = set()
            for index in range(30):
                while True:
                    rows = generator.permutation(model["matrix"].shape[0])
                    columns = generator.permutation(model["matrix"].shape[1])
                    signature = (tuple(rows), tuple(columns))
                    if signature not in seen and not (
                        np.array_equal(rows, np.arange(len(rows)))
                        and np.array_equal(columns, np.arange(len(columns)))
                    ):
                        seen.add(signature)
                        break
                transformed = LinearModelRepresentation.permute(model, rows, columns)
                LinearModelRepresentation.assert_equivalent(
                    model, transformed, rows, columns
                )
                LinearModelRepresentation.save(
                    directory / f"permutation-{index:02d}.npz", transformed
                )
                np.savez_compressed(
                    directory / f"mapping-{index:02d}.npz", rows=rows, columns=columns
                )
            for axis, values in (
                ("seed", seeds),
                ("representation", list(range(30))),
                ("control", list(range(3))),
            ):
                for index, value in enumerate(values):
                    for budget in (10, 30):
                        jobs.append(
                            {
                                "run_id": f"{name}/{axis}/{index:02d}/budget-{budget}",
                                "instance": name,
                                "axis": axis,
                                "replicate": index,
                                "solver_seed": value if axis == "seed" else 0,
                                "permutation": value
                                if axis == "representation"
                                else None,
                                "budget_sec": budget,
                            }
                        )
        random.Random(20260909).shuffle(jobs)
        manifest = {
            "experiment": "original HiGHS nuisance exploration",
            "solver": HighsCollection.solver_identity(),
            "code_state": "original release, no patch",
            "budgets_sec": [10, 30],
            "solver_options": options,
            "solver_seeds": seeds,
            "seed_sampling": "30 distinct IDs uniformly sampled from [0, 2^31-1] and retained",
            "representation_solver_seed": 0,
            "permutation_generator": "numpy.default_rng(20260909); stored mappings are authoritative",
            "permutations_per_instance": 30,
            "control_repeats_per_instance_budget": 3,
            "model_input": "HiGHS parses original MPS; all runs pass preserved or permuted numeric models through the same API",
            "equivalence_checks": "all 300 matrices, vectors, names, types, bounds, sense and offset restore exactly",
            "verification": "independent sparse arithmetic on original model; shared HiGHS input parser; no independent dual/optimality certificate check",
            "statistics": "deferred; retain all runs without imputation or replacement",
            "timing": "fresh process per run; budget applies to Highs.run; input loading and independent verification timed separately",
            "trajectory": "native MIP logging and improving-incumbent events plus final state; not a continuous trace",
            "environment": {
                "python": sys.version,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "platform": platform.platform(),
                "cpu_model": platform.processor(),
            },
            "instances": instances,
            "jobs": jobs,
        }
        for path in (Path(__file__), ROOT / "pitbench/evaluator/representations.py"):
            target = output / "collector_source" / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
        write_json(manifest_path, manifest)
        print(
            f"Prepared {len(instances)} instances, 300 equivalent models, {len(jobs)} runs.",
            flush=True,
        )

    @staticmethod
    def worker(output: Path, run_id: str, cpu: int) -> None:
        os.sched_setaffinity(0, {cpu})
        started = time.perf_counter()
        import highspy
        import numpy as np

        from pitbench.evaluator.representations import LinearModelRepresentation

        manifest = json.loads((output / "experiment.json").read_text())
        job = next(item for item in manifest["jobs"] if item["run_id"] == run_id)
        directory = output / "runs" / run_id
        directory.mkdir(parents=True, exist_ok=True)
        result = {
            **job,
            "cpu": cpu,
            "pid": os.getpid(),
            "started_unix": time.time(),
            "solver": HighsCollection.solver_identity(),
        }
        try:
            original = LinearModelRepresentation.load(
                output / "models" / job["instance"] / "original.npz"
            )
            if job["permutation"] is None:
                model = original
                columns = np.arange(len(original["cost"]))
            else:
                model_root = output / "models" / job["instance"]
                model = LinearModelRepresentation.load(
                    model_root / f"permutation-{job['permutation']:02d}.npz"
                )
                with np.load(
                    model_root / f"mapping-{job['permutation']:02d}.npz"
                ) as mapping:
                    columns = mapping["columns"]
            solver = highspy.Highs()
            options = {
                **manifest["solver_options"],
                "random_seed": job["solver_seed"],
                "time_limit": job["budget_sec"],
                "log_to_console": False,
                "log_file": str(directory / "solver.log"),
            }
            for key, value in options.items():
                if solver.setOptionValue(key, value) != highspy.HighsStatus.kOk:
                    raise ValueError(f"HiGHS rejected {key}={value}")
            if (
                solver.passModel(LinearModelRepresentation.to_highs_lp(model))
                != highspy.HighsStatus.kOk
            ):
                raise ValueError("HiGHS rejected prepared model")
            solver.writeOptions(str(directory / "options.txt"))
            trajectory_path = directory / "trajectory.jsonl"
            trajectory_path.write_text("")

            def observe(event):
                data = event.data_out
                record = {
                    "event": int(event.callback_type),
                    "time_sec": finite(data.running_time),
                    "primal_bound": finite(data.mip_primal_bound),
                    "dual_bound": finite(data.mip_dual_bound),
                    "solver_gap": finite(data.mip_gap),
                    "nodes": int(data.mip_node_count),
                    "simplex_iterations": int(data.simplex_iteration_count),
                }
                with trajectory_path.open("a") as handle:
                    handle.write(json.dumps(record, allow_nan=False) + "\n")

            solver.cbMipLogging.subscribe(observe)
            solver.cbMipImprovingSolution.subscribe(observe)
            result["setup_wall_sec"] = time.perf_counter() - started
            solve_started = time.perf_counter()
            cpu_started = time.process_time()
            status = solver.run()
            result["solve_wall_sec"] = time.perf_counter() - solve_started
            result["solve_cpu_sec"] = time.process_time() - cpu_started
            info = solver.getInfo()
            model_status = solver.getModelStatus()
            solution = solver.getSolution()
            result.update(
                {
                    "execution_status": "returned",
                    "call_status": str(status),
                    "model_status": solver.modelStatusToString(model_status),
                    "solver_runtime_sec": solver.getRunTime(),
                    "primal_solution_status": int(info.primal_solution_status),
                    "solver_objective": finite(info.objective_function_value)
                    if solution.value_valid
                    else None,
                    "dual_bound": finite(info.mip_dual_bound),
                    "solver_gap": finite(info.mip_gap),
                    "nodes": int(info.mip_node_count),
                    "simplex_iterations": int(info.simplex_iteration_count),
                    "optimal_with_configured_tolerances": model_status
                    == highspy.HighsModelStatus.kOptimal,
                    "time_to_solver_optimal_sec": solver.getRunTime()
                    if model_status == highspy.HighsModelStatus.kOptimal
                    else None,
                    "solution_value_valid": solution.value_valid,
                }
            )
            verification_started = time.perf_counter()
            tolerance = manifest["solver_options"]["mip_feasibility_tolerance"]
            result["verification"] = None
            result["transformed_verification"] = None
            if solution.value_valid:
                values = np.asarray(solution.col_value)
                mapped = LinearModelRepresentation.map_solution(values, columns)
                np.savez_compressed(
                    directory / "solution.npz", values=values, original_values=mapped
                )
                result["verification"] = LinearModelRepresentation.verify_primal(
                    original, mapped, tolerance
                )
                result["transformed_verification"] = (
                    LinearModelRepresentation.verify_primal(model, values, tolerance)
                )
                result["objective_mapping_difference"] = result["verification"].get(
                    "objective", 0
                ) - result["transformed_verification"].get("objective", 0)
                if "objective" in result["verification"]:
                    result["objective_reporting_difference"] = (
                        result["verification"]["objective"]
                        - info.objective_function_value
                    )
            result["verified_feasible"] = bool(
                (result["verification"] or {}).get("feasible")
            )
            result["verification_wall_sec"] = time.perf_counter() - verification_started
            final_event = {
                "event": "final",
                "time_sec": solver.getRunTime(),
                "primal_bound": result["solver_objective"],
                "dual_bound": result["dual_bound"],
                "solver_gap": result["solver_gap"],
                "nodes": result["nodes"],
            }
            with trajectory_path.open("a") as handle:
                handle.write(json.dumps(final_event, allow_nan=False) + "\n")
        except Exception:
            result.update(
                execution_status="collector_error",
                error=traceback.format_exc(),
                verified_feasible=False,
            )
        result["total_worker_wall_sec"] = time.perf_counter() - started
        write_json(directory / "result.json", result)

    @staticmethod
    def summarize(output: Path) -> dict:
        manifest = json.loads((output / "experiment.json").read_text())
        records = []
        for job in manifest["jobs"]:
            path = output / "runs" / job["run_id"] / "result.json"
            if path.exists():
                record = json.loads(path.read_text())
                if any(record.get(key) != value for key, value in job.items()):
                    raise ValueError(f"result identity changed: {path}")
                records.append(record)
        summary = {
            "expected_runs": len(manifest["jobs"]),
            "completed_runs": len(records),
            "complete": len(records) == len(manifest["jobs"]),
            "verified_feasible_runs": sum(
                bool(item.get("verified_feasible")) for item in records
            ),
            "execution_statuses": dict(
                Counter(item["execution_status"] for item in records)
            ),
            "model_statuses": dict(
                Counter(item.get("model_status", "unavailable") for item in records)
            ),
            "by_axis": dict(Counter(item["axis"] for item in records)),
            "statistics": "deferred",
            "updated_unix": time.time(),
        }
        write_json(output / "collection_summary.json", summary)
        with (output / "results.jsonl").open("w") as handle:
            for record in records:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
        return summary

    @staticmethod
    def run(output: Path, cpus: list[int], axis: str | None, limit: int | None) -> None:
        if (
            not cpus
            or len(cpus) != len(set(cpus))
            or not set(cpus) <= os.sched_getaffinity(0)
        ):
            raise ValueError("CPU IDs must be distinct and available")
        manifest = json.loads((output / "experiment.json").read_text())
        if manifest["solver"] != HighsCollection.solver_identity():
            raise ValueError("solver changed")
        jobs = [
            item for item in manifest["jobs"] if axis is None or item["axis"] == axis
        ]
        jobs = [
            item
            for item in jobs
            if not (output / "runs" / item["run_id"] / "result.json").exists()
        ]
        if limit is not None:
            jobs = jobs[:limit]
        queue = Queue()
        for cpu in cpus:
            queue.put(cpu)
        environment = {
            **os.environ,
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }

        def execute(job):
            cpu = queue.get()
            directory = output / "runs" / job["run_id"]
            directory.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            try:
                with (directory / "process.log").open("w") as log:
                    completed = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "scripts.collect_nuisance_results",
                            "highs",
                            "worker",
                            "--output",
                            str(output),
                            "--run-id",
                            job["run_id"],
                            "--cpu",
                            str(cpu),
                        ],
                        cwd=ROOT,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=job["budget_sec"] + 60,
                        check=False,
                    )
                if not (directory / "result.json").exists():
                    write_json(
                        directory / "result.json",
                        {
                            **job,
                            "cpu": cpu,
                            "execution_status": "process_error",
                            "returncode": completed.returncode,
                            "verified_feasible": False,
                            "process_wall_sec": time.perf_counter() - started,
                        },
                    )
            except subprocess.TimeoutExpired:
                write_json(
                    directory / "result.json",
                    {
                        **job,
                        "cpu": cpu,
                        "execution_status": "watchdog_timeout",
                        "verified_feasible": False,
                        "process_wall_sec": time.perf_counter() - started,
                    },
                )
            finally:
                queue.put(cpu)
            return json.loads((directory / "result.json").read_text())

        print(
            f"Running {len(jobs)} outstanding observations on CPUs {cpus}.", flush=True
        )
        with ThreadPoolExecutor(max_workers=len(cpus)) as pool:
            futures = [pool.submit(execute, job) for job in jobs]
            for count, future in enumerate(as_completed(futures), 1):
                result = future.result()
                print(
                    f"{count}/{len(jobs)} {result['run_id']} {result.get('model_status', result['execution_status'])} "
                    f"verified={result.get('verified_feasible')} time={result.get('solver_runtime_sec')}",
                    flush=True,
                )
                if count % 30 == 0:
                    HighsCollection.summarize(output)
        print(json.dumps(HighsCollection.summarize(output)), flush=True)

    @staticmethod
    def main(argv: list[str] | None = None) -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("command", choices=("prepare", "worker", "run", "summary"))
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--instances", nargs="+")
        parser.add_argument("--cpus", type=int, nargs="+", default=[0])
        parser.add_argument("--axis", choices=("seed", "representation", "control"))
        parser.add_argument("--limit", type=int)
        parser.add_argument("--run-id")
        parser.add_argument("--cpu", type=int)
        args = parser.parse_args(argv)
        output = args.output.resolve()
        if args.command == "prepare":
            if args.instances is None:
                parser.error("prepare requires --instances")
            HighsCollection.prepare(output, args.instances)
        elif args.command == "worker":
            if args.run_id is None or args.cpu is None:
                parser.error("worker requires --run-id and --cpu")
            HighsCollection.worker(output, args.run_id, args.cpu)
        elif args.command == "run":
            if args.limit is not None and args.limit < 1:
                parser.error("limit must be positive")
            HighsCollection.run(output, args.cpus, args.axis, args.limit)
        else:
            print(json.dumps(HighsCollection.summarize(output)), flush=True)


class PyVRPCollection:
    """Collect the approved PyVRP representation panel using the regular judge."""

    SOLVER_SEED = 0
    RELABELING_SEED = 20260907
    RELABELING_COUNT = 30

    @staticmethod
    def run_identity(observation: RunObservation) -> tuple:
        return (
            observation.instance_set,
            observation.instance_id,
            observation.code_state,
            observation.solver_seed,
            observation.budget_sec,
        )

    @staticmethod
    def load_results(path: Path) -> list[dict]:
        from pitbench.schema.observation import RunObservation

        if not path.exists():
            return []
        records = []
        identities = set()
        lines = path.read_text().splitlines(keepends=True)
        for index, line in enumerate(lines):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1 and not line.endswith("\n"):
                    path.write_text("".join(lines[:index]))
                    break
                raise
            identity = PyVRPCollection.run_identity(
                RunObservation.model_validate(record["observation"])
            )
            if identity in identities:
                raise ValueError("duplicate run in results checkpoint")
            identities.add(identity)
            records.append(record)
        if records and not path.read_bytes().endswith(b"\n"):
            with path.open("a") as handle:
                handle.write("\n")
        return records

    @staticmethod
    def main(argv: list[str] | None = None) -> None:
        from adapters.pitbench.adapter import PitBenchAdapter
        from pitbench.evaluator.representation import (
            RecordingJudge,
            prepare_cases,
            preserve_json,
            result_record,
            write_json,
        )
        from pitbench.evaluator.storage import ObservationStore
        from pitbench.schema.observation import CodeState, RunObservation
        from pitbench.schema.task import PitBenchTask

        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--repository", type=Path, required=True)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--private-root", type=Path, default=ROOT / "private")
        parser.add_argument("--parallel-runs", type=int, default=4)
        parser.add_argument("--prepare-only", action="store_true")
        parser.add_argument(
            "--case-limit",
            type=int,
            help="Run only the first N relabelings for a smoke check.",
        )
        args = parser.parse_args(argv)
        if args.parallel_runs < 1 or (
            args.case_limit is not None and args.case_limit < 1
        ):
            parser.error("parallel-runs and case-limit must be positive")
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        task = PitBenchTask.from_yaml(ROOT / "configs/tasks/pyvrp_v0_14_0.yaml")
        if task.release.version != "0.14.0" or task.evaluation.budgets_sec != [
            5.0,
            10.0,
        ]:
            raise ValueError("task configuration differs from the approved experiment")
        PitBenchAdapter.validate_repository(task, args.repository)
        manifest = {
            "task_id": task.task_id,
            "source_commit": task.release.base_commit,
            "solver_seed": PyVRPCollection.SOLVER_SEED,
            "relabeling_generation_seed": PyVRPCollection.RELABELING_SEED,
            "relabelings_per_instance": PyVRPCollection.RELABELING_COUNT,
            "instance_count": 10,
            "budgets_sec": [5.0, 10.0],
            "code_states": ["base", "agent"],
            "candidate_patch": "empty",
            "task_configuration": task.model_dump(mode="json"),
        }
        preserve_json(output_dir / "experiment.json", manifest)
        cases, transformations = prepare_cases(task, args.private_root, output_dir)
        expected = {
            (
                case.instance_set.name,
                case.instance_id,
                state,
                PyVRPCollection.SOLVER_SEED,
                budget,
            )
            for case in cases
            for state in CodeState
            for budget in (5.0, 10.0)
        }
        print(
            f"Prepared {len(cases)} equivalent inputs; {len(expected)} planned runs.",
            flush=True,
        )
        if args.prepare_only:
            return
        checkpoint = output_dir / "results.jsonl"
        records = PyVRPCollection.load_results(checkpoint)
        observations = [
            RunObservation.model_validate(record["observation"]) for record in records
        ]
        completed = {
            PyVRPCollection.run_identity(observation) for observation in observations
        }
        if completed - expected:
            raise ValueError("results contain runs outside the approved experiment")
        selected_cases = (
            cases[: args.case_limit] if args.case_limit is not None else cases
        )
        selected_expected = {
            item
            for item in expected
            if item[1] in {case.instance_id for case in selected_cases}
        }
        patch = output_dir / "no_change.patch"
        if patch.exists() and patch.read_bytes():
            raise ValueError("experiment requires an empty patch")
        patch.touch()
        if selected_expected - completed:
            judge = RecordingJudge(
                task,
                args.repository,
                ROOT,
                args.private_root,
                patch,
                output_dir / "runs",
                code_states=(CodeState.BASE, CodeState.AGENT),
                parallel_runs=args.parallel_runs,
                run_validation_builds=False,
            )
            with checkpoint.open("a") as handle:

                def save(observation):
                    record = result_record(
                        observation,
                        transformations[observation.instance_id],
                        output_dir,
                    )
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    records.append(record)
                    observations.append(observation)

                judge.run(
                    selected_cases, completed_runs=completed, save_observation=save
                )
        ObservationStore.write(output_dir / "observations.parquet", observations)
        summary = {
            "task_id": task.task_id,
            "expected_run_count": len(expected),
            "completed_run_count": len(observations),
            "complete": len(observations) == len(expected),
            "valid_run_count": sum(item.valid for item in observations),
            "mapped_feasible_run_count": sum(
                bool((record["verification"]["mapped_original"] or {}).get("feasible"))
                for record in records
            ),
            "objective_preserved_run_count": sum(
                record["verification"]["objective_preserved"] is True
                for record in records
            ),
            "statistics": "deferred",
        }
        write_json(output_dir / "collection_summary.json", summary)
        print(json.dumps(summary), flush=True)
