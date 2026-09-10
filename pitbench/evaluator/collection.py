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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

from pitbench.solver_drivers.common import ParameterRejected
from pitbench.solver_drivers.run import HighsDriver

ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def run_collection_process(
    command: list[str],
    directory: Path,
    identity: dict,
    *,
    timeout: float,
    environment: dict | None = None,
) -> dict:
    """Collect one isolated worker, including process failures and watchdog exits."""
    directory.mkdir(parents=True, exist_ok=True)
    result = dict(identity)
    started = time.perf_counter()
    try:
        with (directory / "process.log").open("w") as log:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment
                or {
                    **os.environ,
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                },
                timeout=timeout,
                check=False,
            )
        result.update(execution_status="process_error", returncode=completed.returncode)
        if (directory / "result.json").exists():
            result.update(json.loads((directory / "result.json").read_text()))
            result["returncode"] = completed.returncode
            if completed.returncode:
                result["execution_status"] = "process_error"
    except subprocess.TimeoutExpired:
        result.update(execution_status="watchdog_timeout", verified_feasible=False)
    except OSError as error:
        result.update(execution_status="collector_error", error=str(error))
    result.update(
        process_wall_sec=time.perf_counter() - started,
        result_path=str(directory / "result.json"),
    )
    write_json(directory / "result.json", result)
    return result


def run_collection_jobs(jobs: list[dict], cpus: list[int], execute) -> list[dict]:
    """Shared bounded scheduling for isolated observation workers."""
    if (
        not cpus
        or len(cpus) != len(set(cpus))
        or not set(cpus) <= os.sched_getaffinity(0)
    ):
        raise ValueError("cpus must be distinct available CPUs")
    slots = Queue()
    for cpu in cpus:
        slots.put(cpu)

    def run(job):
        cpu = slots.get()
        try:
            return execute(job, cpu)
        finally:
            slots.put(cpu)

    with ThreadPoolExecutor(max_workers=len(cpus)) as pool:
        return list(pool.map(run, jobs))


class NuisanceCollection:
    """Configured axes and transformations, with one retained job/result contract."""

    @staticmethod
    def prepare(
        config_path: Path,
        output: Path,
        *,
        repository: Path | None = None,
        private_root: Path = ROOT / "private",
        panel: str | None = None,
    ) -> dict:
        import yaml

        from pitbench.evaluator.representation import prepare_transformations
        from pitbench.evaluator.representations import representation_type
        from pitbench.instances.generate import prepare_collection_instances
        from pitbench.repositories.base import RepositoryPluginRegistry
        from pitbench.schema.task import PitBenchTask

        config = yaml.safe_load(config_path.read_text())
        if "panels" in config:
            if panel not in config["panels"]:
                raise ValueError("select a configured panel with --panel")
            config = config["panels"][panel]
        task = PitBenchTask.from_yaml(ROOT / config["task_config"])
        plugin = RepositoryPluginRegistry.load(task.repository.plugin)
        transform = representation_type(config["representation"]["kind"])
        if transform.family != task.problem_family or config[
            "execution"
        ] not in plugin.representations.get(transform.name, ()):
            raise ValueError(
                "representation is not supported by the configured repository"
            )
        if (
            transform.requires_tolerance
            and config["representation"].get("verification_tolerance") is None
        ):
            raise ValueError("numeric representation requires a verification tolerance")
        executor = COLLECTION_EXECUTORS[config["execution"]]
        identity = executor.identity(task, repository)
        if identity["version"] != task.release.version or (
            identity.get("binding_git_hash")
            and not task.release.base_commit.startswith(identity["binding_git_hash"])
        ):
            raise ValueError("runtime identity differs from configured task release")
        manifest_path = output / "experiment.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if (
                manifest["config"] != config
                or manifest["solver"] != identity
                or manifest["task_configuration"] != task.model_dump(mode="json")
                or manifest["repository"]
                != (str(repository.resolve()) if repository else None)
            ):
                raise ValueError(
                    "existing experiment differs from requested configuration"
                )
            return manifest
        instances = prepare_collection_instances(
            ROOT / config["instance_source"],
            output / "sources",
            path_template=config.get("instance_path_template"),
        )
        if len(instances) != config["instance_count"] or len(
            {i["id"] for i in instances}
        ) != len(instances):
            raise ValueError(
                "instance panel differs from configured count or contains duplicates"
            )
        seeds = config["solver_seeds"]
        if not seeds or len(seeds) != len(set(seeds)):
            raise ValueError("solver seeds must be nonempty and unique")
        generator = transform.generator(config["representation"]["generation_seed"])
        jobs, transformations = [], {}
        for instance in instances:
            prepared = prepare_transformations(
                Path(instance["path"]),
                instance["id"],
                instance["bks"],
                output,
                transform.name,
                config["representation"]["count"],
                generator,
                tolerance=config["representation"].get("verification_tolerance"),
            )
            transformations.update(prepared)
            original_path = next(iter(prepared.values()))["original_instance_path"]
            variants = [
                ("seed", i, seed, original_path, None)
                for i, seed in enumerate(seeds)
                if "seed" in config["axes"]
            ]
            variants += [
                (
                    "representation",
                    i,
                    config["representation"]["solver_seed"],
                    mapping["transformed_instance_path"],
                    mapping,
                )
                for i, mapping in enumerate(prepared.values())
                if "representation" in config["axes"]
            ]
            variants += [
                (
                    "control",
                    i,
                    config["representation"]["solver_seed"],
                    original_path,
                    None,
                )
                for i in range(config["control_repeats"])
                if "control" in config["axes"]
            ]
            for axis, replicate, seed, path, mapping in variants:
                case_id = f"{instance['id']}__{axis}_{replicate:02d}"
                for budget in task.evaluation.budgets_sec:
                    for state in config["code_states"]:
                        jobs.append(
                            {
                                "run_id": f"{instance['id']}/{axis}/{replicate:02d}/{state}/budget-{budget:g}",
                                "instance": instance["id"],
                                "instance_id": case_id,
                                "axis": axis,
                                "replicate": replicate,
                                "solver_seed": seed,
                                "permutation": mapping["transform_id"]
                                if mapping
                                else None,
                                "budget_sec": budget,
                                "code_state": state,
                                "path": path,
                                "original_path": original_path,
                                "transformation": mapping,
                                "bks": instance["bks"],
                            }
                        )
        random.Random(config["schedule_seed"]).shuffle(jobs)
        manifest = {
            "config": config,
            "task_id": task.task_id,
            "task_configuration": task.model_dump(mode="json"),
            "solver": identity,
            "repository": str(repository.resolve()) if repository else None,
            "private_root": str(private_root.resolve()),
            "budgets_sec": task.evaluation.budgets_sec,
            "solver_seeds": seeds,
            "code_states": config["code_states"],
            "statistics": "deferred",
            "environment": {"python": sys.version, "platform": platform.platform()},
            "instances": instances,
            "jobs": jobs,
        }
        write_json(output / "transformations.json", transformations)
        write_json(manifest_path, manifest)
        return manifest

    @staticmethod
    def run(
        output: Path,
        cpus: list[int],
        *,
        axis: str | None = None,
        limit: int | None = None,
    ) -> dict:
        manifest = json.loads((output / "experiment.json").read_text())
        executor = COLLECTION_EXECUTORS[manifest["config"]["execution"]]
        if not cpus or not set(cpus) <= os.sched_getaffinity(0):
            raise ValueError("requested CPUs are unavailable")
        jobs = []
        for job in manifest["jobs"]:
            path = output / "runs" / job["run_id"] / "result.json"
            if path.exists():
                result = json.loads(path.read_text())
                if any(result.get(key) != value for key, value in job.items()):
                    raise ValueError(f"saved run identity differs: {path}")
            elif axis is None or job["axis"] == axis:
                jobs.append(job)
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be positive")
            jobs = jobs[:limit]
        if jobs:
            executor.run(output, manifest, jobs, cpus)
        return NuisanceCollection.summarize(output)

    @staticmethod
    def summarize(output: Path) -> dict:
        from pitbench.metrics.nuisance_report import report_nuisance_results

        summary = report_nuisance_results(output, output / "report")
        write_json(output / "collection_summary.json", summary)
        return summary

    @staticmethod
    def main(argv: list[str] | None = None) -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        commands = parser.add_subparsers(dest="command", required=True)
        prepare = commands.add_parser("prepare")
        prepare.add_argument("--config", type=Path, required=True)
        prepare.add_argument("--panel")
        prepare.add_argument("--repository", type=Path)
        prepare.add_argument("--private-root", type=Path, default=ROOT / "private")
        run = commands.add_parser("run")
        run.add_argument("--cpus", type=int, nargs="+", required=True)
        run.add_argument("--axis", choices=("seed", "representation", "control"))
        run.add_argument("--limit", type=int)
        summary = commands.add_parser("summary")
        for command in (prepare, run, summary):
            command.add_argument("--output", type=Path, required=True)
        worker = commands.add_parser("worker")
        worker.add_argument("--job", type=Path, required=True)
        worker.add_argument("--cpu", type=int, required=True)
        args = parser.parse_args(argv)
        if args.command == "worker":
            configuration_worker(args.job, args.cpu)
            return
        output = args.output.resolve()
        if args.command == "prepare":
            result = NuisanceCollection.prepare(
                args.config,
                output,
                repository=args.repository,
                private_root=args.private_root,
                panel=args.panel,
            )
            print(f"Prepared {len(result['jobs'])} runs; no solver runs executed.")
        else:
            result = (
                NuisanceCollection.run(
                    output, args.cpus, axis=args.axis, limit=args.limit
                )
                if args.command == "run"
                else NuisanceCollection.summarize(output)
            )
            print(json.dumps({k: v for k, v in result.items() if k != "groups"}))


class IsolatedCollection:
    """Execute a saved panel through installed solver backends in fresh processes."""

    @staticmethod
    def identity(task, repository):
        if repository is not None:
            raise ValueError(
                "isolated collection uses the current solver Python; use judge execution for a source repository"
            )
        return collection_backend(task.repository.plugin).identity()

    @staticmethod
    def run(output, manifest, jobs, cpus):
        from pitbench.schema.task import PitBenchTask

        task = PitBenchTask.model_validate(manifest["task_configuration"])
        if IsolatedCollection.identity(task, None) != manifest["solver"]:
            raise ValueError("solver identity changed since preparation")
        if set(manifest["code_states"]) != {"base"}:
            raise ValueError("installed collection supports original release runs only")

        def execute(job, cpu):
            directory = output / "runs" / job["run_id"]
            worker = {
                **job,
                "nuisance_identity": job,
                "instance": {
                    "id": job["instance"],
                    "path": str(output / job["path"]),
                    "bks": job["bks"],
                },
                "original_instance_path": str(output / job["original_path"]),
                "parameters": {},
                "fixed_options": manifest["config"].get("fixed_options", {}),
                "threads": task.evaluation.threads,
                "solver": manifest["solver"],
                "repository_plugin": task.repository.plugin,
            }
            write_json(directory / "job.json", worker)
            result = run_collection_process(
                [
                    sys.executable,
                    "-m",
                    "scripts.collect_nuisance_results",
                    "worker",
                    "--job",
                    str(directory / "job.json"),
                    "--cpu",
                    str(cpu),
                ],
                directory,
                job,
                timeout=job["budget_sec"] + manifest["config"]["watchdog_grace_sec"],
            )
            # Manifest identities remain the same across all executor formats.
            result.update(job)
            write_json(directory / "result.json", result)
            return result

        run_collection_jobs(jobs, cpus, execute)


class JudgeCollection:
    """Execute a saved panel through the existing repository build and judge."""

    @staticmethod
    def identity(task, repository):
        from adapters.pitbench.adapter import PitBenchAdapter

        if repository is None:
            raise ValueError("judge collection requires --repository")
        PitBenchAdapter.validate_repository(task, repository)
        return {
            "version": task.release.version,
            "source_commit": task.release.base_commit,
        }

    @staticmethod
    def run(output, manifest, jobs, cpus):
        from pitbench.evaluator.judge import InstanceCase, LocalProcessJudge
        from pitbench.evaluator.representation import result_record
        from pitbench.evaluator.representations import representation_type
        from pitbench.evaluator.storage import ObservationStore
        from pitbench.schema.observation import CodeState
        from pitbench.schema.task import InstanceSetSpec, PitBenchTask

        task = PitBenchTask.model_validate(manifest["task_configuration"])
        repository = Path(manifest["repository"])
        JudgeCollection.identity(task, repository)
        transform = representation_type(manifest["config"]["representation"]["kind"])
        verifier = transform.verifier(
            manifest["config"]["representation"].get("verification_tolerance")
        )
        instance_set = InstanceSetSpec(
            name="nuisance",
            kind="agent_dev",
            instance_set_config="experiment.json",
            size=len(manifest["instances"]),
        )
        cases = {}
        for job in manifest["jobs"]:
            cases[job["instance_id"]] = InstanceCase(
                instance_set=instance_set,
                instance_id=job["instance_id"],
                path=output / job["path"],
                anchor=job["bks"],
                solver_seeds=(job["solver_seed"],),
                budgets_sec=tuple(manifest["budgets_sec"]),
                equivalence_parent_id=job["instance"]
                if job["transformation"]
                else None,
                equivalence_transform=job["permutation"],
                verifier=verifier,
            )

        def key(job):
            return (
                instance_set.name,
                job["instance_id"],
                CodeState(job["code_state"]),
                job["solver_seed"],
                job["budget_sec"],
            )

        selected = {key(job): job for job in jobs}
        skipped = {key(job) for job in manifest["jobs"]} - selected.keys()

        def save(observation):
            job = selected[
                (
                    observation.instance_set,
                    observation.instance_id,
                    observation.code_state,
                    observation.solver_seed,
                    observation.budget_sec,
                )
            ]
            checked = {
                "feasible": observation.valid,
                "objective": observation.objective,
            }
            record = {
                **job,
                "observation": observation.model_dump(mode="json"),
                "execution_status": observation.status.value,
                "model_status": observation.solver_status,
                "solver_objective": observation.objective,
                "solver_runtime_sec": observation.wall_time_sec,
                "dual_bound": observation.dual_bound,
                "nodes": observation.nodes,
                "verification": checked,
                "verified_feasible": observation.valid,
                "peak_rss_bytes": observation.peak_rss_bytes,
                "resource_scope": observation.resource_scope,
                "cpu_time_sec": observation.cpu_time_sec,
                "trajectory_path": observation.trajectory_path,
            }
            if job["transformation"]:
                detailed = result_record(
                    observation,
                    job["transformation"],
                    output,
                    runs_dir=output / "judge",
                )
                record["representation_verification"] = detailed["verification"]
                record["artifacts"] = detailed["artifacts"]
                record["verification"] = detailed["verification"]["mapped_original"]
                record["verified_feasible"] = bool(
                    observation.valid
                    and (record["verification"] or {}).get("feasible")
                    and detailed["verification"]["objective_preserved"]
                )
            write_json(output / "runs" / job["run_id"] / "result.json", record)

        affinity = os.sched_getaffinity(0)
        try:
            os.sched_setaffinity(0, set(cpus))
            judge = LocalProcessJudge(
                task,
                repository,
                ROOT,
                Path(manifest["private_root"]),
                None,
                output / "judge",
                code_states=tuple(CodeState(s) for s in manifest["code_states"]),
                parallel_runs=len(cpus),
                run_validation_builds=False,
                evaluation_seeds=tuple(manifest["solver_seeds"]),
                family=verifier,
            )
            judge.run(
                list(cases.values()), completed_runs=skipped, save_observation=save
            )
        finally:
            os.sched_setaffinity(0, affinity)
        records = [
            json.loads(path.read_text())["observation"]
            for path in (output / "runs").glob("**/result.json")
        ]
        from pitbench.schema.observation import RunObservation

        ObservationStore.write(
            output / "observations.parquet",
            [RunObservation.model_validate(record) for record in records],
        )


COLLECTION_EXECUTORS = {"judge": JudgeCollection, "isolated": IsolatedCollection}


class AnchorCollection:
    """Generate empirical anchors with the task's existing judge and verifier."""

    @staticmethod
    def main(argv: list[str] | None = None) -> None:
        from datetime import UTC, datetime

        import yaml

        from adapters.pitbench.adapter import PitBenchAdapter
        from pitbench.evaluator.judge import InstanceCase, LocalProcessJudge
        from pitbench.evaluator.storage import ObservationStore
        from pitbench.instances.generate import materialize_generated_instance_set
        from pitbench.schema.observation import CodeState, RunStatus
        from pitbench.schema.task import InstanceSetKind, InstanceSetSpec, PitBenchTask

        parser = argparse.ArgumentParser(
            description="Generate independently verified empirical BKS anchors."
        )
        parser.add_argument("--task-config", type=Path, required=True)
        parser.add_argument("--repository", type=Path, required=True)
        parser.add_argument("--instance-set-config", type=Path, required=True)
        parser.add_argument("--private-root", type=Path, default=ROOT / "private")
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--budget-sec", type=float, required=True)
        parser.add_argument("--seeds", type=int, nargs="+", required=True)
        parser.add_argument("--workers", type=int, default=5)
        args = parser.parse_args(argv)
        if (
            not math.isfinite(args.budget_sec)
            or args.budget_sec <= 0
            or args.workers <= 0
        ):
            parser.error("budget and worker count must be positive")
        if len(set(args.seeds)) != len(args.seeds):
            parser.error("seeds must be unique")
        task = PitBenchTask.from_yaml(args.task_config)
        if task.oracle.objective_sense is None:
            raise ValueError("anchor generation requires an objective sense")
        PitBenchAdapter.validate_repository(task, args.repository)
        output = args.output_dir.resolve()
        payload = yaml.safe_load(args.instance_set_config.read_text())
        paths = materialize_generated_instance_set(
            payload,
            output / "inputs",
            expected_visibility="judge",
            stem_prefix="judge_shift",
        )
        instance_set = InstanceSetSpec(
            name="anchor_generation",
            kind=InstanceSetKind.JUDGE_SHIFT,
            instance_set_config=str(args.instance_set_config.resolve()),
            size=len(paths),
        )
        cases = [
            InstanceCase(
                instance_set=instance_set,
                instance_id=path.stem,
                path=path,
                anchor=None,
                solver_seeds=tuple(args.seeds),
                budgets_sec=(args.budget_sec,),
            )
            for path in paths
        ]
        judge = LocalProcessJudge(
            task=task,
            base_repository=args.repository.resolve(),
            public_root=ROOT,
            private_root=args.private_root,
            candidate_patch=None,
            output_dir=output / "runs",
            code_states=(CodeState.BASE,),
            parallel_runs=args.workers,
            run_validation_builds=False,
            evaluation_seeds=tuple(args.seeds),
        )
        # Keep every completed observation even if a later run or build fails.
        with (output / "observations.jsonl").open("w") as checkpoint:

            def save(observation):
                checkpoint.write(observation.model_dump_json() + "\n")
                checkpoint.flush()

            observations = judge.run(cases, save_observation=save)
        ObservationStore.write(output / "observations.parquet", observations)
        expected = {(path.stem, seed) for path in paths for seed in args.seeds}
        observed = {(item.instance_id, item.solver_seed) for item in observations}
        if observed != expected or len(observations) != len(expected):
            raise ValueError("anchor generation did not complete the declared run grid")
        if any(
            item.status != RunStatus.COMPLETED
            or not item.valid
            or item.objective is None
            or not math.isfinite(item.objective)
            or item.solution_path is None
            for item in observations
        ):
            raise ValueError("invalid BKS candidate; inspect the retained observations")
        # Resolve all paths before publishing an oracle so missing evidence cannot pass.
        solutions = {
            (item.instance_id, item.solver_seed): Path(item.solution_path)
            for item in observations
        }
        if any(not path.is_file() for path in solutions.values()):
            raise ValueError("BKS candidate solution is missing")
        private_root = args.private_root.resolve()
        if output.is_relative_to(private_root):
            solution_prefix = "private://" + output.relative_to(private_root).as_posix()
        else:
            solution_prefix = None
        anchors = []
        for path in paths:
            candidates = [
                item for item in observations if item.instance_id == path.stem
            ]
            direction = 1 if task.oracle.objective_sense == "minimize" else -1
            best = min(
                candidates,
                key=lambda item: (direction * item.objective, item.solver_seed),
            )
            destination = output / f"{path.stem}.bks.solution.json"
            destination.write_bytes(
                solutions[(path.stem, best.solver_seed)].read_bytes()
            )
            anchors.append(
                {
                    "id": path.stem,
                    "bks": best.objective,
                    **(
                        {"bks_solution_uri": f"{solution_prefix}/{destination.name}"}
                        if solution_prefix
                        else {"bks_solution_file": destination.name}
                    ),
                    "instance_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "bks_solution_sha256": hashlib.sha256(
                        destination.read_bytes()
                    ).hexdigest(),
                    "winning_seed": best.solver_seed,
                    "verified_objectives": {
                        str(item.solver_seed): item.objective for item in candidates
                    },
                }
            )
        family = type(judge.family)
        oracle = {
            "schema_version": "1.0",
            "kind": "empirical_best_known_solution",
            "problem_family": task.problem_family.value,
            "objective_sense": task.oracle.objective_sense,
            "generated_at": datetime.now(UTC).isoformat(),
            "instance_set_config_sha256": hashlib.sha256(
                args.instance_set_config.read_bytes()
            ).hexdigest(),
            "solver": {
                "name": judge.repository.name,
                "version": task.release.version,
                "source_commit": task.release.base_commit,
            },
            "protocol": {
                "budget_sec": args.budget_sec,
                "seeds": args.seeds,
                "workers": args.workers,
                "threads": task.evaluation.threads,
                "independent_verifier": f"{family.__module__}:{family.__name__}",
            },
            "anchors": anchors,
        }
        metric = payload["generator"].get("distance_metric")
        if metric is not None:
            oracle["distance_metric"] = metric
        oracle_path = output / "oracle.yaml"
        oracle_path.write_text(yaml.safe_dump(oracle, sort_keys=False))
        print(oracle_path)


class PyVRPCollectionBackend:
    """Parameter panels reuse the normal routing driver and verifier."""

    @staticmethod
    def identity() -> dict:
        import importlib.metadata

        import pyvrp

        return {
            "version": importlib.metadata.version("pyvrp"),
            "module": pyvrp.__file__,
        }

    @staticmethod
    def run(job: dict, directory: Path, result: dict) -> None:
        from pitbench.problem_families.verification import CVRPFamily
        from pitbench.solver_drivers.run import PyVRPDriver

        result["error_stage"] = "parameters"
        try:
            params = PyVRPDriver.parameters(job["parameters"])
        except (ValueError, TypeError, AttributeError) as error:
            raise ParameterRejected(str(error)) from error
        result["effective_parameters"] = PyVRPDriver.parameter_values(params)
        parameters = directory / "parameters.json"
        write_json(parameters, job["parameters"])
        output = directory / "driver.json"
        result["error_stage"] = "solve"
        try:
            PyVRPDriver.main(
                [
                    "--instance",
                    job["instance"]["path"],
                    "--output",
                    str(output),
                    "--trajectory",
                    str(directory / "trajectory.jsonl"),
                    "--parameters",
                    str(parameters),
                    "--seed",
                    str(job["solver_seed"]),
                    "--budget",
                    str(job["budget_sec"]),
                    "--threads",
                    str(job["threads"]),
                ]
            )
        finally:
            if output.exists():
                result.update(json.loads(output.read_text()))
        result.update(
            execution_status="returned",
            model_status=result.get("solver_status"),
            solution_value_valid=result.get("has_solution", False),
        )
        result["error_stage"] = "verification"
        solution = output.with_suffix(".solution.json")
        result["solution_path"] = str(solution) if solution.exists() else None
        result["verification"] = None
        if solution.exists():
            result["verification"] = (
                CVRPFamily()
                .verify(Path(job["instance"]["path"]), solution)
                .model_dump()
            )
        result["verified_feasible"] = bool(
            (result["verification"] or {}).get("feasible")
        )
        result.pop("error_stage", None)


class HighsCollectionBackend:
    """Parameter panels reuse the numeric-model collector and its verifier."""

    @staticmethod
    def identity() -> dict:
        import highspy

        solver = highspy.Highs()
        return {"version": solver.version(), "binding_git_hash": solver.githash()}

    @staticmethod
    def run(job: dict, directory: Path, result: dict) -> None:
        import numpy as np

        from pitbench.evaluator.representations import LinearModelRepresentation

        started = time.perf_counter()
        model = LinearModelRepresentation.read_input(Path(job["instance"]["path"]))
        original = LinearModelRepresentation.read_input(
            Path(job.get("original_instance_path", job["instance"]["path"]))
        )
        options = {
            **job["fixed_options"],
            **job["parameters"],
            "threads": job["threads"],
            "random_seed": job["solver_seed"],
            "time_limit": job["budget_sec"],
            "log_to_console": False,
            "log_file": str(directory / "solver.log"),
        }
        HighsDriver.solve_numeric(
            model,
            original,
            (job.get("transformation") or {}).get(
                "columns", np.arange(len(original["cost"]))
            ),
            directory,
            options,
            result,
            started,
        )


def collection_backend(repository_plugin: str):
    from pitbench.repositories.base import RepositoryPluginRegistry

    return RepositoryPluginRegistry.load(repository_plugin).load_collection_backend()


def configuration_worker(job_path: Path, cpu: int) -> None:
    """One fresh process per solver run; keep failures distinct from missing data."""
    os.sched_setaffinity(0, {cpu})
    job = json.loads(job_path.read_text())
    result = {**job, "cpu": cpu, "pid": os.getpid(), "started_unix": time.time()}
    started = time.perf_counter()
    try:
        collector = collection_backend(job["repository_plugin"])
        result["error_stage"] = "setup"
        result["solver"] = collector.identity()
        if result["solver"] != job["solver"]:
            raise ValueError("solver identity changed since preparation")
        collector.run(job, job_path.parent, result)
        if job.get("transformation") and result.get("solution_path"):
            from pitbench.evaluator.representations import representation_type

            mapping = job["transformation"]
            result["error_stage"] = "verification"
            checked = representation_type(mapping["representation"]).verify_files(
                Path(job["original_instance_path"]),
                Path(job["instance"]["path"]),
                Path(result["solution_path"]),
                job_path.parent / "mapped.solution.json",
                mapping,
                tolerance=mapping.get("verification_tolerance"),
            )
            result["representation_verification"] = checked
            result["verification"] = checked["mapped_original"]
            result["verified_feasible"] = bool(
                checked["transformed"]["feasible"]
                and checked["mapped_original"]["feasible"]
            )
            result.pop("error_stage", None)
    except ParameterRejected as error:
        result.update(execution_status="parameter_rejected", error=str(error))
    except Exception:
        result.update(
            execution_status="solver_error"
            if result.get("error_stage") == "solve"
            else "collector_error",
            error=traceback.format_exc(),
        )
    result["total_worker_wall_sec"] = time.perf_counter() - started
    result.update(job.get("nuisance_identity", {}))
    write_json(job_path.parent / "result.json", result)
