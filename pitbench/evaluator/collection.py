"""Approved raw-result collection protocols, separated by solver backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from threading import Condition

from pitbench.evaluator.artifacts import write_json
from pitbench.problem_families.verification import _check_bin_packing_solution
from pitbench.solver_drivers.common import ParameterRejected
from pitbench.solver_drivers.run import HighsDriver

ROOT = Path(__file__).resolve().parents[2]


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
            process = subprocess.Popen(
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
                start_new_session=True,
            )
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        result.update(execution_status="process_error", returncode=process.returncode)
        if (directory / "result.json").exists():
            result.update(json.loads((directory / "result.json").read_text()))
            result["returncode"] = process.returncode
            if process.returncode:
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


def run_collection_jobs(
    jobs: list[dict], cpus: list[int], execute, *, cpu_request=None
) -> list[dict]:
    """Shared bounded scheduling for isolated observation workers."""
    if (
        not cpus
        or len(cpus) != len(set(cpus))
        or not set(cpus) <= os.sched_getaffinity(0)
    ):
        raise ValueError("cpus must be distinct available CPUs")
    if cpu_request is not None:
        requests = [cpu_request(job) for job in jobs]
        if any(
            type(count) is not int or not 1 <= count <= len(cpus) for count in requests
        ):
            raise ValueError("job CPU requests must fit the available CPU pool")
        available = cpus.copy()
        changed = Condition()

        def run_group(pair):
            job, count = pair
            with changed:
                changed.wait_for(lambda: len(available) >= count)
                allocated = available[:count]
                del available[:count]
            try:
                return execute(job, allocated)
            finally:
                with changed:
                    available.extend(allocated)
                    changed.notify_all()

        with ThreadPoolExecutor(max_workers=len(cpus)) as pool:
            return list(pool.map(run_group, zip(jobs, requests, strict=True)))
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


def reference_resource_state(limits: dict, root: Path) -> dict:
    """Admission limits for reference preparation, independent of scoring budgets."""
    import shutil

    memory = {
        line.split(":")[0]: int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    }
    temperatures = []
    for path in Path("/sys/class/hwmon").glob("hwmon*/temp*_input"):
        try:
            value = int(path.read_text()) / 1000
            if 0 < value < 150:
                temperatures.append(value)
        except (OSError, ValueError):
            pass
    state = {
        "load1": os.getloadavg()[0],
        "memory_available_gib": memory["MemAvailable"] / 1024**2,
        "disk_free_gib": shutil.disk_usage(root).free / 1024**3,
        "max_temperature_celsius": max(temperatures, default=None),
        "blocked_by": [],
    }
    for name, actual, limit, greater in [
        ("load", state["load1"], limits.get("max_load1"), True),
        (
            "temperature",
            state["max_temperature_celsius"],
            limits.get("max_temperature_celsius"),
            True,
        ),
        (
            "memory",
            state["memory_available_gib"],
            limits.get("min_memory_available_gib"),
            False,
        ),
        ("disk", state["disk_free_gib"], limits.get("min_disk_free_gib"), False),
    ]:
        if (
            actual is not None
            and limit is not None
            and (actual > limit if greater else actual < limit)
        ):
            state["blocked_by"].append(name)
    return state


class ReferenceCollection:
    """Resumable reference portfolio using the common isolated-worker scheduler."""

    @staticmethod
    def recover_interrupted(output: Path) -> list[dict]:
        """Retain stale attempts after a reboot; never start over live workers."""
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        recovered = []
        for directory in sorted((output / "jobs").glob("**/attempt-*")):
            if (directory / "result.json").exists() or not (
                directory / "running.json"
            ).exists():
                continue
            record = json.loads((directory / "running.json").read_text())
            command_path = Path(f"/proc/{record['pid']}/cmdline")
            if (
                record.get("worker_boot_id") in {None, boot_id}
                and command_path.exists()
            ):
                try:
                    command = (
                        command_path.read_bytes()
                        .replace(b"\x00", b" ")
                        .decode(errors="replace")
                    )
                except OSError:
                    command = ""
                if "--worker-job" in command and str(directory) in command:
                    raise RuntimeError(
                        "prior reference workers are still active; waiting avoids duplicate searches"
                    )
            if (directory / "incumbent.json").exists():
                record.update(json.loads((directory / "incumbent.json").read_text()))
            record.update(
                execution_status="interrupted",
                recovered_at=time.time(),
                recovery_boot_id=boot_id,
                interruption="controller_or_host_restart",
            )
            write_json(directory / "result.json", record)
            recovered.append({"job_id": record["job_id"], "directory": str(directory)})
        if recovered:
            with (output / "recovery.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {"time": time.time(), "boot_id": boot_id, "attempts": recovered}
                    )
                    + "\n"
                )
        return recovered

    @staticmethod
    def prepare(config_path: Path, output: Path) -> dict:
        import importlib.metadata
        import shutil

        import yaml

        from pitbench.solver_drivers.reference import (
            generated_reference_instances,
            verify_reference_file,
        )

        config_path, output = config_path.resolve(), output.resolve()
        config = yaml.safe_load(config_path.read_text())
        if (output / "campaign.json").exists():
            saved = json.loads((output / "campaign.json").read_text())
            if saved["config"] != config:
                raise ValueError("existing campaign configuration differs")
            return saved
        output.mkdir(parents=True, exist_ok=False)
        cpus = config.get("cpus")
        if cpus is None:
            physical = {}
            for cpu in sorted(os.sched_getaffinity(0)):
                topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
                key = tuple(
                    (topology / name).read_text().strip()
                    for name in ("physical_package_id", "core_id")
                )
                physical.setdefault(key, cpu)
            cpus = list(physical.values())[: config["physical_cores"]]
        if (
            not cpus
            or len(cpus) != len(set(cpus))
            or not set(cpus) <= os.sched_getaffinity(0)
        ):
            raise ValueError("campaign CPUs must be distinct available CPUs")
        instances = []
        for source in config["manifests"]:
            instances.extend(generated_reference_instances(ROOT / source))
        if len({instance["id"] for instance in instances}) != len(instances):
            raise ValueError(
                "campaign instance identities must be unique across sources"
            )
        for instance in instances:
            if (
                hashlib.sha256(Path(instance["path"]).read_bytes()).hexdigest()
                != instance["sha256"]
            ):
                raise ValueError(f"input hash mismatch: {instance['id']}")
        snapshot = output / "code_snapshot"
        shutil.copytree(
            ROOT / "pitbench",
            snapshot / "pitbench",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        source_hashes = {
            str(path.relative_to(snapshot)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(snapshot.rglob("*"))
            if path.is_file()
        }
        backends = []
        for backend in config["backends"]:
            identity = dict(backend)
            if backend.get("binary"):
                binary = (ROOT / backend["binary"]).resolve()
                identity["binary"] = str(binary)
                identity["binary_sha256"] = hashlib.sha256(
                    binary.read_bytes()
                ).hexdigest()
                source = ROOT / backend.get("repository", str(binary.parent))
                identity["source_commit"] = subprocess.run(
                    ["git", "-C", str(source), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()
            elif backend.get("package"):
                identity["package_version"] = importlib.metadata.version(
                    backend["package"]
                )
            backends.append(identity)
        jobs = []
        for phase in config["phases"]:
            for instance in instances:
                for backend in backends:
                    if instance["generator"]["kind"] not in backend["kinds"]:
                        continue
                    if (
                        phase.get("backends")
                        and backend["name"] not in phase["backends"]
                    ):
                        continue
                    for seed in phase.get("seeds", config["seeds"]):
                        identity = f"{phase['name']}/{instance['id']}/{backend['name']}/seed-{seed}"
                        jobs.append(
                            {
                                "id": identity,
                                "phase": phase["name"],
                                "instance": instance,
                                "backend": backend,
                                "seed": seed,
                                "seconds": phase["seconds"],
                                "threads": backend.get("threads", 1),
                            }
                        )
        manifest = {
            "schema_version": "1.0",
            "config": config,
            "config_path": str(config_path),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "cpus": cpus,
            "instances": instances,
            "backends": backends,
            "jobs": jobs,
            "source_hashes": source_hashes,
            "python": sys.executable,
            "python_version": platform.python_version(),
            "created_at": time.time(),
        }
        write_json(output / "campaign.json", manifest)
        imported = {}
        known = {instance["id"]: instance for instance in instances}
        for source_name in config.get("prior_references", []):
            source = (ROOT / source_name).resolve()
            source_records = json.loads(source.read_text())["references"]
            if isinstance(source_records, dict):
                source_records = source_records.values()
            for record in source_records:
                instance = known.get(record["instance"])
                if instance is None or record["instance_sha256"] != instance["sha256"]:
                    continue
                solution = Path(record["solution"])
                if not solution.is_absolute():
                    solution = source.parent / solution
                if (
                    hashlib.sha256(solution.read_bytes()).hexdigest()
                    != record["solution_sha256"]
                ):
                    raise ValueError("prior reference solution hash mismatch")
                checked = verify_reference_file(
                    Path(instance["path"]), solution, instance["generator"]["kind"]
                )
                if (
                    not checked["feasible"]
                    or checked["objective"] != record["objective"]
                ):
                    raise ValueError("prior reference failed independent verification")
                imported[record["instance"]] = {
                    **record,
                    "solution": str(solution.resolve()),
                    "reference_source": str(source),
                    "verification": checked,
                }
        write_json(output / "prior_references.json", imported)
        write_json(output / "bks.json", {"references": imported})
        write_json(
            output / "status.json",
            {
                "state": "prepared",
                "instances": len(instances),
                "jobs": len(jobs),
                "cpus": cpus,
            },
        )
        return manifest

    @staticmethod
    def status(output: Path) -> dict:
        from collections import Counter

        output = output.resolve()
        manifest = json.loads((output / "campaign.json").read_text())
        prior = output / "prior_references.json"
        best = json.loads(prior.read_text()) if prior.exists() else {}
        finished, running = {}, []
        for directory in sorted((output / "jobs").glob("**/attempt-*")):
            incumbent = directory / "incumbent.json"
            if incumbent.exists():
                record = json.loads(incumbent.read_text())
                old = best.get(record["instance"])
                if old is None or (
                    record["objective"],
                    not record["proven_optimal"],
                ) < (old["objective"], not old["proven_optimal"]):
                    best[record["instance"]] = record
            result = directory / "result.json"
            if result.exists():
                record = json.loads(result.read_text())
                finished[record["job_id"]] = record
            elif (directory / "running.json").exists():
                running.append(json.loads((directory / "running.json").read_text()))
        counts = dict(Counter(r["execution_status"] for r in finished.values()))
        controller = output / "controller.lock"
        controller_pid = controller.read_text().strip() if controller.exists() else ""
        summary = {
            "updated_at": time.time(),
            "pid": int(controller_pid) if controller_pid else None,
            "instances": len(manifest["instances"]),
            "jobs": len(manifest["jobs"]),
            "finished_jobs": len(finished),
            "running_jobs": len(running),
            "pending_jobs": len(manifest["jobs"]) - len(finished) - len(running),
            "execution_counts": counts,
            "instances_with_reference": len(best),
            "proven_optimal": sum(r["proven_optimal"] for r in best.values()),
            "best_found": sum(not r["proven_optimal"] for r in best.values()),
            "cpus": manifest["cpus"],
            "running": running,
        }
        write_json(
            output / "bks.json",
            {
                "schema_version": "1.0",
                "updated_at": time.time(),
                "objective_sense": "minimize",
                "references": best,
            },
        )
        write_json(output / "status.json", summary)
        return summary

    @staticmethod
    def run(config_path: Path, output: Path) -> dict:
        import fcntl
        import threading

        output = output.resolve()
        manifest = ReferenceCollection.prepare(config_path, output)
        lockfile = (output / "controller.lock").open("a+")
        fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lockfile.seek(0)
        lockfile.truncate()
        lockfile.write(str(os.getpid()))
        lockfile.flush()
        ReferenceCollection.recover_interrupted(output)
        state_lock = threading.Lock()
        stop_heartbeat = threading.Event()
        snapshot = output / "code_snapshot"
        # Keep the source of old attempts while allowing corrected resumed workers.
        current_hashes = {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / "pitbench").rglob("*"))
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        }
        revision = hashlib.sha256(
            json.dumps(current_hashes, sort_keys=True).encode()
        ).hexdigest()
        if current_hashes != manifest["source_hashes"]:
            import shutil

            snapshot = output / "code_revisions" / revision
            if not snapshot.exists():
                shutil.copytree(
                    ROOT / "pitbench",
                    snapshot / "pitbench",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )
                write_json(
                    snapshot / "provenance.json",
                    {
                        "created_at": time.time(),
                        "source_hashes": current_hashes,
                        "revision": revision,
                    },
                )

        def heartbeat():
            while not stop_heartbeat.wait(30):
                with state_lock:
                    ReferenceCollection.status(output)

        ticker = threading.Thread(target=heartbeat, daemon=True)
        ticker.start()

        def execute(job, cpus):
            base = output / "jobs" / job["id"]
            attempts = sorted(base.glob("attempt-*"))
            if attempts and (attempts[-1] / "result.json").exists():
                previous = json.loads((attempts[-1] / "result.json").read_text())
                if previous["execution_status"] in {
                    "completed",
                    "skipped_proven_optimal",
                }:
                    return previous
            limits = manifest["config"].get("resource_guard", {})
            while limits and not (output / "STOP").exists():
                resource = reference_resource_state(limits, output)
                with state_lock:
                    write_json(
                        output / "resource_state.json",
                        {**resource, "time": time.time()},
                    )
                if not resource["blocked_by"]:
                    break
                time.sleep(15)
            directory = base / f"attempt-{len(attempts):04d}"
            directory.mkdir(parents=True)
            with state_lock:
                best_file = output / "bks.json"
                best = (
                    json.loads(best_file.read_text())["references"].get(
                        job["instance"]["id"]
                    )
                    if best_file.exists()
                    else None
                )
            if (best and best["proven_optimal"]) or (output / "STOP").exists():
                result = {
                    "job_id": job["id"],
                    "instance": job["instance"]["id"],
                    "execution_status": "skipped_proven_optimal"
                    if best and best["proven_optimal"]
                    else "stopped_before_start",
                    "proof_reference": best,
                    "cpus": cpus,
                }
                write_json(directory / "result.json", result)
            else:
                worker_job = {
                    **job,
                    "cpus": cpus,
                    "directory": str(directory),
                    "warm_start": best["solution"] if best else None,
                    "code_revision": revision,
                    "host_id": manifest.get("host_id"),
                }
                write_json(directory / "job.json", worker_job)
                command = [
                    manifest["python"],
                    str(snapshot / "pitbench/solver_drivers/reference.py"),
                    "--worker-job",
                    str(directory / "job.json"),
                ]
                environment = {
                    **os.environ,
                    "PYTHONPATH": str(snapshot),
                    "OMP_NUM_THREADS": str(len(cpus)),
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "MPLCONFIGDIR": str(output / "matplotlib"),
                }
                result = run_collection_process(
                    command,
                    directory,
                    {"job_id": job["id"], "instance": job["instance"]["id"]},
                    timeout=job["seconds"]
                    + manifest["config"].get("watchdog_grace_sec", 300),
                    environment=environment,
                )
            with state_lock:
                with (output / "attempts.jsonl").open("a") as stream:
                    stream.write(json.dumps(result) + "\n")
            return result

        try:
            for phase in manifest["config"]["phases"]:
                jobs = [
                    job for job in manifest["jobs"] if job["phase"] == phase["name"]
                ]
                random.Random(manifest["config"].get("schedule_seed", 0)).shuffle(jobs)
                run_collection_jobs(
                    jobs,
                    manifest["cpus"],
                    execute,
                    cpu_request=lambda job: job["threads"],
                )
                with state_lock:
                    ReferenceCollection.status(output)
                if (output / "STOP").exists():
                    break
        finally:
            stop_heartbeat.set()
            ticker.join()
            with state_lock:
                summary = ReferenceCollection.status(output)
            lockfile.close()
        return summary


class NuisanceCollection:
    """Configured axes and transformations, with one retained job/result contract."""

    @staticmethod
    def prepare(
        config_path: Path,
        output: Path,
        *,
        repository: Path | None = None,
        candidate_patch: Path | None = None,
        private_root: Path = ROOT / "private",
        panel: str | None = None,
    ) -> dict:
        import yaml

        from pitbench.evaluator.representation import (
            prepare_precomputed_transformations,
            prepare_transformations,
        )
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
        if candidate_patch is not None and config["execution"] != "judge":
            raise ValueError("candidate patches require judge execution")
        if (
            candidate_patch is None
            and config["execution"] == "judge"
            and "agent" in config["code_states"]
        ):
            output.mkdir(parents=True, exist_ok=True)
            candidate_patch = output / "empty.patch"
            if candidate_patch.exists():
                if candidate_patch.read_bytes():
                    raise ValueError("saved empty candidate patch has changed")
            else:
                candidate_patch.write_bytes(b"")
        if candidate_patch is not None:
            candidate_patch = candidate_patch.resolve(strict=True)
            candidate_patch_sha256 = hashlib.sha256(
                candidate_patch.read_bytes()
            ).hexdigest()
        else:
            candidate_patch_sha256 = None
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
                or type(task)
                .model_validate(manifest["task_configuration"])
                .model_dump(mode="json")
                != task.model_dump(mode="json")
                or manifest["repository"]
                != (str(repository.resolve()) if repository else None)
                or manifest.get("candidate_patch_sha256") != candidate_patch_sha256
            ):
                raise ValueError(
                    "existing experiment differs from requested configuration"
                )
            return manifest
        instances = prepare_collection_instances(
            ROOT / config["instance_source"],
            output / "sources",
            path_template=config.get("instance_path_template"),
            generated_stem_prefix=config.get("generated_stem_prefix"),
        )
        if config.get("reference_solver_commit") is not None:
            from pitbench.problem_families.verification import (
                validate_bin_packing_optimum_record,
            )

            for instance in instances:
                proof_path = instance.get("proof_path")
                solution_path = instance.get("bks_solution_path")
                if proof_path is None or solution_path is None:
                    raise ValueError("representation reference files are missing")
                proof = json.loads(Path(proof_path).read_text())
                source = Path(instance["path"])
                if (
                    proof["solver_commit"] != config["reference_solver_commit"]
                    or proof["adapter_source_sha256"]
                    != config["reference_adapter_source_sha256"]
                    or not 0
                    < proof["budget_sec"]
                    <= config["reference_proof_budget_sec"]
                ):
                    raise ValueError("representation reference proof run differs")
                payload = json.loads(source.read_text())
                reference_solution = json.loads(Path(solution_path).read_text())
                optimum = validate_bin_packing_optimum_record(
                    proof,
                    instance_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    weights=payload["weights"],
                    capacity=payload["capacity"],
                    reference_bins=reference_solution["bins"],
                )
                if (
                    instance.get("bks") != optimum
                    or instance.get("proven_optimal") is not True
                ):
                    raise ValueError("representation reference target differs")
                instance["reference_verified"] = True
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
        precomputed_manifest = config["representation"].get("precomputed_manifest")
        jobs, transformations = [], {}
        for instance in instances:
            if precomputed_manifest is not None:
                prepared = prepare_precomputed_transformations(
                    Path(instance["path"]),
                    instance["id"],
                    instance["bks"],
                    output,
                    transform.name,
                    ROOT / precomputed_manifest,
                    config["representation"]["count"],
                    generation_seed=config["representation"]["generation_seed"],
                    protocol_version=config["representation"].get("protocol_version"),
                    solver_commit=task.release.base_commit,
                    reference_solution=(
                        Path(instance["bks_solution_path"])
                        if instance.get("bks_solution_path")
                        else None
                    ),
                    proven_optimal=instance.get("proven_optimal"),
                )
            else:
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
                                "reference_verified": instance.get(
                                    "reference_verified", True
                                ),
                            }
                        )
        random.Random(config["schedule_seed"]).shuffle(jobs)
        manifest = {
            "config": config,
            "task_id": task.task_id,
            "task_configuration": task.model_dump(mode="json"),
            "solver": identity,
            "repository": str(repository.resolve()) if repository else None,
            "candidate_patch": str(candidate_patch) if candidate_patch else None,
            "candidate_patch_sha256": candidate_patch_sha256,
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
        prepare.add_argument("--candidate-patch", type=Path)
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
                candidate_patch=args.candidate_patch,
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
        candidate_patch = (
            Path(manifest["candidate_patch"])
            if manifest.get("candidate_patch") is not None
            else None
        )
        if (
            candidate_patch is not None
            and hashlib.sha256(candidate_patch.read_bytes()).hexdigest()
            != manifest["candidate_patch_sha256"]
        ):
            raise ValueError("candidate patch changed since collection preparation")
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
                    and observation.objective
                    == (record["verification"] or {}).get("objective")
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
                candidate_patch,
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

        ObservationStore.write_jsonl(
            output / "observations.jsonl",
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


class CollectionBackend(ABC):
    @staticmethod
    @abstractmethod
    def identity() -> dict: ...

    @staticmethod
    @abstractmethod
    def run(job: dict, directory: Path, result: dict) -> None: ...


def _external_identity(environment_key: str) -> dict:
    configured = os.environ.get(environment_key)
    if not configured:
        raise ValueError(f"missing external collector runner: {environment_key}")
    completed = subprocess.run(
        shlex.split(configured),
        input=json.dumps({"probe": True}),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode:
        raise RuntimeError(
            completed.stderr.strip()
            or f"external identity probe failed with {completed.returncode}"
        )
    try:
        identity = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("external identity probe returned invalid JSON") from error
    if not isinstance(identity, dict) or not isinstance(identity.get("version"), str):
        raise RuntimeError("external identity probe omitted a solver version")
    return identity


def _vroom_solver_command() -> str:
    return os.environ.get("PITBENCH_VROOM_SOLVER", "vroom")


def _vroom_identity() -> dict:
    command = shlex.split(_vroom_solver_command())
    completed = subprocess.run(
        [*command, "--version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode:
        raise RuntimeError(
            completed.stderr.strip()
            or f"VROOM identity probe failed with {completed.returncode}"
        )
    match = re.search(r"vroom\s+([0-9]+\.[0-9]+\.[0-9]+)", completed.stdout)
    if match is None:
        raise RuntimeError("VROOM identity probe omitted a semantic version")
    return {"version": match.group(1)}


def _verify_bin_packing_solution(instance_path: Path, solution_path: Path) -> dict:
    try:
        instance = json.loads(instance_path.read_text())
        solution = json.loads(solution_path.read_text())
        return _check_bin_packing_solution(
            solution, weights=instance["weights"], capacity=instance["capacity"]
        ).model_dump(mode="json")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return {"feasible": False, "detail": f"solution verification failed: {error}"}


def _validate_choco_parameters(parameters: dict) -> None:
    if set(parameters) != {"search_strategy"} or parameters["search_strategy"] not in {
        "input_order_lb",
        "min_dom_lb",
        "dom_over_wdeg",
        "activity_based",
    }:
        raise ParameterRejected("invalid Choco search_strategy parameter")


def _validate_cp_sat_parameters(parameters: dict) -> None:
    expected = {
        "cp_model_presolve",
        "cp_model_probing_level",
        "linearization_level",
        "symmetry_level",
        "search_branching",
    }
    if set(parameters) != expected:
        raise ParameterRejected("invalid CP-SAT parameter set")
    if type(parameters["cp_model_presolve"]) is not bool:
        raise ParameterRejected("cp_model_presolve must be boolean")
    for name, upper in (
        ("cp_model_probing_level", 2),
        ("linearization_level", 2),
        ("symmetry_level", 4),
    ):
        value = parameters[name]
        if type(value) is not int or not 0 <= value <= upper:
            raise ParameterRejected(f"{name} is outside the approved domain")
    if parameters["search_branching"] not in {
        "automatic_search",
        "lp_search",
        "pseudo_cost_search",
        "portfolio_search",
    }:
        raise ParameterRejected("search_branching is outside the approved domain")


def _run_external_configuration(
    driver,
    job: dict,
    directory: Path,
    result: dict,
    parameters: dict,
    verifier,
    solver_command: str | None = None,
) -> None:
    result["error_stage"] = "parameters"
    parameter_path = directory / "parameters.json"
    write_json(parameter_path, parameters)
    output = directory / "driver.json"
    result["error_stage"] = "solve"
    try:
        arguments = []
        if solver_command is not None:
            arguments.extend(["--solver", solver_command])
        arguments.extend(["--instance", job["instance"]["path"]])
        if driver.records_trajectory:
            arguments.extend(["--trajectory", str(directory / "trajectory.jsonl")])
        arguments.extend(
            [
                "--output",
                str(output),
                "--seed",
                str(job["solver_seed"]),
                "--budget",
                str(job["budget_sec"]),
                "--threads",
                str(job["threads"]),
                "--parameters",
                str(parameter_path),
            ]
        )
        driver.main(arguments)
    finally:
        if output.exists():
            result.update(json.loads(output.read_text()))
    result.update(
        execution_status="returned",
        model_status=result.get("solver_status"),
        termination=result.get("solver_termination"),
        solution_value_valid=result.get("has_solution", False),
    )
    result.setdefault("effective_parameters", parameters)
    result["error_stage"] = "verification"
    solution = output.with_suffix(".solution.json")
    result["solution_path"] = str(solution) if solution.exists() else None
    result["verification"] = (
        verifier(Path(job["instance"]["path"]), solution) if solution.exists() else None
    )
    if (result["verification"] or {}).get("infrastructure_error"):
        raise RuntimeError(
            "independent verifier infrastructure error: "
            + str(result["verification"].get("detail") or "no detail")
        )
    result["verified_feasible"] = bool((result["verification"] or {}).get("feasible"))
    result.pop("error_stage", None)


class ChocoCollectionBackend(CollectionBackend):
    """Apply the approved Choco search-strategy space through the JVM runner."""

    @staticmethod
    def identity() -> dict:
        return _external_identity("PITBENCH_CHOCO_RUNNER")

    @staticmethod
    def run(job: dict, directory: Path, result: dict) -> None:
        from pitbench.solver_drivers.run import ChocoDriver

        _validate_choco_parameters(job["parameters"])
        _run_external_configuration(
            ChocoDriver,
            job,
            directory,
            result,
            job["parameters"],
            _verify_bin_packing_solution,
        )


class VroomCollectionBackend(CollectionBackend):
    """Apply the approved VROOM exploration-level space."""

    @staticmethod
    def identity() -> dict:
        return _vroom_identity()

    @staticmethod
    def run(job: dict, directory: Path, result: dict) -> None:
        from pitbench.problem_families.verification import CVRPFamily
        from pitbench.solver_drivers.run import VroomDriver

        parameters = job["parameters"]
        if set(parameters) != {"exploration_level"}:
            raise ParameterRejected("invalid VROOM parameter set")
        if (
            type(parameters["exploration_level"]) is not int
            or not 0 <= parameters["exploration_level"] <= 5
        ):
            raise ParameterRejected("exploration_level must be an integer in [0, 5]")
        _run_external_configuration(
            VroomDriver,
            job,
            directory,
            result,
            parameters,
            lambda instance, solution: (
                CVRPFamily().verify(instance, solution).model_dump(mode="json")
            ),
            solver_command=_vroom_solver_command(),
        )


class OrToolsCpSatCollectionBackend(CollectionBackend):
    """Apply the approved CP-SAT solve-parameter space through the JVM runner."""

    @staticmethod
    def identity() -> dict:
        return _external_identity("PITBENCH_ORTOOLS_CP_SAT_RUNNER")

    @staticmethod
    def run(job: dict, directory: Path, result: dict) -> None:
        from pitbench.repositories.base import RepositoryPluginRegistry
        from pitbench.solver_drivers.run import OrToolsCpSatExactDriver

        _validate_cp_sat_parameters(job["parameters"])

        def verify_cp_sat(instance_path: Path, solution_path: Path) -> dict:
            repository = RepositoryPluginRegistry.load(job["repository_plugin"])
            with tempfile.TemporaryDirectory(
                prefix="pitbench-cp-sat-verifier-"
            ) as temporary:
                verifier = repository.prepare_independent_verifier(Path(temporary))
                if verifier is None:
                    raise RuntimeError("CP-SAT independent verifier is unavailable")
                return verifier.verify(instance_path, solution_path).model_dump(
                    mode="json"
                )

        _run_external_configuration(
            OrToolsCpSatExactDriver,
            job,
            directory,
            result,
            job["parameters"],
            verify_cp_sat,
        )


class PyVRPCollectionBackend(CollectionBackend):
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
            termination=result.get("solver_termination"),
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


class HighsCollectionBackend(CollectionBackend):
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
