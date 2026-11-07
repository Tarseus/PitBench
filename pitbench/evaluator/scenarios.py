"""Prepare, collect and verify source-bound metric scenarios with shared workers."""

from __future__ import annotations

import copy
import gc
import hashlib
import itertools
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import yaml

from pitbench.evaluator.artifacts import write_json
from pitbench.evaluator.collection import run_collection_jobs, run_collection_process
from pitbench.evaluator.docker_judge import docker_cli_image, judge_harness_sha256
from pitbench.evaluator.private_assets import PrivateAssetResolver
from pitbench.evaluator.representations import representation_type
from pitbench.evaluator.storage import cache_lock, file_manifest
from pitbench.metrics.scenario_report import (
    ReportedBoundTrace,
    compute_scenario_report,
    sequence_key,
)
from pitbench.schema.scenarios import (
    ScenarioCampaign,
    ScenarioEvent,
    ScenarioProtocol,
    ScenarioRun,
    ScenarioRunKey,
    ScenarioStep,
)
from pitbench.schema.task import PitBenchTask
from pitbench.solver_drivers.common import ScenarioSession, ScenarioUnsupported


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def task_digest(task: PitBenchTask) -> str:
    return digest(task.model_dump(mode="json"))


def _read(path):
    return yaml.safe_load(Path(path).read_text())


def resolve_model_inputs(
    config: dict, task: PitBenchTask, *, private_root: Path, config_dir: Path
) -> tuple[dict, list[dict]]:
    """Resolve transport paths through the declared asset namespace and frozen hashes."""
    resolved = copy.deepcopy(config)
    resolver = PrivateAssetResolver(private_root)
    mappings = {}
    panels = {panel.name: panel for panel in task.instance_sets}
    for protocol in task.metric_scenarios:
        for panel in protocol.instance_sets:
            cases = resolved["instances"][panel]
            if len(cases) != panels[panel].size or len({case["id"] for case in cases}) != len(cases):
                raise ValueError("scenario model inventory differs from the frozen task")
            frozen = protocol.model_sequences_sha256.get(panel, {})
            if set(frozen) != {case["id"] for case in cases}:
                raise ValueError("scenario instance identities differ from the frozen task")
            for case in cases:
                by_scenario = case.setdefault("models_by_scenario", {})
                names = by_scenario.get(protocol.id, case.get("models", []))
                hashes = frozen[case["id"]]
                if not names or len(names) != len(hashes):
                    raise ValueError("scenario model sequence differs from the frozen task")
                paths = []
                for name, expected in zip(names, hashes, strict=True):
                    known = mappings.get(name)
                    if known is not None:
                        if known["sha256"] != expected:
                            raise ValueError(f"conflicting frozen scenario hashes for {name}")
                        paths.append(known["resolved"])
                        continue
                    if name.startswith("private://"):
                        path = resolver.resolve(name, expected)
                    else:
                        path = Path(name)
                        path = (config_dir / path).resolve() if not path.is_absolute() else path.resolve()
                        if not path.is_file() and path.is_relative_to(resolver.logical_root):
                            relative = path.relative_to(resolver.logical_root).as_posix()
                            path = resolver.resolve("private://" + relative, expected)
                        if not path.is_file():
                            raise ValueError(f"scenario model input is missing: {name}")
                        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                            raise ValueError(f"scenario model input hash mismatch: {name}")
                    paths.append(str(path))
                    mappings[name] = {"declared": name, "resolved": str(path), "sha256": expected}
                by_scenario[protocol.id] = paths
    return resolved, list(mappings.values())


def _rss():
    fields = Path("/proc/self/statm").read_text().split()
    return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")


def _hardware_identity(cpus):
    machine_id = Path("/etc/machine-id")
    if not machine_id.is_file() or not machine_id.read_text().strip():
        raise ValueError(
            "scenario collection requires a real host machine-id; mount /etc/machine-id read-only when running in Docker"
        )
    return {
        "host_sha256": hashlib.sha256(machine_id.read_bytes()).hexdigest(),
        "architecture": platform.machine(),
        "cpu_model": next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            "unavailable",
        ),
        "cpu_pool": cpus,
        "memory_total_kib": next(
            line.split()[1]
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemTotal:")
        ),
        "cgroup_limits": {
            name: (Path("/sys/fs/cgroup") / name).read_text().strip()
            if (Path("/sys/fs/cgroup") / name).is_file()
            else None
            for name in ("cpu.max", "memory.max")
        },
    }


class _SampledPeak:
    def __init__(self, interval):
        self.interval = interval
        self.peak = 0
        self.stop = threading.Event()

    def __enter__(self):
        self.peak = _rss()

        def sample():
            while not self.stop.wait(self.interval):
                self.peak = max(self.peak, _rss())

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.peak = max(self.peak, _rss())
        self.stop.set()
        self.thread.join()


def prepare(config_path: Path, output: Path) -> dict:
    config_path, output = config_path.resolve(), output.resolve()
    config = _read(config_path)
    if output.exists():
        raise ValueError("scenario output already exists; resume its saved manifest")
    task_path = Path(config["task_config"]).resolve()
    payload = _read(task_path)
    if config.get("protocols"):
        if (
            payload.get("metric_scenarios")
            and payload["metric_scenarios"] != config["protocols"]
        ):
            raise ValueError("scenario overrides differ from the frozen task")
        payload["metric_scenarios"] = config["protocols"]
    task = PitBenchTask.model_validate(payload)
    protocols = task.metric_scenarios
    if not protocols:
        raise ValueError("at least one explicit metric scenario protocol is required")
    if any(
        representation_type(p.representation).family != task.problem_family
        or p.objective_sense != task.oracle.objective_sense
        for p in protocols
    ):
        raise ValueError(
            "scenario representation or objective sense differs from the task"
        )
    panels = {item.name: item for item in task.instance_sets}
    if any(set(p.instance_sets) - set(panels) for p in protocols):
        raise ValueError("scenario refers to an unknown task panel")
    cpus = config["cpus"]
    if (
        not cpus
        or len(set(cpus)) != len(cpus)
        or not set(cpus) <= os.sched_getaffinity(0)
    ):
        raise ValueError("scenario CPU pool must be explicitly available and unique")
    if max(n for p in protocols for n in p.thread_counts) > len(cpus):
        raise ValueError("thread grid exceeds the explicitly allocated CPU pool")
    candidate = Path(config["candidate_patch"]).resolve()
    candidate_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
    order = config["order"]
    if order not in {"base_first", "agent_first"}:
        raise ValueError("declare Base/Agent order")
    runtimes, identities = {}, {}
    for state in ("base", "agent"):
        runtime = dict(config["runtimes"][state])
        if runtime["source_commit"] != task.release.base_commit:
            raise ValueError("scenario runtime is not bound to the task source")
        expected_patch = (
            hashlib.sha256(b"").hexdigest() if state == "base" else candidate_hash
        )
        if runtime["patch_sha256"] != expected_patch:
            raise ValueError(
                "scenario runtime is not bound to the selected candidate patch"
            )
        probe = subprocess.run(
            [
                runtime["python"],
                "-m",
                "scripts.collect_metric_scenarios",
                "probe",
                "--session-plugin",
                runtime["session_plugin"],
                *[
                    arg
                    for name in sorted({p.representation for p in protocols})
                    for arg in ("--representation", name)
                ],
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
            env={**os.environ, **runtime.get("environment", {})},
        )
        identity = json.loads(probe.stdout.strip().splitlines()[-1])
        if identity.get("repository_plugin") != task.repository.plugin:
            raise ValueError(
                "scenario API runtime belongs to another solver repository"
            )
        if identity.get(
            "version"
        ) != task.release.version or not task.release.base_commit.startswith(
            identity.get("binding_git_hash") or "missing"
        ):
            raise ValueError(
                "scenario API runtime differs from the declared source release"
            )
        if runtime.get("expected_binding_sha256") != identity.get("binding_sha256"):
            raise ValueError(
                "scenario binding differs from the declared built artifact"
            )
        if any(
            p.representation not in identity.get("representations", [])
            for p in protocols
        ):
            raise ValueError("scenario API does not support a declared representation")
        identities[state] = digest(
            {
                "probe": identity,
                "source_commit": runtime["source_commit"],
                "patch_sha256": runtime["patch_sha256"],
                "options": runtime.get("options", {}),
                **(
                    {"options_by_scenario": runtime["options_by_scenario"]}
                    if runtime.get("options_by_scenario")
                    else {}
                ),
                "environment": runtime.get("environment", {}),
            }
        )
        runtimes[state] = {**runtime, "probe": identity}
    if candidate_hash != hashlib.sha256(b"").hexdigest() and all(
        runtimes["base"]["probe"].get(field) == runtimes["agent"]["probe"].get(field)
        for field in ("binding_sha256", "package_code_sha256")
    ):
        raise ValueError(
            "nonempty candidate patch has no distinct built API artifact; refuse to measure the installed Base binding as Agent"
        )
    sequences, instances = {}, {}
    wanted = {panel for p in protocols for panel in p.instance_sets}
    for panel in wanted:
        cases = config["instances"][panel]
        if len(cases) != panels[panel].size:
            raise ValueError(
                "scenario inventory must retain the full declared task panel"
            )
        instances[panel] = [case["id"] for case in cases]
        if len(set(instances[panel])) != len(cases):
            raise ValueError("scenario instance IDs must be unique within a panel")
        for case in cases:
            for protocol in [p for p in protocols if panel in p.instance_sets]:
                files = []
                for name in case.get("models_by_scenario", {}).get(
                    protocol.id, case.get("models", [])
                ):
                    path = Path(name).resolve()
                    files.append(
                        {
                            "path": str(path),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        }
                    )
                if not files:
                    raise ValueError("scenario model sequences cannot be empty")
                if protocol.model_sequences_sha256.get(panel, {}).get(case["id"]) != [
                    item["sha256"] for item in files
                ]:
                    raise ValueError(
                        "scenario model sequence was not frozen in the task protocol"
                    )
                sequences[sequence_key(panel, case["id"], protocol.id)] = files
    expected = []
    for p in protocols:
        for panel in p.instance_sets:
            for iid in instances[panel]:
                if (
                    p.kind == "scaling" or "time_to_target_sec" in p.metrics
                ) and iid not in p.target_by_instance:
                    raise ValueError("missing preregistered quality target")
                if any("integral" in m for m in p.metrics) and (
                    iid not in p.reference_by_instance
                    or iid not in p.loss_scale_by_instance
                ):
                    raise ValueError("missing integral normalization for an instance")
                for state, seed, threads, repeat, mode in itertools.product(
                    ("base", "agent"),
                    p.solver_seeds,
                    p.thread_counts,
                    range(p.repeats),
                    p.modes,
                ):
                    expected.append(
                        ScenarioRunKey(
                            scenario_id=p.id,
                            task_id=task.task_id,
                            instance_set=panel,
                            instance_id=iid,
                            code_state=state,
                            solver_seed=seed,
                            threads=threads,
                            repeat=repeat,
                            mode=mode,
                        )
                    )
    output.mkdir(parents=True)
    for sequence in sequences.values():
        for item in sequence:
            source = Path(item["path"])
            path = output / "instances" / (item["sha256"] + source.suffix)
            path.parent.mkdir(exist_ok=True)
            if not path.exists():
                shutil.copy2(source, path)
            item["path"] = str(path)
    campaign = ScenarioCampaign(
        task_id=task.task_id,
        task_sha256=task_digest(task),
        candidate_sha256=candidate_hash,
        protocols=protocols,
        expected_runs=expected,
        runs=[],
        model_sequences=sequences,
        instances_by_set=instances,
        runtime_identities=identities,
        hardware_identity=_hardware_identity(cpus),
        order=order,
    )
    write_json(output / "campaign.json", campaign.model_dump(mode="json"))
    write_json(
        output / "runtime.json",
        {
            "runtimes": runtimes,
            "cpus": cpus,
            "watchdog_grace_sec": config["watchdog_grace_sec"],
            "base_cache_path": config.get("base_cache_path"),
        },
    )
    (output / "task.yaml").write_text(
        yaml.safe_dump(task.model_dump(mode="json"), sort_keys=False)
    )
    return {
        "jobs": len(expected),
        "protocols": len(protocols),
        "task_sha256": campaign.task_sha256,
    }


def worker(job_path: Path) -> dict:
    job = json.loads(job_path.read_text())
    os.sched_setaffinity(0, job["cpus"])
    p = ScenarioProtocol.model_validate(job["protocol"])
    key = ScenarioRunKey.model_validate(job["key"])
    directory = job_path.parent
    runtime = job["runtime"]
    implementation = ScenarioSession.load(runtime["session_plugin"])
    steps, error, status = [], None, "returned"
    completed_steps = 0
    try:
        if implementation.probe() != runtime["probe"]:
            raise ValueError("worker runtime artifact changed after preparation")
        session, previous = None, None
        for index, item in enumerate(job["models"]):
            path = Path(item["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("scenario model changed after preparation")
            model = representation_type(p.representation).read_input(path)
            setup_time, update_time = 0.0, 0.0
            with _SampledPeak(p.memory_sample_interval_sec) as memory:
                started = time.perf_counter()
                if session is None or key.mode == "cold":
                    session = implementation(
                        seed=key.solver_seed,
                        threads=key.threads,
                        budget=p.budget_sec,
                        tolerance=p.feasibility_tolerance,
                        options={
                            **runtime.get("options", {}),
                            **runtime.get("options_by_scenario", {}).get(p.id, {}),
                        },
                    )
                    session.load_model(model)
                    setup_time = time.perf_counter() - started
                else:
                    session.update_model(previous, model)
                    update_time = time.perf_counter() - started
            events = []
            session.trace_directory = directory
            session.trace_step = index

            def emit(timestamp, values, multipliers):
                event = ScenarioEvent(
                    time_sec=timestamp,
                    solution_values=values if not isinstance(values, dict) else None,
                    solution=values if isinstance(values, dict) else None,
                    row_multipliers=multipliers,
                )
                events.append(event)
                with (directory / "trajectory.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            {"step": index, **event.model_dump(mode="json")},
                            allow_nan=False,
                        )
                        + "\n"
                    )

            result = session.solve(emit)
            payload = ScenarioStep(
                instance_path=str(path),
                instance_sha256=item["sha256"],
                process_id=os.getpid(),
                process_boot_id=Path("/proc/sys/kernel/random/boot_id")
                .read_text()
                .strip(),
                setup_time_sec=setup_time,
                setup_peak_rss_bytes=memory.peak,
                solve_time_sec=result["solve_time_sec"],
                update_time_sec=update_time,
                retained_rss_bytes=0,
                termination=result["termination"],
                events=events,
                final_solution_values=result.get("values"),
                final_solution=result.get("solution"),
            ).model_dump(mode="json")
            write_json(directory / f"step-{index:04d}.payload.json", payload)
            # Do not retain an ever-growing observer history inside the measured
            # long-lived process. Reconstruct final records after sampling ends.
            del payload, result
            events.clear()
            previous = model
            if p.reset_policy == "clear_search":
                session.reset_search()
            if p.memory_cleanup == "gc_collect":
                gc.collect()
            write_json(
                directory / f"step-{index:04d}.memory.json",
                {"retained_rss_bytes": _rss()},
            )
            completed_steps += 1
    except ScenarioUnsupported as exc:
        status, error = "unsupported", str(exc)
    except Exception as exc:
        status, error = "collector_error", f"{type(exc).__name__}: {exc}"
    for index in range(completed_steps):
        payload = json.loads((directory / f"step-{index:04d}.payload.json").read_text())
        payload.update(
            json.loads((directory / f"step-{index:04d}.memory.json").read_text())
        )
        step = ScenarioStep.model_validate(payload)
        steps.append(step)
        write_json(directory / f"step-{index:04d}.json", step.model_dump(mode="json"))
    run = ScenarioRun(
        key=key,
        protocol_sha256=p.digest(),
        runtime_identity_sha256=job["runtime_identity_sha256"],
        status=status,
        steps=steps,
        error=error,
    )
    payload = {
        "execution_status": "completed" if status == "returned" else "collector_error",
        "run": run.model_dump(mode="json"),
    }
    write_json(directory / "result.json", payload)
    return payload


def run(output: Path, *, retry_errors: bool = False) -> dict:
    output = output.resolve()
    campaign = ScenarioCampaign.model_validate_json(
        (output / "campaign.json").read_text()
    )
    if campaign.started_at is None:
        campaign.started_at = time.time()
    runtime = json.loads((output / "runtime.json").read_text())
    collector_sha256 = (
        judge_harness_sha256(Path(__file__).resolve().parents[2])
        if runtime.get("base_cache_path")
        else None
    )
    if campaign.hardware_identity != _hardware_identity(runtime["cpus"]):
        raise ValueError(
            "scenario hardware or resource allocation changed after preparation"
        )
    protocols = {p.id: p for p in campaign.protocols}
    existing = {record.key: record for record in campaign.runs}
    launched = threading.Event()
    states = ["base", "agent"] if campaign.order == "base_first" else ["agent", "base"]

    def execute_uncached(key, cpus):
        p = protocols[key.scenario_id]
        directory = output / "jobs" / digest(key.model_dump(mode="json"))
        attempts = sorted(directory.glob("attempt-*/result.json"))
        if attempts:
            saved = json.loads(attempts[-1].read_text())
            if saved.get("run"):
                old = ScenarioRun.model_validate(saved["run"])
                if old.key != key or old.protocol_sha256 != p.digest():
                    raise ValueError(
                        "saved scenario attempt differs from the frozen job"
                    )
                if old.status == "returned" or not retry_errors:
                    return old
            elif not retry_errors:
                return ScenarioRun(
                    key=key,
                    protocol_sha256=p.digest(),
                    runtime_identity_sha256=campaign.runtime_identities[key.code_state],
                    status="watchdog_timeout"
                    if saved.get("execution_status") == "watchdog_timeout"
                    else "process_error",
                    error=str(saved.get("error", saved.get("execution_status"))),
                )
        if p.kind == "setup":
            signature = execution_signature(key)
            # Select by the frozen plan, never by completion time or success.
            # A shared failure remains a failure rather than a hidden retry.
            source = next(
                (
                    existing[planned]
                    for planned in campaign.expected_runs
                    if protocols[planned.scenario_id].kind != "setup"
                    and execution_signature(planned) == signature
                    and planned in existing
                ),
                None,
            )
            if source is not None:
                reused = source.model_copy(
                    update={
                        "key": key,
                        "protocol_sha256": p.digest(),
                        "reused_from": source.key,
                    }
                )
                attempt = (
                    directory / f"attempt-{len(list(directory.glob('attempt-*'))):04d}"
                )
                write_json(
                    attempt / "result.json",
                    {
                        "execution_status": "reused",
                        "run": reused.model_dump(mode="json"),
                    },
                )
                return reused
        attempt = directory / f"attempt-{len(list(directory.glob('attempt-*'))):04d}"
        launched.set()
        attempt.mkdir(parents=True)
        selected = runtime["runtimes"][key.code_state]
        models = campaign.model_sequences[
            sequence_key(key.instance_set, key.instance_id, p.id)
        ]
        job = {
            "key": key.model_dump(mode="json"),
            "protocol": p.model_dump(mode="json"),
            "models": models,
            "runtime": selected,
            "cpus": cpus,
            "runtime_identity_sha256": campaign.runtime_identities[key.code_state],
        }
        write_json(attempt / "job.json", job)
        result = run_collection_process(
            [
                selected["python"],
                "-m",
                "scripts.collect_metric_scenarios",
                "worker",
                "--job",
                str(attempt / "job.json"),
            ],
            attempt,
            {"scenario_run": key.model_dump(mode="json")},
            timeout=len(models) * p.budget_sec + runtime["watchdog_grace_sec"],
            environment={
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                **selected.get("environment", {}),
            },
        )
        if result.get("run"):
            return ScenarioRun.model_validate(result["run"])
        failed = ScenarioRun(
            key=key,
            protocol_sha256=p.digest(),
            runtime_identity_sha256=campaign.runtime_identities[key.code_state],
            status="watchdog_timeout"
            if result.get("execution_status") == "watchdog_timeout"
            else "process_error",
            error=str(result.get("error", result.get("execution_status"))),
        )
        write_json(
            attempt / "result.json", {**result, "run": failed.model_dump(mode="json")}
        )
        return failed

    def execution_signature(key):
        p = protocols[key.scenario_id]
        selected = runtime["runtimes"][key.code_state]
        return digest(
            {
                "state": key.code_state,
                "instance_set": key.instance_set,
                "instance_id": key.instance_id,
                "seed": key.solver_seed,
                "threads": key.threads,
                "repeat": key.repeat,
                "mode": key.mode,
                "models": [
                    row["sha256"]
                    for row in campaign.model_sequences[
                        sequence_key(key.instance_set, key.instance_id, p.id)
                    ]
                ],
                "budget": p.budget_sec,
                "tolerance": p.feasibility_tolerance,
                "sample_interval": p.memory_sample_interval_sec,
                "reset": p.reset_policy,
                "cleanup": p.memory_cleanup,
                "options": {
                    **selected.get("options", {}),
                    **selected.get("options_by_scenario", {}).get(p.id, {}),
                },
                "runtime": campaign.runtime_identities[key.code_state],
                "python": selected.get("python"),
                "session_plugin": selected.get("session_plugin"),
                "environment_sha256": digest(selected.get("environment", {})),
                "watchdog_grace_sec": runtime["watchdog_grace_sec"],
            }
        )

    def execute(key, cpus):
        cache = runtime.get("base_cache_path")
        if key.code_state != "base" or not cache:
            return execute_uncached(key, cpus)
        hardware = {
            k: v for k, v in campaign.hardware_identity.items() if k != "cpu_pool"
        }
        hardware["boot_id"] = (
            Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        )
        identity = {
            "collector_sha256": collector_sha256,
            "task_sha256": campaign.task_sha256,
            "protocol_sha256": protocols[key.scenario_id].digest(),
            "hardware": hardware,
            "cpu_count": len(runtime["cpus"]),
            "execution": execution_signature(key),
        }
        cache_key = digest(identity)
        entry = Path(cache) / cache_key
        directory = output / "jobs" / digest(key.model_dump(mode="json"))
        with cache_lock(Path(cache) / (cache_key + ".lock")):
            receipt_path = entry / "ready.json"
            saved = _read(receipt_path) if receipt_path.is_file() else None
            own_attempts = sorted(directory.glob("attempt-*/result.json"))
            if (
                not own_attempts
                and saved
                and saved["identity"] == identity
                and file_manifest(Path(saved["raw_artifacts"])) == saved["files"]
            ):
                record = ScenarioRun.model_validate(saved["run"])
                models = campaign.model_sequences[
                    sequence_key(key.instance_set, key.instance_id, key.scenario_id)
                ]
                record = record.model_copy(
                    update={
                        "key": key,
                        "steps": [
                            step.model_copy(update={"instance_path": model["path"]})
                            for step, model in zip(record.steps, models, strict=True)
                        ],
                        "measurement_source": {
                            "cache_key": cache_key,
                            "receipt": saved["receipt"],
                            "measurement_id": saved["measurement_id"],
                            "collection_id": saved["collection_id"],
                            "reused": True,
                        },
                    }
                )
                write_json(
                    directory / "attempt-0000/result.json",
                    {
                        "execution_status": "reused_base",
                        "run": record.model_dump(mode="json"),
                    },
                )
                return record
            record = execute_uncached(key, cpus)
            if (
                record.status == "returned"
                and record.reused_from is None
                and record.measurement_source is None
            ):
                attempt = sorted(directory.glob("attempt-*/result.json"))[-1].parent
                measurement_id = uuid.uuid4().hex
                artifacts = entry / ("artifacts-" + measurement_id)
                measurement_receipt = entry / (
                    "measurement-" + measurement_id + ".json"
                )
                shutil.copytree(attempt, artifacts)
                for index, model in enumerate(
                    campaign.model_sequences[
                        sequence_key(key.instance_set, key.instance_id, key.scenario_id)
                    ]
                ):
                    target = (
                        artifacts / "models" / (str(index) + Path(model["path"]).suffix)
                    )
                    target.parent.mkdir(exist_ok=True)
                    shutil.copy2(model["path"], target)
                saved = {
                    "identity": identity,
                    "run": record.model_dump(mode="json"),
                    "raw_artifacts": str(artifacts),
                    "files": file_manifest(artifacts),
                    "collection_id": campaign.collection_id,
                    "source_campaign": str(output / "campaign.json"),
                    "receipt": str(measurement_receipt),
                    "measurement_id": measurement_id,
                }
                write_json(measurement_receipt, saved)
                write_json(receipt_path, saved)
                record = record.model_copy(
                    update={
                        "measurement_source": {
                            "cache_key": cache_key,
                            "receipt": str(measurement_receipt),
                            "measurement_id": measurement_id,
                            "collection_id": campaign.collection_id,
                            "reused": False,
                        }
                    }
                )
                write_json(
                    attempt / "result.json",
                    {
                        "execution_status": "completed",
                        "run": record.model_dump(mode="json"),
                    },
                )
            return record

    for state in states:
        for setup in (False, True):
            keys = [
                key
                for key in campaign.expected_runs
                if key.code_state == state
                and (protocols[key.scenario_id].kind == "setup") == setup
            ]
            keys.sort(
                key=lambda key: (
                    -protocols[key.scenario_id].budget_sec
                    * len(
                        campaign.model_sequences[
                            sequence_key(
                                key.instance_set, key.instance_id, key.scenario_id
                            )
                        ]
                    )
                )
            )
            records = run_collection_jobs(
                keys, runtime["cpus"], execute, cpu_request=lambda key: key.threads
            )
            existing.update({record.key: record for record in records})
            campaign.runs = list(existing.values())
            write_json(output / "campaign.json", campaign.model_dump(mode="json"))
    if launched.is_set() or campaign.finished_at is None:
        campaign.finished_at = time.time()
    write_json(output / "campaign.json", campaign.model_dump(mode="json"))
    return report(output)


def load_reported_bound_traces(
    output: Path,
    campaign: ScenarioCampaign,
    *,
    native_infinity: float | None = None,
) -> dict[ScenarioRunKey, ReportedBoundTrace]:
    """Recover native diagnostic bounds, retaining trace hashes and cache provenance."""
    if native_infinity is not None and (
        not math.isfinite(native_infinity) or native_infinity <= 0
    ):
        raise ValueError("native bound infinity must be finite and positive")
    protocols = {p.id: p for p in campaign.protocols}
    traces = {}
    for run in campaign.runs:
        p = protocols[run.key.scenario_id]
        if (
            p.kind != "trajectory"
            or "dual_progress_integral" not in p.metrics
            or run.status != "returned"
            or len(run.steps) != 1
        ):
            continue
        directory = output / "jobs" / digest(run.key.model_dump(mode="json"))
        attempts = sorted(directory.glob("attempt-*/result.json"))
        source = attempts[-1].parent if attempts else None
        manifest = None
        if run.measurement_source:
            receipt = json.loads(Path(run.measurement_source["receipt"]).read_text())
            saved = ScenarioRun.model_validate(receipt["run"])
            source = Path(receipt["raw_artifacts"])
            manifest = receipt["files"]
        elif attempts:
            saved = ScenarioRun.model_validate(json.loads(attempts[-1].read_text())["run"])
        else:
            continue
        if (
            saved.key != run.key
            or saved.protocol_sha256 != run.protocol_sha256
            or saved.runtime_identity_sha256 != run.runtime_identity_sha256
            or [step.instance_sha256 for step in saved.steps]
            != [step.instance_sha256 for step in run.steps]
        ):
            raise ValueError("reported bound trace belongs to another scenario execution")
        path = source / "native-progress.jsonl"
        if not path.is_file():
            continue
        raw = path.read_bytes()
        if manifest is not None:
            fingerprint = hashlib.sha256(
                str(path.lstat().st_mode & 0o777).encode() + b"\0" + raw
            ).hexdigest()
            if manifest.get(path.name) != fingerprint:
                raise ValueError("shared reported bound trace hash mismatch")
        events, observed, unavailable = [], 0, 0
        previous = -1.0
        for line in raw.splitlines():
            row = json.loads(line)
            if row["step"] != 0:
                raise ValueError("trajectory bound trace contains another model step")
            timestamp = float(row["time_sec"])
            if (
                not math.isfinite(timestamp)
                or timestamp < previous
                or timestamp < 0
                or timestamp > run.steps[0].solve_time_sec
            ):
                raise ValueError("reported bound timestamps are outside the measured solve")
            previous = timestamp
            if timestamp > p.budget_sec:
                continue
            observed += 1
            value = row.get("dual_bound")
            if value is None or not math.isfinite(float(value)) or (
                native_infinity is not None and abs(float(value)) >= native_infinity
            ):
                unavailable += 1
                continue
            events.append((timestamp, float(value)))
        traces[run.key] = ReportedBoundTrace(
            events=events,
            source_path=str(path.resolve()),
            source_sha256=hashlib.sha256(raw).hexdigest(),
            observed_events=observed,
            unavailable_bound_events=unavailable,
            native_infinity=native_infinity,
        )
    return traces


def report(
    output: Path,
    *,
    scaling_time_policy: str = "observed",
    dual_bound_source: str = "independently_verified",
    reported_bound_infinity: float | None = None,
    report_path: Path | None = None,
) -> dict:
    campaign = ScenarioCampaign.model_validate_json(
        (output / "campaign.json").read_text()
    )
    traces = (
        load_reported_bound_traces(output, campaign, native_infinity=reported_bound_infinity)
        if dual_bound_source == "solver_reported"
        else None
    )
    result = compute_scenario_report(
        campaign,
        scaling_time_policy=scaling_time_policy,
        dual_bound_source=dual_bound_source,
        reported_bound_traces=traces,
    )
    suffix = "-budget-capped-v1" if scaling_time_policy == "budget_capped_v1" else ""
    if dual_bound_source == "solver_reported":
        suffix += "-solver-reported"
    destination = report_path or output / f"scenario-report{suffix}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, result.model_dump(mode="json"))
    return {
        "expected_runs": len(campaign.expected_runs),
        "recorded_runs": len(campaign.runs),
        "complete": result.complete,
        "invalid_runs": len(result.invalid_runs),
        "reused_runs": sum(run.reused_from is not None for run in campaign.runs),
        "shared_base_runs": sum(
            bool(run.measurement_source and run.measurement_source.get("reused"))
            for run in campaign.runs
        ),
        "metrics": len(result.results),
        "scaling_time_policy": scaling_time_policy,
        "dual_bound_source": dual_bound_source,
        "report": str(destination),
    }


def load_report(campaign_path: Path, task: PitBenchTask):
    campaign = ScenarioCampaign.model_validate_json(campaign_path.read_text())
    if campaign.task_id != task.task_id or campaign.task_sha256 != task_digest(task):
        raise ValueError("scenario campaign belongs to another frozen task")
    if campaign.protocols != task.metric_scenarios:
        raise ValueError("scenario protocols differ from the frozen task")
    panels = {panel.name: panel for panel in task.instance_sets}
    for protocol in task.metric_scenarios:
        for panel in protocol.instance_sets:
            instances = campaign.instances_by_set.get(panel, [])
            if panel not in panels or len(instances) != panels[panel].size:
                raise ValueError("scenario inventory differs from the frozen task")
            frozen = protocol.model_sequences_sha256.get(panel)
            if frozen is not None and set(instances) != set(frozen):
                raise ValueError("scenario instance identities differ from the frozen task")
    return compute_scenario_report(campaign)


def scenario_evidence_failures(task: PitBenchTask, output_dir: Path, payload: dict) -> list[str]:
    """Recheck requested scenarios from raw campaigns, independent of report flags."""
    failures = []
    artifacts = payload.get("artifacts", {})
    summary = payload.get("summary", {})
    if task.metric_scenarios:
        if summary.get("metric_scenario_error"):
            failures.append("metric scenario collection failed: " + str(summary["metric_scenario_error"]))
        campaign_ref = artifacts.get("metric_scenario_campaign")
        if not isinstance(campaign_ref, dict) or not campaign_ref.get("path"):
            failures.append("metric scenario campaign is missing")
        else:
            campaign_path = (output_dir / campaign_ref["path"]).resolve()
            try:
                if not campaign_path.is_relative_to(output_dir.resolve()):
                    raise ValueError("campaign artifact escapes evaluation output")
                if hashlib.sha256(campaign_path.read_bytes()).hexdigest() != campaign_ref.get("sha256"):
                    raise ValueError("campaign artifact hash mismatch")
                report = load_report(campaign_path, task)
                candidate_ref = artifacts.get("candidate_patch") or {}
                candidate_path = (output_dir / candidate_ref.get("path", "candidate.patch")).resolve()
                if not candidate_path.is_relative_to(output_dir.resolve()):
                    raise ValueError("candidate artifact escapes evaluation output")
                if hashlib.sha256(candidate_path.read_bytes()).hexdigest() != report.candidate_sha256:
                    raise ValueError("scenario campaign belongs to another candidate patch")
                if not report.complete:
                    failures.append("metric scenario report is incomplete")
                if report.invalid_runs:
                    failures.append(f"{len(report.invalid_runs)} invalid metric scenario runs")
            except (OSError, ValueError, TypeError, KeyError) as error:
                failures.append(f"metric scenario evidence is invalid: {error}")
    return failures


def collect_for_evaluation(
    task: PitBenchTask,
    candidate_patch: Path,
    output: Path,
    config_path: Path,
    *,
    reverse: bool = False,
    build_context: dict | None = None,
):
    """Collect the explicit API scenarios alongside a normal independent Judge."""
    config = _read(config_path)
    if config.get("source_build"):
        return _collect_source_build(
            task, candidate_patch, output, config_path, build_context, reverse
        )
    if config.get("protocols") and [
        ScenarioProtocol.model_validate(p).model_dump(mode="json")
        for p in config["protocols"]
    ] != [p.model_dump(mode="json") for p in task.metric_scenarios]:
        raise ValueError("evaluation scenario settings differ from the frozen task")
    output.mkdir(parents=True, exist_ok=True)
    task_path = output / "scenario-task.yaml"
    task_path.write_text(yaml.safe_dump(task.model_dump(mode="json"), sort_keys=False))
    config.update(
        task_config=str(task_path),
        candidate_patch=str(candidate_patch.resolve()),
        protocols=[p.model_dump(mode="json") for p in task.metric_scenarios],
    )
    if reverse:
        config["order"] = (
            "agent_first" if config["order"] == "base_first" else "base_first"
        )
    saved_config = output / "scenario-config.yaml"
    if saved_config.exists() and _read(saved_config) != config:
        raise ValueError(
            "saved scenario runtime configuration differs from this evaluation"
        )
    saved_config.write_text(yaml.safe_dump(config, sort_keys=False))
    destination = output / "metric-scenarios"
    if not destination.exists():
        prepare(saved_config, destination)
    else:
        saved = ScenarioCampaign.model_validate_json(
            (destination / "campaign.json").read_text()
        )
        if (
            saved.candidate_sha256
            != hashlib.sha256(candidate_patch.read_bytes()).hexdigest()
        ):
            raise ValueError("saved scenario campaign belongs to another candidate")
    run(destination)
    return load_report(
        destination / "campaign.json", task
    ), destination / "campaign.json"


def _collect_source_build(task, candidate_patch, output, config_path, context, reverse):
    """Build and collect in the same pinned solver image as the independent Judge."""
    if not context or not context.get("image"):
        raise ValueError("source-bound collection requires an evaluation build context")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    code_root = Path(__file__).resolve().parents[2]
    source_config = Path(config_path).resolve()
    config, inputs = resolve_model_inputs(
        _read(source_config),
        task,
        private_root=Path(context["private_root"]),
        config_dir=source_config.parent,
    )
    if context.get("base_cache_path"):
        config["base_cache_path"] = str(
            Path(context["base_cache_path"]).resolve() / "scenarios"
        )
    config_path = output / "resolved-scenario-config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(
        output / "scenario-input-resolution.json",
        {
            "source_config": str(source_config),
            "source_config_sha256": hashlib.sha256(
                source_config.read_bytes()
            ).hexdigest(),
            "resolved_config_sha256": hashlib.sha256(
                config_path.read_bytes()
            ).hexdigest(),
            "model_inputs": inputs,
        },
    )
    private_root = PrivateAssetResolver(Path(context["private_root"])).root
    task_path = output / "scenario-source-task.yaml"
    task_path.write_text(yaml.safe_dump(task.model_dump(mode="json"), sort_keys=False))
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--cpus",
        str(context["cpus"]),
        "--memory",
        context["memory"],
        "--env",
        "PYTHONPATH=" + str(code_root),
        "--workdir",
        str(code_root),
    ]
    if context.get("build_cache_path"):
        cache = Path(context["build_cache_path"]).resolve()
        cache.mkdir(parents=True, exist_ok=True)
        command += [
            "--env",
            f"PITBENCH_BUILD_CACHE={cache}",
            "--env",
            "PITBENCH_BUILD_CONTEXT="
            + json.dumps(
                {
                    "judge_image": context["image"],
                    "harness_sha256": judge_harness_sha256(code_root),
                    "cpus": float(context["cpus"]),
                    "memory": context["memory"],
                }
            ),
        ]
    if context.get("cpuset_cpus"):
        command += ["--cpuset-cpus", context["cpuset_cpus"]]
    mounts = {
        str(code_root): "ro",
        str(Path(context["base_repository"]).resolve()): "ro",
        str(private_root): "ro",
        str(config_path.resolve()): "ro",
        str(candidate_patch.resolve()): "ro",
        str(output): "rw",
        "/etc/machine-id": "ro",
    }
    if context.get("build_cache_path"):
        mounts[str(Path(context["build_cache_path"]).resolve())] = "rw"
    if context.get("base_cache_path"):
        cache = Path(context["base_cache_path"]).resolve()
        cache.mkdir(parents=True, exist_ok=True)
        mounts[str(cache)] = "rw"
    mounted = [Path(name) for name in mounts]
    for item in inputs:
        path = Path(item["resolved"])
        if not any(path.is_relative_to(root) for root in mounted):
            mounts[str(path)] = "ro"
    for path, mode in mounts.items():
        command += ["--volume", path + ":" + path + ":" + mode]
    command += [
        docker_cli_image(context["image"]),
        "python3",
        "-m",
        "scripts.collect_metric_scenarios",
        "evaluate-source",
        "--task",
        str(task_path),
        "--candidate",
        str(candidate_patch.resolve()),
        "--base",
        str(Path(context["base_repository"]).resolve()),
        "--private-root",
        str(private_root),
        "--config",
        str(config_path.resolve()),
        "--output",
        str(output),
    ]
    if reverse:
        command.append("--reverse")
    write_json(output / "scenario-source-command.json", {"argv": command})
    with (output / "scenario-source.log").open("a") as log:
        result = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    if result.returncode:
        raise RuntimeError(
            f"source-bound build/collection failed; see {output / 'scenario-source.log'}"
        )
    campaign = output / "metric-scenarios/campaign.json"
    return load_report(campaign, task), campaign


def evaluate_source_build(
    *,
    task_path,
    candidate_patch,
    base_repository,
    private_root,
    config_path,
    output,
    reverse=False,
):
    from pitbench.evaluator.judge import LocalProcessJudge
    from pitbench.repositories.base import RepositoryPluginRegistry

    task = PitBenchTask.from_yaml(task_path)
    config = _read(config_path)
    for protocol in task.metric_scenarios:
        representation_type(protocol.representation).check_runtime()
    recipe = config.pop("source_build")
    if (
        recipe["source_commit"] != task.release.base_commit
        or recipe["repository_plugin"] != task.repository.plugin
    ):
        raise ValueError("source build recipe differs from the frozen task")
    plugin = RepositoryPluginRegistry.load(task.repository.plugin)
    judge = LocalProcessJudge(
        task=task,
        base_repository=base_repository,
        public_root=Path(__file__).resolve().parents[2],
        private_root=private_root,
        candidate_patch=candidate_patch,
        output_dir=output / "api-build",
    )
    manifest_path = output / "api-build/scenario-build.json"
    built = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else judge.build_scenario_sources()
    )
    if built["status"] != "passed":
        raise RuntimeError("actual Base/candidate API build failed")
    runtimes = {}
    for state, row in built["states"].items():
        source = Path(row["workspace"])
        receipt = source / ".pitbench/scenario-binding.json"
        python = (
            str(source / plugin.driver_python)
            if "/" in plugin.driver_python
            else sys.executable
        )
        environment = {"PITBENCH_SCENARIO_BINDING": str(receipt)}
        probe = subprocess.run(
            [
                python,
                "-m",
                "scripts.collect_metric_scenarios",
                "probe",
                "--session-plugin",
                plugin.scenario_session_plugin,
            ],
            env={**os.environ, **environment},
            text=True,
            capture_output=True,
            check=True,
        )
        identity = json.loads(probe.stdout.strip().splitlines()[-1])
        runtimes[state] = {
            "python": python,
            "session_plugin": plugin.scenario_session_plugin,
            "source_commit": task.release.base_commit,
            "patch_sha256": row["binding"]["source_patch_sha256"],
            "expected_binding_sha256": identity["binding_sha256"],
            "environment": environment,
            "options_by_scenario": config.get("options_by_scenario", {}),
        }
    config["runtimes"] = runtimes
    config["cpus"] = sorted(os.sched_getaffinity(0))
    concrete = output / "built-scenario-config.yaml"
    concrete.write_text(yaml.safe_dump(config, sort_keys=False))
    return collect_for_evaluation(
        task, candidate_patch, output, concrete, reverse=reverse
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    prepare_parser = subs.add_parser("prepare")
    prepare_parser.add_argument("--config", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    for action in ("run", "report"):
        sub = subs.add_parser(action)
        sub.add_argument("--output", type=Path, required=True)
        if action == "run":
            sub.add_argument("--retry-errors", action="store_true")
        else:
            sub.add_argument(
                "--scaling-time-policy",
                choices=("observed", "budget_capped_v1"),
                default="observed",
            )
            sub.add_argument(
                "--dual-bound-source",
                choices=("independently_verified", "solver_reported"),
                default="independently_verified",
            )
            sub.add_argument("--reported-bound-infinity", type=float)
            sub.add_argument("--report-output", type=Path)
    sub = subs.add_parser("worker")
    sub.add_argument("--job", type=Path, required=True)
    sub = subs.add_parser("probe")
    sub.add_argument("--session-plugin", required=True)
    sub.add_argument("--representation", action="append", default=[])
    source = subs.add_parser("evaluate-source")
    for name in ("task", "candidate", "base", "private-root", "config", "output"):
        source.add_argument("--" + name, type=Path, required=True)
    source.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    if args.action == "evaluate-source":
        measured, path = evaluate_source_build(
            task_path=args.task,
            candidate_patch=args.candidate,
            base_repository=args.base,
            private_root=args.private_root,
            config_path=args.config,
            output=args.output,
            reverse=args.reverse,
        )
        result = {"campaign": str(path), "metrics": len(measured.results)}
    elif args.action == "prepare":
        result = prepare(args.config, args.output)
    elif args.action == "run":
        result = run(args.output, retry_errors=args.retry_errors)
    elif args.action == "report":
        result = report(
            args.output,
            scaling_time_policy=args.scaling_time_policy,
            dual_bound_source=args.dual_bound_source,
            reported_bound_infinity=args.reported_bound_infinity,
            report_path=args.report_output,
        )
    elif args.action == "worker":
        result = worker(args.job)
    else:
        for name in args.representation:
            representation_type(name).check_runtime()
        result = ScenarioSession.load(args.session_plugin).probe()
    print(json.dumps(result, allow_nan=False))
