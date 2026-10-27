"""Prepare, collect and verify source-bound metric scenarios with shared workers."""

from __future__ import annotations

import gc
import hashlib
import itertools
import json
import os
import platform
import shutil
import subprocess
import threading
import time
from pathlib import Path

import yaml

from pitbench.evaluator.artifacts import write_json
from pitbench.evaluator.collection import run_collection_jobs, run_collection_process
from pitbench.evaluator.representations import representation_type
from pitbench.metrics.scenario_report import compute_scenario_report, sequence_key
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


def _rss():
    fields = Path("/proc/self/statm").read_text().split()
    return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")


def _hardware_identity(cpus):
    return {
        "host_sha256": hashlib.sha256(Path("/etc/machine-id").read_bytes()).hexdigest(),
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
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
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
            }
        )
        runtimes[state] = {**runtime, "probe": identity}
    if (
        candidate_hash != hashlib.sha256(b"").hexdigest()
        and runtimes["base"]["probe"] == runtimes["agent"]["probe"]
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
        if implementation.identity() != runtime["probe"]:
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
                        options=runtime.get("options", {}),
                    )
                    session.load_model(model)
                    setup_time = time.perf_counter() - started
                else:
                    session.update_model(previous, model)
                    update_time = time.perf_counter() - started
            events = []

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
    if campaign.hardware_identity != _hardware_identity(runtime["cpus"]):
        raise ValueError(
            "scenario hardware or resource allocation changed after preparation"
        )
    protocols = {p.id: p for p in campaign.protocols}
    existing = {record.key: record for record in campaign.runs}
    launched = threading.Event()
    states = ["base", "agent"] if campaign.order == "base_first" else ["agent", "base"]

    def execute(key, cpus):
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

    for state in states:
        keys = [key for key in campaign.expected_runs if key.code_state == state]
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


def report(output: Path) -> dict:
    campaign = ScenarioCampaign.model_validate_json(
        (output / "campaign.json").read_text()
    )
    result = compute_scenario_report(campaign)
    write_json(output / "scenario-report.json", result.model_dump(mode="json"))
    return {
        "expected_runs": len(campaign.expected_runs),
        "recorded_runs": len(campaign.runs),
        "complete": result.complete,
        "invalid_runs": len(result.invalid_runs),
        "metrics": len(result.results),
    }


def load_report(campaign_path: Path, task: PitBenchTask):
    campaign = ScenarioCampaign.model_validate_json(campaign_path.read_text())
    if campaign.task_id != task.task_id or campaign.task_sha256 != task_digest(task):
        raise ValueError("scenario campaign belongs to another frozen task")
    return compute_scenario_report(campaign)


def collect_for_evaluation(
    task: PitBenchTask,
    candidate_patch: Path,
    output: Path,
    config_path: Path,
    *,
    reverse: bool = False,
):
    """Collect the explicit API scenarios alongside a normal independent Judge."""
    config = _read(config_path)
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
    sub = subs.add_parser("worker")
    sub.add_argument("--job", type=Path, required=True)
    sub = subs.add_parser("probe")
    sub.add_argument("--session-plugin", required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.config, args.output)
    elif args.action == "run":
        result = run(args.output, retry_errors=args.retry_errors)
    elif args.action == "report":
        result = report(args.output)
    elif args.action == "worker":
        result = worker(args.job)
    else:
        result = ScenarioSession.load(args.session_plugin).identity()
    print(json.dumps(result, allow_nan=False))
