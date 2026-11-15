#!/usr/bin/env python3
"""Build a manifest's releases, run full unchanged-source Judges, then model runs."""
# Script-path execution needs the checkout on sys.path before project imports.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
# An absolute script path also works when launched outside the checkout.
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from adapters.pitbench.adapter import PitBenchAdapter
from pitbench.agent_tools import AgentTool
from pitbench.cli.evaluate_config import EvaluationConfig, EvaluationPaths
from pitbench.cli.main import (
    _configuration_dry_run_failures,
    _evaluation_metric_failures,
)
from pitbench.cli.solver_images import build_solver_image
from pitbench.evaluator.artifacts import write_json
from pitbench.evaluator.configuration import (
    attach_configuration_report,
    run_candidate_configuration,
)
from pitbench.evaluator.evaluator import PitBenchEvaluator
from pitbench.evaluator.storage import ObservationStore, cache_lock
from pitbench.harness.evaluation import EvaluationRequest
from pitbench.metrics.campaign_report import require_plotting, write_campaign_report
from pitbench.research import (
    prepare_generalization_run_views,
    prepare_generalization_seed_views,
    study_evaluation_matches,
)
from pitbench.schema.observation import RunStatus
from pitbench.schema.task import PitBenchTask
from pitbench.tasks import TaskCatalog


def now():
    return datetime.now(timezone.utc).isoformat()


def study_cpu_ids(pool):
    cpus = set()
    for part in pool.split(","):
        bounds = part.split("-")
        if len(bounds) not in (1, 2) or any(not b.isdecimal() for b in bounds):
            raise ValueError(f"invalid CPU pool: {pool}")
        first, last = int(bounds[0]), int(bounds[-1])
        selected = set(range(first, last + 1))
        if not selected or selected & cpus:
            raise ValueError(f"invalid or overlapping CPU pool: {pool}")
        cpus.update(selected)
    return cpus


@contextmanager
def study_cpu_lease(pools, *, lock_root=None):
    """Reserve host CPUs across study managers, including idle worker slots."""
    occupied = set()
    for pool in pools:
        cpus = study_cpu_ids(pool)
        if cpus & occupied:
            raise ValueError("study CPU pools overlap")
        occupied.update(cpus)
    if not occupied <= os.sched_getaffinity(0):
        raise ValueError("study CPU pools must be available on this host")
    root = lock_root or Path(f"/tmp/pitbench-study-cpus-{os.getuid()}")
    root.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for cpu in sorted(occupied):
            handle = stack.enter_context((root / f"cpu-{cpu}.lock").open("a+"))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    f"CPU {cpu} is reserved by another study manager"
                ) from error
        yield


def refresh_study_index(output):
    """Publish a snapshot from live statuses under a separate writer lock."""
    cohort_path = output / "cohort.json"
    if not cohort_path.is_file():
        return []
    with cache_lock(output / "study-index.lock"):
        cohort = json.loads(cohort_path.read_text())
        records = []
        for job in cohort["jobs"]:
            path = output / "agent-status" / job["run_id"] / "status.json"
            state = (
                json.loads(path.read_text())
                if path.is_file()
                else {**job, "status": "not_started"}
            )
            records.append(
                {
                    **state,
                    "condition_id": job["run_id"],
                    "model": job["model"]["model"],
                    "setting": job["model"].get("setting"),
                    "arm": job["arm"],
                    "task_id": job["task_id"],
                }
            )
        write_json(output / "model-runs.json", records)
        write_json(
            output / "study-status.json",
            {
                "updated_at": now(),
                "stop_requested": (output / "STOP").exists(),
                "total": len(records),
                "statuses": dict(Counter(row["status"] for row in records)),
            },
        )
        return records


def checked_run(command, log):
    return subprocess.run(
        command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, text=True
    )


def host_snapshot():
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        memory[key] = int(value.split()[0])
    temperatures = []
    for hw in Path("/sys/class/hwmon").glob("hwmon*"):
        try:
            if (hw / "name").read_text().strip() not in {
                "coretemp",
                "k10temp",
                "cpu_thermal",
            }:
                continue
            for sensor in hw.glob("temp*_input"):
                temperatures.append(int(sensor.read_text()) / 1000)
        except (OSError, ValueError):
            continue
    return {
        "at": now(),
        "available_memory_gib": memory["MemAvailable"] / (1024 * 1024),
        "swap_used_gib": (memory["SwapTotal"] - memory["SwapFree"]) / (1024 * 1024),
        "load_1min": float(Path("/proc/loadavg").read_text().split()[0]),
        "cpu_temperature_c": max(temperatures, default=None),
    }


def wait_for_capacity(args):
    policy_path = args.output / "resource-policy.json"
    if not policy_path.is_file():
        return
    policy = json.loads(policy_path.read_text())
    waiting = False
    while True:
        health = host_snapshot()
        reasons = []
        if health["available_memory_gib"] < policy["min_available_memory_gib"]:
            reasons.append("memory reserve")
        if (
            health["cpu_temperature_c"] is not None
            and health["cpu_temperature_c"] >= policy["max_cpu_temperature_c"]
        ):
            reasons.append("CPU temperature")
        if health["load_1min"] >= policy["max_load_1min"]:
            reasons.append("host CPU load")
        if not reasons:
            return
        if not waiting:
            print("Waiting before dispatch: " + ", ".join(reasons), flush=True)
            waiting = True
        with (args.output / "admission-waits.jsonl").open("a") as log:
            log.write(json.dumps({**health, "reasons": reasons}) + "\n")
        time.sleep(policy.get("check_every_sec", 10))


def build(task_id, args, config):
    record = TaskCatalog(ROOT).validate_one(task_id)
    directory = args.output / "build" / task_id
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "build.log").open("a") as log:

        def run(command, **kwargs):
            if kwargs.get("capture_output"):
                return subprocess.run(command, **kwargs)
            return subprocess.run(
                command, **kwargs, stdout=log, stderr=subprocess.STDOUT
            )

        image = build_solver_image(
            task_id, ROOT, config=config.solver_image_build, run=run
        )
        identity = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--entrypoint",
                "git",
                image.image_id,
                "-C",
                "/workspace/repo",
                "rev-parse",
                "HEAD",
                "HEAD^{tree}",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        if identity != [record.task.release.base_commit, record.task.release.tree_sha]:
            raise ValueError(
                f"{task_id}: image source commit/tree differs from manifest"
            )
    return {
        "image_id": image.image_id,
        "image_tag": image.image_tag,
        "source_commit": identity[0],
        "source_tree": identity[1],
    }


def judge_worker(task_id, args, config):
    record = TaskCatalog(ROOT).validate_one(task_id)
    resources = config.resources_for(task_id)
    directory = args.output / "baseline" / task_id
    directory.mkdir(parents=True, exist_ok=True)
    patch = directory / "candidate.patch"
    patch.write_bytes(b"")
    source = PitBenchAdapter(
        ROOT, ROOT / config.paths.private_root
    ).ensure_source_cache(record.task)
    evaluator_config = {
        "task_config_path": str(record.task_config_path),
        "base_repository": str(source),
        "private_root": str((ROOT / config.paths.private_root).resolve()),
        "judge_image": resources.judge_image,
        "judge_cpus": config.judge_cpus,
        "judge_memory": config.judge_memory or "32g",
        "judge_parallel_runs": config.judge_parallel_runs,
        "use_base_cache": False,
        "_progress_callback": lambda message: print(message, flush=True),
    }
    if args.cpu_pool:
        evaluator_config["judge_cpuset_cpus"] = args.cpu_pool
    if task_id in config.metric_scenario_configs:
        evaluator_config["metric_scenario_config"] = str(
            (ROOT / config.metric_scenario_configs[task_id]).resolve()
        )
    envelope = PitBenchEvaluator().envelope(
        EvaluationRequest(
            task_id=task_id,
            task_path=ROOT,
            candidate_patch_path=patch,
            candidate_patch_sha256=hashlib.sha256(b"").hexdigest(),
            output_dir=directory,
            agent_name="release-baseline",
            model_name=None,
            evaluator_config=evaluator_config,
        )
    )
    write_json(directory / "evaluation.json", envelope.model_dump(mode="json"))
    if not envelope.completed:
        raise RuntimeError(envelope.error)
    failures = _evaluation_metric_failures(record.task, directory, envelope.payload)
    if not envelope.payload["validity"]["accepted"]:
        failures.append("independent solution/objective verification failed")
    observations = ObservationStore.read(
        directory / envelope.payload["artifacts"]["observations"]["path"]
    )
    for status in (RunStatus.INFRASTRUCTURE_ERROR, RunStatus.INVALID):
        count = sum(item.status == status for item in observations)
        if count:
            failures.append(f"{count} {status.value} observations")
    if failures:
        raise RuntimeError("; ".join(failures))
    return {
        "observation_count": len(observations),
        "valid_observation_count": sum(item.valid for item in observations),
        "performance_classification": envelope.payload["summary"]["performance"][
            "classification"
        ],
        "scope": "full_declared_instance_seed_budget_grid_unchanged_source",
    }


def run_phase(phase, task_ids, args, config):
    results = {}
    cpu_slots = queue.SimpleQueue()
    for pool in args.cpu_pools:
        cpu_slots.put(pool)

    def one(index, task_id):
        receipt = args.output / phase / task_id / "status.json"
        record = TaskCatalog(ROOT).validate_one(task_id)
        identity = {"task_config_sha256": record.task_config_sha256}
        previous = json.loads(receipt.read_text()) if receipt.exists() else None
        if previous is not None:
            if previous.get("identity") != identity:
                raise ValueError(
                    f"{task_id}: previous task identity differs; use a new output directory"
                )
            if previous["status"] == "passed":
                return task_id, previous
            if not args.retry_errors:
                return task_id, previous
            attempts = receipt.parent / "attempts"
            attempts.mkdir(exist_ok=True)
            write_json(
                attempts / f"{len(list(attempts.glob('*.json'))) + 1:04d}.json",
                previous,
            )
        state = {
            "task_id": task_id,
            "phase": phase,
            "identity": identity,
            "status": "running",
            "started_at": now(),
        }
        write_json(receipt, state)
        try:
            if phase == "build":
                wait_for_capacity(args)
                state.update(build(task_id, args, config))
            else:
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--phase",
                    "_judge",
                    "--task-id",
                    task_id,
                    "--manifest",
                    str(args.manifest),
                    "--config",
                    str(args.config),
                    "--output",
                    str(args.output),
                ]
                cpu_pool = cpu_slots.get() if args.cpu_pools else None
                try:
                    if cpu_pool is not None:
                        command += ["--cpu-pool", cpu_pool]
                    with (receipt.parent / "judge.log").open("a") as log:
                        checked_run(command, log)
                finally:
                    if cpu_pool is not None:
                        cpu_slots.put(cpu_pool)
                state.update(json.loads((receipt.parent / "result.json").read_text()))
            state["status"] = "passed"
        except Exception as error:
            state.update(status="failed", error=f"{type(error).__name__}: {error}")
        state["finished_at"] = now()
        write_json(receipt, state)
        print(f"{phase} {task_id}: {state['status']}", flush=True)
        return task_id, state

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for task_id, result in pool.map(lambda item: one(*item), enumerate(task_ids)):
            results[task_id] = result
    write_json(args.output / f"{phase}.json", results)
    if phase == "build":
        for task_id, result in results.items():
            if result["status"] == "passed":
                agent_image = "pitbench-agent:" + result["image_id"].replace(":", "-")
                subprocess.run(
                    ["docker", "image", "tag", result["image_id"], agent_image],
                    cwd=ROOT,
                    check=True,
                )
                result["agent_image_tag"] = agent_image
                write_json(args.output / phase / task_id / "status.json", result)
                config.tasks[task_id] = config.resources_for(task_id).model_copy(
                    update={
                        "agent_image": agent_image,
                        "judge_image": result["image_id"],
                    }
                )
        config = EvaluationConfig.model_validate(config.model_dump())
        args.config.write_text(
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
        )
        write_json(args.output / f"{phase}.json", results)
    return all(result["status"] == "passed" for result in results.values())


def audit_model_run(run_dir, task_id, *, evaluation_required=True):
    """A successful CLI exit alone does not prove that an Agent executed."""
    path = run_dir / "results.json"
    if not path.is_file():
        return {
            "complete": False,
            "retryable": not run_dir.exists() or not any(run_dir.iterdir()),
            "reason": "no terminal harness result",
        }
    rows = [
        r
        for r in json.loads(path.read_text()).get("results", [])
        if r.get("task_id") == task_id
    ]
    if len(rows) != 1:
        return {
            "complete": False,
            "retryable": False,
            "reason": "expected one harness trial",
        }
    trial = rows[0]
    failure = trial.get("failure_mode")
    tokens = int(trial.get("total_input_tokens") or 0) + int(
        trial.get("total_output_tokens") or 0
    )
    folder = run_dir / task_id / trial["trial_name"]
    transcript = folder / "agent-logs/codex.jsonl"
    completed_turn = False
    rate_limited = False
    transport_failed = False
    trace_closed_during_cleanup = False
    if transcript.is_file():
        for line in transcript.read_text(errors="replace").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            completed_turn |= event.get("type") == "turn.completed"
            if event.get("type") in ("error", "turn.failed"):
                rate_limited |= "429" in line or "rate_limit" in line
                transport_failed |= (
                    "stream disconnected" in line or "error sending request" in line
                )
    pipeline_trace = run_dir / "pipeline_trace.jsonl"
    if pipeline_trace.is_file():
        for line in pipeline_trace.read_text(errors="replace").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            error = event.get("error") or {}
            trace_closed_during_cleanup |= (
                error.get("type") == "RuntimeError"
                and error.get("message") == "agent trace is already closed"
            )
    evidence = {
        "failure_mode": failure,
        "tokens": tokens,
        "completed_turn": completed_turn,
        "trace_closed_during_cleanup": trace_closed_during_cleanup,
        "harness_result": str(path),
    }
    candidate = folder / "evaluation/candidate.patch"
    if (
        not evaluation_required
        and failure in ("unset", "none", "agent_timeout")
        and (tokens > 0 or completed_turn)
        and candidate.is_file()
    ):
        return {
            **evidence,
            "complete": True,
            "retryable": False,
            "development_complete": True,
            "evaluation_pending": True,
            "candidate_patch": str(candidate),
            "candidate_patch_sha256": hashlib.sha256(
                candidate.read_bytes()
            ).hexdigest(),
        }
    if failure == "unknown_agent_error" and tokens == 0 and rate_limited:
        return {
            **evidence,
            "complete": False,
            "retryable": True,
            "reason": "initial model request rate limited; no model execution",
        }
    if (
        failure == "unknown_agent_error"
        and trace_closed_during_cleanup
        and not completed_turn
    ):
        return {
            **evidence,
            "complete": False,
            "retryable": True,
            "reason": "agent trace closed during timeout cleanup; retain failed attempt",
        }
    if failure == "unknown_agent_error" and transport_failed and not completed_turn:
        return {
            **evidence,
            "complete": False,
            "retryable": True,
            "reason": "model stream disconnected; retain partial infrastructure-failed attempt",
        }
    evaluation = trial.get("evaluation") or {}
    if (
        failure in ("unset", "none", "agent_timeout")
        and (tokens > 0 or completed_turn)
        and evaluation.get("completed")
    ):
        return {
            **evidence,
            "complete": True,
            "retryable": False,
            "validity_accepted": evaluation.get("payload", {})
            .get("validity", {})
            .get("accepted"),
        }
    return {
        **evidence,
        "complete": False,
        "retryable": False,
        "evaluation_pending": failure in ("unset", "none", "agent_timeout")
        and (tokens > 0 or completed_turn),
        "reason": evaluation.get("error") or f"unresolved Agent outcome: {failure}",
    }


def incomplete_condition_retry_reason(original_status, audit):
    """Return why a failed condition can be restarted in a fresh workspace."""
    if original_status != "development_failed":
        return None
    stream_disconnect = (
        audit.get("reason")
        == "model stream disconnected; retain partial infrastructure-failed attempt"
        and int(audit.get("tokens") or 0) == 0
        and not audit.get("completed_turn")
        and not audit.get("candidate_patch")
    )
    trace_close_race = (
        audit.get("reason")
        == "agent trace closed during timeout cleanup; retain failed attempt"
        and audit.get("trace_closed_during_cleanup")
        and not audit.get("completed_turn")
    )
    incomplete_timeout = (
        audit.get("failure_mode") == "agent_timeout"
        and not audit.get("complete")
        and not audit.get("evaluation_pending")
        and not audit.get("completed_turn")
    )
    if incomplete_timeout:
        return "incomplete_agent_timeout"
    if trace_close_race:
        return "trace_close_race"
    if stream_disconnect:
        return "stream_disconnect"
    return None


def acquire_condition_recovery_lock(retry_root):
    """Prevent two recovery launches from racing on the same condition."""
    retry_root.mkdir(parents=True, exist_ok=True)
    lock_file = (retry_root / "active.lock").open("a+")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_file.close()
        raise ValueError(
            "a recovery attempt is already running for this condition"
        ) from error
    return lock_file


def models(task_ids, args):
    baseline = args.output / "baseline.json"
    if not baseline.is_file():
        raise ValueError("full baseline results are missing; model runs are gated")
    evidence = json.loads(baseline.read_text())
    if set(evidence) != set(task_ids) or any(
        row["status"] != "passed" for row in evidence.values()
    ):
        raise ValueError(
            "every manifest task must pass its real baseline before model runs"
        )
    for task_id in task_ids:
        record = TaskCatalog(ROOT).validate_one(task_id)
        if (
            evidence[task_id]["identity"]["task_config_sha256"]
            != record.task_config_sha256
        ):
            raise ValueError(f"{task_id}: task changed after its real baseline")
    matrix = yaml.safe_load(args.models.read_text())["runs"]
    require_plotting()
    statuses = []
    config_template = EvaluationConfig.from_yaml(args.config)
    defaults = EvaluationPaths()
    if config_template.paths.output_path in {defaults.output_path, Path("runs")}:
        config_template.paths.output_path = args.output / "runs"
    if config_template.paths.workspace_path in {
        defaults.workspace_path,
        Path(".pitbench/tasks"),
    }:
        config_template.paths.workspace_path = args.output / "workspaces"
    run_root = (ROOT / config_template.paths.output_path).resolve()
    workspace_root = (ROOT / config_template.paths.workspace_path).resolve()
    if getattr(args, "resume_models", False):
        for process in Path("/proc").glob("[0-9]*/cmdline"):
            try:
                if process.stat().st_uid != os.getuid():
                    continue
                words = [
                    w.decode(errors="replace")
                    for w in process.read_bytes().split(b"\0")
                    if w
                ]
            except OSError:
                continue
            if (
                "pitbench.cli.main" in words
                and "evaluate" in words
                and "--run-id" in words
            ):
                existing = words[words.index("--run-id") + 1]
                if (
                    existing.startswith(args.output.name)
                    and "\nState:\tZ" not in (process.parent / "status").read_text()
                ):
                    raise RuntimeError(
                        "existing campaign trials are live; wait before resuming"
                    )
    slots = queue.SimpleQueue()
    for cpu_pool in args.cpu_pools or [None] * args.workers:
        slots.put(cpu_pool)

    launch_lock = threading.Lock()
    next_launch = [0.0]

    def one(task_id, row, run_id):
        cpu_pool = slots.get()
        condition_id = f"{args.output.name}-{row['label']}-{task_id}"
        state = {
            "run_id": run_id,
            "task_id": task_id,
            "model": row["model"],
            "setting": row["setting"],
            "cpu_pool": cpu_pool,
            "started_at": now(),
            "status": "command_running",
        }
        status_path = args.output / "model-status" / condition_id / "status.json"
        write_json(status_path, state)
        try:
            wait_for_capacity(args)
            config = config_template.model_copy(deep=True)
            if cpu_pool is not None:
                config.tasks[task_id].judge_cpuset_cpus = cpu_pool
                config.agent_cpuset_cpus = cpu_pool
            local_config = args.output / "model-configs" / f"{run_id}.local.yaml"
            local_config.parent.mkdir(exist_ok=True)
            local_config.write_text(
                yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
            )
            command = [
                sys.executable,
                "-m",
                "pitbench.cli.main",
                "evaluate",
                task_id,
                "--config",
                str(local_config),
                "--agent",
                row["agent"],
                "--model",
                row["model"],
                "--agent-kwarg",
                row["setting"],
                "--run-id",
                run_id,
                "--n-attempts",
                "1",
                "--n-concurrent",
                "1",
            ]
            with (args.output / f"{run_id}.log").open("a") as log:
                with launch_lock:
                    delay = max(0.0, next_launch[0] - time.monotonic())
                    if delay:
                        time.sleep(delay)
                    next_launch[0] = time.monotonic() + getattr(
                        args, "dispatch_interval_sec", 0
                    )
                checked_run(command, log)
            audit = audit_model_run(run_root / run_id, task_id)
            state["audit"] = audit
            state["status"] = (
                "command_completed" if audit["complete"] else "command_failed"
            )
            if not audit["complete"]:
                state["error"] = audit["reason"]
        except Exception as error:
            state.update(status="command_failed", error=str(error))
        finally:
            slots.put(cpu_pool)
        state["finished_at"] = now()
        write_json(status_path, state)
        return state

    jobs = []
    for row in matrix:
        for task_id in task_ids:
            condition_id = f"{args.output.name}-{row['label']}-{task_id}"
            status_path = args.output / "model-status" / condition_id / "status.json"
            previous = (
                json.loads(status_path.read_text()) if status_path.exists() else None
            )
            run_id = previous.get("run_id", condition_id) if previous else condition_id
            if getattr(args, "resume_models", False) and (run_root / run_id).exists():
                audit = audit_model_run(run_root / run_id, task_id)
                if audit["complete"]:
                    state = {
                        **(previous or {}),
                        "run_id": run_id,
                        "task_id": task_id,
                        "model": row["model"],
                        "setting": row["setting"],
                        "status": "command_completed",
                        "audit": audit,
                    }
                    write_json(status_path, state)
                    statuses.append(state)
                    continue
                if getattr(args, "defer_evaluation_incomplete", False) and audit.get(
                    "evaluation_pending"
                ):
                    state = {
                        **(previous or {}),
                        "run_id": run_id,
                        "task_id": task_id,
                        "model": row["model"],
                        "setting": row["setting"],
                        "status": "evaluation_pending",
                        "audit": audit,
                    }
                    write_json(status_path, state)
                    statuses.append(state)
                    continue
                if not audit["retryable"]:
                    raise RuntimeError(
                        f"{run_id}: {audit['reason']}; preserve and diagnose before retrying"
                    )
            if getattr(args, "resume_models", False) and (
                (previous and previous.get("status") != "queued")
                or (run_root / run_id).exists()
                or (workspace_root / run_id).exists()
            ):
                if previous:
                    history = status_path.parent / "attempts"
                    history.mkdir(exist_ok=True)
                    write_json(
                        history / f"{len(list(history.glob('*.json'))) + 1:04d}.json",
                        previous,
                    )
                number = 2
                run_id = f"{condition_id}-infra{number}"
                while (run_root / run_id).exists() or (
                    workspace_root / run_id
                ).exists():
                    number += 1
                    run_id = f"{condition_id}-infra{number}"
            jobs.append((task_id, row, run_id))
            write_json(
                status_path,
                {
                    "run_id": run_id,
                    "task_id": task_id,
                    "model": row["model"],
                    "setting": row["setting"],
                    "status": "queued",
                },
            )
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(one, task_id, row, run_id) for task_id, row, run_id in jobs
        ]
        for future in concurrent.futures.as_completed(futures):
            statuses.append(future.result())
            write_json(args.output / "model-runs.json", statuses)
    write_json(args.output / "model-runs.json", statuses)
    report(task_ids, args, config_template)
    if any(row["status"] != "command_completed" for row in statuses):
        raise SystemExit(1)


def report(task_ids, args, config):
    """Aggregate completed, failed and missing trials without rerunning them."""
    defaults = EvaluationPaths()
    runs_root = config.paths.output_path
    if runs_root in {defaults.output_path, Path("runs")}:
        runs_root = args.output / "runs"
    if not runs_root.is_absolute():
        runs_root = ROOT / runs_root
    result = write_campaign_report(
        args.output,
        args.output / "report",
        repository_root=ROOT,
        runs_root=runs_root,
        task_configs={
            task_id: TaskCatalog(ROOT).validate_one(task_id).task_config_path
            for task_id in task_ids
        },
    )
    print(f"Report: {args.output / 'report/report.html'}", flush=True)
    return result


def study_agent_arguments(job):
    """Select the configured Agent through the existing CLI interface."""
    custom = job.get("agent_import_path")
    return (
        ["--agent-import-path", custom]
        if custom
        else ["--agent", job["model"]["agent"]]
    )


def configure_study_agent(config, job, budget):
    original = job["model"]["agent"]
    selected = job.get("agent_import_path") or original
    config.agents[selected] = {
        **config.kwargs_for(original),
        **config.kwargs_for(selected),
        "timeout_sec": budget.wall_time_sec,
    }


def configure_study_scenarios(config, task_id, views):
    reference = views["tasks"][task_id].get("metric_scenario_config")
    if reference:
        config.metric_scenario_configs[task_id] = Path(reference)
    else:
        config.metric_scenario_configs.pop(task_id, None)


def evaluate_study_candidate(state, args, cpu_pool):
    """Claim a condition before changing its status or starting any Judge work."""
    status_path = args.output / "agent-status" / state["run_id"] / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with (status_path.parent / "evaluation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return (
                json.loads(status_path.read_text()) if status_path.is_file() else state
            )
        previous = (
            json.loads(status_path.read_text()) if status_path.is_file() else state
        )
        # A queued snapshot must not overwrite a condition finished by another manager.
        if any(
            previous.get(key) != state.get(key)
            for key in ("status", "evaluation_attempt")
        ) or (previous.get("audit") and previous["audit"] != state.get("audit")):
            return previous
        if (args.output / "STOP").exists():
            return previous
        previous = {**previous, "audit": state["audit"]}
        return _evaluate_study_candidate(previous, args, cpu_pool)


def _evaluate_study_candidate(state, args, cpu_pool):
    """Consume one captured patch through the existing independent evaluator."""
    job = dict(state)
    run_id, task_id = job["run_id"], job["task_id"]
    execution_id = job.get("execution_run_id") or run_id
    status_path = args.output / "agent-status" / run_id / "status.json"
    source_patch = Path(job["audit"]["candidate_patch"])
    task_path = args.output / "workspaces" / execution_id / task_id
    previous = (
        json.loads(status_path.read_text()) if status_path.is_file() else dict(job)
    )
    if any(
        key in previous
        for key in ("evaluation_started_at", "evaluation_error", "metrics_complete")
    ):
        archive = (
            status_path.parent / "evaluation-attempts" / (str(time.time_ns()) + ".json")
        )
        write_json(archive, previous)
    job.update(
        status="evaluating",
        evaluation_attempt=int(previous.get("evaluation_attempt", 0)) + 1,
        evaluation_cpu_pool=cpu_pool,
        evaluation_started_at=now(),
        evaluation_finished_at=None,
        evaluation_error=None,
        metric_gaps=[],
        metrics_complete=False,
        evaluation_output=None,
        configuration_robustness_output=None,
    )
    write_json(status_path, job)
    refresh_study_index(args.output)
    try:
        expected = job["audit"]["candidate_patch_sha256"]
        if (
            not source_patch.is_file()
            or hashlib.sha256(source_patch.read_bytes()).hexdigest() != expected
        ):
            raise ValueError("final captured candidate is missing or changed")
        # Each invocation retains its own Judge identity and complete artifacts.
        output = (
            args.output
            / "evaluations"
            / run_id
            / (f"attempt-{job['evaluation_attempt']:04d}-{uuid.uuid4().hex}")
        )
        output.mkdir(parents=True)
        patch = output / "candidate.patch"
        shutil.copy2(source_patch, patch)
        if hashlib.sha256(patch.read_bytes()).hexdigest() != expected:
            raise ValueError("copied candidate hash changed")
        job["evaluation_output"] = str(output)
        write_json(
            output / "attempt.json",
            {
                "run_id": run_id,
                "attempt": job["evaluation_attempt"],
                "source_candidate": str(source_patch),
                "candidate_patch_sha256": expected,
                "cpu_pool": cpu_pool,
                "started_at": job["evaluation_started_at"],
            },
        )
        write_json(status_path, job)
        refresh_study_index(args.output)
        configured = yaml.safe_load((task_path / "task.yaml").read_text())[
            "evaluator_config"
        ]
        configured.update(judge_cpuset_cpus=cpu_pool)
        configured.update(
            use_base_cache=True,
            base_cache_path=str((args.output / "shared-cache/base").resolve()),
            build_cache_path=str((args.output / "shared-cache/build").resolve()),
        )
        request = EvaluationRequest(
            task_id=task_id,
            task_path=task_path,
            candidate_patch_path=patch,
            candidate_patch_sha256=job["audit"]["candidate_patch_sha256"],
            output_dir=patch.parent,
            agent_name=job.get("agent_import_path") or job["model"]["agent"],
            model_name=job["model"]["model"],
            evaluator_config=configured,
        )
        result = PitBenchEvaluator().envelope(request)
        write_json(
            patch.parent / "evaluation-envelope.json", result.model_dump(mode="json")
        )
        results_path = args.output / "runs" / execution_id / "results.json"
        harness = json.loads(results_path.read_text())
        trials = [row for row in harness["results"] if row["task_id"] == task_id]
        if len(trials) != 1:
            raise ValueError("expected one captured Agent trial")
        trials[0]["evaluation"] = result.model_dump(mode="json")
        write_json(results_path, harness)
        gaps = (
            _evaluation_metric_failures(
                PitBenchTask.from_yaml(Path(configured["task_config_path"])),
                patch.parent,
                result.payload,
            )
            if result.completed
            else []
        )
        if result.completed:
            settings = EvaluationConfig.from_yaml(args.config).configuration_robustness
            try:
                configuration_output = run_candidate_configuration(request, settings)
                if configuration_output is not None:
                    gaps.extend(
                        "parameter robustness: " + value
                        for value in _configuration_dry_run_failures(
                            configuration_output
                        )
                    )
                    job["configuration_robustness_output"] = str(configuration_output)
                    attach_configuration_report(
                        result.payload, configuration_output, output_root=patch.parent
                    )
                    trials[0]["evaluation"] = result.model_dump(mode="json")
                    write_json(
                        patch.parent / "evaluation-envelope.json",
                        result.model_dump(mode="json"),
                    )
                    write_json(results_path, harness)
            except Exception as error:
                gaps.append(f"parameter robustness: {type(error).__name__}: {error}")
        job.update(
            status=("evaluation_incomplete" if gaps else "evaluation_completed")
            if result.completed
            else "evaluation_failed",
            evaluation_error=result.error,
            metric_gaps=gaps,
            metrics_complete=result.completed and not gaps,
            scientific_interpretation="optimization_space_qualification_pending",
        )
    except Exception as error:
        job.update(
            status="evaluation_failed",
            evaluation_error=f"{type(error).__name__}: {error}",
        )
    job["evaluation_finished_at"] = now()
    write_json(status_path, job)
    write_json(args.output / "evaluation-queue" / (run_id + ".json"), job)
    refresh_study_index(args.output)
    return job


def study_evaluations(args):
    if (
        args.evaluation_workers < 1
        or len(args.evaluation_cpu_pools) != args.evaluation_workers
    ):
        raise ValueError(
            "candidate evaluation requires one CPU pool per evaluation worker"
        )
    with study_cpu_lease(args.evaluation_cpu_pools):
        return _study_evaluations(args)


def _study_evaluations(args):
    """Evaluate retained final patches without allocating another Agent start."""
    output = args.output
    cohort = json.loads((output / "cohort.json").read_text())
    slots = queue.SimpleQueue()
    for cpus in args.evaluation_cpu_pools:
        slots.put(cpus)
    pending = []
    run_id = getattr(args, "retry_run_id", None)
    jobs = [job for job in cohort["jobs"] if run_id is None or job["run_id"] == run_id]
    if run_id is not None and not jobs:
        raise ValueError(f"study run ID is absent from the cohort: {run_id}")
    for job in jobs:
        status_path = output / "agent-status" / job["run_id"] / "status.json"
        if not status_path.is_file():
            continue
        state = json.loads(status_path.read_text())
        retryable_statuses = set(
            getattr(args, "retry_evaluation_statuses", None) or ["evaluation_failed"]
        )
        if state.get("status") not in retryable_statuses:
            continue
        audit = state.get("audit") or {
            key: state[key]
            for key in (
                "candidate_patch",
                "candidate_patch_sha256",
                "tokens",
                "completed_turn",
            )
            if key in state
        }
        state["audit"] = audit
        pending.append(state)

    def run(state):
        cpus = slots.get()
        try:
            return evaluate_study_candidate(state, args, cpus)
        finally:
            slots.put(cpus)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.evaluation_workers
    ) as pool:
        for future in concurrent.futures.as_completed(
            [pool.submit(run, state) for state in pending]
        ):
            future.result()
    records = refresh_study_index(output)
    print(
        json.dumps(
            {
                "evaluation_candidates": len(pending),
                "without_agent_calls": True,
                "statuses": dict(Counter(row["status"] for row in records)),
            }
        )
    )


def retry_unexecuted_condition(args):
    """Retry an incomplete condition after a model disconnect or hard timeout."""
    from pitbench.schema.research import GeneralizationStudyPlan

    plan = GeneralizationStudyPlan.model_validate_json(args.study_plan.read_text())
    view_dir = (
        "evaluation-views"
        if plan.evaluation_run_target is not None
        else "frozen-seed-views"
    )
    views = json.loads((args.output / view_dir / "task-views.json").read_text())
    cohort = json.loads((args.output / "cohort.json").read_text())
    matches = [row for row in cohort["jobs"] if row["run_id"] == args.retry_run_id]
    if len(matches) != 1:
        raise ValueError("retry run ID is absent from the frozen cohort")
    job = matches[0]
    retry_root = args.output / "infrastructure-retries" / job["run_id"]
    recovery_lock = acquire_condition_recovery_lock(retry_root)
    status_path = args.output / "agent-status" / job["run_id"] / "status.json"
    original = json.loads(status_path.read_text())
    audit = audit_model_run(
        args.output / "runs" / job["run_id"],
        job["task_id"],
        evaluation_required=False,
    )
    retry_reason = incomplete_condition_retry_reason(original.get("status"), audit)
    if retry_reason is None:
        raise ValueError(
            "retry requires a failed condition with an incomplete model stream "
            "or a hard-timeout result without a completed turn"
        )
    retry_number = 1
    while (retry_root / f"infra{retry_number}" / "retry.json").exists():
        retry_number += 1
    receipt_dir = retry_root / f"infra{retry_number}"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt = receipt_dir / "retry.json"
    write_json(receipt_dir / "initial-status.json", original)
    retry_id = f"{job['run_id']}-infra{retry_number}"
    run_root = args.output / "runs"
    workspace_root = args.output / "workspaces"
    if (run_root / retry_id).exists() or (workspace_root / retry_id).exists():
        raise RuntimeError("retry run ID already exists")
    template = EvaluationConfig.from_yaml(args.config)
    config = template.model_copy(deep=True)
    config.paths.output_path = run_root.resolve()
    config.paths.workspace_path = workspace_root.resolve()
    config.research_budget = plan.research_budget
    config.research_track = None
    config.candidate_hypotheses_required = job["hypotheses_required"]
    config.agent_tools = [AgentTool(tool) for tool in plan.shared_agent_tools]
    resources = config.resources_for(job["task_id"]).model_copy(deep=True)
    arm = views["arms"][job["arm"]][job["task_id"]]
    resources.task_config_path = Path(arm["task_config_path"])
    resources.agent_task_config_path = Path(arm["agent_task_config_path"])
    resources.research_budget = plan.research_budget
    resources.judge_cpuset_cpus = args.cpu_pool
    config.tasks[job["task_id"]] = resources
    config.agent_cpuset_cpus = args.cpu_pool
    configure_study_scenarios(config, job["task_id"], views)
    configure_study_agent(config, job, plan.research_budget)
    attempt = receipt_dir / retry_id
    attempt.mkdir()
    config_path = attempt / "evaluation.local.yaml"
    config_path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
    )
    command = [
        sys.executable,
        "-m",
        "pitbench.cli.main",
        "evaluate",
        job["task_id"],
        "--config",
        str(config_path),
        *study_agent_arguments(job),
        "--model",
        job["model"]["model"],
        "--agent-kwarg",
        job["model"]["setting"],
        "--run-id",
        retry_id,
        "--n-attempts",
        "1",
        "--n-concurrent",
        "1",
        "--defer-evaluation",
    ]
    write_json(
        receipt,
        {
            "status": "retry_running",
            "original_run_id": job["run_id"],
            "retry_run_id": retry_id,
            "retry_reason": retry_reason,
            "audit": audit,
            "cpu_pool": args.cpu_pool,
            "agent_attempts": 1,
            "budget": plan.research_budget.model_dump(mode="json"),
        },
    )
    with (attempt / "controller.log").open("w") as log:
        code = subprocess.run(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    recovered = audit_model_run(
        run_root / retry_id, job["task_id"], evaluation_required=False
    )
    record = json.loads(receipt.read_text())
    record.update(
        returncode=code,
        recovered_audit=recovered,
        status="evaluation_pending" if recovered["complete"] else "retry_failed",
        finished_at=now(),
    )
    write_json(receipt, record)
    if recovered["complete"]:
        original.update(
            status="evaluation_pending",
            audit=recovered,
            execution_run_id=retry_id,
            infrastructure_retry=str(receipt),
        )
        write_json(status_path, original)
        write_json(
            args.output / "evaluation-queue" / (job["run_id"] + ".json"),
            {
                **job,
                **recovered,
                "original_run_id": job["run_id"],
                "execution_run_id": retry_id,
                "evaluation_config": str(config_path),
                "status": "queued_for_independent_measurement",
            },
        )
    print(
        json.dumps(
            {
                "original_condition": job["run_id"],
                "retry_run": retry_id,
                "status": record["status"],
                "model_calls": 1,
            }
        )
    )
    recovery_lock.close()


def study_models(args):
    """Run a frozen development cohort; queue hidden evaluation separately."""
    import itertools
    import random

    from pitbench.schema.research import GeneralizationStudyPlan

    plan = GeneralizationStudyPlan.model_validate_json(args.study_plan.read_text())
    config_template = EvaluationConfig.from_yaml(args.config)
    private_root = config_template.paths.private_root
    if not private_root.is_absolute():
        private_root = ROOT / private_root
    views_path = prepare_generalization_seed_views(
        plan=plan,
        views_path=args.study_views,
        private_root=private_root,
        destination=args.output / "frozen-seed-views",
    )
    if plan.evaluation_run_target is not None:
        views_path = prepare_generalization_run_views(
            views_path=views_path,
            private_root=private_root,
            destination=args.output / "evaluation-views",
            target_runs=plan.evaluation_run_target,
            supplemental_seed_count=plan.supplemental_seed_count,
            supplemental_primary_budget_only=plan.supplemental_primary_budget_only,
        )
    views = json.loads(views_path.read_text())
    if views.get("status") != "development_frozen_evaluation_gated":
        raise ValueError("development requires frozen scenario/task views")
    matrix = yaml.safe_load((ROOT / plan.model_matrix).read_text())["runs"]
    task_ids = [Path(name).stem for name in plan.task_configs]
    missing = set(task_ids) - set(views["tasks"])
    if missing:
        raise ValueError(
            f"study tasks are absent from the frozen views: {sorted(missing)}"
        )
    snapshots = {}
    model_files = {}
    for task_id in task_ids:
        row = views["tasks"][task_id]
        task_file = Path(row["evaluator_task_config"])
        record = TaskCatalog(ROOT).validate_file(task_file)
        original = TaskCatalog(ROOT).validate_one(task_id).task
        if not study_evaluation_matches(original, record.task, plan):
            raise ValueError(
                "study changed the frozen source or original primary contract"
            )
        snapshots[str(task_file)] = hashlib.sha256(task_file.read_bytes()).hexdigest()
        for arm in plan.arms:
            path = Path(views["arms"][arm.id][task_id]["agent_task_config_path"])
            TaskCatalog(ROOT).validate_file(path)
            snapshots[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        scenario_reference = row.get("metric_scenario_config")
        if not scenario_reference:
            if record.task.metric_scenarios:
                raise ValueError(
                    "declared task scenarios require a frozen runtime configuration"
                )
            continue
        scenario_file = Path(scenario_reference)
        scenario = yaml.load(
            scenario_file.read_text(),
            Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader),
        )
        from pitbench.evaluator.scenarios import resolve_model_inputs

        scenario, _ = resolve_model_inputs(
            scenario,
            record.task,
            private_root=private_root,
            config_dir=scenario_file.parent,
        )
        if scenario["source_build"]["source_commit"] != record.task.release.base_commit:
            raise ValueError("scenario source differs from study task")
        snapshots[str(task_file)] = hashlib.sha256(task_file.read_bytes()).hexdigest()
        snapshots[str(scenario_file)] = hashlib.sha256(
            scenario_file.read_bytes()
        ).hexdigest()
        for panel, cases in scenario["instances"].items():
            for case in cases:
                for pid, paths in case["models_by_scenario"].items():
                    protocol = next(
                        p for p in record.task.metric_scenarios if p.id == pid
                    )
                    hashes = protocol.model_sequences_sha256[panel][case["id"]]
                    if len(paths) != len(hashes):
                        raise ValueError("scenario sequence changed")
                    for path, expected in zip(paths, hashes, strict=True):
                        model_files[path] = expected
    for path, expected in model_files.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError("frozen scenario model changed")
    blocks = list(itertools.product(task_ids, matrix))
    random.Random(args.schedule_seed).shuffle(blocks)
    jobs = []
    run_prefix = (
        args.output.name
        + "-"
        + hashlib.sha256(str(args.output.resolve()).encode()).hexdigest()[:8]
    )
    for index, (task_id, model) in enumerate(blocks):
        arms = list(plan.arms)
        arms = arms[index % len(arms) :] + arms[: index % len(arms)]
        for arm in arms:
            for attempt in range(1, plan.agent_attempts_per_condition + 1):
                jobs.append(
                    {
                        "task_id": task_id,
                        "model": model,
                        "arm": arm.id,
                        "hypotheses_required": arm.require_candidate_hypotheses,
                        "run_id": f"{run_prefix}-{len(jobs) + 1:03d}",
                        "attempt": attempt,
                    }
                )
                if arm.agent_import_path:
                    jobs[-1]["agent_import_path"] = arm.agent_import_path
                if arm.agent_provenance:
                    jobs[-1]["agent_provenance"] = arm.agent_provenance
                if task_id in plan.task_targets:
                    jobs[-1]["target_problem"] = plan.task_targets[
                        task_id
                    ].problem_variant
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "kind": "generalization_agent_cohort",
        "study_id": plan.study_id,
        "plan_sha256": hashlib.sha256(args.study_plan.read_bytes()).hexdigest(),
        "views_sha256": hashlib.sha256(views_path.read_bytes()).hexdigest(),
        "input_files_sha256": snapshots,
        "scenario_model_files_sha256": model_files,
        "jobs": jobs,
        "research_budget": plan.research_budget.model_dump(mode="json"),
        "schedule_seed": args.schedule_seed,
        "agent_attempts_per_condition": plan.agent_attempts_per_condition,
        "comparison_mode": plan.comparison_mode,
        "execution_mode": "development_then_separate_hidden_evaluation",
        "scientific_evaluation_ready": False,
        "inference_scope": plan.inference_scope,
    }
    saved = args.output / "cohort.json"
    if saved.exists() and json.loads(saved.read_text()) != manifest:
        raise ValueError("cohort inputs changed; use a distinct output directory")
    write_json(saved, manifest)
    if args.prepare_only:
        print(
            json.dumps(
                {"planned_agent_runs": len(jobs), "status": "frozen_not_started"}
            ),
            flush=True,
        )
        return
    pools = args.cpu_pools or [None] * args.workers
    slots = queue.SimpleQueue()
    for cpu_pool in pools:
        slots.put(cpu_pool)
    launch_lock = threading.Lock()
    next_launch = [0.0]

    def one(job):
        condition = args.output / "agent-status" / job["run_id"]
        path = condition / "status.json"
        condition.mkdir(parents=True, exist_ok=True)
        existing = json.loads(path.read_text()) if path.exists() else None
        if (args.output / "STOP").exists():
            return existing or {**job, "status": "queued"}
        output_root = args.output / "runs"
        run_path = output_root / job["run_id"]
        workspace_path = args.output / "workspaces" / job["run_id"]
        if existing is None and (run_path.exists() or workspace_path.exists()):
            raise RuntimeError(
                "existing Agent artifacts lack a condition receipt; preserve them instead of resampling"
            )
        if existing is not None:
            audit = audit_model_run(run_path, job["task_id"], evaluation_required=False)
            if audit["complete"] or (
                existing.get("status") not in {"queued"} and not audit.get("retryable")
            ):
                return existing
            if existing.get("pid"):
                process = Path(f"/proc/{existing['pid']}/cmdline")
                if process.exists() and job["run_id"] in process.read_bytes().decode(
                    errors="replace"
                ):
                    raise RuntimeError("cohort condition still has a live process")
            if run_path.exists() or workspace_path.exists():
                raise RuntimeError(
                    "a started Agent attempt cannot be resampled; diagnose its retained run"
                )
        cpu_pool = slots.get()
        state = {
            **job,
            "status": "preparing",
            "started_at": now(),
            "cpu_pool": cpu_pool,
        }
        write_json(path, state)
        try:
            config = config_template.model_copy(deep=True)
            config.paths.output_path = output_root.resolve()
            config.paths.workspace_path = (args.output / "workspaces").resolve()
            config.research_budget = plan.research_budget
            config.research_track = None
            config.candidate_hypotheses_required = job["hypotheses_required"]
            config.agent_tools = [AgentTool(tool) for tool in plan.shared_agent_tools]
            resource = config.resources_for(job["task_id"]).model_copy(deep=True)
            arm = views["arms"][job["arm"]][job["task_id"]]
            resource.task_config_path = Path(arm["task_config_path"])
            resource.agent_task_config_path = Path(arm["agent_task_config_path"])
            resource.research_budget = plan.research_budget
            resource.judge_cpuset_cpus = cpu_pool
            config.tasks[job["task_id"]] = resource
            config.agent_cpuset_cpus = cpu_pool
            configure_study_scenarios(config, job["task_id"], views)
            configure_study_agent(config, job, plan.research_budget)
            config_path = condition / "evaluation.local.yaml"
            config_path.write_text(
                yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
            )
            command = [
                sys.executable,
                "-m",
                "pitbench.cli.main",
                "evaluate",
                job["task_id"],
                "--config",
                str(config_path),
                *study_agent_arguments(job),
                "--model",
                job["model"]["model"],
                "--agent-kwarg",
                job["model"]["setting"],
                "--run-id",
                job["run_id"],
                "--n-attempts",
                "1",
                "--n-concurrent",
                "1",
                "--defer-evaluation",
            ]
            with (condition / "controller.log").open("a") as log:
                with launch_lock:
                    delay = max(0.0, next_launch[0] - time.monotonic())
                    if delay:
                        time.sleep(delay)
                    process = subprocess.Popen(
                        command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
                    )
                    next_launch[0] = time.monotonic() + args.dispatch_interval_sec
                    state.update(status="development_running", pid=process.pid)
                    write_json(path, state)
                code = process.wait()
            audit = audit_model_run(run_path, job["task_id"], evaluation_required=False)
            state.update(
                returncode=code,
                audit=audit,
                status="evaluation_pending"
                if audit["complete"]
                else "development_failed",
            )
            if audit["complete"]:
                write_json(
                    args.output / "evaluation-queue" / (job["run_id"] + ".json"),
                    {
                        **job,
                        **audit,
                        "evaluation_config": str(config_path),
                        "status": "queued_for_independent_measurement"
                        if getattr(args, "evaluate_candidates", False)
                        else "awaiting_scientific_gate",
                    },
                )
        except Exception as error:
            state.update(
                status="development_failed", error=f"{type(error).__name__}: {error}"
            )
        finally:
            slots.put(cpu_pool)
        state["finished_at"] = now()
        write_json(path, state)
        print(
            json.dumps(
                {
                    "run_id": job["run_id"],
                    "task": job["task_id"],
                    "arm": job["arm"],
                    "status": state["status"],
                }
            ),
            flush=True,
        )
        return state

    evaluate = getattr(args, "evaluate_candidates", False)
    evaluation_pools = getattr(args, "evaluation_cpu_pools", [])
    evaluation_slots = queue.SimpleQueue()
    for cpu_pool in evaluation_pools:
        evaluation_slots.put(cpu_pool)

    def collect(state):
        cpu_pool = evaluation_slots.get()
        try:
            return evaluate_study_candidate(state, args, cpu_pool)
        finally:
            evaluation_slots.put(cpu_pool)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=getattr(args, "evaluation_workers", 1)
    ) as judges:
        measured = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(args.workers, len(pools))
        ) as agents:
            futures = [agents.submit(one, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                state = future.result()
                if evaluate and state["status"] == "evaluation_pending":
                    measured.append(judges.submit(collect, state))
        for future in concurrent.futures.as_completed(measured):
            future.result()
    if evaluate:
        records = []
        for job in jobs:
            path = args.output / "agent-status" / job["run_id"] / "status.json"
            state = json.loads(path.read_text()) if path.exists() else job
            records.append(
                {
                    **state,
                    "condition_id": job["run_id"],
                    "model": job["model"]["model"],
                    "setting": job["model"]["setting"],
                }
            )
        index = args.output / "model-runs.json"
        write_json(index, records)
        write_campaign_report(
            args.output,
            args.output / "report",
            repository_root=ROOT,
            index=index,
            runs_root=args.output / "runs",
            task_configs={
                key: Path(row["evaluator_task_config"])
                for key, row in views["tasks"].items()
            },
        )
        write_json(
            args.output / "execution-complete.json",
            {
                "finished_at": now(),
                "conditions": len(jobs),
                "statuses": {
                    status: sum(row.get("status") == status for row in records)
                    for status in {row.get("status", "missing") for row in records}
                },
                "scientific_interpretation": "optimization_space_qualification_pending",
            },
        )


def study_parameters(args):
    """Supplement captured patches in a new campaign, without another Agent."""
    source = args.source_cohort.resolve()
    if args.output.resolve().is_relative_to(source):
        raise ValueError(
            "parameter supplementation needs an output outside the original cohort"
        )
    cohort = json.loads((source / "cohort.json").read_text())
    config = EvaluationConfig.from_yaml(args.config)
    if not config.configuration_robustness.enabled:
        raise ValueError(
            "parameter supplementation is disabled in the supplied configuration"
        )
    slots = queue.SimpleQueue()
    for pool in args.evaluation_cpu_pools:
        slots.put(pool)
    jobs = [
        job
        for job in cohort["jobs"]
        if not args.task_id or job["task_id"] == args.task_id
    ]
    write_json(
        args.output / "supplementation.json",
        {"source_cohort": str(source), "jobs": jobs, "new_agent_starts": 0},
    )

    def run(job):
        output = args.output / job["run_id"]
        output.mkdir(parents=True, exist_ok=True)
        status = {
            **job,
            "started_at": now(),
            "status": "preparing",
            "new_agent_starts": 0,
        }
        pool = slots.get()
        try:
            original = json.loads(
                (source / "agent-status" / job["run_id"] / "status.json").read_text()
            )
            if original["status"] not in {
                "evaluation_completed",
                "evaluation_incomplete",
            }:
                status.update(
                    status="not_run",
                    reason="source development or native evaluation did not complete",
                )
                return status
            execution = original.get("execution_run_id", job["run_id"])
            audit = original["audit"]
            patch = Path(audit["candidate_patch"])
            task_path = source / "workspaces" / execution / job["task_id"]
            context = yaml.safe_load((task_path / "task.yaml").read_text())[
                "evaluator_config"
            ]
            resources = config.resources_for(job["task_id"])
            for key in (
                "judge_image",
                "judge_cpus",
                "judge_memory",
                "judge_parallel_runs",
            ):
                value = getattr(resources, key, None) or getattr(config, key, None)
                if value is not None:
                    context[key] = value
            context.update(
                judge_cpuset_cpus=pool,
                build_cache_path=str((args.output / "shared-cache/build").resolve()),
            )
            request = EvaluationRequest(
                task_id=job["task_id"],
                task_path=task_path,
                candidate_patch_path=patch,
                candidate_patch_sha256=audit["candidate_patch_sha256"],
                output_dir=output,
                agent_name=job.get("agent_import_path") or job["model"]["agent"],
                model_name=job["model"]["model"],
                evaluator_config=context,
            )
            status.update(status="running", cpu_pool=pool)
            write_json(output / "status.json", status)
            directory = run_candidate_configuration(
                request, config.configuration_robustness
            )
            gaps = _configuration_dry_run_failures(directory)
            status.update(
                status="completed" if not gaps else "incomplete",
                metric_gaps=gaps,
                report=str(directory / "configuration_report.json"),
            )
        except Exception as error:
            status.update(status="failed", error=f"{type(error).__name__}: {error}")
        finally:
            status["finished_at"] = now()
            write_json(output / "status.json", status)
            slots.put(pool)
        return status

    records = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.evaluation_workers
    ) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())
            write_json(args.output / "parameter-results.json", records)
    if any(record["status"] != "completed" for record in records):
        raise SystemExit(1)


def parse_run_arguments(parser, argv):
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--profile", type=Path)
    selected, _ = preliminary.parse_known_args(argv)
    profile_path = selected.profile or ROOT / "config/study.local.yaml"
    if selected.profile is not None or profile_path.is_file():
        payload = yaml.safe_load(profile_path.read_text()) or {}
        allowed = {action.dest for action in parser._actions}
        unknown = set(payload) - allowed
        if unknown:
            parser.error(f"unknown run profile options: {sorted(unknown)}")
        parser.set_defaults(**payload)
    args = parser.parse_args(argv)
    required = (
        ("output", "phase")
        if args.phase == "study-status"
        else ("manifest", "config", "output", "phase")
    )
    for name in required:
        if getattr(args, name) is None:
            parser.error(f"--{name} is required, directly or through --profile")
    return args


def launch_background(args, argv):
    receipt = args.output / "launcher-process.json"
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if Path(f"/proc/{old['pid']}").exists():
            print(json.dumps({**old, "status": "already_running"}), flush=True)
            return
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        *argv,
        "--foreground",
    ]
    with (args.output / "launcher.log").open("a") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    result = {
        "pid": process.pid,
        "started_at": now(),
        "status": "started",
        "log": str(args.output / "launcher.log"),
    }
    write_json(receipt, result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--foreground", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--evaluate-candidates", action="store_true")
    parser.add_argument("--evaluation-workers", type=int, default=1)
    parser.add_argument("--evaluation-cpu-pools", action="append", default=[])
    parser.add_argument(
        "--retry-evaluation-status",
        dest="retry_evaluation_statuses",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--phase",
        choices=[
            "build",
            "baseline",
            "models",
            "report",
            "all",
            "_judge",
            "study-models",
            "study-evaluations",
            "study-status",
            "study-agent-recovery",
            "study-parameters",
        ],
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--study-plan", type=Path)
    parser.add_argument("--study-views", type=Path)
    parser.add_argument("--source-cohort", type=Path)
    parser.add_argument("--schedule-seed", type=int, default=20261101)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume-models", action="store_true")
    parser.add_argument("--defer-evaluation-incomplete", action="store_true")
    parser.add_argument("--dispatch-interval-sec", type=float, default=0)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--retry-run-id")
    parser.add_argument("--task-id")
    parser.add_argument("--cpu-pool")
    parser.add_argument("--cpu-pools", action="append", default=[])
    args = parse_run_arguments(parser, sys.argv[1:])
    for name in ("manifest", "config", "output"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.phase == "study-status":
        if not (args.output / "cohort.json").is_file():
            parser.error("study-status requires an existing cohort.json")
        records = refresh_study_index(args.output)
        print(
            json.dumps(
                {
                    "total": len(records),
                    "statuses": dict(Counter(row["status"] for row in records)),
                }
            ),
            flush=True,
        )
        return
    if args.workers < 1 or (args.cpu_pools and len(args.cpu_pools) != args.workers):
        parser.error("positive workers and one CPU pool per worker are required")
    if (
        args.phase in ("models", "all", "study-models")
        and args.workers > 1
        and not args.cpu_pools
    ):
        parser.error("parallel model runs require disjoint --cpu-pools")
    if args.evaluate_candidates or args.phase in {
        "study-parameters",
        "study-evaluations",
    }:
        if (
            args.evaluation_workers < 1
            or len(args.evaluation_cpu_pools) != args.evaluation_workers
        ):
            parser.error(
                "candidate evaluation requires one CPU pool per evaluation worker"
            )

        occupied = set()
        agent_pools = (
            args.cpu_pools
            if args.phase not in {"study-evaluations", "study-parameters"}
            else []
        )
        for pool in agent_pools + args.evaluation_cpu_pools:
            try:
                cpus = study_cpu_ids(pool)
            except ValueError as error:
                parser.error(str(error))
            if not cpus or cpus & occupied:
                parser.error(
                    "Agent and evaluation CPU pools must be nonempty and disjoint"
                )
            occupied |= cpus
    args.output.mkdir(parents=True, exist_ok=True)
    if args.background and not args.foreground and not args.prepare_only:
        launch_background(args, sys.argv[1:])
        return
    manifest = json.loads(args.manifest.read_text())
    task_ids = [row["task_id"] for row in manifest["task_matrix"]]
    if len(task_ids) != len(set(task_ids)):
        parser.error("manifest task IDs must be unique")
    config = EvaluationConfig.from_yaml(args.config)
    if args.phase == "study-parameters":
        if args.source_cohort is None:
            parser.error("parameter supplementation requires --source-cohort")
        with study_cpu_lease(args.evaluation_cpu_pools):
            study_parameters(args)
        return
    if args.phase in {"study-models", "study-evaluations", "study-agent-recovery"}:
        if args.study_plan is None or args.study_views is None:
            parser.error("study phases require --study-plan and --study-views")
        args.study_plan = args.study_plan.resolve()
        args.study_views = args.study_views.resolve()
        if args.phase == "study-models":
            pools = args.cpu_pools + (
                args.evaluation_cpu_pools if args.evaluate_candidates else []
            )
            with study_cpu_lease(pools):
                study_models(args)
        elif args.phase == "study-agent-recovery":
            if not args.retry_run_id or not args.cpu_pool:
                parser.error("a retry run ID and a dedicated CPU pool are required")
            with study_cpu_lease([args.cpu_pool]):
                retry_unexecuted_condition(args)
        else:
            study_evaluations(args)
        return
    if args.phase in {"models", "all", "report"}:
        require_plotting()
    for row in manifest["task_matrix"]:
        task = TaskCatalog(ROOT).validate_one(row["task_id"]).task
        release = manifest["selected_releases"][row["repository"]][row["role"]]
        if (task.release.base_commit, task.release.tree_sha) != (
            release["commit_sha"],
            release["tree_sha"],
        ):
            parser.error(f"{task.task_id}: source differs from pinned release")
    if args.phase == "_judge":
        if args.task_id not in task_ids:
            parser.error("worker task is absent from manifest")
        wait_for_capacity(args)
        result = judge_worker(args.task_id, args, config)
        write_json(args.output / "baseline" / args.task_id / "result.json", result)
        return
    if args.phase == "report":
        report(task_ids, args, config)
        return
    for phase in ("build", "baseline"):
        if args.phase in (phase, "all"):
            if not run_phase(phase, task_ids, args, config):
                raise SystemExit(1)
            config = EvaluationConfig.from_yaml(args.config)
    if args.phase in ("models", "all"):
        if args.models is None:
            parser.error("--models is required for model runs")
        models(task_ids, args)


if __name__ == "__main__":
    main()
