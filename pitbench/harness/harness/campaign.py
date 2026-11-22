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
import math
import multiprocessing
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import Counter
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path.cwd().resolve()
from adapters.pitbench.adapter import PitBenchAdapter
from pitbench.agent_tools import AgentTool
from pitbench.cli.evaluate_config import (
    EvaluationConfig,
    EvaluationPaths,
    resolve_repository_path,
)
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


def read_study_plan(path):
    from pitbench.schema.research import GeneralizationStudyPlan

    return GeneralizationStudyPlan.model_validate(yaml.safe_load(Path(path).read_text()))


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


def available_study_sockets(
    *, topology=Path("/sys/devices/system/cpu"), lock_root=None, excluded=()
):
    """One available hardware thread per physical core, grouped by socket."""
    allowed = sorted(os.sched_getaffinity(0))
    lock_root = lock_root or Path(f"/tmp/pitbench-study-cpus-{os.getuid()}")
    reserved = set(excluded)
    for cpu in allowed:
        path = lock_root / f"cpu-{cpu}.lock"
        if path.exists():
            with path.open("a") as handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    reserved.add(cpu)
    cores = {}
    for cpu in allowed:
        folder = topology / f"cpu{cpu}" / "topology"
        if folder.is_dir():
            socket = int((folder / "physical_package_id").read_text())
            core = int((folder / "core_id").read_text())
        else:
            socket, core = 0, cpu
        cores.setdefault((socket, core), []).append(cpu)
    sockets = {}
    for (socket, _), siblings in sorted(cores.items()):
        if not reserved.intersection(siblings):
            sockets.setdefault(socket, []).append(min(siblings))
    return list(sockets.values())


def automatic_study_pools(
    sockets, cpus_per_worker, memory_gib, available_gib, reserve_gib=0
):
    """Spread each worker over sockets instead of filling one hot socket first."""
    count = math.ceil(cpus_per_worker)
    if count < 1 or memory_gib <= 0:
        raise ValueError(
            "automatic scheduling requires positive CPU allocations and memory"
        )
    ordered = []
    for index in range(max(map(len, sockets), default=0)):
        ordered.extend(group[index] for group in sockets if index < len(group))
    slots = min(
        len(ordered) // count, int(max(0, available_gib - reserve_gib) // memory_gib)
    )
    return [
        ",".join(map(str, ordered[start : start + count]))
        for start in range(0, slots * count, count)
    ]


def study_judge_resources(config, task_ids):
    from docker.utils import parse_bytes

    cpus, memory = [], []
    for task_id in task_ids:
        resources = config.resources_for(task_id)
        cpus.append(float(resources.judge_cpus or config.judge_cpus or 8))
        memory.append(
            parse_bytes(resources.judge_memory or config.judge_memory or "8g") / 1024**3
        )
    return max(cpus, default=8), max(memory, default=8)


def configure_study_parallelism(args):
    """Resolve omitted worker counts from host capacity; explicit pools stay explicit."""
    args.dynamic_evaluation_workers = args.phase in {"study-evaluations", "study-parameters"} and not args.evaluation_cpu_pools
    args.dynamic_evaluation_limit = args.evaluation_workers
    if args.workers < 0 or args.evaluation_workers < 0:
        raise ValueError("worker counts must be nonnegative (0 means automatic)")
    if args.phase == "baseline":
        if args.cpu_pools:
            args.workers = args.workers or len(args.cpu_pools)
            if len(args.cpu_pools) != args.workers:
                raise ValueError("baseline requires one CPU pool per worker")
            return
        config = EvaluationConfig.from_yaml(args.config)
        manifest = json.loads(args.manifest.read_text())
        task_ids = [row["task_id"] for row in manifest["task_matrix"]]
        if not task_ids:
            raise ValueError("BASE manifest contains no tasks")
        from docker.utils import parse_bytes

        cpus = max(
            float(config.resources_for(task_id).judge_cpus or config.judge_cpus or 8)
            for task_id in task_ids
        )
        memory_gib = max(
            parse_bytes(
                config.resources_for(task_id).judge_memory
                or config.judge_memory
                or "32g"
            )
            / 1024**3
            for task_id in task_ids
        )
        policy = (
            args.resource_policy
            if isinstance(args.resource_policy, dict)
            else json.loads(Path(args.resource_policy).read_text())
            if args.resource_policy
            else {}
        )
        pools = automatic_study_pools(
            available_study_sockets(),
            cpus,
            memory_gib,
            host_snapshot()["available_memory_gib"],
            policy.get("min_available_memory_gib", 0),
        )
        if args.workers:
            if len(pools) < args.workers:
                raise ValueError(
                    f"requested {args.workers} BASE workers but only {len(pools)} fit current CPU and memory capacity"
                )
            pools = pools[: args.workers]
        if not pools:
            raise ValueError("no BASE worker fits current CPU and memory capacity")
        args.cpu_pools = pools
        args.workers = len(pools)
        write_json(
            args.output / "baseline-parallelism.json",
            {
                "workers": args.workers,
                "cpu_pools": args.cpu_pools,
                "judge_cpus_per_worker": cpus,
                "judge_memory_gib_per_worker": memory_gib,
            },
        )
        return
    if args.phase not in {
        "study-models",
        "study-evaluations",
        "study-parameters",
        "study-agent-recovery",
    }:
        args.workers = args.workers or 1
        args.evaluation_workers = args.evaluation_workers or 1
        return
    if args.cpu_pools:
        args.workers = args.workers or len(args.cpu_pools)
    if args.evaluation_cpu_pools:
        args.evaluation_workers = args.evaluation_workers or len(
            args.evaluation_cpu_pools
        )
    needs_agents = (
        args.phase == "study-models"
        and not args.cpu_pools
        and (not args.workers or args.workers > 1)
    )
    needs_judges = (
        args.phase in {"study-evaluations", "study-parameters"}
        and not args.evaluation_cpu_pools
    )
    if needs_agents or needs_judges:
        config = EvaluationConfig.from_yaml(args.config)
        plan = (
            read_study_plan(args.study_plan)
            if args.study_plan
            else None
        )
        matrix_path = getattr(args, "target_task_matrix", None)
        if needs_judges and matrix_path:
            matrix = yaml.safe_load(Path(matrix_path).read_text())
            task_ids = sorted({task_id for targets in matrix.values() for task_id in targets})
        elif plan is not None:
            task_ids = [Path(path).stem for path in plan.task_configs]
        else:
            source = args.source_cohort or args.output
            task_ids = sorted(
                {
                    job["task_id"]
                    for job in json.loads((source / "cohort.json").read_text())["jobs"]
                }
            )
        judge_cpus, judge_memory = study_judge_resources(config, task_ids)
        policy_path = args.output / "resource-policy.json"
        policy = json.loads(policy_path.read_text()) if policy_path.is_file() else {}
        reserve = policy.get("min_available_memory_gib", 0)
        excluded = (
            set().union(*(study_cpu_ids(pool) for pool in args.evaluation_cpu_pools))
            if needs_agents
            else set()
        )
        sockets = available_study_sockets(excluded=excluded)
        if needs_agents:
            cpus = (
                max(plan.research_budget.cpus, judge_cpus)
                if args.evaluate_candidates
                else plan.research_budget.cpus
            )
            memory = (
                max(plan.research_budget.memory_gib, judge_memory)
                if args.evaluate_candidates
                else plan.research_budget.memory_gib
            )
        else:
            cpus, memory = judge_cpus, judge_memory
        available = host_snapshot()["available_memory_gib"]
        if needs_agents:
            available -= len(args.evaluation_cpu_pools) * judge_memory
        pools = automatic_study_pools(sockets, cpus, memory, available, reserve)
        if not pools and not args.dynamic_evaluation_workers:
            raise ValueError(
                "no free CPU/memory allocation fits the configured study budget"
            )
        if needs_agents:
            dedicated = (
                min(args.evaluation_workers or 1, max(0, len(pools) - 1))
                if args.evaluate_candidates and not args.evaluation_cpu_pools
                else 0
            )
            if args.workers:
                pools = pools[: args.workers + dedicated]
            if args.evaluate_candidates and not args.evaluation_cpu_pools:
                args.evaluation_cpu_pools = pools[:dedicated]
            args.cpu_pools = pools[dedicated:]
            args.workers = len(args.cpu_pools)
            args.evaluation_workers = len(args.evaluation_cpu_pools) or 1
        else:
            args.evaluation_cpu_pools = pools[: args.evaluation_workers or len(pools)]
            args.evaluation_workers = len(args.evaluation_cpu_pools)
    args.workers = args.workers or 1
    args.evaluation_workers = (
        len(args.evaluation_cpu_pools) if args.dynamic_evaluation_workers else
        args.evaluation_workers or len(args.evaluation_cpu_pools) or 1
    )
    write_json(
        parallelism_path(args),
        {
            "agent_workers": args.workers,
            "agent_cpu_pools": args.cpu_pools,
            "evaluation_workers": args.evaluation_workers,
            "evaluation_cpu_pools": args.evaluation_cpu_pools,
            "reuse_agent_pools": args.reuse_agent_pools,
        },
    )


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


def study_live_snapshot(output):
    """Read current campaign status and the latest recorded Agent activity."""
    records = refresh_study_index(output)
    activities = []
    effective_statuses = Counter(row["status"] for row in records)
    for row in records:
        original_id = row.get("run_id") or row["condition_id"]
        execution_id = original_id
        if row.get("status") != "development_running":
            receipts = sorted(
                (output / "infrastructure-retries" / original_id).glob("infra*/retry.json"),
                key=lambda path: int(path.parent.name[5:]),
            )
            retry = json.loads(receipts[-1].read_text()) if receipts else None
            if not retry or retry.get("status") != "retry_running":
                continue
            execution_id = retry["retry_run_id"]
            effective_statuses[row["status"]] -= 1
            effective_statuses["recovery_running"] += 1
        run = output / "runs" / execution_id
        traces = sorted(run.glob("*/*/agent-trace/*/events.jsonl"))
        detail = None
        if traces:
            trace = traces[-1]
            with trace.open("rb") as stream:
                stream.seek(max(0, trace.stat().st_size - 2 * 1024 * 1024))
                lines = stream.read().decode(errors="replace").splitlines()
            for line in reversed(lines):
                try:
                    event = json.loads(line)
                    data = event.get("data") or {}
                    if data.get("path"):
                        data = json.loads((trace.parent / data["path"]).read_text())
                    if event.get("event") == "agent.progress":
                        detail = data.get("detail")
                    elif event.get("event") in {"tool.started", "tool.result"}:
                        inputs = data.get("inputs") or {}
                        tool = (
                            inputs.get("ToolName")
                            if isinstance(inputs, dict)
                            else None
                        ) or data.get("name") or "tool"
                        phase = "Running" if event["event"] == "tool.started" else "Returned"
                        detail = f"Agent: {phase} {tool}"
                    else:
                        continue
                    break
                except (OSError, ValueError, TypeError):
                    continue
        activities.append({
            "run_id": original_id, "execution_run_id": execution_id,
            "task_id": row["task_id"], "model": row["model"], "activity": detail,
        })
    return {"total": len(records), "statuses": {key: value for key, value in effective_statuses.items() if value},
            "active_agents": activities}


def study_live_events(output, snapshot, cursors):
    """Stream newly recorded Agent tool calls without replaying historical logs."""
    events = []
    for activity in snapshot["active_agents"]:
        run = output / "runs" / activity["execution_run_id"]
        for trace in sorted(run.glob("*/*/agent-trace/*/events.jsonl")):
            size = trace.stat().st_size
            if trace not in cursors:
                cursors[trace] = size
                continue
            position = cursors[trace]
            if position > size:
                position = 0
            with trace.open("rb") as stream:
                stream.seek(position)
                while line := stream.readline():
                    if not line.endswith(b"\n"):
                        break
                    cursors[trace] = stream.tell()
                    try:
                        event = json.loads(line)
                        if event.get("event") not in {
                            "mcp.started", "mcp.result", "tool.started"
                        }:
                            continue
                        data = event.get("data") or {}
                        if data.get("path"):
                            data = json.loads((trace.parent / data["path"]).read_text())
                        events.append({"timestamp": event.get("timestamp"),
                            "run_id": activity["run_id"], "task_id": activity["task_id"],
                            "model": activity["model"], "event": event["event"], "data": data})
                    except (OSError, ValueError, TypeError):
                        continue
    return events


def _live_event_summary(event):
    """Render a compact terminal line for an Agent tool event."""
    data = event.get("data") or {}
    event_type = event.get("event", "")
    if event_type in {"tool.started", "tool.result"}:
        inputs = data.get("inputs") or {}
        tool = (
            inputs.get("ToolName")
            if isinstance(inputs, dict)
            else None
        ) or data.get("name") or "tool"
        detail = ""
        if event_type == "tool.started":
            arguments = inputs.get("Arguments") if isinstance(inputs, dict) else None
            command = arguments.get("command") if isinstance(arguments, dict) else None
            if command:
                detail = " ".join(str(command).split())
        elif data.get("result_present"):
            detail = " ".join(str(data.get("result", "")).split())
        detail = detail[:180] + ("…" if len(detail) > 180 else "")
        stamp = str(event.get("timestamp") or "").replace("T", " ")[:19]
        identity = f"{event.get('model', '?')} / {event.get('task_id', '?')}"
        phase = "→" if event_type == "tool.started" else "←"
        return f"{stamp}  {identity}  {phase} {tool}" + (f"  {detail}" if detail else "")
    nested = data.get("result") if isinstance(data, dict) else None
    payload = nested if isinstance(nested, dict) else data
    command = payload.get("command") if isinstance(payload, dict) else None
    tool = data.get("tool") or data.get("name") or data.get("tool_name")
    if isinstance(tool, dict):
        tool = tool.get("name") or tool.get("tool")
    if command and tool in (None, "result", "mcp.result"):
        tool = "command"
    tool = str(tool or event.get("event", "tool").removeprefix("mcp."))
    detail = ""
    if command:
        detail = " ".join(str(command).split())
    elif isinstance(payload, dict):
        result = payload.get("content")
        if isinstance(result, list):
            texts = [item.get("text", "") for item in result if isinstance(item, dict)]
            detail = " ".join(" ".join(texts).split())
        elif payload.get("error"):
            detail = str(payload["error"])
    detail = detail[:180] + ("…" if len(detail) > 180 else "")
    stamp = str(event.get("timestamp") or "").replace("T", " ")[:19]
    identity = f"{event.get('model', '?')} / {event.get('task_id', '?')}"
    phase = "→" if event_type == "mcp.started" else "←"
    return f"{stamp}  {identity}  {phase} {tool}" + (f"  {detail}" if detail else "")


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
        if (args.output / "STOP").exists():
            return False
        health = host_snapshot()
        reasons = study_admission_reasons(policy, health)
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
    resources = config.resources_for(task_id)
    configured_task = resources.task_config_path
    record = (
        TaskCatalog(ROOT).validate_file(resolve_repository_path(ROOT, configured_task))
        if configured_task is not None
        else TaskCatalog(ROOT).validate_one(task_id)
    )
    if record.task.task_id != task_id:
        raise ValueError(f"{task_id}: BASE task view has a different task ID")
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
    if getattr(args, "base_only", False):
        evaluator_config["code_states"] = ["base"]
    if args.cpu_pool:
        evaluator_config["judge_cpuset_cpus"] = args.cpu_pool
    if not getattr(args, "base_only", False) and task_id in config.metric_scenario_configs:
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
    if getattr(args, "base_only", False):
        from pitbench.schema.observation import ExpectedRunGrid

        payload = envelope.payload
        if not payload.get("validity", {}).get("accepted"):
            raise RuntimeError("BASE-only empty-patch validity check failed")
        artifact = payload.get("artifacts", {}).get("observations") or {}
        grid_artifact = payload.get("artifacts", {}).get("expected_run_grid") or {}
        if not artifact.get("path") or not grid_artifact.get("path"):
            raise RuntimeError("BASE-only observations or expected grid are missing")
        observations = ObservationStore.read(directory / artifact["path"])
        grid = ExpectedRunGrid.model_validate_json(
            (directory / grid_artifact["path"]).read_text()
        )

        def identity(row):
            return (
                row.task_id,
                row.code_state,
                row.instance_set,
                row.instance_id,
                row.solver_seed,
                row.budget_sec,
                row.test_suite,
                row.equivalence_parent_id,
            )

        expected = [row for row in grid.runs if row.code_state.value == "base"]
        observed = [row for row in observations if row.code_state.value == "base"]
        if (
            len(observed) != len(expected)
            or len({identity(row) for row in observed}) != len(observed)
            or {identity(row) for row in observed}
            != {identity(row) for row in expected}
        ):
            raise RuntimeError(
                f"BASE-only grid incomplete: observed={len(observed)} expected={len(expected)}"
            )
        primary_expected = [
            row
            for row in expected
            if row.test_suite is None and row.equivalence_parent_id is None
        ]
        primary_observed = [
            row
            for row in observed
            if row.test_suite is None and row.equivalence_parent_id is None
        ]
        infrastructure = [
            row
            for row in observed
            if row.status in {RunStatus.INFRASTRUCTURE_ERROR, RunStatus.BUILD_FAILED}
        ]
        if infrastructure:
            raise RuntimeError(
                f"{len(infrastructure)} BASE observations are infrastructure/build failures"
            )
        performance = payload.get("summary", {}).get("performance") or {}
        primary = performance.get("primary", {})
        base = primary.get("base")
        if base is None:
            budgets = performance.get("by_budget", {})
            primary_budget = record.task.evaluation.primary_budget_sec
            base = next(
                (
                    value.get("base")
                    for key, value in budgets.items()
                    if float(key) == float(primary_budget)
                ),
                None,
            )
        from collections import defaultdict

        difficulty_by_budget = {}
        if record.task.task_type.value == "exact_solver":
            from pitbench.evaluator.private_assets import PrivateAssetResolver
            from pitbench.metrics.performance_report import compute_exact_run_outcome
            from pitbench.problem_families.verification import TrustedOptimumOracle

            oracle_path = PrivateAssetResolver(
                Path(evaluator_config["private_root"])
            ).resolve(record.task.oracle.source, record.task.oracle.source_sha256)
            oracle = TrustedOptimumOracle.from_yaml(oracle_path)
            target_kind_counts = dict(
                Counter(item.target_kind for item in oracle.records)
            )

            grouped = defaultdict(list)
            for row in primary_observed:
                grouped[float(row.budget_sec)].append(
                    compute_exact_run_outcome(
                        row, time_basis=record.task.evaluation.exact_time_basis
                    )
                )
            for budget, outcomes in sorted(grouped.items()):
                solved = sum(item.solved is True for item in outcomes)
                expected_count = sum(
                    float(row.budget_sec) == budget for row in primary_expected
                )
                timed = [
                    item.penalized_runtime_sec
                    for item in outcomes
                    if item.penalized_runtime_sec is not None
                ]
                difficulty_by_budget[str(budget)] = {
                    "protocol": "solver_optimality_and_target_match_par2_diagnostic",
                    "oracle_kind": record.task.oracle.kind,
                    "oracle_record_target_kinds": target_kind_counts,
                    "interpretation": (
                        "An OPTIMAL solver status matching a best-found target is a "
                        "BASE difficulty diagnostic, not independent proof that the "
                        "target is globally optimal."
                    ),
                    "solved": solved,
                    "expected": expected_count,
                    "verified_solved_coverage": solved / expected_count
                    if expected_count
                    else None,
                    "mean_par2_sec": sum(timed) / expected_count
                    if expected_count and len(timed) == expected_count
                    else None,
                    "outcomes": dict(Counter(item.kind.value for item in outcomes)),
                }
        else:
            for budget in sorted({float(row.budget_sec) for row in primary_expected}):
                rows = [row for row in primary_observed if float(row.budget_sec) == budget]
                expected_count = sum(
                    float(row.budget_sec) == budget for row in primary_expected
                )
                gaps = [
                    row.normalized_gap
                    for row in rows
                    if row.valid
                    and row.normalized_gap is not None
                    and math.isfinite(row.normalized_gap)
                ]
                difficulty_by_budget[str(budget)] = {
                    "protocol": "fixed_budget_normalized_gap",
                    "valid_gap_observations": len(gaps),
                    "expected": expected_count,
                    "mean_normalized_gap": sum(gaps) / len(gaps) if gaps else None,
                }
        return {
            "observation_count": len(observations),
            "valid_observation_count": sum(item.valid for item in observations),
            "base_observation_count": len(observed),
            "base_performance_protocol": record.task.evaluation.performance_protocol.value,
            "base_primary": base,
            "base_difficulty_diagnostic": difficulty_by_budget,
            "base_by_budget": {
                key: value.get("base")
                for key, value in performance.get("by_budget", {}).items()
            },
            "scope": "full_declared_BASE_grid_no_agent_calls",
        }
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
        catalog = TaskCatalog(ROOT)
        configured = config.resources_for(task_id).task_config_path
        record = (
            catalog.validate_file(resolve_repository_path(ROOT, configured))
            if configured is not None
            else catalog.validate_one(task_id)
        )
        if record.task.task_id != task_id:
            raise ValueError(f"{task_id}: configured task has a different task ID")
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
                    "-m", "pitbench.harness.harness.campaign",
                    "--root", str(ROOT),
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
                if getattr(args, "base_only", False):
                    command.append("--base-only")
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
    relay_network_failed = False
    trace_closed_during_cleanup = False
    quota_delay = None
    quota_log = None
    conversation_id = None
    agent_duration_sec = 0.0
    interrupted_after_quota = False
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
    agy_transcript = folder / "agent-logs/antigravity.jsonl"
    agy_stderr = folder / "agent-logs/antigravity.stderr.log"
    if agy_transcript.is_file() or agy_stderr.is_file():
        from pitbench.harness.agents.antigravity_mcp_agent import AntigravityMCPAgent

        stdout = agy_transcript.read_text(errors="replace") if agy_transcript.is_file() else ""
        stderr = agy_stderr.read_text(errors="replace") if agy_stderr.is_file() else ""
        agy_events = list(AntigravityMCPAgent._events(stdout))
        input_tokens, output_tokens = AntigravityMCPAgent._parse_usage(stdout)
        result_positions = [i for i, event in enumerate(agy_events) if event.get("event") == "result"]
        last_result = result_positions[-1] if result_positions else -1
        after_last_result = agy_events[last_result + 1:]
        resumed_after_result = any(event.get("event") == "init" for event in after_last_result)
        trailing_usage = {}
        if resumed_after_result:
            for event in after_last_result:
                step = event.get("step_update") or {}
                if step.get("step_type") == "agent_response" and step.get("state") == "DONE" and step.get("usage"):
                    trailing_usage[step.get("step_index")] = step["usage"]
        tokens = max(tokens, input_tokens + output_tokens + sum(
            int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)
            for usage in trailing_usage.values()
        ))
        completed_turn |= AntigravityMCPAgent._has_successful_result(stdout) and not resumed_after_result
        waited_quota_sec = 0.0
        for position in result_positions:
            event = agy_events[position]
            delay = AntigravityMCPAgent._quota_retry_delay(json.dumps(event), "")
            if delay is not None and any(item.get("event") == "init" for item in agy_events[position + 1:]):
                waited_quota_sec += delay + 5
        interrupted_after_quota = resumed_after_result and waited_quota_sec > 0 and failure == "agent_timeout"
        quota_delay = None if resumed_after_result else AntigravityMCPAgent._quota_retry_delay(
            json.dumps(agy_events[last_result]) if last_result >= 0 else "", stderr
        )
        quota_log = agy_stderr if agy_stderr.is_file() else agy_transcript
        for event in agy_events:
            if event.get("event") == "result":
                result = event.get("result") or {}
                conversation_id = result.get("conversation_id") or conversation_id
                agent_duration_sec += float(result.get("duration_seconds") or 0)
        if trial.get("agent_started_at") and trial.get("agent_ended_at"):
            start = datetime.fromisoformat(trial["agent_started_at"])
            end = datetime.fromisoformat(trial["agent_ended_at"])
            agent_duration_sec = max(agent_duration_sec, (end - start).total_seconds() - waited_quota_sec)
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
    for trace_path in run_dir.glob(
        f"{task_id}/*/agent-trace/*/events.jsonl"
    ):
        for line in trace_path.read_text(errors="replace").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("event") != "model.rejected":
                continue
            reference = (event.get("data") or {}).get("path")
            if not reference:
                continue
            detail_path = trace_path.parent / reference
            try:
                detail = json.loads(detail_path.read_text())
            except (OSError, ValueError):
                continue
            message = str(detail.get("message") or "").lower()
            relay_network_failed |= (
                detail.get("status") == 502
                and "model relay failed" in message
                and any(
                    marker in message
                    for marker in (
                        "network is unreachable",
                        "connection refused",
                        "connection timed out",
                        "timed out",
                        "unexpected_eof",
                        "eof occurred",
                    )
                )
            )
    evidence = {
        "failure_mode": failure,
        "tokens": tokens,
        "completed_turn": completed_turn,
        "trace_closed_during_cleanup": trace_closed_during_cleanup,
        "harness_result": str(path),
    }
    candidate = folder / "evaluation/candidate.patch"
    if (quota_delay is not None or interrupted_after_quota) and not completed_turn:
        return {
            **evidence,
            "complete": False,
            "retryable": tokens == 0,
            "reason": "antigravity quota reached" if quota_delay is not None else "antigravity interrupted after quota wait",
            "quota_retry_after": datetime.fromtimestamp(
                quota_log.stat().st_mtime + (quota_delay or 0) + 5, tz=timezone.utc
            ).isoformat(),
            "continuation_required": tokens > 0,
            "candidate_patch": str(candidate) if candidate.is_file() else None,
            "conversation_id": conversation_id,
            "agent_duration_sec": agent_duration_sec,
            "development_storage": str(folder / "development-storage")
            if (folder / "development-storage").is_dir() else None,
        }
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
    if failure == "unknown_agent_error" and tokens == 0 and relay_network_failed:
        return {
            **evidence,
            "complete": False,
            "retryable": True,
            "reason": "Codex relay upstream network unavailable; no model execution",
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
    if audit.get("reason") in {"antigravity quota reached", "antigravity interrupted after quota wait"}:
        retry_after = audit.get("quota_retry_after")
        if retry_after and datetime.fromisoformat(retry_after) <= datetime.now(timezone.utc):
            if audit.get("retryable"):
                return "quota_reset"
            if audit.get("continuation_required") and audit.get("development_storage"):
                return "quota_workspace_continuation"
        return None
    stream_disconnect = (
        audit.get("reason")
        == "model stream disconnected; retain partial infrastructure-failed attempt"
        and int(audit.get("tokens") or 0) == 0
        and not audit.get("completed_turn")
        and not audit.get("candidate_patch")
    )
    relay_network_error = (
        audit.get("reason")
        == "Codex relay upstream network unavailable; no model execution"
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
    if relay_network_error:
        return "codex_relay_network_unavailable"
    return None


def preflight_retry_reason(original_status, audit, controller_log):
    """Permit a fresh attempt only when the preflight confirmed no Agent start."""
    if (
        original_status == "development_failed"
        and audit.get("reason") == "no terminal harness result"
        and audit.get("retryable")
        and controller_log.is_file()
        and "NOT READY: no agent was started." in controller_log.read_text(errors="replace")
    ):
        return "preflight_agent_not_started"
    return None


def write_continuation_context(run_dir, destination, *, append=False):
    """Preserve the interrupted Agent's recorded tool history for its next session."""
    traces = sorted(run_dir.glob("*/*/agent-trace/*/events.jsonl"))
    if not traces:
        raise ValueError("interrupted Agent has no retained tool trace")
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("a" if append else "w") as out:
        for trace in traces:
            with trace.open() as stream:
                for line in stream:
                    try:
                        event = json.loads(line)
                        if event.get("event") not in {"mcp.started", "mcp.result", "agent.progress"}:
                            continue
                        data = event.get("data") or {}
                        if data.get("path"):
                            data = json.loads((trace.parent / data["path"]).read_text())
                        out.write(json.dumps({"event": event["event"], "timestamp": event.get("timestamp"), "data": data}, ensure_ascii=False) + "\n")
                        count += 1
                    except (OSError, ValueError, TypeError, KeyError):
                        continue
    destination.chmod(0o600)
    return count


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


def launch_study_process(command, args, status_path, log, dispatch_at, previous):
    """Honor STOP while waiting and immediately before spending an Agent start."""
    if not wait_for_study_dispatch(args, dispatch_at):
        write_json(status_path, previous)
        return None
    return subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)


def wait_for_study_dispatch(args, dispatch_at):
    """Apply the same STOP, spacing and host admission policy to both stages."""
    while not (args.output / "STOP").exists():
        remaining = dispatch_at - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(remaining, 1.0))
    if (args.output / "STOP").exists() or wait_for_capacity(args) is False:
        return False
    if (args.output / "STOP").exists():
        return False
    policy_path = args.output / "resource-policy.json"
    policy = json.loads(policy_path.read_text()) if policy_path.is_file() else {}
    while not (args.output / "STOP").exists():
        admitted, _ = claim_campaign_dispatch_slot(args.output, policy)
        if admitted:
            return True
        time.sleep(1)
    return False


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
        source_id = previous.get("source_run_id")
        source_root = previous.get("source_cohort")
        if source_id and source_root:
            source_status = Path(source_root) / "agent-status" / source_id / "status.json"
            source = json.loads(source_status.read_text()) if source_status.is_file() else {}
            if not source_candidate_ready(source):
                if previous.get("status") == "evaluation_pending":
                    previous.update(status="awaiting_candidate", evaluation_error="source has no captured candidate",
                                    source_agent_status=source.get("status"))
                    write_json(status_path, previous)
                return previous
        # A queued snapshot must not overwrite a condition finished by another manager.
        if any(
            previous.get(key) != state.get(key)
            for key in ("status", "evaluation_attempt")
        ) or (previous.get("audit") and previous["audit"] != state.get("audit")):
            return previous
        if (args.output / "STOP").exists():
            return previous
        if state.get("interrupted_by_reboot"):
            import copy

            args = copy.copy(args)
            prior_output = previous.get("evaluation_output")
            prior_pool = previous.get("evaluation_cpu_pool")
            args.resume_evaluation = bool(
                prior_output
                and prior_pool
                and study_cpu_ids(prior_pool) == study_cpu_ids(cpu_pool)
                and (Path(prior_output) / "judge-observations.jsonl").is_file()
            )
        previous = {**previous, "audit": state["audit"]}
        return _evaluate_study_candidate(previous, args, cpu_pool)


def _study_metric_status(primary_gaps, extension_gaps, *, completed=True):
    return {
        "status": (
            "evaluation_failed" if not completed else
            "evaluation_incomplete" if primary_gaps else "evaluation_completed"
        ),
        "primary_metrics_status": (
            "unavailable" if not completed else "incomplete" if primary_gaps else "complete"
        ),
        "extension_metrics_status": (
            "unavailable" if not completed else "incomplete" if extension_gaps else "no_gaps_reported"
        ),
        "primary_metric_gaps": primary_gaps,
        "extension_metric_gaps": extension_gaps,
        "metric_gaps": [*primary_gaps, *extension_gaps],
        "metrics_complete": completed and not primary_gaps and not extension_gaps,
    }


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
        primary_metric_gaps=[],
        extension_metric_gaps=[],
        primary_metrics_status="running",
        extension_metrics_status="running",
        metrics_complete=False,
        evaluation_output=None,
        configuration_robustness_output=None,
    )
    write_json(status_path, job)
    refresh_study_index(args.output)
    try:
        if not source_patch.is_file():
            raise ValueError("final captured candidate is missing")
        resume = bool(getattr(args, "resume_evaluation", False) and previous.get("evaluation_output"))
        if resume:
            output = Path(previous["evaluation_output"])
            if not output.is_dir() or not (output / "judge-observations.jsonl").is_file():
                raise ValueError("retained evaluation checkpoint is unavailable")
            if study_cpu_ids(cpu_pool) != study_cpu_ids(previous["evaluation_cpu_pool"]):
                raise ValueError("checkpoint resume requires the original CPU allocation")
            patch = output / "candidate.patch"
            if not patch.is_file() or patch.read_bytes() != source_patch.read_bytes():
                raise ValueError("retained checkpoint belongs to another candidate")
        else:
            # Each fresh invocation retains its own Judge identity and artifacts.
            output = (
                args.output / "evaluations" / run_id
                / (f"attempt-{job['evaluation_attempt']:04d}-{uuid.uuid4().hex}")
            )
            output.mkdir(parents=True)
            patch = output / "candidate.patch"
            shutil.copy2(source_patch, patch)
        expected = hashlib.sha256(patch.read_bytes()).hexdigest()
        job["audit"] = {**job["audit"], "candidate_patch_sha256": expected}
        job["evaluation_output"] = str(output)
        write_json(
            output / (("resume-" + str(time.time_ns()) + ".json") if resume else "attempt.json"),
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
        extension_gaps = []
        gaps = (
            _evaluation_metric_failures(
                PitBenchTask.from_yaml(Path(configured["task_config_path"])),
                patch.parent,
                result.payload,
                extension_failures=extension_gaps,
            )
            if result.completed
            else []
        )
        if result.completed:
            settings = EvaluationConfig.from_yaml(args.config).configuration_robustness
            try:
                configuration_output = run_candidate_configuration(request, settings)
                if configuration_output is not None:
                    extension_gaps.extend(
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
                extension_gaps.append(f"parameter robustness: {type(error).__name__}: {error}")
        job.update(
            **_study_metric_status(gaps, extension_gaps, completed=result.completed),
            evaluation_error=result.error,
            scientific_interpretation="optimization_space_qualification_pending",
        )
    except Exception as error:
        job.update(
            **_study_metric_status([], [], completed=False),
            evaluation_error=f"{type(error).__name__}: {error}",
        )
    job["evaluation_finished_at"] = now()
    write_json(status_path, job)
    write_json(args.output / "evaluation-queue" / (run_id + ".json"), job)
    refresh_study_index(args.output)
    return job


def source_candidate_ready(state):
    """Do not treat a patch from an interrupted Agent as its final candidate."""
    audit = state.get("audit") or {}
    if state.get("status") in {"development_failed", "development_running", "preparing", "queued", "not_started"}:
        return False
    if audit.get("complete") is False:
        return False
    patch = audit.get("candidate_patch")
    return bool(patch and Path(patch).is_file())


def prepare_study_replay(args):
    """Seed a new evaluation queue from retained candidates, without Agent work."""
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with cache_lock(output / "replay-refresh.lock"):
        return _prepare_study_replay_locked(args)


def _prepare_study_replay_locked(args):
    source = args.source_cohort.resolve()
    output = args.output.resolve()
    if source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("saved-patch evaluation needs a separate output directory")
    matrix_path = getattr(args, "target_task_matrix", None)
    mapping = None
    if matrix_path:
        matrix_path = Path(matrix_path)
        matrix_path = matrix_path if matrix_path.is_absolute() else ROOT / matrix_path
        mapping = yaml.safe_load(matrix_path.read_text())
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("target task matrix must map source task IDs to lists")
    existing_replay = (output / "cohort.json").is_file()
    if existing_replay:
        receipt = json.loads((output / "replay.json").read_text())
        if receipt["source_cohort"] != str(source):
            raise ValueError("existing replay belongs to another source cohort")
        if receipt.get("target_task_matrix") != (str(matrix_path.resolve()) if matrix_path else None):
            raise ValueError("existing replay uses a different target task matrix")
    cohort = json.loads((source / "cohort.json").read_text())
    config = EvaluationConfig.from_yaml(args.config)
    source_ids = {job["task_id"] for job in cohort["jobs"]}
    if mapping is not None:
        if set(mapping) != source_ids:
            raise ValueError("target task matrix must cover exactly the source task IDs")
        if any(not isinstance(targets, list) or not targets or
               any(not isinstance(task, str) or not task for task in targets) or
               len(set(targets)) != len(targets) for targets in mapping.values()):
            raise ValueError("each source task requires a nonempty list of unique target task IDs")
    replay_jobs = []
    newly_available = False
    for job in cohort["jobs"]:
        source_run_id, source_task_id = job["run_id"], job["task_id"]
        status_path = source / "agent-status" / source_run_id / "status.json"
        original = json.loads(status_path.read_text()) if status_path.is_file() else {}
        execution = original.get("execution_run_id") or source_run_id
        audit = original.get("audit") or {}
        ready = source_candidate_ready(original)
        for task_id in (mapping[source_task_id] if mapping is not None else [source_task_id]):
            run_id = f"{source_run_id}--{task_id}" if mapping is not None else source_run_id
            replay_job = {**job, "run_id": run_id, "task_id": task_id}
            if mapping is not None:
                replay_job.update(source_run_id=source_run_id, source_task_id=source_task_id)
            replay_jobs.append(replay_job)
            existing_status = output / "agent-status" / run_id / "status.json"
            if existing_replay and existing_status.is_file():
                previous = json.loads(existing_status.read_text())
                if previous.get("status") == "evaluation_pending" and not ready:
                    lock_path = existing_status.parent / "evaluation.lock"
                    with lock_path.open("a") as lock:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                        current = json.loads(existing_status.read_text())
                        if current.get("status") == "evaluation_pending":
                            write_json(existing_status.parent / "evaluation-attempts" / (str(time.time_ns()) + ".json"), current)
                            current.update(status="awaiting_candidate", evaluation_error="source has no captured candidate",
                                           source_agent_status=original.get("status"))
                            write_json(existing_status, current)
                    continue
                missing_source = (
                    previous.get("status") in {"development_failed", "awaiting_candidate"}
                    and previous.get("evaluation_error") == "source has no captured candidate"
                )
                if missing_source and not ready and previous.get("status") == "development_failed":
                    previous["status"] = "awaiting_candidate"
                    write_json(existing_status, previous)
                if not missing_source or not ready:
                    continue
                newly_available = True
            state = {**replay_job, "source_cohort": str(source), "new_agent_starts": 0}
            if not ready:
                state.update(status="awaiting_candidate", evaluation_error="source has no captured candidate",
                             source_agent_status=original.get("status"))
            else:
                task_path = output / "workspaces" / run_id / task_id
                task_path.mkdir(parents=True, exist_ok=True)
                context = yaml.safe_load((source / "workspaces" / execution / source_task_id / "task.yaml").read_text())
                settings = context["evaluator_config"]
                resources = config.resources_for(task_id)
                selected = resources.task_config_path
                if mapping is not None and selected is None:
                    selected = ROOT / "configs/tasks" / (task_id + ".yaml")
                if selected is not None:
                    selected = selected if selected.is_absolute() else ROOT / selected
                    before = PitBenchTask.from_yaml(Path(settings["task_config_path"]))
                    after = PitBenchTask.from_yaml(selected)
                    if after.task_id != task_id or before.release.base_commit != after.release.base_commit or before.repository.plugin != after.repository.plugin:
                        raise ValueError("saved patch and replay task use different solver sources or task IDs")
                    settings["task_config_path"] = str(selected.resolve())
                for key in ("judge_image", "judge_cpus", "judge_memory", "judge_tmpfs_size", "judge_parallel_runs"):
                    value = getattr(resources, key) or getattr(config, key, None)
                    if value is not None:
                        settings[key] = value
                settings["private_root"] = str(config.paths.private_root if config.paths.private_root.is_absolute() else ROOT / config.paths.private_root)
                settings.pop("metric_scenario_config", None)
                if task_id in config.metric_scenario_configs:
                    scenario = config.metric_scenario_configs[task_id]
                    settings["metric_scenario_config"] = str(scenario if scenario.is_absolute() else ROOT / scenario)
                (task_path / "task.yaml").write_text(yaml.safe_dump(context, sort_keys=False))
                result_path = source / "runs" / execution / "results.json"
                result = json.loads(result_path.read_text())
                for trial in result.get("results", []):
                    if mapping is not None and trial.get("task_id") == source_task_id:
                        trial["task_id"] = task_id
                    trial["evaluation"] = None
                write_json(output / "runs" / run_id / "results.json", result)
                state.update(status="evaluation_pending", audit=audit, evaluation_attempt=0)
            write_json(output / "agent-status" / run_id / "status.json", state)
    matrix = str(matrix_path.resolve()) if matrix_path else None
    write_json(output / "replay.json", {"source_cohort": str(source), "target_task_matrix": matrix, "new_agent_starts": 0, "conditions": len(replay_jobs)})
    write_json(output / "cohort.json", {**cohort, "jobs": replay_jobs, "source_cohort": str(source), "target_task_matrix": matrix, "new_agent_starts": 0})
    if newly_available:
        (output / "execution-complete.json").unlink(missing_ok=True)


def run_evaluation_jobs(args, jobs, worker, *, on_complete=None):
    """Share CPU leases and live admission across native and parameter evaluation."""
    if not jobs:
        return []
    if not getattr(args, "dynamic_evaluation_workers", False):
        slots = queue.SimpleQueue()
        for pool in args.evaluation_cpu_pools:
            slots.put(pool)
        launch_lock = threading.Lock()
        dispatch_at = [0.0]

        def run(job):
            cpus = slots.get()
            try:
                with launch_lock:
                    if not wait_for_study_dispatch(args, dispatch_at[0]):
                        return None
                    dispatch_at[0] = time.monotonic() + getattr(args, "dispatch_interval_sec", 0)
                return worker(job, args, cpus)
            finally:
                slots.put(cpus)

        records = []
        with study_cpu_lease(args.evaluation_cpu_pools):
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.evaluation_workers) as executor:
                for future in concurrent.futures.as_completed([executor.submit(run, job) for job in jobs]):
                    result = future.result()
                    if result is not None:
                        records.append(result)
                        if on_complete:
                            on_complete(result)
        return records

    config = EvaluationConfig.from_yaml(args.config)
    cpus, memory = study_judge_resources(config, sorted({job["task_id"] for job in jobs}))
    policy_path = args.output / "resource-policy.json"
    policy = json.loads(policy_path.read_text()) if policy_path.is_file() else {}
    interval = max(0.01, min(float(policy.get("check_every_sec", 10)), 10))
    reserve = policy.get("min_available_memory_gib", 0)
    pending, active, records = list(jobs), {}, []
    cap = getattr(args, "dynamic_evaluation_limit", 0) or len(jobs)
    dispatch_at, inspect_at, peak = 0.0, 0.0, 0
    dispatch_gap = max(
        float(getattr(args, "dispatch_interval_sec", 0)),
        interval if policy.get("max_cpu_temperature_c") is not None else 0.0,
    )

    def publish(reasons=()):
        write_json(parallelism_path(args), {
            "dynamic": True, "pid": os.getpid(), "evaluation_workers": len(active),
            "evaluation_cpu_pools": [value[0] for value in active.values()],
            "peak_evaluation_workers": peak, "pending_jobs": len(pending),
            "worker_limit": getattr(args, "dynamic_evaluation_limit", 0),
            "admission_waits": list(reasons), "updated_at": now(),
        })

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(jobs), cap)) as executor:
        try:
            while pending or active:
                changed = False
                for future in [f for f in active if f.done()]:
                    _, lease = active.pop(future)
                    try:
                        result = future.result()
                        records.append(result)
                        if on_complete:
                            on_complete(result)
                    finally:
                        lease.__exit__(None, None, None)
                    changed = True
                if (args.output / "STOP").exists():
                    pending.clear()
                current = time.monotonic()
                if pending and len(active) < cap and (changed or current >= inspect_at) and current >= dispatch_at:
                    health = host_snapshot()
                    reasons = study_admission_reasons(policy, health)
                    pools = [] if reasons else automatic_study_pools(
                        available_study_sockets(), cpus, memory,
                        # Reserve headroom for this manager's already admitted limits.
                        max(0, health["available_memory_gib"] - len(active) * memory), reserve,
                    )
                    for pool in pools[: min(cap - len(active), len(pending))]:
                        if (args.output / "STOP").exists() or time.monotonic() < dispatch_at:
                            break
                        current_reasons = study_admission_reasons(policy, host_snapshot())
                        if current_reasons:
                            reasons = current_reasons
                            break
                        lease = study_cpu_lease([pool])
                        try:
                            lease.__enter__()
                        except RuntimeError:
                            continue  # A peer manager claimed the same core first.
                        admitted, shared_reasons = claim_campaign_dispatch_slot(
                            args.output, policy
                        )
                        if not admitted:
                            lease.__exit__(None, None, None)
                            reasons = shared_reasons
                            break
                        job = pending.pop(0)
                        try:
                            future = executor.submit(worker, job, args, pool)
                        except BaseException:
                            lease.__exit__(None, None, None)
                            raise
                        active[future] = (pool, lease)
                        peak = max(peak, len(active))
                        dispatch_at = time.monotonic() + dispatch_gap
                    inspect_at = current + interval
                    publish(reasons or ([] if pools else ["no free CPU/memory allocation"]))
                elif changed:
                    publish()
                if active:
                    concurrent.futures.wait(active, timeout=min(interval, 1), return_when=concurrent.futures.FIRST_COMPLETED)
                elif pending:
                    time.sleep(min(interval, 1))
        finally:
            # Running Judges keep their leases until execution actually finishes.
            concurrent.futures.wait(active)
            for _, lease in active.values():
                lease.__exit__(None, None, None)
    publish()
    return records


def parallelism_path(args):
    group = getattr(args, "worker_group", None)
    return args.output / (f"parallelism-{group}.json" if group else "parallelism.json")


def study_admission_reasons(policy, health):
    reasons = []
    if health["available_memory_gib"] < policy.get("min_available_memory_gib", 0):
        reasons.append("memory reserve")
    if health.get("cpu_temperature_c") is not None and health["cpu_temperature_c"] >= policy.get("max_cpu_temperature_c", math.inf):
        reasons.append("CPU temperature")
    if health["load_1min"] >= policy.get("max_load_1min", math.inf):
        reasons.append("host CPU load")
    return reasons


def claim_campaign_dispatch_slot(output, policy):
    """Serialize thermal admission across Agent, Judge, and recovery managers."""
    if policy.get("max_cpu_temperature_c") is None:
        return True, []
    campaign_root = Path(output).resolve().parent
    lock_path = campaign_root / "dispatch-admission.lock"
    state_path = campaign_root / "dispatch-admission.json"
    gap = max(0.0, float(policy.get("check_every_sec", 10)))
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        now_epoch = time.time()
        try:
            state = json.loads(state_path.read_text()) if state_path.is_file() else {}
        except (OSError, ValueError):
            state = {}
        if now_epoch < float(state.get("next_dispatch_after", 0)):
            return False, ["shared thermal dispatch spacing"]
        reasons = study_admission_reasons(policy, host_snapshot())
        if reasons:
            return False, reasons
        write_json(
            state_path,
            {
                "last_dispatch_at": now(),
                "next_dispatch_after": now_epoch + gap,
                "pid": os.getpid(),
            },
        )
        return True, []


def study_evaluations(args):
    if getattr(args, "refresh_queue", False):
        if getattr(args, "source_cohort", None) is None:
            raise ValueError("queue refresh requires a source cohort")
        while not (args.output / "QUEUE_REFRESH_STOP").exists():
            prepare_study_replay(args)
            refresh_study_index(args.output)
            time.sleep(max(1, getattr(args, "follow_interval_sec", 30)))
        return
    if getattr(args, "prepare_only", False):
        if getattr(args, "source_cohort", None) is not None:
            prepare_study_replay(args)
        return
    if not getattr(args, "dynamic_evaluation_workers", False) and (
        args.evaluation_workers < 1
        or len(args.evaluation_cpu_pools) != args.evaluation_workers
    ):
        raise ValueError(
            "candidate evaluation requires one CPU pool per evaluation worker"
        )
    last_report_state = None
    while True:
        if getattr(args, "source_cohort", None) is not None:
            prepare_study_replay(args)
        _study_evaluations(args)
        if not getattr(args, "follow_source", False) or (args.output / "STOP").exists():
            break
        records = refresh_study_index(args.output)
        missing = any(
            row.get("status") in {"development_failed", "awaiting_candidate"}
            and row.get("evaluation_error") == "source has no captured candidate"
            for row in records
        )
        if not missing:
            break
        # Publish useful results so far without claiming the source cohort is complete.
        report_state = tuple(sorted((row["condition_id"], row.get("status"), row.get("evaluation_finished_at")) for row in records))
        if report_state != last_report_state and any(row.get("status") in {"evaluation_completed", "evaluation_incomplete", "evaluation_failed"} for row in records):
            finalize_study_report(args)
            (args.output / "execution-complete.json").unlink(missing_ok=True)
            last_report_state = report_state
        time.sleep(max(1, getattr(args, "follow_interval_sec", 30)))
    # A batch manager also owns report completion while peer managers finish.
    # Per-condition invocations return after their own measurement.
    if not getattr(args, "retry_run_id", None):
        while True:
            records = refresh_study_index(args.output)
            if (args.output / "STOP").exists():
                return
            if not any(row.get("status") == "evaluating" for row in records):
                break
            time.sleep(5)
    finalize_study_report(args)


def replace_study_observation_suite(original, replacement, suite):
    """Replace exactly one complete run grid while retaining all other observations."""
    def identity(item):
        return (item.task_id, item.code_state, item.instance_set, item.instance_id,
                item.solver_seed, item.budget_sec, item.test_suite, item.equivalence_parent_id)

    previous = [row for row in original if row.test_suite == suite]
    new = [row for row in replacement if row.test_suite == suite]
    if not previous or len(previous) != len(new) or len({identity(row) for row in new}) != len(new) or {identity(row) for row in previous} != {identity(row) for row in new}:
        raise ValueError("recollected boundary grid differs from retained observations")
    return [*[row for row in original if row.test_suite != suite], *new]


def recover_study_collected_results(args):
    """Rebuild reports from completed grids whose final aggregation failed."""
    from pitbench.harness.evaluation import EvaluationEnvelope

    output = args.output
    cohort = json.loads((output / "cohort.json").read_text())
    recovered = []
    for job in cohort["jobs"]:
        run_id, task_id = job["run_id"], job["task_id"]
        state_path = output / "agent-status" / run_id / "status.json"
        if not state_path.is_file():
            continue
        state = json.loads(state_path.read_text())
        error = state.get("evaluation_error") or ""
        if state.get("status") != "evaluation_failed" or not (
            "KeyError: 'representation_robustness'" in error
            or "cannot import name 'snapshot_judge_code'" in error
        ):
            continue
        source = Path(state["evaluation_output"])
        task_path = output / "workspaces" / run_id / task_id
        context = yaml.safe_load((task_path / "task.yaml").read_text())["evaluator_config"]
        task = PitBenchTask.from_yaml(Path(context["task_config_path"]))
        observations = ObservationStore.read_jsonl(source / "judge-observations.jsonl")
        if any(row.test_suite == "operational_reliability" and row.error and "unsupported or noncanonical MIP instance" in row.error for row in observations):
            continue
        target = output / "checkpoint-recoveries" / run_id / ("attempt-" + uuid.uuid4().hex)
        request = EvaluationRequest(task_id=task_id, task_path=task_path,
            candidate_patch_path=source / "candidate.patch", output_dir=target,
            agent_name=job["model"]["agent"], model_name=job["model"]["model"], evaluator_config=context)
        try:
            result = PitBenchEvaluator().rebuild_reports(request, source)
            envelope = EvaluationEnvelope(evaluator="pitbench", evaluator_version=PitBenchEvaluator.version,
                                          completed=True, payload=result.model_dump(mode="json"))
            extension_gaps = []
            gaps = _evaluation_metric_failures(task, target, envelope.payload, extension_failures=extension_gaps)
            if EvaluationConfig.from_yaml(args.config).configuration_robustness.enabled:
                extension_gaps.append("parameter robustness: not yet collected")
            write_json(target / "evaluation-envelope.json", envelope.model_dump(mode="json"))
            result_path = output / "runs" / run_id / "results.json"
            results = json.loads(result_path.read_text())
            trials = [row for row in results["results"] if row["task_id"] == task_id]
            if len(trials) != 1:
                raise ValueError("expected exactly one retained Agent trial")
            trials[0]["evaluation"] = envelope.model_dump(mode="json")
            write_json(result_path, results)
            write_json(state_path.parent / "evaluation-attempts" / (str(time.time_ns()) + ".json"), state)
            state.update(**_study_metric_status(gaps, extension_gaps), evaluation_output=str(target),
                         evaluation_error=None,
                         prior_evaluation_output=str(source), recovered_observations=True)
            write_json(state_path, state)
            recovered.append(run_id)
        except Exception as error:
            print(json.dumps({"run_id": run_id, "recovery_error": f"{type(error).__name__}: {error}"}), flush=True)
    refresh_study_index(output)
    return recovered


def repair_study_boundary_observations(args):
    """Rerun only broken boundary cases and reuse complete performance observations."""
    from pitbench.evaluator.docker_judge import DockerJudge
    from pitbench.harness.evaluation import EvaluationEnvelope
    from pitbench.schema.observation import CodeState

    output = args.output
    cohort = json.loads((output / "cohort.json").read_text())
    jobs = []
    for job in cohort["jobs"]:
        path = output / "agent-status" / job["run_id"] / "status.json"
        if not path.is_file():
            continue
        state = json.loads(path.read_text())
        if state.get("status") == "evaluation_failed" and "judge harness code was updated" in (state.get("evaluation_error") or ""):
            source = Path(state["evaluation_output"])
            if (source / "judge-observations.jsonl").is_file():
                jobs.append({**job, **state})

    def repair(job, current_args, cpu_pool):
        run_id, task_id = job["run_id"], job["task_id"]
        root = output / "reliability-repairs" / run_id
        receipt = root / "status.json"
        if receipt.is_file() and json.loads(receipt.read_text()).get("status") == "completed":
            return json.loads(receipt.read_text())
        root.mkdir(parents=True, exist_ok=True)
        task_path = output / "workspaces" / run_id / task_id
        context = yaml.safe_load((task_path / "task.yaml").read_text())["evaluator_config"]
        source = Path(job["evaluation_output"])
        attempt = root / ("attempt-" + uuid.uuid4().hex)
        attempt.mkdir()
        status = {"run_id": run_id, "task_id": task_id, "status": "running", "started_at": now(), "output": str(attempt)}
        write_json(receipt, status)
        try:
            task = PitBenchTask.from_yaml(Path(context["task_config_path"]))
            judge = DockerJudge(
                image=context["judge_image"], task_config_path=Path(context["task_config_path"]),
                base_repository=Path(context["base_repository"]), private_root=Path(context["private_root"]),
                candidate_patch=source / "candidate.patch", output_dir=attempt / "boundary",
                cpus=float(context.get("judge_cpus") or 8), memory=str(context.get("judge_memory") or "8g"),
                tmpfs_size=str(context.get("judge_tmpfs_size") or "4g"), cpuset_cpus=cpu_pool,
                code_states=tuple(CodeState), parallel_runs=int(context.get("judge_parallel_runs") or 1),
                reliability_only=True,
            )
            replacement = judge.run()
            old = ObservationStore.read_jsonl(source / "judge-observations.jsonl")
            new = [row for row in replacement if row.test_suite == "operational_reliability"]
            combined = replace_study_observation_suite(old, replacement, "operational_reliability")
            if any(row.error and "unsupported or noncanonical MIP instance" in row.error for row in new):
                raise ValueError("boundary solver still cannot parse generated instances")
            merged = attempt / "merged-source"
            merged.mkdir()
            for name in ("candidate.patch", "expected-run-grid.json", "shared-base.json"):
                if (source / name).is_file():
                    shutil.copy2(source / name, merged / name)
            for name in ("representation", "runs", "build-logs", "metric-scenarios"):
                if (source / name).is_dir():
                    (merged / name).symlink_to(source / name, target_is_directory=True)
            ObservationStore.write_jsonl(merged / "judge-observations.jsonl", combined)
            report = attempt / "report"
            request = EvaluationRequest(task_id=task_id, task_path=task_path,
                candidate_patch_path=merged / "candidate.patch", output_dir=report,
                agent_name=job["model"]["agent"], model_name=job["model"]["model"], evaluator_config=context)
            rebuilt = PitBenchEvaluator().rebuild_reports(request, merged)
            envelope = EvaluationEnvelope(evaluator="pitbench", evaluator_version=PitBenchEvaluator.version,
                                          completed=True, payload=rebuilt.model_dump(mode="json"))
            extension_gaps = []
            gaps = _evaluation_metric_failures(task, report, envelope.payload, extension_failures=extension_gaps)
            write_json(report / "evaluation-envelope.json", envelope.model_dump(mode="json"))
            results_path = output / "runs" / run_id / "results.json"
            results = json.loads(results_path.read_text())
            trials = [row for row in results["results"] if row["task_id"] == task_id]
            if len(trials) != 1:
                raise ValueError("expected exactly one retained Agent trial")
            trials[0]["evaluation"] = envelope.model_dump(mode="json")
            write_json(results_path, results)
            status_path = output / "agent-status" / run_id / "status.json"
            current = json.loads(status_path.read_text())
            write_json(status_path.parent / "evaluation-attempts" / (str(time.time_ns()) + ".json"), current)
            current.update(**_study_metric_status(gaps, extension_gaps),
                           evaluation_output=str(report), evaluation_error=None,
                           prior_evaluation_output=str(source), boundary_repair_output=str(attempt))
            write_json(status_path, current)
            status.update(status="completed", replaced_boundary_runs=len(new), metric_gaps=[*gaps, *extension_gaps])
        except Exception as error:
            status.update(status="failed", error=f"{type(error).__name__}: {error}")
        status["finished_at"] = now()
        write_json(receipt, status)
        refresh_study_index(output)
        return status

    return run_evaluation_jobs(args, jobs, repair)


def supplement_rebuilt_study_metrics(args):
    """Collect missing scenario and parameter metrics without repeating performance runs."""
    from pitbench.harness.evaluation import EvaluationEnvelope
    from pitbench.schema.observation import CodeState

    output = args.output

    def supplement(job, current_args, cpu_pool):
        run_id, task_id = job["run_id"], job["task_id"]
        directory = output / "metric-supplements" / run_id
        directory.mkdir(parents=True, exist_ok=True)
        receipt = directory / "status.json"
        status = {"run_id": run_id, "task_id": task_id, "status": "running", "started_at": now()}
        write_json(receipt, status)
        try:
            task_path = output / "workspaces" / run_id / task_id
            context = yaml.safe_load((task_path / "task.yaml").read_text())["evaluator_config"]
            context.update(judge_cpuset_cpus=cpu_pool,
                           build_cache_path=str((output / "shared-cache/build").resolve()))
            report = Path(job["evaluation_output"])
            patch = report / "candidate.patch"
            task = PitBenchTask.from_yaml(Path(context["task_config_path"]))
            observations = ObservationStore.read_jsonl(report / "judge-observations.jsonl")
            request = EvaluationRequest(task_id=task_id, task_path=task_path, candidate_patch_path=patch,
                output_dir=report, agent_name=job["model"]["agent"],
                model_name=job["model"]["model"], evaluator_config=context)
            result = PitBenchEvaluator()._aggregate_reports(
                request, task, observations, actual_patch_sha256=hashlib.sha256(patch.read_bytes()).hexdigest(),
                build_cache_path=Path(context["build_cache_path"]), code_states=tuple(CodeState),
            )
            envelope = EvaluationEnvelope(evaluator="pitbench", evaluator_version=PitBenchEvaluator.version,
                                          completed=True, payload=result.model_dump(mode="json"))
            extension_gaps = []
            gaps = _evaluation_metric_failures(task, report, envelope.payload, extension_failures=extension_gaps)
            settings = EvaluationConfig.from_yaml(current_args.config).configuration_robustness
            try:
                configuration_output = run_candidate_configuration(request, settings)
                if configuration_output is not None:
                    extension_gaps.extend("parameter robustness: " + value for value in _configuration_dry_run_failures(configuration_output))
                    attach_configuration_report(envelope.payload, configuration_output, output_root=report)
                    status["configuration_robustness_output"] = str(configuration_output)
            except Exception as error:
                extension_gaps.append(f"parameter robustness: {type(error).__name__}: {error}")
            write_json(report / "evaluation-envelope.json", envelope.model_dump(mode="json"))
            results_path = output / "runs" / run_id / "results.json"
            results = json.loads(results_path.read_text())
            trials = [row for row in results["results"] if row["task_id"] == task_id]
            if len(trials) != 1:
                raise ValueError("expected exactly one retained Agent trial")
            trials[0]["evaluation"] = envelope.model_dump(mode="json")
            write_json(results_path, results)
            state_path = output / "agent-status" / run_id / "status.json"
            current = json.loads(state_path.read_text())
            write_json(state_path.parent / "evaluation-attempts" / (str(time.time_ns()) + ".json"), current)
            current.update(**_study_metric_status(gaps, extension_gaps),
                           evaluation_error=None)
            if status.get("configuration_robustness_output"):
                current["configuration_robustness_output"] = status["configuration_robustness_output"]
            write_json(state_path, current)
            status.update(status="completed", metric_gaps=[*gaps, *extension_gaps])
        except Exception as error:
            status.update(status="failed", error=f"{type(error).__name__}: {error}")
        status["finished_at"] = now()
        write_json(receipt, status)
        refresh_study_index(output)
        return status

    while not (output / "SUPPLEMENT_STOP").exists():
        cohort = json.loads((output / "cohort.json").read_text())
        pending = []
        for job in cohort["jobs"]:
            path = output / "agent-status" / job["run_id"] / "status.json"
            receipt = output / "metric-supplements" / job["run_id"] / "status.json"
            if not path.is_file() or receipt.is_file():
                continue
            state = json.loads(path.read_text())
            if getattr(args, "supplement_recovered_only", False) and not state.get("recovered_observations"):
                continue
            if state.get("status") == "evaluation_incomplete" and (state.get("boundary_repair_output") or state.get("recovered_observations")):
                pending.append({**job, **state})
        run_evaluation_jobs(args, pending, supplement)
        if not getattr(args, "follow_source", False):
            return
        time.sleep(max(1, getattr(args, "follow_interval_sec", 30)))


def refresh_retained_study_metrics(args):
    """Reaggregate completed scenarios after repairing generated-file ownership."""
    from pitbench.evaluator.docker_judge import docker_cli_image
    from pitbench.harness.evaluation import EvaluationEnvelope
    from pitbench.metrics.nuisance_report import report_nuisance_results

    output = args.output.resolve()
    recovered = []
    for path in (output / "agent-status").glob("*/status.json"):
        state = json.loads(path.read_text())
        if state.get("status") != "evaluation_incomplete" or not state.get("evaluation_output"):
            continue
        run_id, task_id = state["run_id"], state["task_id"]
        supplement = output / "metric-supplements" / run_id / "status.json"
        if not supplement.is_file() or json.loads(supplement.read_text()).get("status") != "completed":
            continue
        report = Path(state["evaluation_output"]).resolve()
        envelope_file = report / "evaluation-envelope.json"
        if not report.is_relative_to(output) or not envelope_file.is_file():
            continue
        previous = json.loads(envelope_file.read_text())
        scenario = previous.get("payload", {}).get("summary", {}).get("metric_scenarios") or {}
        if not any("Permission denied" in issue for result in scenario.get("results", []) for issue in result.get("issues", [])):
            continue
        task_path = output / "workspaces" / run_id / task_id
        context = yaml.safe_load((task_path / "task.yaml").read_text())["evaluator_config"]
        task = PitBenchTask.from_yaml(Path(context["task_config_path"]))
        scenario_dir = report / "metric-scenarios"
        if not scenario_dir.resolve().is_relative_to(output):
            continue
        try:
            subprocess.run([
                "docker", "run", "--rm", "--network", "none", "--user", "0",
                "--entrypoint", "chown", "--volume", f"{scenario_dir}:/repair",
                docker_cli_image(context["judge_image"]), "-h", "-R", "-P",
                f"{os.getuid()}:{os.getgid()}", "/repair",
            ], check=True, capture_output=True, text=True)
            representation = report / "representation"
            if representation.is_dir():
                nuisance = report_nuisance_results(representation, representation / "report-refreshed", task=task)
                details = json.loads((representation / "details.json").read_text())
                if "representation_robustness" in nuisance:
                    details["representation_robustness"] = nuisance["representation_robustness"]
                    write_json(representation / "details.json", details)
            patch = report / "candidate.patch"
            request = EvaluationRequest(task_id=task_id, task_path=task_path, candidate_patch_path=patch,
                output_dir=report, agent_name=state["model"]["agent"],
                model_name=state["model"]["model"], evaluator_config=context)
            observations = ObservationStore.read_jsonl(report / "judge-observations.jsonl")
            result = PitBenchEvaluator()._aggregate_reports(
                request, task, observations, actual_patch_sha256=hashlib.sha256(patch.read_bytes()).hexdigest(),
                allow_scenario_execution=False,
            )
            envelope = EvaluationEnvelope(evaluator="pitbench", evaluator_version=PitBenchEvaluator.version,
                                          completed=True, payload=result.model_dump(mode="json"))
            if state.get("configuration_robustness_output"):
                attach_configuration_report(envelope.payload, Path(state["configuration_robustness_output"]), output_root=report)
            extension_gaps = []
            gaps = _evaluation_metric_failures(task, report, envelope.payload, extension_failures=extension_gaps)
            write_json(report / ("evaluation-envelope-before-ownership-repair-" + str(time.time_ns()) + ".json"), previous)
            write_json(envelope_file, envelope.model_dump(mode="json"))
            result_file = output / "runs" / run_id / "results.json"
            results = json.loads(result_file.read_text())
            trials = [item for item in results["results"] if item["task_id"] == task_id]
            if len(trials) != 1:
                raise ValueError("expected exactly one retained Agent trial")
            trials[0]["evaluation"] = envelope.model_dump(mode="json")
            write_json(result_file, results)
            state.update(**_study_metric_status(gaps, extension_gaps), permission_report_rebuilt=True)
            write_json(path, state)
            recovered.append(run_id)
        except Exception as error:
            print(json.dumps({"run_id": run_id, "refresh_error": f"{type(error).__name__}: {error}"}), flush=True)
    refresh_study_index(output)
    return recovered


def watch_study_primary_results(args):
    """Publish complete primary grids while slower extra metrics continue."""
    from pitbench.metrics.performance_report import compute_task_performance_report
    from pitbench.schema.observation import ExpectedRunGrid

    output = args.output
    destination = output / "early-performance"
    destination.mkdir(parents=True, exist_ok=True)
    while not (output / "EARLY_REPORT_STOP").exists():
        for status_path in (output / "agent-status").glob("*/status.json"):
            state = json.loads(status_path.read_text())
            source = state.get("evaluation_output")
            if not source or state.get("status") not in {"evaluating", "evaluation_completed", "evaluation_incomplete", "evaluation_failed"}:
                continue
            source = Path(source)
            grid_path, observations_path = source / "expected-run-grid.json", source / "judge-observations.jsonl"
            if not grid_path.is_file() or not observations_path.is_file():
                continue
            receipt = destination / (state["run_id"] + ".json")
            if receipt.is_file() and json.loads(receipt.read_text()).get("measurement_source") == str(source):
                continue
            try:
                grid = ExpectedRunGrid.model_validate_json(grid_path.read_text())
                observations = ObservationStore.read_jsonl(observations_path)
                def identity(row):
                    return (row.task_id, row.code_state, row.instance_set, row.instance_id,
                            row.solver_seed, row.budget_sec, row.test_suite, row.equivalence_parent_id)
                planned = {identity(row) for row in grid.runs}
                observed = {identity(row) for row in observations}
                if len(planned) != len(grid.runs) or len(observed) != len(observations) or planned != observed:
                    continue
                task_path = output / "workspaces" / state["run_id"] / state["task_id"] / "task.yaml"
                context = yaml.safe_load(task_path.read_text())["evaluator_config"]
                task = PitBenchTask.from_yaml(Path(context["task_config_path"]))
                report = compute_task_performance_report(
                    [row for row in observations if row.equivalence_parent_id is None and row.test_suite is None],
                    task=task, expected_run_grid=grid,
                )
                write_json(receipt, {"run_id": state["run_id"], "task_id": state["task_id"],
                    "model": state.get("model"), "measurement_source": str(source),
                    "provisional": True, "other_metrics_pending": True,
                    "observation_count": len(observations),
                    "performance": report.model_dump(mode="json") if report else None})
                print(json.dumps({"run_id": state["run_id"], "primary_result": str(receipt)}), flush=True)
            except (OSError, ValueError, KeyError, TypeError) as error:
                print(json.dumps({"run_id": state["run_id"], "primary_result_error": f"{type(error).__name__}: {error}"}), flush=True)
        if not getattr(args, "follow_source", False):
            return
        time.sleep(max(5, getattr(args, "follow_interval_sec", 30)))


def rebalance_study_agents(args):
    """Resume evaluation after our temporary priority pause has served its purpose."""
    output = args.output
    marker = output / "agent-priority-pause.json"
    if args.source_cohort is None or not marker.is_file():
        raise ValueError("agent-priority balancing requires its recorded pause and source cohort")
    status_path = getattr(args, "recovery_status", None) or args.source_cohort / "recovery-status.json"
    while marker.is_file():
        if status_path.is_file() and time.time() - status_path.stat().st_mtime < 60:
            status = json.loads(status_path.read_text())
            if not status.get("eligible_waiting") and not status.get("recovery_errors"):
                (output / "STOP").unlink(missing_ok=True)
                marker.unlink(missing_ok=True)
                command = [sys.executable, "-m", "scripts.run_benchmark_release",
                    "--profile", str(args.profile), "--phase", "study-evaluations",
                    "--source-cohort", str(args.source_cohort), "--output", str(output),
                    "--follow-source", "--evaluation-workers", "0", "--worker-group", "follow-balanced", "--background"]
                launched = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
                if launched.returncode:
                    raise RuntimeError(launched.stderr.strip() or "could not resume the evaluation manager")
                print(json.dumps({"status": "evaluation_dispatch_resumed", "launcher": launched.stdout.strip()}), flush=True)
                return
        time.sleep(5)


def resume_thermal_study_evaluations(args):
    """Resume only recorded thermal interruptions on their original CPU pools."""
    import copy

    output = args.output
    active = {}
    completed = {}
    next_dispatch_at = 0.0
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        while True:
            for future in [item for item in active if item.done()]:
                run_id, lease, interruption_at = active.pop(future)
                try:
                    result = future.result()
                    write_json(output / "agent-status" / run_id / "thermal-resume.json", {
                        "finished_at": now(), "status": result.get("status"),
                        "evaluation_output": result.get("evaluation_output"),
                        "error": result.get("evaluation_error"),
                        "interruption_at": interruption_at,
                    })
                except Exception as error:
                    write_json(output / "agent-status" / run_id / "thermal-resume.json", {
                        "finished_at": now(), "status": "failed", "error": f"{type(error).__name__}: {error}",
                        "interruption_at": interruption_at,
                    })
                finally:
                    lease.__exit__(None, None, None)
                    completed[run_id] = interruption_at
            pending = []
            for marker in (output / "agent-status").glob("*/thermal-interruption.json"):
                run_id = marker.parent.name
                interruption = json.loads(marker.read_text())
                interruption_at = interruption.get("recorded_at")
                if completed.get(run_id) != interruption_at:
                    completed.pop(run_id, None)
                if run_id in completed or any(item[0] == run_id for item in active.values()):
                    continue
                receipt = marker.parent / "thermal-resume.json"
                if receipt.is_file() and json.loads(receipt.read_text()).get("interruption_at") == interruption_at:
                    completed[run_id] = interruption_at
                    continue
                state = json.loads((marker.parent / "status.json").read_text())
                if state.get("status") == "evaluation_failed":
                    state["thermal_interruption_at"] = interruption_at
                    pending.append(state)
            if not pending and not active:
                if not getattr(args, "follow_source", False):
                    return
                time.sleep(10)
                continue
            policy_path = output / "resource-policy.json"
            policy = json.loads(policy_path.read_text()) if policy_path.is_file() else {}
            dispatch_gap = max(
                float(getattr(args, "recovery_interval_sec", 2)),
                float(policy.get("check_every_sec", 10)),
            )
            if pending and time.monotonic() >= next_dispatch_at and not (output / "STOP").exists():
                for state in pending:
                    if study_admission_reasons(policy, host_snapshot()):
                        break
                    pool = state.get("evaluation_cpu_pool")
                    if not pool:
                        completed[state["run_id"]] = state.get("thermal_interruption_at")
                        continue
                    lease = study_cpu_lease([pool])
                    try:
                        lease.__enter__()
                    except RuntimeError:
                        continue
                    admitted, _ = claim_campaign_dispatch_slot(output, policy)
                    if not admitted:
                        lease.__exit__(None, None, None)
                        break
                    child = copy.copy(args)
                    child.resume_evaluation = True
                    try:
                        future = executor.submit(evaluate_study_candidate, state, child, pool)
                    except BaseException:
                        lease.__exit__(None, None, None)
                        raise
                    active[future] = (state["run_id"], lease, state.get("thermal_interruption_at"))
                    next_dispatch_at = time.monotonic() + dispatch_gap
                    break
            write_json(output / "thermal-recovery-status.json", {
                "updated_at": now(), "running": [item[0] for item in active.values()],
                "waiting": [state["run_id"] for state in pending if state["run_id"] not in {item[0] for item in active.values()}],
            })
            time.sleep(2)


def _study_evaluations(args):
    """Evaluate retained final patches without allocating another Agent start."""
    output = args.output
    cohort = json.loads((output / "cohort.json").read_text())
    pending = []
    retry_statuses = getattr(args, "retry_evaluation_statuses", None)
    boot_time = next(
        (int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines()
         if line.startswith("btime ")),
        None,
    ) if not retry_statuses else None
    run_id = getattr(args, "retry_run_id", None)
    jobs = [job for job in cohort["jobs"] if run_id is None or job["run_id"] == run_id]
    if run_id is not None and not jobs:
        raise ValueError(f"study run ID is absent from the cohort: {run_id}")
    for job in jobs:
        status_path = output / "agent-status" / job["run_id"] / "status.json"
        if not status_path.is_file():
            continue
        state = json.loads(status_path.read_text())
        retryable_statuses = set(retry_statuses or ["evaluation_pending"])
        started_at = state.get("evaluation_started_at")
        interrupted = (
            boot_time is not None and state.get("status") == "evaluating"
            and started_at and datetime.fromisoformat(started_at).timestamp() < boot_time
        )
        if state.get("status") not in retryable_statuses and not interrupted:
            continue
        if interrupted:
            state["interrupted_by_reboot"] = True
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

    run_evaluation_jobs(args, pending, evaluate_study_candidate)
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


def finalize_study_report(args, views=None):
    """The last evaluator writes the final report, including across managers."""
    terminal = {
        "evaluation_completed",
        "evaluation_incomplete",
        "evaluation_failed",
        "development_failed",
        "awaiting_candidate",
    }
    with cache_lock(args.output / "study-report.lock"):
        records = refresh_study_index(args.output)
        if not records or any(row.get("status") not in terminal for row in records):
            return False
        if views is None:
            candidates = [
                args.output / "evaluation-views/task-views.json",
                args.output / "frozen-seed-views/task-views.json",
                Path(args.study_views),
            ]
            views = json.loads(
                next(path for path in candidates if path.is_file()).read_text()
            )
        write_campaign_report(
            args.output,
            args.output / "report",
            repository_root=ROOT,
            index=args.output / "model-runs.json",
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
                "conditions": len(records),
                "statuses": dict(Counter(row["status"] for row in records)),
                "scientific_interpretation": "optimization_space_qualification_pending",
            },
        )
    return True


def reusable_agent_pools(args, config, task_ids, agent_pools):
    if not getattr(args, "reuse_agent_pools", True):
        return []
    cpus, memory = study_judge_resources(config, task_ids)
    candidates = [
        pool for pool in agent_pools if pool and len(study_cpu_ids(pool)) >= cpus
    ]
    policy_path = args.output / "resource-policy.json"
    policy = json.loads(policy_path.read_text()) if policy_path.is_file() else {}
    maximum = int(
        max(
            0,
            host_snapshot()["available_memory_gib"]
            - policy.get("min_available_memory_gib", 0),
        )
        // memory
    )
    return candidates[: max(0, maximum - len(args.evaluation_cpu_pools))]


def run_study_jobs(jobs, one, args, agent_pools, config, task_ids):
    """Recycle the Agent CPU allocation into the evaluation queue after coding."""
    evaluate = getattr(args, "evaluate_candidates", False)
    dedicated = getattr(args, "evaluation_cpu_pools", [])
    slots = queue.SimpleQueue()
    for pool in dedicated:
        slots.put(pool)
    dispatch_lock = threading.Lock()
    next_dispatch = [0.0]

    def collect(state):
        pool = slots.get()
        try:
            with dispatch_lock:
                if not wait_for_study_dispatch(args, next_dispatch[0]):
                    return state
                next_dispatch[0] = time.monotonic() + getattr(
                    args, "dispatch_interval_sec", 0
                )
            return evaluate_study_candidate(state, args, pool)
        finally:
            slots.put(pool)

    maximum = len(dedicated) + (
        len(agent_pools) if getattr(args, "reuse_agent_pools", True) else 0
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, maximum)) as judges:
        measured = []
        deferred = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(args.workers, len(agent_pools))
        ) as agents:
            for future in concurrent.futures.as_completed(
                [agents.submit(one, job) for job in jobs]
            ):
                state = future.result()
                if evaluate and state["status"] == "evaluation_pending":
                    if dedicated:
                        measured.append(judges.submit(collect, state))
                    else:
                        deferred.append(state)
        if evaluate:
            recycled = reusable_agent_pools(args, config, task_ids, agent_pools)
            for pool in recycled:
                slots.put(pool)
            if recycled:
                measured.extend(judges.submit(collect, state) for state in deferred)
            elif deferred:
                print(
                    "No evaluation allocation fits current memory; captured candidates remain pending",
                    flush=True,
                )
            write_json(
                args.output / "evaluation-parallelism.json",
                {
                    "agent_phase_finished_at": now(),
                    "dedicated_cpu_pools": dedicated,
                    "reused_agent_cpu_pools": recycled,
                    "evaluation_workers": len(dedicated) + len(recycled),
                },
            )
        for future in concurrent.futures.as_completed(measured):
            future.result()


def retry_unexecuted_condition(args):
    """Retry an incomplete condition after a model disconnect or hard timeout."""
    plan = read_study_plan(args.study_plan)
    view_dir = (
        "evaluation-views"
        if plan.evaluation_run_target is not None
        else "frozen-seed-views"
    )
    views_path = args.output / view_dir / "task-views.json"
    if not views_path.is_file():
        views_path = Path(args.study_views)
    views = json.loads(views_path.read_text())
    cohort = json.loads((args.output / "cohort.json").read_text())
    matches = [row for row in cohort["jobs"] if row["run_id"] == args.retry_run_id]
    if len(matches) != 1:
        raise ValueError("retry run ID is absent from the frozen cohort")
    job = matches[0]
    retry_root = args.output / "infrastructure-retries" / job["run_id"]
    recovery_lock = acquire_condition_recovery_lock(retry_root)
    status_path = args.output / "agent-status" / job["run_id"] / "status.json"
    original = json.loads(status_path.read_text())
    original_run = args.output / "runs" / job["run_id"]
    audit = audit_model_run(original_run, job["task_id"], evaluation_required=False)
    prior_runs = [(original_run, audit)]
    for prior in sorted(retry_root.glob("infra*/retry.json"), key=lambda path: int(path.parent.name[5:])):
        previous = json.loads(prior.read_text())
        if previous.get("status") == "retry_failed":
            previous_run = args.output / "runs" / previous["retry_run_id"]
            prior_audit = audit_model_run(previous_run, job["task_id"], evaluation_required=False)
            prior_runs.append((previous_run, prior_audit))
            if prior_audit.get("reason") in {"antigravity quota reached", "antigravity interrupted after quota wait"}:
                audit = prior_audit
    prior_continuation = next((item for _, item in reversed(prior_runs)
                               if item.get("continuation_required") and item.get("development_storage")), None)
    if audit.get("reason") in {"antigravity quota reached", "antigravity interrupted after quota wait"} and audit.get("retryable") and prior_continuation:
        audit = {**audit, "retryable": False, "continuation_required": True,
                 "development_storage": prior_continuation["development_storage"]}
    retry_reason = incomplete_condition_retry_reason(original.get("status"), audit)
    if retry_reason is None and audit.get("reason") not in {"antigravity quota reached", "antigravity interrupted after quota wait"}:
        retry_reason = preflight_retry_reason(
            original.get("status"), audit, status_path.parent / "controller.log"
        )
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
    continuation = retry_reason == "quota_workspace_continuation"
    budget = plan.research_budget
    extension_path = status_path.parent / "budget-extension.json"
    extension = json.loads(extension_path.read_text()) if extension_path.is_file() else None
    original_budget_sec = float(budget.wall_time_sec)
    additional_budget_sec = 0.0
    if extension and extension.get("run_id") == job["run_id"]:
        additional_budget_sec = max(
            0.0, float(extension.get("additional_active_time_sec") or 0)
        )
    if continuation:
        consumed = sum(float(item.get("agent_duration_sec") or 0) for _, item in prior_runs)
        total_authorized = original_budget_sec + additional_budget_sec
        remaining = total_authorized - consumed
        if remaining <= 0:
            raise ValueError("interrupted Agent has exhausted its authorized active-time budget")
        budget = budget.model_copy(update={"wall_time_sec": remaining})
    config.paths.output_path = run_root.resolve()
    config.paths.workspace_path = workspace_root.resolve()
    config.research_budget = budget
    config.research_track = None
    config.candidate_hypotheses_required = job["hypotheses_required"]
    config.agent_tools = [AgentTool(tool) for tool in plan.shared_agent_tools]
    resources = config.resources_for(job["task_id"]).model_copy(deep=True)
    arm = views["arms"][job["arm"]][job["task_id"]]
    resources.task_config_path = Path(arm["task_config_path"])
    resources.agent_task_config_path = Path(arm["agent_task_config_path"])
    resources.research_budget = budget
    resources.judge_cpuset_cpus = args.cpu_pool
    config.tasks[job["task_id"]] = resources
    config.agent_cpuset_cpus = args.cpu_pool
    configure_study_scenarios(config, job["task_id"], views)
    configure_study_agent(config, job, budget)
    child_env = None
    context_events = 0
    if continuation:
        source_run, source_audit = next(
            (run, item) for run, item in reversed(prior_runs)
            if item.get("continuation_required") and item.get("development_storage")
        )
        context_path = receipt_dir / "continuation-context.jsonl"
        for previous_run, previous_audit in prior_runs:
            if previous_audit.get("continuation_required"):
                context_events += write_continuation_context(previous_run, context_path, append=context_path.exists())
        if not context_events:
            raise ValueError("interrupted Agent has no recorded tool history")
        selected = job.get("agent_import_path") or job["model"]["agent"]
        config.agents[selected]["continuation_context_available"] = True
        child_env = {**os.environ, "PITBENCH_CONTINUATION_STORAGE": source_audit["development_storage"],
                     "PITBENCH_CONTINUATION_CONTEXT": str(context_path)}
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
            "budget": budget.model_dump(mode="json"),
            "budget_extension": (
                {
                    "record_path": str(extension_path),
                    "original_active_time_sec": original_budget_sec,
                    "additional_active_time_sec": additional_budget_sec,
                    "authorized_total_active_time_sec": original_budget_sec + additional_budget_sec,
                    "active_time_used_before_retry_sec": sum(
                        float(item.get("agent_duration_sec") or 0)
                        for _, item in prior_runs
                    ),
                }
                if additional_budget_sec
                else None
            ),
            "continuation_source": str(source_run) if continuation else None,
            "continuation_context_events": context_events,
        },
    )
    with (attempt / "controller.log").open("w") as log:
        code = subprocess.run(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=child_env,
            check=False,
        ).returncode
    recovered = audit_model_run(
        run_root / retry_id, job["task_id"], evaluation_required=False
    )
    if continuation:
        recovered["prior_tokens"] = sum(int(item.get("tokens") or 0) for _, item in prior_runs)
        recovered["continuation_tokens"] = int(recovered.get("tokens") or 0)
        recovered["tokens"] = recovered["prior_tokens"] + recovered["continuation_tokens"]
        recovered["continuation_source"] = str(source_run)
        recovered["continuation_kind"] = "restored_workspace_and_recorded_tool_history"
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
    elif recovered.get("reason") in {"antigravity quota reached", "antigravity interrupted after quota wait"}:
        original.update(recovery_audit=recovered, quota_retry_after=recovered.get("quota_retry_after"))
        write_json(status_path, original)
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


def recover_pending_conditions(args):
    """Keep unstarted recoveries queued and respect per-model quota reset times."""
    import copy

    plan = read_study_plan(args.study_plan)
    output = args.output
    group = getattr(args, "worker_group", None)
    status_name = f"recovery-status-{group}.json" if group and group != "auto" else "recovery-status.json"
    previous_status = output / status_name
    previous = json.loads(previous_status.read_text()) if previous_status.is_file() else {}
    active = {}
    recovery_errors = dict(previous.get("recovery_errors") or {})
    for run_id in list(recovery_errors):
        extension_path = output / "agent-status" / run_id / "budget-extension.json"
        if not extension_path.is_file():
            continue
        try:
            extension = json.loads(extension_path.read_text())
        except (OSError, ValueError):
            continue
        if (
            extension.get("run_id") == run_id
            and float(extension.get("additional_active_time_sec") or 0) > 0
        ):
            recovery_errors.pop(run_id, None)
    last_snapshot = None
    next_dispatch_at = 0.0
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        while not (output / "RECOVERY_STOP").exists():
            for future in [item for item in active if item.done()]:
                run_id, lease = active.pop(future)
                try:
                    future.result()
                except Exception as error:
                    recovery_errors[run_id] = f"{type(error).__name__}: {error}"
                    print(json.dumps({"run_id": run_id, "recovery_error": str(error)}), flush=True)
                finally:
                    lease.__exit__(None, None, None)
            rows = refresh_study_index(output)
            model_holds = {}
            audits = {}
            for row in rows:
                if row.get("status") != "development_failed":
                    continue
                run_id = row["run_id"]
                original = audit_model_run(output / "runs" / run_id, row["task_id"], evaluation_required=False)
                retry_root = output / "infrastructure-retries" / run_id
                receipts = sorted(retry_root.glob("infra*/retry.json"), key=lambda p: int(p.parent.name[5:]))
                recent = json.loads(receipts[-1].read_text()) if receipts else None
                if recent and recent.get("status") == "retry_failed":
                    recovered = audit_model_run(output / "runs" / recent["retry_run_id"], row["task_id"], evaluation_required=False)
                else:
                    recovered = None
                audits[run_id] = (original, recent, recovered)
                for audit in (original, recovered):
                    if audit and audit.get("quota_retry_after"):
                        hold = datetime.fromisoformat(audit["quota_retry_after"])
                        model = row["model"]
                        model_holds[model] = max(model_holds.get(model, hold), hold)
            pending = []
            now_utc = datetime.now(timezone.utc)
            active_ids = {run_id for run_id, _ in active.values()}
            for row in rows:
                run_id = row["run_id"]
                if row.get("status") != "development_failed" or run_id in active_ids or run_id in recovery_errors:
                    continue
                audit, recent, recovered = audits[run_id]
                if getattr(args, "continuations_only", False) and not (
                    audit.get("continuation_required")
                    or (recovered and recovered.get("continuation_required"))
                ):
                    continue
                if model_holds.get(row["model"], now_utc) > now_utc:
                    continue
                if recent and recent.get("status") == "retry_running":
                    continue
                if recent and recent.get("status") == "retry_failed":
                    if not recovered:
                        continue
                    if recovered.get("reason") in {"antigravity quota reached", "antigravity interrupted after quota wait"}:
                        if recovered.get("retryable") and audit.get("continuation_required") and audit.get("development_storage"):
                            recovered = {**recovered, "retryable": False, "continuation_required": True,
                                         "development_storage": audit["development_storage"]}
                    elif incomplete_condition_retry_reason(row.get("status"), recovered) is None:
                        continue
                    audit = recovered
                reason = incomplete_condition_retry_reason(row.get("status"), audit)
                if reason is None and audit.get("reason") not in {"antigravity quota reached", "antigravity interrupted after quota wait"}:
                    reason = preflight_retry_reason(row.get("status"), audit, output / "agent-status" / run_id / "controller.log")
                if reason:
                    pending.append(row)
            health = host_snapshot()
            policy_file = output / "resource-policy.json"
            policy = json.loads(policy_file.read_text()) if policy_file.is_file() else {}
            reasons = study_admission_reasons(policy, health)
            pools = [] if reasons else automatic_study_pools(
                available_study_sockets(), plan.research_budget.cpus,
                plan.research_budget.memory_gib, health["available_memory_gib"],
                policy.get("min_available_memory_gib", 0),
            )
            dispatch_gap = max(
                float(getattr(args, "recovery_interval_sec", 2)),
                float(policy.get("check_every_sec", 10)),
            )
            if pending and time.monotonic() >= next_dispatch_at:
                for row, pool in zip(pending, pools):
                    current_health = host_snapshot()
                    if study_admission_reasons(policy, current_health):
                        break
                    lease = study_cpu_lease([pool])
                    try:
                        lease.__enter__()
                    except RuntimeError:
                        continue
                    admitted, _ = claim_campaign_dispatch_slot(output, policy)
                    if not admitted:
                        lease.__exit__(None, None, None)
                        break
                    child = copy.copy(args)
                    child.retry_run_id, child.cpu_pool = row["run_id"], pool
                    try:
                        future = executor.submit(retry_unexecuted_condition, child)
                    except BaseException:
                        lease.__exit__(None, None, None)
                        raise
                    active[future] = (row["run_id"], lease)
                    next_dispatch_at = time.monotonic() + dispatch_gap
                    break
            snapshot = {
                "updated_at": now(), "running": [run_id for run_id, _ in active.values()],
                "eligible_waiting": max(0, len(pending) - len(pools)),
                "quota_resets": {key: value.isoformat() for key, value in model_holds.items() if value > now_utc},
                "continuation_required": [
                    run_id for run_id, (audit, _, recovered) in audits.items()
                    if audit.get("continuation_required") or (recovered and recovered.get("continuation_required"))
                ],
                "recovery_errors": recovery_errors,
                "agent_statuses": dict(Counter(row["status"] for row in rows)),
            }
            write_json(output / status_name, snapshot)
            summary = {key: value for key, value in snapshot.items() if key != "updated_at"}
            if summary != last_snapshot:
                print(json.dumps(summary), flush=True)
                last_snapshot = summary
            if rows and all(row.get("status") in {"evaluation_pending", "evaluation_completed", "evaluation_incomplete", "evaluation_failed"} for row in rows) and not active:
                break
            time.sleep(max(0.5, getattr(args, "recovery_interval_sec", 2)))


def adopt_running_study_agents(args):
    """Take over already-running Agents after their dispatcher exits."""
    output = args.output
    cohort = json.loads((output / "cohort.json").read_text())
    jobs = {job["run_id"]: job for job in cohort["jobs"]}
    active = {}
    try:
        for run_id, job in jobs.items():
            path = output / "agent-status" / run_id / "status.json"
            if not path.is_file():
                continue
            state = json.loads(path.read_text())
            if state.get("status") != "development_running":
                continue
            pid, pool = state.get("pid"), state.get("cpu_pool")
            if not pid or not pool:
                raise ValueError(f"running condition {run_id} lacks process or CPU allocation")
            cmdline = Path(f"/proc/{pid}/cmdline")
            if cmdline.is_file() and run_id not in cmdline.read_bytes().decode(errors="replace"):
                raise ValueError(f"running condition {run_id} no longer owns its recorded process")
            lease = study_cpu_lease([pool])
            lease.__enter__()
            active[run_id] = (state, lease)
        write_json(output / "adoption-ready.json", {
            "started_at": now(), "manager_pid": os.getpid(),
            "conditions": {run_id: {"pid": state["pid"], "cpu_pool": state["cpu_pool"]}
                           for run_id, (state, _) in active.items()},
        })
        while active:
            for run_id, (state, lease) in list(active.items()):
                pid = state["pid"]
                proc = Path(f"/proc/{pid}")
                stat = proc / "stat"
                try:
                    if stat.read_text().split()[2] != "Z" and run_id in (proc / "cmdline").read_bytes().decode(errors="replace"):
                        continue
                except FileNotFoundError:
                    pass
                path = output / "agent-status" / run_id / "status.json"
                current = json.loads(path.read_text())
                if current.get("status") == "development_running" and current.get("pid") == pid:
                    job = jobs[run_id]
                    audit = audit_model_run(output / "runs" / run_id, job["task_id"], evaluation_required=False)
                    current.update(status="evaluation_pending" if audit["complete"] else "development_failed",
                                   audit=audit, finished_at=now(), adopted_dispatcher=True)
                    write_json(path, current)
                    if audit["complete"]:
                        write_json(output / "evaluation-queue" / (run_id + ".json"), {
                            **job, **audit, "status": "awaiting_scientific_gate",
                        })
                    refresh_study_index(output)
                    print(json.dumps({"run_id": run_id, "status": current["status"]}), flush=True)
                lease.__exit__(None, None, None)
                del active[run_id]
            if active:
                time.sleep(2)
        write_json(output / "adoption-complete.json", {"finished_at": now()})
    finally:
        for _, lease in active.values():
            lease.__exit__(None, None, None)


def study_models(args):
    """Run a frozen development cohort; queue hidden evaluation separately."""
    import itertools
    import random

    plan = read_study_plan(args.study_plan)
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
    source_configs = {Path(name).stem: ROOT / name for name in plan.task_configs}
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
        original = TaskCatalog(ROOT).validate_file(source_configs[task_id]).task
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
                    previous = existing or {**job, "status": "queued"}
                    process = launch_study_process(
                        command, args, path, log, next_launch[0], previous
                    )
                    if process is None:
                        return previous
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

    run_study_jobs(jobs, one, args, pools, config_template, task_ids)
    if getattr(args, "evaluate_candidates", False):
        finalize_study_report(args, views)


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
    jobs = [
        job
        for job in cohort["jobs"]
        if not args.task_id or job["task_id"] == args.task_id
    ]
    write_json(
        args.output / "supplementation.json",
        {"source_cohort": str(source), "jobs": jobs, "new_agent_starts": 0},
    )

    def run(job, _args, pool):
        output = args.output / job["run_id"]
        output.mkdir(parents=True, exist_ok=True)
        status = {
            **job,
            "started_at": now(),
            "status": "preparing",
            "new_agent_starts": 0,
        }
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
        return status

    records = []

    def completed(result):
        records.append(result)
        write_json(args.output / "parameter-results.json", records)

    run_evaluation_jobs(args, jobs, run, on_complete=completed)
    if any(record["status"] != "completed" for record in records):
        raise SystemExit(1)


def rebuild_study_condition_reports(job, source, target, parameter_root):
    from pitbench.harness.evaluation import EvaluationEnvelope
    from pitbench.metrics.configuration_report import write_configuration_report

    run_id, task_id = job["run_id"], job["task_id"]
    original = json.loads((source / "agent-status" / run_id / "status.json").read_text())
    status = {**original, "source_evaluation_status": original["status"],
              "source_evaluation_error": original.get("evaluation_error"),
              "source_evaluation_output": original.get("evaluation_output"),
              "new_agent_starts": 0, "new_solver_runs": 0, "report_started_at": now()}
    output = target / "evaluations" / run_id
    try:
        execution = original.get("execution_run_id") or run_id
        task_manifest = source / "workspaces" / execution / task_id / "task.yaml"
        context = yaml.safe_load(task_manifest.read_text())["evaluator_config"]
        retained_manifest = target / "workspaces" / execution / task_id / "task.yaml"
        retained_manifest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(task_manifest, retained_manifest)
        measured = Path(original["evaluation_output"])
        request = EvaluationRequest(task_id=task_id, task_path=source / "workspaces" / execution / task_id,
            candidate_patch_path=measured / "candidate.patch", output_dir=output,
            agent_name=job.get("agent_import_path") or job["model"]["agent"],
            model_name=job["model"]["model"], evaluator_config=context)
        result = PitBenchEvaluator().rebuild_reports(request, measured)
        envelope = EvaluationEnvelope(evaluator="pitbench", evaluator_version=PitBenchEvaluator.version,
            completed=True, payload=result.model_dump(mode="json"))
        extension_gaps = []
        gaps = _evaluation_metric_failures(PitBenchTask.from_yaml(Path(context["task_config_path"])), output, envelope.payload, extension_failures=extension_gaps)
        parameters = parameter_root / run_id / "configuration-robustness" if parameter_root else None
        if parameters and (parameters / "experiment.json").is_file():
            destination = output / "configuration-robustness"
            write_configuration_report(parameters, destination=destination)
            attach_configuration_report(envelope.payload, destination, output_root=output)
            extension_gaps.extend(_configuration_dry_run_failures(destination))
            status["configuration_robustness_output"] = str(destination)
        write_json(output / "evaluation-envelope.json", envelope.model_dump(mode="json"))
        harness = json.loads((source / "runs" / execution / "results.json").read_text())
        trials = [trial for trial in harness["results"] if trial["task_id"] == task_id]
        if len(trials) != 1:
            raise ValueError("expected exactly one retained Agent trial")
        trials[0]["evaluation"] = envelope.model_dump(mode="json")
        write_json(target / "runs" / execution / "results.json", harness)
        status.update(**_study_metric_status(gaps, extension_gaps),
            evaluation_output=str(output), evaluation_error=None)
    except Exception as error:
        status.update(**_study_metric_status([], [], completed=False), evaluation_error=f"{type(error).__name__}: {error}")
    status["report_finished_at"] = now()
    write_json(target / "agent-status" / run_id / "status.json", status)
    return status


def rebuild_study_reports(args):
    """Create a new reporting cohort; never modify retained measurement sources."""
    source = args.output.resolve()
    target = (args.report_output or source.parent / ("recovered-reports-" + datetime.now().strftime("%Y%m%d-%H%M%S"))).resolve()
    if source.is_relative_to(target) or target.is_relative_to(source):
        raise ValueError("recovered reports need a separate output directory")
    target.mkdir(parents=True, exist_ok=False)
    cohort = json.loads((source / "cohort.json").read_text())
    write_json(target / "cohort.json", {**cohort, "measurement_source_cohort": str(source), "new_agent_starts": 0, "new_solver_runs": 0})
    write_json(target / "replay.json", {
        "source_cohort": str(args.source_cohort.resolve()) if args.source_cohort else str(source),
        "measurement_source_cohort": str(source), "new_agent_starts": 0,
    })
    parameter_root = None
    plan_path = source.parent / "metric-plan.json"
    if plan_path.is_file():
        value = json.loads(plan_path.read_text()).get("parameter_supplementation")
        parameter_root = Path(value) if value else None

    workers = min(len(cohort["jobs"]), max(1, len(os.sched_getaffinity(0))))
    numeric_threads = {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    try:
        # Each report is CPU-bound; spawn independent interpreters with one BLAS
        # thread each. This affects aggregation only, never measured solver work.
        os.environ.update({key: "1" for key in numeric_threads})
        with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, workers), mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = [executor.submit(rebuild_study_condition_reports, job, source, target, parameter_root) for job in cohort["jobs"]]
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                record = future.result()
                print(f"Reports {index}/{len(futures)}: {record['run_id']} {record['status']}", flush=True)
                refresh_study_index(target)
    finally:
        for key, value in numeric_threads.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    report_args = argparse.Namespace(**vars(args))
    report_args.output = target
    finalize_study_report(report_args)
    write_json(target / "report-recovery.json", {"source_cohort": str(source),
        "statuses": dict(Counter(row["status"] for row in refresh_study_index(target))),
        "new_agent_starts": 0, "new_solver_runs": 0})
    print(json.dumps({"recovered_report_cohort": str(target), "new_solver_runs": 0}), flush=True)


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


def study_launcher_paths(args):
    name = (
        "launcher"
        if args.phase in {"study-models", "models", "all"}
        else "launcher-" + args.phase
    )
    if getattr(args, "worker_group", None):
        name += "-" + args.worker_group
    receipt = args.output / (name + "-process.json")
    return receipt, args.output / (name + ".log")


def active_study_launcher(args):
    receipt, _ = study_launcher_paths(args)
    if receipt.exists():
        old = json.loads(receipt.read_text())
        stat = Path(f"/proc/{old['pid']}/stat")
        if stat.is_file() and stat.read_text().split()[2] != "Z":
            return old
    return None


def launch_background(args, argv):
    receipt, log_path = study_launcher_paths(args)
    with cache_lock(receipt.with_suffix(".lock")):
        old = active_study_launcher(args)
        if old:
            print(json.dumps({**old, "status": "already_running"}), flush=True)
            return
        command = [
            sys.executable,
            "-m", "pitbench.harness.harness.campaign",
            "--root", str(ROOT),
            *argv,
            "--foreground",
        ]
        with log_path.open("a") as log:
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
            "log": str(log_path),
            "dynamic_evaluation_workers": getattr(args, "dynamic_evaluation_workers", False),
        }
        write_json(receipt, result)
        print(json.dumps(result), flush=True)



def run_study_cohort(config, *, workers=None, prepare_only=False, status=False,
                     foreground=False, livestream=False, root=None):
    """Launch or inspect a study using the existing resource-aware scheduler."""
    config = Path(config).expanduser().resolve()
    argv = ["--profile", str(config), "--phase", "study-status" if status else "study-models"]
    if workers is not None:
        argv.extend(["--workers", str(workers)])
    if prepare_only:
        argv.append("--prepare-only")
    if livestream:
        if not status:
            raise ValueError("livestream requires --status for a running study")
        argv.extend(["--watch", "--live-stream"])
    argv.append("--foreground" if foreground or status or prepare_only else "--background")
    return main(argv, root=root or Path.cwd())


def evaluate_saved_patch_cohort(
    config: Path, *, source_cohort: Path | None = None, output: Path | None = None,
    workers: int | None = None, prepare_only: bool = False, status: bool = False,
    foreground: bool = False, root: Path | None = None,
    reports_only: bool = False, report_output: Path | None = None,
):
    """Public configuration-based entry point; only evaluate captured candidates."""
    config = config.expanduser().resolve()
    settings = yaml.safe_load(config.read_text()) or {}
    phase = settings.get("phase", "study-evaluations")
    if phase not in {"study-evaluations", "study-parameters", "study-reports"}:
        raise ValueError("saved-patch configuration must select study-evaluations or study-parameters")
    if reports_only:
        phase = "study-reports"
    argv = ["--profile", str(config), "--phase", "study-status" if status else phase]
    if report_output is not None:
        argv.extend(["--report-output", str(report_output)])
    for option, value in (("--source-cohort", source_cohort), ("--output", output), ("--evaluation-workers", workers)):
        if value is not None:
            argv.extend([option, str(value)])
    if prepare_only:
        if phase != "study-evaluations":
            raise ValueError("prepare-only is available for the native saved-patch queue")
        argv.append("--prepare-only")
    argv.append("--foreground" if foreground else "--background")
    return main(argv, root=root or Path.cwd())


def main(argv=None, *, root=None):
    global ROOT
    if root is not None:
        ROOT = Path(root).resolve()
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--foreground", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-group", help=argparse.SUPPRESS)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--evaluate-candidates", action="store_true")
    parser.add_argument(
        "--evaluation-workers",
        type=int,
        default=0,
        help="Initial evaluation slots; 0 selects from host resources",
    )
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
            "study-adopt-running",
            "study-parameters",
            "study-reports",
        ],
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Agent slots; 0 selects from host resources",
    )
    parser.add_argument(
        "--reuse-agent-pools", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--temporary-root",
        type=Path,
        help="Host temporary directory (use a short path for Unix sockets)",
    )
    parser.add_argument(
        "--resource-policy",
        type=Path,
        help="JSON admission policy copied into the campaign",
    )
    parser.add_argument("--models", type=Path)
    parser.add_argument("--study-plan", type=Path)
    parser.add_argument("--study-views", type=Path)
    parser.add_argument("--source-cohort", type=Path)
    parser.add_argument("--target-task-matrix", type=Path)
    parser.add_argument("--cross-evaluation-output", type=Path)
    parser.add_argument("--schedule-seed", type=int, default=20261101)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Collect only unmodified BASE solver observations for a BASE calibration sweep",
    )
    parser.add_argument("--resume-models", action="store_true")
    parser.add_argument("--defer-evaluation-incomplete", action="store_true")
    parser.add_argument("--dispatch-interval-sec", type=float, default=None)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--resume-evaluation", action="store_true")
    parser.add_argument("--resume-thermal", action="store_true")
    parser.add_argument("--retry-run-id")
    parser.add_argument("--recover-all", action="store_true")
    parser.add_argument("--continuations-only", action="store_true")
    parser.add_argument("--recovery-interval-sec", type=float, default=2)
    parser.add_argument("--follow-source", action="store_true")
    parser.add_argument("--refresh-queue", action="store_true")
    parser.add_argument("--repair-boundary", action="store_true")
    parser.add_argument("--recover-observations", action="store_true")
    parser.add_argument("--refresh-retained-metrics", action="store_true")
    parser.add_argument("--provisional-performance", action="store_true")
    parser.add_argument("--rebalance-agents", action="store_true")
    parser.add_argument("--recovery-status", type=Path)
    parser.add_argument("--supplement-metrics", action="store_true")
    parser.add_argument("--supplement-recovered-only", action="store_true")
    parser.add_argument("--follow-interval-sec", type=float, default=30)
    parser.add_argument("--watch", action="store_true", help="Continuously stream study status and Agent activity")
    parser.add_argument("--watch-interval-sec", type=float, default=2)
    parser.add_argument("--live-stream", action="store_true", help="Render a compact live Agent activity stream")
    parser.add_argument("--task-id")
    parser.add_argument("--cpu-pool")
    parser.add_argument("--cpu-pools", action="append", default=[])
    args = parse_run_arguments(parser, argv)
    if args.worker_group and (len(args.worker_group) > 64 or not args.worker_group.replace("-", "").isalnum()):
        parser.error("invalid worker group")
    if args.root is not None:
        ROOT = args.root.resolve()
    if args.dispatch_interval_sec is None:
        args.dispatch_interval_sec = (
            20 if args.phase in {"study-models", "study-agent-recovery"} else 0
        )
    for name in ("manifest", "config", "output"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.phase == "study-status":
        if not (args.output / "cohort.json").is_file():
            parser.error("study-status requires an existing cohort.json")
        if args.live_stream and not args.watch:
            parser.error("--live-stream requires --watch")
        if args.watch:
            previous = None
            previous_live_status = None
            previous_live_activities = {}
            cursors = {}
            try:
                while True:
                    snapshot = study_live_snapshot(args.output)
                    if args.live_stream:
                        status_key = (
                            snapshot["total"],
                            tuple(sorted(snapshot["statuses"].items())),
                        )
                        if status_key != previous_live_status:
                            statuses = ", ".join(
                                f"{name}={count}" for name, count in sorted(snapshot["statuses"].items())
                            ) or "no active conditions"
                            print(
                                f"\n[{now()}] {snapshot['total']} conditions: {statuses}",
                                flush=True,
                            )
                            previous_live_status = status_key
                        for activity in snapshot["active_agents"]:
                            key = activity["run_id"]
                            detail = activity.get("activity") or "active"
                            if previous_live_activities.get(key) != detail:
                                print(
                                    f"  {activity['model']} / {activity['task_id']}: "
                                    f"{detail}",
                                    flush=True,
                                )
                                previous_live_activities[key] = detail
                    elif snapshot != previous:
                        print(json.dumps({"updated_at": now(), **snapshot}, ensure_ascii=False), flush=True)
                        previous = snapshot
                    events = study_live_events(args.output, snapshot, cursors)
                    if args.live_stream:
                        latest = {}
                        for event in events:
                            run_id = event["run_id"]
                            count, _ = latest.get(run_id, (0, None))
                            latest[run_id] = (count + 1, event)
                        for count, event in latest.values():
                            line = _live_event_summary(event)
                            if line:
                                suffix = f"  (+{count - 1} calls)" if count > 1 else ""
                                print(line + suffix, flush=True)
                    else:
                        for event in events:
                            print(json.dumps(event, ensure_ascii=False), flush=True)
                    time.sleep(max(0.1, args.watch_interval_sec))
            except KeyboardInterrupt:
                return
        else:
            records = refresh_study_index(args.output)
            print(json.dumps({"total": len(records), "statuses": dict(Counter(row["status"] for row in records))}), flush=True)
        return
    if args.phase == "study-adopt-running":
        if args.background and not args.foreground:
            launch_background(args, argv)
        else:
            adopt_running_study_agents(args)
        return
    if args.phase == "study-reports":
        if not (args.output / "cohort.json").is_file():
            parser.error("report reconstruction requires an existing cohort")
        if args.background and not args.foreground:
            launch_background(args, argv)
        else:
            rebuild_study_reports(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if args.temporary_root is not None:
        temporary = Path(args.temporary_root).expanduser().absolute()
        temporary.mkdir(parents=True, exist_ok=True)
        for key in ("TMPDIR", "TMP", "TEMP"):
            os.environ[key] = str(temporary)
        tempfile.tempdir = None
    if args.phase in {"study-models", "study-evaluations", "study-agent-recovery"}:
        if args.study_plan is None or args.study_views is None:
            parser.error("study phases require --study-plan and --study-views")
    if args.phase == "study-parameters" and args.source_cohort is None:
        parser.error("parameter supplementation requires --source-cohort")
    if args.resource_policy is not None:
        policy = (
            args.resource_policy
            if isinstance(args.resource_policy, dict)
            else json.loads(Path(args.resource_policy).read_text())
        )
        write_json(args.output / "resource-policy.json", policy)
    if args.background and not args.foreground:
        old = active_study_launcher(args)
        if old and args.phase == "study-evaluations" and args.evaluation_workers == 0 and not args.evaluation_cpu_pools and not args.worker_group and not old.get("dynamic_evaluation_workers", False):
            args.worker_group = "elastic"
            argv.extend(["--worker-group", args.worker_group])
            old = active_study_launcher(args)
        if old:
            print(json.dumps({**old, "status": "already_running"}), flush=True)
            return
    try:
        configure_study_parallelism(args)
    except ValueError as error:
        parser.error(str(error))
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
        if not args.dynamic_evaluation_workers and (args.evaluation_workers < 1 or (
            len(args.evaluation_cpu_pools) != args.evaluation_workers
            and not (
                args.phase == "study-models"
                and args.reuse_agent_pools
                and not args.evaluation_cpu_pools
                and args.cpu_pools
            )
        )):
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
        launch_background(args, argv)
        return
    manifest = json.loads(args.manifest.read_text())
    task_ids = [row["task_id"] for row in manifest["task_matrix"]]
    if len(task_ids) != len(set(task_ids)):
        parser.error("manifest task IDs must be unique")
    config = EvaluationConfig.from_yaml(args.config)
    if args.phase == "study-parameters":
        if args.source_cohort is None:
            parser.error("parameter supplementation requires --source-cohort")
        study_parameters(args)
        return
    if args.phase in {"study-models", "study-evaluations", "study-agent-recovery"}:
        if args.study_plan is None or args.study_views is None:
            parser.error("study phases require --study-plan and --study-views")
        args.study_plan = args.study_plan.resolve()
        args.study_views = args.study_views.resolve()
        if args.phase == "study-models":
            if bool(args.target_task_matrix) != bool(args.cross_evaluation_output):
                parser.error("cross-task evaluation requires both target_task_matrix and cross_evaluation_output")
            if args.target_task_matrix and args.evaluate_candidates:
                parser.error("cross-task evaluation requires evaluate_candidates: false")
            pools = args.cpu_pools + (
                args.evaluation_cpu_pools if args.evaluate_candidates else []
            )
            with study_cpu_lease(pools):
                study_models(args)
            if args.target_task_matrix and not args.prepare_only and not (args.output / "STOP").exists():
                import copy

                replay = copy.copy(args)
                replay.phase = "study-evaluations"
                replay.source_cohort = args.output
                replay.output = Path(args.cross_evaluation_output).resolve()
                replay.output.mkdir(parents=True, exist_ok=True)
                policy_path = args.output / "resource-policy.json"
                if policy_path.is_file():
                    shutil.copy2(policy_path, replay.output / "resource-policy.json")
                replay.cpu_pools = []
                replay.evaluation_cpu_pools = []
                replay.evaluation_workers = 0
                replay.dynamic_evaluation_workers = True
                replay.dispatch_interval_sec = 0
                configure_study_parallelism(replay)
                study_evaluations(replay)
        elif args.phase == "study-agent-recovery":
            if args.recover_all:
                if args.retry_run_id or args.cpu_pool:
                    parser.error("--recover-all cannot be combined with a single retry or CPU pool")
                recover_pending_conditions(args)
            elif not args.retry_run_id or not args.cpu_pool:
                parser.error("a retry run ID and a dedicated CPU pool are required")
            else:
                with study_cpu_lease([args.cpu_pool]):
                    retry_unexecuted_condition(args)
        else:
            if args.repair_boundary:
                repair_study_boundary_observations(args)
            elif args.recover_observations:
                print(json.dumps({"recovered": recover_study_collected_results(args)}), flush=True)
            elif args.refresh_retained_metrics:
                print(json.dumps({"refreshed": refresh_retained_study_metrics(args)}), flush=True)
            elif args.provisional_performance:
                watch_study_primary_results(args)
            elif args.rebalance_agents:
                rebalance_study_agents(args)
            elif args.resume_thermal:
                resume_thermal_study_evaluations(args)
            elif args.supplement_metrics:
                supplement_rebuilt_study_metrics(args)
            else:
                study_evaluations(args)
        return
    if args.phase in {"models", "all", "report"}:
        require_plotting()
    for row in manifest["task_matrix"]:
        task_id = row["task_id"]
        configured_task = config.resources_for(task_id).task_config_path
        task_record = (
            TaskCatalog(ROOT).validate_file(
                resolve_repository_path(ROOT, configured_task)
            )
            if configured_task is not None
            else TaskCatalog(ROOT).validate_one(task_id)
        )
        task = task_record.task
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
            if phase == "baseline":
                with study_cpu_lease(args.cpu_pools):
                    passed = run_phase(phase, task_ids, args, config)
            else:
                passed = run_phase(phase, task_ids, args, config)
            if not passed:
                raise SystemExit(1)
            config = EvaluationConfig.from_yaml(args.config)
    if args.phase in ("models", "all"):
        if args.models is None:
            parser.error("--models is required for model runs")
        models(task_ids, args)


if __name__ == "__main__":
    main()
