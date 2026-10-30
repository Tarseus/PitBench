#!/usr/bin/env python3
"""Build a manifest's releases, run full unchanged-source Judges, then model runs."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from adapters.pitbench.adapter import PitBenchAdapter
from pitbench.cli.evaluate_config import EvaluationConfig, EvaluationPaths
from pitbench.cli.main import _evaluation_metric_failures
from pitbench.cli.solver_images import build_solver_image
from pitbench.evaluator.evaluator import PitBenchEvaluator
from pitbench.evaluator.storage import ObservationStore
from pitbench.harness.evaluation import EvaluationRequest
from pitbench.metrics.campaign_report import require_plotting, write_campaign_report
from pitbench.schema.observation import RunStatus
from pitbench.tasks import TaskCatalog

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


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


def audit_model_run(run_dir, task_id):
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
    evidence = {
        "failure_mode": failure,
        "tokens": tokens,
        "completed_turn": completed_turn,
        "harness_result": str(path),
    }
    if failure == "unknown_agent_error" and tokens == 0 and rate_limited:
        return {
            **evidence,
            "complete": False,
            "retryable": True,
            "reason": "initial model request rate limited; no model execution",
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
                "--no-configuration-robustness",
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=["build", "baseline", "models", "report", "all", "_judge"],
        required=True,
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--resume-models", action="store_true")
    parser.add_argument("--defer-evaluation-incomplete", action="store_true")
    parser.add_argument("--dispatch-interval-sec", type=float, default=0)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--task-id")
    parser.add_argument("--cpu-pool")
    parser.add_argument("--cpu-pools", action="append", default=[])
    args = parser.parse_args()
    for name in ("manifest", "config", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.workers < 1 or (args.cpu_pools and len(args.cpu_pools) != args.workers):
        parser.error("positive workers and one CPU pool per worker are required")
    if args.phase in ("models", "all") and args.workers > 1 and not args.cpu_pools:
        parser.error("parallel model runs require disjoint --cpu-pools")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    task_ids = [row["task_id"] for row in manifest["task_matrix"]]
    if len(task_ids) != len(set(task_ids)):
        parser.error("manifest task IDs must be unique")
    config = EvaluationConfig.from_yaml(args.config)
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
