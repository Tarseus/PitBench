from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import shutil
import statistics
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq

from adapters.pitbench.adapter import PitBenchAdapter
from pitbench.cli.evaluate_config import (
    EvaluationConfig,
    PipelineResources,
    resolve_repository_path,
)
from pitbench.cli.task_image import prepare_task_image
from pitbench.evaluator.evaluator import PitBenchEvaluator
from pitbench.evaluator.storage import ObservationStore
from pitbench.harness.agents import AgentName
from pitbench.harness.agents.failure_mode import FailureMode
from pitbench.harness.evaluation import EvaluationRequest
from pitbench.harness.handlers.trial_handler import Task
from pitbench.harness.harness.harness import Harness
from pitbench.matrix.models import (
    MatrixAgentSpec,
    MatrixSpec,
    MatrixTrialState,
    PipelineStatus,
)
from pitbench.schema.observation import CodeState
from pitbench.tasks import TaskCatalog, TaskRecord


@dataclass(frozen=True)
class MatrixJob:
    task_id: str
    agent: MatrixAgentSpec
    repeat: int

    @property
    def id(self) -> str:
        return f"{self.task_id}__{self.agent.id}__r{self.repeat}"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _parse_cpu_set(value: str) -> tuple[int, ...]:
    cpus: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"invalid CPU range: {token}")
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(token))
    if not cpus:
        raise ValueError("CPU set cannot be empty")
    return tuple(sorted(cpus))


def _physical_cpus() -> tuple[int, ...]:
    available = sorted(os.sched_getaffinity(0))
    selected: list[int] = []
    seen_siblings: set[tuple[int, ...]] = set()
    for cpu in available:
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
        try:
            siblings = _parse_cpu_set(path.read_text().strip())
        except (OSError, ValueError):
            siblings = (cpu,)
        key = tuple(item for item in siblings if item in available)
        if key not in seen_siblings:
            seen_siblings.add(key)
            selected.append(cpu)
    return tuple(selected)


def _format_cpu_set(cpus: tuple[int, ...]) -> str:
    return ",".join(str(cpu) for cpu in cpus)


class MatrixRunner:
    """Durable producer-consumer runner for PitBench experiment matrices."""

    def __init__(
        self,
        *,
        repository_root: Path,
        spec: MatrixSpec,
        config: EvaluationConfig,
        run_id: str,
        rebuild: bool = False,
        progress: Callable[[str], None] = print,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.spec = spec
        self.config = config
        self.run_id = run_id
        self.rebuild = rebuild
        self.progress = progress
        self.output_root = resolve_repository_path(
            self.repository_root, config.paths.output_path
        ).resolve()
        self.workspace_root = resolve_repository_path(
            self.repository_root, config.paths.workspace_path
        ).resolve()
        self.private_root = resolve_repository_path(
            self.repository_root, config.paths.private_root
        ).resolve()
        self.run_root = self.output_root / run_id
        self.dataset_root = self.workspace_root / run_id
        self._records = {record.task.task_id: record for record in TaskCatalog(
            self.repository_root
        ).validate_all()}
        self._state_lock = threading.Lock()
        self._agent_slots, self._judge_slots = self._resolve_cpu_slots(config.pipeline)

    def _resolve_cpu_slots(
        self, resources: PipelineResources
    ) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
        physical = _physical_cpus()
        available = set(os.sched_getaffinity(0))
        agent_count = resources.agent_workers * resources.agent_cpus_per_worker
        judge_count = resources.judge_workers * resources.judge_parallel_runs

        agent_pool = (
            _parse_cpu_set(resources.agent_cpu_pool)
            if resources.agent_cpu_pool
            else physical[:agent_count]
        )
        remaining = tuple(cpu for cpu in physical if cpu not in agent_pool)
        judge_pool = (
            _parse_cpu_set(resources.judge_cpu_pool)
            if resources.judge_cpu_pool
            else remaining[:judge_count]
        )
        if not set(agent_pool) <= available or not set(judge_pool) <= available:
            raise ValueError("configured CPU pool includes CPUs outside process affinity")
        if set(agent_pool) & set(judge_pool):
            raise ValueError("agent and judge CPU pools must not overlap")
        if len(agent_pool) < agent_count:
            raise ValueError(f"agent CPU pool requires at least {agent_count} CPUs")
        if len(judge_pool) < judge_count:
            raise ValueError(f"judge CPU pool requires at least {judge_count} CPUs")
        physical_set = set(physical)
        if not set(agent_pool) <= physical_set or not set(judge_pool) <= physical_set:
            raise ValueError("CPU pools must use one hardware thread per physical core")

        agent_slots = [
            tuple(agent_pool[index : index + resources.agent_cpus_per_worker])
            for index in range(0, agent_count, resources.agent_cpus_per_worker)
        ]
        judge_slots = [
            tuple(judge_pool[index : index + resources.judge_parallel_runs])
            for index in range(0, judge_count, resources.judge_parallel_runs)
        ]
        return agent_slots, judge_slots

    def _resolved_lock(self) -> dict[str, Any]:
        tasks = {}
        for task_id in self.spec.tasks:
            record = self._record(task_id)
            resources = self.config.resources_for(task_id)
            judge_image = resources.judge_image or record.task.repository.judge_image
            if not judge_image:
                raise ValueError(
                    f"task {task_id} requires a pinned judge_image in local config"
                )
            tasks[task_id] = {
                "base_commit": record.task.release.base_commit,
                "manifest": str(record.manifest_path.resolve()),
                "manifest_sha256": record.manifest_sha256,
                "repository_source": (
                    str(resolve_repository_path(
                        self.repository_root, resources.repository_source
                    ).resolve())
                    if resources.repository_source is not None
                    else None
                ),
                "agent_image": resources.agent_image
                or record.task.repository.agent_image,
                "judge_image": judge_image,
            }
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.repository_root, text=True
        ).strip()
        return {
            "lock_version": "1.0",
            "pitbench_commit": commit,
            "evaluator_version": PitBenchEvaluator.version,
            "run_id": self.run_id,
            "spec": self.spec.model_dump(mode="json"),
            "pipeline": self.config.pipeline.model_dump(mode="json"),
            "agent_slots": [list(slot) for slot in self._agent_slots],
            "judge_slots": [list(slot) for slot in self._judge_slots],
            "tasks": tasks,
        }

    def _record(self, task_id: str) -> TaskRecord:
        try:
            record = self._records[task_id]
        except KeyError as exc:
            raise ValueError(f"unknown matrix task: {task_id}") from exc
        if record.task.evaluation.threads != 1:
            raise ValueError(
                f"matrix throughput protocol requires {task_id} evaluation.threads=1"
            )
        return record

    def _prepare(self) -> None:
        for agent in self.spec.agents:
            try:
                AgentName(agent.agent)
            except ValueError as exc:
                raise ValueError(f"unknown matrix agent: {agent.agent}") from exc

        lock = self._resolved_lock()
        lock_path = self.run_root / "matrix.lock.json"
        if lock_path.is_file():
            existing = json.loads(lock_path.read_text())
            if existing != lock:
                raise ValueError("existing matrix run lock does not match this invocation")
        else:
            self.run_root.mkdir(parents=True, exist_ok=True)
            _atomic_json(lock_path, lock)

        adapter = PitBenchAdapter(self.repository_root, self.private_root)
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        for task_id in self.spec.tasks:
            record = self._record(task_id)
            resources = self.config.resources_for(task_id)
            task_path = self.dataset_root / task_id
            if not (task_path / "task.yaml").is_file():
                if task_path.exists():
                    shutil.rmtree(task_path)
                repository_source = (
                    resolve_repository_path(
                        self.repository_root, resources.repository_source
                    ).resolve()
                    if resources.repository_source is not None
                    else None
                )
                judge_image = resources.judge_image or record.task.repository.judge_image
                try:
                    adapter.materialize(
                        task_id,
                        task_path,
                        repository_source=repository_source,
                        agent_image=resources.agent_image,
                        judge_image=judge_image,
                    )
                except Exception:
                    if task_path.exists() and not (task_path / "task.yaml").is_file():
                        shutil.rmtree(task_path)
                    raise
            prepare_task_image(
                task_path,
                rebuild=self.rebuild,
                progress=self.progress,
            )

    def _jobs(self) -> list[MatrixJob]:
        jobs: list[MatrixJob] = []
        for repeat in range(1, self.spec.repeats + 1):
            for task_index, task_id in enumerate(self.spec.tasks):
                agents = list(self.spec.agents)
                random.Random(
                    self.spec.schedule_seed + repeat * 1009 + task_index * 9176
                ).shuffle(agents)
                jobs.extend(MatrixJob(task_id, agent, repeat) for agent in agents)
        return jobs

    def _trial_dir(self, job: MatrixJob) -> Path:
        return self.run_root / "trials" / job.task_id / job.agent.id / f"repeat-{job.repeat}"

    def _state_path(self, job: MatrixJob) -> Path:
        return self._trial_dir(job) / "state.json"

    def _load_state(self, job: MatrixJob) -> MatrixTrialState:
        path = self._state_path(job)
        if path.is_file():
            return MatrixTrialState.model_validate_json(path.read_text())
        return MatrixTrialState(
            task_id=job.task_id,
            agent_id=job.agent.id,
            repeat=job.repeat,
            updated_at=_utcnow(),
        )

    def _save_state(self, job: MatrixJob, state: MatrixTrialState) -> None:
        state.updated_at = _utcnow()
        with self._state_lock:
            _atomic_json(self._state_path(job), state.model_dump(mode="json"))

    def _task_evaluator_config(self, task_id: str) -> dict[str, Any]:
        task = Task.from_yaml(self.dataset_root / task_id / "task.yaml")
        config = task.evaluator_config.copy()
        if "judge_image" not in config:
            raise ValueError(f"materialized task {task_id} has no judge_image")
        return config

    def _slot_for(self, task_id: str, repeat: int) -> int:
        task_index = self.spec.tasks.index(task_id)
        return (task_index + repeat - 1) % len(self._judge_slots)

    def _run_base(self, task_id: str, repeat: int, slot_index: int) -> Path:
        root = self.run_root / "baselines" / task_id / f"repeat-{repeat}"
        state_path = root / "state.json"
        canonical = root / "trials.parquet"
        if state_path.is_file() and canonical.is_file():
            state = json.loads(state_path.read_text())
            if state.get("status") == "completed":
                return canonical

        max_attempts = self.config.pipeline.infrastructure_retries + 1
        failures: list[str] = []
        for attempt in range(1, max_attempts + 1):
            _atomic_json(
                state_path,
                {
                    "status": "evaluating",
                    "attempt": attempt,
                    "failures": failures,
                    "updated_at": _utcnow(),
                },
            )
            attempt_dir = root / "attempts" / str(attempt)
            attempt_dir.mkdir(parents=True, exist_ok=True)
            patch = attempt_dir / "empty.patch"
            patch.write_text("")
            config = self._task_evaluator_config(task_id)
            config.update(
                {
                    "code_states": [CodeState.BASE.value],
                    "judge_cpus": float(len(self._judge_slots[slot_index])),
                    "judge_cpuset_cpus": _format_cpu_set(
                        self._judge_slots[slot_index]
                    ),
                    "judge_parallel_runs": self.config.pipeline.judge_parallel_runs,
                    "_progress_callback": lambda message: self.progress(
                        f"{task_id}/base/repeat-{repeat} — {message}"
                    ),
                }
            )
            envelope = PitBenchEvaluator().envelope(
                EvaluationRequest(
                    task_id=task_id,
                    task_path=self.dataset_root / task_id,
                    candidate_patch_path=patch,
                    output_dir=attempt_dir,
                    agent_name="base-cache",
                    model_name=None,
                    evaluator_config=config,
                )
            )
            _atomic_json(
                attempt_dir / "evaluation.json", envelope.model_dump(mode="json")
            )
            if envelope.completed:
                observations = ObservationStore.read(attempt_dir / "trials.parquet")
                if not observations or any(
                    item.code_state != CodeState.BASE for item in observations
                ):
                    envelope_error = "BASE evaluation produced invalid observations"
                else:
                    ObservationStore.write(canonical, observations)
                    _atomic_json(
                        state_path,
                        {
                            "status": "completed",
                            "attempt": attempt,
                            "failures": failures,
                            "observations": len(observations),
                            "updated_at": _utcnow(),
                        },
                    )
                    self.progress(f"BASE ready {task_id} repeat {repeat}")
                    return canonical
            else:
                envelope_error = envelope.error or "unknown BASE evaluator failure"
            failures.append(envelope_error)

        _atomic_json(
            state_path,
            {
                "status": "failed",
                "attempt": max_attempts,
                "failures": failures,
                "updated_at": _utcnow(),
            },
        )
        raise RuntimeError(f"BASE failed for {task_id} repeat {repeat}: {failures[-1]}")

    @staticmethod
    def _retryable_generation_failure(mode: FailureMode) -> bool:
        return mode in {
            FailureMode.UNKNOWN_AGENT_ERROR,
            FailureMode.AGENT_INSTALLATION_FAILED,
        }

    @staticmethod
    def _retryable_evaluator_error(error: str) -> bool:
        lowered = error.lower()
        deterministic = (
            "build failed",
            "patch failed",
            "candidate patch rejected",
            "protected path",
        )
        return not any(marker in lowered for marker in deterministic)

    def _generate_candidate(
        self, job: MatrixJob, agent_slot: tuple[int, ...]
    ) -> MatrixTrialState:
        state = self._load_state(job)
        patch_path = Path(state.candidate_patch) if state.candidate_patch else None
        if patch_path is not None and patch_path.is_file() and state.status in {
            PipelineStatus.CANDIDATE_READY,
            PipelineStatus.EVALUATING,
        }:
            return state
        if state.status == PipelineStatus.COMPLETED:
            return state

        max_attempts = self.config.pipeline.infrastructure_retries + 1
        started = time.monotonic()
        while state.generation_attempts < max_attempts:
            state.generation_attempts += 1
            attempt = state.generation_attempts
            state.status = PipelineStatus.GENERATING
            state.error = None
            self._save_state(job, state)
            attempt_root = self._trial_dir(job) / "agent-attempts"
            attempt_run_id = f"{job.agent.id}-r{job.repeat}-attempt-{attempt}"
            kwargs: dict[str, Any] = {
                "no_rebuild": False,
                **self.config.kwargs_for(job.agent.agent),
                **job.agent.agent_kwargs,
            }
            try:
                harness = Harness(
                    output_path=attempt_root,
                    run_id=attempt_run_id,
                    dataset_path=self.dataset_root,
                    task_ids=[job.task_id],
                    agent_name=AgentName(job.agent.agent),
                    model_name=job.agent.model,
                    agent_kwargs=kwargs,
                    no_rebuild=True,
                    cleanup=False,
                    log_level=logging.INFO,
                    livestream=False,
                    upload_results=False,
                    n_concurrent_trials=1,
                    n_attempts=1,
                    defer_evaluation=True,
                    agent_cpuset_cpus=_format_cpu_set(agent_slot),
                    progress_bar=False,
                    progress_callback=lambda detail: self.progress(
                        f"{job.id} — {detail}"
                    ),
                )
                benchmark = harness.run()
                result = benchmark.results[0]
                state.total_input_tokens += result.total_input_tokens or 0
                state.total_output_tokens += result.total_output_tokens or 0
                state.total_cost += result.total_cost or 0.0
                result_path = (
                    attempt_root
                    / attempt_run_id
                    / job.task_id
                    / result.trial_name
                    / "results.json"
                )
                candidate = result_path.parent / "evaluation" / "candidate.patch"
                retryable = self._retryable_generation_failure(result.failure_mode)
                if retryable:
                    detail = f"attempt {attempt}: {result.failure_mode.value}"
                    state.infrastructure_failures.append(detail)
                    state.error = detail
                    self._save_state(job, state)
                    continue
                if not candidate.is_file():
                    state.status = PipelineStatus.FAILED
                    state.error = (
                        f"agent finished with {result.failure_mode.value} without "
                        "a candidate patch"
                    )
                    self._save_state(job, state)
                    return state
                canonical = self._trial_dir(job) / "candidate.patch"
                canonical.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(candidate, canonical)
                agent_result = self._trial_dir(job) / "agent-result.json"
                shutil.copyfile(result_path, agent_result)
                state.candidate_patch = str(canonical)
                state.candidate_sha256 = hashlib.sha256(canonical.read_bytes()).hexdigest()
                state.agent_result = str(agent_result)
                state.agent_duration_sec = time.monotonic() - started
                state.status = PipelineStatus.CANDIDATE_READY
                state.error = None
                self._save_state(job, state)
                self.progress(f"Candidate ready {job.id}")
                return state
            except Exception as exc:
                detail = f"attempt {attempt}: {type(exc).__name__}: {exc}"
                state.infrastructure_failures.append(detail)
                state.error = detail
                self._save_state(job, state)

        state.status = PipelineStatus.FAILED
        state.agent_duration_sec = time.monotonic() - started
        state.error = state.error or "agent infrastructure retries exhausted"
        self._save_state(job, state)
        return state

    def _evaluate_candidate(
        self,
        job: MatrixJob,
        base_future: Future[Path],
        slot_index: int,
    ) -> MatrixTrialState:
        state = self._load_state(job)
        if state.status == PipelineStatus.COMPLETED:
            return state
        if state.candidate_patch is None or not Path(state.candidate_patch).is_file():
            state.status = PipelineStatus.FAILED
            state.error = "candidate patch is missing"
            self._save_state(job, state)
            return state
        try:
            base_path = base_future.result()
        except Exception as exc:
            state.status = PipelineStatus.FAILED
            state.error = f"BASE unavailable: {exc}"
            self._save_state(job, state)
            return state

        max_attempts = self.config.pipeline.infrastructure_retries + 1
        started = time.monotonic()
        while state.evaluation_attempts < max_attempts:
            state.evaluation_attempts += 1
            attempt = state.evaluation_attempts
            state.status = PipelineStatus.EVALUATING
            state.error = None
            self._save_state(job, state)
            attempt_dir = self._trial_dir(job) / "evaluation" / "attempts" / str(attempt)
            attempt_dir.mkdir(parents=True, exist_ok=True)
            config = self._task_evaluator_config(job.task_id)
            config.update(
                {
                    "code_states": [CodeState.AGENT.value],
                    "base_observations_path": str(base_path),
                    "judge_cpus": float(len(self._judge_slots[slot_index])),
                    "judge_cpuset_cpus": _format_cpu_set(
                        self._judge_slots[slot_index]
                    ),
                    "judge_parallel_runs": self.config.pipeline.judge_parallel_runs,
                    "_progress_callback": lambda message: self.progress(
                        f"{job.id} — {message}"
                    ),
                }
            )
            envelope = PitBenchEvaluator().envelope(
                EvaluationRequest(
                    task_id=job.task_id,
                    task_path=self.dataset_root / job.task_id,
                    candidate_patch_path=Path(state.candidate_patch),
                    output_dir=attempt_dir,
                    agent_name=job.agent.agent,
                    model_name=job.agent.model,
                    evaluator_config=config,
                    telemetry={
                        "total_input_tokens": state.total_input_tokens,
                        "total_output_tokens": state.total_output_tokens,
                        "total_cost": state.total_cost,
                    },
                )
            )
            result_path = attempt_dir / "evaluation.json"
            _atomic_json(result_path, envelope.model_dump(mode="json"))
            if envelope.completed:
                state.status = PipelineStatus.COMPLETED
                state.evaluation_result = str(result_path)
                state.evaluation_duration_sec = time.monotonic() - started
                state.error = None
                self._save_state(job, state)
                self.progress(f"Evaluation complete {job.id}")
                return state
            error = envelope.error or "unknown evaluator failure"
            if self._retryable_evaluator_error(error):
                state.infrastructure_failures.append(f"judge attempt {attempt}: {error}")
                state.error = error
                self._save_state(job, state)
                continue
            state.status = PipelineStatus.FAILED
            state.error = error
            state.evaluation_result = str(result_path)
            state.evaluation_duration_sec = time.monotonic() - started
            self._save_state(job, state)
            return state

        state.status = PipelineStatus.FAILED
        state.evaluation_duration_sec = time.monotonic() - started
        state.error = state.error or "judge infrastructure retries exhausted"
        self._save_state(job, state)
        return state

    def _row(self, job: MatrixJob, state: MatrixTrialState) -> dict[str, Any]:
        row: dict[str, Any] = {
            "task_id": job.task_id,
            "agent_id": job.agent.id,
            "agent": job.agent.agent,
            "model": job.agent.model,
            "repeat": job.repeat,
            "status": state.status.value,
            "generation_attempts": state.generation_attempts,
            "evaluation_attempts": state.evaluation_attempts,
            "candidate_sha256": state.candidate_sha256,
            "total_input_tokens": state.total_input_tokens,
            "total_output_tokens": state.total_output_tokens,
            "total_cost": state.total_cost,
            "agent_duration_sec": state.agent_duration_sec,
            "evaluation_duration_sec": state.evaluation_duration_sec,
            "error": state.error,
            "is_resolved": False,
            "classification": None,
            "validity_accepted": False,
            "id_gap_reduction": None,
            "shift_gap_reduction": None,
            "shift_minus_id_gap_reduction": None,
        }
        if state.evaluation_result is None or not Path(state.evaluation_result).is_file():
            return row
        envelope = json.loads(Path(state.evaluation_result).read_text())
        if not envelope.get("completed"):
            return row
        payload = envelope["payload"]
        summary = payload["summary"]
        decision = summary.get("decision") or {}
        performance = summary.get("performance") or {}
        row["is_resolved"] = bool(envelope.get("is_resolved"))
        row["classification"] = decision.get("classification")
        row["validity_accepted"] = bool(payload.get("validity", {}).get("accepted"))
        primary = performance.get("primary") or {}
        row["id_gap_reduction"] = (primary.get("paired") or {}).get(
            "mean_gap_reduction"
        )
        primary_budget = performance.get("primary_budget_sec")
        shift = (performance.get("populations") or {}).get("judge_shift") or {}
        if primary_budget is not None:
            shift_budget = (shift.get("by_budget") or {}).get(f"{primary_budget:g}")
            if shift_budget:
                row["shift_gap_reduction"] = (
                    shift_budget.get("paired") or {}
                ).get("mean_gap_reduction")
        for retention in performance.get("held_out_retention") or []:
            if retention.get("budget_sec") == primary_budget:
                row["shift_minus_id_gap_reduction"] = retention.get(
                    "shift_minus_id_gap_reduction"
                )
                break
        return row

    def _write_summary(self, jobs: list[MatrixJob]) -> None:
        rows = [self._row(job, self._load_state(job)) for job in jobs]
        parquet = self.run_root / "matrix-trials.parquet"
        temporary = parquet.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
        temporary.replace(parquet)

        cells = []
        for task_id in self.spec.tasks:
            for agent in self.spec.agents:
                selected = [
                    row
                    for row in rows
                    if row["task_id"] == task_id and row["agent_id"] == agent.id
                ]
                id_values = [
                    row["id_gap_reduction"]
                    for row in selected
                    if row["id_gap_reduction"] is not None
                ]
                shift_values = [
                    row["shift_gap_reduction"]
                    for row in selected
                    if row["shift_gap_reduction"] is not None
                ]
                cells.append(
                    {
                        "task_id": task_id,
                        "agent_id": agent.id,
                        "completed": sum(
                            row["status"] == PipelineStatus.COMPLETED.value
                            for row in selected
                        ),
                        "valid": sum(row["validity_accepted"] for row in selected),
                        "resolved": sum(row["is_resolved"] for row in selected),
                        "id_gap_reduction_mean": (
                            statistics.fmean(id_values) if id_values else None
                        ),
                        "id_gap_reduction_median": (
                            statistics.median(id_values) if id_values else None
                        ),
                        "id_gap_reduction_std": (
                            statistics.stdev(id_values) if len(id_values) > 1 else None
                        ),
                        "shift_gap_reduction_mean": (
                            statistics.fmean(shift_values) if shift_values else None
                        ),
                        "shift_gap_reduction_median": (
                            statistics.median(shift_values) if shift_values else None
                        ),
                        "shift_gap_reduction_std": (
                            statistics.stdev(shift_values)
                            if len(shift_values) > 1
                            else None
                        ),
                    }
                )
        _atomic_json(
            self.run_root / "matrix-summary.json",
            {"run_id": self.run_id, "trials": rows, "cells": cells},
        )
        lines = [
            f"# Matrix report: {self.run_id}",
            "",
            "Task | Configuration | Completed | Valid | Resolved | ID mean | Shift mean",
            "--- | --- | ---: | ---: | ---: | ---: | ---:",
        ]
        for cell in cells:
            id_mean = cell["id_gap_reduction_mean"]
            shift_mean = cell["shift_gap_reduction_mean"]
            lines.append(
                f"{cell['task_id']} | {cell['agent_id']} | {cell['completed']}/"
                f"{self.spec.repeats} | {cell['valid']}/{self.spec.repeats} | "
                f"{cell['resolved']}/{self.spec.repeats} | "
                f"{'-' if id_mean is None else f'{100 * id_mean:.3f}%'} | "
                f"{'-' if shift_mean is None else f'{100 * shift_mean:.3f}%'}"
            )
        report = self.run_root / "matrix-report.md"
        temporary_report = report.with_suffix(".md.tmp")
        temporary_report.write_text("\n".join(lines) + "\n")
        temporary_report.replace(report)

    def run(self) -> Path:
        self._prepare()
        jobs = self._jobs()
        self.progress(
            f"Matrix {self.run_id}: {len(jobs)} candidates, "
            f"{len(self._agent_slots)} agent workers, "
            f"{len(self._judge_slots)} judge workers"
        )
        slot_executors = [
            ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"judge-{index}")
            for index in range(len(self._judge_slots))
        ]
        base_futures: dict[tuple[str, int], Future[Path]] = {}
        evaluation_futures: list[Future[MatrixTrialState]] = []
        try:
            for repeat in range(1, self.spec.repeats + 1):
                for task_id in self.spec.tasks:
                    slot = self._slot_for(task_id, repeat)
                    base_futures[(task_id, repeat)] = slot_executors[slot].submit(
                        self._run_base, task_id, repeat, slot
                    )

            pending_generation = []
            agent_executors = [
                ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"agent-{index}")
                for index in range(len(self._agent_slots))
            ]
            try:
                pending_index = 0
                for job in jobs:
                    state = self._load_state(job)
                    if state.status == PipelineStatus.COMPLETED:
                        continue
                    if state.candidate_patch and Path(state.candidate_patch).is_file():
                        slot = self._slot_for(job.task_id, job.repeat)
                        evaluation_futures.append(
                            slot_executors[slot].submit(
                                self._evaluate_candidate,
                                job,
                                base_futures[(job.task_id, job.repeat)],
                                slot,
                            )
                        )
                        continue
                    agent_slot_index = pending_index % len(self._agent_slots)
                    pending_index += 1
                    agent_slot = self._agent_slots[agent_slot_index]
                    future = agent_executors[agent_slot_index].submit(
                        self._generate_candidate, job, agent_slot
                    )
                    pending_generation.append((future, job))

                future_to_job = {future: job for future, job in pending_generation}
                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        state = future.result()
                    except Exception as exc:
                        state = self._load_state(job)
                        state.status = PipelineStatus.FAILED
                        state.error = f"generation worker failed: {exc}"
                        self._save_state(job, state)
                        continue
                    if state.status != PipelineStatus.CANDIDATE_READY:
                        continue
                    slot = self._slot_for(job.task_id, job.repeat)
                    evaluation_futures.append(
                        slot_executors[slot].submit(
                            self._evaluate_candidate,
                            job,
                            base_futures[(job.task_id, job.repeat)],
                            slot,
                        )
                    )
            finally:
                for executor in agent_executors:
                    executor.shutdown(wait=True, cancel_futures=False)

            for future in base_futures.values():
                try:
                    future.result()
                except Exception as exc:
                    self.progress(f"BASE failure: {exc}")
            for future in as_completed(evaluation_futures):
                future.result()
        finally:
            for executor in slot_executors:
                executor.shutdown(wait=True, cancel_futures=False)

        self._write_summary(jobs)
        self.progress(f"Matrix complete: {self.run_root / 'matrix-report.md'}")
        return self.run_root
