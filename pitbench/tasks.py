from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

from pitbench.instances import verify_public_file
from pitbench.problem_families.base import ProblemFamilyRegistry
from pitbench.repositories.base import RepositoryPluginRegistry
from pitbench.schema.task import InstanceSetKind, PitBenchTask


class TaskValidationError(ValueError):
    pass


class TaskNotFoundError(TaskValidationError):
    pass


@dataclass(frozen=True)
class TaskRecord:
    task_config_path: Path
    task: PitBenchTask

    @property
    def task_config_sha256(self) -> str:
        return hashlib.sha256(self.task_config_path.read_bytes()).hexdigest()


class TaskCatalog:
    """Loads and validates the public, agent-safe portion of task definitions."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.task_config_dir = self.root / "configs" / "tasks"

    @staticmethod
    def _load_record(path: Path) -> TaskRecord:
        record = TaskRecord(path, PitBenchTask.from_yaml(path))
        if record.task.task_id != path.stem:
            raise TaskValidationError(
                f"{path}: task ID {record.task.task_id!r} must match config filename"
            )
        return record

    def records(self) -> list[TaskRecord]:
        records = [
            self._load_record(path)
            for path in sorted(self.task_config_dir.glob("*.yaml"))
        ]
        ids = [record.task.task_id for record in records]
        if len(ids) != len(set(ids)):
            raise TaskValidationError("task IDs must be unique")
        return records

    def validate(self, record: TaskRecord) -> None:
        task = record.task
        if len(task.release.base_commit) != 40:
            raise TaskValidationError(
                f"{task.task_id}: release commit must be a full SHA-1"
            )
        if task.evaluation.threads != 1:
            raise TaskValidationError(
                f"{task.task_id}: current tasks require one solver thread per run; "
                "the judge parallelizes independent processes"
            )
        RepositoryPluginRegistry.load(task.repository.plugin)
        ProblemFamilyRegistry.load(task.problem_family)

        for instance_set in task.instance_sets:
            if instance_set.kind == InstanceSetKind.AGENT_DEV:
                path = (self.root / instance_set.instance_set_config).resolve()
                if self.root not in path.parents or not path.is_file():
                    raise TaskValidationError(
                        f"{task.task_id}: missing visible instance set "
                        f"{instance_set.instance_set_config}"
                    )
                payload = yaml.safe_load(path.read_text())
                if (
                    instance_set.instance_set_config_sha256 is not None
                    and hashlib.sha256(path.read_bytes()).hexdigest()
                    != instance_set.instance_set_config_sha256
                ):
                    raise TaskValidationError(
                        f"{task.task_id}: visible instance-set config hash mismatch"
                    )
                if payload.get("visibility") != "agent":
                    raise TaskValidationError(
                        f"{task.task_id}: development instance set must be agent-visible"
                    )
                if "private://" in path.read_text():
                    raise TaskValidationError(
                        f"{task.task_id}: visible instance set leaks a private URI"
                    )
                if "instances" in payload:
                    if len(payload["instances"]) != instance_set.size:
                        raise TaskValidationError(
                            f"{task.task_id}: visible instance-set size mismatch"
                        )
                    for item in payload["instances"]:
                        verify_public_file(
                            path.parent,
                            item["instance_file"],
                            item["instance_file_sha256"],
                        )
                        verify_public_file(
                            path.parent,
                            item["bks_solution_file"],
                            item["bks_solution_file_sha256"],
                        )
                        if "proof_file" in item:
                            verify_public_file(
                                path.parent,
                                item["proof_file"],
                                item["proof_file_sha256"],
                            )
            elif not instance_set.instance_set_config.startswith("private://"):
                raise TaskValidationError(
                    f"{task.task_id}: judge instance set must use private:// storage"
                )

    def validate_one(self, task_id: str) -> TaskRecord:
        if not task_id or Path(task_id).name != task_id:
            raise TaskNotFoundError(f"unknown task ID: {task_id}")
        task_config_path = self.task_config_dir / f"{task_id}.yaml"
        if not task_config_path.is_file():
            raise TaskNotFoundError(f"unknown task ID: {task_id}")
        record = self._load_record(task_config_path)
        self.validate(record)
        return record

    def validate_all(self) -> list[TaskRecord]:
        records = self.records()
        if not records:
            raise TaskValidationError("no PitBench task configs found")
        for record in records:
            self.validate(record)
        return records

    def inventory(
        self, *, registry_path: Path | None = None, baseline_commit: str | None = None
    ) -> dict:
        """Record existing task definitions and recheck supplied historical evidence.

        Registry stages are reported verbatim, independently of recheck results.
        Missing evidence stays unknown and no task configuration is rewritten.
        """
        from pitbench.repositories.pr_curation import (
            read_jsonl,
            validate_evidence_stage,
        )

        rows = read_jsonl(registry_path) if registry_path is not None else []
        indexed: dict[str, list[dict]] = {}
        for row in rows:
            if row.get("task_id"):
                indexed.setdefault(row["task_id"], []).append(row)
        baseline_configs = {}
        if baseline_commit is not None:
            # Resolve to a commit first; option-like or non-commit inputs cannot
            # turn the subsequent path lookups into arbitrary Git commands.
            baseline_commit = subprocess.check_output(
                [
                    "git",
                    "rev-parse",
                    "--verify",
                    "--end-of-options",
                    f"{baseline_commit}^{{commit}}",
                ],
                cwd=self.root,
                text=True,
            ).strip()
            paths = subprocess.check_output(
                [
                    "git",
                    "ls-tree",
                    "-r",
                    "--name-only",
                    baseline_commit,
                    "--",
                    "configs/tasks",
                ],
                cwd=self.root,
                text=True,
            ).splitlines()
            for path in paths:
                if path.endswith(".yaml"):
                    raw = subprocess.check_output(
                        ["git", "show", f"{baseline_commit}:{path}"], cwd=self.root
                    )
                    baseline_configs[Path(path).stem] = raw
        tasks = []
        for record in self.validate_all():
            task = record.task
            evidence = []
            for row in indexed.get(task.task_id, []):
                error = None
                try:
                    if (row.get("repository"), row.get("base_commit")) != (
                        task.release.repository,
                        task.release.base_commit,
                    ):
                        raise ValueError("registry source differs from task")
                    validate_evidence_stage(row, self.root)
                except (ValueError, OSError) as failure:
                    error = str(failure)
                evidence.append(
                    {
                        "repository": row["repository"],
                        "pr_number": row["pr_number"],
                        "recorded_stage": row.get("stage"),
                        "recorded_track": row.get("task_track", "expert_reference"),
                        "recheck": "failed" if error else "passed",
                        "error": error,
                        "artifacts": {
                            key: value
                            for key, value in row.items()
                            if key.endswith(("_evidence_path", "_evidence_sha256"))
                            or key in {"source_patch_path", "source_patch_sha256"}
                        },
                    }
                )
            baseline = baseline_configs.get(task.task_id)
            current = yaml.safe_load(record.task_config_path.read_text())
            old = yaml.safe_load(baseline) if baseline is not None else None
            changed = (
                sorted(
                    key
                    for key in current.keys() | old.keys()
                    if current.get(key) != old.get(key)
                )
                if old is not None
                else None
            )
            tasks.append(
                {
                    "task_id": task.task_id,
                    "task_config_sha256": record.task_config_sha256,
                    "base_commit": task.release.base_commit,
                    "repository": task.release.repository,
                    "baseline_task_config_sha256": hashlib.sha256(baseline).hexdigest()
                    if baseline is not None
                    else None,
                    "changed_fields": changed,
                    "evidence": evidence,
                    "evidence_status": "recorded" if evidence else "unknown",
                }
            )
        current_ids = {item["task_id"] for item in tasks}
        return {
            "schema_version": "1.0",
            "baseline_commit": baseline_commit,
            "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest()
            if registry_path
            else None,
            "task_count": len(tasks),
            "removed_task_ids": sorted(set(baseline_configs) - current_ids),
            "tasks": tasks,
            "claim_boundary": "Historical evidence recheck only; no new solver/Agent execution or readiness promotion.",
        }
