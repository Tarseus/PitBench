from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from pitbench.families.base import ProblemFamilyRegistry
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
