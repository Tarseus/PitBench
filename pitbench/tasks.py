from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from pitbench.families.base import ProblemFamilyRegistry
from pitbench.repositories.base import RepositoryPluginRegistry
from pitbench.schema.task import PitBenchTask, PopulationKind


class TaskValidationError(ValueError):
    pass


class TaskNotFoundError(TaskValidationError):
    pass


@dataclass(frozen=True)
class TaskRecord:
    manifest_path: Path
    task: PitBenchTask

    @property
    def manifest_sha256(self) -> str:
        return hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()


class TaskCatalog:
    """Loads and validates the public, agent-safe portion of task definitions."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.manifest_dir = self.root / "manifests" / "tasks"

    @staticmethod
    def _load_record(path: Path) -> TaskRecord:
        record = TaskRecord(path, PitBenchTask.from_yaml(path))
        if record.task.task_id != path.stem:
            raise TaskValidationError(
                f"{path}: task ID {record.task.task_id!r} must match manifest filename"
            )
        return record

    def records(self) -> list[TaskRecord]:
        records = [
            self._load_record(path) for path in sorted(self.manifest_dir.glob("*.yaml"))
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
        ProblemFamilyRegistry.load(task.evaluation.family_plugin)

        for population in task.populations:
            if population.kind == PopulationKind.AGENT_DEV:
                path = (self.root / population.manifest).resolve()
                if self.root not in path.parents or not path.is_file():
                    raise TaskValidationError(
                        f"{task.task_id}: missing visible population "
                        f"{population.manifest}"
                    )
                payload = yaml.safe_load(path.read_text())
                if (
                    population.manifest_sha256 is not None
                    and hashlib.sha256(path.read_bytes()).hexdigest()
                    != population.manifest_sha256
                ):
                    raise TaskValidationError(
                        f"{task.task_id}: visible population hash mismatch"
                    )
                if payload.get("visibility") != "agent":
                    raise TaskValidationError(
                        f"{task.task_id}: development population must be agent-visible"
                    )
                if "private://" in path.read_text():
                    raise TaskValidationError(
                        f"{task.task_id}: visible population leaks a private URI"
                    )
            elif not population.manifest.startswith("private://"):
                raise TaskValidationError(
                    f"{task.task_id}: judge population must use private:// storage"
                )

    def validate_one(self, task_id: str) -> TaskRecord:
        if not task_id or Path(task_id).name != task_id:
            raise TaskNotFoundError(f"unknown task ID: {task_id}")
        manifest_path = self.manifest_dir / f"{task_id}.yaml"
        if not manifest_path.is_file():
            raise TaskNotFoundError(f"unknown task ID: {task_id}")
        record = self._load_record(manifest_path)
        self.validate(record)
        return record

    def validate_all(self) -> list[TaskRecord]:
        records = self.records()
        if not records:
            raise TaskValidationError("no PitBench task manifests found")
        for record in records:
            self.validate(record)
        return records
