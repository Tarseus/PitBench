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

    def records(self) -> list[TaskRecord]:
        records = [
            TaskRecord(path, PitBenchTask.from_yaml(path))
            for path in sorted(self.manifest_dir.glob("*.yaml"))
        ]
        ids = [record.task.task_id for record in records]
        if len(ids) != len(set(ids)):
            raise TaskValidationError("task IDs must be unique")
        return records

    def validate(self, record: TaskRecord) -> None:
        task = record.task
        if len(task.event.base_commit) != 40 or len(task.event.human_commit) != 40:
            raise TaskValidationError(f"{task.task_id}: commits must be full SHA-1s")
        if task.event.pr_merged_at < task.event.pr_created_at:
            raise TaskValidationError(f"{task.task_id}: merge predates creation")
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

    def validate_all(self) -> list[TaskRecord]:
        records = self.records()
        if not records:
            raise TaskValidationError("no PitBench task manifests found")
        for record in records:
            self.validate(record)
        return records
