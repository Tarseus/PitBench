"""Persistence and upload operations for a harness run."""

from __future__ import annotations

import re
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import boto3
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from pitbench.harness.harness.models import (
    BenchmarkResults,
    RunMetadata,
    TrialResults,
)


def git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def current_user() -> str:
    try:
        git_user = (
            subprocess.check_output(
                ["git", "config", "user.name"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        if git_user:
            return git_user
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    try:
        return (
            subprocess.check_output(["whoami"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


class RunArtifactStore:
    """Own run-level result, metadata, resume, and S3 artifact I/O."""

    def __init__(
        self,
        *,
        run_path: Path,
        run_id: str,
        results_path: Path,
        metadata_path: Path,
        s3_bucket: str | None,
        logger,
        trace: Callable[..., None],
    ) -> None:
        self.run_path = run_path
        self.run_id = run_id
        self.results_path = results_path
        self.metadata_path = metadata_path
        self.s3_bucket = s3_bucket
        self.logger = logger
        self.trace = trace

    def write_results(self, results: BenchmarkResults) -> None:
        self.results_path.write_text(results.model_dump_json(indent=4))
        self.trace(
            stage="results.aggregate",
            status="checkpoint",
            inputs={"trial_results": results.results},
            outputs={
                "benchmark_results": results,
                "results_path": self.results_path,
                "completed_trial_count": len(results.results),
            },
            execution={
                "component": f"{type(self).__module__}.{type(self).__name__}",
                "operation": (
                    "Recompute run-level metrics from all completed trial results and "
                    "persist results.json."
                ),
            },
        )

    def write_metadata(self, metadata: RunMetadata) -> None:
        self.metadata_path.write_text(metadata.model_dump_json(indent=4))

    def finish_metadata(self) -> None:
        if not self.metadata_path.exists():
            return
        try:
            metadata = RunMetadata.model_validate_json(self.metadata_path.read_text())
            metadata.end_time = datetime.now(timezone.utc).isoformat()
            self.write_metadata(metadata)
        except Exception as error:
            self.logger.warning("Failed to update metadata: %s", error)

    @staticmethod
    def _is_multi_agent_sub_trial(trial_name: str) -> bool:
        return bool(re.search(r"\.agent-\d+-", trial_name))

    def load_previous_results(self) -> BenchmarkResults | None:
        if not self.run_path.exists():
            self.logger.warning(
                "Previous run directory %s does not exist", self.run_path
            )
            return None
        all_results = []
        for task_dir in self.run_path.iterdir():
            if not task_dir.is_dir():
                continue
            for trial_dir in task_dir.iterdir():
                if not trial_dir.is_dir() or self._is_multi_agent_sub_trial(
                    trial_dir.name
                ):
                    continue
                result_path = trial_dir / "results.json"
                if not result_path.exists():
                    continue
                try:
                    all_results.append(
                        TrialResults.model_validate_json(result_path.read_text())
                    )
                except Exception as error:
                    self.logger.warning(
                        "Failed to load trial result from %s: %s",
                        result_path,
                        error,
                    )
        if not all_results:
            self.logger.warning("No previous results found to load")
            return None
        self.logger.info(
            "Loaded %s results from individual task directories", len(all_results)
        )
        return BenchmarkResults(results=all_results)

    def filter_completed_tasks(
        self,
        *,
        dataset,
        attempts: int,
        trial_name: Callable[[Path, int], str],
    ) -> None:
        """Keep incomplete tasks and archive any interrupted task artifacts."""
        if not self.run_path.exists():
            self.logger.warning(
                "Resume output directory %s does not exist. Starting a fresh run.",
                self.run_path,
            )
            return
        completed_tasks = set()
        incomplete_tasks = []
        for task_path in dataset._tasks:
            task_id = task_path.name
            task_completed = True
            task_has_partial_artifacts = False
            for attempt in range(1, attempts + 1):
                task_run_path = self.run_path / task_id / trial_name(task_path, attempt)
                if not (task_run_path / "results.json").exists():
                    task_completed = False
                    task_has_partial_artifacts = task_run_path.exists()
                    break
            if task_completed:
                completed_tasks.add(task_id)
                continue
            incomplete_tasks.append(task_path)
            if task_has_partial_artifacts:
                self.logger.info(
                    "Task %s has partial artifacts from previous run. "
                    "Archiving them to prevent data corruption.",
                    task_id,
                )
                self.archive_interrupted_task(task_id)
        dataset._tasks = incomplete_tasks
        if completed_tasks:
            self.logger.info(
                "Resuming run %s. Skipping %s completed tasks: %s",
                self.run_id,
                len(completed_tasks),
                ", ".join(sorted(completed_tasks)),
            )
        if not incomplete_tasks:
            self.logger.info("All tasks have been completed in the previous run.")

    def archive_interrupted_task(self, task_id: str) -> None:
        task_run_path = self.run_path / task_id
        if not task_run_path.exists():
            return
        try:
            archive = self.run_path / "interrupted" / uuid.uuid4().hex / task_id
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(task_run_path), str(archive))
            self.logger.info("Archived interrupted task %s to %s", task_id, archive)
        except Exception as error:
            raise RuntimeError(
                f"Could not archive interrupted evidence for task {task_id}"
            ) from error

    def handle_upload(self, enabled: bool) -> None:
        if not enabled:
            return
        if self.s3_bucket:
            self.upload_to_s3()
        else:
            self.logger.warning(
                "Upload results requested but no S3 bucket configured. "
                "Set the S3_BUCKET_NAME environment variable in .env"
            )

    def upload_to_s3(self) -> None:
        files = [path for path in self.run_path.rglob("*") if path.is_file()]
        if not files:
            self.logger.warning("No files found to upload in %s", self.run_path)
            return
        try:
            s3_client = boto3.client("s3")
            self.logger.info("Uploading run results to S3 bucket: %s", self.s3_bucket)
            failures = []
            with Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
            ) as progress:
                task = progress.add_task("Uploading files to S3", total=len(files))
                for path in files:
                    relative = path.relative_to(self.run_path)
                    key = f"{self.run_id}/{relative}"
                    try:
                        self.logger.info(
                            "Uploading %s to s3://%s/%s",
                            relative,
                            self.s3_bucket,
                            key,
                        )
                        s3_client.upload_file(str(path), self.s3_bucket, key)
                    except Exception as error:
                        self.logger.warning("Failed to upload %s: %s", relative, error)
                        failures.append(str(relative))
                    progress.advance(task)
            if not failures:
                self.logger.info(
                    "Successfully uploaded all %s files to s3://%s/%s/",
                    len(files),
                    self.s3_bucket,
                    self.run_id,
                )
            else:
                self.logger.warning(
                    "Uploaded %s of %s files. %s files failed to upload.",
                    len(files) - len(failures),
                    len(files),
                    len(failures),
                )
                self.logger.warning(
                    "Failed files: %s%s",
                    ", ".join(failures[:5]),
                    "..." if len(failures) > 5 else "",
                )
        except Exception as error:
            self.logger.error(
                "Failed to upload %s results to S3: %s", self.run_path, error
            )
