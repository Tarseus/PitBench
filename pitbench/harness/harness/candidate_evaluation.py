"""Candidate capture, independent evaluation, and snapshot handling."""

from __future__ import annotations

import hashlib
from functools import partial
from pathlib import Path
from typing import Any, Callable

from pitbench.harness.evaluation import EvaluationRequest, EvaluatorFactory
from pitbench.harness.handlers.trial_handler import TrialHandler
from pitbench.harness.harness.models import TrialResults
from pitbench.harness.utils.repository import capture_repository_patch


def trace_file_input(path: Path) -> dict[str, Any]:
    """Describe a pipeline input file, including readable text."""
    file_input: dict[str, Any] = {"path": path, "exists": path.exists()}
    if path.is_file():
        try:
            file_input["content"] = path.read_text(errors="replace")
        except OSError as error:
            file_input["read_error"] = error
    return file_input


def trace_artifact(path: Path, *, include_text: bool = True) -> dict[str, Any]:
    """Describe an output artifact without decoding binary content."""
    artifact: dict[str, Any] = {"path": path, "exists": path.exists()}
    if not path.is_file():
        return artifact
    try:
        payload = path.read_bytes()
    except OSError as error:
        artifact["read_error"] = error
        return artifact
    artifact.update(size_bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    if include_text:
        try:
            artifact["content"] = payload.decode("utf-8")
        except UnicodeDecodeError:
            artifact["content"] = None
    return artifact


class CandidateEvaluator:
    """Own candidate state capture and evaluator-side artifact lifecycle."""

    def __init__(
        self,
        *,
        defer_evaluation: bool,
        remote_build: bool,
        snapshot_bucket: str | None,
        logger,
        trace: Callable[..., None],
        progress_stage: Callable[..., None],
        progress_detail: Callable[..., None],
    ) -> None:
        self.defer_evaluation = defer_evaluation
        self.remote_build = remote_build
        self.snapshot_bucket = snapshot_bucket
        self.logger = logger
        self.trace = trace
        self.progress_stage = progress_stage
        self.progress_detail = progress_detail

    @staticmethod
    def repository_head(terminal) -> str:
        container = terminal.container
        if container is None:
            raise RuntimeError("agent container is not running")
        working_dir = container.attrs.get("Config", {}).get("WorkingDir") or None
        result = container.exec_run(["git", "rev-parse", "HEAD"], workdir=working_dir)
        if result.exit_code != 0:
            detail = result.output.decode("utf-8", errors="replace")
            raise RuntimeError(f"repository HEAD capture failed: {detail}")
        if not isinstance(result.output, bytes):
            raise TypeError("container returned a non-bytes repository HEAD")
        head = result.output.decode("ascii").strip()
        if not head:
            raise RuntimeError("repository HEAD capture returned an empty value")
        return head

    def evaluate(
        self,
        *,
        terminal,
        trial_handler: TrialHandler,
        results: TrialResults,
        expected_repository_head: str,
        agent_label: str,
        model_name: str | None,
    ) -> None:
        self.progress_stage(
            stage="candidate.capture",
            status="started",
            task_id=trial_handler.task_id,
            trial_name=trial_handler.trial_name,
        )
        output_dir = trial_handler.trial_paths.task_output_path / "evaluation"
        output_dir.mkdir(parents=True, exist_ok=True)
        candidate_patch = output_dir / "candidate.patch"
        container = terminal.container
        if container is None:
            raise RuntimeError("agent container is not running")
        working_dir = container.attrs.get("Config", {}).get("WorkingDir") or None
        actual_repository_head = self.repository_head(terminal)
        if actual_repository_head != expected_repository_head:
            raise RuntimeError(
                "agent changed repository HEAD: "
                f"{actual_repository_head} != {expected_repository_head}"
            )
        payload = capture_repository_patch(container, workdir=working_dir)
        candidate_patch_sha256 = hashlib.sha256(payload).hexdigest()
        candidate_patch.write_bytes(payload)

        if self.defer_evaluation:
            self.trace(
                stage="candidate.capture",
                status="completed",
                outputs={"candidate_patch": trace_artifact(candidate_patch)},
                task_id=trial_handler.task_id,
                trial_name=trial_handler.trial_name,
            )
            return

        evaluator = EvaluatorFactory.from_import_path(
            trial_handler.task.evaluator_import_path
        )
        request = EvaluationRequest(
            task_id=trial_handler.task_id,
            task_path=trial_handler.task_paths.input_path,
            candidate_patch_path=candidate_patch,
            candidate_patch_sha256=candidate_patch_sha256,
            output_dir=output_dir,
            agent_name=agent_label,
            model_name=model_name,
            evaluator_config={
                **trial_handler.task.evaluator_config,
                "_progress_callback": partial(
                    self.progress_detail,
                    trial_handler.task_id,
                    trial_handler.trial_name,
                ),
            },
            telemetry={
                "total_input_tokens": results.total_input_tokens,
                "total_output_tokens": results.total_output_tokens,
                "total_cost": results.total_cost,
            },
        )
        self.trace(
            stage="evaluator.execute",
            status="started",
            inputs={
                "evaluator_import_path": trial_handler.task.evaluator_import_path,
                "candidate_patch_path": candidate_patch,
            },
            task_id=trial_handler.task_id,
            trial_name=trial_handler.trial_name,
        )
        try:
            results.evaluation = evaluator.envelope(request)
        except Exception as error:
            self.trace(
                stage="evaluator.execute",
                status="failed",
                task_id=trial_handler.task_id,
                trial_name=trial_handler.trial_name,
                error=error,
            )
            raise
        self.trace(
            stage="evaluator.execute",
            status="completed" if results.evaluation.completed else "failed",
            inputs={
                "evaluator_import_path": trial_handler.task.evaluator_import_path,
                "candidate_patch": trace_artifact(candidate_patch),
            },
            outputs={"evaluation": results.evaluation},
            execution={
                "component": "pitbench.harness.evaluation.Evaluator",
                "operation": "Delegate evaluation and persist opaque payload",
            },
            task_id=trial_handler.task_id,
            trial_name=trial_handler.trial_name,
        )

    def maybe_save_snapshot(
        self,
        *,
        terminal,
        trial_handler: TrialHandler,
        results: TrialResults,
        snapshot_name: str,
        snapshot_s3_key: str,
        reason: str,
    ) -> None:
        if not self.remote_build:
            return
        if not self.snapshot_bucket:
            self.logger.info(
                "Skipping evaluation snapshot (%s) for task %s. "
                "Set S3_EVALUATION_SNAPSHOTS_BUCKET_NAME to enable snapshot uploads.",
                reason,
                trial_handler.task_id,
            )
            return
        try:
            self.logger.info(
                "Saving evaluation snapshot (%s) for task %s as S3 key %s",
                reason,
                trial_handler.task_id,
                snapshot_s3_key,
            )
            terminal.save_container_image(snapshot_s3_key=snapshot_s3_key)
            results.evaluation_snapshot_bucket_name = self.snapshot_bucket
            results.evaluation_snapshot_s3_keys[snapshot_name] = snapshot_s3_key
            self.logger.info(
                "Evaluation snapshot saved to S3: s3://%s/%s",
                self.snapshot_bucket,
                snapshot_s3_key,
            )
        except Exception as error:
            self.logger.warning(
                "Failed to save evaluation snapshot for task %s: %s",
                trial_handler.task_id,
                error,
                exc_info=True,
            )
