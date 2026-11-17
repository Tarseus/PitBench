from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from pitbench.schema.evaluation import ArtifactRef


class FreshEvaluationRequired(RuntimeError):
    """Retained measurements need a fresh attempt under the current inputs."""


def selected_output_attempt(output: Path) -> Path:
    """Follow the retained attempt selection, keeping the original root usable."""
    root = output.resolve()
    current, seen = root, set()
    while current not in seen:
        seen.add(current)
        try:
            selected = Path(
                json.loads((current / "latest-attempt.json").read_text())["output"]
            ).resolve()
        except (OSError, ValueError, KeyError, TypeError):
            break
        if not selected.is_relative_to(root) or not selected.is_dir():
            break
        current = selected
    return current


def new_output_attempt(output: Path, reason: str) -> Path:
    """Retain prior files in place and select a separate directory automatically."""
    directory = output / "attempts" / uuid.uuid4().hex
    directory.mkdir(parents=True)
    write_json(
        output / "latest-attempt.json",
        {
            "output": str(directory.resolve()),
            "reason": reason,
        },
    )
    return directory


def write_text_atomic(path: Path, payload: str) -> None:
    """Replace one text artifact without exposing a partial destination file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    write_text_atomic(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )


def artifact_ref(
    path: Path,
    *,
    root: Path | None = None,
    media_type: str | None = None,
    private: bool = False,
) -> ArtifactRef:
    payload = path.read_bytes()
    display_path = path
    if root is not None:
        try:
            display_path = path.resolve().relative_to(root.resolve())
        except ValueError:
            display_path = path.resolve()
    return ArtifactRef(
        path=str(display_path),
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        media_type=media_type,
        private=private,
    )


def resolve_saved_candidate(
    source: Path, *, task_id: str, trial_name: str | None = None
) -> tuple[Path, dict[str, Any]]:
    """Select a captured candidate from a run or trial without executing an Agent."""
    source = source.expanduser().resolve()
    manifest = source / "candidate.json"
    if manifest.is_file():
        metadata = json.loads(manifest.read_text())
        if metadata["task_id"] != task_id:
            raise ValueError("saved candidate belongs to another task")
        patch = source / metadata["candidate_patch"]
    elif (source / "results.json").is_file():
        payload = json.loads((source / "results.json").read_text())
        trials = payload.get("results", [payload])
        trials = [row for row in trials if row.get("task_id") == task_id
                  and (trial_name is None or row.get("trial_name") == trial_name)]
        if len(trials) != 1:
            raise ValueError("select exactly one saved attempt with --trial-name")
        trial = trials[0]
        name = trial["trial_name"]
        directory = source / task_id / name / "evaluation"
        if not directory.is_dir():
            directory = source / "evaluation"
        if (directory / "candidate.json").is_file():
            return resolve_saved_candidate(directory, task_id=task_id)
        patch = directory / "candidate.patch"
        metadata = {"task_id": task_id, "trial_name": name, "source_run": str(source)}
        workspace = source.parent.parent / "workspaces" / source.name / task_id / "task.yaml"
        if workspace.is_file():
            import yaml

            metadata["evaluator_config"] = yaml.safe_load(workspace.read_text())["evaluator_config"]
    else:
        raise ValueError("saved run has no results.json or candidate.json")
    patch = patch.resolve()
    if not patch.is_relative_to(source) or not patch.is_file():
        raise ValueError("saved candidate patch is missing or outside the run")
    return patch, metadata
