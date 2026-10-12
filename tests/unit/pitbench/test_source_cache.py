from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from adapters.pitbench.adapter import PitBenchAdapter
from pitbench.cli.evaluate_config import EvaluationConfig


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=repository, text=True
    ).strip()


def test_source_override_is_rejected_by_evaluation_config() -> None:
    with pytest.raises(ValidationError, match="repository_source"):
        EvaluationConfig.model_validate(
            {"tasks": {"solver": {"repository_source": "/tmp/solver"}}}
        )


def test_pinned_source_is_fetched_once_then_reused_without_origin(
    tmp_path: Path,
) -> None:
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "--quiet")
    _git(origin, "config", "user.name", "PitBench Test")
    _git(origin, "config", "user.email", "pitbench@example.invalid")
    (origin / "solver.txt").write_text("release source\n")
    _git(origin, "add", "solver.txt")
    _git(origin, "commit", "--quiet", "-m", "release")
    commit = _git(origin, "rev-parse", "HEAD")
    task = SimpleNamespace(
        task_id="test_solver",
        release=SimpleNamespace(base_commit=commit, tree_sha=None),
        repository=SimpleNamespace(clone_url=str(origin)),
    )
    adapter = PitBenchAdapter(tmp_path, tmp_path / "private")
    first = tmp_path / "first"
    adapter._materialize_cached_repository(task, first, tmp_path / "first.bundle")
    cache = adapter.source_cache_path(tmp_path, task)
    assert cache.is_dir()
    assert _git(first, "rev-parse", "HEAD") == commit
    assert (tmp_path / "first.bundle").is_file()

    origin.rename(tmp_path / "origin-offline")
    second = tmp_path / "second"
    adapter._materialize_cached_repository(task, second, tmp_path / "second.bundle")
    assert (second / "solver.txt").read_text() == "release source\n"
    assert (tmp_path / "second.bundle").is_file()

    (cache / "untracked.txt").write_text("unexpected\n")
    with pytest.raises(ValueError, match="local modifications"):
        adapter._materialize_cached_repository(
            task, tmp_path / "third", tmp_path / "third.bundle"
        )
