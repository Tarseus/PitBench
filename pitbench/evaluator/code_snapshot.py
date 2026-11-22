"""Stable evaluator-code snapshots for isolated running measurements."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path


def snapshot_judge_code(package_root: Path, directory: Path) -> Path:
    """Keep code used by a running judge stable while the checkout changes."""
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = Path(tempfile.mkdtemp(prefix="judge-code-", dir=directory))
    for name in ("pitbench", "adapters", "scripts"):
        shutil.copytree(
            package_root / name, snapshot / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.log"),
        )
    return snapshot
