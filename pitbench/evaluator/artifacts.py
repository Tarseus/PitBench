from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pitbench.schema.evaluation import ArtifactRef


def write_text_atomic(path: Path, payload: str) -> None:
    """Replace one text artifact without exposing a partial destination file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload)
    temporary.replace(path)


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
