from __future__ import annotations

import hashlib
from pathlib import Path

from pitbench.schema.evaluation import ArtifactRef


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
