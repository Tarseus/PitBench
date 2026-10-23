from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pitbench.schema.evaluation import ArtifactRef


def write_text_atomic(path: Path, payload: str) -> None:
    """Replace one text artifact without exposing a partial destination file."""
    write_bytes_atomic(path, payload.encode())


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    """Replace an artifact without following an old destination/temp symlink."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=".pitbench-", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            os.fchmod(handle.fileno(), 0o644)
            handle.write(payload)
            handle.flush()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def output_directory(root: Path, relative: Path) -> Path:
    """Create a trusted artifact directory, rejecting old symlink components."""
    root.mkdir(parents=True, exist_ok=True)
    current = root.resolve()
    if relative.is_absolute():
        raise OSError("absolute artifact directory is not allowed")
    for component in relative.parts:
        current = current / component
        if component in {".", ".."} or current.is_symlink():
            raise OSError("unsafe existing output directory")
        current.mkdir(exist_ok=True)
    return current


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
