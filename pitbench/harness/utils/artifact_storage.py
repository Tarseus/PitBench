"""Lossless maintenance of retained experiment artifacts."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Callable, Iterable


def trace_directories(roots: Iterable[Path]) -> list[Path]:
    traces = set()
    for root in roots:
        for directory, children, files in os.walk(root, followlinks=False):
            path = Path(directory)
            if "events.jsonl" in files and (path / "objects").is_dir():
                traces.add(path.resolve())
                children[:] = []
            else:
                children[:] = [
                    name
                    for name in children
                    if name not in {".git", ".venv", "node_modules", "__pycache__"}
                    and not (path / name).is_symlink()
                ]
    return sorted(traces)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def deduplicate_objects(
    traces: Iterable[Path], *, progress: Callable[[dict], None] | None = None
) -> dict:
    """Hard-link identical immutable trace objects, checking bytes before replacement."""
    canonical = {}
    verified = set()
    linked = 0
    released = 0
    for trace in traces:
        for directory in (trace / "objects", trace / "native-inbox" / "objects"):
            if not directory.is_dir():
                continue
            changed = False
            for path in sorted(directory.iterdir()):
                if path.is_symlink() or not re.fullmatch("[0-9a-f]{64}", path.name):
                    continue
                current = path.stat()
                key = (current.st_dev, path.name)
                original = canonical.setdefault(key, path)
                if original == path or original.stat().st_ino == current.st_ino:
                    continue
                if key not in verified:
                    if _digest(original) != path.name:
                        raise ValueError(f"corrupt trace object: {original}")
                    verified.add(key)
                if _digest(path) != path.name:
                    raise ValueError(f"corrupt trace object: {path}")
                temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
                try:
                    os.link(original, temporary)
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
                linked += 1
                if current.st_nlink == 1:
                    released += current.st_blocks * 512
                changed = True
            if changed:
                descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        if progress:
            progress(
                {"trace": str(trace), "linked": linked, "released_bytes": released}
            )
    return {"linked": linked, "released_bytes": released}


def _tree_signature(directory: Path) -> str:
    """Detect files added, removed, or changed while an archive is being made."""
    digest = hashlib.sha256()
    for parent, children, files in os.walk(directory, followlinks=False):
        children.sort()
        for name in [".", *sorted(children + files)]:
            path = Path(parent) / name
            info = path.lstat()
            digest.update(os.fsencode(str(path.relative_to(directory))))
            digest.update(
                f"\0{info.st_mode}:{info.st_size}:{info.st_mtime_ns}\0".encode()
            )
            if path.is_symlink():
                digest.update(os.fsencode(os.readlink(path)))
    return digest.hexdigest()


def archive_directory(directory: Path) -> dict:
    """Verify a complete tar.zst against its source before removing loose files."""
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"expected a directory, not a file or symlink: {directory}")
    directory = directory.resolve()
    destination = directory.with_name(directory.name + ".tar.zst")
    if destination.exists():
        raise FileExistsError(destination)
    if shutil.which("tar") is None or shutil.which("zstd") is None:
        raise RuntimeError("archiving requires the tar and zstd commands")
    signature = _tree_signature(directory)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        subprocess.run(
            [
                "tar",
                "--zstd",
                "-cf",
                str(temporary),
                "-C",
                str(directory.parent),
                "--",
                directory.name,
            ],
            check=True,
        )
        subprocess.run(
            [
                "tar",
                "--zstd",
                "--compare",
                "-f",
                str(temporary),
                "-C",
                str(directory.parent),
            ],
            check=True,
        )
        if _tree_signature(directory) != signature:
            raise RuntimeError(f"artifacts changed while archiving: {directory}")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    result = {
        "source": str(directory),
        "archive": str(destination),
        "archive_bytes": destination.stat().st_size,
    }
    try:
        shutil.rmtree(directory)
        result["source_removed"] = True
    except OSError as error:
        # The verified archive remains available even if container-owned files
        # require an administrator to finish removing the loose directory.
        result.update(source_removed=False, cleanup_error=str(error))
    return result


def restore_archive(archive: Path) -> Path:
    """Restore one retained directory without overwriting existing artifacts."""
    archive = archive.resolve()
    if not archive.name.endswith(".tar.zst"):
        raise ValueError("expected a .tar.zst archive")
    destination = archive.with_name(archive.name[:-8])
    if destination.exists():
        raise FileExistsError(destination)
    with tempfile.TemporaryDirectory(
        prefix=".restore-", dir=archive.parent
    ) as temporary:
        subprocess.run(
            ["tar", "--zstd", "-xf", str(archive), "-C", temporary], check=True
        )
        os.replace(Path(temporary) / destination.name, destination)
    return destination
