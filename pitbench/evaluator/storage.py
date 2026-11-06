from __future__ import annotations

import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

from pitbench.schema.observation import RunObservation


@contextmanager
def cache_lock(path: Path):
    """Serialize publishers and readers of one evaluator-owned cache entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def file_manifest(root: Path, *, ignore: tuple[str, ...] = ()) -> dict[str, str]:
    """Hash retained files, excluding interpreter/test caches that grow on use."""
    result = {}
    for directory, dirs, files in os.walk(root):
        dirs[:] = sorted(
            d for d in dirs if d not in {".git", "__pycache__", ".pytest_cache"}
        )
        for name in sorted(
            files + [d for d in dirs if (Path(directory) / d).is_symlink()]
        ):
            path = Path(directory) / name
            if path.suffix == ".pyc" or path.relative_to(root).as_posix() in ignore:
                continue
            data = (
                os.readlink(path).encode() if path.is_symlink() else path.read_bytes()
            )
            result[path.relative_to(root).as_posix()] = hashlib.sha256(
                str(path.lstat().st_mode & 0o777).encode() + b"\0" + data
            ).hexdigest()
    return result


class ObservationStore:
    """Canonical columnar storage for raw run observations."""

    @staticmethod
    def write(path: Path, observations: list[RunObservation]) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [item.model_dump(mode="json") for item in observations]
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, path, compression="zstd")

    @staticmethod
    def read(path: Path) -> list[RunObservation]:
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        return [RunObservation.model_validate(row) for row in table.to_pylist()]

    @staticmethod
    def write_jsonl(path: Path, observations: list[RunObservation]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w") as handle:
            for observation in observations:
                handle.write(observation.model_dump_json())
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)

    @staticmethod
    def read_jsonl(path: Path) -> list[RunObservation]:
        return [
            RunObservation.model_validate_json(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]

    @staticmethod
    def schema_json(path: Path) -> None:
        path.write_text(json.dumps(RunObservation.model_json_schema(), indent=2))
