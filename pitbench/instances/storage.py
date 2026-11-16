"""Provision public input bundles kept outside version control."""

from __future__ import annotations

import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen


def prepare_public_inputs(source: str, output: Path) -> int:
    """Import a tar bundle from a local path or a direct HTTP(S) download URL."""
    with tempfile.TemporaryDirectory(prefix="pitbench-inputs-") as temporary:
        temporary = Path(temporary)
        if source.startswith(("https://", "http://")):
            archive_path = temporary / "inputs.tar"
            with (
                urlopen(source, timeout=60) as response,
                archive_path.open("wb") as target,
            ):
                shutil.copyfileobj(response, target)
        else:
            archive_path = Path(source).expanduser()
        staged = temporary / "inputs"
        with tarfile.open(archive_path, "r:*") as archive:
            members = archive.getmembers()
            if any(not (member.isfile() or member.isdir()) for member in members):
                raise ValueError(
                    "public input bundles must contain regular files and directories"
                )
            archive.extractall(staged, members=members, filter="data")
        files = sum(path.is_file() for path in staged.rglob("*"))
        if not files:
            raise ValueError("public input bundle contains no files")
        shutil.copytree(staged, output, dirs_exist_ok=True)
        return files
