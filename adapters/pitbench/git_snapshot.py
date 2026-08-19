from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class GitSnapshotError(RuntimeError):
    pass


@dataclass(frozen=True)
class GitSnapshot:
    """Build a repository whose refs and reachable history stop at ``base_commit``."""

    clone_url: str
    base_commit: str

    @staticmethod
    def _run(argv: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise GitSnapshotError(
                f"{' '.join(argv)} failed: {completed.stderr.strip()}"
            )
        return completed

    def create(self, destination: Path, bundle_path: Path | None = None) -> Path:
        if destination.exists():
            raise GitSnapshotError(f"snapshot destination exists: {destination}")
        destination.mkdir(parents=True)
        self._run(["git", "init", "--quiet"], destination)
        self._run(["git", "remote", "add", "origin", self.clone_url], destination)
        try:
            self._run(
                [
                    "git",
                    "fetch",
                    "--quiet",
                    "--no-tags",
                    "origin",
                    self.base_commit,
                ],
                destination,
            )
            self._run(
                ["git", "checkout", "--quiet", "--detach", "FETCH_HEAD"], destination
            )
        finally:
            subprocess.run(
                ["git", "remote", "remove", "origin"],
                cwd=destination,
                check=False,
                capture_output=True,
            )
        self._run(["git", "reflog", "expire", "--expire=now", "--all"], destination)
        self._run(["git", "gc", "--prune=now", "--quiet"], destination)
        self._assert_censored(destination)
        if bundle_path is not None:
            bundle_path.parent.mkdir(parents=True, exist_ok=True)
            self._run(
                ["git", "bundle", "create", str(bundle_path.resolve()), "HEAD"],
                destination,
            )
        return destination

    def _assert_censored(self, repository: Path) -> None:
        head = self._run(["git", "rev-parse", "HEAD"], repository).stdout.strip()
        if head != self.base_commit:
            raise GitSnapshotError(f"snapshot HEAD {head} != base {self.base_commit}")
        refs = self._run(["git", "for-each-ref", "--format=%(refname)"], repository)
        unexpected = [line for line in refs.stdout.splitlines() if line.strip()]
        if unexpected:
            raise GitSnapshotError(f"snapshot contains named refs: {unexpected}")


def clone_bundle(bundle: Path, destination: Path) -> Path:
    if destination.exists():
        raise GitSnapshotError(f"clone destination exists: {destination}")
    completed = subprocess.run(
        ["git", "clone", "--quiet", str(bundle), str(destination)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        shutil.rmtree(destination, ignore_errors=True)
        raise GitSnapshotError(completed.stderr.strip())
    return destination
