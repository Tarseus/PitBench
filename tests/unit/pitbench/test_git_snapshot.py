import subprocess
from pathlib import Path

from adapters.pitbench.git_snapshot import GitSnapshot


def _git(repository: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repository, text=True).strip()


def test_snapshot_stops_at_release_commit(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "--quiet")
    _git(source, "config", "user.name", "PitBench Test")
    _git(source, "config", "user.email", "pitbench@example.invalid")
    (source / "solver.py").write_text("BASE = True\n")
    _git(source, "add", "solver.py")
    _git(source, "commit", "--quiet", "-m", "base")
    base = _git(source, "rev-parse", "HEAD")
    (source / "solver.py").write_text("FUTURE = True\n")
    _git(source, "commit", "--quiet", "-am", "future")
    future = _git(source, "rev-parse", "HEAD")

    snapshot = tmp_path / "snapshot"
    bundle = tmp_path / "agent_repo.bundle"
    GitSnapshot(clone_url=str(source), base_commit=base).create(snapshot, bundle)

    assert _git(snapshot, "rev-parse", "HEAD") == base
    assert _git(snapshot, "remote") == ""
    assert bundle.is_file()
    hidden = subprocess.run(
        ["git", "cat-file", "-e", f"{future}^{{commit}}"], cwd=snapshot, check=False
    )
    assert hidden.returncode != 0
