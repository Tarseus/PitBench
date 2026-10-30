from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from pitbench.harness.utils import artifact_storage


def test_deduplication_preserves_paths_and_checks_duplicate_contents(tmp_path):
    content = b"immutable evidence"
    digest = hashlib.sha256(content).hexdigest()
    traces = [tmp_path / "first", tmp_path / "second"]
    for trace in traces:
        (trace / "objects").mkdir(parents=True)
        (trace / "objects" / digest).write_bytes(content)
    paths = [trace / "objects" / digest for trace in traces]
    inbox = traces[1] / "native-inbox" / "objects"
    inbox.mkdir(parents=True)
    (inbox / digest).write_bytes(content)
    assert paths[0].stat().st_ino != paths[1].stat().st_ino
    assert artifact_storage.deduplicate_objects(traces)["linked"] == 2
    assert paths[0].stat().st_ino == paths[1].stat().st_ino
    assert (inbox / digest).stat().st_ino == paths[0].stat().st_ino
    assert paths[1].read_bytes() == content
    paths[1].unlink()
    paths[1].write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt trace object"):
        artifact_storage.deduplicate_objects(traces)
    assert paths[1].read_bytes() == b"corrupt"
    assert paths[0].read_bytes() == content


@pytest.mark.skipif(not shutil.which("zstd"), reason="requires zstd")
def test_archive_restore_keeps_all_records_links_and_failures(tmp_path):
    source = tmp_path / "records"
    source.mkdir()
    (source / "failure.json").write_text('{"status":"failed","missing":[1]}\n')
    (source / "raw.bin").write_bytes(bytes(range(256)) * 100)
    (source / "empty").mkdir()
    (source / "link").symlink_to("raw.bin")
    os.link(source / "raw.bin", source / "same.bin")
    result = artifact_storage.archive_directory(source)
    assert result["source_removed"]
    assert not source.exists()
    archive = Path(result["archive"])
    assert stat.S_IMODE(archive.stat().st_mode) == 0o600
    restored = artifact_storage.restore_archive(archive)
    assert restored == source
    assert (source / "raw.bin").read_bytes() == bytes(range(256)) * 100
    assert (source / "same.bin").stat().st_ino == (source / "raw.bin").stat().st_ino
    assert (source / "link").is_symlink()
    assert (
        source / "failure.json"
    ).read_text() == '{"status":"failed","missing":[1]}\n'
    assert (source / "empty").is_dir()
    with pytest.raises(FileExistsError):
        artifact_storage.restore_archive(archive)


@pytest.mark.skipif(not shutil.which("zstd"), reason="requires zstd")
def test_archive_verification_failure_preserves_source(tmp_path, monkeypatch):
    source = tmp_path / "records"
    source.mkdir()
    raw = source / "raw.txt"
    raw.write_text("original")
    run = subprocess.run

    def change_after_packing(args, **kwargs):
        result = run(args, **kwargs)
        if "-cf" in args:
            raw.write_text("changed after packing")
        return result

    monkeypatch.setattr(artifact_storage.subprocess, "run", change_after_packing)
    with pytest.raises(subprocess.CalledProcessError):
        artifact_storage.archive_directory(source)
    assert raw.read_text() == "changed after packing"
    assert not source.with_name("records.tar.zst").exists()
