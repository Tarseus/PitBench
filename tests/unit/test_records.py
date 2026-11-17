"""Functional unit/component contracts consolidated by responsibility."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from pitbench.harness.utils import artifact_storage
from pitbench.harness.utils.artifact_storage import export_campaign_records
from pitbench.metrics.campaign_report import collect_campaign_diagnostics


def test_evaluator_artifacts_allow_concurrent_atomic_writers(tmp_path, monkeypatch):
    import json
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from pitbench.evaluator.artifacts import write_json

    path = tmp_path / "status.json"
    barrier = Barrier(2)
    replace = Path.replace

    def together(source, target):
        if target == path:
            barrier.wait(timeout=5)
        return replace(source, target)

    monkeypatch.setattr(Path, "replace", together)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda value: write_json(path, {"writer": value}), [1, 2]))
    assert json.loads(path.read_text())["writer"] in {1, 2}
    assert not list(tmp_path.glob("*.tmp"))


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


@pytest.fixture(params=[False, True], ids=["inline-data", "external-data"])
def campaign_records(tmp_path, request):
    project = tmp_path / "project"
    project.mkdir()
    data = tmp_path / "data" if request.param else project
    campaign = data / "campaign"
    runs = data / "runs"
    campaign.mkdir(parents=True)
    rows = [
        {"condition_id": name, "run_id": name, "task_id": "task", "outcome": status}
        for name, status in [
            ("ok", "completed"),
            ("failed", "terminal_failure"),
            ("missing", "not_started"),
        ]
    ]
    (campaign / "final-model-audit-summary.json").write_text(
        json.dumps({"conditions": rows})
    )
    for name in ["ok", "failed", "failed-infra1"]:
        run = runs / name
        run.mkdir(parents=True)
        (run / "results.json").write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "task_id": "task",
                            "trial_name": name,
                            "failure_mode": "failed" if name != "ok" else "unset",
                        }
                    ]
                }
            )
        )
        (run / "raw.log").write_text(name)
    trace = runs / "failed/task/trial/agent-trace/execution"
    (trace / "objects").mkdir(parents=True)
    content = json.dumps({"failure": "retained"}).encode()
    digest = hashlib.sha256(content).hexdigest()
    (trace / "objects" / digest).write_bytes(content)
    (trace / "events.jsonl").write_text(
        json.dumps(
            {
                "event": "execution.finished",
                "sequence": 1,
                "timestamp": "2026-01-01T00:00:00Z",
                "data": {
                    "path": "objects/" + digest,
                    "sha256": digest,
                    "bytes": len(content),
                },
            }
        )
        + "\n"
    )
    return project, campaign, runs


def test_diagnostics_keep_failed_and_missing_conditions(campaign_records, tmp_path):
    project, campaign, runs = campaign_records
    report = collect_campaign_diagnostics(
        project, "example", runs, campaign, tmp_path / "diagnostics"
    )
    assert len(report["conditions"]) == 3
    missing = next(
        row for row in report["conditions"] if row["condition"]["run_id"] == "missing"
    )
    failed = next(
        row for row in report["conditions"] if row["condition"]["run_id"] == "failed"
    )
    assert missing["missing_result"]
    assert failed["trace"]["traces"][0]["execution_end_events"] == [
        {"failure": "retained"}
    ]


@pytest.mark.skipif(not shutil.which("zstd"), reason="requires zstd")
def test_full_export_includes_failure_trajectories_infra_attempts_and_missing_metadata(
    campaign_records, tmp_path
):
    project, campaign, runs = campaign_records
    data = campaign.parent
    originals = {
        p.relative_to(data): p.read_bytes() for p in data.rglob("*") if p.is_file()
    }
    output = tmp_path / "export"
    result = export_campaign_records(project, "example", runs, campaign, output)
    manifest = json.loads((output / "EXPORT_MANIFEST.json").read_text())
    assert manifest["planned_condition_count"] == 3
    assert manifest["completed_condition_count"] == 1
    assert manifest["missing_run_ids"] == ["missing"]
    assert manifest["earlier_attempt_count"] == 1
    assert manifest["trace_execution_count"] == 1
    names = subprocess.check_output(
        ["tar", "--zstd", "-tf", result["archive"]], text=True
    )
    assert "runs/failed/raw.log" in names and "runs/failed-infra1/raw.log" in names
    assert "runs/failed/task/trial/agent-trace/execution/objects/" in names
    assert originals == {
        p.relative_to(data): p.read_bytes() for p in data.rglob("*") if p.is_file()
    }


def test_saved_candidate_selects_legacy_attempt_and_requires_explicit_trial(tmp_path):
    from pitbench.evaluator.artifacts import resolve_saved_candidate, write_json

    trials = [{"task_id": "task", "trial_name": "one"}, {"task_id": "task", "trial_name": "two"}]
    write_json(tmp_path / "results.json", {"results": trials})
    patch = tmp_path / "task/one/evaluation/candidate.patch"
    patch.parent.mkdir(parents=True)
    patch.write_bytes(b"")
    with pytest.raises(ValueError, match="--trial-name"):
        resolve_saved_candidate(tmp_path, task_id="task")
    selected, metadata = resolve_saved_candidate(tmp_path, task_id="task", trial_name="one")
    assert selected == patch
    assert metadata["trial_name"] == "one"


def test_saved_candidate_manifest_and_task_identity(tmp_path):
    from pitbench.evaluator.artifacts import resolve_saved_candidate, write_json

    patch = tmp_path / "candidate.patch"
    patch.write_text("retained patch")
    write_json(tmp_path / "candidate.json", {
        "task_id": "task", "candidate_patch": "candidate.patch", "source_commit": "abc"
    })
    selected, metadata = resolve_saved_candidate(tmp_path, task_id="task")
    assert selected == patch
    assert metadata["source_commit"] == "abc"
    with pytest.raises(ValueError, match="another task"):
        resolve_saved_candidate(tmp_path, task_id="different")
