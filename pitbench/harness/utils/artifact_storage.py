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


def file_identity(path: Path, root: Path) -> dict:
    display = path.relative_to(root) if path.is_relative_to(root) else path
    return {"path": str(display), "bytes": path.stat().st_size, "sha256": _digest(path)}


def create_verified_archive(
    groups, destination: Path, *, log_directory: Path | None = None, exclude=()
) -> None:
    """One tar/zstd writer and comparison path for compaction and exports."""
    from contextlib import ExitStack

    if shutil.which("tar") is None or shutil.which("zstd") is None:
        raise RuntimeError("archiving requires the tar and zstd commands")
    options = ["--exclude=" + pattern for pattern in exclude]
    command = [
        "tar",
        "--format=gnu",
        "--use-compress-program=zstd -T2 -3",
        *options,
        "-cf",
        str(destination),
    ]
    for root, paths in groups:
        command.extend(["-C", str(root), *["./" + str(path) for path in paths]])
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    with ExitStack() as stack:
        create_log = (
            stack.enter_context((log_directory / "archive-create.log").open("w"))
            if log_directory
            else None
        )
        verify_log = (
            stack.enter_context((log_directory / "archive-verify.log").open("w"))
            if log_directory
            else None
        )
        subprocess.run(command, check=True, stdout=create_log, stderr=create_log)
        for root, paths in groups:
            subprocess.run(
                [
                    "tar",
                    "--use-compress-program=zstd -T2 -d",
                    *options,
                    "--compare",
                    "-f",
                    str(destination),
                    "-C",
                    str(root),
                    *["./" + str(path) for path in paths],
                ],
                check=True,
                stdout=verify_log,
                stderr=verify_log,
            )


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
        create_verified_archive([(directory.parent, [Path(directory.name)])], temporary)
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


def export_campaign_records(
    project: Path,
    campaign_name: str,
    runs_directory: Path,
    campaign_directory: Path,
    output: Path,
) -> dict:
    """Non-destructively export all outcomes, earlier attempts and complete evidence."""
    import collections
    import datetime
    import json
    from types import SimpleNamespace

    from pitbench.evaluator.artifacts import write_json
    from pitbench.metrics.campaign_report import retained_conditions

    args = SimpleNamespace(
        project=project,
        campaign=campaign_name,
        runs_directory=runs_directory,
        campaign_directory=campaign_directory,
        output=output,
    )
    project_root = args.project.resolve()
    root = project_root
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    output.chmod(0o700)
    campaign = (root / args.campaign_directory).resolve()
    runs_directory = (root / args.runs_directory).resolve()
    root = Path(
        os.path.commonpath([str(project_root), str(campaign), str(runs_directory)])
    )
    audit_path = campaign / "final-model-audit-summary.json"
    if output.is_relative_to(campaign) or output.is_relative_to(runs_directory):
        raise ValueError("export output must be outside its source campaign and runs")
    rows = retained_conditions(campaign, runs_directory)
    final_ids = {row.get("execution_run_id") or row.get("run_id") for row in rows}
    final_ids.discard(None)
    run_paths = (
        sorted(path for path in runs_directory.iterdir() if path.is_dir())
        if runs_directory.is_dir()
        else []
    )
    missing_ids = sorted(final_ids - {path.name for path in run_paths})
    existing_final_ids = final_ids & {path.name for path in run_paths}
    completed_count = sum(
        row.get("outcome", row.get("status")) in {"completed", "evaluation_completed"}
        for row in rows
    )
    name = f"pitbench-{len(rows)}-experiments-{datetime.datetime.now(datetime.timezone.utc):%Y%m%d}"
    archive = output / (name + ".tar.zst")
    partial = archive.with_suffix(archive.suffix + ".partial")
    if archive.exists() or partial.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing archive or partial export"
        )

    # Health sampling continues after model runs finish; freeze this context only.
    snapshot_root = output / "context-snapshot"
    snapshot_names = []
    snapshot_details = []
    for source in sorted(campaign.glob("host-health*.json*")):
        relative = source.relative_to(root)
        target = snapshot_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        snapshot_names.append(str(relative))
        snapshot_details.append(
            {
                "path": str(relative),
                "bytes": target.stat().st_size,
                "sha256": _digest(target),
                "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        )

    def status(phase, **details):
        state = {
            "phase": phase,
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            **details,
        }
        write_json(output / "export-status.json", state)
        print(json.dumps(state, ensure_ascii=False), flush=True)

    includes = [path.relative_to(root) for path in run_paths]
    includes.extend(
        child.relative_to(root)
        for child in sorted(campaign.iterdir())
        if child.resolve() != runs_directory
        and child.name not in {"workspaces", "shared-cache", "judge-tmp"}
    )
    if (project_root / "configs/benchmarks").exists():
        includes.append((project_root / "configs/benchmarks").relative_to(root))
    includes.extend(
        path.relative_to(root)
        for row in rows
        if row.get("task_id")
        for path in [project_root / "configs/tasks" / (row["task_id"] + ".yaml")]
        if path.is_file()
    )
    optional = [
        "configs/experiments/gpt_6_1_matrix.yaml",
        "configs/experiments/model_matrix.yaml",
        "AGENT_TRACING.md",
        "pitbench/harness/utils/agent_trace.py",
        "pitbench/harness/utils/recording.py",
        "pitbench/harness/utils/trace_validation.py",
    ]
    includes.extend(
        (project_root / path).relative_to(root)
        for path in optional
        if (project_root / path).exists()
    )
    write_json(output / "selected-paths.json", [str(path) for path in includes])
    status(
        "inventory",
        planned_conditions=len(rows),
        completed_conditions=completed_count,
        run_directories=len(run_paths),
    )
    inventory = []
    traces = []
    for relative in includes:
        source = root / relative
        files = 0
        size = 0
        directories = 0
        if source.is_dir():
            for directory, dirs, names in os.walk(source):
                dirs[:] = [
                    name
                    for name in dirs
                    if name
                    not in {"workspaces", "shared-cache", "judge-tmp", "__pycache__"}
                ]
                directories += 1
                for entry in dirs:
                    if (Path(directory) / entry).is_symlink():
                        raise RuntimeError(f"Unexpected symlink: {directory}/{entry}")
                for entry in names:
                    path = Path(directory) / entry
                    if path.is_symlink() or not path.is_file():
                        raise RuntimeError(f"Unexpected non-regular file: {path}")
                    files += 1
                    snapshot = snapshot_root / path.relative_to(root)
                    size += (
                        (
                            snapshot
                            if str(path.relative_to(root)) in snapshot_names
                            else path
                        )
                        .stat()
                        .st_size
                    )
                    if (
                        entry == "events.jsonl"
                        and path.parent.parent.name == "agent-trace"
                    ):
                        traces.append(path)
        else:
            files = 1
            size = source.stat().st_size
        inventory.append(
            {
                "path": str(relative),
                "files": files,
                "directories": directories,
                "bytes": size,
            }
        )

    status("checking_trajectory_references", traces=len(traces))
    trace_reports = []
    for trace in traces:
        coverage_path = trace.parent / "coverage.json"
        coverage = (
            json.loads(coverage_path.read_text()) if coverage_path.exists() else {}
        )
        count = 0
        artifact_count = 0
        artifacts_verified = 0
        artifact_differences = []
        from pitbench.harness.utils.agent_trace import read_trace

        with trace.open("rb") as index:
            index.seek(0, 2)
            size = index.tell()
            if size:
                index.seek(-1, 2)
            incomplete_tail = bool(size) and index.read(1) != b"\n"
        for event in read_trace(trace):
            count += 1
            if event["event"] == "source.artifact":
                artifact_count += 1
                artifact = event["data"]
                source = Path(artifact["retained_at"])
                if not source.exists():
                    source = (
                        trace.parent.parent.parent
                        / artifact["source"]
                        / artifact["path"]
                    )
                if not source.is_file() or not any(
                    source.is_relative_to(path) for path in run_paths
                ):
                    raise RuntimeError(
                        f"Retained source artifact outside exported runs: {source}"
                    )
                actual_size = source.stat().st_size
                actual_hash = _digest(source)
                exact_match = (
                    actual_size == artifact["bytes"]
                    and actual_hash == artifact["sha256"]
                )
                prefix_match = False
                if actual_size > artifact["bytes"]:
                    with source.open("rb") as stream:
                        prefix_match = (
                            hashlib.sha256(stream.read(artifact["bytes"])).hexdigest()
                            == artifact["sha256"]
                        )
                if exact_match or prefix_match:
                    artifacts_verified += 1
                if not exact_match:
                    artifact_differences.append(
                        {
                            "path": str(source.relative_to(root)),
                            "recorded_bytes": artifact["bytes"],
                            "recorded_sha256": artifact["sha256"],
                            "exported_bytes": actual_size,
                            "exported_sha256": actual_hash,
                            "recorded_prefix_verified": prefix_match,
                            "classification": "bytes_appended_after_indexing"
                            if prefix_match
                            else "retained_file_differs_from_indexed_version",
                        }
                    )
        missing = collections.Counter(
            reason
            for op in coverage.get("operations", [])
            for reason in op.get("missing", [])
        )
        trace_reports.append(
            {
                "path": str(trace.relative_to(root)),
                "run_id": next(
                    run.name for run in run_paths if trace.is_relative_to(run)
                ),
                "events_with_verified_payloads": count,
                "incomplete_final_index_line": incomplete_tail,
                "retained_artifacts": artifact_count,
                "retained_artifacts_verified": artifacts_verified,
                "retained_artifact_differences": artifact_differences,
                "coverage_path": str(coverage_path.relative_to(root))
                if coverage_path.exists()
                else None,
                "execution_finished": coverage.get("execution_finished"),
                "observed_operations_paired": coverage.get(
                    "observed_operations_paired"
                ),
                "collection_gaps": coverage.get("collection_gaps", []),
                "missing_operation_fields": dict(missing),
                "scope": coverage.get("scope"),
            }
        )
    write_json(output / "EXPORT_TRACE_COVERAGE.json", trace_reports)
    manifest = {
        "schema_version": 2,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_project": str(project_root),
        "archive_root": str(root),
        "campaign": args.campaign,
        "selection_audit": str(audit_path.relative_to(root))
        if audit_path.is_file()
        else None,
        "selection_audit_sha256": _digest(audit_path) if audit_path.is_file() else None,
        "selection": "all planned conditions and every retained run directory; includes failures and missing records",
        "planned_condition_count": len(rows),
        "missing_run_ids": missing_ids,
        "outcome_counts": dict(
            collections.Counter(
                row.get("outcome", row.get("status", "missing")) for row in rows
            )
        ),
        "completed_condition_count": completed_count,
        "final_run_count": len(existing_final_ids),
        "earlier_attempt_count": len(run_paths) - len(existing_final_ids),
        "conditions": rows,
        "run_directories": [path.name for path in run_paths],
        "earlier_attempt_run_ids": [
            path.name for path in run_paths if path.name not in final_ids
        ],
        "inventory": inventory,
        "source_file_count": sum(item["files"] for item in inventory),
        "source_directory_count": sum(item["directories"] for item in inventory),
        "source_content_bytes": sum(item["bytes"] for item in inventory),
        "trace_execution_count": len(trace_reports),
        "trace_payload_events_verified": sum(
            item["events_with_verified_payloads"] for item in trace_reports
        ),
        "retained_provider_artifacts_verified": sum(
            item["retained_artifacts_verified"] for item in trace_reports
        ),
        "retained_artifacts_appended_after_indexing": sum(
            item["classification"] == "bytes_appended_after_indexing"
            for trace in trace_reports
            for item in trace["retained_artifact_differences"]
        ),
        "retained_artifacts_with_changed_indexed_content": sum(
            item["classification"] == "retained_file_differs_from_indexed_version"
            for trace in trace_reports
            for item in trace["retained_artifact_differences"]
        ),
        "compression": "GNU tar + zstd level 3, 2 threads",
        "live_server_health_context_snapshots": snapshot_details,
        "original_records_modified": False,
        "context_note": "All outcomes and available run trajectories are retained; missing run IDs are listed explicitly.",
        "not_included": [
            "source/build working directories",
            "Docker images",
            "SSH/provider credential files",
            "transient judge/build caches",
        ],
        "coverage_note": "All available recorded files are exported without truncation. Original coverage gaps and unpaired operations remain; unavailable native actions or hidden reasoning cannot be reconstructed.",
    }
    write_json(output / "EXPORT_MANIFEST.json", manifest)
    (output / "EXPORT_README.md").write_text(
        f"# PitBench records export\n\nContains {len(rows)} planned conditions and {len(run_paths)} retained run directories, including failures and infrastructure attempts. Missing run IDs are listed in EXPORT_MANIFEST.json. Raw observations, seeds, patches, provider logs and full trace objects are preserved. Source records are not deleted. Transient build/source working directories, caches, Docker images and credentials are excluded.\n"
    )
    shutil.copy2(Path(__file__), output / "exporter_source.py")
    metadata_names = [
        "EXPORT_MANIFEST.json",
        "EXPORT_TRACE_COVERAGE.json",
        "EXPORT_README.md",
        "exporter_source.py",
    ]
    file_list = output / "archive-includes.nul"
    tar_includes = []
    for path in includes:
        if str(path) in snapshot_names:
            continue
        if path == campaign.relative_to(root):
            tar_includes.extend(
                child.relative_to(root)
                for child in sorted(campaign.iterdir())
                if str(child.relative_to(root)) not in snapshot_names
            )
        else:
            tar_includes.append(path)
    file_list.write_bytes(
        b"\0".join(os.fsencode(path) for path in tar_includes) + b"\0"
    )
    status(
        "compressing",
        **{
            key: manifest[key]
            for key in [
                "source_file_count",
                "source_content_bytes",
                "trace_execution_count",
            ]
        },
    )
    groups = [(root, tar_includes)]
    if snapshot_names:
        groups.append((snapshot_root, [Path(name) for name in snapshot_names]))
    groups.append((output, [Path(name) for name in metadata_names]))
    create_verified_archive(
        groups,
        partial,
        log_directory=output,
        exclude=("*/workspaces", "*/shared-cache", "*/judge-tmp", "*/__pycache__"),
    )
    partial.rename(archive)
    digest = _digest(archive)
    (output / (archive.name + ".sha256")).write_text(f"{digest}  {archive.name}\n")
    verification = {
        "archive": str(archive),
        "compressed_bytes": archive.stat().st_size,
        "sha256": digest,
        "source_file_count": manifest["source_file_count"],
        "source_content_bytes": manifest["source_content_bytes"],
        "completed_conditions": completed_count,
        "earlier_attempts": len(run_paths) - len(existing_final_ids),
        "trace_executions": len(trace_reports),
        "payload_events_verified": manifest["trace_payload_events_verified"],
        "retained_provider_artifacts_verified": manifest[
            "retained_provider_artifacts_verified"
        ],
        "retained_artifacts_appended_after_indexing": manifest[
            "retained_artifacts_appended_after_indexing"
        ],
        "retained_artifacts_with_changed_indexed_content": manifest[
            "retained_artifacts_with_changed_indexed_content"
        ],
        "archive_vs_originals": "passed",
        "live_server_health_context": "frozen snapshots compared separately; experiment records compared with originals",
        "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    write_json(output / "export-verification.json", verification)
    status("complete", **verification)
    return verification
