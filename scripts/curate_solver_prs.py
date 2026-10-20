"""Harvest and audit real merged PRs for solver improvement task curation."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pitbench.repositories.pr_curation import (
    build_baseline_opportunity_evidence,
    build_mechanism_review_queue,
    build_paired_correctness_evidence,
    build_selection_queue,
    build_task_integration_evidence,
    load_repository_catalog,
    materialize_git_backed_checkout,
    merge_candidates,
    normalize_search_issue,
    read_jsonl,
    summarize_registry,
    validate_evidence_stage,
    validate_selection_review,
    validate_workload_exposure_evidence,
    verify_source_patch,
    write_jsonl,
)


def _github_token(name: str) -> str | None:
    configured = os.environ.get(name)
    if configured:
        return configured
    try:
        result = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _queue(args: argparse.Namespace) -> None:
    catalog = load_repository_catalog(args.catalog)
    queue = build_selection_queue(read_jsonl(args.registry), catalog, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(queue, indent=2) + "\n")
    print(
        json.dumps(
            {
                "eligible_discovered_prs": queue["eligible_discovered_prs"],
                "pending_evidence_prs": queue["pending_evidence_prs"],
                "queue_size": queue["queue_size"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


def _triage(args: argparse.Namespace) -> None:
    catalog = load_repository_catalog(args.catalog)
    details = [row for path in args.details for row in read_jsonl(path)]
    queue = build_mechanism_review_queue(
        read_jsonl(args.registry),
        catalog,
        details,
        args.limit,
        audit_fraction=args.audit_fraction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(queue, indent=2) + "\n")
    print(
        json.dumps(
            {
                "queue_size": queue["queue_size"],
                "manual_diff_queue_size": queue["manual_diff_queue_size"],
                "mechanism_hint_pool": queue["mechanism_hint_pool"],
                "low_signal_pool": queue["low_signal_pool"],
                "eligible_without_completed_details": queue[
                    "eligible_without_completed_details"
                ],
                "pending_evidence_prs": queue["pending_evidence_prs"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


def _screening_audit(args: argparse.Namespace) -> None:
    """Measure review coverage by metadata signal without admitting candidates."""
    catalog = load_repository_catalog(args.catalog)
    rows = read_jsonl(args.registry)
    detail_rows = [row for path in args.details for row in read_jsonl(path)]
    review_frame = [
        {**row, "stage": "discovered", "selection_decision": "unreviewed"}
        for row in rows
    ]
    queue = build_mechanism_review_queue(
        review_frame, catalog, detail_rows, max(1, len(rows)), audit_fraction=0.0
    )
    indexed = {(row["repository"], row["pr_number"]): row for row in rows}
    bands: dict[str, Counter[str]] = defaultdict(Counter)
    repositories: dict[str, Counter[str]] = defaultdict(Counter)
    phase_hints: dict[str, Counter[str]] = defaultdict(Counter)
    low_signal_cases: list[dict[str, Any]] = []
    for item in queue["queue"]:
        key = (item["repository"], item["pr_number"])
        row = indexed[key]
        decision = row.get("selection_decision") or (
            "legacy_unlabeled" if row.get("stage") == "task_ready" else "unreviewed"
        )
        band = str(item["signal_band"])
        bands[band][decision] += 1
        repositories[item["repository"]][decision] += 1
        for phase in item["solver_phase_path_hints"] or ["unspecified"]:
            phase_hints[phase][decision] += 1
        if band in {"2", "3"} and decision not in {
            "unreviewed",
            "legacy_unlabeled",
        }:
            low_signal_cases.append(
                {
                    "repository": key[0],
                    "pr_number": key[1],
                    "signal_band": item["signal_band"],
                    "selection_decision": decision,
                    "review_path": row.get("selection_review_path"),
                }
            )
    report = {
        "schema_version": 1,
        "registry_sha256": hashlib.sha256(args.registry.read_bytes()).hexdigest(),
        "detail_batches": [
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "record_count": len(read_jsonl(path)),
            }
            for path in args.details
        ],
        "completed_metadata_candidates": queue["eligible_with_completed_details"],
        "missing_completed_metadata": queue["eligible_without_completed_details"],
        "manual_full_diff_candidates": queue["manual_diff_queue_size"],
        "by_signal_band": {
            band: dict(sorted(counts.items())) for band, counts in sorted(bands.items())
        },
        "by_repository": {
            repository: dict(sorted(counts.items()))
            for repository, counts in sorted(repositories.items())
        },
        "by_phase_path_hint": {
            phase: dict(sorted(counts.items()))
            for phase, counts in sorted(phase_hints.items())
        },
        "low_signal_reviewed": sum(
            count
            for band in ("2", "3")
            for decision, count in bands[band].items()
            if decision not in {"unreviewed", "legacy_unlabeled"}
        ),
        "low_signal_advanced": sum(
            bands[band]["advance"] for band in ("2", "3")
        ),
        "reviewed_low_signal_cases": sorted(
            low_signal_cases,
            key=lambda case: (case["repository"], case["pr_number"]),
        ),
        "claim_boundary": (
            "Metadata bands measure review coverage only. The deterministic "
            "low-signal lane is an audit sample, not an unbiased recall estimate; "
            "unreviewed, missing-detail, and manual-full-diff PRs remain unknown. "
            "No candidate counts as a task without all downstream evidence gates."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "completed_metadata_candidates": report[
                    "completed_metadata_candidates"
                ],
                "low_signal_reviewed": report["low_signal_reviewed"],
                "low_signal_advanced": report["low_signal_advanced"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


def _repository_path(path: Path, repository_root: Path) -> Path:
    return path if path.is_absolute() else repository_root / path


def _evidence_output(path: Path, repository_root: Path) -> tuple[Path, str]:
    root = repository_root.resolve()
    destination = _repository_path(path, root).resolve()
    if not destination.is_relative_to(root) or destination.exists():
        raise ValueError("evidence output must be a new file inside the repository")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination, destination.relative_to(root).as_posix()


def _evidence_candidate(
    args: argparse.Namespace,
    allowed_stages: tuple[str, ...] = ("correctness_verified",),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = read_jsonl(args.registry)
    matching = [
        row
        for row in rows
        if row["repository"] == args.repository and row["pr_number"] == args.pr_number
    ]
    if len(matching) != 1:
        raise ValueError("evidence builder requires one indexed PR")
    row = matching[0]
    if row.get("stage") not in allowed_stages:
        raise ValueError(f"evidence builder requires stage {allowed_stages}")
    if not row.get("task_id") or not row.get("task_config_path"):
        raise ValueError("evidence builder requires a pinned task configuration")
    return row, rows


def _public_correctness(args: argparse.Namespace) -> None:
    catalog = load_repository_catalog(args.catalog)
    row, rows = _evidence_candidate(args, ("workload_verified",))
    destination, relative = _evidence_output(args.output, args.repository_root)
    ledger, _ = _evidence_output(args.ledger_output, args.repository_root)
    if destination == ledger:
        raise ValueError("correctness evidence and run ledger need separate files")
    test_results = json.loads(
        _repository_path(args.test_results, args.repository_root).read_text()
    )
    semantics = (
        json.loads(_repository_path(args.semantics, args.repository_root).read_text())
        if args.semantics
        else {}
    )
    checker_source = _repository_path(args.checker_artifact, args.repository_root)
    checker_bytes = checker_source.read_bytes()
    checker_digest = hashlib.sha256(checker_bytes).hexdigest()
    checker_snapshot = (
        destination.parent
        / "checker_snapshots"
        / f"{checker_source.stem}_{checker_digest}{checker_source.suffix}"
    )
    checker_snapshot.parent.mkdir(parents=True, exist_ok=True)
    if checker_snapshot.exists():
        if checker_snapshot.read_bytes() != checker_bytes:
            raise ValueError("frozen correctness checker hash collision")
    else:
        checker_snapshot.write_bytes(checker_bytes)
    evidence = build_paired_correctness_evidence(
        row,
        args.repository_root,
        _repository_path(args.pairs, args.repository_root),
        ledger,
        checker_snapshot,
        test_results,
        semantics,
    )
    destination.write_text(json.dumps(evidence, indent=2) + "\n")
    row["correctness_evidence_path"] = relative
    row["stage"] = "correctness_verified"
    summary = summarize_registry(
        rows, catalog["target_validated_pr_tasks"], args.repository_root
    )
    write_jsonl(args.registry, rows)
    print(
        json.dumps(
            {
                "repository": row["repository"],
                "pr_number": row["pr_number"],
                "stage": row["stage"],
                "verified_solution_count": evidence["verified_solution_count"],
                "normal_timeout_count": evidence["normal_timeout_count"],
                "remaining_task_ready_prs": summary["remaining_task_ready_prs"],
            }
        )
    )


def _baseline_evidence(args: argparse.Namespace) -> None:
    catalog = load_repository_catalog(args.catalog)
    row, rows = _evidence_candidate(args)
    if row.get("task_track") != "open_opportunity":
        raise ValueError("baseline builder requires an open-opportunity task")
    destination, relative = _evidence_output(args.output, args.repository_root)
    evidence = build_baseline_opportunity_evidence(
        row,
        args.repository_root,
        _repository_path(args.manifest, args.repository_root),
        _repository_path(args.analysis, args.repository_root),
        _repository_path(args.runs, args.repository_root),
    )
    destination.write_text(json.dumps(evidence, indent=2) + "\n")
    row["opportunity_evidence_path"] = relative
    summary = summarize_registry(
        rows, catalog["target_validated_pr_tasks"], args.repository_root
    )
    write_jsonl(args.registry, rows)
    print(
        json.dumps(
            {
                "task_id": row["task_id"],
                "stage": row["stage"],
                "remaining_task_ready_prs": summary["remaining_task_ready_prs"],
            }
        )
    )


def _integration_evidence(args: argparse.Namespace) -> None:
    catalog = load_repository_catalog(args.catalog)
    row, rows = _evidence_candidate(args)
    if row.get("task_track") != "open_opportunity":
        raise ValueError("real Judge integration requires an open-opportunity task")
    if not row.get("opportunity_evidence_path"):
        raise ValueError("real Judge integration requires hidden baseline evidence")
    destination, relative = _evidence_output(args.output, args.repository_root)
    evidence = build_task_integration_evidence(
        row,
        args.repository_root,
        _repository_path(args.agent_identity, args.repository_root),
        _repository_path(args.checkout_audit, args.repository_root),
        _repository_path(args.run_directory, args.repository_root),
        args.dry_run_exit_code,
        args.dry_run_exit_note,
    )
    destination.write_text(json.dumps(evidence, indent=2) + "\n")
    row["integration_evidence_path"] = relative
    row["integration_status"] = "passed"
    row["stage"] = "task_ready"
    summary = summarize_registry(
        rows, catalog["target_validated_pr_tasks"], args.repository_root
    )
    write_jsonl(args.registry, rows)
    print(
        json.dumps(
            {
                "task_id": row["task_id"],
                "stage": row["stage"],
                "remaining_task_ready_prs": summary["remaining_task_ready_prs"],
            }
        )
    )


def _screen(args: argparse.Namespace) -> None:
    catalog = load_repository_catalog(args.catalog)
    rows = read_jsonl(args.registry)
    review = json.loads(args.review_file.read_text())
    if not isinstance(review, dict):
        raise ValueError("selection review must be a JSON object")
    matches = [
        row
        for row in rows
        if row["repository"] == review.get("repository")
        and row["pr_number"] == review.get("pr_number")
    ]
    if len(matches) != 1:
        raise ValueError("selection review requires exactly one indexed PR")
    row = matches[0]
    if row["stage"] not in {"discovered", "intent_reviewed"}:
        raise ValueError("selection review cannot rewrite a later evidence stage")
    decision = validate_selection_review(review, row)
    scope_status = {
        entry["repository"]: entry["scope_status"] for entry in catalog["repositories"]
    }.get(row["repository"])
    if scope_status is None:
        raise ValueError("selection PR source is absent from the repository catalog")
    if decision == "advance" and scope_status not in {"task_backed", "approved_source"}:
        raise ValueError("repository needs domain approval before a PR can advance")
    review_path = args.review_file.resolve().relative_to(args.repository_root.resolve())
    row["selection_review_path"] = str(review_path)
    row["selection_review_sha256"] = hashlib.sha256(
        args.review_file.read_bytes()
    ).hexdigest()
    row["selection_decision"] = decision
    if decision == "advance":
        row["intent_evidence_path"] = str(review_path)
        row["stage"] = "intent_reviewed"
    else:
        row.pop("intent_evidence_path", None)
        row["stage"] = "discovered"
    summary = summarize_registry(
        rows, catalog["target_validated_pr_tasks"], args.repository_root
    )
    write_jsonl(args.registry, rows)
    print(
        json.dumps(
            {
                "repository": row["repository"],
                "pr_number": row["pr_number"],
                "decision": decision,
                "stage": row["stage"],
                "remaining_task_ready_prs": summary["remaining_task_ready_prs"],
            },
            sort_keys=True,
        )
    )


def _checkout(args: argparse.Namespace) -> None:
    catalog = load_repository_catalog(args.catalog)
    names = {entry["repository"] for entry in catalog["repositories"]}
    if args.repository not in names:
        raise ValueError("checkout repository is absent from catalog")
    matches = [
        row
        for row in read_jsonl(args.registry)
        if row["repository"] == args.repository and row["pr_number"] == args.pr_number
    ]
    if len(matches) != 1:
        raise ValueError("checkout needs exactly one indexed PR")
    sources = {}
    for value in args.submodule_source or []:
        path, separator, source = value.partition("=")
        if not separator or not path or not source or path in sources:
            raise ValueError(f"invalid submodule source mapping: {value!r}")
        sources[path] = Path(source)
    audit = materialize_git_backed_checkout(
        source_repository=args.repository_checkout,
        base_commit=matches[0]["base_commit"],
        destination=args.output_dir,
        submodule_sources=sources,
    )
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2) + "\n")
    print(
        json.dumps(
            {
                "repository": args.repository,
                "pr_number": args.pr_number,
                "base_commit": audit["base_commit"],
                "submodule_count": len(audit["submodule_status"]),
            }
        )
    )


def _github_search_page(
    repository: str, query: str, page: int, token: str | None
) -> tuple[dict[str, Any], int | None]:
    expression = f"repo:{repository} is:pr is:merged {query}"
    url = "https://api.github.com/search/issues?" + urllib.parse.urlencode(
        {"q": expression, "per_page": 100, "page": page}
    )
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "PitBench-PR-curation",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
        remaining = response.headers.get("X-RateLimit-Remaining")
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError("GitHub search returned an unexpected response")
    return payload, int(remaining) if remaining is not None else None


def _github_pulls_page(
    repository: str, page: int, token: str | None
) -> tuple[list[dict[str, Any]], int | None]:
    url = f"https://api.github.com/repos/{repository}/pulls?" + urllib.parse.urlencode(
        {"state": "closed", "per_page": 100, "page": page}
    )
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "PitBench-PR-curation",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
        remaining = response.headers.get("X-RateLimit-Remaining")
    if not isinstance(payload, list):
        raise ValueError("GitHub pulls endpoint returned an unexpected response")
    return payload, int(remaining) if remaining is not None else None


def _github_pull_detail(
    repository: str, number: int, token: str | None
) -> tuple[dict[str, Any], list[dict[str, Any]] | None]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "PitBench-PR-curation",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    def get(url: str) -> Any:
        for attempt in range(3):
            try:
                with urllib.request.urlopen(
                    urllib.request.Request(url, headers=headers), timeout=30
                ) as response:
                    return json.load(response)
            except (http.client.IncompleteRead, urllib.error.URLError) as error:
                if isinstance(error, urllib.error.HTTPError) and error.code in {
                    403,
                    429,
                }:
                    raise
                if attempt == 2:
                    try:
                        fallback = subprocess.run(
                            ["gh", "api", url],
                            capture_output=True,
                            text=True,
                            check=False,
                            timeout=90,
                        )
                    except (FileNotFoundError, subprocess.TimeoutExpired):
                        raise error from None
                    if fallback.returncode == 0:
                        return json.loads(fallback.stdout)
                    raise error
                time.sleep(2**attempt)
        raise AssertionError("unreachable GitHub request retry")

    base = f"https://api.github.com/repos/{repository}/pulls/{number}"
    detail = get(base)
    if not isinstance(detail, dict):
        raise ValueError("GitHub PR endpoint returned an unexpected response")
    if (
        detail.get("changed_files", 0) > 1000
        or detail.get("additions", 0) + detail.get("deletions", 0) > 100000
    ):
        return detail, None
    files = []
    page = 0
    while True:
        page += 1
        batch = get(
            base + "/files?" + urllib.parse.urlencode({"per_page": 100, "page": page})
        )
        if not isinstance(batch, list):
            raise ValueError("GitHub PR files endpoint returned an unexpected response")
        files.extend(batch)
        if len(batch) < 100:
            break
    return detail, files


def _enrich(args: argparse.Namespace) -> None:
    """Cache PR body and changed-file evidence without making an admission decision."""
    catalog = load_repository_catalog(args.catalog)
    known = {entry["repository"] for entry in catalog["repositories"]}
    queue = json.loads(args.queue.read_text())
    selected = queue.get("queue")
    if not isinstance(selected, list) or args.limit <= 0:
        raise ValueError("enrichment needs a queue and positive limit")
    registry = {
        (row["repository"], row["pr_number"]): row for row in read_jsonl(args.registry)
    }
    details = read_jsonl(args.output)
    completed = {
        (row["repository"], row["pr_number"])
        for row in details
        if row.get("status") in {"completed", "manual_large_pr"}
    }
    token = _github_token(args.token_env)
    for item in selected[: args.limit]:
        key = (item["repository"], item["pr_number"])
        if key[0] not in known or key not in registry:
            raise ValueError(f"queue PR is absent from the catalog or registry: {key}")
        if key in completed:
            continue
        try:
            pull, files = _github_pull_detail(key[0], key[1], token)
            indexed = registry[key]
            if (
                pull.get("merged_at") != indexed.get("merged_at")
                or pull.get("base", {}).get("sha") != indexed.get("base_commit")
                or pull.get("head", {}).get("sha") != indexed.get("head_commit")
            ):
                raise ValueError(f"GitHub PR identity differs from registry: {key}")
            if files is not None and pull.get("changed_files") != len(files):
                files = None
            record = {
                "repository": key[0],
                "pr_number": key[1],
                "status": "manual_large_pr" if files is None else "completed",
                "title": pull.get("title"),
                "body": pull.get("body"),
                "url": pull.get("html_url"),
                "base_commit": indexed["base_commit"],
                "head_commit": indexed["head_commit"],
                "merged_at": indexed["merged_at"],
                "changed_file_count": pull.get("changed_files"),
                "additions": pull.get("additions"),
                "deletions": pull.get("deletions"),
                "files": [
                    {
                        "path": file.get("filename"),
                        "status": file.get("status"),
                        "additions": file.get("additions"),
                        "deletions": file.get("deletions"),
                        "patch_sha256": (
                            hashlib.sha256(file["patch"].encode()).hexdigest()
                            if isinstance(file.get("patch"), str)
                            else None
                        ),
                    }
                    for file in files or []
                ],
                "claim_boundary": (
                    "PR metadata is review evidence only; oversized PRs require "
                    "manual diff inspection, and no solver mechanism, correctness, "
                    "workload exposure, or performance decision follows."
                ),
            }
        except (
            http.client.IncompleteRead,
            urllib.error.URLError,
            ValueError,
            KeyError,
        ) as error:
            rate_limited = isinstance(error, urllib.error.HTTPError) and error.code in {
                403,
                429,
            }
            record = {
                "repository": key[0],
                "pr_number": key[1],
                "status": "rate_limited" if rate_limited else "failed",
                "error": str(error),
            }
        details.append(record)
        write_jsonl(args.output, details)
        print(f"{key[0]}#{key[1]}: {record['status']}", flush=True)
        if record["status"] == "rate_limited":
            break


def _inventory(args: argparse.Namespace) -> None:
    catalog = load_repository_catalog(args.catalog)
    known = {row["repository"]: row["scope_status"] for row in catalog["repositories"]}
    selected = args.repository or [
        name for name, status in known.items() if status != "rejected"
    ]
    unknown = sorted(set(selected) - known.keys())
    if unknown:
        raise ValueError(f"repositories absent from catalog: {unknown}")
    token = os.environ.get(args.token_env)
    log = read_jsonl(args.query_log)
    completed = {row["repository"] for row in log if row.get("status") == "completed"}
    for repository in selected:
        if repository in completed:
            continue
        output = args.output_dir / f"{repository.replace('/', '__')}.jsonl"
        rows = read_jsonl(output)
        page = 0
        closed_count = 0
        try:
            while True:
                page += 1
                pulls, remaining = _github_pulls_page(repository, page, token)
                closed_count += len(pulls)
                for pull in pulls:
                    if not pull.get("merged_at"):
                        continue
                    rows.append(
                        {
                            "repository": repository,
                            "pr_number": pull["number"],
                            "title": pull["title"],
                            "url": pull["html_url"],
                            "created_at": pull.get("created_at"),
                            "closed_at": pull.get("closed_at"),
                            "merged_at": pull["merged_at"],
                            "base_commit": pull["base"]["sha"],
                            "head_commit": pull["head"]["sha"],
                            "merge_commit": pull.get("merge_commit_sha"),
                            "matched_queries": ["all_merged"],
                            "stage": "discovered",
                        }
                    )
                rows = merge_candidates(rows)
                write_jsonl(output, rows)
                if len(pulls) < 100:
                    break
            log.append(
                {
                    "repository": repository,
                    "status": "completed",
                    "pages": page,
                    "closed_prs": closed_count,
                    "merged_prs": len(rows),
                    "rate_remaining": remaining,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            write_jsonl(args.query_log, log)
            print(f"{repository}: {len(rows)} merged PRs, {page} pages")
        except (urllib.error.URLError, ValueError, KeyError) as error:
            log.append(
                {
                    "repository": repository,
                    "status": "failed",
                    "pages_completed": page - 1,
                    "closed_prs_seen": closed_count,
                    "merged_prs_retained": len(rows),
                    "error": str(error),
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            write_jsonl(args.query_log, log)
            raise


def _harvest(args: argparse.Namespace) -> None:
    catalog = load_repository_catalog(args.catalog)
    catalog_names = {
        row["repository"]: row["scope_status"] for row in catalog["repositories"]
    }
    selected = args.repository or [
        name for name, status in catalog_names.items() if status != "rejected"
    ]
    unknown = sorted(set(selected) - catalog_names.keys())
    if unknown:
        raise ValueError(f"repositories absent from catalog: {unknown}")
    queries = args.query or catalog.get("search_queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError("no PR discovery queries configured")
    token = os.environ.get(args.token_env)
    existing = read_jsonl(args.output)
    query_log = read_jsonl(args.query_log)
    completed = {
        (row["repository"], row["query"])
        for row in query_log
        if row.get("status") == "completed"
    }
    for repository in selected:
        for query in queries:
            if (repository, query) in completed:
                continue
            received = []
            pages = 0
            try:
                while True:
                    payload, remaining = _github_search_page(
                        repository, query, pages + 1, token
                    )
                    total = payload["total_count"]
                    if type(total) is not int or total < 0:
                        raise ValueError("GitHub search omitted its total count")
                    if total > 1000:
                        raise ValueError(
                            f"GitHub caps search at 1000 results: {repository} {query} "
                            "must be partitioned by date"
                        )
                    pages += 1
                    for issue in payload["items"]:
                        row = normalize_search_issue(repository, issue, query)
                        if row is not None:
                            received.append(row)
                    if pages * 100 >= total:
                        break
                existing = merge_candidates([*existing, *received])
                write_jsonl(args.output, existing)
                query_log.append(
                    {
                        "repository": repository,
                        "query": query,
                        "status": "completed",
                        "total_count": total,
                        "retrieved_count": len(received),
                        "pages": pages,
                        "rate_remaining": remaining,
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                write_jsonl(args.query_log, query_log)
                print(f"{repository}: {query}: {len(received)}; pool={len(existing)}")
            except (urllib.error.URLError, ValueError) as error:
                if received:
                    existing = merge_candidates([*existing, *received])
                    write_jsonl(args.output, existing)
                query_log.append(
                    {
                        "repository": repository,
                        "query": query,
                        "status": "failed",
                        "pages_completed": pages,
                        "retrieved_count": len(received),
                        "error": str(error),
                        "failed_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                write_jsonl(args.query_log, query_log)
                raise


def _import(args: argparse.Namespace) -> None:
    catalog = load_repository_catalog(args.catalog)
    known = {row["repository"] for row in catalog["repositories"]}
    incoming = []
    for path in args.input:
        for row in read_jsonl(path):
            if row["repository"] not in known:
                raise ValueError(f"unknown repository in import: {row['repository']}")
            incoming.append(
                {
                    **row,
                    "matched_queries": row.get("matched_queries")
                    or row.get("matched_terms")
                    or ["all_merged"],
                    "stage": row.get("stage", "discovered"),
                }
            )
    merged = merge_candidates([*read_jsonl(args.output), *incoming])
    summary = summarize_registry(
        merged, catalog["target_validated_pr_tasks"], args.repository_root
    )
    write_jsonl(args.output, merged)
    print(json.dumps(summary, sort_keys=True))


def _audit(args: argparse.Namespace) -> None:
    catalog = load_repository_catalog(args.catalog)
    rows = read_jsonl(args.registry)
    summary = summarize_registry(
        rows, catalog["target_validated_pr_tasks"], args.repository_root
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))


def _source_verify(args: argparse.Namespace) -> None:
    catalog = load_repository_catalog(args.catalog)
    known = {row["repository"] for row in catalog["repositories"]}
    if args.repository not in known:
        raise ValueError(f"repository absent from catalog: {args.repository}")
    rows = read_jsonl(args.registry)
    matching = [
        row
        for row in rows
        if row["repository"] == args.repository and row["pr_number"] == args.pr_number
    ]
    if len(matching) != 1:
        raise ValueError("source verification requires exactly one indexed PR")
    row = matching[0]
    if row["stage"] == "discovered":
        raise ValueError(
            "run structured selection screening before source verification"
        )
    if row["stage"] == "intent_reviewed" and row.get("selection_decision") != "advance":
        raise ValueError("source verification requires selection approval")
    if row["stage"] not in {"intent_reviewed", "source_verified"}:
        raise ValueError("source verification cannot downgrade a later evidence stage")
    intent_note = args.intent_note
    if not intent_note and row.get("selection_review_path"):
        review = json.loads(
            (args.repository_root / row["selection_review_path"]).read_text()
        )
        intent_note = review["solver_behavior_hypothesis"]
    evidence = verify_source_patch(
        repository_checkout=args.repository_checkout,
        record=row,
        editable_roots=args.editable_root,
        patch_path=args.patch_output,
        audit_path=args.audit_output,
        intent_note=intent_note or "",
    )
    row.update(
        {
            "stage": "source_verified",
            "intent_evidence_path": row.get("intent_evidence_path")
            or str(args.audit_output),
            "source_patch_path": str(args.patch_output),
            "source_patch_sha256": evidence["source_patch_sha256"],
            "changed_source_files": evidence["changed_source_files"],
        }
    )
    summary = summarize_registry(
        rows, catalog["target_validated_pr_tasks"], args.repository_root
    )
    write_jsonl(args.registry, rows)
    print(
        json.dumps(
            {"pr_number": args.pr_number, "stage": row["stage"], "summary": summary}
        )
    )


def _workload_evidence(args: argparse.Namespace) -> None:
    """Freeze and validate measured solver-path activation evidence."""
    catalog = load_repository_catalog(args.catalog)
    rows = read_jsonl(args.registry)
    matching = [
        row
        for row in rows
        if row["repository"] == args.repository and row["pr_number"] == args.pr_number
    ]
    if len(matching) != 1:
        raise ValueError("workload evidence requires exactly one indexed PR")
    row = matching[0]
    if row.get("stage") != "build_verified":
        raise ValueError("workload evidence requires a build-verified candidate")
    review_path = _repository_path(args.repository_root / row["selection_review_path"], args.repository_root)
    if hashlib.sha256(review_path.read_bytes()).hexdigest() != row.get(
        "selection_review_sha256"
    ):
        raise ValueError("workload evidence selection review hash mismatch")
    review = json.loads(review_path.read_text())
    if validate_selection_review(review, row) != "advance":
        raise ValueError("workload evidence requires an advanced selection review")

    root = args.repository_root.resolve()
    evidence: dict[str, Any] = {
        "schema_version": 1,
        **{
            field: row[field]
            for field in (
                "repository",
                "pr_number",
                "base_commit",
                "head_commit",
                "source_patch_sha256",
            )
        },
        "instance_selection_rule_sha256": hashlib.sha256(
            review["instance_selection_rule"].encode()
        ).hexdigest(),
    }
    for name, input_path in (
        ("instance_manifest", args.manifest),
        ("activation_artifact", args.activation),
    ):
        actual = _repository_path(input_path, root).resolve()
        if not actual.is_relative_to(root) or not actual.is_file():
            raise ValueError(f"workload {name} is missing or outside the repository")
        evidence[f"{name}_path"] = actual.relative_to(root).as_posix()
        evidence[f"{name}_sha256"] = hashlib.sha256(actual.read_bytes()).hexdigest()
    activation = json.loads(_repository_path(args.activation, root).read_text())
    evidence["cases"] = activation.get("cases")
    validate_workload_exposure_evidence(evidence, row, review, root)

    destination, relative = _evidence_output(args.output, root)
    destination.write_text(json.dumps(evidence, indent=2) + "\n")
    updated_row = {**row, "workload_evidence_path": relative, "stage": "workload_verified"}
    try:
        validate_evidence_stage(updated_row, root)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    rows[rows.index(row)] = updated_row
    summary = summarize_registry(rows, catalog["target_validated_pr_tasks"], root)
    write_jsonl(args.registry, rows)
    print(
        json.dumps(
            {
                "repository": args.repository,
                "pr_number": args.pr_number,
                "stage": updated_row["stage"],
                "workload_evidence_path": relative,
                "remaining_task_ready_prs": summary["remaining_task_ready_prs"],
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (
        "inventory",
        "harvest",
        "import",
        "audit",
        "queue",
        "triage",
        "screening-audit",
        "enrich",
        "checkout",
        "screen",
        "source-verify",
        "workload-evidence",
        "public-correctness",
        "baseline-evidence",
        "integrate",
    ):
        subparser = commands.add_parser(name)
        subparser.add_argument("--catalog", type=Path, required=True)
        if name == "inventory":
            subparser.add_argument("--output-dir", type=Path, required=True)
            subparser.add_argument("--query-log", type=Path, required=True)
            subparser.add_argument("--repository", action="append")
            subparser.add_argument("--token-env", default="GITHUB_TOKEN")
        elif name == "harvest":
            subparser.add_argument("--output", type=Path, required=True)
            subparser.add_argument("--query-log", type=Path, required=True)
            subparser.add_argument("--repository", action="append")
            subparser.add_argument("--query", action="append")
            subparser.add_argument("--token-env", default="GITHUB_TOKEN")
        elif name == "import":
            subparser.add_argument("--input", type=Path, required=True, action="append")
            subparser.add_argument("--output", type=Path, required=True)
            subparser.add_argument("--repository-root", type=Path, default=Path.cwd())
        elif name == "audit":
            subparser.add_argument("--registry", type=Path, required=True)
            subparser.add_argument("--output", type=Path)
            subparser.add_argument("--repository-root", type=Path, default=Path.cwd())
        elif name == "queue":
            subparser.add_argument("--registry", type=Path, required=True)
            subparser.add_argument("--limit", type=int, default=100)
            subparser.add_argument("--output", type=Path, required=True)
        elif name in {"triage", "screening-audit"}:
            subparser.add_argument("--registry", type=Path, required=True)
            subparser.add_argument(
                "--details", type=Path, action="append", required=True
            )
            if name == "triage":
                subparser.add_argument("--limit", type=int, default=100)
                subparser.add_argument("--audit-fraction", type=float, default=0.1)
            subparser.add_argument("--output", type=Path, required=True)
        elif name == "enrich":
            subparser.add_argument("--registry", type=Path, required=True)
            subparser.add_argument("--queue", type=Path, required=True)
            subparser.add_argument("--limit", type=int, default=100)
            subparser.add_argument("--output", type=Path, required=True)
            subparser.add_argument("--token-env", default="GITHUB_TOKEN")
        elif name == "checkout":
            subparser.add_argument("--registry", type=Path, required=True)
            subparser.add_argument("--repository", required=True)
            subparser.add_argument("--pr-number", type=int, required=True)
            subparser.add_argument("--repository-checkout", type=Path, required=True)
            subparser.add_argument("--output-dir", type=Path, required=True)
            subparser.add_argument("--submodule-source", action="append")
            subparser.add_argument("--audit-output", type=Path, required=True)
        elif name == "screen":
            subparser.add_argument("--registry", type=Path, required=True)
            subparser.add_argument("--review-file", type=Path, required=True)
            subparser.add_argument("--repository-root", type=Path, default=Path.cwd())
        elif name in {
            "workload-evidence",
            "public-correctness",
            "baseline-evidence",
            "integrate",
        }:
            subparser.add_argument("--registry", type=Path, required=True)
            subparser.add_argument("--repository", required=True)
            subparser.add_argument("--pr-number", type=int, required=True)
            subparser.add_argument("--repository-root", type=Path, default=Path.cwd())
            subparser.add_argument("--output", type=Path, required=True)
            if name == "workload-evidence":
                subparser.add_argument("--manifest", type=Path, required=True)
                subparser.add_argument("--activation", type=Path, required=True)
            elif name == "public-correctness":
                subparser.add_argument("--pairs", type=Path, required=True)
                subparser.add_argument("--ledger-output", type=Path, required=True)
                subparser.add_argument("--checker-artifact", type=Path, required=True)
                subparser.add_argument("--test-results", type=Path, required=True)
                subparser.add_argument("--semantics", type=Path)
            elif name == "baseline-evidence":
                subparser.add_argument("--manifest", type=Path, required=True)
                subparser.add_argument("--analysis", type=Path, required=True)
                subparser.add_argument("--runs", type=Path, required=True)
            else:
                subparser.add_argument("--agent-identity", type=Path, required=True)
                subparser.add_argument("--checkout-audit", type=Path, required=True)
                subparser.add_argument("--run-directory", type=Path, required=True)
            if name == "integrate":
                subparser.add_argument("--dry-run-exit-code", type=int, required=True)
                subparser.add_argument("--dry-run-exit-note")
        else:
            subparser.add_argument("--registry", type=Path, required=True)
            subparser.add_argument("--repository", required=True)
            subparser.add_argument("--pr-number", type=int, required=True)
            subparser.add_argument("--repository-checkout", type=Path, required=True)
            subparser.add_argument("--editable-root", action="append", required=True)
            subparser.add_argument("--patch-output", type=Path, required=True)
            subparser.add_argument("--audit-output", type=Path, required=True)
            subparser.add_argument("--intent-note")
            subparser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        {
            "inventory": _inventory,
            "harvest": _harvest,
            "import": _import,
            "audit": _audit,
            "queue": _queue,
            "triage": _triage,
            "screening-audit": _screening_audit,
            "enrich": _enrich,
            "checkout": _checkout,
            "screen": _screen,
            "source-verify": _source_verify,
            "workload-evidence": _workload_evidence,
            "public-correctness": _public_correctness,
            "baseline-evidence": _baseline_evidence,
            "integrate": _integration_evidence,
        }[args.command](args)
    except Exception as error:
        print(f"PR curation failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
