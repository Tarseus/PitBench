"""Harvest and audit real merged PRs for solver improvement task curation."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
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

from pitbench.repositories.base import SOURCE_SUFFIXES
from pitbench.repositories.classification import classify_pr_objectives
from pitbench.repositories.pr_curation import (
    append_jsonl,
    apply_selection_reviews,
    build_baseline_opportunity_evidence,
    build_mechanism_review_queue,
    build_objective_qualification_evidence,
    build_objective_scoring_registration,
    build_paired_correctness_evidence,
    build_selection_queue,
    build_task_integration_evidence,
    candidate_key,
    inspect_pinned_review_diff,
    load_repository_catalog,
    materialize_git_backed_checkout,
    merge_candidates,
    normalize_search_issue,
    read_jsonl,
    render_selection_review_card,
    render_selection_review_index,
    selection_review_draft,
    summarize_registry,
    validate_evidence_stage,
    validate_selection_review,
    validate_workload_exposure_evidence,
    verify_source_patch,
    workload_selection_rule_sha256,
    write_jsonl,
)
from pitbench.schema.objectives import objective_catalog, objective_catalog_sha256


def _type_audit(args: argparse.Namespace) -> None:
    catalog = load_repository_catalog(args.catalog)
    scopes = {row["repository"]: row["scope_status"] for row in catalog["repositories"]}
    taxonomy = objective_catalog()
    taxonomy_sha256 = objective_catalog_sha256()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    rows, counts, kind_counts, scope_counts = [], Counter(), Counter(), Counter()
    for record in read_jsonl(args.registry):
        candidate_key(record)
        classification = classify_pr_objectives(record, taxonomy)
        classification["taxonomy_sha256"] = taxonomy_sha256
        scope = scopes.get(record["repository"], "not_cataloged")
        counts.update(classification["objectives"] or ["unclassified"])
        kind_counts.update(classification["change_kinds"])
        scope_counts[scope] += 1
        rows.append(
            {
                "repository": record["repository"],
                "pr_number": record["pr_number"],
                "title": record.get("title", ""),
                "base_commit": record.get("base_commit"),
                "head_commit": record.get("head_commit"),
                "scope_status": scope,
                "classification": classification,
            }
        )
    records_path = output / "pr_type_hints.jsonl"
    write_jsonl(records_path, rows)
    summary = {
        "schema_version": 1,
        "input_registry_sha256": hashlib.sha256(args.registry.read_bytes()).hexdigest(),
        "taxonomy_version": taxonomy["taxonomy_version"],
        "taxonomy_sha256": taxonomy_sha256,
        "classification_implementation_sha256": hashlib.sha256(
            (
                Path(__file__).resolve().parent.parent
                / "pitbench/repositories/classification.py"
            ).read_bytes()
        ).hexdigest(),
        "records": len(rows),
        "counts_by_objective_hint": dict(sorted(counts.items())),
        "counts_by_change_kind_hint": dict(sorted(kind_counts.items())),
        "counts_by_scope_status": dict(sorted(scope_counts.items())),
        "classified_records_path": records_path.name,
        "classified_records_sha256": hashlib.sha256(
            records_path.read_bytes()
        ).hexdigest(),
        "claim_boundary": "Title hints across the retained registry, not semantic classifications or a random sample. Labels overlap. Scope, source, mathematical correctness, measurable benefit and task admission are not inferred. Unclassified records remain retained.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))


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
        scope_statuses=set(args.scope_status) if args.scope_status else None,
        review_order=args.review_order,
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


def _prepare_review_batch(args: argparse.Namespace) -> None:
    """Freeze a review cohort and prefill factual fields without deciding admission."""
    if args.limit < 1:
        raise ValueError("review batch limit must be positive")
    root = args.repository_root.resolve()
    output_dir = args.output_dir.resolve()
    if not output_dir.is_relative_to(root):
        raise ValueError("review batch output must be inside the repository")
    queue_bytes = args.queue.read_bytes()
    queue = json.loads(queue_bytes)
    selected = queue.get("queue")
    if not isinstance(selected, list):
        raise ValueError("review batch requires a triage queue")
    selected = selected[: args.limit]
    if not selected:
        raise ValueError("review batch queue is empty")
    manifest_path = output_dir / "manifest.json"
    previous_manifest = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    )
    catalog = load_repository_catalog(args.catalog)
    scope_status = {
        entry["repository"]: entry["scope_status"] for entry in catalog["repositories"]
    }
    registry_rows = read_jsonl(args.registry)
    rows = {candidate_key(row): row for row in registry_rows}
    if len(rows) != len(registry_rows):
        raise ValueError("duplicate PR in evidence registry")
    details: dict[tuple[str, int], dict[str, Any]] = {}
    detail_inputs = []
    for path in args.details:
        detail_inputs.append(
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
        for detail in read_jsonl(path):
            if detail.get("status") != "completed":
                continue
            key = candidate_key(detail)
            if key in details and details[key] != detail:
                raise ValueError(f"conflicting complete PR detail for {key}")
            details[key] = detail
    cards = []
    card_files = []
    drafts = []
    seen: set[tuple[str, int]] = set()
    for item in selected:
        key = candidate_key(item)
        if key in seen:
            raise ValueError(f"duplicate PR in review queue: {key}")
        seen.add(key)
        row = rows.get(key)
        detail = details.get(key)
        if row is None or detail is None or item.get("detail_sha256") is None:
            raise ValueError(f"{key} needs an indexed PR and complete triage detail")
        if key[0] not in scope_status:
            raise ValueError(f"{key} repository is absent from the catalog")
        digest = hashlib.sha256(
            json.dumps(detail, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if digest != item["detail_sha256"]:
            raise ValueError(f"{key} detail differs from frozen triage queue")
        if previous_manifest is None and (
            row.get("stage") != "discovered"
            or row.get("selection_decision") in {"reject", "needs_evidence"}
        ):
            raise ValueError(f"{key} already has a selection decision")
        draft = selection_review_draft(row, detail)
        draft_path = output_dir / "drafts" / key[0] / f"pr-{key[1]}.json"
        drafts.append((draft_path, draft))
        card = {
            "repository": key[0],
            "pr_number": key[1],
            "base_commit": row["base_commit"],
            "head_commit": row["head_commit"],
            "repository_scope_status": scope_status.get(key[0]),
            "review_lane": item.get("review_lane"),
            "signal_band": item.get("signal_band"),
            "solver_phase_path_hints": item.get("solver_phase_path_hints", []),
            "caution_flags": item.get("caution_flags", []),
            "related_pr_hints": item.get("related_pr_hints", []),
            "title": detail.get("title"),
            "body": detail.get("body"),
            "url": detail.get("url"),
            "changed_files": detail["files"],
            "changed_source_paths": draft["changed_source_paths"],
            "draft_path": draft_path.relative_to(root).as_posix(),
            "detail_sha256": digest,
        }
        card_path = output_dir / "cards" / key[0] / f"pr-{key[1]}.md"
        card_content = render_selection_review_card(card)
        card["card_path"] = card_path.relative_to(root).as_posix()
        card["card_sha256"] = hashlib.sha256(card_content.encode("utf-8")).hexdigest()
        cards.append(card)
        card_files.append((card_path, card_content))
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "registry_sha256_at_selection": hashlib.sha256(
            args.registry.read_bytes()
        ).hexdigest(),
        "queue_sha256": hashlib.sha256(queue_bytes).hexdigest(),
        "detail_inputs": detail_inputs,
        "cards_sha256": hashlib.sha256(
            "".join(json.dumps(card, sort_keys=True) + "\n" for card in cards).encode()
        ).hexdigest(),
        "cohort": [
            {
                "repository": card["repository"],
                "pr_number": card["pr_number"],
                "base_commit": card["base_commit"],
                "head_commit": card["head_commit"],
                "review_lane": card["review_lane"],
                "signal_band": card["signal_band"],
                "draft_path": card["draft_path"],
                "card_path": card["card_path"],
                "card_sha256": card["card_sha256"],
            }
            for card in cards
        ],
        "cohort_by_repository": dict(
            sorted(Counter(card["repository"] for card in cards).items())
        ),
        "cohort_by_signal_band": dict(
            sorted(Counter(str(card["signal_band"]) for card in cards).items())
        ),
        "cohort_by_review_lane": dict(
            sorted(Counter(str(card["review_lane"]) for card in cards).items())
        ),
        "claim_boundary": "Review cards and drafts do not approve PRs or count as tasks.",
    }
    if previous_manifest is not None:
        for field in ("queue_sha256", "detail_inputs", "cards_sha256", "cohort"):
            if previous_manifest.get(field) != manifest[field]:
                raise ValueError("existing review batch has a different frozen cohort")
        manifest = previous_manifest
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(output_dir / "cards.jsonl", cards)
    cards_path = output_dir / "cards.jsonl"
    if (
        not cards_path.is_file()
        or hashlib.sha256(cards_path.read_bytes()).hexdigest()
        != manifest["cards_sha256"]
    ):
        raise ValueError("existing review batch lacks its frozen review cards")
    for card_path, card_content in card_files:
        card_path.parent.mkdir(parents=True, exist_ok=True)
        if card_path.exists():
            if (
                hashlib.sha256(card_path.read_bytes()).hexdigest()
                != hashlib.sha256(card_content.encode("utf-8")).hexdigest()
            ):
                raise ValueError(f"review card differs from frozen inputs: {card_path}")
            continue
        temporary = card_path.with_suffix(".md.tmp")
        temporary.write_text(card_content, encoding="utf-8")
        temporary.replace(card_path)
    for draft_path, draft in drafts:
        draft_path.parent.mkdir(parents=True, exist_ok=True)
        if draft_path.exists():
            existing = json.loads(draft_path.read_text())
            if any(
                existing.get(field) != draft[field]
                for field in ("repository", "pr_number", "base_commit", "head_commit")
            ):
                raise ValueError(f"review draft identity differs at {draft_path}")
            continue
        temporary = draft_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(draft, indent=2) + "\n")
        temporary.replace(draft_path)
    if previous_manifest is None:
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(manifest_path)
    index_path = output_dir / "index.md"
    index_content = render_selection_review_index(cards, output_dir.relative_to(root))
    if index_path.exists():
        if index_path.read_text(encoding="utf-8") != index_content:
            raise ValueError(f"review index differs from frozen inputs: {index_path}")
    else:
        temporary = index_path.with_suffix(".md.tmp")
        temporary.write_text(index_content, encoding="utf-8")
        temporary.replace(index_path)
    print(
        json.dumps(
            {
                "cohort_size": len(manifest["cohort"]),
                "manifest": str(manifest_path),
                "index": str(index_path),
            },
            sort_keys=True,
        )
    )


def _review_batch_status(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema_version") != 1 or not isinstance(
        manifest.get("cohort"), list
    ):
        raise ValueError("unsupported review batch manifest")
    registry_rows = read_jsonl(args.registry)
    rows = {candidate_key(row): row for row in registry_rows}
    if len(rows) != len(registry_rows):
        raise ValueError("duplicate PR in evidence registry")
    stages = Counter()
    decisions = Counter()
    by_lane: dict[str, Counter[str]] = defaultdict(Counter)
    by_repository: dict[str, Counter[str]] = defaultdict(Counter)
    by_signal_band: dict[str, Counter[str]] = defaultdict(Counter)
    review_duration_sec = 0.0
    timed_reviews = 0
    seen: set[tuple[str, int]] = set()
    for item in manifest["cohort"]:
        key = candidate_key(item)
        if key in seen:
            raise ValueError(f"duplicate PR in review batch: {key}")
        seen.add(key)
        row = rows.get(key)
        if row is None or any(
            row.get(field) != item.get(field)
            for field in ("base_commit", "head_commit")
        ):
            raise ValueError(f"review batch PR identity differs from registry: {key}")
        stage = str(row.get("stage", "discovered"))
        decision = str(row.get("selection_decision") or "unreviewed")
        stages[stage] += 1
        decisions[decision] += 1
        by_lane[str(item.get("review_lane") or "unspecified")][stage] += 1
        by_repository[key[0]][stage] += 1
        by_signal_band[str(item.get("signal_band"))][stage] += 1
        review_path = row.get("selection_review_path")
        if review_path:
            review_file = (args.repository_root / review_path).resolve()
            if not review_file.is_relative_to(args.repository_root.resolve()):
                raise ValueError(f"review path escapes repository: {key}")
            review_bytes = review_file.read_bytes()
            if hashlib.sha256(review_bytes).hexdigest() != row.get(
                "selection_review_sha256"
            ):
                raise ValueError(f"review hash differs from registry: {key}")
            review = json.loads(review_bytes)
            duration = review.get("review_duration_sec")
            if duration is not None:
                if (
                    type(duration) not in {int, float}
                    or not math.isfinite(duration)
                    or duration < 0
                ):
                    raise ValueError(f"invalid review duration for {key}")
                timed_reviews += 1
                review_duration_sec += duration
    report = {
        "schema_version": 1,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "cohort_size": len(seen),
        "reviewed": len(seen) - decisions["unreviewed"],
        "task_ready": stages["task_ready"],
        "timed_reviews": timed_reviews,
        "reported_review_hours": round(review_duration_sec / 3600, 3),
        "stage_counts": dict(sorted(stages.items())),
        "decision_counts": dict(sorted(decisions.items())),
        "stage_counts_by_review_lane": {
            lane: dict(sorted(counts.items()))
            for lane, counts in sorted(by_lane.items())
        },
        "stage_counts_by_repository": {
            repository: dict(sorted(counts.items()))
            for repository, counts in sorted(by_repository.items())
        },
        "stage_counts_by_signal_band": {
            band: dict(sorted(counts.items()))
            for band, counts in sorted(by_signal_band.items())
        },
        "claim_boundary": "This report describes a frozen review cohort; it does not estimate unseen PR yield.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


def _cache_local_diffs(args: argparse.Namespace) -> None:
    """Cache pinned source diffs from Git objects, optionally fetching missing SHAs."""
    if args.limit < 1 or args.git_timeout_sec <= 0:
        raise ValueError("Git diff cache limit and timeout must be positive")
    root = args.repository_root.resolve()
    output_dir = args.output_dir.resolve()
    if not output_dir.is_relative_to(root):
        raise ValueError("local diff cache must be inside the repository")
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported review batch manifest")
    cards_path = args.manifest.parent / "cards.jsonl"
    if hashlib.sha256(cards_path.read_bytes()).hexdigest() != manifest.get(
        "cards_sha256"
    ):
        raise ValueError("review cards differ from frozen manifest")
    cards = read_jsonl(cards_path)
    if len(cards) != len(manifest["cohort"]):
        raise ValueError("review cards differ from frozen cohort")
    registry = {candidate_key(row): row for row in read_jsonl(args.registry)}
    checkouts: dict[str, Path] = {}
    for mapping in args.repository_checkout:
        repository, separator, raw_path = mapping.partition("=")
        if not separator or not repository or not raw_path or repository in checkouts:
            raise ValueError(f"invalid repository checkout mapping: {mapping!r}")
        path = Path(raw_path)
        checkouts[repository] = (
            (root / path).resolve() if not path.is_absolute() else path
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.jsonl"
    previous = {candidate_key(row): row for row in read_jsonl(status_path)}
    saved = dict(previous)

    def save_status(record: dict[str, Any]) -> None:
        saved[candidate_key(record)] = record
        write_jsonl(
            status_path,
            [
                saved[candidate_key(item)]
                for item in manifest["cohort"]
                if candidate_key(item) in saved
            ],
        )

    results = []
    counts = Counter()
    index = [
        "# Local source diffs",
        "",
        "These pinned source diffs support human review; they do not approve PRs.",
        "",
    ]
    selected = list(zip(manifest["cohort"], cards))[: args.limit]
    probe_env = {**os.environ, "GIT_NO_LAZY_FETCH": "1", "GIT_TERMINAL_PROMPT": "0"}
    for ordinal, (item, card) in enumerate(selected, 1):
        key = candidate_key(item)
        if candidate_key(card) != key or any(
            card.get(field) != item.get(field)
            for field in ("base_commit", "head_commit", "card_sha256")
        ):
            raise ValueError(f"review card differs from frozen cohort: {key}")
        row = registry.get(key)
        if row is None or any(
            row.get(field) != item.get(field)
            for field in ("base_commit", "head_commit")
        ):
            raise ValueError(f"registry identity differs from review card: {key}")
        checkout = checkouts.get(key[0])
        prior = previous.get(key)
        same_identity = prior is not None and (
            prior.get("card_sha256") == item["card_sha256"]
            and prior.get("diff_identity_version") == 4
            and prior.get("merge_commit") == row.get("merge_commit")
            and prior.get("checkout") == (str(checkout) if checkout else None)
        )
        if (
            same_identity
            and prior.get("status") == "unfetchable_merge_commit"
            and not args.retry_unavailable
        ):
            results.append(prior)
            counts[prior["status"]] += 1
            card_link = os.path.relpath(root / item["card_path"], output_dir)
            index.append(
                f"{ordinal}. [{key[0]} #{key[1]}]({card_link}): {prior['status']}"
            )
            (output_dir / "index.md").write_text("\n".join(index) + "\n")
            continue
        if same_identity and prior.get("status") in {"cached", "cached_branch_base"}:
            patch_path = root / str(prior.get("patch_path"))
            if patch_path.is_file() and hashlib.sha256(
                patch_path.read_bytes()
            ).hexdigest() == prior.get("patch_sha256"):
                results.append(prior)
                counts[prior["status"]] += 1
                card_link = os.path.relpath(root / item["card_path"], output_dir)
                patch_link = os.path.relpath(patch_path, output_dir)
                index.append(
                    f"{ordinal}. [{key[0]} #{key[1]}]({card_link}): {prior['status']} "
                    f"([source diff]({patch_link}))"
                )
                save_status(prior)
                (output_dir / "index.md").write_text("\n".join(index) + "\n")
                continue
        fetch_failed = False
        fetch_failure_status = "git_fetch_failed"
        integration_base_candidate = None
        if (
            args.fetch_missing
            and checkout is not None
            and card.get("changed_source_paths")
        ):
            remote = subprocess.run(
                ["git", "-C", str(checkout), "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
                check=False,
            )
            expected_remote = f"https://github.com/{key[0]}"
            if (
                remote.returncode
                or remote.stdout.strip().removesuffix(".git") != expected_remote
            ):
                raise ValueError(
                    f"{key} checkout origin is not the pinned GitHub repository"
                )
            missing = [
                commit
                for commit in (
                    item["base_commit"],
                    item["head_commit"],
                    row.get("merge_commit"),
                )
                if commit
                if subprocess.run(
                    [
                        "git",
                        "-C",
                        str(checkout),
                        "cat-file",
                        "-e",
                        f"{commit}^{{commit}}",
                    ],
                    capture_output=True,
                    check=False,
                    env=probe_env,
                ).returncode
            ]
            if missing:
                try:
                    fetched = subprocess.run(
                        [
                            "git",
                            "-C",
                            str(checkout),
                            "fetch",
                            "--filter=blob:none",
                            "--no-tags",
                            "--depth=1",
                            "origin",
                            *missing,
                        ],
                        capture_output=True,
                        check=False,
                        timeout=args.git_timeout_sec,
                    )
                    fetch_failed = fetched.returncode != 0
                    if (
                        fetch_failed
                        and missing == [row.get("merge_commit")]
                        and b"not our ref" in fetched.stderr
                    ):
                        fetch_failure_status = "unfetchable_merge_commit"
                except subprocess.TimeoutExpired:
                    fetch_failed = True
            merge = row.get("merge_commit")
            if merge and not fetch_failed:
                metadata = subprocess.run(
                    ["git", "-C", str(checkout), "cat-file", "-p", merge],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                parents = [
                    line[7:]
                    for line in metadata.stdout.splitlines()
                    if line.startswith("parent ")
                ]
                integration_base_candidate = parents[0] if parents else None
                if parents and parents[0] != item["base_commit"]:
                    integration_base = parents[0]
                    present = subprocess.run(
                        [
                            "git",
                            "-C",
                            str(checkout),
                            "cat-file",
                            "-e",
                            f"{integration_base}^{{commit}}",
                        ],
                        capture_output=True,
                        check=False,
                        env=probe_env,
                    )
                    if present.returncode:
                        try:
                            fetched = subprocess.run(
                                [
                                    "git",
                                    "-C",
                                    str(checkout),
                                    "fetch",
                                    "--filter=blob:none",
                                    "--no-tags",
                                    "--depth=1",
                                    "origin",
                                    integration_base,
                                ],
                                capture_output=True,
                                check=False,
                                timeout=args.git_timeout_sec,
                            )
                            fetch_failed = fetched.returncode != 0
                        except subprocess.TimeoutExpired:
                            fetch_failed = True
        status, patch, observed = (
            (fetch_failure_status, None, [])
            if fetch_failed
            else inspect_pinned_review_diff(
                card,
                checkout,
                merge_commit=row.get("merge_commit"),
                timeout_sec=args.git_timeout_sec,
            )
            if checkout is not None
            else ("missing_checkout", None, [])
        )
        record = {
            "repository": key[0],
            "pr_number": key[1],
            "base_commit": item["base_commit"],
            "head_commit": item["head_commit"],
            "merge_commit": row.get("merge_commit"),
            "card_sha256": item["card_sha256"],
            "diff_identity_version": 4,
            "status": status,
            "observed_changed_paths": [] if status == "base_drift" else observed,
            "integration_base_commit": (
                integration_base_candidate
                or (
                    observed[0]
                    if status in {"base_drift", "missing_integration_base_commit"}
                    and observed
                    else None
                )
            ),
            "checkout": str(checkout) if checkout is not None else None,
            "patch_path": None,
            "patch_sha256": None,
        }
        if patch is not None:
            path = output_dir / "diffs" / key[0] / f"pr-{key[1]}.patch"
            path.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256(patch).hexdigest()
            if path.exists():
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise ValueError(f"cached source diff changed: {path}")
            else:
                temporary = path.with_suffix(".patch.tmp")
                temporary.write_bytes(patch)
                temporary.replace(path)
            record["patch_path"] = path.relative_to(root).as_posix()
            record["patch_sha256"] = digest
        results.append(record)
        counts[status] += 1
        card_link = os.path.relpath(root / item["card_path"], output_dir)
        line = f"{ordinal}. [{key[0]} #{key[1]}]({card_link}): {status}"
        if record["patch_path"]:
            patch_link = os.path.relpath(root / record["patch_path"], output_dir)
            line += f" ([source diff]({patch_link}))"
        index.append(line)
        save_status(record)
        (output_dir / "index.md").write_text("\n".join(index) + "\n")
    print(
        json.dumps(
            {
                "cohort_size": len(results),
                "status_counts": dict(sorted(counts.items())),
                "index": str(output_dir / "index.md"),
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
        "low_signal_advanced": sum(bands[band]["advance"] for band in ("2", "3")),
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
    validate_evidence_stage(row, args.repository_root)
    summary = summarize_registry(
        rows,
        catalog["target_validated_pr_tasks"],
        args.repository_root,
        validate_evidence_files=False,
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
    validate_evidence_stage(row, args.repository_root)
    summary = summarize_registry(
        rows,
        catalog["target_validated_pr_tasks"],
        args.repository_root,
        validate_evidence_files=False,
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


def _objective_register(args: argparse.Namespace) -> None:
    load_repository_catalog(args.catalog)
    row, rows = _evidence_candidate(
        args,
        (
            "source_verified",
            "build_verified",
            "workload_verified",
            "correctness_verified",
        ),
    )
    validate_evidence_stage(row, args.repository_root)
    destination, relative = _evidence_output(args.output, args.repository_root)
    evidence = build_objective_scoring_registration(row, args.repository_root)
    destination.write_text(json.dumps(evidence, indent=2) + "\n")
    if row.get("objective_registration_path"):
        row.setdefault("objective_registration_history", []).append(
            {
                "path": row["objective_registration_path"],
                "sha256": row["objective_registration_sha256"],
            }
        )
    row["objective_registration_path"] = relative
    row["objective_registration_sha256"] = hashlib.sha256(
        destination.read_bytes()
    ).hexdigest()
    row["objective_status"] = (
        "needs_protocol" if evidence["protocol_gaps"] else "registered"
    )
    write_jsonl(args.registry, rows)
    print(
        json.dumps(
            {
                "task_id": row["task_id"],
                "stage": row["stage"],
                "objective_status": row["objective_status"],
                "protocol_gaps": evidence["protocol_gaps"],
            }
        )
    )


def _objective_qualify(args: argparse.Namespace) -> None:
    load_repository_catalog(args.catalog)
    row, rows = _evidence_candidate(args)
    validate_evidence_stage(row, args.repository_root)
    if not row.get("objective_registration_path"):
        raise ValueError(
            "register the task scoring plan before collecting qualification runs"
        )
    destination, relative = _evidence_output(args.output, args.repository_root)
    evidence = build_objective_qualification_evidence(
        row,
        args.repository_root,
        _repository_path(args.run_directory, args.repository_root),
    )
    destination.write_text(json.dumps(evidence, indent=2) + "\n")
    report = evidence["objective_qualification"]
    row["objective_evidence_path"] = relative
    row["objective_status"] = report["classification"]
    # Open opportunities retain the independent Base-only admission route.
    if (
        report["performance_confirmed"]
        and row.get("task_track", "expert_reference") == "expert_reference"
    ):
        row["performance_evidence_path"] = relative
        row["stage"] = "performance_confirmed"
        validate_evidence_stage(row, args.repository_root)
    write_jsonl(args.registry, rows)
    print(
        json.dumps(
            {
                "task_id": row["task_id"],
                "stage": row["stage"],
                "objective_qualification": report,
            }
        )
    )


def _integration_evidence(args: argparse.Namespace) -> None:
    catalog = load_repository_catalog(args.catalog)
    row, rows = _evidence_candidate(
        args, ("correctness_verified", "performance_confirmed")
    )
    if row.get("task_track") == "open_opportunity" and not row.get(
        "opportunity_evidence_path"
    ):
        raise ValueError("real Judge integration requires hidden baseline evidence")
    if (
        row.get("task_track", "expert_reference") == "expert_reference"
        and row["stage"] != "performance_confirmed"
    ):
        raise ValueError(
            "expert integration requires confirmed objective/performance evidence"
        )
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
    validate_evidence_stage(row, args.repository_root)
    summary = summarize_registry(
        rows,
        catalog["target_validated_pr_tasks"],
        args.repository_root,
        validate_evidence_files=False,
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
    review_paths = list(args.review_file or [])
    if args.review_dir is not None:
        review_paths.extend(sorted(args.review_dir.rglob("*.json")))
    decisions = apply_selection_reviews(
        rows, catalog, review_paths, args.repository_root
    )
    summary = summarize_registry(
        rows,
        catalog["target_validated_pr_tasks"],
        args.repository_root,
        validate_evidence_files=False,
    )
    write_jsonl(args.registry, rows)
    if len(decisions) == 1:
        report = {
            **decisions[0],
            "remaining_task_ready_prs": summary["remaining_task_ready_prs"],
        }
    else:
        report = {
            "screened": len(decisions),
            "decision_counts": dict(Counter(item["decision"] for item in decisions)),
            "remaining_task_ready_prs": summary["remaining_task_ready_prs"],
        }
    print(json.dumps(report, sort_keys=True))


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
                        "patch": (
                            file["patch"]
                            if isinstance(file.get("patch"), str)
                            and Path(str(file.get("filename", ""))).suffix.lower()
                            in SOURCE_SUFFIXES
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
        append_jsonl(args.output, record)
        if record["status"] in {"completed", "manual_large_pr"}:
            completed.add(key)
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
        merged,
        catalog["target_validated_pr_tasks"],
        args.repository_root,
        validate_evidence_files=False,
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
        intent_note = review.get("solver_behavior_hypothesis") or review.get("reason", "")
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
    validate_evidence_stage(row, args.repository_root)
    summary = summarize_registry(
        rows,
        catalog["target_validated_pr_tasks"],
        args.repository_root,
        validate_evidence_files=False,
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
    review_path = _repository_path(
        args.repository_root / row["selection_review_path"], args.repository_root
    )
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
        "instance_selection_rule_sha256": workload_selection_rule_sha256(review, row, root),
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
    updated_row = {
        **row,
        "workload_evidence_path": relative,
        "stage": "workload_verified",
    }
    try:
        validate_evidence_stage(updated_row, root)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    rows[rows.index(row)] = updated_row
    summary = summarize_registry(
        rows,
        catalog["target_validated_pr_tasks"],
        root,
        validate_evidence_files=False,
    )
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
        "type-audit",
        "prepare-review-batch",
        "review-batch-status",
        "cache-local-diffs",
        "cache-git-diffs",
        "screening-audit",
        "enrich",
        "checkout",
        "screen",
        "source-verify",
        "workload-evidence",
        "public-correctness",
        "baseline-evidence",
        "objective-register",
        "objective-qualify",
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
        elif name == "type-audit":
            subparser.add_argument("--registry", type=Path, required=True)
            subparser.add_argument("--output-dir", type=Path, required=True)
        elif name in {"triage", "screening-audit"}:
            subparser.add_argument("--registry", type=Path, required=True)
            subparser.add_argument(
                "--details", type=Path, action="append", required=True
            )
            if name == "triage":
                subparser.add_argument("--limit", type=int, default=100)
                subparser.add_argument("--audit-fraction", type=float, default=0.1)
                subparser.add_argument(
                    "--scope-status",
                    action="append",
                    choices=("task_backed", "approved_source", "pending_domain_review"),
                    help="Schedule reviews from these catalog states; other PRs remain eligible",
                )
                subparser.add_argument(
                    "--review-order",
                    choices=("coverage", "throughput"),
                    default="coverage",
                    help="Balance source coverage or prioritize explicit optimization signals",
                )
            subparser.add_argument("--output", type=Path, required=True)
        elif name == "prepare-review-batch":
            subparser.add_argument("--registry", type=Path, required=True)
            subparser.add_argument(
                "--details", type=Path, action="append", required=True
            )
            subparser.add_argument("--queue", type=Path, required=True)
            subparser.add_argument("--limit", type=int, default=100)
            subparser.add_argument("--output-dir", type=Path, required=True)
            subparser.add_argument("--repository-root", type=Path, default=Path.cwd())
        elif name == "review-batch-status":
            subparser.add_argument("--registry", type=Path, required=True)
            subparser.add_argument("--manifest", type=Path, required=True)
            subparser.add_argument("--output", type=Path, required=True)
            subparser.add_argument("--repository-root", type=Path, default=Path.cwd())
        elif name in {"cache-local-diffs", "cache-git-diffs"}:
            subparser.add_argument("--registry", type=Path, required=True)
            subparser.add_argument("--manifest", type=Path, required=True)
            subparser.add_argument(
                "--repository-checkout", action="append", required=True
            )
            subparser.add_argument("--output-dir", type=Path, required=True)
            subparser.add_argument("--repository-root", type=Path, default=Path.cwd())
            subparser.add_argument("--fetch-missing", action="store_true")
            subparser.add_argument("--retry-unavailable", action="store_true")
            subparser.add_argument("--limit", type=int, default=1000000)
            subparser.add_argument("--git-timeout-sec", type=float, default=120)
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
            subparser.add_argument("--review-file", type=Path, action="append")
            subparser.add_argument("--review-dir", type=Path)
            subparser.add_argument("--repository-root", type=Path, default=Path.cwd())
        elif name in {
            "workload-evidence",
            "public-correctness",
            "baseline-evidence",
            "objective-register",
            "objective-qualify",
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
            elif name == "objective-register":
                pass
            elif name == "objective-qualify":
                subparser.add_argument("--run-directory", type=Path, required=True)
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
            "type-audit": _type_audit,
            "prepare-review-batch": _prepare_review_batch,
            "review-batch-status": _review_batch_status,
            "cache-local-diffs": _cache_local_diffs,
            "cache-git-diffs": _cache_local_diffs,
            "screening-audit": _screening_audit,
            "enrich": _enrich,
            "checkout": _checkout,
            "screen": _screen,
            "source-verify": _source_verify,
            "workload-evidence": _workload_evidence,
            "public-correctness": _public_correctness,
            "baseline-evidence": _baseline_evidence,
            "objective-register": _objective_register,
            "objective-qualify": _objective_qualify,
            "integrate": _integration_evidence,
        }[args.command](args)
    except Exception as error:
        print(f"PR curation failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
