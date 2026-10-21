from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml

import pitbench.repositories.pr_curation as pr_curation
from pitbench.repositories.pr_curation import (
    build_baseline_opportunity_evidence,
    build_mechanism_review_queue,
    build_paired_correctness_evidence,
    build_selection_queue,
    build_task_integration_evidence,
    load_repository_catalog,
    merge_candidates,
    summarize_registry,
    validate_evidence_stage,
    validate_independent_correctness_evidence,
    validate_open_opportunity_evidence,
    validate_pinned_build_evidence,
    validate_selection_review,
    validate_task_integration_evidence,
    validate_workload_exposure_evidence,
)


@pytest.mark.parametrize(
    ("history", "expected_status"),
    [
        ("merge", "cached"),
        ("squash", "cached"),
        ("advanced_base", "cached"),
        ("equivalent_branch", "cached_branch_base"),
        ("squash_equivalent_branch", "cached_branch_base"),
        ("squash_changed_context", "base_drift"),
        ("changed_context", "base_drift"),
        ("extra_file", "pinned_file_mismatch"),
    ],
)
def test_review_diff_tracks_actual_git_integration(tmp_path, history, expected_status):
    repo = tmp_path / "solver"
    repo.mkdir()

    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=repo, text=True, stderr=subprocess.PIPE
        ).strip()

    git("init", "-q")
    git("config", "user.name", "PitBench test")
    git("config", "user.email", "test@example.invalid")
    (repo / "src").mkdir()
    source = repo / "src/engine.cpp"
    source.write_text("first\n" + "middle\n" * 12 + "before\n")
    (repo / "build.txt").write_text("initial\n")
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    git("checkout", "-qb", "feature")
    source.write_text(source.read_text().replace("before", "after"))
    if history == "extra_file":
        (repo / "src/other.cpp").write_text("another change\n")
    git("add", ".")
    git("commit", "-qm", "optimization")
    head = git("rev-parse", "HEAD")
    git("checkout", "-q", "--detach", base)
    if history in {"advanced_base", "equivalent_branch", "changed_context", "squash_equivalent_branch", "squash_changed_context"}:
        if history in {"changed_context", "squash_changed_context"}:
            source.write_text(source.read_text().replace("first", "new first"))
        else:
            (repo / "build.txt").write_text("updated\n")
        git("add", ".")
        git("commit", "-qm", "base advances")
    integration_base = git("rev-parse", "HEAD")
    if history == "advanced_base":
        base = integration_base
    if history.startswith("squash"):
        git("merge", "--squash", head)
        git("commit", "-qm", "squash optimization")
    else:
        git("merge", "--no-ff", "--no-edit", head)
    merge = git("rev-parse", "HEAD")
    card = {
        "base_commit": base,
        "head_commit": head,
        "changed_source_paths": ["src/engine.cpp"],
        "changed_files": [{"path": "src/engine.cpp"}],
    }
    status, patch, _ = pr_curation.inspect_pinned_review_diff(
        card, repo, merge_commit=merge
    )
    assert status == expected_status
    if status not in {"cached", "cached_branch_base"}:
        assert patch is None
        if history == "changed_context":
            with pytest.raises(ValueError, match="source patches differ"):
                pr_curation.verify_source_patch(
                    repository_checkout=repo,
                    record={**card, "repository": "owner/solver", "pr_number": 1,
                            "merge_commit": merge},
                    editable_roots=["src"],
                    patch_path=tmp_path / "rejected.patch",
                    audit_path=tmp_path / "rejected.json",
                    intent_note="Reduce repeated search work",
                )
            assert not (tmp_path / "rejected.patch").exists()
        return
    audit = pr_curation.verify_source_patch(
        repository_checkout=repo,
        record={**card, "repository": "owner/solver", "pr_number": 1,
                "merge_commit": merge},
        editable_roots=["src"],
        patch_path=tmp_path / "source.patch",
        audit_path=tmp_path / "source.json",
        intent_note="Reduce repeated search work",
    )
    assert (tmp_path / "source.patch").read_bytes() == patch
    assert audit["integrated_commit"] == merge
    assert audit["integration_base_commit"] == integration_base
    assert audit["changed_source_files"] == ["src/engine.cpp"]
    assert git("rev-parse", "HEAD") == merge


def test_interrupted_diff_cache_keeps_unvisited_results(tmp_path, monkeypatch):
    from argparse import Namespace

    import scripts.curate_solver_prs as cli

    batch = tmp_path / "batch"
    batch.mkdir()
    checkout = tmp_path / "mirror"
    checkout.mkdir()
    output = batch / "diffs"
    output.mkdir()
    cards = [
        {"repository": "owner/solver", "pr_number": number,
         "base_commit": "a" * 40, "head_commit": "b" * 40,
         "card_sha256": str(number) * 64, "card_path": f"batch/{number}.md",
         "changed_source_paths": ["src/engine.cpp"]}
        for number in (1, 2, 3)
    ]
    cli.write_jsonl(batch / "cards.jsonl", cards)
    manifest = {
        "schema_version": 1, "cohort": cards,
        "cards_sha256": hashlib.sha256((batch / "cards.jsonl").read_bytes()).hexdigest(),
    }
    (batch / "manifest.json").write_text(json.dumps(manifest))
    cli.write_jsonl(tmp_path / "registry.jsonl", cards)
    prior = [
        {**card, "status": "git_timeout", "diagnostic": f"prior-{card['pr_number']}"}
        for card in cards
    ]
    cli.write_jsonl(output / "status.jsonl", prior)

    def inspect(card, *_args, **_kwargs):
        if card["pr_number"] == 2:
            raise RuntimeError("interrupted")
        return "cached", b"source patch", ["src/engine.cpp"]

    monkeypatch.setattr(cli, "inspect_pinned_review_diff", inspect)
    args = Namespace(
        limit=3, git_timeout_sec=10, repository_root=tmp_path, output_dir=output,
        manifest=batch / "manifest.json", registry=tmp_path / "registry.jsonl",
        repository_checkout=[f"owner/solver={checkout}"], fetch_missing=False,
        retry_unavailable=False,
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        cli._cache_local_diffs(args)
    saved = cli.read_jsonl(output / "status.jsonl")
    assert len(saved) == 3
    assert saved[0]["status"] == "cached"
    assert saved[1:] == prior[1:]

    def finish(card, *_args, **_kwargs):
        assert card["pr_number"] != 1, "the completed diff must be reused"
        return "cached", b"source patch", ["src/engine.cpp"]

    monkeypatch.setattr(cli, "inspect_pinned_review_diff", finish)
    cli._cache_local_diffs(args)
    assert [row["status"] for row in cli.read_jsonl(output / "status.jsonl")] == ["cached"] * 3


def test_repeated_discovery_does_not_inflate_distinct_pr_count() -> None:
    candidate = {
        "repository": "PyVRP/PyVRP",
        "pr_number": 1156,
        "title": "Group neighbours by route in perturbation",
        "url": "https://github.com/PyVRP/PyVRP/pull/1156",
        "merged_at": "2026-08-10T08:28:08Z",
        "matched_queries": ["all_merged"],
        "stage": "discovered",
    }
    records = merge_candidates(
        [candidate, {**candidate, "matched_queries": ["performance in:title,body"]}]
    )

    assert len(records) == 1
    assert len(records[0]["matched_queries"]) == 2
    assert summarize_registry(records, 957)["stage_counts"]["task_ready"] == 0


def test_full_inventory_enriches_earlier_search_record_without_downgrading() -> None:
    identity = {
        "repository": "PyVRP/PyVRP",
        "pr_number": 1156,
        "url": "https://github.com/PyVRP/PyVRP/pull/1156",
        "merged_at": "2026-08-10T08:28:08Z",
    }
    records = merge_candidates(
        [
            {**identity, "stage": "source_verified", "matched_queries": ["manual"]},
            {
                **identity,
                "stage": "discovered",
                "matched_queries": ["all_merged"],
                "base_commit": "a" * 40,
                "head_commit": "b" * 40,
            },
        ]
    )

    assert len(records) == 1
    assert records[0]["stage"] == "source_verified"
    assert records[0]["base_commit"] == "a" * 40
    assert records[0]["matched_queries"] == ["all_merged", "manual"]


def test_task_ready_claim_requires_every_evidence_gate() -> None:
    candidate = {
        "repository": "PyVRP/PyVRP",
        "pr_number": 1156,
        "merged_at": "2026-08-10T08:28:08Z",
        "stage": "task_ready",
        "task_id": "pyvrp_search_20260806",
        "task_config_path": "configs/tasks/pyvrp_search_20260806.yaml",
    }

    with pytest.raises(ValueError, match="lacks intent_evidence_path"):
        summarize_registry([candidate], 957)


def test_same_snapshot_cannot_count_identical_agent_requests(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(pr_curation, "validate_evidence_stage", lambda row, root: None)
    task = {
        "instruction": "Improve MIP solver performance",
        "optimization_scope": "mip_search",
        "problem_family": "mip",
        "repository": {"editable_paths": ["src"]},
        "evaluation": {
            "performance_protocol": "exact_verified_solve",
            "budgets_sec": [5.0, 20.0],
        },
        "instance_sets": [
            {
                "name": "judge_id",
                "kind": "judge_id",
                "size": 30,
                "instance_set_config_sha256": "c" * 64,
            }
        ],
    }
    (tmp_path / "first.yaml").write_text(yaml.safe_dump({**task, "task_id": "first"}))
    (tmp_path / "second.yaml").write_text(yaml.safe_dump({**task, "task_id": "second"}))
    rows = [
        {
            "repository": "owner/solver",
            "pr_number": number,
            "base_commit": "a" * 40,
            "merged_at": f"2026-01-0{number}T00:00:00Z",
            "stage": "task_ready",
            "task_config_path": path,
        }
        for number, path in ((1, "first.yaml"), (2, "second.yaml"))
    ]
    with pytest.raises(ValueError, match="duplicate Agent task input"):
        summarize_registry(rows, 1001, tmp_path)

    (tmp_path / "second.yaml").write_text(
        yaml.safe_dump(
            {**task, "task_id": "second", "instruction": "Improve MIP branching"}
        )
    )
    with pytest.raises(ValueError, match="documented distinct goal"):
        summarize_registry(rows, 1001, tmp_path)

    (tmp_path / "review.json").write_text(
        json.dumps(
            {
                "same_snapshot_distinction": "Different branching target and workload",
                "related_prs": [1],
            }
        )
    )
    rows[1]["selection_review_path"] = "review.json"
    assert summarize_registry(rows, 1001, tmp_path)["stage_counts"]["task_ready"] == 2

    rows[1].pop("selection_review_path")
    (tmp_path / "first_review.json").write_text(
        json.dumps(
            {
                "same_snapshot_distinction": "First task targets a distinct search phase",
                "related_prs": [2],
            }
        )
    )
    rows[0]["selection_review_path"] = "first_review.json"
    assert summarize_registry(rows, 1001, tmp_path)["stage_counts"]["task_ready"] == 2


def test_selection_requires_solver_mechanism_and_independent_check_plan() -> None:
    candidate = {
        "repository": "ERGO-Code/HiGHS",
        "pr_number": 2861,
        "base_commit": "a" * 40,
        "head_commit": "b" * 40,
    }
    review = {
        "schema_version": 2,
        **candidate,
        "decision": "advance",
        "reason": "Changes enumeration presolve used during MIP search.",
        "repository_role": "solver_engine",
        "mechanism": "algorithmic_search",
        "solver_phase": "presolve",
        "solver_entry_points": ["HPresolve::enumerateSolutions"],
        "effect_channels": ["search_tree", "proof_time"],
        "correctness_failure_modes": ["invalid_bound"],
        "changed_source_paths": ["highs/presolve/HPresolve.cpp"],
        "solver_behavior_hypothesis": "Fewer duplicate enumeration operations.",
        "causal_chain": "Better row selection changes reductions and the search tree.",
        "structural_trigger": "Binary MIP rows eligible for enumeration.",
        "activation_probe": "Count enumeration calls and reductions.",
        "instance_selection_rule": "Freeze structural families before hidden scoring.",
        "workload_exposure_plan": "MIP instances exercising enumeration presolve.",
        "mathematical_risk": "A missed reduction can change primal or dual bounds.",
        "independent_check_plan": "Check original model solutions and certified bounds.",
        "metric_family": "exact",
        "hidden_generalization_plan": "Held-out MIP families and seeds.",
        "overlap_review": "Inspect adjacent presolve PRs for shared source changes.",
        "overlap_group": "highs-enumeration",
        "related_prs": [],
        "resource_risk": "Additional row scoring work.",
        "source_evidence_urls": ["https://github.com/ERGO-Code/HiGHS/pull/2861"],
    }
    assert validate_selection_review(review, candidate) == "advance"
    invalid_channels = {**review, "effect_channels": ["latency"]}
    with pytest.raises(ValueError, match="effect channels"):
        validate_selection_review(invalid_channels, candidate)
    invalid_risk = {**review, "correctness_failure_modes": ["invalid_primal"]}
    with pytest.raises(ValueError, match="correctness risk for its solver phase"):
        validate_selection_review(invalid_risk, candidate)
    del review["independent_check_plan"]
    with pytest.raises(ValueError, match="independent_check_plan"):
        validate_selection_review(review, candidate)


def test_selection_queue_balances_repositories_without_promoting_prs() -> None:
    catalog = {
        "repositories": [
            {"repository": "owner/solver", "scope_status": "task_backed"},
            {"repository": "owner/other", "scope_status": "pending_domain_review"},
        ]
    }
    rows = [
        {
            "repository": "owner/solver",
            "pr_number": number,
            "title": "Improve branch performance",
            "stage": "discovered",
        }
        for number in range(1, 4)
    ] + [
        {
            "repository": "owner/other",
            "pr_number": 4,
            "title": "Speed up presolve",
            "stage": "discovered",
        },
        {
            "repository": "owner/other",
            "pr_number": 5,
            "title": "Update CI cache",
            "stage": "discovered",
            "selection_decision": "reject",
        },
        {
            "repository": "owner/other",
            "pr_number": 6,
            "title": "Investigate solver path",
            "stage": "discovered",
            "selection_decision": "needs_evidence",
        },
    ]
    result = build_selection_queue(rows, catalog, 3)
    assert result["eligible_discovered_prs"] == 4
    assert result["pending_evidence_prs"] == 1
    assert [(item["repository"], item["pr_number"]) for item in result["queue"]] == [
        ("owner/solver", 1),
        ("owner/other", 4),
        ("owner/solver", 2),
    ]
    assert all(row["stage"] == "discovered" for row in rows)


def test_mechanism_triage_audits_low_signal_prs_without_admitting_them() -> None:
    catalog = {
        "repositories": [
            {"repository": "owner/solver", "scope_status": "task_backed"},
            {"repository": "owner/other", "scope_status": "approved_source"},
        ]
    }
    rows = [
        {
            "repository": "owner/solver" if number < 10 else "owner/other",
            "pr_number": number,
            "title": "Update solver internals",
            "stage": "discovered",
            "base_commit": "a" * 40,
            "head_commit": "b" * 40,
        }
        for number in range(1, 13)
    ]
    details = [
        {
            "repository": row["repository"],
            "pr_number": row["pr_number"],
            "base_commit": row["base_commit"],
            "head_commit": row["head_commit"],
            "status": "completed",
            "files": [
                {
                    "path": (
                        "src/search/Branching.cpp"
                        if row["pr_number"] < 11
                        else "docs/benchmark.md"
                    )
                }
            ],
        }
        for row in rows
    ]

    result = build_mechanism_review_queue(rows, catalog, details, 10)

    assert result["queue_size"] == 10
    assert result["mechanism_hint_pool"] == 10
    assert result["low_signal_pool"] == 2
    assert (
        sum(item["review_lane"] == "low_signal_audit" for item in result["queue"]) == 1
    )
    assert all(item["signal_band"] == 0 for item in result["queue"][:-1])
    assert all(row["stage"] == "discovered" for row in rows)
    assert result == build_mechanism_review_queue(rows, catalog, details, 10)


def test_low_signal_audit_spans_repositories_before_repeating_one() -> None:
    catalog = {
        "repositories": [
            {"repository": repository, "scope_status": "task_backed"}
            for repository in ("owner/first", "owner/second")
        ]
    }
    rows = [
        {
            "repository": repository,
            "pr_number": number,
            "title": "Update internals",
            "stage": "discovered",
            "base_commit": "a" * 40,
            "head_commit": "b" * 40,
        }
        for repository in ("owner/first", "owner/second")
        for number in range(1, 6)
    ]
    details = [
        {
            "repository": row["repository"],
            "pr_number": row["pr_number"],
            "base_commit": row["base_commit"],
            "head_commit": row["head_commit"],
            "status": "completed",
            "files": [{"path": "docs/notes.md"}],
        }
        for row in rows
    ]

    queue = build_mechanism_review_queue(
        rows, catalog, details, 10, audit_fraction=0.2
    )["queue"]

    assert [item["review_lane"] for item in queue[:2]] == [
        "low_signal_audit",
        "low_signal_audit",
    ]
    assert {item["repository"] for item in queue[:2]} == {
        "owner/first",
        "owner/second",
    }


def test_triage_keeps_missing_evidence_out_of_new_review_queue() -> None:
    catalog = {
        "repositories": [{"repository": "owner/solver", "scope_status": "task_backed"}]
    }
    row = {
        "repository": "owner/solver",
        "pr_number": 3,
        "title": "Improve presolve",
        "stage": "discovered",
        "selection_decision": "needs_evidence",
        "base_commit": "a" * 40,
        "head_commit": "b" * 40,
    }
    detail = {
        "repository": row["repository"],
        "pr_number": row["pr_number"],
        "base_commit": row["base_commit"],
        "head_commit": row["head_commit"],
        "status": "completed",
        "files": [{"path": "src/presolve.cpp"}],
    }

    result = build_mechanism_review_queue([row], catalog, [detail], 10)

    assert result["queue_size"] == 0
    assert result["pending_evidence_prs"] == 1


def test_mechanism_triage_flags_reverts_and_related_prs() -> None:
    catalog = {
        "repositories": [{"repository": "owner/solver", "scope_status": "task_backed"}]
    }
    rows = [
        {
            "repository": "owner/solver",
            "pr_number": number,
            "title": (
                "Improve branch choice"
                if number == 1
                else "Revert branch change"
                if number == 2
                else "Merge routes in local search"
            ),
            "stage": "discovered",
            "base_commit": "a" * 40,
            "head_commit": "b" * 40,
        }
        for number in (1, 2, 3)
    ]
    details = [
        {
            "repository": row["repository"],
            "pr_number": row["pr_number"],
            "base_commit": row["base_commit"],
            "head_commit": row["head_commit"],
            "status": "completed",
            "body": "Reverts #1" if row["pr_number"] == 2 else "",
            "files": [{"path": "src/search/Branching.cpp"}],
        }
        for row in rows
    ]

    result = build_mechanism_review_queue(rows, catalog, details, 3)
    by_number = {item["pr_number"]: item for item in result["queue"]}

    assert by_number[1]["signal_band"] == 0
    assert by_number[2]["signal_band"] == 2
    assert by_number[2]["caution_flags"] == ["lineage_review"]
    assert by_number[2]["related_pr_hints"] == [1]
    assert by_number[3]["signal_band"] == 0
    assert by_number[3]["caution_flags"] == []
    assert all(row["stage"] == "discovered" for row in rows)


def test_oversized_pr_requires_manual_diff_and_stays_eligible() -> None:
    catalog = {
        "repositories": [
            {"repository": "owner/solver", "scope_status": "approved_source"}
        ]
    }
    row = {
        "repository": "owner/solver",
        "pr_number": 7,
        "title": "Improve cut pool",
        "stage": "discovered",
        "base_commit": "a" * 40,
        "head_commit": "b" * 40,
    }
    detail = {
        "repository": row["repository"],
        "pr_number": row["pr_number"],
        "base_commit": row["base_commit"],
        "head_commit": row["head_commit"],
        "status": "manual_large_pr",
        "changed_file_count": 1001,
        "additions": 4000,
        "deletions": 2000,
        "files": [],
    }

    result = build_mechanism_review_queue([row], catalog, [detail], 10)

    assert result["queue_size"] == 0
    assert result["manual_diff_queue_size"] == 1
    assert result["manual_diff_queue"][0]["review_lane"] == "manual_full_diff"
    assert result["manual_diff_queue"][0]["changed_file_count"] == 1001
    assert row["stage"] == "discovered"


def test_selection_review_hash_detects_later_edits(tmp_path) -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "configs/pr_curation/reviews/highs_pr2861.json"
    )
    review_path = tmp_path / "review.json"
    review_path.write_bytes(source.read_bytes())
    review = json.loads(review_path.read_text())
    row = {
        "repository": review["repository"],
        "pr_number": review["pr_number"],
        "base_commit": review["base_commit"],
        "head_commit": review["head_commit"],
        "merged_at": "2025-01-01T00:00:00Z",
        "stage": "discovered",
        "selection_decision": "advance",
        "selection_review_path": "review.json",
        "selection_review_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
    }
    validate_evidence_stage(row, tmp_path)
    review_path.write_text(review_path.read_text() + "\n")
    with pytest.raises(ValueError, match="selection review hash mismatch"):
        validate_evidence_stage(row, tmp_path)


def test_pinned_build_evidence_rejects_recipe_drift(tmp_path) -> None:
    dockerfile = tmp_path / "Overlay.Dockerfile"
    dockerfile.write_text("FROM pinned-image\n")
    candidate = {
        "repository": "ERGO-Code/HiGHS",
        "pr_number": 2861,
        "base_commit": "a" * 40,
        "head_commit": "b" * 40,
        "source_patch_sha256": "c" * 64,
    }
    evidence = {
        "schema_version": 1,
        **candidate,
        "dockerfile_path": dockerfile.name,
        "dockerfile_sha256": hashlib.sha256(dockerfile.read_bytes()).hexdigest(),
        "parent_image_sha256": "d" * 64,
        "build_network": "none",
        "variants": {
            name: {
                "image_sha256": character * 64,
                "build_exit_code": 0,
                "executable_check_exit_code": 0,
            }
            for name, character in (("base", "e"), ("human_pr", "f"))
        },
    }
    validate_pinned_build_evidence(evidence, candidate, tmp_path)
    dockerfile.write_text("FROM changed-image\n")
    with pytest.raises(ValueError, match="Dockerfile hash"):
        validate_pinned_build_evidence(evidence, candidate, tmp_path)


def test_domain_approval_requires_recorded_evidence(tmp_path) -> None:
    path = tmp_path / "repositories.json"
    catalog = {
        "schema_version": 1,
        "selection_review_schema_version": 2,
        "target_validated_pr_tasks": 1001,
        "repositories": [
            {"repository": "owner/solver", "scope_status": "approved_source"}
        ],
    }
    path.write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="domain review evidence"):
        load_repository_catalog(path)


def test_workload_exposure_requires_frozen_rule_and_active_solver_path(
    tmp_path,
) -> None:
    manifest = tmp_path / "instances.json"
    activation = tmp_path / "activation.json"
    case = {
        "instance_id": "case-1",
        "family": "binary_mip",
        "base_precondition_count": 12,
        "pr_precondition_count": 12,
        "pr_changed_path_count": 2,
    }
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [{"instance_id": "case-1", "family": "binary_mip"}],
            }
        )
    )
    activation.write_text(json.dumps({"schema_version": 1, "cases": [case]}))
    candidate = {
        "repository": "ERGO-Code/HiGHS",
        "pr_number": 2861,
        "base_commit": "a" * 40,
        "head_commit": "b" * 40,
        "source_patch_sha256": "c" * 64,
    }
    review = {"instance_selection_rule": "Freeze binary-row families first."}
    evidence = {
        "schema_version": 1,
        **candidate,
        "instance_selection_rule_sha256": hashlib.sha256(
            review["instance_selection_rule"].encode()
        ).hexdigest(),
        "instance_manifest_path": manifest.name,
        "instance_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "activation_artifact_path": activation.name,
        "activation_artifact_sha256": hashlib.sha256(
            activation.read_bytes()
        ).hexdigest(),
        "cases": [case.copy()],
    }
    validate_workload_exposure_evidence(evidence, candidate, review, tmp_path)
    evidence["cases"][0]["pr_changed_path_count"] = 0
    with pytest.raises(ValueError, match="never exercised"):
        validate_workload_exposure_evidence(evidence, candidate, review, tmp_path)
    evidence["cases"][0]["pr_changed_path_count"] = 2
    trace = tmp_path / "trace.log"
    trace.write_text("changed solver branch reached\n")
    activation.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [case],
                "trace_artifacts": [
                    {
                        "path": trace.name,
                        "sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                    }
                ],
            }
        )
    )
    evidence["activation_artifact_sha256"] = hashlib.sha256(
        activation.read_bytes()
    ).hexdigest()
    validate_workload_exposure_evidence(evidence, candidate, review, tmp_path)
    trace.write_text("trace altered\n")
    with pytest.raises(ValueError, match="trace artifact hash mismatch"):
        validate_workload_exposure_evidence(evidence, candidate, review, tmp_path)


def test_correctness_gate_rejects_invalid_paired_solution(tmp_path) -> None:
    checker = tmp_path / "checker.py"
    instance = tmp_path / "instance.json"
    checker.write_text("# independent checker\n")
    instance.write_text('{"objective": 1}')

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    rows = []
    for variant in ("base", "human_pr"):
        output = tmp_path / f"{variant}.json"
        raw = tmp_path / f"{variant}.raw.json"
        solution = tmp_path / f"{variant}.solution.json"
        output.write_text('{"valid": true, "objective": 1}')
        raw.write_text('{"solver_objective": 1}')
        solution.write_text('{"solution": 1}')
        rows.append(
            {
                "instance_id": "case-1",
                "instance_path": instance.name,
                "instance_sha256": digest(instance),
                "variant": variant,
                "solver_seed": 0,
                "budget_sec": 1.0,
                "output_path": output.name,
                "output_sha256": digest(output),
                "raw_output_path": raw.name,
                "raw_output_sha256": digest(raw),
                "solution_path": solution.name,
                "solution_sha256": digest(solution),
                "driver_valid": True,
                "driver_objective": 1,
                "independent_feasible": True,
                "independent_objective": 1,
                "objective_matches": True,
            }
        )
    ledger = tmp_path / "ledger.jsonl"
    candidate = {
        "repository": "owner/solver",
        "pr_number": 1,
        "base_commit": "a" * 40,
        "head_commit": "b" * 40,
        "source_patch_sha256": "c" * 64,
    }
    evidence = {
        "schema_version": 1,
        **candidate,
        "checker_artifact_path": checker.name,
        "checker_module_sha256": digest(checker),
        "run_ledger_path": ledger.name,
        "expected_runs": 2,
        "retained_runs": 2,
        "verified_solution_count": 2,
        "invalid_solution_count": 0,
        "upstream_targeted_tests": {
            variant: {"passed": 1, "failed": 0} for variant in ("base", "human_pr")
        },
    }

    def save_ledger():
        ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
        evidence["run_ledger_sha256"] = digest(ledger)

    save_ledger()
    validate_independent_correctness_evidence(evidence, candidate, tmp_path)
    rows[0]["independent_objective"] = 1.0 + 1e-12
    save_ledger()
    validate_independent_correctness_evidence(evidence, candidate, tmp_path)
    rows[0]["independent_objective"] = 1.0 + 1e-3
    save_ledger()
    with pytest.raises(ValueError, match="independently invalid"):
        validate_independent_correctness_evidence(evidence, candidate, tmp_path)
    rows[0]["independent_objective"] = 1
    save_ledger()
    raw.write_text('{"solver_objective": 2}')
    with pytest.raises(ValueError, match="correctness file hash mismatch"):
        validate_independent_correctness_evidence(evidence, candidate, tmp_path)
    raw.write_text('{"solver_objective": 1}')
    for row in rows:
        row["round"] = 0
    rows.extend({**row, "round": 1} for row in rows[:])
    evidence.update(expected_runs=4, retained_runs=4, verified_solution_count=4)
    save_ledger()
    validate_independent_correctness_evidence(evidence, candidate, tmp_path)
    del rows[2:]
    evidence.update(expected_runs=2, retained_runs=2, verified_solution_count=2)
    save_ledger()
    with pytest.raises(ValueError, match="lacks mathematical semantic checks"):
        validate_independent_correctness_evidence(
            evidence,
            candidate,
            tmp_path,
            required_semantic_obligations={"bound_validity"},
        )
    bound_record = tmp_path / "bounds.json"
    bound_record.write_text('{"case_count": 1, "invalid_count": 0}')
    evidence["semantic_obligations"] = {
        "bound_validity": {
            "independent_checker": "small-instance enumeration",
            "case_count": 1,
            "invalid_count": 0,
            "artifact_path": bound_record.name,
            "artifact_sha256": digest(bound_record),
        }
    }
    validate_independent_correctness_evidence(
        evidence,
        candidate,
        tmp_path,
        required_semantic_obligations={"bound_validity"},
    )
    rows[1]["independent_feasible"] = False
    save_ledger()
    with pytest.raises(ValueError, match="independently invalid"):
        validate_independent_correctness_evidence(evidence, candidate, tmp_path)


def test_public_builder_rechecks_original_cvrp_solutions(tmp_path) -> None:
    candidate = {
        "repository": "owner/solver",
        "pr_number": 1,
        "base_commit": "a" * 40,
        "head_commit": "b" * 40,
        "source_patch_sha256": "c" * 64,
        "task_config_path": "task.yaml",
        "build_evidence_path": "build.json",
    }
    (tmp_path / "task.yaml").write_text(
        yaml.safe_dump(
            {
                "problem_family": "cvrp",
                "evaluation": {"performance_protocol": "heuristic_fixed_budget"},
                "repository": {"judge_image": "sha256:" + "d" * 64},
            }
        )
    )
    (tmp_path / "build.json").write_text(
        json.dumps(
            {
                "variants": {
                    "base": {"image_sha256": "d" * 64},
                    "human_pr": {"image_sha256": "e" * 64},
                }
            }
        )
    )
    checker = tmp_path / "checker.py"
    checker.write_text("# evaluator-owned checker\n")
    instance = tmp_path / "instance.json"
    instance.write_text(
        json.dumps(
            {
                "coordinates": [[0, 0], [1, 0]],
                "demands": [0, 1],
                "capacity": 2,
                "depot": 0,
                "distance_metric": "EUC_2D",
            }
        )
    )
    pair = {
        "round": 0,
        "instance_id": "case-1",
        "solver_seed": 0,
        "budget_sec": 1.0,
        "instance_path": str(instance),
    }
    for label in ("old", "new"):
        output = tmp_path / f"{label}.json"
        solution = tmp_path / f"{label}.solution.json"
        output.write_text(json.dumps({"valid": True, "objective": 2.0}))
        solution.write_text(json.dumps({"routes": [[1]]}))
        pair[label] = {
            "status": "completed",
            "commit": candidate["base_commit"]
            if label == "old"
            else candidate["head_commit"],
            "image": "sha256:" + ("d" if label == "old" else "e") * 64,
            "container_exit_code": 0,
            "output_path": str(output),
            "solution_path": str(solution),
            "reported_valid": True,
            "reported_objective": 2.0,
            "verified_objective": 2.0,
        }
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(json.dumps(pair) + "\n")
    tests = {variant: {"passed": 1, "failed": 0} for variant in ("base", "human_pr")}

    evidence = build_paired_correctness_evidence(
        candidate, tmp_path, pairs, tmp_path / "ledger.jsonl", checker, tests, {}
    )
    validate_independent_correctness_evidence(evidence, candidate, tmp_path)
    assert evidence["verified_solution_count"] == 2
    assert evidence["normal_timeout_count"] == 0

    pair["new"]["image"] = pair["old"]["image"]
    pairs.write_text(json.dumps(pair) + "\n")
    with pytest.raises(ValueError, match="images are identical"):
        build_paired_correctness_evidence(
            candidate,
            tmp_path,
            pairs,
            tmp_path / "same-image-ledger.jsonl",
            checker,
            tests,
            {},
        )
    pair["new"]["image"] = "sha256:" + "f" * 64
    pairs.write_text(json.dumps(pair) + "\n")
    with pytest.raises(ValueError, match="image differs from paired build"):
        build_paired_correctness_evidence(
            candidate,
            tmp_path,
            pairs,
            tmp_path / "wrong-image-ledger.jsonl",
            checker,
            tests,
            {},
        )
    pair["new"]["image"] = "sha256:" + "e" * 64
    pairs.write_text(json.dumps(pair) + "\n")
    (tmp_path / "new.solution.json").write_text(json.dumps({"routes": []}))
    with pytest.raises(ValueError, match="independent original-model check"):
        build_paired_correctness_evidence(
            candidate,
            tmp_path,
            pairs,
            tmp_path / "invalid-ledger.jsonl",
            checker,
            tests,
            {},
        )


def test_public_builder_retains_only_normal_exact_timeouts(tmp_path) -> None:
    candidate = {
        "repository": "owner/solver",
        "pr_number": 2,
        "base_commit": "a" * 40,
        "head_commit": "b" * 40,
        "source_patch_sha256": "c" * 64,
        "task_config_path": "task.yaml",
    }
    (tmp_path / "task.yaml").write_text(
        yaml.safe_dump(
            {
                "problem_family": "cvrp",
                "evaluation": {"performance_protocol": "exact_verified_solve"},
            }
        )
    )
    checker = tmp_path / "checker.py"
    checker.write_text("# independent checker\n")
    instance = tmp_path / "instance.json"
    instance.write_text(
        json.dumps(
            {
                "coordinates": [[0, 0], [1, 0]],
                "demands": [0, 1],
                "capacity": 2,
            }
        )
    )
    solved = tmp_path / "solved.json"
    timeout = tmp_path / "timeout.json"
    solution = tmp_path / "solved.solution.json"
    solved.write_text(
        json.dumps({"valid": True, "objective": 2.0, "solver_termination": "optimal"})
    )
    timeout.write_text(
        json.dumps(
            {
                "valid": False,
                "has_solution": False,
                "objective": None,
                "solver_termination": "time_limit",
            }
        )
    )
    solution.write_text(json.dumps({"routes": [[1]]}))
    pair = {
        "round": 0,
        "instance_id": "case-1",
        "solver_seed": 0,
        "budget_sec": 1.0,
        "instance_path": str(instance),
        "old": {
            "status": "completed",
            "commit": candidate["base_commit"],
            "image": "sha256:" + "d" * 64,
            "container_exit_code": 0,
            "output_path": str(solved),
            "solution_path": str(solution),
            "reported_valid": True,
            "reported_objective": 2.0,
            "verified_objective": 2.0,
        },
        "new": {
            "status": "no_solution",
            "commit": candidate["head_commit"],
            "image": "sha256:" + "e" * 64,
            "container_exit_code": 0,
            "output_path": str(timeout),
            "reported": json.loads(timeout.read_text()),
        },
    }
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(json.dumps(pair) + "\n")
    tests = {variant: {"passed": 0, "failed": 0} for variant in ("base", "human_pr")}

    evidence = build_paired_correctness_evidence(
        candidate, tmp_path, pairs, tmp_path / "ledger.jsonl", checker, tests, {}
    )
    validate_independent_correctness_evidence(evidence, candidate, tmp_path)
    assert evidence["verified_solution_count"] == 1
    assert evidence["normal_timeout_count"] == 1

    pair["new"]["reported"]["solver_termination"] = "error"
    pairs.write_text(json.dumps(pair) + "\n")
    with pytest.raises(ValueError, match="unqualified timeout"):
        build_paired_correctness_evidence(
            candidate,
            tmp_path,
            pairs,
            tmp_path / "bad-ledger.jsonl",
            checker,
            tests,
            {},
        )

    timeout.write_text(
        json.dumps({"valid": True, "objective": 2.0, "solver_termination": "time_limit"})
    )
    pair["new"].update(
        status="completed",
        solution_path=str(solution),
        reported_valid=True,
        reported_objective=2.0,
        verified_objective=2.0,
    )
    pair["new"].pop("reported")
    pairs.write_text(json.dumps(pair) + "\n")
    incumbent = build_paired_correctness_evidence(
        candidate,
        tmp_path,
        pairs,
        tmp_path / "incumbent-ledger.jsonl",
        checker,
        tests,
        {},
    )
    validate_independent_correctness_evidence(incumbent, candidate, tmp_path)
    assert incumbent["verified_solution_count"] == 2
    assert incumbent["normal_timeout_count"] == 1


def test_open_opportunity_keeps_human_reference_status_separate(tmp_path) -> None:
    candidate = {
        "repository": "PyVRP/PyVRP",
        "pr_number": 966,
        "base_commit": "a" * 40,
        "head_commit": "b" * 40,
        "source_patch_sha256": "c" * 64,
    }
    manifest = tmp_path / "manifest.json"
    analysis = tmp_path / "analysis.json"
    pairs = tmp_path / "pairs.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "old": {"commit": candidate["base_commit"]},
                "new": {"commit": candidate["head_commit"]},
            }
        )
    )

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    report = {
        "manifest_sha256": digest(manifest),
        "expected_pairs": 1,
        "observed_unique_pairs": 1,
        "retained_pair_attempts": 1,
        "missing_pairs": [],
        "failed_pairs": [],
        "pitbench_performance": {"confirmed_classification": "inconclusive"},
    }
    analysis.write_text(json.dumps(report))
    pairs.write_text(
        json.dumps(
            {
                "round": 0,
                "instance_id": "case-1",
                "solver_seed": 0,
                "old": {
                    "status": "completed",
                    "reported_valid": True,
                    "reported_objective": 10,
                    "verified_objective": 10,
                    "normalized_gap": 0.1,
                },
                "new": {
                    "status": "completed",
                    "reported_valid": True,
                    "reported_objective": 10,
                    "verified_objective": 10,
                },
            }
        )
        + "\n"
    )
    evidence = {
        "schema_version": 1,
        **candidate,
        "opportunity_kind": "remaining_verified_gap",
        "manifest_path": manifest.name,
        "manifest_sha256": digest(manifest),
        "analysis_path": analysis.name,
        "analysis_sha256": digest(analysis),
        "pairs_path": pairs.name,
        "pairs_sha256": digest(pairs),
        "base_mean_normalized_gap": 0.1,
        "human_reference_classification": "inconclusive",
    }
    validate_open_opportunity_evidence(evidence, candidate, tmp_path)
    evidence["base_mean_normalized_gap"] = 0
    with pytest.raises(ValueError, match="no independently measured improvement room"):
        validate_open_opportunity_evidence(evidence, candidate, tmp_path)
    evidence["base_mean_normalized_gap"] = 0.1
    report["pitbench_performance"]["confirmed_classification"] = "improved"
    analysis.write_text(json.dumps(report))
    evidence["analysis_sha256"] = digest(analysis)
    evidence["human_reference_classification"] = "improved"
    with pytest.raises(ValueError, match="positive human PR belongs"):
        validate_open_opportunity_evidence(evidence, candidate, tmp_path)


def test_open_opportunity_accepts_complete_baseline_only_qualification(
    tmp_path,
) -> None:
    candidate = {
        "repository": "PyVRP/PyVRP",
        "pr_number": 967,
        "base_commit": "a" * 40,
        "head_commit": "b" * 40,
        "source_patch_sha256": "c" * 64,
    }
    manifest = tmp_path / "manifest.json"
    analysis = tmp_path / "analysis.json"
    runs = tmp_path / "baseline_runs.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "run_mode": "baseline_qualification",
                "old": {"commit": candidate["base_commit"]},
            }
        )
    )

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    analysis.write_text(
        json.dumps(
            {
                "manifest_sha256": digest(manifest),
                "expected_runs": 1,
                "observed_unique_runs": 1,
                "retained_run_attempts": 1,
                "missing_runs": [],
                "failed_runs": [],
                "invalid_runs": [],
                "conclusion": "verified_improvement_room",
                "base_mean_normalized_gap": 0.1,
            }
        )
    )
    runs.write_text(
        json.dumps(
            {
                "instance_id": "case-1",
                "solver_seed": 0,
                "base": {
                    "status": "completed",
                    "reported_valid": True,
                    "reported_objective": 10,
                    "verified_objective": 10,
                    "normalized_gap": 0.1,
                },
            }
        )
        + "\n"
    )
    evidence = {
        "schema_version": 1,
        **candidate,
        "opportunity_kind": "remaining_verified_gap",
        "qualification_mode": "baseline_qualification",
        "manifest_path": manifest.name,
        "manifest_sha256": digest(manifest),
        "analysis_path": analysis.name,
        "analysis_sha256": digest(analysis),
        "runs_path": runs.name,
        "runs_sha256": digest(runs),
        "base_mean_normalized_gap": 0.1,
        "human_reference_classification": "not_evaluated",
    }
    validate_open_opportunity_evidence(evidence, candidate, tmp_path)
    built = build_baseline_opportunity_evidence(
        candidate, tmp_path, manifest, analysis, runs
    )
    assert built["base_mean_normalized_gap"] == 0.1
    assert built["runs_sha256"] == digest(runs)
    evidence["base_mean_normalized_gap"] = 0
    with pytest.raises(ValueError, match="no verified improvement room"):
        validate_open_opportunity_evidence(evidence, candidate, tmp_path)


def test_exact_open_opportunity_requires_verified_solves_and_time_limits(
    tmp_path,
) -> None:
    candidate = {
        "repository": "ERGO-Code/HiGHS",
        "pr_number": 2861,
        "base_commit": "a" * 40,
        "head_commit": "b" * 40,
        "source_patch_sha256": "c" * 64,
    }
    manifest = tmp_path / "manifest.json"
    analysis = tmp_path / "analysis.json"
    runs = tmp_path / "baseline_runs.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "run_mode": "baseline_qualification",
                "old": {"commit": candidate["base_commit"]},
            }
        )
    )

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    analysis.write_text(
        json.dumps(
            {
                "manifest_sha256": digest(manifest),
                "expected_runs": 3,
                "observed_unique_runs": 3,
                "retained_run_attempts": 3,
                "missing_runs": [],
                "failed_runs": [],
                "invalid_runs": [],
                "conclusion": "verified_unsolved_room",
                "base_verified_solved_runs": 1,
                "base_time_limit_runs": 2,
                "base_time_limit_with_incumbent_runs": 1,
            }
        )
    )
    rows = [
        {
            "instance_id": "solved",
            "solver_seed": 0,
            "target": -5,
            "base": {
                "status": "completed",
                "solver_termination": "optimal",
                "reported_valid": True,
                "reported_objective": -5,
                "verified_objective": -5,
            },
        },
        {
            "instance_id": "timeout",
            "solver_seed": 0,
            "base": {
                "status": "no_solution",
                "reported": {"solver_termination": "time_limit"},
            },
        },
        {
            "instance_id": "timeout_with_incumbent",
            "solver_seed": 0,
            "target": -7,
            "base": {
                "status": "completed",
                "solver_termination": "time_limit",
                "reported_valid": True,
                "reported_objective": -6,
                "verified_objective": -6,
            },
        },
    ]
    runs.write_text("".join(json.dumps(row) + "\n" for row in rows))
    evidence = {
        "schema_version": 1,
        **candidate,
        "opportunity_kind": "unsolved_exact",
        "qualification_mode": "baseline_qualification",
        "manifest_path": manifest.name,
        "manifest_sha256": digest(manifest),
        "analysis_path": analysis.name,
        "analysis_sha256": digest(analysis),
        "runs_path": runs.name,
        "runs_sha256": digest(runs),
        "base_verified_solved_runs": 1,
        "base_time_limit_runs": 2,
        "human_reference_classification": "not_evaluated",
    }
    validate_open_opportunity_evidence(evidence, candidate, tmp_path)
    built = build_baseline_opportunity_evidence(
        candidate, tmp_path, manifest, analysis, runs
    )
    assert built["opportunity_kind"] == "unsolved_exact"
    assert built["base_verified_solved_runs"] == 1
    (tmp_path / "task.yaml").write_text(
        yaml.safe_dump(
            {"evaluation": {"performance_protocol": "heuristic_fixed_budget"}}
        )
    )
    candidate["task_config_path"] = "task.yaml"
    with pytest.raises(ValueError, match="task metric protocol"):
        build_baseline_opportunity_evidence(
            candidate, tmp_path, manifest, analysis, runs
        )
    candidate.pop("task_config_path")
    evidence["base_time_limit_runs"] = 0
    with pytest.raises(ValueError, match="no verified solve room"):
        validate_open_opportunity_evidence(evidence, candidate, tmp_path)


def test_integration_builder_requires_complete_real_judge_grid(tmp_path) -> None:
    candidate = {
        "repository": "owner/solver",
        "pr_number": 1,
        "base_commit": "a" * 40,
        "task_id": "solver_task",
        "task_track": "open_opportunity",
        "task_config_path": "task.yaml",
    }
    task = {
        "task_id": candidate["task_id"],
        "release": {"base_commit": candidate["base_commit"]},
        "repository": {
            "agent_image": "agent:local",
            "judge_image": "sha256:" + "c" * 64,
        },
        "evaluation": {"performance_protocol": "heuristic_fixed_budget"},
    }
    (tmp_path / "task.yaml").write_text(yaml.safe_dump(task))
    agent = tmp_path / "agent.json"
    checkout = tmp_path / "checkout.json"
    agent.write_text(
        json.dumps(
            {
                "image_reference": "agent:local",
                "image_id": "sha256:" + "d" * 64,
                "git_head": candidate["base_commit"],
                "submodule_status": [],
            }
        )
    )
    checkout.write_text(
        json.dumps({"base_commit": candidate["base_commit"], "submodule_status": []})
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    observation = {
        "task_id": candidate["task_id"],
        "code_state": "base",
        "instance_set": "judge_id",
        "instance_id": "case-1",
        "solver_seed": 0,
        "budget_sec": 1.0,
    }
    (run_dir / "expected-run-grid.json").write_text(
        json.dumps({"task_id": candidate["task_id"], "runs": [observation]})
    )
    (run_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "completed": True,
                "payload": {
                    "task_id": candidate["task_id"],
                    "validity": {"accepted": True},
                    "summary": {"observation_count": 1, "valid_observation_count": 1},
                },
            }
        )
    )
    observations = run_dir / "judge-observations.jsonl"
    observations.write_text(
        json.dumps({**observation, "status": "completed", "valid": True}) + "\n"
    )

    evidence = build_task_integration_evidence(
        candidate,
        tmp_path,
        agent,
        checkout,
        run_dir,
        dry_run_exit_code=1,
        dry_run_exit_note="Unrelated auxiliary check failed after the task completed.",
    )
    validate_task_integration_evidence(evidence, candidate, tmp_path)
    assert evidence["observations_path"] == "run/judge-observations.jsonl"
    assert evidence["dry_run_exit_code"] == 1
    incomplete_exit_record = {**evidence, "dry_run_exit_note": ""}
    with pytest.raises(ValueError, match="limited Judge run is incomplete"):
        validate_task_integration_evidence(
            incomplete_exit_record, candidate, tmp_path
        )

    timeout_observation = {
        **observation,
        "instance_id": "case-timeout",
        "status": "no_solution",
        "valid": False,
        "solver_termination": "time_limit",
    }
    (run_dir / "expected-run-grid.json").write_text(
        json.dumps(
            {
                "task_id": candidate["task_id"],
                "runs": [observation, {**observation, "instance_id": "case-timeout"}],
            }
        )
    )
    (run_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "completed": True,
                "payload": {
                    "task_id": candidate["task_id"],
                    "validity": {"accepted": True},
                    "summary": {"observation_count": 2, "valid_observation_count": 1},
                },
            }
        )
    )
    observations.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {**observation, "status": "completed", "valid": True},
                timeout_observation,
            )
        )
        + "\n"
    )
    build_task_integration_evidence(
        candidate, tmp_path, agent, checkout, run_dir, dry_run_exit_code=0
    )

    observations.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {**observation, "status": "completed", "valid": False},
                timeout_observation,
            )
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="invalid or infrastructure observations"):
        build_task_integration_evidence(
            candidate,
            tmp_path,
            agent,
            checkout,
            run_dir,
            dry_run_exit_code=1,
            dry_run_exit_note="Unrelated auxiliary check failed after the task completed.",
        )
