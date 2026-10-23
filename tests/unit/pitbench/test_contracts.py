from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from pitbench.contracts import (
    compatibility_matrix,
    export_contract,
    import_measurement_bundle,
)
from pitbench.repositories.classification import metadata_selection_review
from pitbench.repositories.pr_curation import validate_selection_review
from pitbench.schema.objectives import (
    objective_catalog,
    objective_catalog_sha256,
    qualification_protocols,
    qualification_protocols_sha256,
)
from pitbench.tasks import TaskCatalog

ROOT = Path(__file__).resolve().parents[3]


def _write(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(root: Path, *, count: int = 2, research: bool = False) -> dict:
    root.mkdir()
    entries, candidates = [], []
    metric = "fixed_budget_normalized_gap"
    for number in range(1, count + 1):
        candidate = {
            "repository": "owner/solver",
            "pr_number": number,
            "base_commit": "a" * 40,
            "head_commit": "b" * 40,
            "stage": "discovered",
        }
        candidates.append(candidate)
        card = {
            **candidate,
            "title": "Reuse search data",
            "body": "Avoid repeated computation.",
            "changed_files": [{"path": "src/search.cpp"}],
            "changed_source_paths": ["src/search.cpp"],
        }
        patch_path = f"patches/pr-{number}.patch"
        patch = root / patch_path
        patch.parent.mkdir(exist_ok=True)
        patch.write_text(
            f"diff --git a/src/search.cpp b/src/search.cpp\n+reuse{number}\n"
        )
        patch_sha = hashlib.sha256(patch.read_bytes()).hexdigest()
        review = metadata_selection_review(candidate, card, patch.read_bytes())
        packet = {
            "protocol_version": 1,
            "kind": "research_case_request" if research else "measurement_request",
            "candidate": candidate,
            "public_pr": card,
            "git_evidence": {
                "source_status": "cached",
                "artifact_path": patch_path,
                "artifact_sha256": patch_sha,
            },
            "evaluation_reference_commit": candidate["head_commit"],
            "optimization_profile_hint": {
                "objectives": ["solution_quality"],
                "taxonomy_sha256": objective_catalog_sha256(),
            },
            "requested_metrics": {metric: objective_catalog()["metrics"][metric]},
            "qualification_requirements": {
                "protocol_version": "1.0",
                "protocols_sha256": qualification_protocols_sha256(),
                "decision_rules": qualification_protocols()["decision_rules"],
                "metric_protocols": {
                    metric: qualification_protocols()["metrics"][metric]
                },
            },
            "pr_metadata": review["metadata"],
            "metadata_selection_review": review,
            "selection_decision": None,
            "task_ready": False,
        }
        request_path = f"requests/pr-{number}.json"
        entries.append(
            {
                "id": f"owner/solver#{number}",
                "request_path": request_path,
                "request_sha256": _write(root / request_path, packet),
                "source_patch_path": patch_path,
                "source_patch_sha256": patch_sha,
            }
        )
    (root / "candidates.jsonl").write_text(
        "".join(json.dumps(c) + "\n" for c in candidates)
    )
    (root / "results.csv").write_text(
        "id,preparation_status\n" + "".join(f"{e['id']},prepared\n" for e in entries)
    )
    manifest = {
        "schema_version": 1,
        "kind": "research_case_requests" if research else "measurement_requests",
        "candidates": count,
        "prepared": count,
        "requests": entries,
        **{
            f"{name}_sha256": hashlib.sha256((root / path).read_bytes()).hexdigest()
            for name, path in (
                ("candidates", "candidates.jsonl"),
                ("results", "results.csv"),
            )
        },
    }
    _write(root / "manifest.json", manifest)
    return manifest


def _replace_request(root: Path, manifest: dict, **changes: object) -> None:
    entry = manifest["requests"][0]
    path = root / entry["request_path"]
    packet = json.loads(path.read_text())
    packet.update(changes)
    entry["request_sha256"] = _write(path, packet)
    _write(root / "manifest.json", manifest)


@pytest.mark.parametrize("research", [False, True])
def test_intake_replay_preserves_requests_without_admitting_tasks(tmp_path, research):
    bundle, store = tmp_path / "bundle", tmp_path / "intake"
    manifest = _bundle(bundle, count=10, research=research)
    before = (bundle / "candidates.jsonl").read_bytes()
    assert import_measurement_bundle(bundle, store) == {
        "imported": 10,
        "reused": 0,
        "tasks_created": 0,
    }
    assert import_measurement_bundle(bundle, store) == {
        "imported": 0,
        "reused": 10,
        "tasks_created": 0,
    }
    assert len(list((store / "requests").iterdir())) == 10
    for entry in manifest["requests"]:
        target = store / "requests" / entry["request_sha256"]
        assert (target / entry["request_path"]).read_bytes() == (
            bundle / entry["request_path"]
        ).read_bytes()
        receipt = json.loads((target / "intake.json").read_text())
        assert receipt["status"] == "pending_measurement"
    assert (bundle / "candidates.jsonl").read_bytes() == before


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("protocol_version", 2, "protocol_version"),
        ("protocol_version", True, "unsupported measurement request protocol"),
        ("task_ready", True, "task_ready"),
        ("selection_decision", "advance", "cannot admit"),
        ("evaluation_reference_commit", "c" * 40, "evaluation reference"),
    ],
)
def test_intake_rejects_unsupported_or_promoted_requests(tmp_path, field, value, error):
    bundle, store = tmp_path / "bundle", tmp_path / "intake"
    manifest = _bundle(bundle)
    _replace_request(bundle, manifest, **{field: value})
    with pytest.raises(ValueError, match=error):
        import_measurement_bundle(bundle, store)
    assert not store.exists()


@pytest.mark.parametrize(
    "tamper",
    ["patch", "protocol", "rules", "public_pr", "source_path", "bundle_version"],
)
def test_intake_rejects_tampering_before_any_import(tmp_path, tamper):
    bundle, store = tmp_path / "bundle", tmp_path / "intake"
    manifest = _bundle(bundle)
    entry = manifest["requests"][0]
    packet = json.loads((bundle / entry["request_path"]).read_text())
    if tamper == "patch":
        (bundle / manifest["requests"][1]["source_patch_path"]).write_text("changed")
    elif tamper == "bundle_version":
        manifest["schema_version"] = 2
        _write(bundle / "manifest.json", manifest)
    else:
        if tamper == "protocol":
            packet["qualification_requirements"]["protocols_sha256"] = "0" * 64
        elif tamper == "rules":
            packet["qualification_requirements"]["decision_rules"] = {}
        elif tamper == "public_pr":
            packet["public_pr"]["body"] = "Unbound replacement context"
        elif tamper == "source_path":
            packet["git_evidence"]["artifact_path"] = "../outside.patch"
        _replace_request(bundle, manifest, **packet)
    with pytest.raises(ValueError):
        import_measurement_bundle(bundle, store)
    assert not store.exists()


def test_intake_revalidates_artifacts_on_replay(tmp_path):
    bundle, store = tmp_path / "bundle", tmp_path / "intake"
    manifest = _bundle(bundle)
    import_measurement_bundle(bundle, store)
    entry = manifest["requests"][0]
    (
        store / "requests" / entry["request_sha256"] / entry["source_patch_path"]
    ).write_text("corrupted cache")
    with pytest.raises(ValueError, match="hash mismatch"):
        import_measurement_bundle(bundle, store)


def test_intake_keeps_unprepared_candidates_and_failure_ledger(tmp_path):
    bundle, store = tmp_path / "bundle", tmp_path / "intake"
    manifest = _bundle(bundle)
    with (bundle / "candidates.jsonl").open("a") as handle:
        handle.write(
            json.dumps(
                {"repository": "owner/solver", "pr_number": 99, "stage": "discovered"}
            )
            + "\n"
        )
    with (bundle / "results.csv").open("a") as handle:
        handle.write("owner/solver#99,source_blocked\n")
    manifest["candidates"] = 3
    for name, path in (("candidates", "candidates.jsonl"), ("results", "results.csv")):
        manifest[f"{name}_sha256"] = hashlib.sha256(
            (bundle / path).read_bytes()
        ).hexdigest()
    _write(bundle / "manifest.json", manifest)
    assert import_measurement_bundle(bundle, store)["imported"] == 2
    ledger = next((store / "bundles").iterdir())
    for name in ("manifest.json", "candidates.jsonl", "results.csv"):
        assert (ledger / name).read_bytes() == (bundle / name).read_bytes()


def test_contract_export_uses_original_catalog_bytes_and_actual_models(tmp_path):
    output = tmp_path / "contract"
    manifest = export_contract(output, repository_root=ROOT)
    for path, digest in manifest["files"].items():
        assert hashlib.sha256((output / path).read_bytes()).hexdigest() == digest
    for name in ("objective_catalog.json", "qualification_protocols.json"):
        assert (output / name).read_bytes() == (
            ROOT / "pitbench/schema" / name
        ).read_bytes()
    task_schema = json.loads((output / "task.schema.json").read_text())
    assert task_schema["properties"]["schema_version"]["const"] == "3.0"
    assert compatibility_matrix()["selection_review"]["accepted_versions"] == [2, 3]
    with pytest.raises(FileExistsError):
        export_contract(output, repository_root=ROOT)


@pytest.mark.parametrize(
    "name",
    [
        "highs_pr577",
        "highs_pr618",
        "highs_pr651",
        "highs_pr685",
        "highs_pr690",
        "highs_pr693",
        "highs_pr1992",
        "highs_highs_pr2027",
        "highs_pr2522",
        "highs_pr2861",
    ],
)
def test_existing_schema_two_reviews_are_accepted_without_migration(name):
    review = json.loads(
        (ROOT / "configs/pr_curation/reviews" / f"{name}.json").read_text()
    )
    original = copy.deepcopy(review)
    assert validate_selection_review(review, review) == review["decision"]
    assert review == original
    review["schema_version"] = 4
    with pytest.raises(ValueError, match="unsupported selection review schema"):
        validate_selection_review(review, review)


def test_task_inventory_preserves_missing_and_invalid_evidence(tmp_path):
    catalog = TaskCatalog(ROOT)
    records = catalog.records()
    task = records[0].task
    registry = tmp_path / "registry.jsonl"
    registry.write_text(
        json.dumps(
            {
                "task_id": task.task_id,
                "repository": task.release.repository,
                "pr_number": 1,
                "base_commit": "0" * 40,
                "stage": "task_ready",
            }
        )
        + "\n"
    )
    before = {r.task.task_id: r.task_config_sha256 for r in records}
    inventory = catalog.inventory(registry_path=registry, baseline_commit="6fba9a99")
    assert inventory["removed_task_ids"] == []
    assert all(item["changed_fields"] == [] for item in inventory["tasks"])
    invalid = inventory["tasks"][0]["evidence"][0]
    assert invalid["recorded_stage"] == "task_ready"
    assert invalid["recheck"] == "failed"
    assert "source differs" in invalid["error"]
    assert inventory["tasks"][1]["evidence_status"] == "unknown"
    assert before == {r.task.task_id: r.task_config_sha256 for r in catalog.records()}
