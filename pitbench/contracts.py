"""Versioned contracts and evaluator-owned intake of curation exports.

Intake preserves source requests; it neither executes them nor admits tasks.
Consumers can use the exported JSON schemas/catalogs without installing the
Agent harness or duplicating the qualification statistics.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from pitbench.repositories.classification import (
    extract_pr_metadata,
    validate_metadata_selection_review,
)
from pitbench.repositories.pr_curation import (
    candidate_key,
    read_jsonl,
)
from pitbench.schema.objectives import (
    OptimizationProfile,
    objective_catalog,
    objective_catalog_sha256,
    qualification_protocols,
    qualification_protocols_sha256,
)
from pitbench.schema.observation import ExpectedRunGrid, RunObservation
from pitbench.schema.research import ResearchCasePlan
from pitbench.schema.task import PitBenchTask

CONTRACT_VERSION = "1.0"


class MeasurementRequest(BaseModel):
    """The existing curation wire format; optional provenance is retained."""

    model_config = ConfigDict(extra="allow")

    protocol_version: Literal[1]
    kind: Literal["measurement_request", "research_case_request"]
    candidate: dict[str, Any]
    public_pr: dict[str, Any]
    git_evidence: dict[str, Any]
    evaluation_reference_commit: str
    optimization_profile_hint: OptimizationProfile
    qualification_requirements: dict[str, Any]
    metadata_selection_review: dict[str, Any]
    pr_metadata: dict[str, Any]
    task_ready: Literal[False]


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compatibility_matrix() -> dict[str, Any]:
    return {
        "selection_review": {
            "accepted_versions": [2, 3],
            "migration": "none; preserve the original review and evidence stage",
            "validator": "pitbench.repositories.pr_curation:validate_selection_review",
        },
        "task": {"accepted_versions": ["3.0"]},
        "measurement_request": {"accepted_versions": [1]},
        "research_case_request": {"accepted_versions": [1]},
        "request_bundle": {"accepted_versions": [1]},
        "unknown_versions": "reject",
        "qualification_protocols_sha256": qualification_protocols_sha256(),
        "objective_catalog_sha256": objective_catalog_sha256(),
    }


def export_contract(destination: Path, *, repository_root: Path) -> dict[str, Any]:
    """Export schemas from their actual models and byte-identical catalogs."""
    payloads = {
        f"{name}.schema.json": _json_bytes(model.model_json_schema())
        for name, model in {
            "measurement_request": MeasurementRequest,
            "task": PitBenchTask,
            "run_observation": RunObservation,
            "expected_run_grid": ExpectedRunGrid,
            "research_case_plan": ResearchCasePlan,
        }.items()
    }
    for name in ("objective_catalog.json", "qualification_protocols.json"):
        payloads[name] = files("pitbench.schema").joinpath(name).read_bytes()
    payloads["compatibility.json"] = _json_bytes(compatibility_matrix())
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repository_root, text=True
        ).strip()
    )
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "kind": "pitbench_contract",
        "producer_commit": commit,
        "producer_dirty": dirty,
        "files": {name: _sha(raw) for name, raw in sorted(payloads.items())},
        "hash_fields": {
            "request_sha256": "exact request JSON bytes",
            "source_patch_sha256": "exact source-only patch bytes",
            "full_integrated_patch_sha256": "exact complete integrated patch bytes",
            "task_config_sha256": "exact task YAML bytes",
            "protocols_sha256": "exact qualification_protocols.json bytes",
            "instance_set_config_sha256": "exact instance-set config bytes",
            "base_commit": "upstream Git SHA-1; not a SHA-256 artifact digest",
        },
        "qualification": "Schemas validate structure; task readiness requires evaluator execution evidence.",
    }
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        staging = Path(temporary) / "contract"
        staging.mkdir()
        for name, raw in payloads.items():
            (staging / name).write_bytes(raw)
        (staging / "manifest.json").write_bytes(_json_bytes(manifest))
        staging.rename(destination)
    return manifest


def _artifact(root: Path, path: str, digest: str) -> tuple[Path, bytes]:
    relative = PurePosixPath(path)
    if not path or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe contract artifact path: {path!r}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"contract artifact escapes bundle: {path}")
    raw = resolved.read_bytes()
    if _sha(raw) != digest:
        raise ValueError(f"contract artifact hash mismatch: {path}")
    return resolved, raw


def _entry_artifacts(entry: dict[str, Any]) -> list[tuple[str, str]]:
    artifacts = []
    for name, value in entry.items():
        if name.endswith("_path"):
            artifacts.append((value, entry[name.removesuffix("_path") + "_sha256"]))
    for log in entry.get("model_logs", []):
        artifacts.append((log["path"], log["sha256"]))
    return artifacts


def validate_measurement_bundle(root: Path) -> list[dict[str, Any]]:
    """Verify the complete export before intake, using existing review rules."""
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
    ):
        raise ValueError("unsupported curation bundle schema")
    kinds = {
        "measurement_requests": "measurement_request",
        "research_case_requests": "research_case_request",
    }
    if manifest.get("kind") not in kinds:
        raise ValueError("unsupported curation bundle kind")
    for name in ("candidates", "results"):
        _artifact(
            root,
            f"{name}.{'jsonl' if name == 'candidates' else 'csv'}",
            manifest[f"{name}_sha256"],
        )
    candidates = read_jsonl(root / "candidates.jsonl")
    indexed = {candidate_key(row): row for row in candidates}
    if len(indexed) != len(candidates):
        raise ValueError("duplicate candidate in curation bundle")
    entries = manifest["requests"]
    if manifest.get("prepared") != len(entries) or manifest.get("candidates") != len(
        candidates
    ):
        raise ValueError("curation bundle counts differ")
    seen = set()
    for entry in entries:
        for path, digest in _entry_artifacts(entry):
            _artifact(root, path, digest)
        _, raw = _artifact(root, entry["request_path"], entry["request_sha256"])
        payload = json.loads(raw)
        if type(payload.get("protocol_version")) is not int:
            raise ValueError("unsupported measurement request protocol")
        request = MeasurementRequest.model_validate(payload)
        if request.kind != kinds[manifest["kind"]]:
            raise ValueError("request kind differs from curation bundle")
        candidate = request.candidate
        key = candidate_key(candidate)
        if key in seen:
            raise ValueError("duplicate request identity in curation bundle")
        seen.add(key)
        if entry["id"] != f"{key[0]}#{key[1]}" or key not in indexed:
            raise ValueError("request identity differs from curation manifest")
        for field in ("base_commit", "head_commit"):
            if candidate.get(field) != indexed[key].get(field):
                raise ValueError(f"request candidate differs: {field}")
        source = request.git_evidence
        if source.get("source_status") not in {"cached", "cached_branch_base"}:
            raise ValueError("request source reconstruction is pending")
        if (source.get("artifact_path"), source.get("artifact_sha256")) != (
            entry["source_patch_path"],
            entry["source_patch_sha256"],
        ):
            raise ValueError("request source differs from manifest")
        _, patch = _artifact(root, source["artifact_path"], source["artifact_sha256"])
        reference = (
            candidate["merge_commit"]
            if candidate.get("merge_commit")
            and source.get("integration_base_commit") == candidate["base_commit"]
            else candidate["head_commit"]
        )
        if request.evaluation_reference_commit != reference:
            raise ValueError("request evaluation reference differs from source")
        for field in ("full_integrated_patch_path", "full_integrated_patch_sha256"):
            if source.get(field) != entry.get(field):
                raise ValueError(f"request complete patch differs: {field}")
        review = request.metadata_selection_review
        validate_metadata_selection_review(review, candidate, patch)
        if request.pr_metadata != review["metadata"]:
            raise ValueError("request metadata differs from review")
        if request.pr_metadata != extract_pr_metadata(
            candidate, request.public_pr, patch
        ):
            raise ValueError("request public PR differs from metadata")
        if "review_path" in entry:
            _, review_raw = _artifact(
                root, entry["review_path"], entry["review_sha256"]
            )
            if json.loads(review_raw) != review:
                raise ValueError("exported review differs from request")
        profile = request.optimization_profile_hint
        if profile.taxonomy_sha256 != objective_catalog_sha256():
            raise ValueError("request objective catalog hash mismatch")
        requirements = request.qualification_requirements
        if (
            requirements.get("protocol_version") != "1.0"
            or requirements.get("protocols_sha256") != qualification_protocols_sha256()
        ):
            raise ValueError("request qualification protocol mismatch")
        protocols = qualification_protocols()
        if requirements.get("decision_rules") != protocols["decision_rules"]:
            raise ValueError("request qualification rules differ from protocol")
        for metric, specification in requirements.get("metric_protocols", {}).items():
            if protocols["metrics"].get(metric) != specification:
                raise ValueError(f"request metric protocol differs: {metric}")
        # The request may carry proposals, never a task admission receipt.
        if payload.get("selection_decision") is not None:
            raise ValueError("measurement request cannot admit or reject a task")
        for metric, specification in payload.get("requested_metrics", {}).items():
            if objective_catalog()["metrics"].get(metric) != specification:
                raise ValueError(f"request metric differs from catalog: {metric}")
    return entries


def import_measurement_bundle(root: Path, destination: Path) -> dict[str, Any]:
    """Persist immutable, content-addressed requests without creating tasks."""
    entries = validate_measurement_bundle(root)
    # Preserve failed/unprepared candidates and the producer's complete ledger,
    # even though only prepared requests can enter the execution queue.
    manifest_raw = (root / "manifest.json").read_bytes()
    ledger = destination / "bundles" / _sha(manifest_raw)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger_files = ("manifest.json", "candidates.jsonl", "results.csv")
    if ledger.exists():
        for name in ledger_files:
            _artifact(ledger, name, _sha((root / name).read_bytes()))
    else:
        with tempfile.TemporaryDirectory(dir=ledger.parent) as temporary:
            staging = Path(temporary) / "bundle"
            staging.mkdir()
            for name in ledger_files:
                shutil.copyfile(root / name, staging / name)
            staging.rename(ledger)
    requests = destination / "requests"
    imported = reused = 0
    for entry in entries:
        target = requests / entry["request_sha256"]
        artifacts = _entry_artifacts(entry)
        if target.exists():
            for path, digest in artifacts:
                _artifact(target, path, digest)
            reused += 1
            continue
        requests.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=requests) as temporary:
            staging = Path(temporary) / "request"
            staging.mkdir()
            for path, digest in artifacts:
                _, raw = _artifact(root, path, digest)
                output = staging / path
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(raw)
            (staging / "intake.json").write_bytes(
                _json_bytes(
                    {
                        "contract_version": CONTRACT_VERSION,
                        "kind": "request_import_receipt",
                        "status": "pending_measurement",
                        "entry": entry,
                        "tasks_created": 0,
                    }
                )
            )
            staging.rename(target)
        imported += 1
    return {"imported": imported, "reused": reused, "tasks_created": 0}
