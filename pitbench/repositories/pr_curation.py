"""Shared PR discovery records and evidence gates for historical solver tasks."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import subprocess
import tarfile
import tempfile
from collections import Counter, deque
from pathlib import Path, PurePosixPath
from statistics import fmean
from typing import Any

import yaml

from pitbench.problem_families.base import ProblemFamilyRegistry

REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
STAGES = (
    "discovered",
    "intent_reviewed",
    "source_verified",
    "build_verified",
    "workload_verified",
    "correctness_verified",
    "performance_confirmed",
    "task_ready",
)
SELECTION_DECISIONS = ("advance", "needs_evidence", "reject")
SELECTION_REVIEW_SCHEMA_VERSION = 2
SELECTION_MECHANISMS = (
    "algorithmic_search",
    "computational_efficiency",
    "combined",
    "other",
    "unknown",
)
SELECTION_METRIC_FAMILIES = ("exact", "anytime", "both", "unknown")
SOLVER_PHASES = (
    "presolve",
    "branching",
    "cut_generation",
    "primal_heuristic",
    "propagation",
    "local_search",
    "search_state",
    "numerical_kernel",
)
EFFECT_CHANNELS = (
    "work_per_event",
    "search_tree",
    "primal_trajectory",
    "dual_trajectory",
    "proof_time",
    "peak_memory",
)
CORRECTNESS_FAILURE_MODES = (
    "invalid_primal",
    "invalid_bound",
    "false_infeasibility",
    "incorrect_optimality",
    "search_incomplete",
    "numerical_instability",
    "resource_failure",
)
SEMANTIC_OBLIGATIONS = {
    "invalid_bound": "bound_validity",
    "false_infeasibility": "status_consistency",
    "incorrect_optimality": "optimality_certificate",
}
PHASE_REQUIRED_RISKS = {
    "presolve": {"invalid_bound", "false_infeasibility", "incorrect_optimality"},
    "branching": {"search_incomplete", "incorrect_optimality"},
    "cut_generation": {"invalid_bound", "false_infeasibility"},
    "propagation": {"invalid_bound", "false_infeasibility"},
    "primal_heuristic": {"invalid_primal", "search_incomplete"},
    "local_search": {"invalid_primal", "search_incomplete"},
    "search_state": {"invalid_primal", "search_incomplete"},
    "numerical_kernel": {
        "numerical_instability",
        "invalid_bound",
        "incorrect_optimality",
    },
}
REJECTION_BASES = (
    "outside_solver_engine",
    "no_optimization_mechanism",
    "correctness_only",
    "model_or_parameter_only",
    "unreproducible_source",
    "duplicate_lineage",
    "other",
)
NEEDED_EVIDENCE_TYPES = (
    "solver_call_path",
    "workload_activation",
    "historical_build",
    "independent_checker",
    "performance_protocol",
    "pr_lineage",
)
TITLE_SIGNALS = {
    "solver_mechanism": re.compile(
        r"\b(?:branch(?:ing)?|presolv(?:e|ing)|heuristic|cut(?:ting)?|"
        r"propagat(?:e|ion|or)|neighbou?r(?:hood)?|prun(?:e|ing)|"
        r"local.search|simplex|search.tree)\b",
        re.IGNORECASE,
    ),
    "performance": re.compile(
        r"\b(?:performance|faster|speed(?:up)?|optimi[sz](?:e|ation)|"
        r"efficien(?:t|cy)|memory|allocat(?:e|ion|ions)|cach(?:e|ing)|"
        r"incremental)\b",
        re.IGNORECASE,
    ),
}
NON_SOLVER_TITLE = re.compile(
    r"\b(?:docs?|readme|changelog|ci|workflow|dependenc(?:y|ies)|build)\b",
    re.IGNORECASE,
)
LINEAGE_TITLE = re.compile(
    r"\b(?:revert|backport|cherry[ -]?pick|forward[ -]?merge|"
    r"merge\s+(?:branch|pull request|master|main|release))\b",
    re.IGNORECASE,
)
RELATED_PR_HINT = re.compile(r"(?<!\w)#([1-9][0-9]*)")
SOURCE_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".cxx",
        ".h",
        ".hpp",
        ".cu",
        ".cuh",
        ".py",
        ".rs",
        ".java",
        ".kt",
        ".scala",
        ".jl",
        ".go",
        ".cs",
    }
)
NON_ENGINE_PATH_PARTS = frozenset(
    {
        ".github",
        "docs",
        "doc",
        "tests",
        "test",
        "examples",
        "example",
        "benchmarks",
        "benchmark",
        "vendor",
        "third_party",
        "cmake",
        "build",
    }
)
PHASE_PATH_HINTS = {
    "presolve": re.compile(r"presolv|preprocess", re.IGNORECASE),
    "branching": re.compile(r"branch|decision", re.IGNORECASE),
    "cut_generation": re.compile(r"(?:^|[/_.-])cut|separat|clique", re.IGNORECASE),
    "primal_heuristic": re.compile(r"heuristic|insertion|construction", re.IGNORECASE),
    "propagation": re.compile(r"propagat|nogood|constraint", re.IGNORECASE),
    "local_search": re.compile(r"local.?search|neighbou?r|operator", re.IGNORECASE),
    "search_state": re.compile(
        r"search|route|solution|population|genetic", re.IGNORECASE
    ),
    "numerical_kernel": re.compile(
        r"simplex|factor|pivot|basis|numerical", re.IGNORECASE
    ),
}
EVIDENCE_BY_STAGE = {
    "intent_reviewed": ("intent_evidence_path",),
    "source_verified": (
        "base_commit",
        "head_commit",
        "source_patch_path",
        "source_patch_sha256",
        "changed_source_files",
    ),
    "build_verified": ("build_evidence_path",),
    "workload_verified": ("workload_evidence_path",),
    "correctness_verified": ("correctness_evidence_path",),
    "performance_confirmed": ("performance_evidence_path",),
    "task_ready": ("task_id", "task_config_path"),
}
TASK_TRACKS = ("expert_reference", "open_opportunity")


def verified_exact_timeout_incumbent(
    run: dict[str, Any],
    target: object,
    *,
    objective_sense: str,
    known_optimum: bool,
) -> bool:
    """Recognize a checked feasible incumbent without treating it as a proof."""
    reported = run.get("reported_objective")
    verified = run.get("verified_objective")
    if (
        run.get("status") != "completed"
        or run.get("solver_termination") != "time_limit"
        or run.get("reported_valid") is not True
        or any(type(value) not in {int, float} for value in (reported, verified, target))
        or not all(math.isfinite(value) for value in (reported, verified, target))
        or not math.isclose(reported, verified, rel_tol=0, abs_tol=1e-6)
    ):
        return False
    if not known_optimum:
        return True
    if objective_sense == "minimize":
        return verified >= target - 1e-6
    if objective_sense == "maximize":
        return verified <= target + 1e-6
    return False


def load_repository_catalog(path: Path) -> dict[str, Any]:
    catalog = json.loads(path.read_text())
    if catalog.get("schema_version") != 1:
        raise ValueError("unsupported PR repository catalog schema")
    if (
        catalog.get("selection_review_schema_version")
        != SELECTION_REVIEW_SCHEMA_VERSION
    ):
        raise ValueError("unsupported PR selection review protocol")
    target = catalog.get("target_validated_pr_tasks")
    if type(target) is not int or target <= 0:
        raise ValueError("target_validated_pr_tasks must be a positive integer")
    repositories = catalog.get("repositories")
    if not isinstance(repositories, list) or not repositories:
        raise ValueError("PR repository catalog is empty")
    names = []
    for entry in repositories:
        name = entry.get("repository") if isinstance(entry, dict) else None
        if not isinstance(name, str) or not REPOSITORY_PATTERN.fullmatch(name):
            raise ValueError(f"invalid repository name: {name!r}")
        if entry.get("scope_status") not in {
            "task_backed",
            "pending_domain_review",
            "approved_source",
            "rejected",
        }:
            raise ValueError(f"invalid scope status for {name}")
        if entry["scope_status"] == "approved_source" and (
            not isinstance(entry.get("domain_review_evidence_url"), str)
            or not entry["domain_review_evidence_url"].startswith("https://")
            or not isinstance(entry.get("domain_review_note"), str)
            or not entry["domain_review_note"].strip()
        ):
            raise ValueError(f"approved source lacks domain review evidence: {name}")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("duplicate repository in PR catalog")
    return catalog


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def candidate_key(row: dict[str, Any]) -> tuple[str, int]:
    repository = row.get("repository")
    number = row.get("pr_number")
    if not isinstance(repository, str) or not REPOSITORY_PATTERN.fullmatch(repository):
        raise ValueError("candidate has invalid repository")
    if type(number) is not int or number <= 0:
        raise ValueError("candidate has invalid PR number")
    return repository, number


def validate_selection_review(review: dict[str, Any], candidate: dict[str, Any]) -> str:
    """Validate a solver-specific PR selection decision, not its later outcome."""
    key = candidate_key(candidate)
    if review.get("schema_version") != SELECTION_REVIEW_SCHEMA_VERSION:
        raise ValueError(f"{key} has unsupported selection review schema")
    if (review.get("repository"), review.get("pr_number")) != key:
        raise ValueError(f"{key} selection review identifies another PR")
    for field in ("base_commit", "head_commit"):
        if review.get(field) != candidate.get(field) or not re.fullmatch(
            r"[0-9a-f]{40}", str(review.get(field, ""))
        ):
            raise ValueError(f"{key} selection review has an unpinned {field}")
    decision = review.get("decision")
    if decision not in SELECTION_DECISIONS:
        raise ValueError(f"{key} selection review has invalid decision")
    if not isinstance(review.get("reason"), str) or not review["reason"].strip():
        raise ValueError(f"{key} selection review needs a reason")
    if decision == "reject" and review.get("rejection_basis") not in REJECTION_BASES:
        raise ValueError(f"{key} rejection needs a solver-specific basis")
    if decision == "needs_evidence" and (
        not isinstance(review.get("missing_evidence"), list)
        or not review["missing_evidence"]
        or any(
            evidence not in NEEDED_EVIDENCE_TYPES
            for evidence in review["missing_evidence"]
        )
    ):
        raise ValueError(f"{key} needs a concrete missing solver evidence type")
    if decision != "advance":
        return decision
    if review.get("repository_role") != "solver_engine":
        raise ValueError(f"{key} cannot advance without solver-engine scope")
    if review.get("mechanism") not in SELECTION_MECHANISMS[:3]:
        raise ValueError(f"{key} cannot advance without an optimization mechanism")
    if review.get("metric_family") not in SELECTION_METRIC_FAMILIES[:3]:
        raise ValueError(f"{key} cannot advance without an exact/anytime metric")
    if review.get("solver_phase") not in SOLVER_PHASES:
        raise ValueError(f"{key} cannot advance without a solver phase")
    channels = review.get("effect_channels")
    if (
        not isinstance(channels, list)
        or not channels
        or any(not isinstance(channel, str) for channel in channels)
        or len(channels) != len(set(channels))
        or any(channel not in EFFECT_CHANNELS for channel in channels)
    ):
        raise ValueError(f"{key} has invalid solver effect channels")
    risks = review.get("correctness_failure_modes")
    if (
        not isinstance(risks, list)
        or not risks
        or any(not isinstance(risk, str) for risk in risks)
        or len(risks) != len(set(risks))
        or any(risk not in CORRECTNESS_FAILURE_MODES for risk in risks)
    ):
        raise ValueError(f"{key} lacks concrete mathematical failure modes")
    if not set(risks).intersection(PHASE_REQUIRED_RISKS[review["solver_phase"]]):
        raise ValueError(f"{key} omits a correctness risk for its solver phase")
    entry_points = review.get("solver_entry_points")
    if (
        not isinstance(entry_points, list)
        or not entry_points
        or any(not isinstance(name, str) or not name.strip() for name in entry_points)
    ):
        raise ValueError(f"{key} lacks solver call-path entry points")
    related_prs = review.get("related_prs")
    if not isinstance(related_prs, list) or any(
        type(number) is not int or number <= 0 for number in related_prs
    ):
        raise ValueError(f"{key} has invalid related PR links")
    paths = review.get("changed_source_paths")
    if not isinstance(paths, list) or not paths:
        raise ValueError(f"{key} cannot advance without source paths")
    for raw in paths:
        if not isinstance(raw, str):
            raise ValueError(f"{key} has invalid changed source path")
        path = PurePosixPath(raw)
        if not raw or path.is_absolute() or ".." in path.parts or raw.endswith("/"):
            raise ValueError(f"{key} has invalid changed source path")
    for field in (
        "solver_behavior_hypothesis",
        "causal_chain",
        "structural_trigger",
        "activation_probe",
        "instance_selection_rule",
        "workload_exposure_plan",
        "mathematical_risk",
        "independent_check_plan",
        "hidden_generalization_plan",
        "overlap_review",
        "overlap_group",
        "resource_risk",
    ):
        if not isinstance(review.get(field), str) or not review[field].strip():
            raise ValueError(f"{key} cannot advance without {field}")
    urls = review.get("source_evidence_urls")
    if (
        not isinstance(urls, list)
        or not urls
        or any(
            not isinstance(url, str) or not url.startswith("https://") for url in urls
        )
    ):
        raise ValueError(f"{key} cannot advance without source evidence URLs")
    return decision


def validate_pinned_build_evidence(
    evidence: dict[str, Any], candidate: dict[str, Any], repository_root: Path
) -> None:
    """Require paired, hash-pinned build evidence for newly screened PRs."""
    key = candidate_key(candidate)
    if evidence.get("schema_version") != 1:
        raise ValueError(f"{key} has unsupported build evidence schema")
    for field in ("repository", "pr_number", "base_commit", "head_commit"):
        if evidence.get(field) != candidate.get(field):
            raise ValueError(f"{key} build evidence has mismatched {field}")
    if evidence.get("source_patch_sha256") != candidate.get("source_patch_sha256"):
        raise ValueError(f"{key} build evidence uses another source patch")
    dockerfile = evidence.get("dockerfile_path")
    if not isinstance(dockerfile, str) or not dockerfile:
        raise ValueError(f"{key} build evidence lacks a Dockerfile")
    dockerfile_parts = PurePosixPath(dockerfile)
    if dockerfile_parts.is_absolute() or ".." in dockerfile_parts.parts:
        raise ValueError(f"{key} build Dockerfile must be inside the repository")
    path = repository_root / dockerfile
    if not path.is_file() or hashlib.sha256(
        path.read_bytes()
    ).hexdigest() != evidence.get("dockerfile_sha256"):
        raise ValueError(f"{key} build Dockerfile hash differs from evidence")
    if not re.fullmatch(r"[0-9a-f]{64}", str(evidence.get("parent_image_sha256", ""))):
        raise ValueError(f"{key} build evidence lacks pinned parent image")
    if evidence.get("build_network") != "none":
        raise ValueError(f"{key} build evidence must use an offline build")
    variants = evidence.get("variants")
    if not isinstance(variants, dict) or set(variants) != {"base", "human_pr"}:
        raise ValueError(f"{key} build evidence needs base and PR variants")
    for name, variant in variants.items():
        if not isinstance(variant, dict) or not re.fullmatch(
            r"[0-9a-f]{64}", str(variant.get("image_sha256", ""))
        ):
            raise ValueError(f"{key} {name} build lacks an image digest")
        if any(
            type(variant.get(field)) is not int or variant[field] != 0
            for field in ("build_exit_code", "executable_check_exit_code")
        ):
            raise ValueError(f"{key} {name} build or executable check failed")


def validate_workload_exposure_evidence(
    evidence: dict[str, Any],
    candidate: dict[str, Any],
    review: dict[str, Any],
    repository_root: Path,
) -> None:
    """Check that a frozen workload exercises the reviewed solver mechanism."""
    key = candidate_key(candidate)
    if evidence.get("schema_version") != 1:
        raise ValueError(f"{key} has unsupported workload evidence schema")
    for field in ("repository", "pr_number", "base_commit", "head_commit"):
        if evidence.get(field) != candidate.get(field):
            raise ValueError(f"{key} workload evidence has mismatched {field}")
    if evidence.get("source_patch_sha256") != candidate.get("source_patch_sha256"):
        raise ValueError(f"{key} workload evidence uses another source patch")
    selection_digest = hashlib.sha256(
        review["instance_selection_rule"].encode()
    ).hexdigest()
    if evidence.get("instance_selection_rule_sha256") != selection_digest:
        raise ValueError(f"{key} workload selection differs from the prior review")
    for field in ("instance_manifest_path", "activation_artifact_path"):
        raw = evidence.get(field)
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"{key} workload evidence lacks {field}")
        path = PurePosixPath(raw)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"{key} workload evidence has unsafe {field}")
        actual = repository_root / raw
        digest_field = field.removesuffix("_path") + "_sha256"
        if not actual.is_file() or hashlib.sha256(
            actual.read_bytes()
        ).hexdigest() != evidence.get(digest_field):
            raise ValueError(f"{key} workload {field} hash mismatch")
    cases = evidence.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{key} workload evidence has no cases")
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError(f"{key} workload case is invalid")
        instance_id, family = case.get("instance_id"), case.get("family")
        if (
            not isinstance(instance_id, str)
            or not instance_id
            or not isinstance(family, str)
            or not family
            or instance_id in seen
        ):
            raise ValueError(f"{key} workload case lacks a unique instance/family")
        seen.add(instance_id)
        for field in (
            "base_precondition_count",
            "pr_precondition_count",
            "pr_changed_path_count",
        ):
            value = case.get(field)
            if type(value) is not int or value < 0:
                raise ValueError(f"{key} workload case has invalid {field}")
    if not any(
        case["base_precondition_count"] > 0
        and case["pr_precondition_count"] > 0
        and case["pr_changed_path_count"] > 0
        for case in cases
    ):
        raise ValueError(f"{key} workload never exercised the reviewed code path")
    manifest = json.loads(
        (repository_root / evidence["instance_manifest_path"]).read_text()
    )
    expected_cases = [
        {"instance_id": case["instance_id"], "family": case["family"]} for case in cases
    ]
    if manifest.get("schema_version") != 1 or manifest.get("cases") != expected_cases:
        raise ValueError(f"{key} measured cases differ from frozen instance manifest")
    activation = json.loads(
        (repository_root / evidence["activation_artifact_path"]).read_text()
    )
    if activation.get("schema_version") != 1 or activation.get("cases") != cases:
        raise ValueError(f"{key} workload counts differ from activation artifact")
    traces = activation.get("trace_artifacts", [])
    if not isinstance(traces, list):
        raise ValueError(f"{key} workload trace artifacts are invalid")
    seen_traces = set()
    for trace in traces:
        if not isinstance(trace, dict):
            raise ValueError(f"{key} workload trace artifact is invalid")
        raw = trace.get("path")
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"{key} workload trace path is invalid")
        relative = PurePosixPath(raw)
        if relative.is_absolute() or ".." in relative.parts or raw in seen_traces:
            raise ValueError(f"{key} workload trace path is unsafe or duplicated")
        seen_traces.add(raw)
        path = repository_root / raw
        if not path.is_file() or hashlib.sha256(
            path.read_bytes()
        ).hexdigest() != trace.get("sha256"):
            raise ValueError(f"{key} workload trace artifact hash mismatch")


def validate_independent_correctness_evidence(
    evidence: dict[str, Any],
    candidate: dict[str, Any],
    repository_root: Path,
    *,
    required_semantic_obligations: set[str] | None = None,
) -> None:
    """Audit a paired public run ledger and its independent solution checks."""
    key = candidate_key(candidate)
    if evidence.get("schema_version") != 1:
        raise ValueError(f"{key} has unsupported correctness evidence schema")
    for field in ("repository", "pr_number", "base_commit", "head_commit"):
        if evidence.get(field) != candidate.get(field):
            raise ValueError(f"{key} correctness evidence has mismatched {field}")
    if evidence.get("source_patch_sha256") != candidate.get("source_patch_sha256"):
        raise ValueError(f"{key} correctness evidence uses another source patch")
    expected_fixed_parameters = None
    if candidate.get("task_config_path"):
        task_relative = PurePosixPath(candidate["task_config_path"])
        if task_relative.is_absolute() or ".." in task_relative.parts:
            raise ValueError(f"{key} correctness task config path is unsafe")
        task = yaml.safe_load((repository_root / task_relative).read_text())
        expected_fixed_parameters = task.get("evaluation", {}).get(
            "fixed_solver_parameters"
        )

    def checked_file(raw: object, digest: object) -> Path:
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"{key} correctness evidence has an invalid file path")
        relative = PurePosixPath(raw)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{key} correctness evidence has an unsafe file path")
        path = repository_root / raw
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise ValueError(f"{key} correctness file hash mismatch: {raw}")
        return path

    checked_file(
        evidence.get("checker_artifact_path"), evidence.get("checker_module_sha256")
    )
    ledger = checked_file(
        evidence.get("run_ledger_path"), evidence.get("run_ledger_sha256")
    )
    rows = read_jsonl(ledger)
    expected = evidence.get("expected_runs")
    if type(expected) is not int or expected <= 0 or len(rows) != expected:
        raise ValueError(f"{key} correctness run grid is incomplete")
    if evidence.get("retained_runs") != len(rows):
        raise ValueError(f"{key} correctness retained-run count differs")
    run_keys = set()
    paired_runs: dict[tuple[Any, Any, Any], set[str]] = {}
    solutions = 0
    for row in rows:
        round_index = row.get("round", 0)
        run_key = (
            round_index,
            row.get("instance_id"),
            row.get("variant"),
            row.get("solver_seed"),
            row.get("budget_sec"),
        )
        if (
            type(round_index) is not int
            or round_index < 0
            or row.get("variant") not in {"base", "human_pr"}
            or run_key in run_keys
            or type(row.get("solver_seed")) is not int
            or not isinstance(row.get("budget_sec"), (int, float))
            or row["budget_sec"] <= 0
        ):
            raise ValueError(f"{key} correctness ledger has invalid run identity")
        run_keys.add(run_key)
        paired_runs.setdefault(
            (run_key[0], run_key[1], run_key[3], run_key[4]), set()
        ).add(row["variant"])
        checked_file(row.get("instance_path"), row.get("instance_sha256"))
        if expected_fixed_parameters is not None:
            parameters = checked_file(
                row.get("fixed_solver_parameters_path"),
                row.get("fixed_solver_parameters_sha256"),
            )
            if json.loads(parameters.read_text()) != expected_fixed_parameters:
                raise ValueError(f"{key} public run used different fixed parameters")
        elif row.get("fixed_solver_parameters_path"):
            raise ValueError(f"{key} public run has undeclared fixed parameters")
        output = checked_file(row.get("output_path"), row.get("output_sha256"))
        if row.get("raw_output_path"):
            checked_file(row["raw_output_path"], row.get("raw_output_sha256"))
        payload = json.loads(output.read_text())
        if payload.get("valid") != row.get("driver_valid") or payload.get(
            "objective"
        ) != row.get("driver_objective"):
            raise ValueError(f"{key} correctness ledger differs from raw output")
        if row.get("solution_path"):
            checked_file(row["solution_path"], row.get("solution_sha256"))
            solutions += 1
            if (
                row.get("independent_feasible") is not True
                or row.get("objective_matches") is not True
                or type(row.get("independent_objective")) not in {int, float}
                or type(row.get("driver_objective")) not in {int, float}
                or not math.isclose(
                    row["independent_objective"],
                    row["driver_objective"],
                    rel_tol=0,
                    abs_tol=1e-6,
                )
                or row.get("driver_valid") is not True
            ):
                raise ValueError(f"{key} has an independently invalid solution")
    if any(states != {"base", "human_pr"} for states in paired_runs.values()):
        raise ValueError(f"{key} correctness ledger lacks paired source states")
    if solutions == 0 or evidence.get("verified_solution_count") != solutions:
        raise ValueError(f"{key} correctness evidence has no verified solutions")
    if evidence.get("invalid_solution_count") != 0:
        raise ValueError(f"{key} correctness evidence reports invalid solutions")
    variants = evidence.get("upstream_targeted_tests")
    if not isinstance(variants, dict) or set(variants) != {"base", "human_pr"}:
        raise ValueError(f"{key} correctness evidence lacks paired tests")
    if any(not isinstance(value, dict) for value in variants.values()):
        raise ValueError(f"{key} correctness evidence has invalid test results")
    if any(
        type(variants[state].get("failed")) is not int
        or variants[state]["failed"] != 0
        or type(variants[state].get("passed")) is not int
        or variants[state]["passed"] < 0
        for state in variants
    ):
        raise ValueError(f"{key} correctness regression tests failed")
    required_semantic_obligations = required_semantic_obligations or set()
    semantic = evidence.get("semantic_obligations", {})
    if not isinstance(semantic, dict) or not required_semantic_obligations <= set(
        semantic
    ):
        raise ValueError(f"{key} lacks mathematical semantic checks")
    for obligation in required_semantic_obligations:
        check = semantic[obligation]
        if (
            not isinstance(check, dict)
            or not isinstance(check.get("independent_checker"), str)
            or not check["independent_checker"].strip()
            or type(check.get("case_count")) is not int
            or check["case_count"] <= 0
            or type(check.get("invalid_count")) is not int
            or check["invalid_count"] != 0
        ):
            raise ValueError(f"{key} has invalid {obligation} check results")
        artifact = checked_file(
            check.get("artifact_path"), check.get("artifact_sha256")
        )
        measured = json.loads(artifact.read_text())
        if (
            measured.get("case_count") != check["case_count"]
            or measured.get("invalid_count") != 0
        ):
            raise ValueError(f"{key} {obligation} artifact differs from its summary")


def validate_open_opportunity_evidence(
    evidence: dict[str, Any], candidate: dict[str, Any], repository_root: Path
) -> None:
    """Qualify an open challenge with a verified baseline gap, not a PR gain."""
    key = candidate_key(candidate)
    if evidence.get("schema_version") != 1:
        raise ValueError(f"{key} has unsupported opportunity evidence schema")
    for field in ("repository", "pr_number", "base_commit", "head_commit"):
        if evidence.get(field) != candidate.get(field):
            raise ValueError(f"{key} opportunity evidence has mismatched {field}")
    if evidence.get("source_patch_sha256") != candidate.get("source_patch_sha256"):
        raise ValueError(f"{key} opportunity evidence uses another source patch")
    opportunity_kind = evidence.get("opportunity_kind")
    if opportunity_kind not in {"remaining_verified_gap", "unsolved_exact"}:
        raise ValueError(f"{key} lacks a supported solver opportunity measure")
    mode = evidence.get("qualification_mode", "paired_reference")
    if mode not in {"paired_reference", "baseline_qualification"}:
        raise ValueError(f"{key} has unknown opportunity qualification mode")
    if opportunity_kind == "unsolved_exact" and mode != "baseline_qualification":
        raise ValueError(f"{key} exact opportunity needs baseline qualification")
    ledger_name = "runs" if mode == "baseline_qualification" else "pairs"
    artifacts = {}
    for name in ("manifest", "analysis", ledger_name):
        raw = evidence.get(f"{name}_path")
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"{key} opportunity evidence lacks {name}_path")
        relative = PurePosixPath(raw)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{key} opportunity evidence has unsafe {name}_path")
        path = repository_root / raw
        if not path.is_file() or hashlib.sha256(
            path.read_bytes()
        ).hexdigest() != evidence.get(f"{name}_sha256"):
            raise ValueError(f"{key} opportunity {name} hash mismatch")
        artifacts[name] = path
    manifest = json.loads(artifacts["manifest"].read_text())
    analysis = json.loads(artifacts["analysis"].read_text())
    if mode == "baseline_qualification":
        expected_conclusion = (
            "verified_unsolved_room"
            if opportunity_kind == "unsolved_exact"
            else "verified_improvement_room"
        )
        if (
            manifest.get("run_mode") != "baseline_qualification"
            or manifest.get("old", {}).get("commit") != candidate["base_commit"]
            or analysis.get("manifest_sha256") != evidence["manifest_sha256"]
            or analysis.get("expected_runs") != analysis.get("observed_unique_runs")
            or analysis.get("missing_runs")
            or analysis.get("failed_runs")
            or analysis.get("invalid_runs")
            or analysis.get("conclusion") != expected_conclusion
            or evidence.get("human_reference_classification") != "not_evaluated"
        ):
            raise ValueError(f"{key} baseline qualification is incomplete")
        attempts = read_jsonl(artifacts["runs"])
        if len(attempts) != analysis.get("retained_run_attempts"):
            raise ValueError(f"{key} baseline raw run attempts differ")
        latest = {}
        for row in attempts:
            run_key = (row.get("instance_id"), row.get("solver_seed"))
            if (
                run_key not in latest
                or row.get("base", {}).get("status") == "completed"
            ):
                latest[run_key] = row
        if len(latest) != analysis["expected_runs"]:
            raise ValueError(f"{key} baseline run ledger is incomplete")
        if opportunity_kind == "unsolved_exact":
            task_config = (
                yaml.safe_load(
                    (repository_root / candidate["task_config_path"]).read_text()
                )
                if candidate.get("task_config_path")
                else {}
            )
            oracle = task_config.get("oracle", {})
            objective_sense = oracle.get("objective_sense", "minimize")
            known_optimum = oracle.get("kind", "known_optimum") == "known_optimum"
            solved = 0
            unsolved = 0
            for row in latest.values():
                run = row.get("base", {})
                if (
                    run.get("status") == "no_solution"
                    and run.get("reported", {}).get("solver_termination")
                    == "time_limit"
                ):
                    unsolved += 1
                    continue
                if verified_exact_timeout_incumbent(
                    run,
                    row.get("target"),
                    objective_sense=objective_sense,
                    known_optimum=known_optimum,
                ):
                    unsolved += 1
                    continue
                verified, reported, target = (
                    run.get("verified_objective"),
                    run.get("reported_objective"),
                    row.get("target"),
                )
                if (
                    run.get("status") != "completed"
                    or run.get("solver_termination") != "optimal"
                    or run.get("reported_valid") is not True
                    or any(
                        type(value) not in {int, float}
                        for value in (verified, reported, target)
                    )
                    or not math.isclose(verified, reported, rel_tol=0, abs_tol=1e-6)
                    or not math.isclose(verified, target, rel_tol=0, abs_tol=1e-6)
                ):
                    raise ValueError(f"{key} exact baseline has invalid outcomes")
                solved += 1
            if (
                solved <= 0
                or unsolved <= 0
                or solved + unsolved != analysis["expected_runs"]
                or evidence.get("base_verified_solved_runs") != solved
                or evidence.get("base_time_limit_runs") != unsolved
                or analysis.get("base_verified_solved_runs") != solved
                or analysis.get("base_time_limit_runs") != unsolved
            ):
                raise ValueError(f"{key} exact baseline has no verified solve room")
            return
        gaps = []
        for row in latest.values():
            run = row.get("base", {})
            gap = run.get("normalized_gap")
            if (
                run.get("status") != "completed"
                or run.get("reported_valid") is not True
                or type(gap) not in {int, float}
                or not math.isfinite(gap)
                or gap < 0
                or type(run.get("verified_objective")) not in {int, float}
                or type(run.get("reported_objective")) not in {int, float}
                or not math.isclose(
                    run["verified_objective"],
                    run["reported_objective"],
                    rel_tol=0,
                    abs_tol=1e-6,
                )
            ):
                raise ValueError(f"{key} baseline contains invalid solver runs")
            gaps.append(gap)
        base_mean_gap = fmean(gaps)
        if (
            base_mean_gap <= 0
            or type(evidence.get("base_mean_normalized_gap")) not in {int, float}
            or type(analysis.get("base_mean_normalized_gap")) not in {int, float}
            or not math.isclose(
                base_mean_gap,
                evidence["base_mean_normalized_gap"],
                abs_tol=1e-12,
            )
            or not math.isclose(
                base_mean_gap,
                analysis["base_mean_normalized_gap"],
                abs_tol=1e-12,
            )
        ):
            raise ValueError(f"{key} baseline has no verified improvement room")
        return
    if (
        manifest.get("old", {}).get("commit") != candidate["base_commit"]
        or manifest.get("new", {}).get("commit") != candidate["head_commit"]
        or analysis.get("manifest_sha256") != evidence["manifest_sha256"]
        or analysis.get("expected_pairs") != analysis.get("observed_unique_pairs")
        or analysis.get("missing_pairs")
        or analysis.get("failed_pairs")
    ):
        raise ValueError(f"{key} opportunity hidden grid is incomplete or mismatched")
    classification = analysis.get("pitbench_performance", {}).get(
        "confirmed_classification"
    )
    if classification != evidence.get("human_reference_classification") or (
        classification == "improved"
    ):
        raise ValueError(f"{key} positive human PR belongs in the expert track")
    rows = read_jsonl(artifacts["pairs"])
    if len(rows) != analysis.get("retained_pair_attempts"):
        raise ValueError(f"{key} opportunity raw pair attempts differ")
    unique = {}
    for row in rows:
        pair_key = (row.get("round"), row.get("instance_id"), row.get("solver_seed"))
        if pair_key not in unique or all(
            row.get(label, {}).get("status") == "completed" for label in ("old", "new")
        ):
            unique[pair_key] = row
    if len(unique) != analysis["expected_pairs"]:
        raise ValueError(f"{key} opportunity pair ledger is incomplete")
    gaps = []
    for pair in unique.values():
        for label in ("old", "new"):
            run = pair.get(label, {})
            if (
                run.get("status") != "completed"
                or run.get("reported_valid") is not True
                or type(run.get("verified_objective")) not in {int, float}
                or type(run.get("reported_objective")) not in {int, float}
                or not math.isclose(
                    run["verified_objective"],
                    run["reported_objective"],
                    rel_tol=0,
                    abs_tol=1e-6,
                )
            ):
                raise ValueError(f"{key} opportunity contains invalid solver runs")
        gap = pair["old"].get("normalized_gap")
        if type(gap) not in {int, float} or not math.isfinite(gap) or gap < 0:
            raise ValueError(f"{key} opportunity base has invalid verified gap")
        gaps.append(gap)
    base_mean_gap = fmean(gaps)
    if (
        base_mean_gap <= 0
        or type(evidence.get("base_mean_normalized_gap")) not in {int, float}
        or not math.isclose(
            evidence["base_mean_normalized_gap"], base_mean_gap, abs_tol=1e-12
        )
    ):
        raise ValueError(f"{key} has no independently measured improvement room")


def validate_task_integration_evidence(
    evidence: dict[str, Any], candidate: dict[str, Any], repository_root: Path
) -> None:
    """Require a Git-backed client and a complete limited real Judge run."""
    key = candidate_key(candidate)
    if evidence.get("schema_version") != 1:
        raise ValueError(f"{key} has unsupported integration evidence schema")
    for field in ("repository", "pr_number", "base_commit", "task_id"):
        if evidence.get(field) != candidate.get(field):
            raise ValueError(f"{key} integration evidence has mismatched {field}")

    def checked_file(name: str) -> Path:
        raw = evidence.get(f"{name}_path")
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"{key} integration evidence lacks {name}_path")
        relative = PurePosixPath(raw)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{key} integration evidence has unsafe {name}_path")
        path = repository_root / raw
        if not path.is_file() or hashlib.sha256(
            path.read_bytes()
        ).hexdigest() != evidence.get(f"{name}_sha256"):
            raise ValueError(f"{key} integration {name} hash mismatch")
        return path

    task_path = checked_file("task_config")
    if evidence["task_config_path"] != candidate.get("task_config_path"):
        raise ValueError(f"{key} integration task config path differs")
    task = yaml.safe_load(task_path.read_text())
    agent = json.loads(checked_file("agent_identity").read_text())
    checkout = json.loads(checked_file("checkout_audit").read_text())
    if (
        task.get("task_id") != candidate["task_id"]
        or task.get("release", {}).get("base_commit") != candidate["base_commit"]
        or task.get("repository", {}).get("agent_image") != agent.get("image_reference")
        or task.get("repository", {}).get("judge_image") != evidence.get("judge_image")
        or agent.get("git_head") != candidate["base_commit"]
        or checkout.get("base_commit") != candidate["base_commit"]
        or not isinstance(agent.get("submodule_status"), list)
        or agent.get("submodule_status") != checkout.get("submodule_status")
        or any(not line.startswith(" ") for line in agent.get("submodule_status", []))
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(agent.get("image_id", "")))
    ):
        raise ValueError(f"{key} integration client image or Git checkout differs")
    evaluation = json.loads(checked_file("evaluation").read_text())
    grid = json.loads(checked_file("expected_grid").read_text())
    observations = read_jsonl(checked_file("observations"))
    summary = evaluation.get("payload", {}).get("summary", {})
    if (
        type(evidence.get("dry_run_exit_code")) is not int
        or (
            evidence.get("dry_run_exit_code") != 0
            and not str(evidence.get("dry_run_exit_note", "")).strip()
        )
        or evaluation.get("completed") is not True
        or evaluation.get("payload", {}).get("task_id") != candidate["task_id"]
        or evaluation.get("payload", {}).get("validity", {}).get("accepted") is not True
        or grid.get("task_id") != candidate["task_id"]
        or not isinstance(grid.get("runs"), list)
        or summary.get("observation_count") != len(observations)
        or len(grid["runs"]) != len(observations)
    ):
        raise ValueError(f"{key} limited Judge run is incomplete")
    fields = (
        "task_id",
        "code_state",
        "instance_set",
        "instance_id",
        "solver_seed",
        "budget_sec",
    )
    expected = {tuple(row.get(field) for field in fields) for row in grid["runs"]}
    observed = {tuple(row.get(field) for field in fields) for row in observations}
    if len(expected) != len(observations) or observed != expected:
        raise ValueError(f"{key} Judge observations differ from expected grid")
    fixed_parameters = task.get("evaluation", {}).get("fixed_solver_parameters")
    if fixed_parameters is not None:
        run_root = (repository_root / evidence["evaluation_path"]).parent
        for row in observations:
            stem = f"seed-{row['solver_seed']}-budget-{row['budget_sec']:g}"
            directory = (
                run_root
                / row["code_state"]
                / row["instance_set"]
                / row["instance_id"]
            ).resolve()
            if not directory.is_relative_to(repository_root.resolve()):
                raise ValueError(f"{key} Judge parameter path escapes repository")
            parameters_path = directory / f"{stem}.parameters.json"
            process_path = directory / f"{stem}.process.json"
            if (
                not parameters_path.is_file()
                or json.loads(parameters_path.read_text()) != fixed_parameters
                or not process_path.is_file()
            ):
                raise ValueError(f"{key} Judge did not retain fixed parameters")
            process = json.loads(process_path.read_text())
            argv = process.get("argv", [])
            if (
                process.get("returncode") != 0
                or "--parameters" not in argv
                or argv.index("--parameters") + 1 >= len(argv)
                or Path(argv[argv.index("--parameters") + 1]).name
                != parameters_path.name
            ):
                raise ValueError(f"{key} Judge did not use fixed parameters")
    valid_count = 0
    for row in observations:
        if row.get("status") == "completed" and row.get("valid") is True:
            valid_count += 1
        elif not (
            row.get("status") == "no_solution"
            and row.get("solver_termination") == "time_limit"
        ):
            raise ValueError(f"{key} Judge has invalid or infrastructure observations")
    if valid_count == 0 or summary.get("valid_observation_count") != valid_count:
        raise ValueError(f"{key} Judge has no independently valid observations")


def _record_artifact(
    evidence: dict[str, Any], name: str, path: Path, repository_root: Path
) -> None:
    """Bind a local evidence file to a safe relative path and its content hash."""
    root = repository_root.resolve()
    actual = path.resolve()
    if not actual.is_relative_to(root) or not actual.is_file():
        raise ValueError(f"evidence artifact is missing or outside repository: {path}")
    evidence[f"{name}_path"] = actual.relative_to(root).as_posix()
    evidence[f"{name}_sha256"] = hashlib.sha256(actual.read_bytes()).hexdigest()


def build_paired_correctness_evidence(
    candidate: dict[str, Any],
    repository_root: Path,
    pairs_path: Path,
    ledger_path: Path,
    checker_path: Path,
    test_results: dict[str, Any],
    semantic_obligations: dict[str, Any],
) -> dict[str, Any]:
    """Recheck public Base/PR solutions and retain every paired run outcome."""
    root = repository_root.resolve()
    task_relative = PurePosixPath(candidate["task_config_path"])
    if task_relative.is_absolute() or ".." in task_relative.parts:
        raise ValueError("correctness task config path is unsafe")
    task = yaml.safe_load((root / task_relative).read_text())
    verifier = ProblemFamilyRegistry.load(task["problem_family"])
    exact = task["evaluation"]["performance_protocol"] == "exact_verified_solve"
    fixed_parameters = task["evaluation"].get("fixed_solver_parameters")
    expected_images = None
    if candidate.get("build_evidence_path"):
        build_relative = PurePosixPath(candidate["build_evidence_path"])
        if build_relative.is_absolute() or ".." in build_relative.parts:
            raise ValueError("public build evidence path is unsafe")
        build = json.loads((root / build_relative).read_text())
        expected_images = {
            variant: "sha256:" + build["variants"][variant]["image_sha256"]
            for variant in ("base", "human_pr")
        }
        if task["repository"]["judge_image"] != expected_images["base"]:
            raise ValueError("task Judge image differs from paired build")
    pairs = read_jsonl(pairs_path)
    if not pairs:
        raise ValueError("public correctness panel has no paired runs")

    def local_path(raw: str) -> Path:
        path = Path(raw)
        return path if path.is_absolute() else root / path

    ledger = []
    seen = set()
    valid_solutions = 0
    normal_timeouts = 0
    for pair in pairs:
        identity = (
            pair.get("round"),
            pair.get("instance_id"),
            pair.get("solver_seed"),
            pair.get("budget_sec"),
        )
        if (
            type(identity[0]) is not int
            or identity[0] < 0
            or not isinstance(identity[1], str)
            or not identity[1]
            or type(identity[2]) is not int
            or type(identity[3]) not in {int, float}
            or identity[3] <= 0
            or identity in seen
        ):
            raise ValueError("public correctness panel has a duplicate or invalid pair")
        seen.add(identity)
        if pair.get("old", {}).get("image") == pair.get("new", {}).get("image"):
            raise ValueError("public Base and PR images are identical")
        for old_label, variant in (("old", "base"), ("new", "human_pr")):
            run = pair.get(old_label)
            if not isinstance(run, dict):
                raise ValueError("public correctness pair lacks a source state")
            expected_commit = (
                candidate["base_commit"]
                if old_label == "old"
                else candidate["head_commit"]
            )
            if (
                run.get("commit") != expected_commit
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(run.get("image", "")))
                or run.get("container_exit_code") != 0
            ):
                raise ValueError("public solver run differs from pinned source image")
            if expected_images is not None and run["image"] != expected_images[variant]:
                raise ValueError("public solver image differs from paired build")
            item = {
                "round": identity[0],
                "instance_id": identity[1],
                "variant": variant,
                "solver_seed": identity[2],
                "budget_sec": identity[3],
            }
            if fixed_parameters is not None:
                raw_parameters = run.get("fixed_solver_parameters_path")
                if not isinstance(raw_parameters, str):
                    raise ValueError("public run lacks fixed solver parameters")
                parameters_path = local_path(raw_parameters)
                _record_artifact(item, "fixed_solver_parameters", parameters_path, root)
                if (
                    item["fixed_solver_parameters_sha256"]
                    != run.get("fixed_solver_parameters_sha256")
                    or json.loads(parameters_path.read_text()) != fixed_parameters
                ):
                    raise ValueError("public run used different fixed parameters")
            elif run.get("fixed_solver_parameters_path"):
                raise ValueError("public run has undeclared fixed parameters")
            _record_artifact(item, "instance", local_path(pair["instance_path"]), root)
            _record_artifact(item, "output", local_path(run["output_path"]), root)
            payload = json.loads((root / item["output_path"]).read_text())
            item["driver_valid"] = payload.get("valid")
            item["driver_objective"] = payload.get("objective")
            item["solver_status"] = payload.get("solver_status")
            item["solver_termination"] = payload.get("solver_termination")
            if run.get("status") == "completed":
                if payload.get("valid") is not True or (
                    exact
                    and payload.get("solver_termination")
                    not in {"optimal", "time_limit"}
                ):
                    raise ValueError("completed public solver run is not valid")
                if (
                    run.get("reported_valid") is not True
                    or type(run.get("reported_objective")) not in {int, float}
                    or type(payload.get("objective")) not in {int, float}
                    or not math.isclose(
                        run["reported_objective"], payload["objective"], abs_tol=1e-6
                    )
                ):
                    raise ValueError("public run ledger differs from raw solver output")
                _record_artifact(
                    item, "solution", local_path(run["solution_path"]), root
                )
                checked = verifier.verify(
                    root / item["instance_path"], root / item["solution_path"]
                )
                objective = checked.objective
                if (
                    not checked.feasible
                    or type(objective) not in {int, float}
                    or type(payload.get("objective")) not in {int, float}
                    or type(run.get("verified_objective")) not in {int, float}
                    or not math.isclose(objective, payload["objective"], abs_tol=1e-6)
                    or not math.isclose(
                        objective, run["verified_objective"], abs_tol=1e-6
                    )
                ):
                    raise ValueError(
                        "public solution failed independent original-model check"
                    )
                item.update(
                    independent_feasible=True,
                    independent_objective=objective,
                    objective_matches=True,
                )
                valid_solutions += 1
                if exact and payload.get("solver_termination") == "time_limit":
                    item["normal_timeout"] = True
                    normal_timeouts += 1
            elif (
                exact
                and run.get("status") == "no_solution"
                and payload.get("valid") is False
                and payload.get("solver_termination") == "time_limit"
                and run.get("reported", {}).get("solver_termination") == "time_limit"
                and run.get("reported") == payload
            ):
                item["normal_timeout"] = True
                normal_timeouts += 1
            else:
                raise ValueError(
                    "public solver run failed or has an unqualified timeout"
                )
            ledger.append(item)

    if set(test_results) != {"base", "human_pr"}:
        raise ValueError("public correctness needs paired test results")
    tests = {}
    for variant, result in test_results.items():
        if not isinstance(result, dict):
            raise ValueError("public test result is invalid")
        tests[variant] = {
            "passed": result.get("passed"),
            "failed": result.get("failed"),
        }
        if result.get("log_path"):
            _record_artifact(
                tests[variant], "log", local_path(result["log_path"]), root
            )
        if result.get("note"):
            tests[variant]["note"] = result["note"]

    semantic = {}
    for name, check in semantic_obligations.items():
        if not isinstance(check, dict):
            raise ValueError("mathematical check summary is invalid")
        semantic[name] = {
            "independent_checker": check.get("independent_checker"),
            "case_count": check.get("case_count"),
            "invalid_count": check.get("invalid_count"),
        }
        _record_artifact(
            semantic[name], "artifact", local_path(check["artifact_path"]), root
        )

    target = ledger_path if ledger_path.is_absolute() else root / ledger_path
    target = target.resolve()
    if not target.is_relative_to(root) or target.exists():
        raise ValueError("correctness ledger must be a new file inside repository")
    write_jsonl(target, ledger)
    evidence = {
        "schema_version": 1,
        **{
            field: candidate[field]
            for field in (
                "repository",
                "pr_number",
                "base_commit",
                "head_commit",
                "source_patch_sha256",
            )
        },
        "expected_runs": len(ledger),
        "retained_runs": len(ledger),
        "verified_solution_count": valid_solutions,
        "invalid_solution_count": 0,
        "normal_timeout_count": normal_timeouts,
        "upstream_targeted_tests": tests,
        "semantic_obligations": semantic,
        "claim_boundary": (
            "Complete paired public original-problem correctness panel only; "
            "no hidden human PR effect is claimed."
        ),
    }
    _record_artifact(evidence, "checker_artifact", checker_path, root)
    evidence["checker_module_sha256"] = evidence.pop("checker_artifact_sha256")
    _record_artifact(evidence, "run_ledger", target, root)
    _record_artifact(evidence, "raw_pairs", pairs_path, root)
    required = set()
    if candidate.get("selection_review_path"):
        review = json.loads((root / candidate["selection_review_path"]).read_text())
        required = {
            SEMANTIC_OBLIGATIONS[risk]
            for risk in review["correctness_failure_modes"]
            if risk in SEMANTIC_OBLIGATIONS
        }
    validate_independent_correctness_evidence(
        evidence, candidate, root, required_semantic_obligations=required
    )
    return evidence


def build_baseline_opportunity_evidence(
    candidate: dict[str, Any],
    repository_root: Path,
    manifest_path: Path,
    analysis_path: Path,
    runs_path: Path,
) -> dict[str, Any]:
    """Turn a complete hidden baseline into a validated open-task claim."""
    analysis = json.loads(analysis_path.read_text())
    conclusion = analysis.get("conclusion")
    if conclusion not in {"verified_improvement_room", "verified_unsolved_room"}:
        raise ValueError("baseline analysis has no verified solver improvement room")
    exact = conclusion == "verified_unsolved_room"
    if candidate.get("task_config_path"):
        task_relative = PurePosixPath(candidate["task_config_path"])
        if task_relative.is_absolute() or ".." in task_relative.parts:
            raise ValueError("baseline task config path is unsafe")
        task = yaml.safe_load((repository_root / task_relative).read_text())
        task_exact = (
            task.get("evaluation", {}).get("performance_protocol")
            == "exact_verified_solve"
        )
        if exact != task_exact:
            raise ValueError("baseline conclusion differs from task metric protocol")
    evidence = {
        "schema_version": 1,
        **{
            field: candidate[field]
            for field in (
                "repository",
                "pr_number",
                "base_commit",
                "head_commit",
                "source_patch_sha256",
            )
        },
        "opportunity_kind": "unsolved_exact" if exact else "remaining_verified_gap",
        "qualification_mode": "baseline_qualification",
        "human_reference_classification": "not_evaluated",
        "claim_boundary": (
            "Complete independently checked hidden baseline only; no positive "
            "human PR effect is claimed."
        ),
    }
    if exact:
        evidence["base_verified_solved_runs"] = analysis["base_verified_solved_runs"]
        evidence["base_time_limit_runs"] = analysis["base_time_limit_runs"]
    else:
        evidence["base_mean_normalized_gap"] = analysis["base_mean_normalized_gap"]
    for name, path in (
        ("manifest", manifest_path),
        ("analysis", analysis_path),
        ("runs", runs_path),
    ):
        _record_artifact(evidence, name, path, repository_root)
    validate_open_opportunity_evidence(evidence, candidate, repository_root)
    return evidence


def build_task_integration_evidence(
    candidate: dict[str, Any],
    repository_root: Path,
    agent_identity_path: Path,
    checkout_audit_path: Path,
    run_directory: Path,
    dry_run_exit_code: int,
    dry_run_exit_note: str | None = None,
) -> dict[str, Any]:
    """Bind and validate a real limited Judge run and Git-backed Agent image."""
    if candidate.get("task_track") != "open_opportunity":
        raise ValueError("integration builder requires an open-opportunity task")
    task_path = repository_root / candidate["task_config_path"]
    task = yaml.safe_load(task_path.read_text())
    evidence = {
        "schema_version": 1,
        "repository": candidate["repository"],
        "pr_number": candidate["pr_number"],
        "base_commit": candidate["base_commit"],
        "task_id": candidate["task_id"],
        "judge_image": task["repository"]["judge_image"],
        "dry_run_exit_code": dry_run_exit_code,
        "dry_run_exit_note": dry_run_exit_note,
        "claim_boundary": (
            "Task-specific limited real Judge integration; the overall dry-run "
            "exit code is recorded separately because auxiliary checks may fail "
            "after this task completes. Final Agent performance requires the "
            "full declared instance panel."
        ),
    }
    for name, path in (
        ("task_config", task_path),
        ("agent_identity", agent_identity_path),
        ("checkout_audit", checkout_audit_path),
        ("evaluation", run_directory / "evaluation.json"),
        ("expected_grid", run_directory / "expected-run-grid.json"),
        ("observations", run_directory / "judge-observations.jsonl"),
    ):
        _record_artifact(evidence, name, path, repository_root)
    validate_task_integration_evidence(evidence, candidate, repository_root)
    return evidence


def build_selection_queue(
    rows: list[dict[str, Any]], catalog: dict[str, Any], limit: int
) -> dict[str, Any]:
    """Round-robin title triage; signals never make a task eligible."""
    if limit <= 0:
        raise ValueError("selection queue limit must be positive")
    status_by_repo = {
        entry["repository"]: entry["scope_status"] for entry in catalog["repositories"]
    }
    buckets: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    eligible = 0
    pending_evidence = 0
    for row in rows:
        status = status_by_repo.get(row.get("repository"))
        if status is None:
            raise ValueError(f"PR source absent from catalog: {row.get('repository')}")
        if status != "rejected" and row.get("stage") == "discovered" and row.get(
            "selection_decision"
        ) == "needs_evidence":
            pending_evidence += 1
            continue
        if (
            row.get("stage") != "discovered"
            or row.get("selection_decision") == "reject"
            or status == "rejected"
        ):
            continue
        eligible += 1
        title = row.get("title") or ""
        signals = [
            name for name, pattern in TITLE_SIGNALS.items() if pattern.search(title)
        ]
        title_tier = 0 if len(signals) == 2 else 1 if signals else 2
        if NON_SOLVER_TITLE.search(title):
            title_tier = max(title_tier, 2)
        scope_tier = (
            0 if status == "task_backed" else 1 if status == "approved_source" else 2
        )
        item = {
            "repository": row["repository"],
            "pr_number": row["pr_number"],
            "title": title,
            "url": row.get("url"),
            "scope_status": status,
            "title_signals": signals,
            "title_only_priority": title_tier,
            "selection_decision": row.get("selection_decision", "unreviewed"),
        }
        buckets.setdefault((title_tier, scope_tier, row["repository"]), []).append(item)
    for items in buckets.values():
        items.sort(key=lambda item: (item["title_only_priority"], item["pr_number"]))
    ordered = {key: deque(items) for key, items in buckets.items()}
    queue = []
    for title_tier in range(3):
        active = {
            scope_tier: deque(
                key
                for key in sorted(ordered)
                if key[0] == title_tier and key[1] == scope_tier
            )
            for scope_tier in range(3)
        }
        while any(active.values()) and len(queue) < limit:
            for scope_tier in range(3):
                if not active[scope_tier] or len(queue) == limit:
                    continue
                key = active[scope_tier].popleft()
                queue.append(ordered[key].popleft())
                if ordered[key]:
                    active[scope_tier].append(key)
    return {
        "eligible_discovered_prs": eligible,
        "pending_evidence_prs": pending_evidence,
        "queue_size": len(queue),
        "queue": queue,
        "claim_boundary": (
            "Title signals order manual review only; no queued PR passes selection "
            "or counts as a benchmark task."
        ),
    }


def build_mechanism_review_queue(
    rows: list[dict[str, Any]],
    catalog: dict[str, Any],
    details: list[dict[str, Any]],
    limit: int,
    *,
    audit_fraction: float = 0.1,
) -> dict[str, Any]:
    """Order metadata-backed human reviews without deciding PR eligibility."""
    if limit <= 0 or not 0 <= audit_fraction < 1:
        raise ValueError("triage needs a positive limit and audit fraction in [0, 1)")
    status_by_repo = {
        entry["repository"]: entry["scope_status"] for entry in catalog["repositories"]
    }
    indexed = {candidate_key(row): row for row in rows}
    detail_status_counts = Counter(
        str(detail.get("status") or "unknown") for detail in details
    )
    completed: dict[tuple[str, int], dict[str, Any]] = {}
    manual_large: dict[tuple[str, int], dict[str, Any]] = {}
    for detail in details:
        status = detail.get("status")
        if status not in {"completed", "manual_large_pr"}:
            continue
        key = candidate_key(detail)
        if key not in indexed:
            raise ValueError(f"PR detail is absent from registry: {key}")
        row = indexed[key]
        if any(
            detail.get(field) != row.get(field)
            for field in ("base_commit", "head_commit")
        ):
            raise ValueError(f"PR detail differs from pinned source: {key}")
        target = completed if status == "completed" else manual_large
        if key in target and target[key] != detail:
            raise ValueError(f"conflicting {status} PR details: {key}")
        target[key] = detail

    core: list[dict[str, Any]] = []
    low_signal: list[dict[str, Any]] = []
    manual_diff: list[dict[str, Any]] = []
    eligible = 0
    pending_evidence = 0
    for row in rows:
        status = status_by_repo.get(row.get("repository"))
        if status is None:
            raise ValueError(f"PR source absent from catalog: {row.get('repository')}")
        if status != "rejected" and row.get("stage") == "discovered" and row.get(
            "selection_decision"
        ) == "needs_evidence":
            pending_evidence += 1
            continue
        if (
            row.get("stage") != "discovered"
            or row.get("selection_decision") == "reject"
            or status == "rejected"
        ):
            continue
        eligible += 1
        detail = completed.get(candidate_key(row))
        if detail is None:
            large = manual_large.get(candidate_key(row))
            if large is not None:
                manual_diff.append(
                    {
                        "repository": row["repository"],
                        "pr_number": row["pr_number"],
                        "title": row.get("title"),
                        "url": row.get("url"),
                        "scope_status": status,
                        "changed_file_count": large.get("changed_file_count"),
                        "additions": large.get("additions"),
                        "deletions": large.get("deletions"),
                        "detail_sha256": hashlib.sha256(
                            json.dumps(
                                large, sort_keys=True, separators=(",", ":")
                            ).encode()
                        ).hexdigest(),
                        "review_lane": "manual_full_diff",
                    }
                )
            continue
        files = detail.get("files")
        if not isinstance(files, list):
            raise ValueError(f"completed PR detail lacks files: {candidate_key(row)}")
        source_paths = []
        for file in files:
            raw = file.get("path") if isinstance(file, dict) else None
            if not isinstance(raw, str):
                continue
            path = PurePosixPath(raw)
            if path.suffix.lower() not in SOURCE_SUFFIXES or any(
                part.lower() in NON_ENGINE_PATH_PARTS for part in path.parts[:-1]
            ):
                continue
            source_paths.append(raw)
        phases = sorted(
            phase
            for phase, pattern in PHASE_PATH_HINTS.items()
            if any(pattern.search(path) for path in source_paths)
        )
        title = row.get("title") or ""
        title_signals = sorted(
            name for name, pattern in TITLE_SIGNALS.items() if pattern.search(title)
        )
        caution_flags = ["lineage_review"] if LINEAGE_TITLE.search(title) else []
        body = detail.get("body")
        related_pr_hints = (
            sorted({int(number) for number in RELATED_PR_HINT.findall(body)})
            if isinstance(body, str)
            else []
        )
        band = (
            0
            if phases
            else 1
            if source_paths and title_signals
            else 2
            if source_paths
            else 3
        )
        if caution_flags and band < 2:
            band = 2
        digest = hashlib.sha256(
            json.dumps(detail, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        item = {
            "repository": row["repository"],
            "pr_number": row["pr_number"],
            "title": title,
            "url": row.get("url"),
            "scope_status": status,
            "source_path_count": len(source_paths),
            "source_path_preview": source_paths[:8],
            "solver_phase_path_hints": phases,
            "title_signals": title_signals,
            "caution_flags": caution_flags,
            "related_pr_hints": related_pr_hints,
            "signal_band": band,
            "detail_sha256": digest,
        }
        (core if band < 2 else low_signal).append(item)

    def stable_key(item: dict[str, Any]) -> str:
        return hashlib.sha256(
            f"pitbench-pr-triage-v1:{item['repository']}#{item['pr_number']}".encode()
        ).hexdigest()

    ordered_core = []
    for band in (0, 1):
        groups: dict[tuple[int, str, str], deque[dict[str, Any]]] = {}
        for item in core:
            if item["signal_band"] != band:
                continue
            scope_tier = {
                "task_backed": 0,
                "approved_source": 1,
                "pending_domain_review": 2,
            }[item["scope_status"]]
            phase = (
                item["solver_phase_path_hints"][0]
                if item["solver_phase_path_hints"]
                else "unspecified"
            )
            groups.setdefault((scope_tier, item["repository"], phase), deque()).append(
                item
            )
        for key, group in groups.items():
            groups[key] = deque(sorted(group, key=stable_key))
        active = deque(sorted(groups))
        while active:
            key = active.popleft()
            ordered_core.append(groups[key].popleft())
            if groups[key]:
                active.append(key)

    # Audit low-signal PRs across source projects instead of allowing the
    # largest repository to dominate the first review batch.
    low_groups: dict[tuple[int, str, int], deque[dict[str, Any]]] = {}
    for item in low_signal:
        scope_tier = {
            "task_backed": 0,
            "approved_source": 1,
            "pending_domain_review": 2,
        }[item["scope_status"]]
        key = (scope_tier, item["repository"], item["signal_band"])
        low_groups.setdefault(key, deque()).append(item)
    for key, group in low_groups.items():
        low_groups[key] = deque(sorted(group, key=stable_key))
    ordered_low = []
    active_low = deque(sorted(low_groups))
    while active_low:
        key = active_low.popleft()
        ordered_low.append(low_groups[key].popleft())
        if low_groups[key]:
            active_low.append(key)
    manual_diff.sort(key=lambda item: (item["repository"], item["pr_number"]))
    audit_quota = (
        min(len(ordered_low), math.ceil(limit * audit_fraction)) if limit >= 10 else 0
    )
    chosen_core = ordered_core[: max(0, limit - audit_quota)]
    chosen_audit = ordered_low[:audit_quota]
    selected_keys = {
        (item["repository"], item["pr_number"])
        for item in (*chosen_core, *chosen_audit)
    }
    remaining_core = [
        item
        for item in ordered_core
        if (item["repository"], item["pr_number"]) not in selected_keys
    ]
    remaining_low = ordered_low[audit_quota:]
    capacity = limit - len(chosen_core) - len(chosen_audit)
    fill_core = remaining_core[:capacity]
    capacity -= len(fill_core)
    fill_low = remaining_low[:capacity]
    queue = [
        {**item, "review_lane": lane}
        for lane, items in (
            ("mechanism_hint", [*chosen_core, *fill_core]),
            ("low_signal_audit", chosen_audit),
            ("remaining_low_signal", fill_low),
        )
        for item in items
    ]
    return {
        "eligible_discovered_prs": eligible,
        "pending_evidence_prs": pending_evidence,
        "completed_detail_prs": len(completed),
        "manual_large_detail_prs": len(manual_large),
        "manual_diff_queue_size": len(manual_diff),
        "manual_diff_queue": manual_diff,
        "detail_record_status_counts": dict(sorted(detail_status_counts.items())),
        "eligible_with_completed_details": len(core) + len(low_signal),
        "eligible_without_completed_details": eligible - len(core) - len(low_signal),
        "mechanism_hint_pool": len(core),
        "low_signal_pool": len(low_signal),
        "audit_fraction": audit_fraction,
        "queue_size": len(queue),
        "queue": queue,
        "claim_boundary": (
            "Paths and titles only order human review. Low-signal candidates remain "
            "eligible; oversized PRs remain in a separate full-diff lane; merge "
            "and revert hints request lineage review. No queue entry "
            "has passed solver scope, causal mechanism, "
            "mathematical correctness, or hidden performance gates."
        ),
    }


def normalize_search_issue(
    repository: str, issue: dict[str, Any], matched_query: str
) -> dict[str, Any] | None:
    pull_request = issue.get("pull_request")
    if not isinstance(pull_request, dict) or not pull_request.get("merged_at"):
        return None
    row = {
        "repository": repository,
        "pr_number": issue["number"],
        "title": issue["title"],
        "url": issue["html_url"],
        "created_at": issue.get("created_at"),
        "closed_at": issue.get("closed_at"),
        "merged_at": pull_request["merged_at"],
        "matched_queries": [matched_query],
        "stage": "discovered",
    }
    candidate_key(row)
    return row


def merge_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = candidate_key(row)
        if not row.get("merged_at"):
            raise ValueError(f"candidate is not a confirmed merged PR: {key}")
        previous = merged.get(key)
        if previous is None:
            merged[key] = {
                **row,
                "matched_queries": sorted(set(row.get("matched_queries", []))),
            }
            continue
        if row.get("url") != previous.get("url"):
            raise ValueError(f"conflicting PR URLs for {key}")
        for field in (
            "base_commit",
            "head_commit",
            "merge_commit",
            "created_at",
            "closed_at",
        ):
            if not previous.get(field) and row.get(field):
                previous[field] = row[field]
            elif (
                field in {"base_commit", "head_commit", "merge_commit"}
                and previous.get(field)
                and row.get(field)
                and previous[field] != row[field]
            ):
                raise ValueError(f"conflicting {field} for {key}")
        for field, value in row.items():
            if field not in previous and field not in {"matched_queries", "stage"}:
                previous[field] = value
        if STAGES.index(row.get("stage", "discovered")) > STAGES.index(
            previous.get("stage", "discovered")
        ):
            previous["stage"] = row["stage"]
        previous["matched_queries"] = sorted(
            set(previous["matched_queries"]) | set(row.get("matched_queries", []))
        )
    return [merged[key] for key in sorted(merged)]


def validate_evidence_stage(
    row: dict[str, Any], repository_root: Path | None = None
) -> None:
    key = candidate_key(row)
    stage = row.get("stage")
    task_track = row.get("task_track", "expert_reference")
    if task_track not in TASK_TRACKS:
        raise ValueError(f"unknown task track for {key}: {task_track!r}")
    if stage not in STAGES:
        raise ValueError(f"unknown evidence stage for {key}: {stage!r}")
    if not row.get("merged_at"):
        raise ValueError(f"missing merge evidence for {key}")
    if row.get("selection_decision") and not row.get("selection_review_path"):
        raise ValueError(f"{key} selection decision has no review file")
    review = None
    if row.get("selection_review_path") and repository_root is not None:
        review_path = repository_root / row["selection_review_path"]
        if hashlib.sha256(review_path.read_bytes()).hexdigest() != row.get(
            "selection_review_sha256"
        ):
            raise ValueError(f"{key} selection review hash mismatch")
        review = json.loads(review_path.read_text())
        decision = validate_selection_review(review, row)
        if decision != row.get("selection_decision"):
            raise ValueError(f"{key} selection decision differs from review")
        if (
            STAGES.index(stage) >= STAGES.index("intent_reviewed")
            and decision != "advance"
        ):
            raise ValueError(f"{key} advanced without selection approval")
    required_stages = [
        current
        for current in STAGES[1 : STAGES.index(stage) + 1]
        if not (task_track == "open_opportunity" and current == "performance_confirmed")
    ]
    for current in required_stages:
        for field in EVIDENCE_BY_STAGE[current]:
            if not row.get(field):
                raise ValueError(f"{key} at {stage} lacks {field}")
    if repository_root is None:
        return
    for current in required_stages:
        for field in EVIDENCE_BY_STAGE[current]:
            if not field.endswith("_path"):
                continue
            path = repository_root / row[field]
            if not path.is_file():
                raise ValueError(f"{key} evidence file missing: {path}")
    if STAGES.index(stage) >= STAGES.index("source_verified"):
        if review is not None and set(review["changed_source_paths"]) != set(
            row["changed_source_files"]
        ):
            raise ValueError(f"{key} source paths differ from selection review")
        patch = repository_root / row["source_patch_path"]
        digest = hashlib.sha256(patch.read_bytes()).hexdigest()
        if digest != row["source_patch_sha256"]:
            raise ValueError(f"{key} source patch hash mismatch")
    if review is not None and STAGES.index(stage) >= STAGES.index("build_verified"):
        evidence = json.loads(
            (repository_root / row["build_evidence_path"]).read_text()
        )
        validate_pinned_build_evidence(evidence, row, repository_root)
    if review is not None and STAGES.index(stage) >= STAGES.index("workload_verified"):
        evidence = json.loads(
            (repository_root / row["workload_evidence_path"]).read_text()
        )
        validate_workload_exposure_evidence(evidence, row, review, repository_root)
    if review is not None and STAGES.index(stage) >= STAGES.index(
        "correctness_verified"
    ):
        evidence = json.loads(
            (repository_root / row["correctness_evidence_path"]).read_text()
        )
        obligations = {
            SEMANTIC_OBLIGATIONS[risk]
            for risk in review["correctness_failure_modes"]
            if risk in SEMANTIC_OBLIGATIONS
        }
        validate_independent_correctness_evidence(
            evidence,
            row,
            repository_root,
            required_semantic_obligations=obligations,
        )
    if row.get("performance_screen_evidence_path"):
        screen_path = repository_root / row["performance_screen_evidence_path"]
        if hashlib.sha256(screen_path.read_bytes()).hexdigest() != row.get(
            "performance_screen_evidence_sha256"
        ):
            raise ValueError(f"{key} hidden screen evidence hash mismatch")
        screen = json.loads(screen_path.read_text())
        classification = screen.get("pitbench_performance", {}).get(
            "confirmed_classification"
        )
        if (
            classification != row.get("performance_screen_classification")
            or screen.get("observed_unique_pairs") != screen.get("expected_pairs")
            or screen.get("missing_pairs")
            or screen.get("failed_pairs")
        ):
            raise ValueError(f"{key} hidden screen classification or grid differs")
    if task_track == "open_opportunity" and stage == "task_ready":
        integration_path = row.get("integration_evidence_path")
        if not integration_path:
            raise ValueError(f"{key} open task lacks real Judge integration evidence")
        integration = json.loads((repository_root / integration_path).read_text())
        validate_task_integration_evidence(integration, row, repository_root)
        opportunity_path = row.get("opportunity_evidence_path")
        if not opportunity_path:
            raise ValueError(f"{key} open task lacks opportunity evidence")
        evidence = json.loads((repository_root / opportunity_path).read_text())
        validate_open_opportunity_evidence(evidence, row, repository_root)
    if task_track == "expert_reference" and STAGES.index(stage) >= STAGES.index(
        "performance_confirmed"
    ):
        evidence = json.loads(
            (repository_root / row["performance_evidence_path"]).read_text()
        )
        confirmed = evidence.get("pitbench_performance", {}).get(
            "confirmed_classification"
        )
        if confirmed != "improved":
            raise ValueError(f"{key} has no confirmed human performance gain")
    if stage == "task_ready":
        task = yaml.safe_load((repository_root / row["task_config_path"]).read_text())
        if (
            task.get("task_id") != row["task_id"]
            or task.get("release", {}).get("base_commit") != row["base_commit"]
            or task.get("release", {}).get("repository") != row["repository"]
        ):
            raise ValueError(f"{key} task identity differs from PR source")


def _ready_task_input_fingerprint(task: dict[str, Any]) -> str:
    """Identify equal Agent improvement requests despite different task IDs/images."""
    evaluation = task.get("evaluation", {})
    identity = {
        "instruction": task.get("instruction"),
        "optimization_scope": task.get("optimization_scope"),
        "task_type": task.get("task_type"),
        "problem_family": task.get("problem_family"),
        "problem_variant": task.get("problem_variant"),
        "formulation_id": task.get("formulation_id"),
        "editable_paths": task.get("repository", {}).get("editable_paths"),
        "objective_sense": task.get("oracle", {}).get("objective_sense"),
        "performance_protocol": evaluation.get("performance_protocol"),
        "budgets_sec": evaluation.get("budgets_sec"),
        "primary_budget_sec": evaluation.get("primary_budget_sec"),
        "solver_seeds": evaluation.get("solver_seeds"),
        "threads": evaluation.get("threads"),
        "development_seeds": (evaluation.get("seed_robustness") or {}).get(
            "development_seeds"
        ),
        "instance_sets": [
            {
                "name": item.get("name"),
                "kind": item.get("kind"),
                "size": item.get("size"),
                "config": item.get("instance_set_config_sha256")
                or item.get("instance_set_config"),
            }
            for item in task.get("instance_sets", [])
        ],
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def summarize_registry(
    rows: list[dict[str, Any]], target: int, repository_root: Path | None = None
) -> dict[str, Any]:
    keys = [candidate_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate PR in evidence registry")
    for row in rows:
        validate_evidence_stage(row, repository_root)
    counts = Counter(row["stage"] for row in rows)
    selection_counts = Counter(
        row.get("selection_decision", "unreviewed")
        for row in rows
        if row["stage"] == "discovered"
    )
    reviewed_counts = Counter(
        row["selection_decision"] for row in rows if row.get("selection_decision")
    )
    ready_by_repository = Counter(
        row["repository"] for row in rows if row["stage"] == "task_ready"
    )
    ready_by_track = Counter(
        row.get("task_track", "expert_reference")
        for row in rows
        if row["stage"] == "task_ready"
    )
    ready_mechanisms = Counter()
    ready_phases = Counter()
    ready_overlap_groups = Counter()
    seen_lineages: set[tuple[str, str]] = set()
    seen_snapshots: dict[
        tuple[str, str], list[tuple[int, str, dict[str, Any]]]
    ] = {}
    ordered_ready = sorted(
        (row for row in rows if row["stage"] == "task_ready"),
        key=lambda row: (
            row["repository"],
            row["base_commit"],
            row.get("merged_at") or "",
            row["pr_number"],
        ),
    )
    for row in ordered_ready:
        review_path = row.get("selection_review_path")
        review = (
            json.loads((repository_root / review_path).read_text())
            if repository_root is not None and review_path
            else {}
        )
        ready_mechanisms[review.get("mechanism", "legacy_unlabeled")] += 1
        ready_phases[review.get("solver_phase", "legacy_unlabeled")] += 1
        group = review.get("overlap_group", "legacy_unlabeled")
        ready_overlap_groups[group] += 1
        if group != "legacy_unlabeled":
            lineage = (row["repository"], group)
            if lineage in seen_lineages:
                raise ValueError(f"duplicate task lineage in ready registry: {lineage}")
            seen_lineages.add(lineage)
        if repository_root is not None:
            task = yaml.safe_load(
                (repository_root / row["task_config_path"]).read_text()
            )
            fingerprint = _ready_task_input_fingerprint(task)
            snapshot = (row["repository"], row["base_commit"])
            previous = seen_snapshots.setdefault(snapshot, [])
            if any(
                prior_fingerprint == fingerprint
                for _, prior_fingerprint, _ in previous
            ):
                raise ValueError(f"duplicate Agent task input for snapshot: {snapshot}")
            for prior_number, _, prior_review in previous:
                current_links_prior = (
                    bool(review.get("same_snapshot_distinction"))
                    and prior_number in review.get("related_prs", [])
                )
                prior_links_current = (
                    bool(prior_review.get("same_snapshot_distinction"))
                    and row["pr_number"] in prior_review.get("related_prs", [])
                )
                if not (current_links_prior or prior_links_current):
                    raise ValueError(
                        "same-snapshot tasks need a documented distinct goal: "
                        f"{snapshot}"
                    )
            previous.append((row["pr_number"], fingerprint, review))
    hidden_screen_counts = Counter(
        row["performance_screen_classification"]
        for row in rows
        if row.get("performance_screen_classification")
    )
    ready = counts["task_ready"]
    return {
        "distinct_merged_prs": len(rows),
        "stage_counts": {stage: counts[stage] for stage in STAGES},
        "discovered_selection_counts": {
            decision: selection_counts[decision]
            for decision in (*SELECTION_DECISIONS, "unreviewed")
        },
        "structured_selection_review_counts": {
            decision: reviewed_counts[decision] for decision in SELECTION_DECISIONS
        },
        "task_ready_by_repository": dict(sorted(ready_by_repository.items())),
        "task_ready_by_track": {track: ready_by_track[track] for track in TASK_TRACKS},
        "task_ready_by_mechanism": dict(sorted(ready_mechanisms.items())),
        "task_ready_by_solver_phase": dict(sorted(ready_phases.items())),
        "task_ready_by_overlap_group": dict(sorted(ready_overlap_groups.items())),
        "hidden_reference_screen_counts": dict(sorted(hidden_screen_counts.items())),
        "target_validated_pr_tasks": target,
        "remaining_task_ready_prs": max(target - ready, 0),
        "claim_boundary": (
            "Only task_ready rows count toward the target; discovered and "
            "screened PRs are source candidates, not benchmark tasks."
        ),
    }


def materialize_git_backed_checkout(
    *,
    source_repository: Path,
    base_commit: str,
    destination: Path,
    submodule_sources: dict[str, Path],
) -> dict[str, Any]:
    """Create a minimal pinned checkout with local, gitlink-exact submodules."""
    if destination.exists():
        raise ValueError(f"checkout destination already exists: {destination}")
    for path, source in submodule_sources.items():
        relative = PurePosixPath(path)
        if relative.is_absolute() or ".." in relative.parts or not source.is_dir():
            raise ValueError(f"invalid local submodule source: {path}")
    source_repository = source_repository.resolve()
    destination.mkdir(parents=True)
    subprocess.run(["git", "init", "--quiet", str(destination)], check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", source_repository.as_uri()],
        cwd=destination,
        check=True,
    )
    file_only = {**os.environ, "GIT_ALLOW_PROTOCOL": "file"}
    subprocess.run(
        ["git", "fetch", "--depth=1", "origin", base_commit],
        cwd=destination,
        env=file_only,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", "--quiet", "--detach", "FETCH_HEAD"],
        cwd=destination,
        check=True,
    )
    actual_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=destination, text=True
    ).strip()
    if actual_commit != base_commit:
        raise ValueError("materialized checkout has the wrong base commit")
    used_submodule_sources = set()

    def initialize_submodules(checkout: Path, prefix: str) -> None:
        if not (checkout / ".gitmodules").is_file():
            return
        listing = subprocess.run(
            [
                "git",
                "config",
                "--file",
                ".gitmodules",
                "--get-regexp",
                r"^submodule\..*\.path$",
            ],
            cwd=checkout,
            capture_output=True,
            text=True,
            check=False,
        )
        if listing.returncode not in {0, 1}:
            raise ValueError(f"cannot read pinned submodules in {checkout}")
        configured = {}
        for line in listing.stdout.splitlines():
            key, path = line.split(" ", 1)
            name = key.removeprefix("submodule.").removesuffix(".path")
            full_path = f"{prefix}/{path}" if prefix else path
            if full_path not in submodule_sources:
                raise ValueError(f"missing local submodule source: {full_path}")
            configured[path] = (name, full_path)
            used_submodule_sources.add(full_path)
            subprocess.run(
                [
                    "git",
                    "config",
                    f"submodule.{name}.url",
                    submodule_sources[full_path].resolve().as_uri(),
                ],
                cwd=checkout,
                check=True,
            )
        if configured:
            subprocess.run(
                [
                    "git",
                    "-c",
                    "protocol.file.allow=always",
                    "submodule",
                    "update",
                    "--init",
                    "--depth=1",
                ],
                cwd=checkout,
                env=file_only,
                check=True,
                capture_output=True,
            )
            for path, (name, full_path) in configured.items():
                original_url = subprocess.check_output(
                    [
                        "git",
                        "config",
                        "--file",
                        ".gitmodules",
                        "--get",
                        f"submodule.{name}.url",
                    ],
                    cwd=checkout,
                    text=True,
                ).strip()
                subprocess.run(
                    ["git", "config", f"submodule.{name}.url", original_url],
                    cwd=checkout,
                    check=True,
                )
                initialize_submodules(checkout / path, full_path)

    initialize_submodules(destination, "")
    if used_submodule_sources != set(submodule_sources):
        raise ValueError(
            "extra local submodule sources are not in pinned gitlinks: "
            f"{sorted(set(submodule_sources) - used_submodule_sources)}"
        )
    status = subprocess.check_output(
        ["git", "submodule", "status", "--recursive"],
        cwd=destination,
        text=True,
    ).splitlines()
    if any(not line.startswith(" ") for line in status):
        raise ValueError("materialized checkout has missing or divergent submodules")
    return {
        "base_commit": actual_commit,
        "base_tree": subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=destination, text=True
        ).strip(),
        "submodule_status": status,
    }


def verify_source_patch(
    *,
    repository_checkout: Path,
    record: dict[str, Any],
    editable_roots: list[str],
    patch_path: Path,
    audit_path: Path,
    intent_note: str,
) -> dict[str, Any]:
    """Extract a source-only PR diff and compare its result with the PR head."""
    key = candidate_key(record)
    base, head = record.get("base_commit"), record.get("head_commit")
    if not base or not head or not editable_roots or not intent_note.strip():
        raise ValueError(
            f"{key} needs pinned commits, editable roots and intent review"
        )
    for root in editable_roots:
        path = PurePosixPath(root)
        if root in {"", "."} or path.is_absolute() or ".." in path.parts:
            raise ValueError(f"invalid editable root for {key}: {root!r}")
    for commit in (base, head):
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=repository_checkout,
            check=True,
            capture_output=True,
        )
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", base, head],
        cwd=repository_checkout,
        text=True,
    ).splitlines()
    source_files = [
        path
        for path in changed
        if any(path == root or path.startswith(root + "/") for root in editable_roots)
    ]
    if not source_files:
        raise ValueError(f"{key} has no changes inside the editable roots")
    patch = subprocess.check_output(
        ["git", "diff", "--binary", base, head, "--", *editable_roots],
        cwd=repository_checkout,
    )
    if not patch:
        raise ValueError(f"{key} produced an empty source patch")
    archive = subprocess.check_output(
        ["git", "archive", "--format=tar", base], cwd=repository_checkout
    )
    with tempfile.TemporaryDirectory(prefix="pitbench-pr-source-") as directory:
        workspace = Path(directory)
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as handle:
            handle.extractall(workspace, filter="data")
        candidate = workspace / "candidate.patch"
        candidate.write_bytes(patch)
        subprocess.run(
            ["git", "apply", "--check", str(candidate)],
            cwd=workspace,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "apply", str(candidate)],
            cwd=workspace,
            check=True,
            capture_output=True,
        )
        for path in source_files:
            expected = subprocess.run(
                ["git", "show", f"{head}:{path}"],
                cwd=repository_checkout,
                check=False,
                capture_output=True,
            )
            actual = workspace / path
            if expected.returncode:
                if actual.exists():
                    raise ValueError(f"{key} did not delete {path}")
            elif not actual.is_file() or actual.read_bytes() != expected.stdout:
                raise ValueError(f"{key} source bytes differ from PR head: {path}")
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_bytes(patch)
    audit = {
        "repository": key[0],
        "pr_number": key[1],
        "pr_title": record.get("title"),
        "base_commit": base,
        "head_commit": head,
        "manual_intent_note": intent_note.strip(),
        "changed_source_files": source_files,
        "excluded_non_source_files": [
            path for path in changed if path not in source_files
        ],
        "source_patch_path": str(patch_path),
        "source_patch_sha256": hashlib.sha256(patch).hexdigest(),
        "source_verification": "standalone_patch_applies_and_source_bytes_match_pr_head",
        "build_status": "untested",
        "workload_exposure": "untested",
        "performance_status": "untested",
        "claim_boundary": "Source and intent review only; this is not a runnable benchmark task.",
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    return audit
