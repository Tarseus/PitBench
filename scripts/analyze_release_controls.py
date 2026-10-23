"""Analyze predeclared paired release controls from retained raw pair records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from pitbench.metrics.performance_report import (
    BOOTSTRAP_SEED as NATIVE_BOOTSTRAP_SEED,
)
from pitbench.metrics.performance_report import (
    PerformanceClassification,
    _cluster_mean_ci,
    adjudicate_performance_confirmation,
    compute_task_performance_report,
)
from pitbench.problem_families.base import ProblemFamilyPlugin, ProblemFamilyRegistry
from pitbench.repositories.pr_curation import verified_exact_timeout_incumbent
from pitbench.schema.observation import CodeState, RunObservation, RunStatus
from pitbench.schema.task import PerformanceProtocol, PitBenchTask

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_SEED = 20261017
BOOTSTRAP_DRAWS = 10_000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relocated_path(value: str) -> Path:
    """Resolve retained paths after moving a complete experiment checkout."""
    path = Path(value)
    if path.is_file():
        return path
    if "PitBench/" in value:
        return ROOT / value.split("PitBench/", 1)[1]
    return _source_path(value)


def _comparison_estimate(values: dict[str, list[float]]) -> dict:
    """Reuse the native paired instance bootstrap; seeds are not independent n."""
    means = {key: statistics.fmean(items) for key, items in values.items()}
    interval = _cluster_mean_ci(
        {(key, key): items for key, items in values.items()},
        seed=NATIVE_BOOTSTRAP_SEED + 2,
    )
    if interval is None:
        return {"classification": "incomplete"}
    return {
        "mean": statistics.fmean(means.values()),
        "ci95": interval.model_dump(mode="json"),
        "classification": (
            "improved"
            if interval.lower > 0
            else "regressed"
            if interval.upper < 0
            else "inconclusive"
        ),
        "instance_count": len(means),
        "positive_instances": sum(value > 0 for value in means.values()),
        "negative_instances": sum(value < 0 for value in means.values()),
        "worst": min(means.values()),
        "best": max(means.values()),
        "per_instance": means,
    }


def _compare_panel(spec: dict, catalog: dict, verification_file) -> dict:
    manifest_path = _source_path(spec["manifest"])
    directory = _source_path(spec["run_directory"])
    manifest = json.loads(manifest_path.read_text())
    identity_path = directory / "identity.json"
    identity = json.loads(identity_path.read_text())
    ledger_path = directory / "pairs.jsonl"
    rows = [json.loads(line) for line in ledger_path.read_text().splitlines() if line]
    issues = []
    if _sha256(manifest_path) != identity["manifest_sha256"]:
        issues.append("manifest_hash_mismatch")
    oracle_path = _relocated_path(manifest["oracle_path"])
    oracle = yaml.safe_load(oracle_path.read_text()) if oracle_path.is_file() else None
    oracle_targets = (
        {item["id"]: item["bks"] for item in oracle["instances"]} if oracle else {}
    )
    if oracle and _sha256(oracle_path) != identity["oracle_sha256"]:
        issues.append("oracle_hash_mismatch")
    ids = (
        sorted(item["id"] for item in oracle["instances"])
        if oracle
        else sorted({row["instance_id"] for row in rows})
    )
    if len(ids) != identity["instance_count"]:
        issues.append("instance_count_mismatch")
    seeds = identity["selected_seeds"]
    if len(seeds) != manifest["seed_count"] or len(seeds) != len(set(seeds)):
        issues.append("seed_selection_mismatch")
    seed_path = _relocated_path(manifest["seed_file"])
    if seed_path.is_file() and _sha256(seed_path) != identity["seed_file_sha256"]:
        issues.append("seed_file_hash_mismatch")
    expected = {
        (round_id, instance, seed)
        for round_id in range(manifest["rounds"])
        for instance in ids
        for seed in seeds
    }
    if len(expected) != identity["expected_pairs"]:
        issues.append("expected_pair_count_mismatch")
    pairs = {}
    duplicates = []
    failed = []
    artifact_count = 0
    verified_count = 0
    verifier = ProblemFamilyRegistry.load(spec["problem_family"])
    for row in rows:
        key = (row["round"], row["instance_id"], row["solver_seed"])
        if key in pairs:
            duplicates.append(key)
        pairs[key] = row
        if row["budget_sec"] != manifest["budget_sec"]:
            issues.append(f"budget_mismatch:{key}")
        for label in ("old", "new"):
            run = row[label]
            audit = {"panel_id": spec["panel_id"], "key": key, "state": label}
            try:
                instance = catalog[row["instance_id"]]
                instance_path = _source_path(instance["path"])
                solution_path = _relocated_path(run["solution_path"])
                output_path = _relocated_path(run["output_path"])
                if _sha256(instance_path) != instance["sha256"]:
                    raise ValueError("instance content differs from comparison catalog")
                checked = verifier.verify(instance_path, solution_path)
                output = json.loads(output_path.read_text())
                artifact_count += 1
                audit.update(
                    instance_path=str(instance_path),
                    instance_sha256=instance["sha256"],
                    solution_path=str(solution_path),
                    solution_sha256=_sha256(solution_path),
                    output_path=str(output_path),
                    output_sha256=_sha256(output_path),
                    feasible=checked.feasible,
                    objective=checked.objective,
                    detail=checked.detail,
                )
                if not checked.feasible or checked.objective is None:
                    raise ValueError("independent verification failed")
                if (
                    run["status"] != "completed"
                    or run.get("reported_valid") is not True
                ):
                    raise ValueError(f"retained run status: {run['status']}")
                if output.get("valid") is not True:
                    raise ValueError("raw solver output is invalid")
                for retained, raw in (
                    ("solver_wall_time_sec", "wall_time_sec"),
                    ("solver_cpu_time_sec", "cpu_time_sec"),
                    ("solver_iterations", "iterations"),
                ):
                    if run.get(retained) != output.get(raw):
                        raise ValueError(
                            f"retained measurement differs from raw output: {raw}"
                        )
                for objective in (
                    run["reported_objective"],
                    run["verified_objective"],
                    output["objective"],
                ):
                    if not math.isclose(
                        checked.objective, objective, rel_tol=0, abs_tol=1e-6
                    ):
                        raise ValueError(
                            "reported or retained objective differs from recomputation"
                        )
                # Keep the original denominator when published BKS versions differ.
                audit["retained_reference_target"] = row["target"]
                audit["catalog_reference_target"] = instance.get("bks")
                audit["reference_version_differs"] = (
                    instance.get("bks") is not None and instance["bks"] != row["target"]
                )
                if oracle is not None:
                    declared = oracle_targets[row["instance_id"]]
                    if row["target"] != declared:
                        raise ValueError("retained target differs from original oracle")
                gap = ProblemFamilyPlugin.normalized_gap(
                    checked.objective, row["target"], objective_sense="minimize"
                )
                if not math.isclose(
                    gap, run["normalized_gap"], rel_tol=0, abs_tol=1e-12
                ):
                    raise ValueError(
                        "retained normalized gap differs from recomputation"
                    )
                for field in ("commit", "image"):
                    if run[field] != manifest[label][field]:
                        raise ValueError(f"source identity differs: {field}")
                audit["accepted"] = True
                verified_count += 1
            except (OSError, ValueError, KeyError, TypeError) as error:
                audit.update(accepted=False, error=str(error))
                failed.append(audit)
            verification_file.write(json.dumps(audit, allow_nan=False) + "\n")
    missing = sorted(expected - pairs.keys())
    unexpected = sorted(pairs.keys() - expected)
    complete = not (issues or duplicates or missing or unexpected or failed)
    result = {
        "panel_id": spec["panel_id"],
        "candidate_id": spec["candidate_id"],
        "role": spec["role"],
        "problem_family": spec["problem_family"],
        "manifest_sha256": _sha256(manifest_path),
        "ledger_sha256": _sha256(ledger_path),
        "identity_sha256": _sha256(identity_path),
        "source_identity": identity,
        "oracle_file_available": oracle is not None,
        "expected_ids_basis": "retained_oracle"
        if oracle
        else "retained_IDs_checked_against_identity_cardinality",
        "original_oracle_hash_revalidated": oracle is not None
        and not any("oracle_hash" in x for x in issues),
        "old_commit": manifest["old"]["commit"],
        "new_commit": manifest["new"]["commit"],
        "budget_sec": manifest["budget_sec"],
        "seed_count": len(seeds),
        "expected_pairs": len(expected),
        "pair_attempts": len(rows),
        "artifact_runs_read": artifact_count,
        "independently_verified_solutions": verified_count,
        "missing_pairs": missing,
        "unexpected_pairs": unexpected,
        "duplicate_pairs": duplicates,
        "issues": issues,
        "failed_runs": failed,
        "complete": complete,
        "rounds": {},
    }
    if not complete:
        result["confirmed_quality_classification"] = "incomplete"
        return result
    for round_id in range(manifest["rounds"]):
        quality, wall, throughput = (
            defaultdict(list),
            defaultdict(list),
            defaultdict(list),
        )
        base_gap, candidate_gap = [], []
        for instance_id in ids:
            for seed in seeds:
                pair = pairs[(round_id, instance_id, seed)]
                old, new = pair["old"], pair["new"]
                quality[instance_id].append(
                    old["normalized_gap"] - new["normalized_gap"]
                )
                base_gap.append(old["normalized_gap"])
                candidate_gap.append(new["normalized_gap"])
                old_time, new_time = (
                    old.get("solver_wall_time_sec"),
                    new.get("solver_wall_time_sec"),
                )
                if all(
                    type(x) in (int, float) and math.isfinite(x) and x > 0
                    for x in (old_time, new_time)
                ):
                    wall[instance_id].append(old_time - new_time)
                numbers = [
                    old.get("solver_iterations"),
                    new.get("solver_iterations"),
                    old.get("solver_cpu_time_sec"),
                    new.get("solver_cpu_time_sec"),
                ]
                if all(
                    type(x) in (int, float) and math.isfinite(x) and x > 0
                    for x in numbers
                ):
                    old_iterations, new_iterations, old_cpu, new_cpu = numbers
                    throughput[instance_id].append(
                        (new_iterations / new_cpu) / (old_iterations / old_cpu) - 1
                    )
        result["rounds"][str(round_id)] = {
            "quality": _comparison_estimate(quality),
            "base_mean_gap": statistics.fmean(base_gap),
            "candidate_mean_gap": statistics.fmean(candidate_gap),
            "wall_reduction_sec": _comparison_estimate(wall)
            if sum(map(len, wall.values())) == len(ids) * len(seeds)
            else {"classification": "unavailable"},
            "iteration_rate_relative_change": _comparison_estimate(throughput)
            if sum(map(len, throughput.values())) == len(ids) * len(seeds)
            else {"classification": "unavailable"},
        }
        result["rounds"][str(round_id)]["subgroups"] = {
            group["group_id"]: _comparison_estimate(
                {instance: quality[instance] for instance in group["instance_ids"]}
            )
            for group in spec.get("subgroups", [])
        }
    reversed_order = manifest["rounds"] == 2 and all(
        pairs[(0, instance_id, seed)]["order"]
        == list(reversed(pairs[(1, instance_id, seed)]["order"]))
        for instance_id in ids
        for seed in seeds
    )
    classes = [r["quality"]["classification"] for r in result["rounds"].values()]
    result["opposite_order_verified"] = reversed_order
    if manifest["rounds"] == 2:
        result["same_state_round_drift"] = {
            label: _comparison_estimate(
                {
                    instance_id: [
                        pairs[(0, instance_id, seed)][label]["normalized_gap"]
                        - pairs[(1, instance_id, seed)][label]["normalized_gap"]
                        for seed in seeds
                    ]
                    for instance_id in ids
                }
            )
            for label in ("old", "new")
        }
    result["confirmed_quality_classification"] = (
        classes[0]
        if reversed_order
        and len(set(classes)) == 1
        and classes[0] in {"improved", "regressed"}
        else "inconclusive"
    )
    return result


def compare_evaluation_protocols(plan_path: Path, output_dir: Path) -> dict:
    """Audit proxy and solution-quality conclusions on explicitly listed panels."""
    plan = json.loads(plan_path.read_text())
    catalog_path = _source_path(plan["instance_catalog"])
    catalog = json.loads(catalog_path.read_text())["instances"]
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with (output_dir / "verification.jsonl").open("w") as verification_file:
        for spec in plan["panels"]:
            result = _compare_panel(spec, catalog, verification_file)
            results.append(result)
            print(
                json.dumps(
                    {
                        "panel": result["panel_id"],
                        "complete": result["complete"],
                        "verified": result["independently_verified_solutions"],
                        "classification": result["confirmed_quality_classification"],
                    }
                ),
                flush=True,
            )
    table = []
    for result in results:
        for round_id, measures in result["rounds"].items():
            quality = measures["quality"]
            table.append(
                {
                    "candidate": result["candidate_id"],
                    "panel": result["panel_id"],
                    "role": result["role"],
                    "round": round_id,
                    "instances": quality["instance_count"],
                    "seeds": result["seed_count"],
                    "gap_reduction_pp": quality["mean"] * 100,
                    "gap_ci_lower_pp": quality["ci95"]["lower"] * 100,
                    "gap_ci_upper_pp": quality["ci95"]["upper"] * 100,
                    "quality": quality["classification"],
                    "wall_time": measures["wall_reduction_sec"]["classification"],
                    "iteration_rate": measures["iteration_rate_relative_change"][
                        "classification"
                    ],
                    "iteration_rate_change_percent": 100
                    * measures["iteration_rate_relative_change"]["mean"]
                    if "mean" in measures["iteration_rate_relative_change"]
                    else None,
                    "confirmed_quality": result["confirmed_quality_classification"],
                    "worse_instances": quality["negative_instances"],
                    "worst_gap_reduction_pp": quality["worst"] * 100,
                }
            )
    summary = {
        "study_kind": plan["study_kind"],
        "plan_sha256": _sha256(plan_path),
        "catalog_sha256": _sha256(catalog_path),
        "analysis_source_sha256": _sha256(Path(__file__)),
        "metric_implementation_sha256": _sha256(
            ROOT / "pitbench/metrics/performance_report.py"
        ),
        "verification_implementation_sha256": _sha256(
            ROOT / "pitbench/problem_families/public_verification.py"
        ),
        "qualification_scope": "Exploratory control-panel comparisons only; no task admission or benchmark qualification is changed.",
        "panels": len(results),
        "candidate_count": len({r["candidate_id"] for r in results}),
        "pair_attempts": sum(r["pair_attempts"] for r in results),
        "independently_verified_solutions": sum(
            r["independently_verified_solutions"] for r in results
        ),
        "incomplete_panels": [r["panel_id"] for r in results if not r["complete"]],
        "claim_boundaries": plan["claim_boundaries"],
        "results": results,
    }
    initial = [r for r in results if r["role"] == "public_initial" and r["complete"]]
    summary["initial_public_cohort_description"] = {
        "panels": len(initial),
        "positive_first_round_point_estimates": sum(
            r["rounds"]["0"]["quality"]["mean"] > 0 for r in initial
        ),
        "positive_first_round_native_intervals": sum(
            r["rounds"]["0"]["quality"]["classification"] == "improved" for r in initial
        ),
        "opposite_order_confirmed_positive": sum(
            r["confirmed_quality_classification"] == "improved" for r in initial
        ),
        "interpretation": "Convenience cohort description; unconfirmed does not mean false or regressed. This is not a population false-positive estimate.",
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    if table:
        with (output_dir / "comparison.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        headers = list(table[0])
        rows_html = "".join(
            "<tr>"
            + "".join(f"<td>{html.escape(str(row[key]))}</td>" for key in headers)
            + "</tr>"
            for row in table
        )
        (output_dir / "comparison.html").write_text(
            '<!doctype html><meta charset="utf-8"><title>Evaluation protocol comparison</title>'
            "<style>body{font:14px system-ui;padding:24px}table{border-collapse:collapse}th,td{border:1px solid #ccc;padding:7px}th{position:sticky;top:0;background:#eee}</style>"
            "<h1>Evaluation protocol comparison</h1><p>"
            + html.escape(" ".join(plan["claim_boundaries"]))
            + "</p><table><thead><tr>"
            + "".join(f"<th>{key}</th>" for key in headers)
            + "</tr></thead><tbody>"
            + rows_html
            + "</tbody></table>"
        )
    return summary


def _source_path(value: str) -> Path:
    if value.startswith("private://"):
        return ROOT / "private" / value.removeprefix("private://")
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _interval(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    ordered = sorted(
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(BOOTSTRAP_DRAWS)
    )
    return {
        "lower": ordered[int(0.025 * BOOTSTRAP_DRAWS)],
        "upper": ordered[int(0.975 * BOOTSTRAP_DRAWS)],
    }


def _sign_test(effects: list[float]) -> float | None:
    positives = sum(value > 1e-12 for value in effects)
    negatives = sum(value < -1e-12 for value in effects)
    total = positives + negatives
    if not total:
        return None
    return sum(math.comb(total, index) for index in range(positives, total + 1)) / (
        2**total
    )


def _distribution(effects: dict[str, float]) -> dict[str, Any]:
    values = list(effects.values())
    ordered = sorted(values)
    largest = max(effects, key=lambda instance: abs(effects[instance]))
    trimmed = ordered[3:-3] if len(ordered) >= 10 else ordered
    return {
        "instance_count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "worst": min(values),
        "best": max(values),
        "trimmed_mean": statistics.fmean(trimmed),
        "better_instances": sum(value > 1e-12 for value in values),
        "equal_instances": sum(abs(value) <= 1e-12 for value in values),
        "worse_instances": sum(value < -1e-12 for value in values),
        "ci95": _interval(values),
        "one_sided_sign_test_p": _sign_test(values),
        "largest_absolute_instance": largest,
        "mean_without_largest_absolute_instance": statistics.fmean(
            value for instance, value in effects.items() if instance != largest
        ),
        "per_instance": dict(sorted(effects.items())),
    }


def _read_pairs(path: Path) -> tuple[dict[tuple[int, str, int], dict], int]:
    latest: dict[tuple[int, str, int], dict] = {}
    attempts = 0
    for line in path.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        key = (row["round"], row["instance_id"], row["solver_seed"])
        attempts += 1
        if key not in latest or all(
            row[label]["status"] == "completed" for label in ("old", "new")
        ):
            latest[key] = row
    return latest, attempts


def analyze_baseline_qualification(
    manifest_path: Path, output_dir: Path
) -> dict[str, Any]:
    """Summarize a one-state hidden baseline without claiming a human PR gain."""
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("run_mode") != "baseline_qualification":
        raise ValueError("baseline analysis requires a baseline qualification manifest")
    identity = json.loads((output_dir / "identity.json").read_text())
    task = PitBenchTask.from_yaml(_source_path(manifest["task_config"]))
    exact = (
        task.evaluation.performance_protocol == PerformanceProtocol.EXACT_VERIFIED_SOLVE
    )
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if identity.get("manifest_sha256") != manifest_sha:
        raise ValueError("baseline manifest differs from retained run identity")
    oracle = yaml.safe_load(_source_path(manifest["oracle_path"]).read_text())
    expected = {
        (item["id"], seed)
        for item in oracle["instances"]
        for seed in identity["selected_seeds"]
    }
    latest = {}
    attempts = 0
    for line in (output_dir / "baseline_runs.jsonl").read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        key = (row["instance_id"], row["solver_seed"])
        attempts += 1
        if (
            key not in latest
            or row["base"]["status"] == "completed"
            or (
                exact
                and row["base"]["status"] == "no_solution"
                and row["base"].get("reported", {}).get("solver_termination")
                == "time_limit"
            )
        ):
            latest[key] = row
    missing = sorted(expected - latest.keys())
    if exact:
        objective_sense = task.oracle.objective_sense or "minimize"
        known_optimum = task.oracle.kind == "known_optimum"
        ordinary_unsolved = {
            key
            for key, row in latest.items()
            if (
                row["base"]["status"] == "no_solution"
                and row["base"].get("reported", {}).get("solver_termination")
                == "time_limit"
            )
            or verified_exact_timeout_incumbent(
                row["base"],
                row.get("target"),
                objective_sense=objective_sense,
                known_optimum=known_optimum,
            )
        }
        failures = [
            {"instance_id": key[0], "seed": key[1], "status": row["base"]["status"]}
            for key, row in latest.items()
            if row["base"]["status"] != "completed" and key not in ordinary_unsolved
        ]
        invalid = []
        solved = 0
        for key, row in latest.items():
            base = row["base"]
            if base["status"] != "completed":
                continue
            if key in ordinary_unsolved:
                continue
            reported, verified, target = (
                base.get("reported_objective"),
                base.get("verified_objective"),
                row.get("target"),
            )
            if (
                base.get("reported_valid") is not True
                or base.get("solver_termination") != "optimal"
                or any(
                    type(value) not in {int, float}
                    for value in (reported, verified, target)
                )
                or not math.isclose(reported, verified, rel_tol=0, abs_tol=1e-6)
                or not math.isclose(verified, target, rel_tol=0, abs_tol=1e-6)
            ):
                invalid.append({"instance_id": key[0], "seed": key[1]})
            else:
                solved += 1
        complete = not missing and not failures and not invalid
        unsolved = len(ordinary_unsolved)
        return {
            "manifest_sha256": manifest_sha,
            "source_identity": identity,
            "expected_runs": len(expected),
            "retained_run_attempts": attempts,
            "observed_unique_runs": len(latest),
            "missing_runs": missing,
            "failed_runs": failures,
            "invalid_runs": invalid,
            "base_verified_solved_runs": solved,
            "base_time_limit_runs": unsolved,
            "base_time_limit_with_incumbent_runs": sum(
                verified_exact_timeout_incumbent(
                    row["base"],
                    row.get("target"),
                    objective_sense=objective_sense,
                    known_optimum=known_optimum,
                )
                for row in latest.values()
            ),
            "base_verified_solved_coverage": (
                solved / len(expected) if complete and expected else None
            ),
            "conclusion": (
                "verified_unsolved_room"
                if complete and solved > 0 and unsolved > 0
                else "no_mixed_exact_outcomes"
                if complete
                else "incomplete"
            ),
            "claim_boundary": (
                "This qualifies an exact-solver open challenge with both verified "
                "optimal and ordinary time-limit outcomes. It does not establish "
                "a human PR gain or Agent success."
            ),
        }
    failures = [
        {"instance_id": key[0], "seed": key[1], "status": row["base"]["status"]}
        for key, row in latest.items()
        if row["base"]["status"] != "completed"
    ]
    gaps = []
    invalid = []
    for key, row in latest.items():
        base = row["base"]
        if base["status"] != "completed":
            continue
        gap = base.get("normalized_gap")
        reported = base.get("reported_objective")
        verified = base.get("verified_objective")
        if (
            base.get("reported_valid") is not True
            or type(gap) not in {int, float}
            or not math.isfinite(gap)
            or gap < 0
            or type(reported) not in {int, float}
            or type(verified) not in {int, float}
            or not math.isclose(reported, verified, rel_tol=0, abs_tol=1e-6)
        ):
            invalid.append({"instance_id": key[0], "seed": key[1]})
        else:
            gaps.append(gap)
    complete = (
        not missing and not failures and not invalid and len(gaps) == len(expected)
    )
    mean_gap = statistics.fmean(gaps) if complete else None
    return {
        "manifest_sha256": manifest_sha,
        "source_identity": identity,
        "expected_runs": len(expected),
        "retained_run_attempts": attempts,
        "observed_unique_runs": len(latest),
        "missing_runs": missing,
        "failed_runs": failures,
        "invalid_runs": invalid,
        "base_mean_normalized_gap": mean_gap,
        "positive_gap_runs": sum(gap > 0 for gap in gaps) if complete else None,
        "conclusion": (
            "verified_improvement_room"
            if mean_gap is not None and mean_gap > 0
            else "no_verified_gap"
            if complete
            else "incomplete"
        ),
        "claim_boundary": (
            "This qualifies an open solver optimization challenge on a pinned "
            "baseline. It does not establish a human PR gain or Agent success."
        ),
    }


def analyze(
    manifest_path: Path,
    output_dir: Path,
    *,
    reference_kind: str = "solver_release",
    exclude_first_seed: bool = False,
) -> dict[str, Any]:
    if reference_kind not in {"solver_release", "human_pr"}:
        raise ValueError("unknown reference kind")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("rounds") != 2:
        raise ValueError("release-control analysis requires two opposite-order rounds")
    identity = json.loads((output_dir / "identity.json").read_text())
    if (
        identity["manifest_sha256"]
        != hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    ):
        raise ValueError("manifest does not match retained run identity")
    oracle = yaml.safe_load(_source_path(manifest["oracle_path"]).read_text())
    instance_ids = [item["id"] for item in oracle["instances"]]
    seeds = identity["selected_seeds"]
    if exclude_first_seed and len(seeds) < 2:
        raise ValueError("excluding the first seed requires at least two seeds")
    expected = {
        (round_index, instance_id, seed)
        for round_index in range(manifest["rounds"])
        for instance_id in instance_ids
        for seed in seeds
    }
    pairs, attempt_count = _read_pairs(output_dir / "pairs.jsonl")
    missing = sorted(expected - pairs.keys())
    failures = [
        {"round": key[0], "instance_id": key[1], "seed": key[2], "row": pair}
        for key, pair in pairs.items()
        if any(pair[label]["status"] != "completed" for label in ("old", "new"))
    ]
    report: dict[str, Any] = {
        "reference_kind": reference_kind,
        "manifest_sha256": identity["manifest_sha256"],
        "source_identity": identity,
        "expected_pairs": len(expected),
        "retained_pair_attempts": attempt_count,
        "observed_unique_pairs": len(pairs),
        "missing_pairs": missing,
        "failed_pairs": failures,
        "method": {
            "primary_metric": "old minus new independently verified normalized gap",
            "unit": "independent instance, after paired-seed averaging",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "rounds_separate": True,
        },
    }
    if missing or failures:
        report["conclusion"] = "incomplete"
        return report

    rounds: dict[str, Any] = {}
    time_only: dict[str, Any] = {}
    for round_index in range(manifest["rounds"]):
        effects: dict[str, float] = {}
        wall_effects: dict[str, float] = {}
        base_wall_by_instance: dict[str, float] = {}
        time_available = True
        order_effects: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        outcomes = Counter()
        objective_mismatch = 0
        for instance_id in instance_ids:
            deltas = []
            wall_deltas = []
            base_walls = []
            for seed in seeds:
                pair = pairs[(round_index, instance_id, seed)]
                old = pair["old"]
                new = pair["new"]
                delta = old["normalized_gap"] - new["normalized_gap"]
                deltas.append(delta)
                old_wall = old.get("solver_wall_time_sec")
                new_wall = new.get("solver_wall_time_sec")
                if (
                    type(old_wall) not in {int, float}
                    or type(new_wall) not in {int, float}
                    or not math.isfinite(old_wall)
                    or not math.isfinite(new_wall)
                ):
                    if reference_kind == "human_pr":
                        raise ValueError("paired solver wall time is unavailable")
                    time_available = False
                else:
                    base_walls.append(old_wall)
                    wall_deltas.append(old_wall - new_wall)
                order_effects[pair["order"][0]][instance_id].append(delta)
                outcomes[
                    "better"
                    if delta > 1e-12
                    else "worse"
                    if delta < -1e-12
                    else "equal"
                ] += 1
                for run in (old, new):
                    if (
                        abs(run["reported_objective"] - run["verified_objective"])
                        > 1e-6
                    ):
                        objective_mismatch += 1
            effects[instance_id] = statistics.fmean(deltas)
            if len(wall_deltas) == len(seeds):
                wall_effects[instance_id] = statistics.fmean(wall_deltas)
                base_wall_by_instance[instance_id] = statistics.fmean(base_walls)
        strata = {
            order: _distribution(
                {
                    instance: statistics.fmean(values)
                    for instance, values in by_instance.items()
                }
            )
            for order, by_instance in order_effects.items()
        }
        rounds[str(round_index)] = {
            "effect": _distribution(effects),
            "paired_run_outcomes": dict(outcomes),
            "reported_verified_objective_mismatches": objective_mismatch,
            "execution_order_strata": strata,
        }
        if time_available and len(wall_effects) == len(instance_ids):
            wall = _distribution(wall_effects)
            time_only[str(round_index)] = {
                "paired_base_minus_candidate_wall_sec": wall,
                "relative_wall_reduction_percent": (
                    100
                    * wall["mean"]
                    / statistics.fmean(base_wall_by_instance.values())
                ),
                "classification": (
                    "faster"
                    if wall["ci95"]["lower"] > 0
                    else "slower"
                    if wall["ci95"]["upper"] < 0
                    else "inconclusive"
                ),
            }
        else:
            time_only[str(round_index)] = {
                "classification": "unavailable",
                "reason": "paired solver wall time is unavailable",
            }
    report["rounds"] = rounds
    report["time_only_by_round"] = time_only
    if exclude_first_seed:
        subset = seeds[1:]
        report["unused_public_seed_subset_by_round"] = {
            str(round_index): _distribution(
                {
                    instance_id: statistics.fmean(
                        pairs[(round_index, instance_id, seed)]["old"]["normalized_gap"]
                        - pairs[(round_index, instance_id, seed)]["new"][
                            "normalized_gap"
                        ]
                        for seed in subset
                    )
                    for instance_id in instance_ids
                }
            )
            for round_index in range(manifest["rounds"])
        }

    null_controls: dict[str, Any] = {}
    for label in ("old", "new"):
        effects = {
            instance_id: statistics.fmean(
                pairs[(0, instance_id, seed)][label]["normalized_gap"]
                - pairs[(1, instance_id, seed)][label]["normalized_gap"]
                for seed in seeds
            )
            for instance_id in instance_ids
        }
        null_controls[label] = _distribution(effects)
    report["same_version_round_drift"] = null_controls

    if "task_config" in manifest:
        task = PitBenchTask.from_yaml(_source_path(manifest["task_config"]))
        formal_reports = []
        for round_index in range(manifest["rounds"]):
            observations = [
                RunObservation(
                    task_id=task.task_id,
                    code_state=(CodeState.BASE if label == "old" else CodeState.AGENT),
                    instance_set="judge_id",
                    instance_set_kind="judge_id",
                    instance_id=instance_id,
                    solver_seed=seed,
                    budget_sec=manifest["budget_sec"],
                    status=RunStatus.COMPLETED,
                    valid=True,
                    objective=pairs[(round_index, instance_id, seed)][label][
                        "verified_objective"
                    ],
                    reported_objective=pairs[(round_index, instance_id, seed)][label][
                        "reported_objective"
                    ],
                    optimal_or_bks=pairs[(round_index, instance_id, seed)]["target"],
                    normalized_gap=pairs[(round_index, instance_id, seed)][label][
                        "normalized_gap"
                    ],
                    wall_time_sec=pairs[(round_index, instance_id, seed)][label].get(
                        "solver_wall_time_sec"
                    ),
                    cpu_time_sec=pairs[(round_index, instance_id, seed)][label].get(
                        "solver_cpu_time_sec"
                    ),
                    iterations=pairs[(round_index, instance_id, seed)][label].get(
                        "solver_iterations"
                    ),
                )
                for instance_id in instance_ids
                for seed in seeds
                for label in ("old", "new")
            ]
            formal = compute_task_performance_report(observations, task=task)
            if formal is None:
                raise ValueError("formal Performance report is unavailable")
            formal_reports.append(formal)
        if formal_reports[0].classification in {
            PerformanceClassification.IMPROVED,
            PerformanceClassification.REGRESSED,
        }:
            confirmed_formal = adjudicate_performance_confirmation(
                formal_reports[0],
                formal_reports[1],
                repeat_observation_count=2 * len(instance_ids) * len(seeds),
                repeat_valid_observation_count=2 * len(instance_ids) * len(seeds),
            )
            final_classification = confirmed_formal.classification.value
        else:
            final_classification = formal_reports[0].classification.value
        report["pitbench_performance"] = {
            "round_classifications": [
                item.classification.value for item in formal_reports
            ],
            "confirmed_classification": final_classification,
            "round_mean_gap_reductions": [
                item.primary.paired.mean_gap_reduction for item in formal_reports
            ],
            "round_gap_ci95": [
                item.primary.paired.mean_gap_reduction_ci95.model_dump()
                if item.primary.paired.mean_gap_reduction_ci95 is not None
                else None
                for item in formal_reports
            ],
            "claim_boundary": (
                "Reduced human-PR grid, not a score on the current task"
                if reference_kind == "human_pr"
                else "Reduced release-control grid, not an Agent patch score"
            ),
        }

    means = [rounds[str(index)]["effect"]["mean"] for index in range(2)]
    null_mean = max(abs(item["mean"]) for item in null_controls.values())
    confirmed = (
        all(
            rounds[str(index)]["effect"]["mean"] >= 0.001
            and rounds[str(index)]["effect"]["ci95"]["lower"] > 0
            and rounds[str(index)]["effect"]["mean_without_largest_absolute_instance"]
            > 0
            and rounds[str(index)]["reported_verified_objective_mismatches"] == 0
            for index in range(2)
        )
        and min(means) > null_mean
    )
    report["version_effect_to_max_null_mean_ratio"] = (
        min(means) / null_mean if null_mean else None
    )
    report["conclusion"] = (
        "strong_positive_human_reference"
        if confirmed and reference_kind == "human_pr"
        else "human_reference_not_confirmed_under_frozen_rule"
        if reference_kind == "human_pr"
        else "strong_positive_control_for_metric_sensitivity"
        if confirmed
        else "release_effect_not_confirmed_under_frozen_rule"
    )
    report["claim_boundary"] = (
        "This one human PR is a reference patch on a historical source commit; "
        "it does not measure Agent ability or establish general solver effects."
        if reference_kind == "human_pr"
        else "A release control tests solver-metric sensitivity. It does not show "
        "that an ordinary Agent benchmark would rank Agent patches differently."
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--comparison-plan", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument(
        "--reference-kind",
        choices=("solver_release", "human_pr"),
        default="solver_release",
    )
    parser.add_argument(
        "--exclude-first-seed",
        action="store_true",
        help="report the public development seeds not used by a one-seed pilot",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if args.comparison_plan is not None:
        if args.manifest is not None:
            parser.error("choose --comparison-plan or --manifest")
        compare_evaluation_protocols(args.comparison_plan, output_dir)
        return
    if args.manifest is None:
        parser.error("--manifest is required without --comparison-plan")
    if args.baseline_only:
        result = analyze_baseline_qualification(args.manifest.resolve(), output_dir)
        output = output_dir / "baseline_analysis.json"
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "conclusion": result["conclusion"],
                    "expected_runs": result["expected_runs"],
                    "missing_runs": len(result["missing_runs"]),
                    "failed_runs": len(result["failed_runs"]),
                    "analysis_path": str(output),
                }
            )
        )
        return
    result = analyze(
        args.manifest.resolve(),
        output_dir,
        reference_kind=args.reference_kind,
        exclude_first_seed=args.exclude_first_seed,
    )
    output = output_dir / "analysis.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "conclusion": result["conclusion"],
                "expected_pairs": result["expected_pairs"],
                "missing_pairs": len(result["missing_pairs"]),
                "failed_pairs": len(result["failed_pairs"]),
                "analysis_path": str(output),
            }
        )
    )


if __name__ == "__main__":
    main()
