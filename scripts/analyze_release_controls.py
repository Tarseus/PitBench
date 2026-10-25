"""Analyze predeclared paired release controls from retained raw pair records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from pitbench.metrics.performance_report import (
    PerformanceClassification,
    adjudicate_performance_confirmation,
    compute_task_performance_report,
)
from pitbench.repositories.pr_curation import verified_exact_timeout_incumbent
from pitbench.schema.observation import CodeState, RunObservation, RunStatus
from pitbench.schema.task import PerformanceProtocol, PitBenchTask

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_SEED = 20261017
BOOTSTRAP_DRAWS = 10_000


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
    parser.add_argument("--manifest", type=Path, required=True)
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
