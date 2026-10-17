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
from pitbench.schema.observation import CodeState, RunObservation, RunStatus
from pitbench.schema.task import PitBenchTask

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_SEED = 20261017
BOOTSTRAP_DRAWS = 10_000


def _source_path(value: str) -> Path:
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


def analyze(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
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
    for round_index in range(manifest["rounds"]):
        effects: dict[str, float] = {}
        order_effects: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        outcomes = Counter()
        objective_mismatch = 0
        for instance_id in instance_ids:
            deltas = []
            for seed in seeds:
                pair = pairs[(round_index, instance_id, seed)]
                old = pair["old"]
                new = pair["new"]
                delta = old["normalized_gap"] - new["normalized_gap"]
                deltas.append(delta)
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
    report["rounds"] = rounds

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
            "claim_boundary": "Reduced release-control grid, not an Agent patch score",
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
        "strong_positive_control_for_metric_sensitivity"
        if confirmed
        else "release_effect_not_confirmed_under_frozen_rule"
    )
    report["claim_boundary"] = (
        "A release control tests solver-metric sensitivity. It does not show "
        "that an ordinary Agent benchmark would rank Agent patches differently."
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    result = analyze(args.manifest.resolve(), output_dir)
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
