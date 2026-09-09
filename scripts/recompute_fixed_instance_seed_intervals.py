"""Recompute seed-only intervals from an existing real-solver validation batch."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

import numpy as np

BOOTSTRAP_SEED = 20260824
BOOTSTRAP_RESAMPLES = 5000


def seed_draws() -> np.ndarray:
    generator = random.Random(BOOTSTRAP_SEED)
    return np.array(
        [[math.floor(30 * generator.random()) for _ in range(30)]
         for _ in range(BOOTSTRAP_RESAMPLES)],
        dtype=np.int32,
    )


def bootstrap_means(gaps: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Return paired Base/Agent mean IQRs; all instance weights stay equal.

    gaps has shape (2, instances, 30). Counts describe shared bootstrap seed
    columns. Weighted order statistics avoid sorting 5000 copies of each row.
    Type 7 Q25 uses ranks 7/8 and Q75 uses ranks 21/22 (zero based).
    """
    means = np.zeros((2, len(counts)))
    for state in range(2):
        for row in gaps[state]:
            order = np.argsort(row, kind="stable")
            ordered = row[order]
            cumulative = np.cumsum(counts[:, order], axis=1)
            lower = ordered[np.argmax(cumulative > 7, axis=1)] * 0.75
            lower += ordered[np.argmax(cumulative > 8, axis=1)] * 0.25
            upper = ordered[np.argmax(cumulative > 21, axis=1)] * 0.25
            upper += ordered[np.argmax(cumulative > 22, axis=1)] * 0.75
            means[state] += upper - lower
    return means / gaps.shape[1]


def scalar_iqr(values: list[float]) -> float:
    ordered = sorted(values)
    def quantile(probability: float) -> float:
        index = (len(ordered) - 1) * probability
        left = math.floor(index)
        right = min(left + 1, len(ordered) - 1)
        return ordered[left] + (index - left) * (ordered[right] - ordered[left])
    return quantile(0.75) - quantile(0.25)


def verify_bootstrap(gaps: np.ndarray, draws: np.ndarray, counts: np.ndarray) -> None:
    actual = bootstrap_means(gaps, counts[:20])
    expected = np.array([
        [statistics.fmean(scalar_iqr([float(row[i]) for i in draw])
                          for row in gaps[state]) for draw in draws[:20]]
        for state in range(2)
    ])
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    identical = np.stack([gaps[0], gaps[0]])
    paired = bootstrap_means(identical, counts[:20])
    np.testing.assert_array_equal(paired[0] - paired[1], np.zeros(20))


def analyze(batch: Path, task_id: str, counts: np.ndarray, draws: np.ndarray) -> None:
    source = batch / task_id
    original = json.loads((source / "validation_summary.json").read_text())
    seeds = json.loads((source / "validation_seeds.json").read_text())
    all_seeds = seeds["reference_seeds"] + seeds["test_seeds"]
    assert len(all_seeds) == len(set(all_seeds)) == 1000
    assert len(seeds["test_seed_lists"]) == 1000
    seed_positions = {seed: index for index, seed in enumerate(all_seeds)}
    observations = [json.loads(line) for line in
                    (source / "observations.checkpoint.jsonl").open() if line.strip()]
    instances = sorted({item["instance_id"] for item in observations})
    assert len(instances) == original["instance_count"] == 10
    instance_positions = {instance: index for index, instance in enumerate(instances)}
    budgets = original["budgets_sec"]
    budget_positions = {budget: index for index, budget in enumerate(budgets)}
    grid = np.full((len(budgets), 2, len(instances), len(all_seeds)), np.nan)
    for item in observations:
        assert item["task_id"] == task_id and item["valid"]
        assert item["instance_set"] == original["instance_set"]
        key = (budget_positions[item["budget_sec"]],
               {"base": 0, "agent": 1}[item["code_state"]],
               instance_positions[item["instance_id"]], seed_positions[item["solver_seed"]])
        assert np.isnan(grid[key]), "duplicate observation"
        grid[key] = item["normalized_gap"]
    assert np.isfinite(grid).all() and len(observations) == grid.size == 60000
    del observations
    reference_quantiles = np.quantile(grid[..., :700], [0.25, 0.75], axis=-1, method="linear")
    references = (reference_quantiles[1] - reference_quantiles[0]).mean(axis=-1)
    references = np.concatenate([references, (references[:, 1] - references[:, 0])[:, None]], axis=1)
    estimates = np.zeros((1000, len(budgets), 3))
    intervals = np.zeros((1000, len(budgets), 3, 2))
    for list_index, seed_list in enumerate(seeds["test_seed_lists"]):
        assert len(seed_list) == len(set(seed_list)) == 30
        assert set(seed_list) <= set(seeds["test_seeds"])
        selected = np.take(grid, [seed_positions[seed] for seed in seed_list], axis=-1)
        for budget_index in range(len(budgets)):
            gaps = selected[budget_index]
            if list_index == 0:
                verify_bootstrap(gaps, draws, counts)
            quartiles = np.quantile(gaps, [0.25, 0.75], axis=-1, method="linear")
            means = (quartiles[1] - quartiles[0]).mean(axis=-1)
            estimates[list_index, budget_index] = [means[0], means[1], means[1] - means[0]]
            samples = bootstrap_means(gaps, counts)
            samples = np.vstack([samples, samples[1] - samples[0]])
            intervals[list_index, budget_index] = np.quantile(
                samples, [0.005, 0.995], axis=1, method="linear"
            ).T
        if (list_index + 1) % 25 == 0:
            print(f"{task_id}: {list_index + 1}/1000 seed lists", flush=True)
    summary = {key: value for key, value in original.items() if key != "by_budget"}
    summary.update({
        "method": "fixed instances; paired shared seed-column percentile bootstrap",
        "confidence_level": 0.99,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "rng_protocol": "random.Random; floor(30 * random()); seed draws only; reset per list and budget",
        "instance_ids": instances,
        "instance_weights": "equal, fixed in every replicate",
        "source_directory": str(source),
        "verification": "vectorized bootstrap checked against scalar Type 7 IQR; unchanged point estimates and references checked against original summaries",
        "by_budget": {},
    })
    for budget_index, budget in enumerate(budgets):
        budget_key = f"{budget:g}"
        result = {"budget_sec": budget}
        for state_index, name in enumerate(["base", "agent", "change"]):
            reference = float(references[budget_index, state_index])
            mean = float(estimates[:, budget_index, state_index].mean())
            lower, upper = intervals[:, budget_index, state_index].T
            old = original["by_budget"][budget_key][name]
            np.testing.assert_allclose([reference, mean],
                                       [old["reference_value"], old["mean_estimate"]],
                                       rtol=1e-10, atol=1e-14)
            result[name] = {
                "reference_value": reference, "test_list_count": 1000,
                "estimate_count": 1000, "interval_count": 1000,
                "mean_estimate": mean, "bias": mean - reference,
                "empirical_coverage": float(((lower <= reference) & (reference <= upper)).mean()),
                "mean_interval_width": float((upper - lower).mean()),
                "original_crossed_coverage": old["empirical_coverage"],
                "original_crossed_mean_interval_width": old["mean_interval_width"],
            }
        summary["by_budget"][budget_key] = result
    destination = source / "fixed_instance_intervals"
    destination.mkdir(exist_ok=True)
    details = [
        {"test_list_index": index, "by_budget": {
            f"{budget:g}": {name: {
                "estimate": float(estimates[index, budget_index, state_index]),
                "lower": float(intervals[index, budget_index, state_index, 0]),
                "upper": float(intervals[index, budget_index, state_index, 1]),
            } for state_index, name in enumerate(["base", "agent", "change"])}
            for budget_index, budget in enumerate(budgets)
        }} for index in range(1000)
    ]
    (destination / "intervals.json").write_text(json.dumps(details, indent=2) + "\n")
    (destination / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("task_ids", nargs="+")
    args = parser.parse_args()
    draws = seed_draws()
    counts = np.zeros((BOOTSTRAP_RESAMPLES, 30), dtype=np.int32)
    np.add.at(counts, (np.arange(BOOTSTRAP_RESAMPLES)[:, None], draws), 1)
    assert (counts.sum(axis=1) == 30).all()
    for task_id in args.task_ids:
        analyze(args.batch, task_id, counts, draws)


if __name__ == "__main__":
    main()
