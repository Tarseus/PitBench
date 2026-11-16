from __future__ import annotations

import hashlib
import json
import random
import statistics
from pathlib import Path

import yaml

from pitbench.evaluator.collection import ReleaseControlCollection as release_runner
from pitbench.metrics.performance_report import (
    ReleaseControlAnalysis as release_analysis,
)
from tests.support import ROOT

TASK_CONFIG = ROOT / "configs/tasks/pyvrp_v0_14_0.yaml"
EXACT_TASK_CONFIG = ROOT / "configs/tasks/highs_multidimensional_knapsack_v1_15_1.yaml"


def test_shared_control_bootstrap_preserves_the_declared_seed_and_draws():
    values = [0.1, -0.3, 0.6]
    rng = random.Random(20261017)
    expected = sorted(
        statistics.fmean(rng.choice(values) for _ in values) for _ in range(10000)
    )
    assert release_analysis._interval(values) == {
        "lower": expected[250],
        "upper": expected[9750],
    }


def test_release_control_reverses_pair_order(monkeypatch, tmp_path: Path) -> None:
    def fake_solver(**arguments):
        return {"status": "completed", "label": arguments["label"]}

    monkeypatch.setattr(release_runner, "_run_solver", fake_solver)
    inputs = {
        "manifest": {"budget_sec": 10.0},
        "task": object(),
        "instance": {"id": "one", "path": tmp_path / "one.json", "target": 1},
        "seed": 7,
        "pair_index": 0,
        "attempt": 1,
        "output_dir": tmp_path,
        "runner_template": [],
    }

    forward = release_runner._run_pair(round_index=0, **inputs)
    reverse = release_runner._run_pair(round_index=1, **inputs)

    assert forward["order"] == ["old", "new"]
    assert reverse["order"] == ["new", "old"]


def test_release_analysis_requires_complete_repeated_instance_gain(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(release_analysis, "ROOT", tmp_path)
    oracle = tmp_path / "oracle.yaml"
    oracle.write_text(
        yaml.safe_dump({"instances": [{"id": f"i{index}"} for index in range(3)]})
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "oracle_path": "oracle.yaml",
                "task_config": str(TASK_CONFIG),
                "rounds": 2,
                "budget_sec": 10.0,
            }
        )
    )
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    (output_dir / "identity.json").write_text(
        json.dumps(
            {
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "selected_seeds": [7, 8],
            }
        )
    )
    rows = [
        {
            "round": round_index,
            "instance_id": f"i{index}",
            "solver_seed": seed,
            "target": 100.0,
            "order": ["old", "new"] if round_index == 0 else ["new", "old"],
            "old": {
                "status": "completed",
                "normalized_gap": 0.05,
                "reported_objective": 105,
                "verified_objective": 105,
            },
            "new": {
                "status": "completed",
                "normalized_gap": 0.03,
                "reported_objective": 103,
                "verified_objective": 103,
            },
        }
        for round_index in range(2)
        for index in range(3)
        for seed in (7, 8)
    ]
    pair_file = output_dir / "pairs.jsonl"
    pair_file.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    complete = release_analysis.analyze(manifest, output_dir)

    assert complete["expected_pairs"] == 12
    assert complete["conclusion"] == "strong_positive_control_for_metric_sensitivity"
    assert complete["rounds"]["0"]["effect"]["better_instances"] == 3
    assert complete["same_version_round_drift"]["old"]["mean"] == 0
    assert complete["pitbench_performance"]["confirmed_classification"] == "improved"

    pair_file.write_text("\n".join(json.dumps(row) for row in rows[:-1]) + "\n")
    incomplete = release_analysis.analyze(manifest, output_dir)
    assert incomplete["conclusion"] == "incomplete"
    assert len(incomplete["missing_pairs"]) == 1


def test_baseline_qualification_keeps_reference_gain_out_of_claim(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(release_analysis, "ROOT", tmp_path)
    oracle = tmp_path / "oracle.yaml"
    oracle.write_text(yaml.safe_dump({"instances": [{"id": "a"}, {"id": "b"}]}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "run_mode": "baseline_qualification",
                "oracle_path": "oracle.yaml",
                "task_config": str(TASK_CONFIG),
            }
        )
    )
    output = tmp_path / "results"
    output.mkdir()
    (output / "identity.json").write_text(
        json.dumps(
            {
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "selected_seeds": [7],
            }
        )
    )
    rows = [
        {
            "instance_id": instance,
            "solver_seed": 7,
            "base": {
                "status": "completed",
                "reported_valid": True,
                "reported_objective": 100,
                "verified_objective": 100,
                "normalized_gap": gap,
            },
        }
        for instance, gap in (("a", 0.1), ("b", 0.0))
    ]
    ledger = output / "baseline_runs.jsonl"
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = release_analysis.analyze_baseline_qualification(manifest, output)
    assert report["conclusion"] == "verified_improvement_room"
    assert report["base_mean_normalized_gap"] == 0.05
    assert "human" not in report

    ledger.write_text(json.dumps(rows[0]) + "\n")
    incomplete = release_analysis.analyze_baseline_qualification(manifest, output)
    assert incomplete["conclusion"] == "incomplete"
    assert incomplete["missing_runs"] == [("b", 7)]


def test_exact_baseline_retains_ordinary_time_limits(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(release_analysis, "ROOT", tmp_path)
    (tmp_path / "oracle.yaml").write_text(
        yaml.safe_dump({"instances": [{"id": "a"}, {"id": "b"}, {"id": "c"}]})
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "run_mode": "baseline_qualification",
                "oracle_path": "oracle.yaml",
                "task_config": str(EXACT_TASK_CONFIG),
            }
        )
    )
    output = tmp_path / "results"
    output.mkdir()
    (output / "identity.json").write_text(
        json.dumps(
            {
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "selected_seeds": [7],
            }
        )
    )
    rows = [
        {
            "instance_id": "a",
            "solver_seed": 7,
            "target": -5,
            "base": {
                "status": "completed",
                "reported_valid": True,
                "solver_termination": "optimal",
                "reported_objective": -5,
                "verified_objective": -5,
            },
        },
        {
            "instance_id": "b",
            "solver_seed": 7,
            "target": -6,
            "base": {
                "status": "no_solution",
                "reported": {"solver_termination": "time_limit"},
            },
        },
        {
            "instance_id": "c",
            "solver_seed": 7,
            "target": -7,
            "base": {
                "status": "completed",
                "reported_valid": True,
                "solver_termination": "time_limit",
                "reported_objective": -6,
                "verified_objective": -6,
            },
        },
    ]
    (output / "baseline_runs.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    report = release_analysis.analyze_baseline_qualification(manifest, output)
    assert report["conclusion"] == "verified_unsolved_room"
    assert report["base_verified_solved_runs"] == 1
    assert report["base_time_limit_runs"] == 2
    assert report["base_time_limit_with_incumbent_runs"] == 1
