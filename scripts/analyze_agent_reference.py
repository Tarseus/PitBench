"""Compare an Agent patch with a human reference on an identical hidden seed subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import yaml

from scripts.analyze_release_controls import _distribution


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(
    human_path: Path,
    human_manifest_path: Path,
    agent_path: Path,
    agent_task_path: Path,
    *,
    budget_sec: float,
    instance_set: str,
) -> dict:
    human = json.loads(human_path.read_text())
    human_manifest = json.loads(human_manifest_path.read_text())
    agent_task = yaml.safe_load(agent_task_path.read_text())
    if human["manifest_sha256"] != _sha256(human_manifest_path):
        raise ValueError("human analysis and manifest identity differ")
    if human_manifest["old"]["commit"] != agent_task["release"]["base_commit"]:
        raise ValueError("human and Agent base commits differ")
    if human_manifest["budget_sec"] != budget_sec:
        raise ValueError("human and Agent budget selections differ")
    if human_manifest["oracle_path"] != agent_task["oracle"]["source"]:
        raise ValueError("human and Agent oracle sources differ")
    seeds = human["source_identity"]["selected_seeds"]
    instances = set(human["rounds"]["0"]["effect"]["per_instance"])
    expected = {(instance, seed) for instance in instances for seed in seeds}
    observations = {}
    duplicates = []
    for line in agent_path.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        if row["task_id"] != agent_task["task_id"]:
            raise ValueError("Agent observation belongs to another task")
        if (
            row["instance_set"] != instance_set
            or row["budget_sec"] != budget_sec
            or row["instance_id"] not in instances
            or row["solver_seed"] not in seeds
        ):
            continue
        key = (row["instance_id"], row["solver_seed"], row["code_state"])
        if key in observations:
            duplicates.append(key)
        observations[key] = row
    missing = [
        (instance, seed, state)
        for instance, seed in sorted(expected)
        for state in ("base", "agent")
        if (instance, seed, state) not in observations
    ]
    failed = [
        {"key": key, "status": row["status"], "valid": row["valid"]}
        for key, row in sorted(observations.items())
        if row["status"] != "completed"
        or not row["valid"]
        or row["normalized_gap"] is None
    ]
    report = {
        "human_analysis_sha256": _sha256(human_path),
        "human_manifest_sha256": _sha256(human_manifest_path),
        "agent_observations_sha256": _sha256(agent_path),
        "agent_task_sha256": _sha256(agent_task_path),
        "base_commit": agent_task["release"]["base_commit"],
        "budget_sec": budget_sec,
        "instance_set": instance_set,
        "instance_count": len(instances),
        "selected_seeds": seeds,
        "expected_pairs": len(expected),
        "observed_pairs": sum(
            all((instance, seed, state) in observations for state in ("base", "agent"))
            for instance, seed in expected
        ),
        "duplicate_observation_keys": duplicates,
        "missing_observation_keys": missing,
        "failed_observations": failed,
        "claim_boundary": "Exploratory matched-instance and matched-seed comparison, not an official full-grid score or an Agent success-rate estimate.",
    }
    if missing or failed or duplicates:
        report["conclusion"] = "incomplete"
        return report
    effects = {
        instance: statistics.fmean(
            observations[(instance, seed, "base")]["normalized_gap"]
            - observations[(instance, seed, "agent")]["normalized_gap"]
            for seed in seeds
        )
        for instance in sorted(instances)
    }
    report["agent_effect"] = _distribution(effects)
    report["human_effect_by_round"] = {
        round_id: human["rounds"][round_id]["effect"]
        for round_id in ("0", "1")
    }
    report["agent_minus_human_effect_by_round"] = {
        round_id: _distribution(
            {
                instance: effects[instance]
                - human["rounds"][round_id]["effect"]["per_instance"][instance]
                for instance in sorted(instances)
            }
        )
        for round_id in ("0", "1")
    }
    report["conclusion"] = "complete_exploratory_comparison"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-analysis", type=Path, required=True)
    parser.add_argument("--human-manifest", type=Path, required=True)
    parser.add_argument("--agent-observations", type=Path, required=True)
    parser.add_argument("--agent-task", type=Path, required=True)
    parser.add_argument("--budget-sec", type=float, required=True)
    parser.add_argument("--instance-set", default="judge_id")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        args.human_analysis,
        args.human_manifest,
        args.agent_observations,
        args.agent_task,
        budget_sec=args.budget_sec,
        instance_set=args.instance_set,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("expected_pairs", "conclusion")}))


if __name__ == "__main__":
    main()
