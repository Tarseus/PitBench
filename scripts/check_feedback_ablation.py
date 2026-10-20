"""Confirm feedback arms differ only in structured public diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pitbench.cli.evaluate_config import EvaluationConfig


def compare(outcome_config: Path, behavior_config: Path) -> dict[str, object]:
    outcome = EvaluationConfig.from_yaml(outcome_config).model_dump(mode="json")
    behavior = EvaluationConfig.from_yaml(behavior_config).model_dump(mode="json")
    if outcome.pop("agent_feedback_mode") != "outcome_only":
        raise ValueError("outcome arm must use outcome_only feedback")
    if behavior.pop("agent_feedback_mode") != "solver_behavior":
        raise ValueError("behavior arm must use solver_behavior feedback")
    if outcome != behavior:
        raise ValueError("feedback arms differ in settings other than feedback mode")
    if "bench" not in outcome["agent_tools"]:
        raise ValueError("matched feedback arms require the same public bench tool")
    digest = hashlib.sha256(json.dumps(outcome, sort_keys=True).encode()).hexdigest()
    return {"matched": True, "common_config_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcome-config", type=Path, required=True)
    parser.add_argument("--behavior-config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.outcome_config, args.behavior_config)))


if __name__ == "__main__":
    main()
