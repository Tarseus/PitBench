"""Configuration and tracing checks through maintained interfaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _feedback() -> None:
    from pitbench.cli.evaluate_config import compare_feedback_configs

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcome-config", type=Path, required=True)
    parser.add_argument("--behavior-config", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(compare_feedback_configs(args.outcome_config, args.behavior_config))
    )


def _tracing():
    from pitbench.harness.agents import AgentFactory
    from pitbench.harness.utils.trace_validation import run_trace_smoke

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent", required=True, choices=sorted(AgentFactory.AGENT_NAME_TO_IMPORT_PATH)
    )
    parser.add_argument("--model")
    parser.add_argument(
        "--config", type=Path, default=Path("config/evaluate.local.yaml")
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument(
        "--user", default="root", help="Task-container user for the smoke run"
    )
    parser.add_argument("--agent-kwarg", action="append", default=[])
    args = parser.parse_args()
    return run_trace_smoke(args)


def main(argv=None):
    if __package__ in (None, ""):
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts import dispatch

    return dispatch(
        {"feedback": _feedback, "tracing": _tracing},
        argv,
        description="Check matched feedback configuration or Agent tracing.",
    )


if __name__ == "__main__":
    main()
