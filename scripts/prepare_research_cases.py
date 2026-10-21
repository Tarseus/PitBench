"""Construct Research Cases or materialize Agent-designed public experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pitbench.research import construct_research_case, materialize_research_experiment
from pitbench.schema.research import ResearchPilotPlan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("construct")
    build.add_argument("--plan", type=Path, required=True)
    build.add_argument("--root", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--agent-view", type=Path, required=True)
    generate.add_argument("--generator-config", type=Path, required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "construct":
        plan = ResearchPilotPlan.model_validate_json(args.plan.read_text())
        results = [
            construct_research_case(
                item,
                root=args.root.resolve(),
                destination=args.output_dir / item.case_id,
                protocol_path=args.root.resolve() / plan.protocol,
            )
            for item in plan.cases
        ]
        print(
            json.dumps(
                {
                    "cases": len(results),
                    "guidance_views": 2 * len(results),
                    "statuses": {item["case_id"]: item["status"] for item in results},
                }
            )
        )
    else:
        result = materialize_research_experiment(
            json.loads(args.agent_view.read_text()),
            json.loads(args.generator_config.read_text()),
            args.output_dir,
        )
        print(json.dumps(result))


if __name__ == "__main__":
    main()
