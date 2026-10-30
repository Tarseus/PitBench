"""Construct Research Cases or materialize Agent-designed public experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pitbench.research import (
    construct_research_case,
    materialize_research_experiment,
    prepare_generalization_study,
)
from pitbench.schema.research import GeneralizationStudyPlan, ResearchPilotPlan


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
    study = sub.add_parser(
        "study",
        help="Materialize a controlled generalization study and readiness inventory",
    )
    study.add_argument("--plan", type=Path, required=True)
    study.add_argument("--root", type=Path, required=True)
    study.add_argument("--output-dir", type=Path, required=True)
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
    elif args.command == "study":
        result = prepare_generalization_study(
            GeneralizationStudyPlan.model_validate_json(args.plan.read_text()),
            root=args.root,
            destination=args.output_dir,
        )
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in [
                        "study_id",
                        "status",
                        "ready_for_agent_execution",
                        "planned_agent_runs",
                        "maximum_agent_wall_hours",
                        "readiness_gaps",
                    ]
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
