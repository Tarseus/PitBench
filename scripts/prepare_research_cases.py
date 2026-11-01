"""Construct Research Cases or materialize Agent-designed public experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pitbench.research import (
    analyze_population_difficulty,
    audit_population_design,
    construct_generalization_populations,
    construct_research_case,
    freeze_generalization_measurements,
    import_population_references,
    materialize_generalization_views,
    materialize_research_experiment,
    measure_population_headroom,
    prepare_generalization_study,
    prepare_population_references,
    qualify_scenario_runtime,
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
    for command in ("validate", "scenario-build"):
        operation = sub.add_parser(
            command,
            help="Prepare source validation or source-bound API builds using the Judge workspace path",
        )
        operation.add_argument("--task-config", type=Path, required=True)
        operation.add_argument("--root", type=Path, required=True)
        operation.add_argument("--source", type=Path, required=True)
        operation.add_argument("--private-root", type=Path, required=True)
        operation.add_argument("--candidate-patch", type=Path)
        operation.add_argument("--output-dir", type=Path, required=True)
        if command == "validate":
            operation.add_argument("--include-excluded-tests", action="store_true")
    binding = sub.add_parser(
        "binding", help="Record artifacts after an evaluator-owned source binding build"
    )
    binding.add_argument("--source", type=Path, required=True)
    package = binding.add_mutually_exclusive_group(required=True)
    package.add_argument("--package")
    package.add_argument("--native-library", type=Path)
    binding.add_argument("--repository-plugin", required=True)
    binding.add_argument("--version", required=True)
    binding.add_argument("--output", type=Path, required=True)
    qualify = sub.add_parser(
        "scenario-qualify",
        help="Run declared analytic fixtures through the real scenario collector",
    )
    qualify.add_argument("--task-config", type=Path, required=True)
    qualify.add_argument("--binding-manifest", type=Path, required=True)
    qualify.add_argument("--specification", type=Path, required=True)
    qualify.add_argument("--output-dir", type=Path, required=True)
    qualify.add_argument("--python")
    populations = sub.add_parser(
        "populations",
        help="Construct declared development and hidden distributions with leakage checks",
    )
    populations.add_argument("--specification", type=Path, required=True)
    populations.add_argument("--output-dir", type=Path, required=True)
    audit = sub.add_parser(
        "audit", help="Audit actual development exposure and scale/structure claims"
    )
    audit.add_argument("--specification", type=Path, required=True)
    audit.add_argument("--populations", type=Path, required=True)
    audit.add_argument("--root", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    measurements = sub.add_parser(
        "measurements", help="Freeze diagnostic scenarios and formal task views"
    )
    for flag in (
        "views",
        "populations",
        "references",
        "specification",
        "root",
        "private-root",
        "output-dir",
    ):
        measurements.add_argument("--" + flag, type=Path, required=True)
    difficulty = sub.add_parser(
        "difficulty", help="Analyze difficulty and select budgets from calibration only"
    )
    difficulty.add_argument("--populations", type=Path, required=True)
    difficulty.add_argument("--results", type=Path, action="append", required=True)
    difficulty.add_argument("--protocol", type=Path, required=True)
    difficulty.add_argument("--output", type=Path, required=True)
    references = sub.add_parser(
        "references", help="Build and independently check population targets"
    )
    references.add_argument("--populations", type=Path, required=True)
    references.add_argument("--task-config", type=Path, required=True)
    references.add_argument("--private-root", type=Path, required=True)
    references.add_argument("--output-dir", type=Path, required=True)
    imported = sub.add_parser(
        "import-references",
        help="Import independently checked distributed BKS and optimal returns",
    )
    imported.add_argument("--populations", type=Path, required=True)
    imported.add_argument("--results", type=Path, action="append", required=True)
    imported.add_argument("--private-root", type=Path, required=True)
    imported.add_argument("--output-dir", type=Path, required=True)
    headroom = sub.add_parser(
        "headroom",
        help="Measure an evaluator-owned source reference with the ordinary Judge",
    )
    for flag in (
        "task-config",
        "source",
        "root",
        "private-root",
        "populations",
        "references",
        "candidate-patch",
        "output-dir",
    ):
        headroom.add_argument("--" + flag, type=Path, required=True)
    headroom.add_argument("--parallel-runs", type=int, default=4)
    headroom.add_argument("--condition", action="append")
    views = sub.add_parser(
        "views", help="Materialize shared evaluator and arm-specific development tasks"
    )
    views.add_argument("--difficulty", type=Path)
    for flag in (
        "plan",
        "root",
        "private-root",
        "populations",
        "references",
        "output-dir",
    ):
        views.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "measurements":
        result = freeze_generalization_measurements(
            views_path=args.views,
            population_root=args.populations,
            references_path=args.references,
            specification_path=args.specification,
            root=args.root,
            private_root=args.private_root,
            destination=args.output_dir,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "tasks": len(result["tasks"]),
                    "arms": len(result["arms"]),
                }
            )
        )
    elif args.command == "import-references":
        result = import_population_references(
            population_root=args.populations,
            result_roots=args.results,
            private_root=args.private_root,
            destination=args.output_dir,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "records": len(result["records"]),
                    "missing": len(result["missing"]),
                }
            )
        )
    elif args.command == "difficulty":
        result = analyze_population_difficulty(
            population_root=args.populations,
            result_roots=args.results,
            protocol_path=args.protocol,
            destination=args.output,
        )
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in [
                        "status",
                        "population_instances",
                        "instances_with_verified_reference",
                        "solver_established_optima",
                    ]
                }
            )
        )
    elif args.command == "audit":
        result = audit_population_design(
            specification_path=args.specification,
            population_root=args.populations,
            root=args.root,
            destination=args.output,
        )
        print(
            json.dumps(
                {key: result[key] for key in ["status", "development_maxima", "gaps"]}
            )
        )
        if result["status"] != "passed":
            raise SystemExit(1)
    elif args.command == "views":
        result = materialize_generalization_views(
            plan=GeneralizationStudyPlan.model_validate_json(args.plan.read_text()),
            root=args.root,
            private_root=args.private_root,
            population_root=args.populations,
            references_path=args.references,
            destination=args.output_dir,
            difficulty_path=args.difficulty,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "tasks": len(result["tasks"]),
                    "arms": len(result["arms"]),
                }
            )
        )
    elif args.command == "headroom":
        result = measure_population_headroom(
            task_path=args.task_config,
            base_repository=args.source,
            public_root=args.root,
            private_root=args.private_root,
            population_root=args.populations,
            references_path=args.references,
            candidate_patch=args.candidate_patch,
            destination=args.output_dir,
            parallel_runs=args.parallel_runs,
            condition_ids=args.condition,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "opportunity_by_panel": result.get("opportunity_by_panel"),
                }
            )
        )
    elif args.command == "references":
        result = prepare_population_references(
            population_root=args.populations,
            task_path=args.task_config,
            private_root=args.private_root,
            destination=args.output_dir,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "records": len(result["records"]),
                    "panels": len(result["panels"]),
                    "headroom_status": result["headroom_status"],
                }
            )
        )
    elif args.command == "populations":
        print(
            json.dumps(
                construct_generalization_populations(
                    specification_path=args.specification, destination=args.output_dir
                )
            )
        )
    elif args.command == "scenario-qualify":
        result = qualify_scenario_runtime(
            task_path=args.task_config,
            binding_path=args.binding_manifest,
            specification_path=args.specification,
            destination=args.output_dir,
            python=args.python,
        )
        print(json.dumps(result))
        if any(value["status"] == "failed" for value in result["scenarios"].values()):
            raise SystemExit(1)
    elif args.command == "binding":
        from pitbench.solver_drivers.common import record_source_binding

        print(
            json.dumps(
                record_source_binding(
                    source=args.source,
                    package=args.package,
                    repository_plugin=args.repository_plugin,
                    version=args.version,
                    destination=args.output,
                    native_library=args.native_library,
                )
            )
        )
    elif args.command in {"validate", "scenario-build"}:
        from pitbench.evaluator.judge import LocalProcessJudge, _development_seeds
        from pitbench.schema.observation import CodeState
        from pitbench.schema.task import PitBenchTask

        if args.output_dir.exists():
            raise FileExistsError(args.output_dir)
        task = PitBenchTask.from_yaml(args.task_config)
        judge = LocalProcessJudge(
            task=task,
            base_repository=args.source,
            public_root=args.root,
            private_root=args.private_root,
            candidate_patch=args.candidate_patch,
            output_dir=args.output_dir,
            code_states=(CodeState.AGENT,)
            if args.candidate_patch
            else (CodeState.BASE,),
            evaluation_seeds=tuple(_development_seeds(task)),
        )
        result = (
            judge.build_scenario_sources()
            if args.command == "scenario-build"
            else judge.validate_sources(
                include_excluded_tests=args.include_excluded_tests
            )
        )
        print(json.dumps(result))
        if result["status"] != "passed":
            raise SystemExit(1)
    elif args.command == "construct":
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
