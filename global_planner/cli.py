"""CLI interface for managing algorithm-space global plans."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from global_planner.adapters.codex import CodexPlannerAdapter
from global_planner.controller import PlanningController
from global_planner.store import PlanStore


def cmd_init(args: argparse.Namespace) -> int:
    store = PlanStore(args.dir)
    controller = PlanningController(store=store)
    controller.on_task_start(
        study_id=args.study_id,
        task_id=args.task_id,
        solver=args.solver,
    )
    print(f"Initialized Global Plan at {store.state_file}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    store = PlanStore(args.dir)
    st = store.state
    print(f"=== Global Plan Status: {st.task_id} ({st.solver}) ===")
    print(f"Revision: {st.revision} | Experiments: {st.experiments_count} | Decisions: {len(st.decisions)}")
    print(f"Active Opportunity: {st.active_opportunity_id or 'None'}")
    print(f"Nodes: {len(st.nodes)} | Opportunities: {len(st.opportunities)} | Evidence: {len(st.evidence)}")
    if st.decisions:
        last = st.decisions[-1]
        print(f"Last Decision: {last.decision.value} -> {last.selected_opportunity}")
    return 0


def cmd_prompt(args: argparse.Namespace) -> int:
    store = PlanStore(args.dir)
    controller = PlanningController(store=store)
    packet = controller.build_context_packet(current_budget_spent=args.budget_spent)
    adapter = CodexPlannerAdapter()
    print(adapter.format_planning_prompt(packet))
    return 0


def cmd_record_decision(args: argparse.Namespace) -> int:
    store = PlanStore(args.dir)
    controller = PlanningController(store=store)
    data = json.loads(Path(args.decision_file).read_text(encoding="utf-8"))
    dec = controller.parse_and_record_decision(data, current_budget_spent=args.budget_spent)
    print(f"Recorded decision: {dec.decision.value} -> {dec.selected_opportunity}")
    return 0


def cmd_record_evidence(args: argparse.Namespace) -> int:
    store = PlanStore(args.dir)
    controller = PlanningController(store=store)
    is_supp = True if args.supported else (False if args.refuted else None)
    metrics = {}
    if args.metrics:
        metrics = json.loads(args.metrics)
    target_nodes = args.target_nodes.split(",") if args.target_nodes else []

    ev, should_replan = controller.on_experiment_completed(
        experiment_id=args.experiment_id,
        target_nodes=target_nodes,
        metrics_summary=metrics,
        is_supported=is_supp,
        current_budget_spent=args.budget_spent,
    )
    print(f"Recorded evidence {ev.id} (Status: {ev.status}). Trigger replan: {should_replan}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Global Planner CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_init = subparsers.add_parser("init", help="Initialize plan store")
    p_init.add_argument("--dir", required=True, help="Storage directory")
    p_init.add_argument("--study-id", default="planning_study_v1", help="Study ID")
    p_init.add_argument("--task-id", required=True, help="Task ID")
    p_init.add_argument("--solver", required=True, help="Target solver")
    p_init.set_defaults(func=cmd_init)

    p_status = subparsers.add_parser("status", help="Show current plan status")
    p_status.add_argument("--dir", required=True, help="Storage directory")
    p_status.set_defaults(func=cmd_status)

    p_prompt = subparsers.add_parser("prompt", help="Generate replanning prompt")
    p_prompt.add_argument("--dir", required=True, help="Storage directory")
    p_prompt.add_argument("--budget-spent", type=float, default=0.0)
    p_prompt.set_defaults(func=cmd_prompt)

    p_dec = subparsers.add_parser("record-decision", help="Record replanning decision from JSON file")
    p_dec.add_argument("--dir", required=True, help="Storage directory")
    p_dec.add_argument("--decision-file", required=True, help="Path to decision JSON file")
    p_dec.add_argument("--budget-spent", type=float, default=0.0)
    p_dec.set_defaults(func=cmd_record_decision)

    p_ev = subparsers.add_parser("record-evidence", help="Record experimental evidence")
    p_ev.add_argument("--dir", required=True, help="Storage directory")
    p_ev.add_argument("--experiment-id", required=True)
    p_ev.add_argument("--target-nodes", default="", help="Comma-separated node IDs")
    p_ev.add_argument("--supported", action="store_true")
    p_ev.add_argument("--refuted", action="store_true")
    p_ev.add_argument("--metrics", default=None, help="JSON metrics summary")
    p_ev.add_argument("--budget-spent", type=float, default=0.0)
    p_ev.set_defaults(func=cmd_record_evidence)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
