"""CLI interface for managing algorithm-space global plans."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

from global_planner.adapters.codex import (
    CodexPlannerAdapter,
    create_default_llm_callable,
)
from global_planner.controller import PlanningController
from global_planner.driver import PlanningSessionDriver, build_autonomous_worker
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
        execution_success=not args.refuted,
        solution_valid=not args.refuted,
        performance_improved=is_supp,
        hypothesis_supported=is_supp,
        current_budget_spent=args.budget_spent,
    )
    print(f"Recorded evidence {ev.id} (Status: {ev.status}). Trigger replan: {should_replan}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    store = PlanStore(args.dir)
    task_config = yaml.safe_load(Path(args.task_config).read_text(encoding="utf-8"))
    repo_root = Path(args.repository).resolve() if args.repository else Path.cwd()

    # Configure model connection
    llm_callable = create_default_llm_callable(
        model_name=getattr(args, "model", None),
        api_key=getattr(args, "api_key", None),
        api_base=getattr(args, "api_base", None),
        backend=getattr(args, "agent", None) or getattr(args, "backend", None),
    )
    if llm_callable is None and not getattr(args, "allow_offline_fallback", False):
        print(
            "[global-plan-error] No LLM connection configured for formal planning mode. "
            "Set DEEPSEEK_API_KEY, OPENAI_API_KEY, or pass --model/--api-key, "
            "or use --allow-offline-fallback for offline testing.",
            file=sys.stderr,
        )
        return 1

    driver = PlanningSessionDriver(
        store=store,
        llm_callable=llm_callable,
        allow_offline_fallback=getattr(args, "allow_offline_fallback", False),
        cooldown_sec=getattr(args, "cooldown_sec", 0.0),
        experiments_between_reviews=getattr(args, "experiments_between_reviews", 1),
    )

    worker = build_autonomous_worker(
        task_config=task_config,
        repository_root=repo_root,
        llm_callable=llm_callable,
    )

    max_steps = getattr(args, "max_steps", 2)
    print(f"Starting closed-loop planning session for {task_config.get('task_id')} (max {max_steps} steps)...")
    decisions = driver.run_session(
        task_config=task_config,
        agent_worker=worker,
        max_steps=max_steps,
        study_id=args.study_id,
        repository_root=repo_root,
    )

    print(f"Completed session: executed {len(decisions)} replanning decisions.")
    for i, dec in enumerate(decisions, 1):
        print(f"  Step {i}: {dec.decision.value} -> {dec.selected_opportunity} (Next: {dec.next_experiment})")
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

    p_run = subparsers.add_parser("run", help="Bootstrap and drive planning session from task config")
    p_run.add_argument("--dir", required=True, help="Storage directory")
    p_run.add_argument("--task-config", required=True, help="Path to task YAML config")
    p_run.add_argument("--study-id", default="planning_study_v1", help="Study ID")
    p_run.add_argument(
        "--agent",
        "--backend",
        dest="agent",
        default="antigravity" if shutil.which("agy") else None,
        choices=("antigravity", "agy", "litellm", "deepseek", "codex"),
        help="Planner LLM or coding agent execution backend (default: antigravity via agy CLI if present)",
    )
    p_run.add_argument("--repository", default=None, help="Solver repository root path")
    p_run.add_argument("--max-steps", type=int, default=2, help="Maximum execution steps in the session")
    p_run.add_argument("--cooldown-sec", type=float, default=0.0, help="Minimum cooldown between replanning reviews")
    p_run.add_argument(
        "--experiments-between-reviews",
        type=int,
        default=1,
        help="Number of completed experiments before triggering a review",
    )
    p_run.add_argument("--model", default=None, help="LLM model name override (e.g. deepseek/deepseek-chat, gpt-4o)")
    p_run.add_argument("--api-key", default=None, help="API key override")
    p_run.add_argument("--api-base", default=None, help="API base URL override")
    p_run.add_argument(
        "--allow-offline-fallback",
        action="store_true",
        help="allow heuristic fallback when no LLM connection is configured (test mode only)",
    )
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
