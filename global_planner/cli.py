"""CLI interface for managing algorithm-space global plans."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shlex
import sys
import time
from pathlib import Path

import yaml

from global_planner.adapters.codex import (
    CodexPlannerAdapter,
    create_default_llm_callable,
)
from global_planner.controller import PlanningController
from global_planner.driver import PlanningSessionDriver
from global_planner.schema import HATSNode, Opportunity
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


def cmd_events(args: argparse.Namespace) -> int:
    store = PlanStore(args.dir)
    events = store.state.events
    if args.since is not None:
        events = events[args.since:]
    for event in events:
        print(json.dumps(event, ensure_ascii=False))
    return 0


def cmd_record_event(args: argparse.Namespace) -> int:
    payload = json.loads(args.payload)
    if not isinstance(payload, dict):
        raise ValueError("event payload must be a JSON object")
    store = PlanStore(args.dir)
    store.record_event(args.event_type, payload)
    print(f"Recorded event {store.state.events[-1]['event_id']}")
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
    if not math.isfinite(args.total_budget_sec) or args.total_budget_sec <= 0:
        print("--total-budget-sec must be a positive finite number", file=sys.stderr)
        return 2
    if not math.isfinite(args.agent_timeout) or args.agent_timeout <= 0:
        print("--agent-timeout must be a positive finite number", file=sys.stderr)
        return 2
    Path(args.dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=Path(args.dir) / "controller.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    store = PlanStore(args.dir)
    task_config = yaml.safe_load(Path(args.task_config).read_text(encoding="utf-8"))
    repo_root = Path(args.repository).resolve() if args.repository else Path.cwd()
    if not args.task:
        print("--task is required: provide the coding task instructions file", file=sys.stderr)
        return 2

    def planner_callable(prompt: str) -> str:
        spent = driver.get_current_budget_spent()
        remaining = max(1.0, args.total_budget_sec - spent)
        callable_fn = create_default_llm_callable(
            model_name=getattr(args, "planner_model", None) or getattr(args, "model", None),
            api_key=getattr(args, "api_key", None),
            api_base=getattr(args, "api_base", None),
            backend=getattr(args, "planner_backend", None),
            repository_root=repo_root,
            timeout_sec=min(300.0, remaining),
            remaining_budget_sec=remaining,
        )
        if callable_fn is None:
            raise RuntimeError("No planner model callable is configured")
        return callable_fn(prompt)

    planner_probe = create_default_llm_callable(
        model_name=getattr(args, "planner_model", None) or getattr(args, "model", None),
        api_key=getattr(args, "api_key", None),
        api_base=getattr(args, "api_base", None),
        backend=getattr(args, "planner_backend", None),
        repository_root=repo_root,
    )
    if planner_probe is None and not getattr(args, "allow_offline_fallback", False):
        print(
            "[global-plan-error] No LLM connection configured for formal planning mode. "
            "Set DEEPSEEK_API_KEY, OPENAI_API_KEY, or pass --model/--api-key, "
            "or use --allow-offline-fallback for offline testing.",
            file=sys.stderr,
        )
        return 1

    driver = PlanningSessionDriver(
        store=store,
        llm_callable=planner_callable if planner_probe is not None else None,
        total_budget_sec=args.total_budget_sec,
        allow_offline_fallback=getattr(args, "allow_offline_fallback", False),
        cooldown_sec=getattr(args, "cooldown_sec", 0.0),
        experiments_between_reviews=getattr(args, "experiments_between_reviews", 1),
    )
    # The coding-agent turn is a real persistent session step. The planner passes
    # its current opportunity and experiment into the agent, then expects the
    # agent to run `pitbench bench --global-plan-dir ...` before the next turn.
    def session_worker(decision, packet):
        evidence_count_before = len(store.state.evidence)
        latest_exp_before = (
            store.state.evidence[-1].experiment_id if store.state.evidence else None
        )
        started = time.monotonic()
        if packet.remaining_budget_sec <= 0:
            return {
                "experiment_id": f"budget_exhausted_{packet.completed_experiments_count:04d}",
                "execution_status": "missing", "solution_validity": "unverified",
                "performance_delta": "unknown", "actual_runs_count": 0,
                "expected_runs_count": 1,
                "metrics_summary": {"reason": "development budget exhausted"},
            }
        agent_callable = create_default_llm_callable(
            model_name=getattr(args, "agent_model", None), backend=args.agent,
            repository_root=repo_root,
            timeout_sec=min(args.agent_timeout, packet.remaining_budget_sec),
            remaining_budget_sec=packet.remaining_budget_sec,
        )
        if agent_callable is None:
            raise RuntimeError(f"Could not initialize '{args.agent}' agent adapter")
        os.environ["PITBENCH_GLOBAL_PLAN_CONTROLLED_SESSION"] = "1"
        prompt = (
            f"Task: {Path(args.task).read_text(encoding='utf-8')}\n\n"
            f"Global planning decision: {decision.model_dump_json()}\n"
            f"Context: {packet.model_dump_json()}\n\n"
            "Work on the selected opportunity in the repository. Run the required "
            "validation and `pitbench bench --global-plan-dir "
            f"{shlex.quote(str(Path(args.dir).resolve()))} --experiment-id "
            f"exp_{packet.completed_experiments_count + 1:04d}`. "
            "Do not claim results that bench did not produce. Return JSON with "
            "hypothesis_id, hypothesis, discovered_nodes, discovered_opportunities, "
            "and revalidated_node_ids. New nodes must include id, level, purpose, "
            "parent_id, and source_refs; new opportunities must include id, title, "
            "target_nodes, proposal, expected_effect, and "
            "cheapest_discriminating_experiment. On resumption, inspect persisted state."
        )
        try:
            agent_output = agent_callable(prompt)
        except Exception as exc:
            outcome = {"execution_status": "fail", "solution_validity": "unverified",
                       "performance_delta": "unknown", "actual_runs_count": 0,
                       "expected_runs_count": 1,
                       "metrics_summary": {"agent_error": str(exc)}}
            store.record_event("agent.turn_failed", {"error": str(exc)})
            return outcome
        agent_summary: dict = {}
        try:
            match = re.search(r"\{.*\}", agent_output, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    agent_summary = parsed
        except (json.JSONDecodeError, TypeError):
            pass
        discovered_nodes = []
        for item in agent_summary.get("discovered_nodes", []):
            try:
                discovered_nodes.append(HATSNode.model_validate(item).model_dump())
            except Exception:
                logging.getLogger("global_planner.agent").warning(
                    "Ignoring invalid discovered HATS node"
                )
        discovered_opportunities = []
        for item in agent_summary.get("discovered_opportunities", []):
            try:
                discovered_opportunities.append(
                    Opportunity.model_validate(item).model_dump()
                )
            except Exception:
                logging.getLogger("global_planner.agent").warning(
                    "Ignoring invalid discovered opportunity"
                )
        model_updates = {
            "hypothesis_id": agent_summary.get("hypothesis_id"),
            "discovered_nodes": discovered_nodes,
            "discovered_opportunities": discovered_opportunities,
            "revalidated_node_ids": agent_summary.get("revalidated_node_ids", []),
        }
        # Read the just-recorded bench boundary from the persistent state. If the
        # agent omitted bench, preserve the missing observation explicitly.
        try:
            store.load()
            if (len(store.state.evidence) == evidence_count_before or
                    store.state.evidence[-1].experiment_id == latest_exp_before):
                raise IndexError
            latest = store.state.evidence[-1]
            return {
                "experiment_id": latest.experiment_id,
                "execution_status": latest.execution_status,
                "solution_validity": latest.solution_validity,
                "performance_delta": latest.performance_delta,
                "hypothesis_supported": latest.hypothesis_supported,
                "hypothesis_id": latest.hypothesis_id,
                "actual_runs_count": latest.metrics_summary.get("runs_count", 0),
                "expected_runs_count": latest.metrics_summary.get("expected_runs_count", 1),
                "metrics_summary": latest.metrics_summary,
                "candidate_patch_sha256": latest.candidate_patch_sha256,
                "observations_path": latest.observations_path,
                "target_nodes": latest.target_nodes,
                "agent_duration_sec": time.monotonic() - started,
                "evidence_recorded": True,
                **model_updates,
            }
        except (IndexError, FileNotFoundError):
            return {
                "experiment_id": f"agent_step_{packet.completed_experiments_count + 1:04d}",
                "execution_status": "missing",
                "solution_validity": "unverified",
                "performance_delta": "unknown",
                "actual_runs_count": 0,
                "expected_runs_count": 1,
                "metrics_summary": {
                    "reason": "Agent completed without recording pitbench bench",
                    "agent_output": agent_output[-2000:],
                },
                **model_updates,
            }

    max_steps = getattr(args, "max_steps", 2)
    print(f"Starting closed-loop planning session for {task_config.get('task_id')} (max {max_steps} steps)...")
    decisions = driver.run_session(
        task_config=task_config,
        agent_worker=session_worker,
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

    p_events = subparsers.add_parser("events", help="Read persistent controller events")
    p_events.add_argument("--dir", required=True, help="Storage directory")
    p_events.add_argument("--since", type=int, default=None, help="zero-based event offset")
    p_events.set_defaults(func=cmd_events)

    p_record_event = subparsers.add_parser("record-event", help="Append a hook or adapter event")
    p_record_event.add_argument("--dir", required=True, help="Storage directory")
    p_record_event.add_argument("--event-type", required=True)
    p_record_event.add_argument("--payload", default="{}", help="JSON object")
    p_record_event.set_defaults(func=cmd_record_event)

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
    p_run.add_argument("--task", required=True, help="Task instructions file passed to the coding agent")
    p_run.add_argument("--agent", choices=("codex", "claude-code", "antigravity"), default="codex",
                       help="coding agent adapter")
    p_run.add_argument("--agent-model", default=None, help="coding agent model override")
    p_run.add_argument("--agent-timeout", type=float, default=7200)
    p_run.add_argument("--total-budget-sec", type=float, default=14400)
    p_run.add_argument("--study-id", default="planning_study_v1", help="Study ID")
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
    p_run.add_argument("--planner-backend", choices=("antigravity", "agy", "litellm", "deepseek", "codex", "claude-code"), default=None)
    p_run.add_argument("--planner-model", default=None, help="planning model override")
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
