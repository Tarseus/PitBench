"""End-to-end planning session driver coordinating bootstrap, triggers, and agent continuation."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from global_planner.adapters.codex import CodexPlannerAdapter
from global_planner.bootstrap import bootstrap_hats_model
from global_planner.controller import PlanningController
from global_planner.schema import (
    ContextPacket,
    DecisionType,
    ExecutionStatus,
    HATSNode,
    Opportunity,
    PerformanceDelta,
    ReplanningDecision,
    SolutionValidity,
)
from global_planner.store import PlanStore


class PlanningSessionDriver:
    """Manages the end-to-end planning loop across task bootstrap, experiment boundaries,

    decision delivery, and automated agent worker continuation.
    """

    def __init__(
        self,
        store: PlanStore,
        total_budget_sec: float = 14400.0,
        cooldown_sec: float = 300.0,
        experiments_between_reviews: int = 1,
        adapter: CodexPlannerAdapter | None = None,
        llm_callable: Callable[[str], str] | None = None,
        allow_offline_fallback: bool = False,
    ):
        self.store = store
        self.controller = PlanningController(
            store=store,
            total_budget_sec=total_budget_sec,
            cooldown_sec=cooldown_sec,
            experiments_between_reviews=experiments_between_reviews,
        )
        self.allow_offline_fallback = allow_offline_fallback
        self.adapter = adapter or CodexPlannerAdapter(
            default_llm_callable=llm_callable,
            allow_offline_fallback=allow_offline_fallback,
        )
        self.llm_callable = llm_callable

        # Preserve cumulative budget across process restarts
        self.initial_budget_spent = 0.0
        if self.store.state_file.exists():
            try:
                self.initial_budget_spent = self.store.state.budget_spent_sec
            except Exception:
                self.initial_budget_spent = 0.0
        self.start_wall_time = time.time()
        if self.store.state_file.exists():
            st = self.store.state
            if st.budget_clock_started_at_utc is None:
                st.budget_clock_started_at_utc = datetime.now(timezone.utc).isoformat()
                self.store.save()
        self._planning_call_started = 0.0

    def get_current_budget_spent(self) -> float:
        """Returns cumulative elapsed budget spent across sessions."""
        if self.store.state_file.exists() and self.store.state.budget_clock_started_at_utc:
            started = datetime.fromisoformat(self.store.state.budget_clock_started_at_utc)
            return max(
                self.initial_budget_spent,
                (datetime.now(timezone.utc) - started).total_seconds(),
            )
        elapsed = max(0.0, time.time() - self.start_wall_time)
        return self.initial_budget_spent + elapsed

    def _request_and_record_decision(self, packet, repository_root=None):
        started = time.monotonic()
        decision = self.adapter.request_decision(
            packet,
            llm_callable=self.llm_callable,
            allow_offline_fallback=self.allow_offline_fallback,
            repository_root=repository_root,
        )
        spent = self.get_current_budget_spent()
        self.store.update_budget_spent(spent)
        self.store.record_event("planner.decision", {
            "decision": decision.decision.value,
            "selected_opportunity": decision.selected_opportunity,
            "evidence_refs": decision.evidence_refs,
            "duration_sec": max(0.0, time.monotonic() - started),
        })
        self.controller.parse_and_record_decision(
            decision.model_dump(), current_budget_spent=spent
        )
        return decision

    def bootstrap(
        self,
        task_config: dict[str, Any],
        study_id: str = "planning_study_v1",
        repository_root: Path | str | None = None,
    ) -> tuple[ContextPacket, ReplanningDecision]:
        """Bootstraps initial HATS model, opportunities, and initial decision from task metadata."""
        task_id = str(task_config.get("task_id", "unknown_task"))
        solver = str(task_config.get("release", {}).get("repository", "solver")).split("/")[-1]
        family = str(task_config.get("problem_family", "general"))
        protocol = str(task_config.get("evaluation", {}).get("performance_protocol", "exact_verified_solve"))

        repo_root = Path(repository_root) if repository_root else None
        goal_stmt, nodes, opps = bootstrap_hats_model(task_config, repository_root=repo_root)
        self.controller.on_task_start(
            study_id=study_id,
            task_id=task_id,
            solver=solver,
            goal_statement=goal_stmt,
            problem_family=family,
            performance_protocol=protocol,
            initial_nodes=nodes,
            initial_opportunities=opps,
        )

        spent = self.get_current_budget_spent()
        self.store.update_budget_spent(spent)
        packet = self.controller.build_context_packet(current_budget_spent=spent)
        decision = self._request_and_record_decision(packet, repo_root)
        return packet, decision

    def on_experiment_boundary(
        self,
        experiment_id: str,
        target_nodes: list[str],
        metrics_summary: dict[str, Any],
        execution_status: ExecutionStatus = ExecutionStatus.PASS,
        solution_validity: SolutionValidity = SolutionValidity.VALID,
        performance_delta: PerformanceDelta = PerformanceDelta.UNKNOWN,
        hypothesis_supported: bool | None = None,
        hypothesis_id: str | None = None,
        observations_path: str | None = None,
        candidate_patch_sha256: str | None = None,
        current_budget_spent: float | None = None,
        # Backward compatibility keyword arguments
        execution_success: bool | None = None,
        solution_valid: bool | None = None,
        performance_improved: bool | None = None,
        repository_root: Path | str | None = None,
    ) -> tuple[Any, ReplanningDecision | None]:
        """Processes an experiment completion, records evidence, and executes replanning if triggered."""
        spent = current_budget_spent if current_budget_spent is not None else self.get_current_budget_spent()
        self.store.update_budget_spent(spent)
        ev, should_replan = self.controller.on_experiment_completed(
            experiment_id=experiment_id,
            target_nodes=target_nodes,
            metrics_summary=metrics_summary,
            execution_status=execution_status,
            solution_validity=solution_validity,
            performance_delta=performance_delta,
            hypothesis_supported=hypothesis_supported,
            hypothesis_id=hypothesis_id,
            observations_path=observations_path,
            candidate_patch_sha256=candidate_patch_sha256,
            current_budget_spent=spent,
            execution_success=execution_success,
            solution_valid=solution_valid,
            performance_improved=performance_improved,
        )

        new_decision: ReplanningDecision | None = None
        if should_replan:
            packet = self.controller.build_context_packet(current_budget_spent=spent)
            new_decision = self._request_and_record_decision(packet, repository_root)

        return ev, new_decision

    def run_session(
        self,
        task_config: dict[str, Any],
        agent_worker: Callable[[ReplanningDecision, ContextPacket], dict[str, Any]],
        max_steps: int = 50,
        study_id: str = "planning_study_v1",
        repository_root: Path | str | None = None,
    ) -> list[ReplanningDecision]:
        """Executes the full end-to-end closed loop driving the coding agent worker across

        decision delivery -> implementation -> experiment evaluation -> dynamic model update -> replanning.
        """
        repo_root = Path(repository_root) if repository_root else None
        resumed = self.store.state_file.exists()
        if resumed:
            self.store.load()
            if self.store.state.task_id != str(task_config.get("task_id", "unknown_task")):
                raise ValueError(
                    f"plan task '{self.store.state.task_id}' does not match requested "
                    f"task '{task_config.get('task_id', 'unknown_task')}'"
                )
        # 1. Bootstrap or resume
        if not self.store.state_file.exists() or len(self.store.state.decisions) == 0:
            packet, decision = self.bootstrap(
                task_config=task_config, study_id=study_id, repository_root=repo_root
            )
        else:
            decision = self.store.state.decisions[-1]
            if decision.decision == DecisionType.STOP:
                self.store.record_event(
                    "session.resumed", {"study_id": study_id, "max_steps": max_steps}
                )
                self.store.record_event(
                    "session.finished", {"executed_steps": 0, "stopped": True}
                )
                return [decision]
            packet = self.controller.build_context_packet(self.get_current_budget_spent())

        executed_decisions: list[ReplanningDecision] = [decision]
        self.store.record_event(
            "session.resumed" if resumed else "session.started",
            {"study_id": study_id, "max_steps": max_steps},
        )

        try:
            # 2. Closed-loop execution loop
            for step_idx in range(max_steps):
                spent = self.get_current_budget_spent()
                self.store.update_budget_spent(spent)
                if spent >= self.controller.total_budget_sec:
                    break
                if decision.decision == DecisionType.STOP:
                    break

                # Always refresh context packet at the start of every step
                # so worker receives up-to-date completed_experiments_count, recent_evidence, and remaining budget
                packet = self.controller.build_context_packet(spent)

                # Hand off decision and fresh context packet to the coding agent worker
                worker_outcome = agent_worker(decision, packet)
                # A real agent may run pitbench bench in a child process, which
                # commits evidence and events through another PlanStore instance.
                # Reload before any controller writes to avoid replacing that state
                # with this process's stale in-memory snapshot.
                if isinstance(worker_outcome, dict) and worker_outcome.get("evidence_recorded"):
                    self.store.load()

                # Validate worker outcome (never default missing/empty worker result to PASS/VALID)
                if not isinstance(worker_outcome, dict) or not worker_outcome:
                    exec_status = ExecutionStatus.MISSING
                    sol_validity = SolutionValidity.UNVERIFIED
                    perf_delta = PerformanceDelta.UNKNOWN
                    hyp_supp = None
                else:
                    exec_status = worker_outcome.get("execution_status", ExecutionStatus.MISSING)
                    sol_validity = worker_outcome.get("solution_validity", SolutionValidity.UNVERIFIED)
                    perf_delta = worker_outcome.get("performance_delta", PerformanceDelta.UNKNOWN)
                    hyp_supp = worker_outcome.get("hypothesis_supported")

                    # Check runs count vs expected runs if specified
                    expected_runs = worker_outcome.get("expected_runs_count")
                    actual_runs = worker_outcome.get("actual_runs_count")
                    if actual_runs is None:
                        actual_runs = worker_outcome.get("metrics_summary", {}).get("runs_count")

                    # If expected_runs is specified, verify actual runs completeness
                    if expected_runs is not None:
                        if actual_runs is None or actual_runs < expected_runs:
                            exec_status = ExecutionStatus.MISSING
                            sol_validity = SolutionValidity.UNVERIFIED
                            perf_delta = PerformanceDelta.UNKNOWN
                            hyp_supp = None

                    # If actual runs count is explicitly 0 or no runs occurred, it cannot be PASS / VALID / IMPROVED
                    if actual_runs is not None and actual_runs == 0:
                        exec_status = ExecutionStatus.MISSING
                        sol_validity = SolutionValidity.UNVERIFIED
                        perf_delta = PerformanceDelta.UNKNOWN
                        hyp_supp = None

                # Process dynamic model updates from worker
                if isinstance(worker_outcome, dict):
                    discovered_nodes = worker_outcome.get("discovered_nodes")
                    if discovered_nodes:
                        parsed_nodes = [
                            HATSNode.model_validate(n) if isinstance(n, dict) else n
                            for n in discovered_nodes
                        ]
                        self.store.register_nodes_batch(parsed_nodes)
                    discovered_opps = worker_outcome.get("discovered_opportunities")
                    if discovered_opps:
                        for opp in discovered_opps:
                            parsed_opp = Opportunity.model_validate(opp) if isinstance(opp, dict) else opp
                            self.store.register_opportunity(parsed_opp)
                    reval_ids = worker_outcome.get("revalidated_node_ids")
                    if reval_ids:
                        for nid in reval_ids:
                            if nid in self.store.state.nodes:
                                self.store.state.nodes[nid].status = "active"

                    self.store.record_event("agent.turn_completed", {
                        "step": step_idx + 1,
                        "experiment_id": worker_outcome.get("experiment_id"),
                        "execution_status": str(worker_outcome.get("execution_status")),
                        "solution_validity": str(worker_outcome.get("solution_validity")),
                    })

                # Invalidate nodes if source files were modified
                modified_files = worker_outcome.get("modified_files") if isinstance(worker_outcome, dict) else None
                patch_sha = worker_outcome.get("candidate_patch_sha256") if isinstance(worker_outcome, dict) else None
                if modified_files or patch_sha:
                    self.store.update_source_state(
                        candidate_patch_sha256=patch_sha,
                        modified_files=modified_files,
                    )

                # Target nodes
                target_nodes = worker_outcome.get("target_nodes") if isinstance(worker_outcome, dict) else None
                if not target_nodes:
                    target_nodes = (
                        list(self.store.state.opportunities[decision.selected_opportunity].target_nodes)
                        if decision.selected_opportunity in self.store.state.opportunities
                        else []
                    )

                metrics_summary = worker_outcome.get("metrics_summary", {}) if isinstance(worker_outcome, dict) else {}
                exp_id = (
                    worker_outcome.get("experiment_id", f"exp_step_{step_idx + 1:03d}")
                    if isinstance(worker_outcome, dict)
                    else f"exp_step_{step_idx + 1:03d}"
                )

                if isinstance(worker_outcome, dict) and worker_outcome.get("evidence_recorded"):
                    self.store.load()
                    ev = self.store.state.evidence[-1]
                    spent = self.get_current_budget_spent()
                    should_replan = self.controller.should_trigger_replanning(
                        current_budget_spent=spent,
                        critical_evidence=ev.status == "refuted",
                    )
                    new_decision = None
                    if should_replan:
                        packet = self.controller.build_context_packet(spent)
                        new_decision = self._request_and_record_decision(packet, repo_root)
                else:
                    ev, new_decision = self.on_experiment_boundary(
                        experiment_id=exp_id,
                        target_nodes=target_nodes,
                        metrics_summary=metrics_summary,
                        execution_status=exec_status,
                        solution_validity=sol_validity,
                        performance_delta=perf_delta,
                        hypothesis_supported=hyp_supp,
                        hypothesis_id=worker_outcome.get("hypothesis_id") if isinstance(worker_outcome, dict) else None,
                        observations_path=worker_outcome.get("observations_path") if isinstance(worker_outcome, dict) else None,
                        candidate_patch_sha256=patch_sha,
                        current_budget_spent=self.get_current_budget_spent(),
                        repository_root=repo_root,
                    )

                # If replanning was triggered, advance to new decision
                if new_decision is not None:
                    decision = new_decision
                    executed_decisions.append(decision)

        finally:
            self.store.update_budget_spent(self.get_current_budget_spent())
            self.store.record_event(
                "session.finished",
                {
                    "executed_steps": len(executed_decisions) - 1,
                    "budget_spent_sec": self.get_current_budget_spent(),
                },
            )

        return executed_decisions


def build_mock_worker(
    task_config: dict[str, Any] | None = None,
    outcomes: list[dict[str, Any]] | None = None,
    default_outcome: dict[str, Any] | None = None,
) -> Callable[[ReplanningDecision, ContextPacket], dict[str, Any]]:
    """Test fixture worker returning predefined outcomes for unit testing driver loops."""
    step_idx = 0
    outcome_list = list(outcomes) if outcomes else []

    def _mock_worker(decision: ReplanningDecision, packet: ContextPacket) -> dict[str, Any]:
        nonlocal step_idx
        step_idx += 1
        if outcome_list:
            idx = min(step_idx - 1, len(outcome_list) - 1)
            return dict(outcome_list[idx])
        if default_outcome is not None:
            return dict(default_outcome)
        return {
            "experiment_id": f"mock_exp_{step_idx:03d}",
            "execution_status": ExecutionStatus.PASS,
            "solution_validity": SolutionValidity.VALID,
            "performance_delta": PerformanceDelta.IMPROVED,
            "hypothesis_supported": True,
            "actual_runs_count": 1,
            "expected_runs_count": 1,
            "metrics_summary": {"runs_count": 1},
        }

    return _mock_worker


def build_autonomous_worker(
    task_config: dict[str, Any],
    repository_root: Path | str,
    llm_callable: Callable[[str], str] | None = None,
    allow_offline_fallback: bool = False,
    benchmark_runner: Callable[[Path], dict[str, Any]] | None = None,
) -> Callable[[ReplanningDecision, ContextPacket], dict[str, Any]]:
    """Builds a production autonomous coding agent worker that invokes the real LLM/agent,
    inspects the repository, applies changes, executes benchmarks or diagnostic experiments,
    and derives empirical evidence from real execution artifacts.
    """
    repo_path = Path(repository_root).resolve()
    declared_paths = list(task_config.get("repository", {}).get("editable_paths", ["src"]))

    def _worker(decision: ReplanningDecision, packet: ContextPacket) -> dict[str, Any]:
        # 1. Ensure LLM connection is available in formal mode
        effective_callable = llm_callable
        if effective_callable is None and not allow_offline_fallback:
            raise RuntimeError(
                "Formal autonomous worker requires a connected LLM/agent callable (e.g. agy). "
                "No LLM callable was provided."
            )

        step_idx = packet.completed_experiments_count + 1
        discovered_nodes: list[HATSNode] = []
        discovered_opps: list[Opportunity] = []
        hypothesis_id = f"hyp_step_{step_idx:03d}"
        hypothesis_desc = ""

        # 2. Invoke real Coding Agent if callable is present
        if effective_callable is not None:
            worker_prompt = f"""### Coding Agent Autonomous Task Execution
You are the autonomous coding agent worker assigned to execute the following strategic decision:

- Active Opportunity: {decision.selected_opportunity}
- Next Proposed Experiment: {decision.next_experiment}
- Strategic Decision: {decision.decision.value}
- Remaining Budget: {packet.remaining_budget_sec:.1f}s
- Repository Path: {repo_path}
- Declared Editable Paths: {', '.join(declared_paths)}

Please inspect the codebase in the workspace, formulate a concrete algorithmic hypothesis, and identify any new mechanism nodes or opportunities.
Respond with a JSON object matching this schema:
```json
{{
  "hypothesis_id": "{hypothesis_id}",
  "hypothesis": "<concrete explanation of proposed algorithm mechanism or observation>",
  "discovered_nodes": [
    {{
      "id": "<new_node_id>",
      "level": "level_2_mechanism | level_3_code",
      "parent_id": "<subsystem_id>",
      "purpose": "<purpose>",
      "source_refs": ["<file_path>"]
    }}
  ],
  "discovered_opportunities": [
    {{
      "id": "<new_opp_id>",
      "title": "<title>",
      "target_nodes": ["<node_id>"],
      "proposal": "<proposal>",
      "expected_effect": "<expected_effect>",
      "cheapest_discriminating_experiment": "<experiment>"
    }}
  ]
}}
```
"""
            try:
                raw_resp = effective_callable(worker_prompt)
                matches = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw_resp, re.DOTALL)
                text = matches[-1] if matches else ""
                if not text:
                    start = raw_resp.find("{")
                    end = raw_resp.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        text = raw_resp[start : end + 1]
                if text:
                    data = json.loads(text)
                    hypothesis_id = data.get("hypothesis_id", hypothesis_id)
                    hypothesis_desc = data.get("hypothesis", "")
                    if "discovered_nodes" in data and isinstance(data["discovered_nodes"], list):
                        for n in data["discovered_nodes"]:
                            if isinstance(n, dict):
                                try:
                                    discovered_nodes.append(HATSNode.model_validate(n))
                                except Exception:
                                    pass
                            elif isinstance(n, HATSNode):
                                discovered_nodes.append(n)
                    if "discovered_opportunities" in data and isinstance(data["discovered_opportunities"], list):
                        for o in data["discovered_opportunities"]:
                            if isinstance(o, dict):
                                try:
                                    discovered_opps.append(Opportunity.model_validate(o))
                                except Exception:
                                    pass
                            elif isinstance(o, Opportunity):
                                discovered_opps.append(o)
            except Exception as e:
                logging.getLogger("global_planner.worker").warning("Error during worker agent invocation: %s", e)
                if not allow_offline_fallback:
                    raise RuntimeError(f"Coding agent invocation failed: {e}") from e

        # 3. Check for real repository modifications (git diff)
        candidate_patch_sha256 = None
        modified_files: list[str] = []

        if (repo_path / ".git").exists():
            diff_proc = subprocess.run(
                ["git", "diff", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
            )
            if diff_proc.returncode == 0 and diff_proc.stdout.strip():
                candidate_patch = diff_proc.stdout
                candidate_patch_sha256 = hashlib.sha256(candidate_patch.encode("utf-8")).hexdigest()
                names_proc = subprocess.run(
                    ["git", "diff", "--name-only", "HEAD"],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                )
                if names_proc.returncode == 0:
                    modified_files = [line.strip() for line in names_proc.stdout.splitlines() if line.strip()]

        # 4. Check for real benchmark execution artifacts or execute benchmark runner if configured
        bench_result: dict[str, Any] | None = None
        if benchmark_runner is not None:
            try:
                bench_result = benchmark_runner(repo_path)
            except Exception as e:
                logging.getLogger("global_planner.worker").error("Benchmark runner failed: %s", e)

        if bench_result is not None:
            return {
                "experiment_id": f"exp_auto_{step_idx:03d}",
                "execution_status": bench_result.get("execution_status", ExecutionStatus.MISSING),
                "solution_validity": bench_result.get("solution_validity", SolutionValidity.UNVERIFIED),
                "performance_delta": bench_result.get("performance_delta", PerformanceDelta.UNKNOWN),
                "hypothesis_supported": bench_result.get("hypothesis_supported"),
                "hypothesis_id": hypothesis_id,
                "actual_runs_count": bench_result.get("actual_runs_count", 0),
                "expected_runs_count": bench_result.get("expected_runs_count", 1),
                "metrics_summary": bench_result.get("metrics_summary", {}),
                "modified_files": modified_files,
                "candidate_patch_sha256": candidate_patch_sha256,
                "discovered_nodes": discovered_nodes,
                "discovered_opportunities": discovered_opps,
            }

        # If no benchmark runner or no benchmark artifacts were produced:
        # DO NOT FABRICATE SUCCESS OR IMPROVEMENT!
        # Record explicitly as MISSING / UNVERIFIED with 0 runs!
        return {
            "experiment_id": f"exp_auto_{step_idx:03d}",
            "execution_status": ExecutionStatus.MISSING,
            "solution_validity": SolutionValidity.UNVERIFIED,
            "performance_delta": PerformanceDelta.UNKNOWN,
            "hypothesis_supported": None,
            "hypothesis_id": hypothesis_id,
            "actual_runs_count": 0,
            "expected_runs_count": 1,
            "metrics_summary": {
                "runs_count": 0,
                "reason": "No benchmark execution artifacts found in repository",
                "hypothesis": hypothesis_desc,
            },
            "modified_files": modified_files,
            "candidate_patch_sha256": candidate_patch_sha256,
            "discovered_nodes": discovered_nodes,
            "discovered_opportunities": discovered_opps,
        }

    return _worker
