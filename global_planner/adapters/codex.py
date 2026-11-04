"""Codex / LLM adapter for global algorithm-space planning."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

from global_planner.adapters.base import BasePlannerAdapter
from global_planner.schema import ContextPacket, DecisionType, ReplanningDecision


def create_default_llm_callable(
    model_name: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    backend: str | None = None,
    timeout_sec: float = 300.0,
) -> Callable[[str], str] | None:
    """Creates a real LLM callable supporting Antigravity CLI (agy), LiteLLM/OpenAI, or Codex CLI."""
    import shutil
    import subprocess

    selected_backend = backend or os.environ.get("PITBENCH_PLANNER_BACKEND")
    if not selected_backend:
        # Default automatically to agy (Antigravity CLI) if installed on system
        if shutil.which("agy"):
            selected_backend = "antigravity"

    if selected_backend in ("agy", "antigravity"):
        agy_bin = shutil.which("agy") or "agy"
        effective_model = model_name or os.environ.get("PITBENCH_PLANNER_MODEL") or "gemini-3.8-flash-low"
        effective_effort = os.environ.get("PITBENCH_PLANNER_EFFORT", "low")

        def _agy_call(prompt: str) -> str:
            cmd = [
                agy_bin,
                "--dangerously-skip-permissions",
                "--disable-slash-commands",
                "--model",
                effective_model,
                "--effort",
                effective_effort,
                "--print",
                prompt,
            ]
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            if res.returncode != 0:
                raise RuntimeError(
                    f"agy execution failed (exit code {res.returncode}): {res.stderr or res.stdout}"
                )
            return res.stdout

        return _agy_call

    if selected_backend == "codex":
        codex_bin = shutil.which("codex") or "codex"

        def _codex_call(prompt: str) -> str:
            res = subprocess.run(
                [codex_bin, "exec", "--ephemeral", "--dangerously-bypass-approvals-and-sandbox", prompt],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            if res.returncode != 0:
                raise RuntimeError(
                    f"codex execution failed (exit code {res.returncode}): {res.stderr or res.stdout}"
                )
            return res.stdout

        return _codex_call

    effective_model = model_name or os.environ.get("PITBENCH_PLANNER_MODEL")
    effective_key = api_key
    effective_base = api_base or os.environ.get("OPENAI_BASE_URL") or os.environ.get("PORTKEY_API_BASE")

    if not effective_model:
        if os.environ.get("DEEPSEEK_API_KEY"):
            effective_model = "deepseek/deepseek-chat"
            effective_key = effective_key or os.environ.get("DEEPSEEK_API_KEY")
        elif os.environ.get("OPENAI_API_KEY"):
            effective_model = "gpt-4o"
            effective_key = effective_key or os.environ.get("OPENAI_API_KEY")
        elif os.environ.get("PORTKEY_API_KEY"):
            effective_model = "openai/gpt-4o"
            effective_key = effective_key or os.environ.get("PORTKEY_API_KEY")

    if not effective_key:
        if effective_model and "deepseek" in effective_model:
            effective_key = os.environ.get("DEEPSEEK_API_KEY")
        elif effective_model and ("openai" in effective_model or "gpt" in effective_model):
            effective_key = os.environ.get("OPENAI_API_KEY")

    if not effective_model or not effective_key:
        return None

    def _call(prompt: str) -> str:
        import litellm
        resp = litellm.completion(
            model=effective_model,
            messages=[{"role": "user", "content": prompt}],
            api_key=effective_key,
            api_base=effective_base,
            timeout=timeout_sec,
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""

    return _call


class CodexPlannerAdapter(BasePlannerAdapter):
    """Formats structured planning prompts and extracts replanning decisions for Codex."""

    def __init__(
        self,
        default_llm_callable: Callable[[str], str] | None = None,
        allow_offline_fallback: bool = False,
    ):
        self.default_llm_callable = default_llm_callable
        self.allow_offline_fallback = allow_offline_fallback

    def format_planning_prompt(self, packet: ContextPacket) -> str:
        recent_ev_text = "None."
        if packet.recent_evidence:
            recent_ev_text = "\n".join(
                f"- [Evidence {e.id}] Exp: {e.experiment_id} | Status: {e.status.upper()} | "
                f"Execution: {e.execution_status.value.upper()} | "
                f"Validity: {e.solution_validity.value.upper()} | "
                f"Performance: {e.performance_delta.value.upper()} | "
                f"Mechanism Verified: {e.hypothesis_supported} | "
                f"Metrics: {json.dumps(e.metrics_summary)}"
                for e in packet.recent_evidence
            )

        affected_nodes_text = "None."
        if packet.affected_nodes:
            affected_nodes_text = "\n".join(
                f"- [Node: {n.id}] (Level: {n.level.value}, Parent: {n.parent_id or 'root'})\n"
                f"  Purpose: {n.purpose}\n"
                f"  Source Refs: {', '.join(n.source_refs) or 'None'}\n"
                f"  Reads: {', '.join(n.reads) or 'None'} | Writes: {', '.join(n.writes) or 'None'}\n"
                f"  Preconditions: {', '.join(n.preconditions) or 'None'}\n"
                f"  Invariants: {', '.join(n.invariants) or 'None'}\n"
                f"  Effects: {'; '.join(f'{ef.target_metric} ({ef.predicted_direction}) [{ef.status}]' for ef in n.effects) or 'None'}"
                for n in packet.affected_nodes
            )

        active_opp_text = "None currently selected."
        if packet.active_opportunity:
            opp = packet.active_opportunity
            active_opp_text = (
                f"- [{opp.id}] {opp.title}\n"
                f"  Target Nodes: {', '.join(opp.target_nodes)}\n"
                f"  Proposal: {opp.proposal}\n"
                f"  Expected Effect: {opp.expected_effect}\n"
                f"  Cheapest Experiment: {opp.cheapest_discriminating_experiment}\n"
                f"  Status: {opp.status.upper()}"
            )

        unexplored_text = "None available."
        if packet.unexplored_opportunities:
            unexplored_text = "\n".join(
                f"- [{opp.id}] {opp.title} (Target: {', '.join(opp.target_nodes)})\n"
                f"  Proposal: {opp.proposal}\n"
                f"  Expected Effect: {opp.expected_effect}"
                for opp in packet.unexplored_opportunities
            )

        prompt = f"""### Algorithm-Space Global Replanning Request

Review the current algorithm state transition model and empirical evidence.
Make an explicit strategic decision for the next development phase.

#### 1. Current Goal & Remaining Budget
- Task: {packet.task_id}
- Objective: {packet.goal_statement}
- Remaining Budget: {packet.remaining_budget_sec:.1f}s (Completed Experiments: {packet.completed_experiments_count})

#### 2. Active Opportunity Under Exploration
{active_opp_text}

#### 3. Recent Empirical Evidence & Controlled Observations
{recent_ev_text}

#### 4. Affected Algorithm-State & Transition Nodes (HATS Level 1-3)
{affected_nodes_text}

#### 5. Unexplored Opportunities
{unexplored_text}

#### Decision Contract
Respond with a concise analysis followed by a JSON block matching this schema:

```json
{{
  "decision": "continue | switch | combine | stop",
  "selected_opportunity": "<opportunity_id>",
  "evidence_refs": ["<evidence_id>", ...],
  "alternatives_considered": [
    {{"id": "<alt_opp_id>", "defer_reason": "<why deferred>"}}
  ],
  "next_experiment": "<concrete discriminating experiment to run next>",
  "switch_justification": "<required if switching: why previous opportunity is abandoned or saturated>",
  "revisit_condition": "<condition under which deferred opportunities should be reconsidered>",
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
        return prompt.strip()

    def extract_decision(self, raw_output: str) -> dict[str, Any]:
        """Extract JSON replanning decision from LLM output, handling markdown fences."""
        matches = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw_output, re.DOTALL)
        if matches:
            text = matches[-1]
        else:
            start = raw_output.find("{")
            end = raw_output.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = raw_output[start : end + 1]
            else:
                raise ValueError("No valid JSON decision object found in agent response")

        data = json.loads(text)
        return data

    def request_decision(
        self,
        packet: ContextPacket,
        llm_callable: Callable[[str], str] | None = None,
        allow_offline_fallback: bool | None = None,
    ) -> ReplanningDecision:
        """Invokes the planning agent via LLM callable, or applies deterministic fallback if allowed."""
        can_fallback = (
            allow_offline_fallback
            if allow_offline_fallback is not None
            else self.allow_offline_fallback
        )

        callable_fn = llm_callable or self.default_llm_callable
        if callable_fn is None and not can_fallback:
            # In formal mode, resolve real LLM connection from environment/credentials
            callable_fn = create_default_llm_callable()

        if callable_fn is not None:
            prompt = self.format_planning_prompt(packet)
            raw_response = callable_fn(prompt)
            data = self.extract_decision(raw_response)
            return ReplanningDecision.model_validate(data)

        if not can_fallback:
            raise RuntimeError(
                "No LLM connection configured for global planning decision. "
                "Formal evaluation requires a connected model callable. "
                "Please configure DEEPSEEK_API_KEY, OPENAI_API_KEY, or pass --model/--api-key, "
                "or explicitly pass allow_offline_fallback=True for offline testing."
            )

        # Fallback heuristic decision logic when offline or in test harnesses
        if packet.active_opportunity and packet.recent_evidence:
            last_ev = packet.recent_evidence[-1]
            if last_ev.status == "refuted":
                # Switch to first unexplored opportunity if available, prioritizing newly discovered or targeted opps
                if packet.unexplored_opportunities:
                    candidate_opps = [
                        opp for opp in packet.unexplored_opportunities
                        if any(nid in opp.target_nodes for nid in packet.affected_nodes)
                        or opp.id.startswith("opp_custom")
                        or opp.id.startswith("opp_optimize")
                    ]
                    next_opp = candidate_opps[0] if candidate_opps else packet.unexplored_opportunities[0]
                    return ReplanningDecision(
                        decision=DecisionType.SWITCH,
                        selected_opportunity=next_opp.id,
                        evidence_refs=[last_ev.id],
                        switch_justification=f"Previous active opportunity was refuted by evidence {last_ev.id}",
                        next_experiment=next_opp.cheapest_discriminating_experiment,
                        alternatives_considered=[{"id": packet.active_opportunity.id, "defer_reason": "refuted"}],
                    )

            if last_ev.status == "supported":
                return ReplanningDecision(
                    decision=DecisionType.CONTINUE,
                    selected_opportunity=packet.active_opportunity.id,
                    evidence_refs=[last_ev.id],
                    next_experiment=f"Differential verification of {packet.active_opportunity.id} across full development seeds",
                )

        # Default start or continue with active/first opportunity
        target_opp = packet.active_opportunity or (
            packet.unexplored_opportunities[0] if packet.unexplored_opportunities else None
        )
        if not target_opp:
            return ReplanningDecision(
                decision=DecisionType.STOP,
                selected_opportunity="none",
                next_experiment="none",
            )

        return ReplanningDecision(
            decision=DecisionType.CONTINUE,
            selected_opportunity=target_opp.id,
            next_experiment=target_opp.cheapest_discriminating_experiment,
        )
