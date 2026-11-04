"""Codex / LLM adapter for global algorithm-space planning."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from global_planner.adapters.base import BasePlannerAdapter
from global_planner.schema import ContextPacket, DecisionType, ReplanningDecision


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
Respond with a concise analysis followed by a JSON block matching this exact schema:

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
  "revisit_condition": "<condition under which deferred opportunities should be reconsidered>"
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
        prompt = self.format_planning_prompt(packet)
        callable_fn = llm_callable or self.default_llm_callable
        if callable_fn is not None:
            raw_response = callable_fn(prompt)
            data = self.extract_decision(raw_response)
            return ReplanningDecision.model_validate(data)

        can_fallback = (
            allow_offline_fallback
            if allow_offline_fallback is not None
            else self.allow_offline_fallback
        )
        if not can_fallback:
            raise RuntimeError(
                "No LLM connection configured for global planning decision. "
                "Formal evaluation requires a connected model callable. "
                "For unit tests or offline simulation, explicitly pass allow_offline_fallback=True."
            )

        # Fallback heuristic decision logic when offline or in test harnesses
        if packet.active_opportunity and packet.recent_evidence:
            last_ev = packet.recent_evidence[-1]
            if last_ev.status == "refuted":
                # Switch to first unexplored opportunity if available
                if packet.unexplored_opportunities:
                    next_opp = packet.unexplored_opportunities[0]
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
