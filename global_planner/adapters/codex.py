"""Codex / LLM adapter for global algorithm-space planning."""

from __future__ import annotations

import json
import re
from typing import Any

from global_planner.adapters.base import BasePlannerAdapter
from global_planner.schema import ContextPacket


class CodexPlannerAdapter(BasePlannerAdapter):
    """Formats structured planning prompts and extracts replanning decisions for Codex."""

    def format_planning_prompt(self, packet: ContextPacket) -> str:
        recent_ev_text = "None"
        if packet.recent_evidence:
            recent_ev_text = "\n".join(
                f"- [Evidence {e.id}] Exp: {e.experiment_id} | Status: {e.status.upper()} | "
                f"Metrics: {json.dumps(e.metrics_summary)}"
                for e in packet.recent_evidence
            )

        affected_nodes_text = "None"
        if packet.affected_nodes:
            affected_nodes_text = "\n".join(
                f"- [Node: {n.id}] Level: {n.level.value} | Purpose: {n.purpose}\n"
                f"  Preconditions: {', '.join(n.preconditions) or 'None'}\n"
                f"  Invariants: {', '.join(n.invariants) or 'None'}"
                for n in packet.affected_nodes
            )

        active_opp_text = "None currently selected."
        if packet.active_opportunity:
            opp = packet.active_opportunity
            active_opp_text = (
                f"- [{opp.id}] {opp.title}\n"
                f"  Proposal: {opp.proposal}\n"
                f"  Expected Effect: {opp.expected_effect}\n"
                f"  Status: {opp.status.upper()}"
            )

        unexplored_text = "None available."
        if packet.unexplored_opportunities:
            unexplored_text = "\n".join(
                f"- [{opp.id}] {opp.title}: {opp.proposal} (Expected: {opp.expected_effect})"
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

#### 3. Recent Empirical Evidence & Test Observations
{recent_ev_text}

#### 4. Affected Algorithm-State & Transition Nodes
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
        # Find json block
        matches = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw_output, re.DOTALL)
        if matches:
            text = matches[-1]
        else:
            # Fallback to brace search
            start = raw_output.find("{")
            end = raw_output.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = raw_output[start : end + 1]
            else:
                raise ValueError("No valid JSON decision object found in agent response")

        data = json.loads(text)
        return data
