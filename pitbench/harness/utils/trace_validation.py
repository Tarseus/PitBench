"""Evidence checks for recorded operations, separate from benchmark metrics."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from pitbench.harness.utils.agent_trace import read_trace


def inspect_trace(path: Path) -> dict:
    counts = Counter()
    calls = {}
    states = set()
    gaps = []
    execution_finished = False
    for event in read_trace(path):
        kind = event["event"]
        counts[kind] += 1
        data = event["data"]
        if kind in {"workspace.snapshot", "tool.target_snapshot"}:
            states.add(data["state_id"])
        if kind == "execution.finished":
            execution_finished = True
        if kind == "collection.gap":
            gaps.append(data)
        group, _, phase = kind.partition(".")
        if group not in {"tool", "command", "terminal", "mcp"} or phase not in {
            "started",
            "result",
            "failed",
            "finished",
        }:
            continue
        call_id = event["call_id"]
        if call_id is None:
            gaps.append(
                {
                    "source": "operation",
                    "error": "missing_call_id",
                    "event_id": event["event_id"],
                }
            )
            continue
        call = calls.setdefault(call_id, {"call_id": call_id, "kind": group})
        call[phase] = data
    operations = []
    for call in calls.values():
        missing = []
        if "started" not in call:
            missing.append("request")
        if "finished" not in call:
            missing.append("finish")
        if "failed" not in call and (
            "result" not in call or call["result"].get("result_present") is False
        ):
            missing.append("result")
        finished = call.get("finished", {})
        for field in ("before_state_id", "after_state_id"):
            if finished.get(field) not in states:
                missing.append(field)
        if finished.get("missing_request"):
            missing.append("native_request")
        if finished.get("correlation") in {
            "unique_arguments_without_provider_id",
            "unresolved",
        }:
            missing.append("provider_call_id")
        operations.append(
            {
                "call_id": call["call_id"],
                "kind": call["kind"],
                "name": call.get("started", {}).get("name"),
                "producer": call.get("started", {}).get("producer", "pitbench"),
                "backend_call_id": finished.get("backend_call_id"),
                "missing": missing,
            }
        )
    return {
        "schema_version": 1,
        "scope": "recorded operations only; does not prove unexposed native activity absent",
        "execution_finished": execution_finished,
        "observed_operations_paired": bool(operations)
        and all(not op["missing"] for op in operations),
        "native_hooks_observed": counts["native.hook"] > 0,
        "event_counts": dict(counts),
        "collection_gaps": gaps,
        "operations": operations,
    }
