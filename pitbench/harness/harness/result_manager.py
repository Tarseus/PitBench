"""Agent metric extraction and multi-agent result aggregation."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from pitbench.harness.agents.agent_name import AgentName
from pitbench.harness.handlers.trial_handler import TrialHandler
from pitbench.harness.harness.models import TrialResults
from pitbench.harness.utils.pricing import cost_from_tokens


def deep_merge(base: dict, updates: dict) -> dict:
    result = base.copy()
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    if "error" in result and len(result) > 1:
        result.pop("error", None)
    return result


def trajectory_length(sessions_dir: Path | None) -> int:
    if sessions_dir is None or not sessions_dir.exists():
        return 0
    cast_files = list(sessions_dir.glob("*.cast"))
    if not cast_files:
        return 0
    cast_file = max(cast_files, key=lambda path: path.stat().st_mtime)
    interactions = 0
    try:
        for line in cast_file.read_text(encoding="utf-8").splitlines()[1:]:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, list) and len(event) >= 3:
                interactions += 1
    except OSError:
        return 0
    return interactions


def parse_agent_metrics(
    agent_logs_dir: Path,
    model_name: str | None = None,
    sessions_dir: Path | None = None,
) -> tuple[int, int, float, int]:
    prompt_tokens = 0
    completion_tokens = 0
    interactions = 0
    if agent_logs_dir.exists():
        try:
            json_files = list(agent_logs_dir.glob("*.json"))
            if json_files:
                log_file = max(json_files, key=lambda path: path.stat().st_mtime)
                trajectory = json.loads(log_file.read_text())
                if isinstance(trajectory, list):
                    interactions = len(trajectory)
                    for entry in reversed(trajectory):
                        usage = (entry.get("llm_metrics") or {}).get(
                            "accumulated_token_usage", {}
                        )
                        prompt_tokens = int(usage.get("prompt_tokens", 0))
                        completion_tokens = int(usage.get("completion_tokens", 0))
                        if prompt_tokens > 0 or completion_tokens > 0:
                            break
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    if interactions == 0:
        interactions = trajectory_length(sessions_dir)
    return (
        prompt_tokens,
        completion_tokens,
        cost_from_tokens(prompt_tokens, completion_tokens, model_name),
        interactions,
    )


def update_agent_metrics(
    agent_results: TrialResults,
    agent_trial_handler: TrialHandler,
    *,
    agent_model_key: Callable[..., str],
    agent_name: AgentName | None = None,
    agent_import_path: str | None = None,
    model_name: str | None = None,
) -> TrialResults:
    input_tokens, output_tokens, cost, interactions = parse_agent_metrics(
        agent_trial_handler.trial_paths.agent_logging_dir,
        model_name,
        agent_trial_handler.trial_paths.sessions_path,
    )
    if input_tokens > 0 or output_tokens > 0:
        agent_results.total_input_tokens = input_tokens
        agent_results.total_output_tokens = output_tokens
    if agent_results.total_cost is None or cost > 0:
        agent_results.total_cost = float(cost)
    if agent_results.costs_by_agent_model is None:
        agent_results.costs_by_agent_model = {}
    key = agent_model_key(
        agent_name=agent_name,
        model_name=model_name,
        agent_import_path=agent_import_path,
    )
    agent_results.costs_by_agent_model[key] = float(agent_results.total_cost or 0.0)
    if agent_results.agent_metrics is None:
        agent_results.agent_metrics = {}
    agent_results.agent_metrics.update(
        trajectory_length=interactions,
        task_cost_usd=float(agent_results.total_cost or 0.0),
        agent_model_key=key,
    )
    return agent_results


def merge_agent_results(
    trial_handler: TrialHandler, all_agent_results: list[TrialResults]
) -> None:
    if len(all_agent_results) <= 1:
        return
    merged = {}
    for agent_result in all_agent_results:
        merged = deep_merge(merged, agent_result.model_dump())
    result = TrialResults.model_validate(merged)
    result.total_input_tokens = sum(
        item.total_input_tokens or 0 for item in all_agent_results
    )
    result.total_output_tokens = sum(
        item.total_output_tokens or 0 for item in all_agent_results
    )
    costs: dict[str, float] = {}
    for item in all_agent_results:
        for key, value in (item.costs_by_agent_model or {}).items():
            costs[key] = costs.get(key, 0.0) + float(value or 0.0)
    result.costs_by_agent_model = costs
    result.total_cost = float(sum(costs.values()))
    trial_handler.trial_paths.results_path.write_text(result.model_dump_json(indent=4))
