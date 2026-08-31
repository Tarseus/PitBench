from __future__ import annotations

from collections.abc import Iterable
from enum import Enum

AGENT_TOOLS_METADATA = "agent_tools.json"


class AgentTool(str, Enum):
    """Optional PitBench commands made available inside an agent task."""

    INSPECT = "inspect"
    WORKSPACE = "workspace"
    BENCH = "bench"
    CHECK = "check"
    VALIDATE = "validate"
    DIFF = "diff"


_DEVELOPMENT_INSTANCE_TOOLS = frozenset({AgentTool.INSPECT, AgentTool.BENCH})
_RUN_DIRECTORY_TOOLS = frozenset({AgentTool.BENCH})


def normalize_agent_tools(
    tools: Iterable[AgentTool | str],
) -> frozenset[AgentTool]:
    return frozenset(AgentTool(tool) for tool in tools)


def agent_tool_names(tools: Iterable[AgentTool | str]) -> list[str]:
    return sorted(tool.value for tool in normalize_agent_tools(tools))


def agent_tools_label(tools: Iterable[AgentTool | str]) -> str:
    names = agent_tool_names(tools)
    return ",".join(names) if names else "none"


def needs_development_instances(tools: Iterable[AgentTool | str]) -> bool:
    return bool(normalize_agent_tools(tools) & _DEVELOPMENT_INSTANCE_TOOLS)


def needs_run_directory(tools: Iterable[AgentTool | str]) -> bool:
    return bool(normalize_agent_tools(tools) & _RUN_DIRECTORY_TOOLS)
