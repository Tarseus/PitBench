"""Base adapter interface for agent-driven global planning."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from global_planner.schema import ContextPacket


class BasePlannerAdapter(ABC):
    """Abstract interface bridging an agent runtime to algorithm-space planning."""

    @abstractmethod
    def format_planning_prompt(self, packet: ContextPacket) -> str:
        """Format the context packet into an actionable planning prompt."""
        raise NotImplementedError

    @abstractmethod
    def extract_decision(self, raw_output: str) -> dict[str, Any]:
        """Extract structured decision payload from the agent's textual response."""
        raise NotImplementedError
