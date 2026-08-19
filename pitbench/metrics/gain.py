from __future__ import annotations


def patch_gain(*, baseline_loss: float, agent_loss: float) -> float:
    return baseline_loss - agent_loss
