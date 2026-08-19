from __future__ import annotations


def normalized_gap(
    objective: float,
    oracle: float,
    *,
    sense: str = "minimize",
    epsilon: float = 1e-12,
) -> float:
    signed = objective - oracle
    if sense == "maximize":
        signed = -signed
    if sense not in {"minimize", "maximize"}:
        raise ValueError(f"unknown objective sense: {sense}")
    return signed / (abs(oracle) + epsilon)
