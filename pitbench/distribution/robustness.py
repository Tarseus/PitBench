from __future__ import annotations


def gain_retention(*, development_gain: float, shifted_gain: float) -> float:
    if development_gain == 0:
        raise ValueError("gain retention is undefined for zero development gain")
    return shifted_gain / development_gain
