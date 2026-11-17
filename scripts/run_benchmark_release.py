#!/usr/bin/env python3
"""Compatibility entry point for the shared campaign controller."""

import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from pitbench.harness.harness import campaign as _campaign  # noqa: E402

if __name__ == "__main__":
    _campaign.main(root=_REPOSITORY_ROOT)
else:
    # Preserve imported callers and monkeypatch targets without another implementation.
    sys.modules[__name__] = _campaign
