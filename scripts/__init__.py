"""Grouped command entry points; implementations live in PitBench modules."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable, Sequence


def dispatch(
    commands: dict[str, Callable | str],
    argv: Sequence[str] | None = None,
    *,
    description: str = "",
):
    """Select one command and forward its arguments, including child --help."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("command", choices=commands)
    selected = parser.parse_args(arguments[:1]).command
    target = commands[selected]
    if isinstance(target, str):
        module, _, name = target.partition(":")
        target = importlib.import_module(module)
        for part in name.split("."):
            target = getattr(target, part)
    previous = sys.argv
    sys.argv = [previous[0] + " " + selected, *arguments[1:]]
    try:
        return target()
    finally:
        sys.argv = previous
