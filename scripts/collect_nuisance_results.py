"""Collect nuisance experiment results with a solver-specific protocol class.

Examples:
    python -m scripts.collect_nuisance_results highs run --output <experiment> --cpus 0
    python -m scripts.collect_nuisance_results pyvrp --repository <repo> --output-dir <experiment>
"""

from __future__ import annotations

import argparse
import sys

from pitbench.evaluator.collection import HighsCollection, PyVRPCollection


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    selector = argparse.ArgumentParser(description=__doc__)
    selector.add_argument("solver", choices=("highs", "pyvrp"))
    args = selector.parse_args(argv[:1])
    collectors = {"highs": HighsCollection, "pyvrp": PyVRPCollection}
    collectors[args.solver].main(argv[1:])


if __name__ == "__main__":
    main()
