"""Render retained nuisance observations without selecting a robustness statistic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pitbench.metrics.nuisance_report import report_nuisance_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = report_nuisance_results(args.source, args.output)
    print(json.dumps({key: value for key, value in summary.items() if key != "groups"}))


if __name__ == "__main__":
    main()
