"""Generate configured instance distributions and scales using shared generators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from pitbench.instances import materialize_instance_grid, materialize_instance_suite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = yaml.safe_load(args.config.read_text())
    if "panels" in plan:
        result = materialize_instance_suite(plan, args.output_dir)
        print(json.dumps(result))
        return
    result = materialize_instance_grid(plan, args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "conditions": len(result["conditions"]),
                "instances": result["generated_instances"],
                "status": result["status"],
                "reference_status": result["reference_status"],
            }
        )
    )


if __name__ == "__main__":
    main()
