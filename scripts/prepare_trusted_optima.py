from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from pitbench.instances.generate import (
    prepare_cp_sat_job_shop_panel,
    prepare_trusted_optimum_oracle,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare evaluator-owned trusted optima for a generated panel."
    )
    parser.add_argument("--task-config", type=Path)
    parser.add_argument("--instance-set-name")
    parser.add_argument("--instance-set-config", type=Path)
    parser.add_argument("--private-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-solutions", type=Path)
    parser.add_argument("--job-shop-source-root", type=Path)
    parser.add_argument("--job-shop-instance-ids")
    parser.add_argument("--task-id")
    parser.add_argument("--model-preparer-image")
    parser.add_argument("--visibility", choices=["agent", "judge"])
    parser.add_argument("--reference-source-dirs")
    args = parser.parse_args()
    if args.job_shop_source_root is not None:
        required = {
            "--instance-set-name": args.instance_set_name,
            "--instance-set-config": args.instance_set_config,
            "--model-preparer-image": args.model_preparer_image,
            "--visibility": args.visibility,
            "--job-shop-instance-ids": args.job_shop_instance_ids,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error(f"job-shop panel preparation requires {', '.join(missing)}")
        instance_ids = [
            value.strip()
            for value in args.job_shop_instance_ids.split(",")
            if value.strip()
        ]
        if args.visibility == "judge":
            if (
                args.private_root is None
                or args.output is None
                or (args.task_config is None and args.task_id is None)
            ):
                parser.error(
                    "judge job-shop preparation requires --task-id or --task-config, --private-root, and --output"
                )
            task_id = args.task_id or args.task_config.stem
        else:
            task_id = "agent-development"
        reference_source_dirs = (
            [Path(value) for value in args.reference_source_dirs.split(",")]
            if args.reference_source_dirs
            else []
        )
        if args.visibility == "agent" and (
            args.reference_solutions is None or not reference_source_dirs
        ):
            parser.error("job-shop panel preparation requires --reference-source-dirs")
        panel = prepare_cp_sat_job_shop_panel(
            task_id=task_id,
            instance_set_name=args.instance_set_name,
            visibility=args.visibility,
            source_root=args.job_shop_source_root,
            instance_ids=instance_ids,
            model_preparer_image=args.model_preparer_image,
            instance_output_dir=args.instance_set_config.with_suffix("") / "instances",
            reference_solution_dir=args.reference_solutions,
            instance_set_config_path=args.instance_set_config,
            private_root=args.private_root if args.visibility == "judge" else None,
            oracle_output_path=args.output if args.visibility == "judge" else None,
            reference_solution_source_dirs=reference_source_dirs,
        )
        count = len(panel["records"]) if "records" in panel else len(panel["instances"])
        print(f"prepared {count} job-shop inputs")
        return
    required = {
        "--task-config": args.task_config,
        "--instance-set-name": args.instance_set_name,
        "--instance-set-config": args.instance_set_config,
        "--private-root": args.private_root,
        "--output": args.output,
        "--reference-solutions": args.reference_solutions,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error(f"trusted optimum preparation requires {', '.join(missing)}")
    oracle = prepare_trusted_optimum_oracle(
        task_config_path=args.task_config,
        instance_set_name=args.instance_set_name,
        instance_set_config_path=args.instance_set_config,
        private_root=args.private_root,
        output_path=args.output,
        reference_solution_dir=args.reference_solutions,
    )
    print(
        f"prepared {len(oracle['records'])} trusted optima; "
        f"sha256={hashlib.sha256(args.output.read_bytes()).hexdigest()}"
    )


if __name__ == "__main__":
    main()
