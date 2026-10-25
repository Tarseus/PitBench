"""Compute independently checked offline references for generated instance grids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pitbench.solver_drivers.reference import collect_reference_solutions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--campaign-config", type=Path)
    parser.add_argument("--remote-config", type=Path)
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--sync-loop", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--max-exact-customers", type=int, default=12)
    parser.add_argument("--hgs-binary", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    if args.remote_config:
        from pitbench.evaluator.remote import RemoteReferenceCollection

        if args.deploy:
            result = RemoteReferenceCollection.deploy(
                args.remote_config, args.output_dir
            )
        elif args.sync_loop:
            RemoteReferenceCollection.sync_loop(args.remote_config, args.output_dir)
            return
        elif args.sync:
            result = RemoteReferenceCollection.sync(args.remote_config, args.output_dir)
        elif args.status:
            result = json.loads((args.output_dir / "status.json").read_text())
        else:
            prepared = RemoteReferenceCollection.prepare(
                args.remote_config, args.output_dir
            )
            result = {
                "hosts": len(prepared["hosts"]),
                "assigned_jobs": sum(len(j) for j in prepared["assignments"].values()),
            }
        if isinstance(result.get("hosts"), list):
            result = {
                **result,
                "hosts": [
                    {k: v for k, v in h.items() if k != "status"}
                    | (
                        {"running_jobs": h["status"].get("running_jobs")}
                        if "status" in h
                        else {}
                    )
                    for h in result["hosts"]
                ],
            }
        print(json.dumps(result))
        return
    if args.campaign_config or args.status:
        from pitbench.evaluator.collection import ReferenceCollection

        if args.status:
            result = ReferenceCollection.status(args.output_dir)
        elif args.prepare_only:
            prepared = ReferenceCollection.prepare(
                args.campaign_config, args.output_dir
            )
            result = {
                "jobs": len(prepared["jobs"]),
                "instances": len(prepared["instances"]),
                "cpus": prepared["cpus"],
            }
        else:
            result = ReferenceCollection.run(args.campaign_config, args.output_dir)
        print(json.dumps({k: v for k, v in result.items() if k != "running"}))
        return
    if args.manifest is None or args.seconds is None:
        parser.error("--manifest and --seconds are required outside campaign mode")
    result = collect_reference_solutions(
        args.manifest,
        args.output_dir,
        seconds=args.seconds,
        max_exact_customers=args.max_exact_customers,
        hgs_binary=args.hgs_binary,
        seeds=tuple(args.seeds),
    )
    print(
        json.dumps(
            {key: result[key] for key in ("counts", "attempts", "failed_attempts")}
        )
    )


if __name__ == "__main__":
    main()
