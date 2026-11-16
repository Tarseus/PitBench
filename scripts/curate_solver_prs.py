"""CLI for the maintained repository-curation interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pitbench.repositories import pr_curation as operations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (
        "inventory",
        "harvest",
        "import",
        "audit",
        "queue",
        "triage",
        "type-audit",
        "prepare-review-batch",
        "review-batch-status",
        "cache-local-diffs",
        "cache-git-diffs",
        "screening-audit",
        "enrich",
        "checkout",
        "screen",
        "source-verify",
        "workload-evidence",
        "public-correctness",
        "baseline-evidence",
        "objective-register",
        "objective-qualify",
        "integrate",
    ):
        subparser = commands.add_parser(name)
        subparser.add_argument("--catalog", type=Path, required=True)
        if name == "inventory":
            subparser.add_argument("--output-dir", type=Path, required=True)
            subparser.add_argument("--query-log", type=Path, required=True)
            subparser.add_argument("--repository", action="append")
            subparser.add_argument("--token-env", default="GITHUB_TOKEN")
        elif name == "harvest":
            subparser.add_argument("--output", type=Path, required=True)
            subparser.add_argument("--query-log", type=Path, required=True)
            subparser.add_argument("--repository", action="append")
            subparser.add_argument("--query", action="append")
            subparser.add_argument("--token-env", default="GITHUB_TOKEN")
        elif name == "import":
            subparser.add_argument("--input", type=Path, required=True, action="append")
            subparser.add_argument("--output", type=Path, required=True)
            subparser.add_argument("--repository-root", type=Path, default=Path.cwd())
        elif name == "audit":
            subparser.add_argument("--registry", type=Path, required=True)
            subparser.add_argument("--output", type=Path)
            subparser.add_argument("--repository-root", type=Path, default=Path.cwd())
        elif name == "queue":
            subparser.add_argument("--registry", type=Path, required=True)
            subparser.add_argument("--limit", type=int, default=100)
            subparser.add_argument("--output", type=Path, required=True)
        elif name == "type-audit":
            subparser.add_argument("--registry", type=Path, required=True)
            subparser.add_argument("--output-dir", type=Path, required=True)
        elif name in {"triage", "screening-audit"}:
            subparser.add_argument("--registry", type=Path, required=True)
            subparser.add_argument(
                "--details", type=Path, action="append", required=True
            )
            if name == "triage":
                subparser.add_argument("--limit", type=int, default=100)
                subparser.add_argument("--audit-fraction", type=float, default=0.1)
                subparser.add_argument(
                    "--scope-status",
                    action="append",
                    choices=("task_backed", "approved_source", "pending_domain_review"),
                    help="Schedule reviews from these catalog states; other PRs remain eligible",
                )
                subparser.add_argument(
                    "--review-order",
                    choices=("coverage", "throughput"),
                    default="coverage",
                    help="Balance source coverage or prioritize explicit optimization signals",
                )
            subparser.add_argument("--output", type=Path, required=True)
        elif name == "prepare-review-batch":
            subparser.add_argument("--registry", type=Path, required=True)
            subparser.add_argument(
                "--details", type=Path, action="append", required=True
            )
            subparser.add_argument("--queue", type=Path, required=True)
            subparser.add_argument("--limit", type=int, default=100)
            subparser.add_argument("--output-dir", type=Path, required=True)
            subparser.add_argument("--repository-root", type=Path, default=Path.cwd())
        elif name == "review-batch-status":
            subparser.add_argument("--registry", type=Path, required=True)
            subparser.add_argument("--manifest", type=Path, required=True)
            subparser.add_argument("--output", type=Path, required=True)
            subparser.add_argument("--repository-root", type=Path, default=Path.cwd())
        elif name in {"cache-local-diffs", "cache-git-diffs"}:
            subparser.add_argument("--registry", type=Path, required=True)
            subparser.add_argument("--manifest", type=Path, required=True)
            subparser.add_argument(
                "--repository-checkout", action="append", required=True
            )
            subparser.add_argument("--output-dir", type=Path, required=True)
            subparser.add_argument("--repository-root", type=Path, default=Path.cwd())
            subparser.add_argument("--fetch-missing", action="store_true")
            subparser.add_argument("--retry-unavailable", action="store_true")
            subparser.add_argument("--limit", type=int, default=1000000)
            subparser.add_argument("--git-timeout-sec", type=float, default=120)
        elif name == "enrich":
            subparser.add_argument("--registry", type=Path, required=True)
            subparser.add_argument("--queue", type=Path, required=True)
            subparser.add_argument("--limit", type=int, default=100)
            subparser.add_argument("--output", type=Path, required=True)
            subparser.add_argument("--token-env", default="GITHUB_TOKEN")
        elif name == "checkout":
            subparser.add_argument("--registry", type=Path, required=True)
            subparser.add_argument("--repository", required=True)
            subparser.add_argument("--pr-number", type=int, required=True)
            subparser.add_argument("--repository-checkout", type=Path, required=True)
            subparser.add_argument("--output-dir", type=Path, required=True)
            subparser.add_argument("--submodule-source", action="append")
            subparser.add_argument("--audit-output", type=Path, required=True)
        elif name == "screen":
            subparser.add_argument("--registry", type=Path, required=True)
            subparser.add_argument("--review-file", type=Path, action="append")
            subparser.add_argument("--review-dir", type=Path)
            subparser.add_argument("--repository-root", type=Path, default=Path.cwd())
        elif name in {
            "workload-evidence",
            "public-correctness",
            "baseline-evidence",
            "objective-register",
            "objective-qualify",
            "integrate",
        }:
            subparser.add_argument("--registry", type=Path, required=True)
            subparser.add_argument("--repository", required=True)
            subparser.add_argument("--pr-number", type=int, required=True)
            subparser.add_argument("--repository-root", type=Path, default=Path.cwd())
            subparser.add_argument("--output", type=Path, required=True)
            if name == "workload-evidence":
                subparser.add_argument("--manifest", type=Path, required=True)
                subparser.add_argument("--activation", type=Path, required=True)
            elif name == "public-correctness":
                subparser.add_argument("--pairs", type=Path, required=True)
                subparser.add_argument("--ledger-output", type=Path, required=True)
                subparser.add_argument("--checker-artifact", type=Path, required=True)
                subparser.add_argument("--test-results", type=Path, required=True)
                subparser.add_argument("--semantics", type=Path)
            elif name == "baseline-evidence":
                subparser.add_argument("--manifest", type=Path, required=True)
                subparser.add_argument("--analysis", type=Path, required=True)
                subparser.add_argument("--runs", type=Path, required=True)
            elif name == "objective-register":
                pass
            elif name == "objective-qualify":
                subparser.add_argument("--run-directory", type=Path, required=True)
            else:
                subparser.add_argument("--agent-identity", type=Path, required=True)
                subparser.add_argument("--checkout-audit", type=Path, required=True)
                subparser.add_argument("--run-directory", type=Path, required=True)
            if name == "integrate":
                subparser.add_argument("--dry-run-exit-code", type=int, required=True)
                subparser.add_argument("--dry-run-exit-note")
        else:
            subparser.add_argument("--registry", type=Path, required=True)
            subparser.add_argument("--repository", required=True)
            subparser.add_argument("--pr-number", type=int, required=True)
            subparser.add_argument("--repository-checkout", type=Path, required=True)
            subparser.add_argument("--editable-root", action="append", required=True)
            subparser.add_argument("--patch-output", type=Path, required=True)
            subparser.add_argument("--audit-output", type=Path, required=True)
            subparser.add_argument("--intent-note")
            subparser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        {
            "inventory": operations._inventory,
            "harvest": operations._harvest,
            "import": operations._import,
            "audit": operations._audit,
            "queue": operations._queue,
            "triage": operations._triage,
            "type-audit": operations._type_audit,
            "prepare-review-batch": operations._prepare_review_batch,
            "review-batch-status": operations._review_batch_status,
            "cache-local-diffs": operations._cache_local_diffs,
            "cache-git-diffs": operations._cache_local_diffs,
            "screening-audit": operations._screening_audit,
            "enrich": operations._enrich,
            "checkout": operations._checkout,
            "screen": operations._screen,
            "source-verify": operations._source_verify,
            "workload-evidence": operations._workload_evidence,
            "public-correctness": operations._public_correctness,
            "baseline-evidence": operations._baseline_evidence,
            "objective-register": operations._objective_register,
            "objective-qualify": operations._objective_qualify,
            "integrate": operations._integration_evidence,
        }[args.command](args)
    except Exception as error:
        print(f"PR curation failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
