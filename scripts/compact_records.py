"""Compact retained records without changing benchmark results or evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pitbench.harness.utils.artifact_storage import (
    archive_directory,
    deduplicate_objects,
    restore_archive,
    trace_directories,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("deduplicate", "archive-traces", "archive", "restore"):
        command = commands.add_parser(name)
        command.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    def report(value):
        print(json.dumps(value, ensure_ascii=False), flush=True)

    if args.command == "deduplicate":
        report(deduplicate_objects(trace_directories(args.paths), progress=report))
    elif args.command == "archive-traces":
        for trace in trace_directories(args.paths):
            # Coverage is written after the writer closes, including failed runs.
            # Interrupted traces without this completion marker remain unpacked.
            if (trace / "coverage.json").is_file():
                result = archive_directory(trace)
                report(result)
                if not result["source_removed"]:
                    raise SystemExit(1)
    elif args.command == "archive":
        for path in args.paths:
            result = archive_directory(path)
            report(result)
            if not result["source_removed"]:
                raise SystemExit(1)
    else:
        for path in args.paths:
            report({"restored": str(restore_archive(path))})


if __name__ == "__main__":
    main()
