"""Batch BASE evaluation and live status for a frozen task manifest."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import yaml

_TERMINAL = {"passed", "failed"}


def _profile(path: Path, root: Path) -> tuple[dict, Path, Path, list[str]]:
    path = path.expanduser().resolve()
    settings = yaml.safe_load(path.read_text()) or {}
    if settings.get("phase", "baseline") != "baseline":
        raise ValueError("BASE profile phase must be 'baseline'")
    missing = [key for key in ("manifest", "config", "output") if not settings.get(key)]
    if missing:
        raise ValueError(f"BASE profile is missing: {', '.join(missing)}")

    def resolve(value: str) -> Path:
        candidate = Path(value).expanduser()
        return (candidate if candidate.is_absolute() else root / candidate).resolve()

    manifest = resolve(settings["manifest"])
    output = resolve(settings["output"])
    task_ids = [row["task_id"] for row in json.loads(manifest.read_text())["task_matrix"]]
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("BASE manifest must contain unique task IDs")
    return settings, manifest, output, task_ids


def _snapshot(output: Path, task_ids: list[str]) -> dict:
    states = {}
    for task_id in task_ids:
        path = output / "baseline" / task_id / "status.json"
        if path.is_file():
            states[task_id] = json.loads(path.read_text())
        else:
            states[task_id] = {"task_id": task_id, "status": "not_started"}
    counts = Counter(row.get("status", "missing") for row in states.values())
    return {
        "total": len(task_ids),
        "statuses": dict(sorted(counts.items())),
        "tasks": states,
        "complete": all(row.get("status") in _TERMINAL for row in states.values()),
    }


def _show_status(output: Path, task_ids: list[str], *, watch: bool, live_stream: bool, interval_sec: float) -> None:
    previous = None
    cursors: dict[str, int] = {}
    try:
        while True:
            snapshot = _snapshot(output, task_ids)
            summary = {"total": snapshot["total"], "statuses": snapshot["statuses"]}
            if summary != previous:
                print(f"\n[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] BASE {json.dumps(summary, ensure_ascii=False)}", flush=True)
                previous = summary
            if live_stream:
                for task_id, state in snapshot["tasks"].items():
                    log_path = output / "baseline" / task_id / "judge.log"
                    if not log_path.is_file():
                        continue
                    size = log_path.stat().st_size
                    cursor = cursors.get(task_id, 0)
                    if size < cursor:
                        cursor = 0
                    if size == cursor:
                        continue
                    with log_path.open("r", errors="replace") as stream:
                        stream.seek(cursor)
                        chunk = stream.read()
                        cursors[task_id] = stream.tell()
                    for line in chunk.splitlines():
                        print(f"[{task_id} {state.get('status')} ] {line}", flush=True)
            if not watch or snapshot["complete"]:
                return
            time.sleep(max(0.5, interval_sec))
    except KeyboardInterrupt:
        return


def run_baseline_profile(
    config: Path,
    *,
    root: Path,
    status: bool = False,
    watch: bool = False,
    live_stream: bool = False,
    interval_sec: float = 10,
    workers: int | None = None,
    foreground: bool = False,
) -> None:
    """Run or monitor the shared baseline campaign controller."""
    if live_stream and not watch:
        raise ValueError("--live-stream requires --watch")
    _settings, _, output, task_ids = _profile(config, root.resolve())
    if status or watch:
        _show_status(
            output,
            task_ids,
            watch=watch,
            live_stream=live_stream,
            interval_sec=interval_sec,
        )
        return

    from pitbench.harness.harness.campaign import main as campaign_main

    argv = [
        "--profile",
        str(config.expanduser().resolve()),
        "--phase",
        "baseline",
        "--base-only",
    ]
    if workers is not None:
        argv.extend(["--workers", str(workers)])
    argv.append("--foreground" if foreground else "--background")
    campaign_main(argv, root=root.resolve())
