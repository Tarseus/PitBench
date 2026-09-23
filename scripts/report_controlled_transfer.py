from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pitbench.evaluator.storage import ObservationStore
from pitbench.metrics.controlled_transfer_report import (
    compute_controlled_transfer_report,
)
from pitbench.schema.observation import ExpectedRunGrid
from pitbench.schema.task import PitBenchTask


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute a Controlled Transfer report from an immutable trial panel."
    )
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--panel-config", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--expected-run-grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _task_with_controlled_transfer(
    task: PitBenchTask,
    panel: dict[str, object],
) -> PitBenchTask:
    required = {"problem_variant", "formulation_id", "conditions"}
    missing = required - set(panel)
    if missing:
        raise ValueError(f"controlled transfer panel is missing fields: {sorted(missing)}")
    if panel.get("task_id") not in {None, task.task_id}:
        raise ValueError("controlled transfer panel belongs to another task")
    payload = task.model_dump(mode="json")
    payload["problem_variant"] = panel["problem_variant"]
    payload["formulation_id"] = panel["formulation_id"]
    payload["evaluation"]["controlled_transfer"] = {
        "conditions": panel["conditions"]
    }
    return PitBenchTask.model_validate(payload)


def _verify_panel_input(
    panel: dict[str, object],
    *,
    path: Path,
    hash_field: str,
) -> None:
    expected_hash = panel.get(hash_field)
    if expected_hash is None:
        return
    if not isinstance(expected_hash, str):
        raise ValueError(f"controlled transfer panel field {hash_field} must be a string")
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(
            f"controlled transfer panel hash mismatch for {path}: "
            f"{actual_hash} != {expected_hash}"
        )


def main() -> None:
    args = _parse_args()
    task = PitBenchTask.from_yaml(args.task_config)
    panel = json.loads(args.panel_config.read_text())
    if not isinstance(panel, dict):
        raise ValueError("controlled transfer panel must be a JSON object")
    _verify_panel_input(
        panel,
        path=args.task_config,
        hash_field="source_task_config_sha256",
    )
    _verify_panel_input(
        panel,
        path=args.observations,
        hash_field="source_observations_sha256",
    )
    _verify_panel_input(
        panel,
        path=args.expected_run_grid,
        hash_field="source_expected_run_grid_sha256",
    )
    candidate_patch = panel.get("candidate_patch")
    if isinstance(candidate_patch, str):
        _verify_panel_input(
            panel,
            path=Path(candidate_patch),
            hash_field="candidate_patch_sha256",
        )
    configured_task = _task_with_controlled_transfer(task, panel)
    report = compute_controlled_transfer_report(
        ObservationStore.read(args.observations),
        task=configured_task,
        expected_run_grid=ExpectedRunGrid.model_validate_json(
            args.expected_run_grid.read_text()
        ),
    )
    if report is None:
        raise ValueError("controlled transfer panel did not configure a report")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2) + "\n")


if __name__ == "__main__":
    main()
