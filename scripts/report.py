"""Shared reporting entry point for retained experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pitbench.schema.task import PitBenchTask


def _controls() -> None:
    import argparse
    from pathlib import Path

    from pitbench.evaluator.artifacts import selected_output_attempt
    from pitbench.metrics.performance_report import ReleaseControlAnalysis

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument(
        "--reference-kind",
        choices=("solver_release", "human_pr"),
        default="solver_release",
    )
    parser.add_argument(
        "--exclude-first-seed",
        action="store_true",
        help="report the public development seeds not used by a one-seed pilot",
    )
    args = parser.parse_args()
    output_dir = selected_output_attempt(args.output_dir.resolve())
    if args.baseline_only:
        result = ReleaseControlAnalysis.analyze_baseline_qualification(
            args.manifest.resolve(), output_dir
        )
        output = output_dir / "baseline_analysis.json"
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "conclusion": result["conclusion"],
                    "expected_runs": result["expected_runs"],
                    "missing_runs": len(result["missing_runs"]),
                    "failed_runs": len(result["failed_runs"]),
                    "analysis_path": str(output),
                }
            )
        )
        return
    result = ReleaseControlAnalysis.analyze(
        args.manifest.resolve(),
        output_dir,
        reference_kind=args.reference_kind,
        exclude_first_seed=args.exclude_first_seed,
    )
    output = output_dir / "analysis.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "conclusion": result["conclusion"],
                "expected_pairs": result["expected_pairs"],
                "missing_pairs": len(result["missing_pairs"]),
                "failed_pairs": len(result["failed_pairs"]),
                "analysis_path": str(output),
            }
        )
    )


def _agent_reference() -> None:
    import argparse
    from pathlib import Path

    from pitbench.metrics.performance_report import ReleaseControlAnalysis

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-analysis", type=Path, required=True)
    parser.add_argument("--human-manifest", type=Path, required=True)
    parser.add_argument("--agent-observations", type=Path, required=True)
    parser.add_argument("--agent-task", type=Path, required=True)
    parser.add_argument("--budget-sec", type=float, required=True)
    parser.add_argument("--instance-set", default="judge_id")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = ReleaseControlAnalysis.compare_agent_reference(
        args.human_analysis,
        args.human_manifest,
        args.agent_observations,
        args.agent_task,
        budget_sec=args.budget_sec,
        instance_set=args.instance_set,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("expected_pairs", "conclusion")}))


def _noise() -> None:
    import argparse
    from pathlib import Path

    from pitbench.metrics.performance_report import calibrate_evaluation_noise

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = calibrate_evaluation_noise(json.loads(args.manifest.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"no-op false directions: {result['control_false_direction_count']}/{result['control_count']} raw, {result['public_control_false_direction_count']}/{result['control_count']} after confirmation; independent Base-pair false directions: {result['baseline_pair_false_direction_count']}/{result['baseline_pair_count']}"
    )


def _patches() -> None:
    import argparse
    from pathlib import Path

    from pitbench.repositories.pr_curation import analyze_patch_mechanisms

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-index", type=Path, required=True)
    parser.add_argument("--coding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_patch_mechanisms(args.attempt_index, args.coding)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("attempts", "class_counts")}))


def _nuisance() -> None:
    import argparse
    from pathlib import Path

    from pitbench.metrics.nuisance_report import report_nuisance_results

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = report_nuisance_results(args.source, args.output)
    print(json.dumps({key: value for key, value in summary.items() if key != "groups"}))


def _transfer_args() -> argparse.Namespace:
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
    from pitbench.schema.task import PitBenchTask

    required = {"problem_variant", "formulation_id", "conditions"}
    missing = required - set(panel)
    if missing:
        raise ValueError(
            f"controlled transfer panel is missing fields: {sorted(missing)}"
        )
    if panel.get("task_id") not in {None, task.task_id}:
        raise ValueError("controlled transfer panel belongs to another task")
    payload = task.model_dump(mode="json")
    payload["problem_variant"] = panel["problem_variant"]
    payload["formulation_id"] = panel["formulation_id"]
    payload["evaluation"]["controlled_transfer"] = {"conditions": panel["conditions"]}
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
        raise ValueError(
            f"controlled transfer panel field {hash_field} must be a string"
        )
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(
            f"controlled transfer panel hash mismatch for {path}: "
            f"{actual_hash} != {expected_hash}"
        )


def _transfer() -> None:
    from pathlib import Path

    from pitbench.evaluator.storage import ObservationStore
    from pitbench.metrics.controlled_transfer_report import (
        compute_controlled_transfer_report,
    )
    from pitbench.schema.observation import ExpectedRunGrid
    from pitbench.schema.task import PitBenchTask

    args = _transfer_args()
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


def main(argv=None):
    if __package__ in (None, ""):
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts import dispatch

    return dispatch(
        {
            "controls": _controls,
            "agent-reference": _agent_reference,
            "noise": _noise,
            "patches": _patches,
            "nuisance": _nuisance,
            "transfer": _transfer,
        },
        argv,
        description="Report retained evidence without running solvers.",
    )


if __name__ == "__main__":
    main()
