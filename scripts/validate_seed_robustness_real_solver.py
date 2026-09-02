from __future__ import annotations

import argparse
import os
import re
import subprocess
from dataclasses import replace
from pathlib import Path

from adapters.pitbench.adapter import PitBenchAdapter
from pitbench.evaluator.judge import JudgePlan, LocalProcessJudge
from pitbench.evaluator.private_assets import PrivateAssetResolver
from pitbench.evaluator.storage import ObservationStore
from pitbench.metrics.seed_robustness_validation import (
    REFERENCE_SEED_COUNT,
    TEST_LIST_COUNT,
    TEST_SEED_COUNT,
    VALIDATION_GENERATION_SEED,
    RealValidationSeeds,
    analyze_real_seed_validation,
    generate_real_validation_seeds,
)
from pitbench.schema.observation import CodeState, RunObservation
from pitbench.schema.task import InstanceSetKind, PitBenchTask

ROOT = Path(__file__).resolve().parents[1]


def _run_identity(
    observation: RunObservation,
) -> tuple[str, str, CodeState, int, float]:
    return (
        observation.instance_set,
        observation.instance_id,
        observation.code_state,
        observation.solver_seed,
        observation.budget_sec,
    )


def _write_json(path: Path, payload: str) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(payload)
    temporary_path.replace(path)


def _load_checkpoint(path: Path) -> list[RunObservation]:
    if not path.exists():
        return []

    lines = path.read_text().splitlines()
    observations: list[RunObservation] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            observations.append(RunObservation.model_validate_json(line))
        except ValueError:
            if line_number == len(lines):
                checkpoint_text = "".join(
                    f"{observation.model_dump_json()}\n"
                    for observation in observations
                )
                _write_json(path, checkpoint_text)
                break
            raise ValueError(
                f"checkpoint contains an invalid line at {line_number}"
            ) from None

    completed_runs = {_run_identity(observation) for observation in observations}
    if len(completed_runs) != len(observations):
        raise ValueError("checkpoint contains duplicate observations")
    return observations


def _show_progress(message: str) -> None:
    if message.startswith("Judge progress:"):
        match = re.search(r"solver runs (\d+)/(\d+)", message)
        if match is not None and int(match.group(1)) % 50:
            return
    print(f"PITBENCH_PROGRESS {message}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Seed Robustness with real PyVRP outcomes."
    )
    parser.add_argument(
        "--task-config",
        type=Path,
        default=ROOT / "configs/tasks/pyvrp_v0_14_0.yaml",
    )
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--judge-image")
    parser.add_argument("--judge-cpus", type=float, default=8.0)
    parser.add_argument("--judge-memory", default="8g")
    parser.add_argument("--private-root", type=Path, default=ROOT / "private")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--instance-limit", type=int, default=10)
    parser.add_argument("--parallel-runs", type=int, default=8)
    parser.add_argument(
        "--reference-seed-count", type=int, default=REFERENCE_SEED_COUNT
    )
    parser.add_argument("--test-seed-count", type=int, default=TEST_SEED_COUNT)
    parser.add_argument("--test-list-count", type=int, default=TEST_LIST_COUNT)
    parser.add_argument(
        "--generation-seed", type=int, default=VALIDATION_GENERATION_SEED
    )
    parser.add_argument("--inside-judge", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _run_in_judge_image(args: argparse.Namespace) -> None:
    if args.repository is None:
        raise ValueError("--repository is required when running the real solver")
    if args.judge_cpus <= 0:
        raise ValueError("judge_cpus must be positive")
    pinned_image = re.fullmatch(r"sha256:[0-9a-f]{64}", args.judge_image)
    if pinned_image is None:
        raise ValueError("judge_image must be pinned by sha256")

    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--cpus",
        str(args.judge_cpus),
        "--memory",
        args.judge_memory,
        "--pids-limit",
        "2048",
        "--tmpfs",
        "/tmp:rw,exec,nosuid,size=4g",
        "--env",
        "PYTHONPATH=/opt/pitbench",
        "--env",
        "GIT_CONFIG_COUNT=1",
        "--env",
        "GIT_CONFIG_KEY_0=safe.directory",
        "--env",
        "GIT_CONFIG_VALUE_0=/input/base",
        "--volume",
        f"{ROOT}:/opt/pitbench:ro",
        "--volume",
        f"{args.task_config.resolve()}:/input/task.yaml:ro",
        "--volume",
        f"{args.repository.resolve()}:/input/base:ro",
        "--volume",
        f"{args.private_root.resolve()}:/private:ro",
        "--volume",
        f"{args.output_dir.resolve()}:/output:rw",
        args.judge_image,
        "python",
        "/opt/pitbench/scripts/validate_seed_robustness_real_solver.py",
        "--inside-judge",
        "--task-config",
        "/input/task.yaml",
        "--repository",
        "/input/base",
        "--private-root",
        "/private",
        "--output-dir",
        "/output",
        "--instance-limit",
        str(args.instance_limit),
        "--parallel-runs",
        str(args.parallel_runs),
        "--reference-seed-count",
        str(args.reference_seed_count),
        "--test-seed-count",
        str(args.test_seed_count),
        "--test-list-count",
        str(args.test_list_count),
        "--generation-seed",
        str(args.generation_seed),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    args = _parse_args()
    if args.instance_limit < 2:
        raise ValueError("instance_limit must be at least two for confidence intervals")
    if args.parallel_runs < 1:
        raise ValueError("parallel_runs must be positive")
    existing_python_path = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = (
        str(ROOT)
        if not existing_python_path
        else f"{ROOT}{os.pathsep}{existing_python_path}"
    )

    task = PitBenchTask.from_yaml(args.task_config)
    if task.task_id != "pyvrp_v0_14_0":
        raise ValueError("real validation is fixed to pyvrp_v0_14_0")
    seed_robustness = task.evaluation.seed_robustness
    if seed_robustness is None:
        raise ValueError("task does not define Seed Robustness")

    validation_seeds = generate_real_validation_seeds(
        seed_min=seed_robustness.seed_selection.seed_min,
        seed_max=seed_robustness.seed_selection.seed_max,
        reference_seed_count=args.reference_seed_count,
        test_seed_count=args.test_seed_count,
        test_list_count=args.test_list_count,
        generation_seed=args.generation_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_seeds_path = args.output_dir / "validation_seeds.json"
    if validation_seeds_path.exists():
        saved_validation_seeds = RealValidationSeeds.model_validate_json(
            validation_seeds_path.read_text()
        )
        if saved_validation_seeds != validation_seeds:
            raise ValueError(
                "output directory contains a different validation seed selection"
            )
    else:
        _write_json(
            validation_seeds_path,
            validation_seeds.model_dump_json(indent=2) + "\n",
        )

    if args.judge_image is not None and not args.inside_judge:
        if args.repository is None:
            raise ValueError("--repository is required when running the real solver")
        PitBenchAdapter.validate_repository(task, args.repository)
        _run_in_judge_image(args)
        return

    if args.observations is not None:
        observations_path = args.observations
        observations = ObservationStore.read(observations_path)
    else:
        if args.repository is None:
            raise ValueError("--repository is required when running the real solver")
        PitBenchAdapter.validate_repository(task, args.repository)
        resolver = PrivateAssetResolver(args.private_root)
        plan = JudgePlan.from_instance_set_configs(
            task,
            resolver,
            public_root=ROOT,
            generated_root=args.output_dir / "generated_instances",
        )
        development_cases = [
            case
            for case in plan.cases
            if case.instance_set.kind is InstanceSetKind.AGENT_DEV
        ]
        if args.instance_limit > len(development_cases):
            raise ValueError("instance_limit exceeds available agent_dev instances")
        all_validation_seeds = (
            *validation_seeds.reference_seeds,
            *validation_seeds.test_seeds,
        )
        selected_cases = [
            replace(
                case,
                solver_seeds=all_validation_seeds,
                budgets_sec=tuple(task.evaluation.budgets_sec),
            )
            for case in development_cases[: args.instance_limit]
        ]
        no_change_patch = args.output_dir / "no_change.patch"
        no_change_patch.write_text("")
        judge = LocalProcessJudge(
            task=task,
            base_repository=args.repository,
            public_root=ROOT,
            private_root=args.private_root,
            candidate_patch=no_change_patch,
            output_dir=args.output_dir / "runs",
            code_states=(CodeState.BASE, CodeState.AGENT),
            parallel_runs=args.parallel_runs,
            run_validation_builds=False,
            progress_callback=_show_progress,
        )
        checkpoint_path = args.output_dir / "observations.checkpoint.jsonl"
        saved_observations = _load_checkpoint(checkpoint_path)
        expected_runs = {
            (
                case.instance_set.name,
                case.instance_id,
                code_state,
                solver_seed,
                budget_sec,
            )
            for case in selected_cases
            for code_state in (CodeState.BASE, CodeState.AGENT)
            for solver_seed in all_validation_seeds
            for budget_sec in task.evaluation.budgets_sec
        }
        completed_runs = {
            _run_identity(observation) for observation in saved_observations
        }
        if completed_runs - expected_runs:
            raise ValueError(
                "checkpoint contains observations outside the requested run"
            )

        missing_run_count = len(expected_runs - completed_runs)
        print(
            f"PITBENCH_PROGRESS Checkpoint: {len(completed_runs)}/"
            f"{len(expected_runs)} solver runs complete, {missing_run_count} remaining",
            flush=True,
        )
        new_observations: list[RunObservation] = []
        if missing_run_count:
            with checkpoint_path.open("a") as checkpoint_file:

                def save_observation(observation: RunObservation) -> None:
                    checkpoint_file.write(observation.model_dump_json())
                    checkpoint_file.write("\n")
                    checkpoint_file.flush()
                    os.fsync(checkpoint_file.fileno())

                new_observations = judge.run(
                    selected_cases,
                    completed_runs=completed_runs,
                    save_observation=save_observation,
                )
        observations = [*saved_observations, *new_observations]
        if len(observations) != len(expected_runs):
            raise RuntimeError("checkpoint is incomplete after solver execution")
        observations_path = args.output_dir / "observations.parquet"
        temporary_observations_path = observations_path.with_suffix(".parquet.tmp")
        ObservationStore.write(temporary_observations_path, observations)
        temporary_observations_path.replace(observations_path)

    summary = analyze_real_seed_validation(
        observations,
        task_id=task.task_id,
        budgets_sec=task.evaluation.budgets_sec,
        validation_seeds=validation_seeds,
    )
    summary_path = args.output_dir / "validation_summary.json"
    _write_json(summary_path, summary.model_dump_json(indent=2) + "\n")
    print(f"Observations: {observations_path}")
    print(f"Seed lists: {validation_seeds_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
