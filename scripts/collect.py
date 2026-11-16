"""One configured collection/worker entry point."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pitbench.schema.observation import CodeState, RunObservation


ROOT = Path(__file__).resolve().parents[1]


def _reference() -> None:
    import argparse
    from pathlib import Path

    from pitbench.solver_drivers.reference import collect_reference_solutions

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
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Read-only live progress, including running attempts",
    )
    parser.add_argument(
        "--job-id",
        action="append",
        default=None,
        help="Select prepared jobs without rewriting the manifest",
    )
    parser.add_argument("--retry-status", action="append", default=None)
    parser.add_argument("--require-queue-drained", action="store_true")
    parser.add_argument("--max-exact-customers", type=int, default=12)
    parser.add_argument("--hgs-binary", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    if args.remote_config:
        from pitbench.evaluator.remote import RemoteReferenceCollection

        if args.progress:
            result = RemoteReferenceCollection.progress(
                args.remote_config, args.output_dir
            )
        elif args.job_id is not None:
            result = RemoteReferenceCollection.retry(
                args.remote_config,
                args.output_dir,
                args.job_id,
                set(args.retry_status or ["process_error"]),
            )
        elif args.deploy:
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
        if not args.progress and isinstance(result.get("hosts"), list):
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
    if args.campaign_config or args.status or args.progress:
        from pitbench.evaluator.collection import ReferenceCollection

        if args.progress:
            from pitbench.evaluator.collection import reference_progress

            result = reference_progress(args.output_dir)
        elif args.status:
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
            result = ReferenceCollection.run(
                args.campaign_config,
                args.output_dir,
                job_ids=args.job_id,
                retry_statuses=set(args.retry_status) if args.retry_status else None,
                require_queue_drained=args.require_queue_drained,
            )
        print(
            json.dumps(
                result
                if args.progress
                else {k: v for k, v in result.items() if k != "running"}
            )
        )
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


def _controls() -> None:
    import argparse
    from pathlib import Path

    from pitbench.evaluator.collection import ReleaseControlCollection

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    ReleaseControlCollection.run(args.manifest, args.output_dir, check=args.check)


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


def _load_checkpoint(path: Path) -> list[RunObservation]:
    from pitbench.evaluator.artifacts import write_text_atomic
    from pitbench.schema.observation import RunObservation

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
                    f"{observation.model_dump_json()}\n" for observation in observations
                )
                write_text_atomic(path, checkpoint_text)
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


def _seed_args() -> argparse.Namespace:
    from pitbench.metrics.seed_robustness_validation import (
        REFERENCE_SEED_COUNT,
        TEST_LIST_COUNT,
        TEST_SEED_COUNT,
        VALIDATION_GENERATION_SEED,
    )

    parser = argparse.ArgumentParser(
        description="Validate Seed Robustness using a configured repository task."
    )
    parser.add_argument(
        "--task-config",
        type=Path,
        required=True,
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
    from pitbench.evaluator.docker_judge import docker_cli_image, judge_container_prefix

    if args.repository is None:
        raise ValueError("--repository is required when running the real solver")
    if args.judge_cpus <= 0:
        raise ValueError("judge_cpus must be positive")
    pinned_image = re.fullmatch(r"sha256:[0-9a-f]{64}", args.judge_image)
    if pinned_image is None:
        raise ValueError("judge_image must be pinned by sha256")

    command = judge_container_prefix(cpus=args.judge_cpus, memory=args.judge_memory) + [
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
        docker_cli_image(args.judge_image),
        "python",
        "/opt/pitbench/scripts/collect.py",
        "seeds",
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


def _seeds() -> None:
    from adapters.pitbench.adapter import PitBenchAdapter
    from pitbench.evaluator.artifacts import write_text_atomic
    from pitbench.evaluator.judge import JudgePlan, LocalProcessJudge
    from pitbench.evaluator.private_assets import PrivateAssetResolver
    from pitbench.evaluator.storage import ObservationStore
    from pitbench.metrics.seed_robustness_validation import (
        RealValidationSeeds,
        analyze_real_seed_validation,
        generate_real_validation_seeds,
    )
    from pitbench.schema.observation import CodeState
    from pitbench.schema.task import InstanceSetKind, PitBenchTask

    args = _seed_args()
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
    seed_robustness = task.evaluation.seed_robustness
    if seed_robustness is None:
        raise ValueError("task does not define Seed Robustness")
    args.judge_image = args.judge_image or task.repository.judge_image

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
        write_text_atomic(
            validation_seeds_path,
            validation_seeds.model_dump_json(indent=2) + "\n",
        )

    if (
        args.observations is None
        and args.judge_image is not None
        and not args.inside_judge
    ):
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
    write_text_atomic(summary_path, summary.model_dump_json(indent=2) + "\n")
    print(f"Observations: {observations_path}")
    print(f"Seed lists: {validation_seeds_path}")
    print(f"Summary: {summary_path}")


def main(argv=None):
    if __package__ in (None, ""):
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts import dispatch

    return dispatch(
        {
            "reference": _reference,
            "configuration": "pitbench.evaluator.configuration:main",
            "scenarios": "pitbench.evaluator.scenarios:main",
            "nuisance": "pitbench.evaluator.collection:NuisanceCollection.main",
            "anchors": "pitbench.evaluator.collection:AnchorCollection.main",
            "controls": _controls,
            "seeds": _seeds,
        },
        argv,
        description="Collect configured references, observations, controls and validation runs.",
    )


if __name__ == "__main__":
    main()
