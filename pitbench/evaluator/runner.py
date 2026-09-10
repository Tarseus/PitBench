from __future__ import annotations

import argparse
from pathlib import Path

from pitbench.evaluator.judge import LocalProcessJudge, _development_seeds
from pitbench.evaluator.reliability import prepare_boundary_cases
from pitbench.evaluator.representation import run_with_representation
from pitbench.evaluator.storage import ObservationStore
from pitbench.schema.observation import CodeState
from pitbench.schema.task import PitBenchTask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--base-repository", type=Path, required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--candidate-patch", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--parallel-runs", type=int, default=1)
    parser.add_argument("--reliability-only", action="store_true")
    parser.add_argument(
        "--code-state",
        action="append",
        choices=[state.value for state in CodeState],
    )
    args = parser.parse_args()
    task = PitBenchTask.from_yaml(args.task_config)
    if args.reliability_only and not task.evaluation.operational_reliability:
        parser.error("task does not enable operational reliability")
    boundary_cases = (
        prepare_boundary_cases(task, args.output_dir)
        if task.evaluation.operational_reliability
        else []
    )
    judge = LocalProcessJudge(
        task=task,
        base_repository=args.base_repository,
        public_root=args.public_root,
        private_root=args.private_root,
        candidate_patch=args.candidate_patch,
        output_dir=args.output_dir,
        code_states=tuple(CodeState(value) for value in (args.code_state or []))
        or tuple(CodeState),
        parallel_runs=args.parallel_runs,
        additional_cases=boundary_cases,
        evaluation_seeds=_development_seeds(task) if args.reliability_only else None,
        family=boundary_cases[0].verifier if args.reliability_only else None,
    )
    if args.reliability_only:
        observations = judge.run([])
    elif task.evaluation.representation_robustness is not None:
        observations = run_with_representation(judge, args.private_root)
    else:
        observations = judge.run()
    ObservationStore.write_jsonl(args.observations, observations)


if __name__ == "__main__":
    main()
