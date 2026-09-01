from __future__ import annotations

import argparse
from pathlib import Path

from pitbench.evaluator.judge import LocalProcessJudge
from pitbench.evaluator.storage import ObservationStore
from pitbench.schema.observation import CodeState
from pitbench.schema.task import PitBenchTask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--base-repository", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--candidate-patch", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--parallel-runs", type=int, default=1)
    parser.add_argument(
        "--code-state",
        action="append",
        choices=[state.value for state in CodeState],
    )
    args = parser.parse_args()
    task = PitBenchTask.from_yaml(args.task_config)
    observations = LocalProcessJudge(
        task=task,
        base_repository=args.base_repository,
        private_root=args.private_root,
        candidate_patch=args.candidate_patch,
        output_dir=args.output_dir,
        code_states=tuple(CodeState(value) for value in (args.code_state or []))
        or tuple(CodeState),
        parallel_runs=args.parallel_runs,
    ).run()
    ObservationStore.write_jsonl(args.observations, observations)


if __name__ == "__main__":
    main()
