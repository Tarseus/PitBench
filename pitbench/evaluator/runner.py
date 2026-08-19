from __future__ import annotations

import argparse
from pathlib import Path

from pitbench.evaluator.judge import LocalProcessJudge
from pitbench.evaluator.storage import ObservationStore
from pitbench.schema.task import PitBenchTask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-repository", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--candidate-patch", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    args = parser.parse_args()
    task = PitBenchTask.from_yaml(args.manifest)
    observations = LocalProcessJudge(
        task=task,
        base_repository=args.base_repository,
        private_root=args.private_root,
        candidate_patch=args.candidate_patch,
        output_dir=args.output_dir,
    ).run()
    ObservationStore.write_jsonl(args.observations, observations)


if __name__ == "__main__":
    main()
