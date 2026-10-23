from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from pitbench.evaluator.isolation import ARTIFACT_OWNER_UID, protect_output
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
    parser.add_argument("--instances-per-instance-set", type=int)
    parser.add_argument("--solver-seed-limit", type=int)
    parser.add_argument("--checkpoint-identity")
    parser.add_argument("--artifact-gid", type=int)
    parser.add_argument(
        "--code-state",
        action="append",
        choices=[state.value for state in CodeState],
    )
    args = parser.parse_args()
    if args.artifact_gid is not None:
        protect_output(args.output_dir, args.artifact_gid)
    measurement_started_at = datetime.now(timezone.utc).isoformat()
    task = PitBenchTask.from_yaml(args.task_config)
    boundary_cases = (
        prepare_boundary_cases(
            task,
            args.output_dir,
            private_root=args.private_root,
            public_root=args.public_root,
            instances_per_instance_set=(
                args.instances_per_instance_set
                if args.solver_seed_limit is not None
                else None
            ),
            solver_seed_limit=args.solver_seed_limit,
        )
        if task.evaluation.operational_reliability or args.reliability_only
        else []
    )
    if (
        args.instances_per_instance_set is not None
        and args.instances_per_instance_set < 1
    ):
        parser.error("--instances-per-instance-set must be positive")
    if args.solver_seed_limit is not None and args.solver_seed_limit < 1:
        parser.error("--solver-seed-limit must be positive")
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
        instances_per_instance_set=args.instances_per_instance_set,
        solver_seed_limit=args.solver_seed_limit,
        observations_path=args.observations,
    )
    if args.reliability_only:
        observations = judge.run([])
    elif task.evaluation.representation_robustness is not None:
        observations = run_with_representation(judge, args.private_root)
    else:
        observations = judge.run()
    ObservationStore.write_jsonl(args.observations, observations)
    if args.checkpoint_identity is not None:
        grid_path = args.output_dir / "expected-run-grid.json"
        checkpoint = {
            "measurement_started_at": measurement_started_at,
            "identity": json.loads(args.checkpoint_identity),
            "observation_count": len(observations),
            "observations_sha256": hashlib.sha256(
                args.observations.read_bytes()
            ).hexdigest(),
            "expected_grid_sha256": hashlib.sha256(grid_path.read_bytes()).hexdigest(),
        }
        marker = args.output_dir / "judge-checkpoint.json"
        temporary = marker.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(checkpoint, sort_keys=True) + "\n")
        temporary.replace(marker)
    for root_dir, dirs, files in os.walk(args.output_dir):
        try:
            if args.artifact_gid is not None:
                os.chown(root_dir, ARTIFACT_OWNER_UID, args.artifact_gid)
            os.chmod(root_dir, 0o770 if args.artifact_gid is not None else 0o777)
        except OSError:
            pass
        for f in files:
            try:
                path = Path(root_dir) / f
                if not path.is_symlink():
                    if args.artifact_gid is not None:
                        os.chown(path, ARTIFACT_OWNER_UID, args.artifact_gid)
                    os.chmod(path, 0o660 if args.artifact_gid is not None else 0o666)
            except OSError:
                pass


if __name__ == "__main__":
    main()
