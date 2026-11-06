from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from pitbench.evaluator.docker_judge import judge_harness_sha256
from pitbench.evaluator.judge import LocalProcessJudge, _development_seeds
from pitbench.evaluator.reliability import prepare_boundary_cases
from pitbench.evaluator.representation import run_with_representation
from pitbench.evaluator.storage import ObservationStore
from pitbench.schema.observation import CodeState, ExpectedRunGrid, RunObservation
from pitbench.schema.task import PitBenchTask


def _run_identity(item: RunObservation) -> tuple[str, str, CodeState, int, float]:
    return (
        item.instance_set,
        item.instance_id,
        item.code_state,
        item.solver_seed,
        item.budget_sec,
    )


def _resume_observations(
    *,
    observations_path: Path,
    output_dir: Path,
    task: PitBenchTask,
    task_config: Path,
    candidate_patch: Path,
    code_states: tuple[CodeState, ...],
    checkpoint_identity: dict[str, object] | None,
    allow_shared_base: bool = False,
) -> tuple[
    list[RunObservation],
    set[tuple[str, str, CodeState, int, float]],
    dict[str, object] | None,
]:
    if not observations_path.is_file():
        raise ValueError("cannot resume: observations file does not exist")
    grid_path = output_dir / "expected-run-grid.json"
    if not grid_path.is_file():
        raise ValueError("cannot resume: expected run grid does not exist")
    expected_grid = ExpectedRunGrid.model_validate_json(grid_path.read_text())
    if expected_grid.task_id != task.task_id:
        raise ValueError("cannot resume: expected run grid belongs to another task")

    expected = {
        (
            run.instance_set,
            run.instance_id,
            run.code_state,
            run.solver_seed,
            run.budget_sec,
        )
        for run in expected_grid.runs
    }
    raw_checkpoint = observations_path.read_bytes()
    raw_lines = raw_checkpoint.splitlines(keepends=True)
    observations: list[RunObservation] = []
    valid_prefix = bytearray()
    corrupt_tail: bytes | None = None
    for index, line in enumerate(raw_lines):
        if not line.strip():
            valid_prefix.extend(line)
            continue
        try:
            observation = RunObservation.model_validate_json(line)
        except ValueError as error:
            if any(remaining.strip() for remaining in raw_lines[index + 1 :]):
                raise ValueError(
                    "cannot resume: observations contain corruption before the final record"
                ) from error
            corrupt_tail = raw_checkpoint[len(valid_prefix) :]
            break
        observations.append(observation)
        valid_prefix.extend(line)
    completed: set[tuple[str, str, CodeState, int, float]] = set()
    allowed_states = set(code_states)
    if allow_shared_base:
        allowed_states.add(CodeState.BASE)
    for observation in observations:
        identity = _run_identity(observation)
        if (
            observation.task_id != task.task_id
            or observation.code_state not in allowed_states
        ):
            raise ValueError(
                "cannot resume: observation task or code state does not match"
            )
        if identity not in expected:
            raise ValueError(
                "cannot resume: observation is outside the expected run grid"
            )
        if identity in completed:
            raise ValueError("cannot resume: duplicate completed run identity")
        completed.add(identity)

    if checkpoint_identity is not None:
        expected_task_sha = hashlib.sha256(task_config.read_bytes()).hexdigest()
        expected_candidate_sha = hashlib.sha256(
            candidate_patch.read_bytes()
        ).hexdigest()
        if checkpoint_identity.get("task_config_sha256") != expected_task_sha:
            raise ValueError("cannot resume: task configuration hash does not match")
        if checkpoint_identity.get("candidate_patch_sha256") != expected_candidate_sha:
            raise ValueError("cannot resume: candidate patch hash does not match")
        identity_states = checkpoint_identity.get("code_states")
        if identity_states is not None and set(identity_states) != {
            state.value for state in code_states
        }:
            raise ValueError("cannot resume: checkpoint code states do not match")
    recovery: dict[str, object] | None = None
    if corrupt_tail is not None or (
        raw_checkpoint and not raw_checkpoint.endswith(b"\n")
    ):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = observations_path.with_name(
            f"{observations_path.stem}.pre-resume-{stamp}.raw"
        )
        suffix = 1
        while backup.exists():
            backup = observations_path.with_name(
                f"{observations_path.stem}.pre-resume-{stamp}-{suffix}.raw"
            )
            suffix += 1
        shutil.copy2(observations_path, backup)
        repaired = observations_path.with_name(observations_path.name + ".tmp")
        with repaired.open("wb") as handle:
            handle.write(valid_prefix)
            if valid_prefix and not valid_prefix.endswith(b"\n"):
                handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        repaired.replace(observations_path)
        recovery = {
            "corrupt_tail_backup": backup.name,
            "corrupt_tail_bytes": len(corrupt_tail or b""),
            "original_observations_sha256": hashlib.sha256(raw_checkpoint).hexdigest(),
        }
    return observations, completed, recovery


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
    parser.add_argument("--resume-existing-observations", action="store_true")
    parser.add_argument("--base-observations", type=Path)
    parser.add_argument(
        "--code-state",
        action="append",
        choices=[state.value for state in CodeState],
    )
    args = parser.parse_args()
    measurement_started_at = datetime.now(timezone.utc).isoformat()
    task = PitBenchTask.from_yaml(args.task_config)
    code_states = tuple(CodeState(value) for value in (args.code_state or [])) or tuple(
        CodeState
    )
    checkpoint_identity = (
        json.loads(args.checkpoint_identity)
        if args.checkpoint_identity is not None
        else None
    )
    previous_observations: list[RunObservation] = []
    completed_runs: set[tuple[str, str, CodeState, int, float]] = set()
    resume_recovery: dict[str, object] | None = None
    if args.resume_existing_observations:
        try:
            previous_observations, completed_runs, resume_recovery = (
                _resume_observations(
                    observations_path=args.observations,
                    output_dir=args.output_dir,
                    task=task,
                    task_config=args.task_config,
                    candidate_patch=args.candidate_patch,
                    code_states=code_states,
                    checkpoint_identity=checkpoint_identity,
                    allow_shared_base=args.base_observations is not None,
                )
            )
        except (OSError, ValueError) as error:
            parser.error(str(error))
        resume_event = {
            "started_at": measurement_started_at,
            "previous_observation_count": len(previous_observations),
            "previous_observations_sha256": (
                resume_recovery["original_observations_sha256"]
                if resume_recovery is not None
                else hashlib.sha256(args.observations.read_bytes()).hexdigest()
            ),
            "checkpoint_identity": checkpoint_identity,
            "recovery": resume_recovery,
            "resume_code_sha256": judge_harness_sha256(
                Path(__file__).resolve().parents[2]
            ),
        }
        event_log = args.output_dir / "judge-resume-events.jsonl"
        event_log.parent.mkdir(parents=True, exist_ok=True)
        with event_log.open("a") as handle:
            handle.write(json.dumps(resume_event, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    elif args.base_observations is not None:
        previous_observations = ObservationStore.read_jsonl(args.base_observations)
        if any(
            row.code_state != CodeState.BASE or row.task_id != task.task_id
            for row in previous_observations
        ):
            parser.error(
                "shared Base contains a different task or non-Base observations"
            )
        completed_runs = {_run_identity(row) for row in previous_observations}
        ObservationStore.write_jsonl(args.observations, previous_observations)
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
        code_states=code_states,
        parallel_runs=args.parallel_runs,
        additional_cases=boundary_cases,
        evaluation_seeds=_development_seeds(task) if args.reliability_only else None,
        family=boundary_cases[0].verifier if args.reliability_only else None,
        instances_per_instance_set=args.instances_per_instance_set,
        solver_seed_limit=args.solver_seed_limit,
        observations_path=args.observations,
    )
    if previous_observations or args.resume_existing_observations:
        # Resume through the existing raw JSONL checkpoint. The judge's normal
        # initialization truncates this path, so resumed writes use the callback.
        judge.observations_path = None

        saved_since_sync = 0

        def save_resumed_observation(observation: RunObservation) -> None:
            nonlocal saved_since_sync
            with args.observations.open("a") as checkpoint:
                checkpoint.write(observation.model_dump_json() + "\n")
                checkpoint.flush()
                saved_since_sync += 1
                if saved_since_sync >= 32:
                    os.fsync(checkpoint.fileno())
                    saved_since_sync = 0

        save_observation = save_resumed_observation
    else:
        save_observation = None
    if args.reliability_only and args.resume_existing_observations:
        observations = judge.run(
            [], completed_runs=completed_runs, save_observation=save_observation
        )
    elif args.reliability_only:
        observations = judge.run([])
    elif task.evaluation.representation_robustness is not None:
        observations = run_with_representation(
            judge,
            args.private_root,
            previous_observations=previous_observations,
            completed_runs=completed_runs,
            save_observation=save_observation,
        )
    elif previous_observations or args.resume_existing_observations:
        observations = judge.run(
            completed_runs=completed_runs, save_observation=save_observation
        )
    else:
        observations = judge.run()
    observations = [*previous_observations, *observations]
    ObservationStore.write_jsonl(args.observations, observations)
    if args.checkpoint_identity is not None:
        grid_path = args.output_dir / "expected-run-grid.json"
        checkpoint = {
            "measurement_started_at": measurement_started_at,
            "identity": json.loads(args.checkpoint_identity),
            "observation_count": len(observations),
            "resume_code_sha256": (
                judge_harness_sha256(Path(__file__).resolve().parents[2])
                if args.resume_existing_observations
                else None
            ),
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
            os.chmod(root_dir, 0o777)
        except OSError:
            pass
        for f in files:
            try:
                os.chmod(Path(root_dir) / f, 0o666)
            except OSError:
                pass


if __name__ == "__main__":
    main()
