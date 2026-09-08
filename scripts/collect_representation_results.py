"""Collect the approved PyVRP relabeling experiment without computing a score."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from adapters.pitbench.adapter import PitBenchAdapter
from pitbench.evaluator.representation import (
    RecordingJudge,
    prepare_cases,
    preserve_json,
    result_record,
    write_json,
)
from pitbench.evaluator.representation import (
    customer_permutations as customer_permutations,
)
from pitbench.evaluator.representation import (
    map_solution as map_solution,
)
from pitbench.evaluator.representation import (
    relabel_instance as relabel_instance,
)
from pitbench.evaluator.storage import ObservationStore
from pitbench.schema.observation import CodeState, RunObservation
from pitbench.schema.task import PitBenchTask

ROOT = Path(__file__).resolve().parents[1]
SOLVER_SEED = 0
RELABELING_SEED = 20260907
RELABELING_COUNT = 30


def run_identity(observation: RunObservation) -> tuple:
    return (
        observation.instance_set,
        observation.instance_id,
        observation.code_state,
        observation.solver_seed,
        observation.budget_sec,
    )


def load_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    identities = set()
    lines = path.read_text().splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not line.endswith("\n"):
                path.write_text("".join(lines[:index]))
                break
            raise
        identity = run_identity(RunObservation.model_validate(record["observation"]))
        if identity in identities:
            raise ValueError("duplicate run in results checkpoint")
        identities.add(identity)
        records.append(record)
    if records and not path.read_bytes().endswith(b"\n"):
        with path.open("a") as handle:
            handle.write("\n")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, default=ROOT / "private")
    parser.add_argument("--parallel-runs", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--case-limit",
        type=int,
        help="Run only the first N relabelings for a smoke check.",
    )
    args = parser.parse_args()
    if args.parallel_runs < 1 or (args.case_limit is not None and args.case_limit < 1):
        parser.error("parallel-runs and case-limit must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    task = PitBenchTask.from_yaml(ROOT / "configs/tasks/pyvrp_v0_14_0.yaml")
    if task.release.version != "0.14.0" or task.evaluation.budgets_sec != [5.0, 10.0]:
        raise ValueError("task configuration differs from the approved experiment")
    PitBenchAdapter.validate_repository(task, args.repository)
    manifest = {
        "task_id": task.task_id,
        "source_commit": task.release.base_commit,
        "solver_seed": SOLVER_SEED,
        "relabeling_generation_seed": RELABELING_SEED,
        "relabelings_per_instance": RELABELING_COUNT,
        "instance_count": 10,
        "budgets_sec": [5.0, 10.0],
        "code_states": ["base", "agent"],
        "candidate_patch": "empty",
        "task_configuration": task.model_dump(mode="json"),
    }
    preserve_json(output_dir / "experiment.json", manifest)
    cases, transformations = prepare_cases(task, args.private_root, output_dir)
    expected = {
        (case.instance_set.name, case.instance_id, state, SOLVER_SEED, budget)
        for case in cases
        for state in CodeState
        for budget in (5.0, 10.0)
    }
    print(
        f"Prepared {len(cases)} equivalent inputs; {len(expected)} planned runs.",
        flush=True,
    )
    if args.prepare_only:
        return
    checkpoint = output_dir / "results.jsonl"
    records = load_results(checkpoint)
    observations = [
        RunObservation.model_validate(record["observation"]) for record in records
    ]
    completed = {run_identity(observation) for observation in observations}
    if completed - expected:
        raise ValueError("results contain runs outside the approved experiment")
    selected_cases = cases[: args.case_limit] if args.case_limit is not None else cases
    selected_expected = {
        item
        for item in expected
        if item[1] in {case.instance_id for case in selected_cases}
    }
    patch = output_dir / "no_change.patch"
    if patch.exists() and patch.read_bytes():
        raise ValueError("experiment requires an empty patch")
    patch.touch()
    if selected_expected - completed:
        judge = RecordingJudge(
            task,
            args.repository,
            ROOT,
            args.private_root,
            patch,
            output_dir / "runs",
            code_states=(CodeState.BASE, CodeState.AGENT),
            parallel_runs=args.parallel_runs,
            run_validation_builds=False,
        )
        with checkpoint.open("a") as handle:

            def save(observation):
                record = result_record(
                    observation, transformations[observation.instance_id], output_dir
                )
                handle.write(json.dumps(record, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                records.append(record)
                observations.append(observation)

            judge.run(selected_cases, completed_runs=completed, save_observation=save)
    ObservationStore.write(output_dir / "observations.parquet", observations)
    summary = {
        "task_id": task.task_id,
        "expected_run_count": len(expected),
        "completed_run_count": len(observations),
        "complete": len(observations) == len(expected),
        "valid_run_count": sum(item.valid for item in observations),
        "mapped_feasible_run_count": sum(
            bool((record["verification"]["mapped_original"] or {}).get("feasible"))
            for record in records
        ),
        "objective_preserved_run_count": sum(
            record["verification"]["objective_preserved"] is True for record in records
        ),
        "statistics": "deferred",
    }
    write_json(output_dir / "collection_summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
