"""Collect the approved PyVRP relabeling experiment without computing a score."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import time
from dataclasses import replace
from pathlib import Path

from adapters.pitbench.adapter import PitBenchAdapter
from pitbench.evaluator.judge import InstanceCase, JudgePlan, LocalProcessJudge
from pitbench.evaluator.private_assets import PrivateAssetResolver
from pitbench.evaluator.storage import ObservationStore
from pitbench.problem_families.cvrp import CVRPFamily
from pitbench.schema.observation import CodeState, RunObservation, RunStatus
from pitbench.schema.task import InstanceSetKind, PitBenchTask

ROOT = Path(__file__).resolve().parents[1]
SOLVER_SEED = 0
RELABELING_SEED = 20260907
RELABELING_COUNT = 30


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def preserve_json(path: Path, payload: object) -> None:
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(f"existing experiment input differs: {path}")
    else:
        write_json(path, payload)


def customer_permutations(
    customer_count: int, count: int, generator: random.Random
) -> list[list[int]]:
    if count < 1 or math.factorial(customer_count) - 1 < count:
        raise ValueError("not enough distinct nonidentity customer permutations")
    original = tuple(range(1, customer_count + 1))
    seen = {original}
    mappings = []
    while len(mappings) < count:
        customers = list(original)
        generator.shuffle(customers)
        permutation = tuple(customers)
        if permutation not in seen:
            seen.add(permutation)
            mappings.append([0, *customers])
    return mappings


def relabel_instance(original: dict, new_to_original: list[int]) -> dict:
    node_count = len(original["coordinates"])
    if (
        original.get("depot", 0) != 0
        or original.get("distance_metric") != "EUC_2D"
        or len(original["demands"]) != node_count
    ):
        raise ValueError("expected a normalized single-depot EUC_2D CVRP instance")
    if new_to_original[0] != 0 or sorted(new_to_original) != list(range(node_count)):
        raise ValueError("mapping must be a bijection fixing depot zero")
    transformed = dict(original)
    # node_ids remain the labels in the new representation; the saved mapping
    # records their relationship to the original customers.
    transformed["coordinates"] = [original["coordinates"][i] for i in new_to_original]
    transformed["demands"] = [original["demands"][i] for i in new_to_original]
    if "node_ids" in original:
        transformed["node_ids"] = list(range(1, node_count + 1))
    return transformed


def map_solution(solution: dict, new_to_original: list[int]) -> dict:
    routes = []
    for route in solution["routes"]:
        if any(type(node) is not int or not 0 < node < len(new_to_original) for node in route):
            raise ValueError("solution contains an invalid customer index")
        routes.append([new_to_original[node] for node in route])
    return {"routes": routes}


def prepare_cases(
    task: PitBenchTask, private_root: Path, output_dir: Path
) -> tuple[list[InstanceCase], dict[str, dict]]:
    development_task = task.model_copy(update={
        "instance_sets": [
            item for item in task.instance_sets if item.kind == InstanceSetKind.AGENT_DEV
        ],
    })
    plan = JudgePlan.from_instance_set_configs(
        development_task, PrivateAssetResolver(private_root), public_root=ROOT,
    )
    originals = sorted(plan.cases, key=lambda item: item.instance_id)
    if len(originals) != 10:
        raise ValueError("this experiment requires the approved ten development instances")
    generator = random.Random(RELABELING_SEED)
    cases = []
    transformations = {}
    for original_case in originals:
        assert original_case.path is not None
        original = json.loads(original_case.path.read_text())
        original_path = output_dir / "inputs" / "original" / f"{original_case.instance_id}.json"
        preserve_json(original_path, original)
        mappings = customer_permutations(len(original["coordinates"]) - 1, RELABELING_COUNT, generator)
        for index, mapping in enumerate(mappings):
            transform_id = f"customer_relabeling_{index:02d}"
            instance_id = f"{original_case.instance_id}__{transform_id}"
            transformed_path = output_dir / "inputs" / "relabeled" / f"{instance_id}.json"
            preserve_json(transformed_path, relabel_instance(original, mapping))
            transformations[instance_id] = {
                "original_instance_id": original_case.instance_id,
                "original_instance_path": str(original_path.relative_to(output_dir)),
                "transformed_instance_path": str(transformed_path.relative_to(output_dir)),
                "transform_id": transform_id,
                "new_to_original": mapping,
                "mapping_index_base": 0,
                "bks": original_case.anchor,
            }
            cases.append(replace(
                original_case, instance_id=instance_id, path=transformed_path,
                equivalence_parent_id=original_case.instance_id,
                equivalence_transform=transform_id, solver_seeds=(SOLVER_SEED,),
                budgets_sec=(5.0, 10.0),
            ))
    preserve_json(output_dir / "transformations.json", transformations)
    return cases, transformations


class RecordingJudge(LocalProcessJudge):
    """Preserve process logs in addition to the existing judge artifacts."""

    @staticmethod
    def _run(command, workspace):
        if "pitbench.solver_drivers.pyvrp" not in command.argv:
            return LocalProcessJudge._run(command, workspace)
        output = Path(command.argv[command.argv.index("--output") + 1])
        started = time.monotonic()
        try:
            completed = LocalProcessJudge._run(command, workspace)
        except subprocess.TimeoutExpired as error:
            stdout, stderr = error.stdout or "", error.stderr or ""
            returncode = None
            raise
        else:
            stdout, stderr = completed.stdout, completed.stderr
            returncode = completed.returncode
            return completed
        finally:
            if "stdout" in locals():
                for suffix, content in ((".stdout.log", stdout), (".stderr.log", stderr)):
                    output.with_suffix(suffix).write_text(
                        content.decode(errors="replace") if isinstance(content, bytes) else content
                    )
                write_json(output.with_suffix(".process.json"), {
                    "argv": command.argv, "returncode": returncode,
                    "timed_out": returncode is None,
                    "elapsed_sec": time.monotonic() - started,
                })

    def _run_case(self, workspace, case, state, seed, budget, *, cpu_ids=None):
        try:
            return super()._run_case(workspace, case, state, seed, budget, cpu_ids=cpu_ids)
        except (ValueError, KeyError, IndexError, TypeError) as error:
            # Malformed solver output remains an individual failed observation.
            # Filesystem and build failures still propagate to the experiment runner.
            return self._failure(case, state, seed, budget, RunStatus.INVALID,
                                 f"output parsing/verification failed: {error}")


def result_record(observation: RunObservation, transformation: dict, output_dir: Path) -> dict:
    output = (output_dir / "runs" / observation.code_state.value / observation.instance_set
              / observation.instance_id / f"seed-{observation.solver_seed}-budget-{observation.budget_sec:g}.json")
    solution = output.with_suffix(".solution.json")
    verification = {"transformed": None, "mapped_original": None, "objective_preserved": None, "error": None}
    if solution.exists():
        try:
            family = CVRPFamily()
            transformed = family.verify(output_dir / transformation["transformed_instance_path"], solution)
            mapped_path = output.with_suffix(".mapped.solution.json")
            write_json(mapped_path, map_solution(json.loads(solution.read_text()), transformation["new_to_original"]))
            mapped = family.verify(output_dir / transformation["original_instance_path"], mapped_path)
            verification.update({
                "transformed": transformed.model_dump(mode="json"),
                "mapped_original": mapped.model_dump(mode="json"),
                "objective_preserved": (
                    transformed.objective == mapped.objective
                    if transformed.objective is not None and mapped.objective is not None else None
                ),
            })
        except (ValueError, KeyError, IndexError, TypeError) as error:
            verification["error"] = f"{type(error).__name__}: {error}"
    write_json(output.with_suffix(".verification.json"), verification)
    artifacts = {}
    for name, suffix in {
        "solver_result": ".json", "solution": ".solution.json",
        "mapped_solution": ".mapped.solution.json", "trajectory": ".trajectory.jsonl",
        "stdout": ".stdout.log", "stderr": ".stderr.log",
        "process": ".process.json", "verification": ".verification.json",
    }.items():
        path = output.with_suffix(suffix)
        artifacts[name] = str(path.relative_to(output_dir)) if path.exists() else None
    return {"observation": observation.model_dump(mode="json"),
            "verification": verification, "artifacts": artifacts}


def run_identity(observation: RunObservation) -> tuple:
    return (observation.instance_set, observation.instance_id, observation.code_state,
            observation.solver_seed, observation.budget_sec)


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
    parser.add_argument("--case-limit", type=int, help="Run only the first N relabelings for a smoke check.")
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
        "task_id": task.task_id, "source_commit": task.release.base_commit,
        "solver_seed": SOLVER_SEED, "relabeling_generation_seed": RELABELING_SEED,
        "relabelings_per_instance": RELABELING_COUNT, "instance_count": 10,
        "budgets_sec": [5.0, 10.0], "code_states": ["base", "agent"],
        "candidate_patch": "empty", "task_configuration": task.model_dump(mode="json"),
    }
    preserve_json(output_dir / "experiment.json", manifest)
    cases, transformations = prepare_cases(task, args.private_root, output_dir)
    expected = {(case.instance_set.name, case.instance_id, state, SOLVER_SEED, budget)
                for case in cases for state in CodeState for budget in (5.0, 10.0)}
    print(f"Prepared {len(cases)} equivalent inputs; {len(expected)} planned runs.", flush=True)
    if args.prepare_only:
        return
    checkpoint = output_dir / "results.jsonl"
    records = load_results(checkpoint)
    observations = [RunObservation.model_validate(record["observation"]) for record in records]
    completed = {run_identity(observation) for observation in observations}
    if completed - expected:
        raise ValueError("results contain runs outside the approved experiment")
    selected_cases = cases[:args.case_limit] if args.case_limit is not None else cases
    selected_expected = {item for item in expected if item[1] in {case.instance_id for case in selected_cases}}
    patch = output_dir / "no_change.patch"
    if patch.exists() and patch.read_bytes():
        raise ValueError("experiment requires an empty patch")
    patch.touch()
    if selected_expected - completed:
        judge = RecordingJudge(
            task, args.repository, ROOT, args.private_root, patch, output_dir / "runs",
            code_states=(CodeState.BASE, CodeState.AGENT), parallel_runs=args.parallel_runs,
            run_validation_builds=False,
        )
        with checkpoint.open("a") as handle:
            def save(observation):
                record = result_record(observation, transformations[observation.instance_id], output_dir)
                handle.write(json.dumps(record, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                records.append(record)
                observations.append(observation)

            judge.run(selected_cases, completed_runs=completed, save_observation=save)
    ObservationStore.write(output_dir / "observations.parquet", observations)
    summary = {
        "task_id": task.task_id, "expected_run_count": len(expected),
        "completed_run_count": len(observations),
        "complete": len(observations) == len(expected),
        "valid_run_count": sum(item.valid for item in observations),
        "mapped_feasible_run_count": sum(
            bool((record["verification"]["mapped_original"] or {}).get("feasible")) for record in records
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
