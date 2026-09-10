"""Customer relabeling and raw result collection for representation experiments."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from dataclasses import replace
from pathlib import Path

from pitbench.evaluator.judge import InstanceCase, JudgePlan, LocalProcessJudge
from pitbench.evaluator.private_assets import PrivateAssetResolver
from pitbench.evaluator.representations import CustomerRepresentation
from pitbench.problem_families.verification import CVRPFamily
from pitbench.schema.observation import RunObservation
from pitbench.schema.task import (
    InstanceSetKind,
    PitBenchTask,
    RepresentationRobustnessConfig,
)

ROOT = Path(__file__).resolve().parents[2]


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


def prepare_cases(
    task: PitBenchTask,
    private_root: Path,
    output_dir: Path,
    *,
    public_root: Path = ROOT,
) -> tuple[list[InstanceCase], dict[str, dict]]:
    config = (
        task.evaluation.representation_robustness or RepresentationRobustnessConfig()
    )
    development_task = task.model_copy(
        update={
            "instance_sets": [
                item
                for item in task.instance_sets
                if item.kind == InstanceSetKind.AGENT_DEV
                and item.name == config.instance_set
            ],
        }
    )
    plan = JudgePlan.from_instance_set_configs(
        development_task,
        PrivateAssetResolver(private_root),
        public_root=public_root,
    )
    originals = sorted(plan.cases, key=lambda item: item.instance_id)
    expected_count = development_task.instance_sets[0].size
    if len(originals) != expected_count:
        raise ValueError(
            "representation instance count differs from the task configuration"
        )
    generator = random.Random(config.relabeling_generation_seed)
    cases = []
    transformations = {}
    for original_case in originals:
        assert original_case.path is not None
        original = json.loads(original_case.path.read_text())
        original_path = (
            output_dir / "inputs" / "original" / f"{original_case.instance_id}.json"
        )
        preserve_json(original_path, original)
        mappings = CustomerRepresentation.permutations(
            len(original["coordinates"]) - 1, config.relabelings_per_instance, generator
        )
        for index, mapping in enumerate(mappings):
            transform_id = f"customer_relabeling_{index:02d}"
            instance_id = f"{original_case.instance_id}__{transform_id}"
            transformed_path = (
                output_dir / "inputs" / "relabeled" / f"{instance_id}.json"
            )
            preserve_json(
                transformed_path, CustomerRepresentation.permute(original, mapping)
            )
            transformations[instance_id] = {
                "original_instance_id": original_case.instance_id,
                "original_instance_path": str(original_path.relative_to(output_dir)),
                "transformed_instance_path": str(
                    transformed_path.relative_to(output_dir)
                ),
                "transform_id": transform_id,
                "new_to_original": mapping,
                "mapping_index_base": 0,
                "bks": original_case.anchor,
            }
            cases.append(
                replace(
                    original_case,
                    instance_id=instance_id,
                    path=transformed_path,
                    equivalence_parent_id=original_case.instance_id,
                    equivalence_transform=transform_id,
                    solver_seeds=(config.solver_seed,),
                    budgets_sec=tuple(task.evaluation.budgets_sec),
                )
            )
    preserve_json(output_dir / "transformations.json", transformations)
    return cases, transformations


def result_record(
    observation: RunObservation,
    transformation: dict,
    output_dir: Path,
    *,
    runs_dir: Path | None = None,
) -> dict:
    runs_dir = runs_dir if runs_dir is not None else output_dir / "runs"
    output = (
        runs_dir
        / observation.code_state.value
        / observation.instance_set
        / observation.instance_id
        / f"seed-{observation.solver_seed}-budget-{observation.budget_sec:g}.json"
    )
    solution = output.with_suffix(".solution.json")
    verification = {
        "transformed": None,
        "mapped_original": None,
        "objective_preserved": None,
        "error": None,
    }
    if solution.exists():
        try:
            family = CVRPFamily()
            transformed = family.verify(
                output_dir / transformation["transformed_instance_path"], solution
            )
            mapped_path = output.with_suffix(".mapped.solution.json")
            write_json(
                mapped_path,
                CustomerRepresentation.map_solution(
                    json.loads(solution.read_text()), transformation["new_to_original"]
                ),
            )
            mapped = family.verify(
                output_dir / transformation["original_instance_path"], mapped_path
            )
            verification.update(
                {
                    "transformed": transformed.model_dump(mode="json"),
                    "mapped_original": mapped.model_dump(mode="json"),
                    "objective_preserved": (
                        transformed.objective == mapped.objective
                        if transformed.objective is not None
                        and mapped.objective is not None
                        else None
                    ),
                }
            )
        except (ValueError, KeyError, IndexError, TypeError) as error:
            verification["error"] = f"{type(error).__name__}: {error}"
    write_json(output.with_suffix(".verification.json"), verification)
    artifacts = {}
    for name, suffix in {
        "solver_result": ".json",
        "solution": ".solution.json",
        "mapped_solution": ".mapped.solution.json",
        "trajectory": ".trajectory.jsonl",
        "stdout": ".stdout.log",
        "stderr": ".stderr.log",
        "process": ".process.json",
        "verification": ".verification.json",
    }.items():
        path = output.with_suffix(suffix)
        artifacts[name] = os.path.relpath(path, output_dir) if path.exists() else None
    return {
        "observation": observation.model_dump(mode="json"),
        "verification": verification,
        "artifacts": artifacts,
    }


def run_with_representation(
    judge: LocalProcessJudge, private_root: Path
) -> list[RunObservation]:
    """Run the standard grid and configured relabelings in the same judge."""
    task = judge.task
    config = task.evaluation.representation_robustness
    if config is None:
        return judge.run()

    output_dir = judge.output_dir / "representation"
    cases, transformations = prepare_cases(
        task,
        private_root,
        output_dir,
        public_root=judge.public_root,
    )
    details = {
        "task_id": task.task_id,
        "source_commit": task.release.base_commit,
        "candidate_patch_sha256": (
            hashlib.sha256(judge.candidate_patch.read_bytes()).hexdigest()
            if judge.candidate_patch is not None
            else None
        ),
        "configuration": config.model_dump(mode="json"),
        "budgets_sec": task.evaluation.budgets_sec,
        "threads": task.evaluation.threads,
        "code_states": [state.value for state in judge.code_states],
        "instance_count": len({case.equivalence_parent_id for case in cases}),
        "expected_run_count": len(cases)
        * len(judge.code_states)
        * len(task.evaluation.budgets_sec),
        "completed_run_count": 0,
        "transformations": "transformations.json",
        "results": "results.jsonl",
        "statistics": "deferred",
    }
    write_json(output_dir / "details.json", details)
    with tempfile.TemporaryDirectory(
        prefix="pitbench-representation-plan-"
    ) as temporary:
        plan = JudgePlan.from_instance_set_configs(
            task,
            judge.resolver,
            public_root=judge.public_root,
            generated_root=Path(temporary),
            evaluation_seeds=judge.evaluation_seeds,
        )
        with (output_dir / "results.jsonl").open("w") as handle:

            def save(observation: RunObservation) -> None:
                if observation.equivalence_parent_id is None:
                    return
                record = result_record(
                    observation,
                    transformations[observation.instance_id],
                    output_dir,
                    runs_dir=judge.output_dir,
                )
                handle.write(json.dumps(record, allow_nan=False) + "\n")
                handle.flush()
                details["completed_run_count"] += 1

            observations = judge.run([*plan.cases, *cases], save_observation=save)
    write_json(output_dir / "details.json", details)
    return observations
