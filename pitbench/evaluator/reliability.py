"""Attach the public boundary suite to the existing isolated judge."""

from __future__ import annotations

import json
from pathlib import Path

from pitbench.evaluator.judge import InstanceCase, _development_seeds
from pitbench.instances.boundary import boundary_suite
from pitbench.schema.task import InstanceSetKind, InstanceSetSpec, PitBenchTask

SUITE_NAME = "operational_reliability"


def prepare_boundary_cases(task: PitBenchTask, output_dir: Path) -> list[InstanceCase]:
    suite = boundary_suite(task.problem_family)
    examples = suite.cases()
    directory = output_dir.resolve() / "reliability" / "inputs"
    directory.mkdir(parents=True, exist_ok=True)
    seeds = _development_seeds(task)
    instance_set = InstanceSetSpec(
        name="boundary_cases",
        kind=InstanceSetKind.AGENT_DEV,
        instance_set_config="reliability/manifest.json",
        size=len(examples),
    )
    verifier = suite.verifier()
    cases = []
    manifest_cases = []
    for example in examples:
        path = suite.write(example, directory)
        reference = directory / f"{example.name}.reference.json"
        reference.write_text(json.dumps(example.reference_solution, indent=2) + "\n")
        manifest_cases.append(
            {
                "instance_id": example.name,
                "description": example.description,
                "input": str(path.relative_to(directory.parent)),
                "reference_solution": str(reference.relative_to(directory.parent)),
                "expected": "normal termination with an independently verified feasible solution",
            }
        )
        cases.append(
            InstanceCase(
                instance_set=instance_set,
                instance_id=example.name,
                path=path,
                anchor=None,
                solver_seeds=seeds,
                budgets_sec=tuple(task.evaluation.budgets_sec),
                test_suite=SUITE_NAME,
                verifier=verifier,
            )
        )
    (directory.parent / "manifest.json").write_text(
        json.dumps(
            {
                "task_id": task.task_id,
                "test_suite": SUITE_NAME,
                "problem_family": task.problem_family.value,
                "source_commit": task.release.base_commit,
                "budgets_sec": task.evaluation.budgets_sec,
                "solver_seeds": list(seeds),
                "threads": task.evaluation.threads,
                "interpretation": "observed boundary test coverage; not a deployment success probability",
                "cases": manifest_cases,
            },
            indent=2,
        )
        + "\n"
    )
    return cases
