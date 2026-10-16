from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import yaml

from pitbench.agent_cli import _public_verification
from pitbench.instances.generate import materialize_generated_instance_set
from pitbench.problem_families.verification import (
    CPFamily,
    MIPFamily,
    TrustedOptimumFamily,
    TrustedOptimumOracle,
    _public_multidimensional_knapsack_data,
)
from pitbench.tasks import TaskCatalog

ROOT = Path(__file__).resolve().parents[3]


def _knapsack_solution(selected: list[int], item_count: int, objective: int) -> dict:
    raw = (
        f"Model status\nOptimal\nObjective {objective}\n"
        f"# Columns {item_count}\n"
        + "".join(f"x{item} {int(item in selected)}\n" for item in range(item_count))
        + "# Rows 0\n"
    )
    return {"raw_solution": raw}


def test_new_exact_task_panels_have_hash_bound_verified_targets() -> None:
    catalog = TaskCatalog(ROOT)
    for task_id in (
        "highs_multidimensional_knapsack_v1_15_1",
        "choco_graph_coloring_v6_0_1",
    ):
        catalog.validate_one(task_id)
        oracle = TrustedOptimumOracle.from_yaml(
            ROOT / "private/oracles" / f"{task_id}.yaml"
        )
        payload = yaml.safe_load(
            (ROOT / "private/hidden_instances" / task_id / "judge_id.yaml").read_text()
        )
        with tempfile.TemporaryDirectory() as temporary:
            paths = materialize_generated_instance_set(
                payload,
                Path(temporary),
                expected_visibility="judge",
                stem_prefix="judge_id",
            )
            assert len(paths) == len(oracle.records) == 30
            for path, record in zip(paths, oracle.records, strict=True):
                reference = (
                    ROOT
                    / "private"
                    / record.reference_solution_uri.removeprefix("private://")
                )
                verifier = TrustedOptimumFamily(record, reference)
                assert verifier._check_oracle(path) is None
                if record.problem_kind == "graph_coloring":
                    assert verifier.verify(path, reference).feasible


def test_new_exact_tasks_reject_semantically_invalid_solutions(tmp_path: Path) -> None:
    knapsack = (
        ROOT
        / "configs/instance_sets/highs_multidimensional_knapsack_v1_15_1_dev"
        / "instances/dev_0000.lp"
    )
    reference = json.loads(
        (
            ROOT
            / "configs/instance_sets/highs_multidimensional_knapsack_v1_15_1_dev"
            / "bks_solutions/dev_0000.solution.json"
        ).read_text()
    )
    weights, capacities = _public_multidimensional_knapsack_data(knapsack)
    solution_path = tmp_path / "knapsack.solution.json"
    solution_path.write_text(
        json.dumps(
            _knapsack_solution(
                reference["selected_items"], len(weights[0]), -sum(capacities)
            )
        )
    )
    assert MIPFamily().verify(knapsack, solution_path).feasible
    development_output = tmp_path / "knapsack.json"
    development_output.with_suffix(".solution.json").write_bytes(
        solution_path.read_bytes()
    )
    feedback = _public_verification(
        {
            "problem_family": "mip",
            "development_targets": {
                "dev_0000.lp": {
                    "sha256": hashlib.sha256(knapsack.read_bytes()).hexdigest()
                }
            },
        },
        knapsack,
        development_output,
    )
    assert feedback["verification"] == "independent_public"
    assert feedback["verified_valid"] is True
    solution_path.write_text(
        json.dumps(
            _knapsack_solution(
                list(range(len(weights[0]))), len(weights[0]), -sum(capacities)
            )
        )
    )
    assert not MIPFamily().verify(knapsack, solution_path).feasible

    graph = (
        ROOT
        / "configs/instance_sets/choco_graph_coloring_v6_0_1_dev"
        / "instances/dev_0000.json"
    )
    graph_data = json.loads(graph.read_text())
    colors = json.loads(
        (
            ROOT
            / "configs/instance_sets/choco_graph_coloring_v6_0_1_dev"
            / "bks_solutions/dev_0000.solution.json"
        ).read_text()
    )["colors"]
    solution_path = tmp_path / "graph.solution.json"
    solution_path.write_text(json.dumps({"colors": colors}))
    assert CPFamily().verify(graph, solution_path).feasible
    development_output = tmp_path / "graph.json"
    development_output.with_suffix(".solution.json").write_bytes(
        solution_path.read_bytes()
    )
    feedback = _public_verification({"problem_family": "cp"}, graph, development_output)
    assert feedback["verification"] == "independent_public"
    assert feedback["verified_valid"] is True
    changed = colors.copy()
    first, second = graph_data["edge_endpoints"][:2]
    changed[second] = changed[first]
    solution_path.write_text(json.dumps({"colors": changed}))
    assert not CPFamily().verify(graph, solution_path).feasible
