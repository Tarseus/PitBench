from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import yaml

from pitbench.agent_cli import _public_verification
from pitbench.instances.generate import (
    binary_clique_cover_data,
    binary_clique_cover_fme_extensions,
    materialize_generated_instance_set,
)
from pitbench.problem_families.verification import (
    CPFamily,
    MIPFamily,
    TrustedOptimumFamily,
    TrustedOptimumOracle,
    TrustedOptimumRecord,
    _public_multidimensional_knapsack_data,
)
from pitbench.tasks import TaskCatalog

ROOT = Path(__file__).resolve().parents[3]


def test_binary_clique_cover_mip_has_a_hash_bound_optimality_proof(
    tmp_path: Path,
) -> None:
    generator = {
        "kind": "binary_clique_cover_mip",
        "count": 1,
        "groups": [4],
        "group_sizes": [5],
        "coefficient_min": 120,
        "coefficient_max": 190,
        "cross_edge_density": [0.1],
        "randomness": {"instance_seed": 910000},
    }
    instance_config = {
        "schema_version": "1.0",
        "visibility": "judge",
        "format": "mps",
        "generator": generator,
    }
    instance = materialize_generated_instance_set(
        instance_config,
        tmp_path / "instances",
        expected_visibility="judge",
        stem_prefix="judge_id",
    )[0]
    groups, planted, _ = binary_clique_cover_data(generator, 0)
    values = {
        name: int(offset == planted[group_index])
        for group_index, group in enumerate(groups)
        for offset, name in enumerate(group)
    }
    reference = {"values": values, "objective": -len(groups)}
    reference_path = tmp_path / "reference.solution.json"
    reference_path.write_text(json.dumps(reference))
    record = TrustedOptimumRecord(
        instance_set="judge_id",
        instance_id=instance.stem,
        instance_sha256=hashlib.sha256(instance.read_bytes()).hexdigest(),
        problem_kind="binary_clique_cover",
        objective_sense="minimize",
        optimal_objective=-len(groups),
        reference_solution_uri="private://reference.solution.json",
        reference_solution_sha256=hashlib.sha256(
            reference_path.read_bytes()
        ).hexdigest(),
        optimality_basis={"kind": "binary_clique_cover", "groups": groups},
        generation_provenance={},
        verification_provenance=(
            "pitbench.problem_families.verification:TrustedOptimumFamily"
        ),
    )
    verifier = TrustedOptimumFamily(record, reference_path)

    assert verifier._check_oracle(instance) is None
    assert verifier.verify(instance, reference_path).objective == -len(groups)

    invalid_groups = [*groups[:-1], groups[-1][:-1]]
    invalid_record = record.model_copy(
        update={
            "optimality_basis": {
                "kind": "binary_clique_cover",
                "groups": invalid_groups,
            }
        }
    )
    assert (
        TrustedOptimumFamily(invalid_record, reference_path)
        .verify(instance, reference_path)
        .infrastructure_error
    )

    invalid_solution = dict(reference)
    invalid_values = dict(values)
    invalid_values[groups[0][0]] = 1
    invalid_values[groups[0][1]] = 1
    invalid_solution["values"] = invalid_values
    invalid_solution["objective"] = -len(groups) - 1
    invalid_solution_path = tmp_path / "invalid.solution.json"
    invalid_solution_path.write_text(json.dumps(invalid_solution))
    assert verifier.verify(instance, invalid_solution_path).feasible is False


def test_binary_clique_cover_fme_extension_keeps_a_checkable_optimum(
    tmp_path: Path,
) -> None:
    generator = {
        "kind": "binary_clique_cover_mip",
        "count": 1,
        "groups": [4],
        "group_sizes": [5],
        "coefficient_min": 120,
        "coefficient_max": 190,
        "cross_edge_density": [0.1],
        "fme_auxiliary_pairs": 2,
        "randomness": {"instance_seed": 910001},
    }
    instance_config = {
        "schema_version": "1.0",
        "visibility": "judge",
        "format": "mps",
        "generator": generator,
    }
    instance = materialize_generated_instance_set(
        instance_config,
        tmp_path / "instances",
        expected_visibility="judge",
        stem_prefix="judge_id",
    )[0]
    groups, planted, _ = binary_clique_cover_data(generator, 0)
    extensions = binary_clique_cover_fme_extensions(generator, 0, groups, planted)
    values = {
        name: int(offset == planted[group_index])
        for group_index, group in enumerate(groups)
        for offset, name in enumerate(group)
    }
    values.update({extension["column"]: 0 for extension in extensions})
    objective = -len(groups)
    reference = {"values": values, "objective": objective}
    reference_path = tmp_path / "reference-fme.solution.json"
    reference_path.write_text(json.dumps(reference))
    record = TrustedOptimumRecord(
        instance_set="judge_id",
        instance_id=instance.stem,
        instance_sha256=hashlib.sha256(instance.read_bytes()).hexdigest(),
        problem_kind="binary_clique_cover",
        objective_sense="minimize",
        optimal_objective=objective,
        reference_solution_uri="private://reference-fme.solution.json",
        reference_solution_sha256=hashlib.sha256(
            reference_path.read_bytes()
        ).hexdigest(),
        optimality_basis={
            "kind": "binary_clique_cover",
            "groups": groups,
            "fme_extensions": extensions,
        },
        generation_provenance={
            "kind": "binary_clique_cover_fme_extension",
            "fme_extension_count": len(extensions),
            "optimality_proof": (
                "the clique-cover bound holds for every feasible assignment and "
                "the planted assignment extends with each auxiliary column zero"
            ),
        },
        verification_provenance=(
            "pitbench.problem_families.verification:TrustedOptimumFamily"
        ),
    )
    verifier = TrustedOptimumFamily(record, reference_path)

    assert len(extensions) == 2
    assert all(
        row["rhs"] == 0
        for extension in extensions
        for row in extension["positive_rows"] + extension["negative_rows"]
    )
    assert verifier._check_oracle(instance) is None
    checked = verifier.verify(instance, reference_path)
    assert checked.feasible is True
    assert checked.objective == objective

    bad_extensions = [dict(item) for item in extensions]
    bad_extensions[0] = {
        **bad_extensions[0],
        "positive_rows": [
            {
                **bad_extensions[0]["positive_rows"][0],
                "binary_coefficients": {
                    "not_a_model_column": 1,
                },
            },
            *bad_extensions[0]["positive_rows"][1:],
        ],
    }
    bad_record = record.model_copy(
        update={
            "optimality_basis": {
                "kind": "binary_clique_cover",
                "groups": groups,
                "fme_extensions": bad_extensions,
            }
        }
    )
    assert TrustedOptimumFamily(bad_record, reference_path)._check_oracle(
        instance
    ).infrastructure_error


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
        "highs_binary_clique_presolve_20260921",
        "choco_graph_coloring_v6_0_1",
    ):
        record = catalog.validate_one(task_id)
        oracle_path = record.task.oracle.source.removeprefix("private://")
        oracle = TrustedOptimumOracle.from_yaml(ROOT / "private" / oracle_path)
        judge_config = next(
            instance_set.instance_set_config.removeprefix("private://")
            for instance_set in record.task.instance_sets
            if instance_set.kind.value == "judge_id"
        )
        payload = yaml.safe_load((ROOT / "private" / judge_config).read_text())
        with tempfile.TemporaryDirectory() as temporary:
            paths = materialize_generated_instance_set(
                payload,
                Path(temporary),
                expected_visibility="judge",
                stem_prefix="judge_id",
            )
            records = [
                record for record in oracle.records if record.instance_set == "judge_id"
            ]
            assert len(paths) == len(records) == 30
            for path, record in zip(paths, records, strict=True):
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

    wrapped = tmp_path / "wrapped_knapsack.lp"
    wrapped_lines = []
    for line in knapsack.read_text().splitlines():
        while len(line) > 120:
            split = line.rfind(" ", 0, 120)
            assert split > 0
            wrapped_lines.append(line[:split])
            line = "  " + line[split + 1 :]
        wrapped_lines.append(line)
    wrapped.write_text("\n".join(wrapped_lines) + "\n")
    assert _public_multidimensional_knapsack_data(wrapped) == (weights, capacities)
    solution_path.write_text(
        json.dumps(
            _knapsack_solution(
                reference["selected_items"], len(weights[0]), -sum(capacities)
            )
        )
    )
    assert MIPFamily().verify(wrapped, solution_path).feasible
    changed = tmp_path / "changed_knapsack.lp"
    changed.write_text(wrapped.read_text().replace("<= 1274", "<= 1275", 1))
    changed_data = _public_multidimensional_knapsack_data(changed)
    assert changed_data is not None
    assert changed_data != (weights, capacities)
    assert changed_data[1][0] == 1275

    mps = ROOT / "tests/fixtures/mip/highs_pr577_knapsack_dev_0000.mps"
    assert _public_multidimensional_knapsack_data(mps) == (weights, capacities)
    selected = set(reference["selected_items"])
    legacy_solution = (
        "60 3 : Number of columns and rows for primal or dual solution or basis\n"
        "T Primal solution\nF Dual solution\nF Basis\nColumns\n"
        + "".join(f"{int(item in selected)}\n" for item in range(60))
    )
    mps_solution_path = tmp_path / "legacy_mps.solution.json"
    mps_solution_path.write_text(
        json.dumps({"raw_solution": legacy_solution, "objective": -4542})
    )
    assert MIPFamily().verify(mps, mps_solution_path).feasible
    hidden_oracle = TrustedOptimumOracle.from_yaml(
        ROOT / "private/oracles/highs_mip_row_aggregation_20210916.yaml"
    )
    trusted_record = next(
        record
        for record in hidden_oracle.records
        if record.instance_set == "judge_id" and record.instance_id == "judge_id_0000"
    )
    trusted_reference = (
        ROOT
        / "private"
        / trusted_record.reference_solution_uri.removeprefix("private://")
    )
    hidden_instance = (
        ROOT
        / "private/hidden_instances/highs_pr577_mps/multidimensional_knapsack/instances/judge_id_0000.mps"
    )
    assert (
        TrustedOptimumFamily(trusted_record, trusted_reference)._check_oracle(
            hidden_instance
        )
        is None
    )

    schedule_mps = ROOT / "tests/fixtures/mip/highs_pr577_schedule_dev_0000.mps"
    schedule_reference = json.loads(
        (
            ROOT
            / "configs/instance_sets/highs_v1_15_1_dev/bks_solutions/dev_0000.solution.json"
        ).read_text()
    )
    order = schedule_reference["order"]
    position = {job: index for index, job in enumerate(order)}
    start_times = schedule_reference["start_times"]
    order_variables = [
        int(position[first] < position[second])
        for first in range(len(order))
        for second in range(first + 1, len(order))
    ]
    schedule_values = [*start_times, *order_variables]
    schedule_raw_solution = (
        f"{len(schedule_values)} 72 : Number of columns and rows for primal or dual solution or basis\n"
        "T Primal solution\nF Dual solution\nF Basis\nColumns\n"
        + "".join(f"{value}\n" for value in schedule_values)
    )
    mps_solution_path.write_text(
        json.dumps(
            {
                "raw_solution": schedule_raw_solution,
                "objective": sum(start_times),
            }
        )
    )
    assert MIPFamily().verify(schedule_mps, mps_solution_path).feasible

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
