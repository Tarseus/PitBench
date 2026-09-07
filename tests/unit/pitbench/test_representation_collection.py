import json
import random

import pytest

from pitbench.problem_families.cvrp import CVRPFamily, _edge_cost
from pitbench.schema.observation import CodeState, RunObservation, RunStatus
from scripts.collect_representation_results import (
    customer_permutations,
    load_results,
    map_solution,
    preserve_json,
    relabel_instance,
    result_record,
    write_json,
)


@pytest.fixture
def instance():
    return {
        "name": "example", "depot": 0, "distance_metric": "EUC_2D",
        "coordinates": [[0, 0], [1, 1], [3, 0], [3, 4]],
        "demands": [0, 1, 2, 3], "capacity": 6, "node_ids": [1, 2, 3, 4],
    }


def test_permutations_fix_depot_and_are_distinct_reproducible_nonidentity():
    mappings = customer_permutations(5, 30, random.Random(20260907))
    assert mappings == customer_permutations(5, 30, random.Random(20260907))
    assert len({tuple(mapping) for mapping in mappings}) == 30
    assert all(mapping[0] == 0 and sorted(mapping) == list(range(6)) for mapping in mappings)
    assert list(range(6)) not in mappings
    with pytest.raises(ValueError, match="not enough"):
        customer_permutations(3, 30, random.Random(0))


def test_mapping_preserves_all_edge_costs_demands_and_verified_route(instance, tmp_path):
    mapping = [0, 3, 1, 2]
    transformed = relabel_instance(instance, mapping)
    for first in range(4):
        for second in range(4):
            assert _edge_cost(transformed["coordinates"], first, second, "EUC_2D") == _edge_cost(
                instance["coordinates"], mapping[first], mapping[second], "EUC_2D"
            )
        assert transformed["demands"][first] == instance["demands"][mapping[first]]
    solution = {"routes": [[1, 2, 3]]}
    mapped = map_solution(solution, mapping)
    assert mapped == {"routes": [[3, 1, 2]]}
    for name, payload in (("original", instance), ("transformed", transformed),
                          ("solution", solution), ("mapped", mapped)):
        write_json(tmp_path / f"{name}.json", payload)
    family = CVRPFamily()
    original_result = family.verify(tmp_path / "original.json", tmp_path / "mapped.json")
    transformed_result = family.verify(tmp_path / "transformed.json", tmp_path / "solution.json")
    assert original_result.feasible and transformed_result.feasible
    assert original_result.objective == transformed_result.objective
    assert instance["coordinates"][1] == [1, 1]


@pytest.mark.parametrize("mapping", [[1, 0, 2, 3], [0, 1, 1, 3]])
def test_invalid_mapping_is_rejected(instance, mapping):
    with pytest.raises(ValueError, match="bijection"):
        relabel_instance(instance, mapping)


def observation(valid=True):
    return RunObservation(
        task_id="pyvrp_v0_14_0", code_state=CodeState.BASE,
        instance_set="agent_dev", instance_set_kind="agent_dev", instance_id="relabeled",
        solver_seed=0, budget_sec=5, status=RunStatus.COMPLETED if valid else RunStatus.CRASHED,
        valid=valid, error=None if valid else "solver crashed",
        equivalence_parent_id="original", equivalence_transform="customer_relabeling_00",
    )


def test_records_keep_mapped_solution_and_failed_run(instance, tmp_path):
    mapping = [0, 3, 1, 2]
    write_json(tmp_path / "original.json", instance)
    write_json(tmp_path / "transformed.json", relabel_instance(instance, mapping))
    transformation = {"original_instance_path": "original.json",
                      "transformed_instance_path": "transformed.json", "new_to_original": mapping}
    output = tmp_path / "runs/base/agent_dev/relabeled/seed-0-budget-5.json"
    write_json(output.with_suffix(".solution.json"), {"routes": [[1, 2, 3]]})
    record = result_record(observation(), transformation, tmp_path)
    assert record["verification"]["mapped_original"]["feasible"]
    assert record["verification"]["objective_preserved"]
    assert json.loads((tmp_path / record["artifacts"]["mapped_solution"]).read_text()) == {"routes": [[3, 1, 2]]}

    failed = observation(False).model_copy(update={"budget_sec": 10.0})
    record = result_record(failed, transformation, tmp_path)
    assert record["observation"]["error"] == "solver crashed"
    assert record["artifacts"]["solution"] is None
    assert record["verification"]["objective_preserved"] is None


def test_checkpoint_retains_failures_and_recovers_partial_final_write(tmp_path):
    path = tmp_path / "results.jsonl"
    record = {"observation": observation(False).model_dump(mode="json")}
    path.write_text(json.dumps(record) + '\n{"observation":')
    assert load_results(path) == [record]
    assert path.read_text() == json.dumps(record) + "\n"
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="duplicate run"):
        load_results(path)


def test_resume_does_not_overwrite_changed_experiment_inputs(tmp_path):
    path = tmp_path / "mapping.json"
    preserve_json(path, [0, 2, 1])
    preserve_json(path, [0, 2, 1])
    with pytest.raises(ValueError, match="differs"):
        preserve_json(path, [0, 1, 2])
    assert json.loads(path.read_text()) == [0, 2, 1]
