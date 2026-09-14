from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest

from pitbench.evaluator import runner
from pitbench.evaluator.judge import InstanceCase, JudgePlan, LocalProcessJudge
from pitbench.evaluator.representation import preserve_json, result_record, write_json
from pitbench.evaluator.representations import (
    CustomerRepresentation,
    LinearModelRepresentation,
    representation_type,
)
from pitbench.evaluator.storage import ObservationStore
from pitbench.problem_families.verification import (
    CVRPFamily,
    NumericModelFamily,
    _edge_cost,
)
from pitbench.schema.observation import CodeState, RunObservation, RunStatus
from pitbench.schema.task import PitBenchTask

# Tests consolidated from tests/unit/pitbench/test_representations.py


def test_representation_plugins_register_and_reject_duplicate_names():
    assert representation_type("customer_relabeling") is CustomerRepresentation
    with pytest.raises(ValueError, match="duplicate representation plugin"):

        class DuplicateRepresentation(CustomerRepresentation):
            name = "customer_relabeling"


@pytest.fixture
def model():
    csc_matrix = pytest.importorskip("scipy.sparse").csc_matrix
    return {
        "matrix": csc_matrix([[1.0, 2.0, 0.0], [-1.0, 0.0, 1.0]]),
        "cost": np.array([3.0, -2.0, 1.0]),
        "col_lower": np.zeros(3),
        "col_upper": np.array([3.0, 2.0, 4.0]),
        "row_lower": np.array([2.0, -np.inf]),
        "row_upper": np.array([np.inf, 1.5]),
        "integrality": np.array([1, 1, 0]),
        "col_names": np.array(["a", "b", "c"]),
        "row_names": np.array(["demand", "link"]),
        "offset": 7.0,
        "sense": 1,
    }


@pytest.mark.parametrize("sense,expected", [(1, 3.0), (-1, 20.0)])
def test_permuted_mip_has_same_optimum_and_mapped_feasible_solution(
    model, sense, expected
):
    highspy = pytest.importorskip("highspy")
    model["sense"] = sense
    rows, columns = [1, 0], [2, 0, 1]
    transformed = LinearModelRepresentation.permute(model, rows, columns)
    LinearModelRepresentation.assert_equivalent(model, transformed, rows, columns)
    for candidate, mapping in ((model, [0, 1, 2]), (transformed, columns)):
        solver = highspy.Highs()
        solver.setOptionValue("output_flag", False)
        solver.setOptionValue("threads", 1)
        assert (
            solver.passModel(LinearModelRepresentation.to_highs_lp(candidate))
            == highspy.HighsStatus.kOk
        )
        assert solver.run() == highspy.HighsStatus.kOk
        assert solver.getModelStatus() == highspy.HighsModelStatus.kOptimal
        result = NumericModelFamily.check_model(
            model,
            LinearModelRepresentation.map_solution(
                solver.getSolution().col_value, mapping
            ),
            1e-6,
        )
        assert result["feasible"]
        assert result["objective"] == pytest.approx(expected)


@pytest.mark.parametrize("field", ["cost", "col_upper", "row_lower", "integrality"])
def test_equivalence_rejects_semantic_changes(model, field):
    transformed = LinearModelRepresentation.permute(model, [1, 0], [2, 0, 1])
    transformed[field] = transformed[field].copy()
    index = int(np.flatnonzero(np.isfinite(transformed[field]))[0])
    transformed[field][index] += 1
    with pytest.raises(ValueError, match="permutation changed"):
        LinearModelRepresentation.assert_equivalent(
            model, transformed, [1, 0], [2, 0, 1]
        )


@pytest.mark.parametrize(
    "values,violation",
    [
        ([0, 0, 0], "max_row_violation"),
        ([0, 3, 0], "max_bound_violation"),
        ([0, 1.5, 0], "max_integrality_violation"),
    ],
)
def test_independent_verifier_rejects_bad_primal(model, values, violation):
    result = NumericModelFamily.check_model(model, values, 1e-6)
    assert not result["feasible"]
    assert result[violation] > 1e-6


def test_verifier_rejects_nonfinite_and_wrong_length(model):
    for values in ([0, np.nan, 0], [0, np.inf, 0], [0, 2]):
        assert not NumericModelFamily.check_model(model, values, 1e-6)["feasible"]


def test_storage_preserves_objective_sense_offset_and_domains(model, tmp_path):
    path = tmp_path / "model.npz"
    LinearModelRepresentation.save(path, model)
    restored = LinearModelRepresentation.load(path)
    LinearModelRepresentation.assert_equivalent(model, restored, [0, 1], [0, 1, 2])


@pytest.mark.parametrize("mapping", [[0, 0, 2], [0.0, 1.0, 2.0], [0, 1], [0, 1, 3]])
def test_nonbijective_mappings_are_rejected(model, mapping):
    with pytest.raises(ValueError, match="bijection"):
        LinearModelRepresentation.permute(model, [0, 1], mapping)


# Tests consolidated from tests/unit/pitbench/test_representation_collection.py


@pytest.fixture
def instance():
    return {
        "name": "example",
        "depot": 0,
        "distance_metric": "EUC_2D",
        "coordinates": [[0, 0], [1, 1], [3, 0], [3, 4]],
        "demands": [0, 1, 2, 3],
        "capacity": 6,
        "node_ids": [1, 2, 3, 4],
    }


def test_permutations_fix_depot_and_are_distinct_reproducible_nonidentity():
    mappings = CustomerRepresentation.permutations(5, 30, random.Random(20260907))
    assert mappings == CustomerRepresentation.permutations(
        5, 30, random.Random(20260907)
    )
    assert len({tuple(mapping) for mapping in mappings}) == 30
    assert all(
        mapping[0] == 0 and sorted(mapping) == list(range(6)) for mapping in mappings
    )
    assert list(range(6)) not in mappings
    with pytest.raises(ValueError, match="not enough"):
        CustomerRepresentation.permutations(3, 30, random.Random(0))


def test_mapping_preserves_all_edge_costs_demands_and_verified_route(
    instance, tmp_path
):
    mapping = [0, 3, 1, 2]
    transformed = CustomerRepresentation.permute(instance, mapping)
    for first in range(4):
        for second in range(4):
            assert _edge_cost(
                transformed["coordinates"], first, second, "EUC_2D"
            ) == _edge_cost(
                instance["coordinates"], mapping[first], mapping[second], "EUC_2D"
            )
        assert transformed["demands"][first] == instance["demands"][mapping[first]]
    solution = {"routes": [[1, 2, 3]]}
    mapped = CustomerRepresentation.map_solution(solution, mapping)
    assert mapped == {"routes": [[3, 1, 2]]}
    for name, payload in (
        ("original", instance),
        ("transformed", transformed),
        ("solution", solution),
        ("mapped", mapped),
    ):
        write_json(tmp_path / f"{name}.json", payload)
    family = CVRPFamily()
    original_result = family.verify(
        tmp_path / "original.json", tmp_path / "mapped.json"
    )
    transformed_result = family.verify(
        tmp_path / "transformed.json", tmp_path / "solution.json"
    )
    assert original_result.feasible and transformed_result.feasible
    assert original_result.objective == transformed_result.objective
    assert instance["coordinates"][1] == [1, 1]


@pytest.mark.parametrize("mapping", [[1, 0, 2, 3], [0, 1, 1, 3]])
def test_invalid_mapping_is_rejected(instance, mapping):
    with pytest.raises(ValueError, match="bijection"):
        CustomerRepresentation.permute(instance, mapping)


def observation(valid=True):
    return RunObservation(
        task_id="pyvrp_v0_14_0",
        code_state=CodeState.BASE,
        instance_set="agent_dev",
        instance_set_kind="agent_dev",
        instance_id="relabeled",
        solver_seed=0,
        budget_sec=5,
        status=RunStatus.COMPLETED if valid else RunStatus.CRASHED,
        valid=valid,
        error=None if valid else "solver crashed",
        equivalence_parent_id="original",
        equivalence_transform="customer_relabeling_00",
    )


def test_records_keep_mapped_solution_and_failed_run(instance, tmp_path):
    mapping = [0, 3, 1, 2]
    write_json(tmp_path / "original.json", instance)
    write_json(
        tmp_path / "transformed.json", CustomerRepresentation.permute(instance, mapping)
    )
    transformation = {
        "original_instance_path": "original.json",
        "transformed_instance_path": "transformed.json",
        "new_to_original": mapping,
    }
    output = tmp_path / "runs/base/agent_dev/relabeled/seed-0-budget-5.json"
    write_json(output.with_suffix(".solution.json"), {"routes": [[1, 2, 3]]})
    record = result_record(observation(), transformation, tmp_path)
    assert record["verification"]["mapped_original"]["feasible"]
    assert record["verification"]["objective_preserved"]
    assert json.loads(
        (tmp_path / record["artifacts"]["mapped_solution"]).read_text()
    ) == {"routes": [[3, 1, 2]]}

    failed = observation(False).model_copy(update={"budget_sec": 10.0})
    record = result_record(failed, transformation, tmp_path)
    assert record["observation"]["error"] == "solver crashed"
    assert record["artifacts"]["solution"] is None
    assert record["verification"]["objective_preserved"] is None


def test_resume_does_not_overwrite_changed_experiment_inputs(tmp_path):
    path = tmp_path / "mapping.json"
    preserve_json(path, [0, 2, 1])
    preserve_json(path, [0, 2, 1])
    with pytest.raises(ValueError, match="differs"):
        preserve_json(path, [0, 1, 2])
    assert json.loads(path.read_text()) == [0, 2, 1]


# Tests consolidated from tests/unit/pitbench/test_representation_pipeline.py


ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("private_panel", "declared agent_dev"),
        ("other_solver", "not supported"),
    ],
)
def test_representation_configuration_rejects_unsupported_scope(change, message):
    payload = PitBenchTask.from_yaml(
        ROOT / "configs/tasks/pyvrp_v0_14_0.yaml"
    ).model_dump(mode="json")
    if change == "private_panel":
        payload["evaluation"]["representation_robustness"]["instance_set"] = "judge_id"
    else:
        payload["repository"]["plugin"] = (
            "pitbench.repositories.plugins:ChocoRepositoryPlugin"
        )
    with pytest.raises(ValueError, match=message):
        PitBenchTask.model_validate(payload)


def test_runner_collects_configured_relabelings_and_preserves_failures(
    tmp_path, monkeypatch
):
    task_path = ROOT / "configs/tasks/pyvrp_v0_14_0.yaml"
    task = PitBenchTask.from_yaml(task_path)
    # This test isolates relabeling; the combined runner is covered separately.
    task.evaluation.operational_reliability = False
    monkeypatch.setattr(PitBenchTask, "from_yaml", lambda path: task)
    assert task.evaluation.representation_robustness is not None
    original = {
        "depot": 0,
        "coordinates": [[0, 0], [1, 1], [3, 0], [3, 4], [5, 1], [6, 3]],
        "demands": [0, 1, 2, 3, 1, 1],
        "capacity": 10,
        "distance_metric": "EUC_2D",
    }
    input_path = tmp_path / "original.json"
    input_path.write_text(json.dumps(original))
    originals = [
        InstanceCase(
            instance_set=task.instance_sets[0],
            instance_id=f"original-{index}",
            path=input_path,
            anchor=10,
            solver_seeds=(11,),
        )
        for index in range(10)
    ]
    normal_case = InstanceCase(
        instance_set=task.instance_sets[1],
        instance_id="normal",
        path=input_path,
        anchor=10,
        solver_seeds=(11,),
    )

    def plan(cls, current_task, resolver, **kwargs):
        cases = (
            originals
            if len(current_task.instance_sets) == 1
            else [*originals, normal_case]
        )
        return JudgePlan(current_task, cases)

    monkeypatch.setattr(JudgePlan, "from_instance_set_configs", classmethod(plan))
    monkeypatch.setattr(
        "pitbench.evaluator.judge._evaluation_seeds", lambda *args: (11,)
    )
    monkeypatch.setattr(LocalProcessJudge, "_workspace", lambda *args: tmp_path)

    def run_case(self, workspace, case, state, seed, budget, **kwargs):
        output = (
            self.output_dir
            / state.value
            / case.instance_set.name
            / case.instance_id
            / f"seed-{seed}-budget-{budget:g}.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.with_suffix(".stdout.log").write_text("solver output")
        output.with_suffix(".stderr.log").write_text("")
        failed = (
            case.instance_id == "original-0__customer_relabeling_00"
            and state == CodeState.AGENT
            and budget == 10
        )
        objective = None
        if not failed:
            solution = output.with_suffix(".solution.json")
            solution.write_text(json.dumps({"routes": [[1, 2, 3, 4, 5]]}))
            objective = CVRPFamily().verify(case.path, solution).objective
            output.write_text(json.dumps({"objective": objective}))
            output.with_suffix(".trajectory.jsonl").write_text(
                json.dumps({"objective": objective}) + "\n"
            )
        return RunObservation(
            task_id=task.task_id,
            code_state=state,
            instance_set=case.instance_set.name,
            instance_set_kind=case.instance_set.kind.value,
            instance_id=case.instance_id,
            solver_seed=seed,
            budget_sec=budget,
            status=RunStatus.CRASHED if failed else RunStatus.COMPLETED,
            valid=not failed,
            error="solver crashed" if failed else None,
            objective=objective,
            optimal_or_bks=case.anchor,
            normalized_gap=None if failed else (objective - case.anchor) / case.anchor,
            equivalence_parent_id=case.equivalence_parent_id,
            equivalence_transform=case.equivalence_transform,
        )

    monkeypatch.setattr(LocalProcessJudge, "_run_case", run_case)
    output_dir = tmp_path / "output"
    patch_path = tmp_path / "candidate.patch"
    patch_path.write_text("candidate patch supplied by the evaluation request")
    observations_path = output_dir / "judge-observations.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "runner",
            "--task-config",
            str(task_path),
            "--base-repository",
            str(tmp_path),
            "--public-root",
            str(ROOT),
            "--private-root",
            str(tmp_path),
            "--candidate-patch",
            str(patch_path),
            "--output-dir",
            str(output_dir),
            "--observations",
            str(observations_path),
            "--parallel-runs",
            "2",
        ],
    )
    runner.main()

    observations = ObservationStore.read_jsonl(observations_path)
    relabeled = [
        item for item in observations if item.equivalence_parent_id is not None
    ]
    assert len(relabeled) == 1200
    assert len(observations) == 1244
    assert {item.solver_seed for item in relabeled} == {0}
    assert {item.budget_sec for item in relabeled} == {5, 10}
    assert {item.code_state for item in relabeled} == set(CodeState)
    assert sum(not item.valid for item in relabeled) == 1

    directory = output_dir / "representation"
    details = json.loads((directory / "details.json").read_text())
    assert details["expected_run_count"] == details["completed_run_count"] == 1200
    assert details["statistics"] == "deferred"
    assert (
        details["candidate_patch_sha256"]
        == hashlib.sha256(patch_path.read_bytes()).hexdigest()
    )
    transformations = json.loads((directory / details["transformations"]).read_text())
    assert len(transformations) == 300
    records = [
        json.loads(line)
        for line in (directory / details["results"]).read_text().splitlines()
    ]
    assert len(records) == 1200
    for record in records:
        if record["observation"]["valid"]:
            assert record["verification"]["mapped_original"]["feasible"] is True
            assert record["verification"]["objective_preserved"] is True
            assert (directory / record["artifacts"]["mapped_solution"]).is_file()
        else:
            assert record["observation"]["error"] == "solver crashed"
            assert record["artifacts"]["solution"] is None
            assert (
                directory / record["artifacts"]["stdout"]
            ).read_text() == "solver output"
    assert json.loads(input_path.read_text()) == original
