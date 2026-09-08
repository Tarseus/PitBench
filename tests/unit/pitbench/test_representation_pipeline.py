import hashlib
import json
from pathlib import Path

import pytest

from pitbench.evaluator import runner
from pitbench.evaluator.judge import InstanceCase, JudgePlan, LocalProcessJudge
from pitbench.evaluator.storage import ObservationStore
from pitbench.problem_families.cvrp import CVRPFamily
from pitbench.schema.observation import CodeState, RunObservation, RunStatus
from pitbench.schema.task import PitBenchTask

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("private_panel", "declared agent_dev"),
        ("other_solver", "PyVRP CVRP"),
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
            "pitbench.repositories.vroom:VroomRepositoryPlugin"
        )
    with pytest.raises(ValueError, match=message):
        PitBenchTask.model_validate(payload)


def test_runner_collects_configured_relabelings_and_preserves_failures(
    tmp_path, monkeypatch
):
    task_path = ROOT / "configs/tasks/pyvrp_v0_14_0.yaml"
    task = PitBenchTask.from_yaml(task_path)
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
