import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from adapters.pitbench.adapter import PitBenchAdapter
from pitbench.evaluator.collection import AnchorCollection
from pitbench.evaluator.judge import LocalProcessJudge
from pitbench.schema.observation import CodeState, RunObservation, RunStatus
from pitbench.schema.task import PitBenchTask

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "task_id,direction,expected",
    [
        ("pyvrp_v0_14_0", "minimize", 7),
        ("highs_v1_15_1", "maximize", 12),
    ],
)
@pytest.mark.parametrize("failed", [False, True])
def test_anchor_collection_uses_task_judge_and_retains_results(
    tmp_path, monkeypatch, task_id, direction, expected, failed
):
    task = PitBenchTask.from_yaml(ROOT / "configs/tasks" / f"{task_id}.yaml")
    task = task.model_copy(
        update={"oracle": task.oracle.model_copy(update={"objective_sense": direction})}
    )
    monkeypatch.setattr(PitBenchTask, "from_yaml", lambda path: task)
    monkeypatch.setattr(PitBenchAdapter, "validate_repository", lambda *args: None)
    private = tmp_path / "private"
    verifier_path = private / task.evaluation.verifier.removeprefix("private://")
    verifier_path.parent.mkdir(parents=True, exist_ok=True)
    verifier_path.touch()
    config = tmp_path / "instances.yaml"
    generator = (
        {
            "kind": "euclidean_cvrp",
            "count": 1,
            "customers": [3],
            "capacity_ratio": 1,
            "randomness": {"coordinate_seed": 1, "demand_seed": 2},
            "distance_metric": "EUC_2D",
        }
        if task.problem_family == "cvrp"
        else {
            "kind": "single_machine_scheduling_mip",
            "count": 1,
            "jobs": [2],
            "randomness": {"instance_seed": 1},
        }
    )
    config.write_text(yaml.safe_dump({"visibility": "judge", "generator": generator}))
    output = private / "oracles" / "custom_panel"

    def run(judge, cases, *, save_observation):
        assert judge.task is task
        assert judge.repository.name == task_id.split("_v")[0]
        assert judge.evaluation_seeds == (8, 3)
        assert judge.code_states == (CodeState.BASE,)
        assert len(cases) == 1
        case = cases[0]
        assert case.budgets_sec == (35,)
        observations = []
        for seed, objective in [(8, 12), (3, 7)]:
            solution = output / f"solution-{seed}.json"
            solution.write_text(json.dumps({"objective": objective}))
            observation = RunObservation(
                task_id=task_id,
                code_state=CodeState.BASE,
                instance_set=case.instance_set.name,
                instance_id=case.instance_id,
                solver_seed=seed,
                budget_sec=35,
                status=RunStatus.INVALID if failed else RunStatus.COMPLETED,
                valid=not failed,
                objective=objective,
                solution_path=str(solution),
            )
            observations.append(observation)
            save_observation(observation)
        return observations

    monkeypatch.setattr(LocalProcessJudge, "run", run)
    argv = [
        "--task-config",
        "task.yaml",
        "--repository",
        str(tmp_path),
        "--instance-set-config",
        str(config),
        "--private-root",
        str(private),
        "--output-dir",
        str(output),
        "--budget-sec",
        "35",
        "--seeds",
        "8",
        "3",
    ]
    if failed:
        with pytest.raises(ValueError, match="invalid BKS candidate"):
            AnchorCollection.main(argv)
        assert not (output / "oracle.yaml").exists()
    else:
        AnchorCollection.main(argv)
        oracle = yaml.safe_load((output / "oracle.yaml").read_text())
        assert oracle["problem_family"] == task.problem_family
        assert oracle["objective_sense"] == direction
        assert oracle["solver"]["version"] == task.release.version
        assert oracle["anchors"][0]["bks"] == expected
        assert (
            oracle["anchors"][0]["bks_solution_uri"]
            == "private://oracles/custom_panel/judge_shift_0000.bks.solution.json"
        )
        assert oracle["protocol"]["seeds"] == [8, 3]
        assert oracle["protocol"]["budget_sec"] == 35
    assert len((output / "observations.jsonl").read_text().splitlines()) == 2
    assert (output / "observations.parquet").exists()


def test_seed_matrix_uses_explicit_tasks_and_their_images(tmp_path):
    batch = tmp_path / "batch with spaces"
    private = tmp_path / "private"
    private.mkdir()
    harness = batch / "harness"
    script = harness / "scripts/validate_seed_robustness_real_solver.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "output=Path(sys.argv[sys.argv.index('--output-dir')+1])\n"
        "(output/'arguments.json').write_text(json.dumps(sys.argv[1:]))\n"
    )
    configs = harness / "configs/tasks"
    configs.mkdir(parents=True)
    for task, image in [
        ("solver_a", "sha256:" + "a" * 64),
        ("solver_b", "sha256:" + "b" * 64),
    ]:
        (batch / "sources" / task / ".git").mkdir(parents=True)
        (configs / f"{task}.yaml").write_text(
            yaml.safe_dump({"repository": {"judge_image": image}})
        )
    environment = {**os.environ, "PITBENCH_PYTHON": sys.executable}
    environment.pop("PITBENCH_JUDGE_IMAGE", None)
    command = [
        "bash",
        str(ROOT / "scripts/run-seed-validation-matrix.sh"),
        str(batch),
        str(private),
    ]
    result = subprocess.run(command, env=environment, capture_output=True, text=True)
    assert result.returncode != 0 and "at least one task ID" in result.stderr
    subprocess.run(
        [*command, "solver_a", "solver_b"],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    for task, letter in [("solver_a", "a"), ("solver_b", "b")]:
        argv = json.loads((batch / task / "arguments.json").read_text())
        assert argv[argv.index("--judge-image") + 1] == "sha256:" + letter * 64
        assert argv[argv.index("--repository") + 1] == str(batch / "sources" / task)
        assert argv[argv.index("--private-root") + 1] == str(private)
