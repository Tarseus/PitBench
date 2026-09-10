import sys
from pathlib import Path

import pytest

from pitbench.schema.observation import CodeState, RunObservation, RunStatus
from pitbench.schema.task import PitBenchTask
from scripts.validate_seed_robustness_real_solver import ROOT, _load_checkpoint, main


def _observation(seed: int) -> RunObservation:
    return RunObservation(
        task_id="pyvrp_v0_14_0",
        code_state=CodeState.BASE,
        instance_set="agent_dev",
        instance_set_kind="agent_dev",
        instance_id="instance",
        solver_seed=seed,
        budget_sec=1,
        status=RunStatus.COMPLETED,
        valid=True,
    )


def test_checkpoint_discards_only_a_truncated_final_line(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "observations.checkpoint.jsonl"
    first_observation = _observation(10)
    checkpoint_path.write_text(f'{first_observation.model_dump_json()}\n{{"incomplete"')

    loaded = _load_checkpoint(checkpoint_path)

    assert loaded == [first_observation]
    assert checkpoint_path.read_text() == f"{first_observation.model_dump_json()}\n"


def test_checkpoint_rejects_duplicate_observations(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "observations.checkpoint.jsonl"
    observation_json = _observation(10).model_dump_json()
    checkpoint_path.write_text(f"{observation_json}\n{observation_json}\n")

    with pytest.raises(ValueError, match="duplicate observations"):
        _load_checkpoint(checkpoint_path)


@pytest.mark.parametrize(
    "task_id",
    ["pyvrp_v0_12_2", "pyvrp_v0_13_0", "pyvrp_v0_13_4", "pyvrp_v0_14_0"],
)
def test_runner_accepts_existing_seed_validation_tasks(
    task_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_seed_robustness_real_solver.py",
            "--task-config",
            str(ROOT / "configs" / "tasks" / f"{task_id}.yaml"),
            "--output-dir",
            str(tmp_path),
            "--reference-seed-count",
            "30",
            "--test-seed-count",
            "30",
            "--test-list-count",
            "1",
        ],
    )
    with pytest.raises(ValueError, match="--repository is required"):
        main()
    assert (tmp_path / "validation_seeds.json").exists()


def test_runner_accepts_new_task_identity_with_seed_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = PitBenchTask.from_yaml(ROOT / "configs/tasks/pyvrp_v0_14_0.yaml")
    task = task.model_copy(update={"task_id": "another_repository_release"})
    monkeypatch.setattr(PitBenchTask, "from_yaml", lambda path: task)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_seed_robustness_real_solver.py",
            "--output-dir",
            str(tmp_path),
            "--task-config",
            "task.yaml",
        ],
    )
    with pytest.raises(ValueError, match="--repository is required"):
        main()
    assert (tmp_path / "validation_seeds.json").exists()


def test_runner_rejects_task_without_seed_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_seed_robustness_real_solver.py",
            "--output-dir",
            str(tmp_path),
            "--task-config",
            str(ROOT / "configs/tasks/highs_v1_15_1.yaml"),
        ],
    )
    with pytest.raises(ValueError, match="task does not define Seed Robustness"):
        main()
    assert not (tmp_path / "validation_seeds.json").exists()
