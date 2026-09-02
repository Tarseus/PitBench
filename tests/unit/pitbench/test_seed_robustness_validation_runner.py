from pathlib import Path

import pytest

from pitbench.schema.observation import CodeState, RunObservation, RunStatus
from scripts.validate_seed_robustness_real_solver import _load_checkpoint


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
    checkpoint_path.write_text(
        f'{first_observation.model_dump_json()}\n{{"incomplete"'
    )

    loaded = _load_checkpoint(checkpoint_path)

    assert loaded == [first_observation]
    assert checkpoint_path.read_text() == f"{first_observation.model_dump_json()}\n"


def test_checkpoint_rejects_duplicate_observations(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "observations.checkpoint.jsonl"
    observation_json = _observation(10).model_dump_json()
    checkpoint_path.write_text(f"{observation_json}\n{observation_json}\n")

    with pytest.raises(ValueError, match="duplicate observations"):
        _load_checkpoint(checkpoint_path)
