import hashlib
from pathlib import Path

import pytest
import yaml

from pitbench.evaluator.judge import JudgePlan
from pitbench.evaluator.private_assets import (
    PrivateAssetError,
    PrivateAssetResolver,
    PrivateSeedRobustnessConfig,
    load_private_seed_robustness_config,
    validate_private_seed_robustness_config,
)
from pitbench.schema.task import PitBenchTask, SeedRobustnessConfig

ROOT = Path(__file__).resolve().parents[3]


def test_private_asset_hash_is_verified(tmp_path: Path) -> None:
    asset = tmp_path / "oracle.json"
    asset.write_bytes(b"trusted")
    expected = hashlib.sha256(b"trusted").hexdigest()
    resolver = PrivateAssetResolver(tmp_path)
    assert resolver.resolve("private://oracle.json", expected) == asset
    with pytest.raises(PrivateAssetError, match="hash mismatch"):
        resolver.resolve("private://oracle.json", "0" * 64)


def _public_seed_robustness_config() -> SeedRobustnessConfig:
    return SeedRobustnessConfig.model_validate(
        {
            "development_seeds": list(range(30)),
            "evaluation_seeds_file": "private://seed_robustness/task.yaml",
            "evaluation_seeds_file_sha256": "a" * 64,
            "seed_selection": {
                "seed_min": 0,
                "seed_max": 2**32 - 1,
                "seed_count": 30,
            },
        }
    )


def test_private_seed_file_provides_hidden_evaluation_seeds(tmp_path: Path) -> None:
    private_seed_file = tmp_path / "task.yaml"
    private_seed_file.write_text(
        yaml.safe_dump(
            {
                "task_id": "task",
                "evaluation_seeds": list(range(100, 130)),
            }
        )
    )

    private_config = PrivateSeedRobustnessConfig.from_yaml(private_seed_file)
    validate_private_seed_robustness_config(
        private_config,
        task_id="task",
        public_config=_public_seed_robustness_config(),
    )

    assert private_config.evaluation_seeds == list(range(100, 130))


@pytest.mark.parametrize(
    "task_id",
    (
        "pyvrp_v0_12_2",
        "pyvrp_v0_13_0",
        "pyvrp_v0_13_4",
        "pyvrp_v0_14_0",
    ),
)
def test_pyvrp_private_evaluation_seeds_match_public_config(task_id: str) -> None:
    task = PitBenchTask.from_yaml(ROOT / f"configs/tasks/{task_id}.yaml")
    public_config = task.evaluation.seed_robustness
    assert public_config is not None

    private_config = load_private_seed_robustness_config(
        PrivateAssetResolver(ROOT / "private"),
        task_id=task_id,
        public_config=public_config,
    )

    assert len(private_config.evaluation_seeds) == 30
    assert not set(private_config.evaluation_seeds) & set(
        public_config.development_seeds
    )


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ("wrong_task", "task_id does not match"),
        ("wrong_count", "must contain seed_count"),
        ("outside_range", "must belong to the seed range"),
        ("overlap", "must be disjoint"),
    ),
)
def test_private_seed_file_must_match_public_parameters(
    change: str, message: str
) -> None:
    private_config = PrivateSeedRobustnessConfig.model_validate(
        {
            "task_id": "task",
            "evaluation_seeds": list(range(100, 130)),
        }
    )
    if change == "wrong_task":
        private_config = private_config.model_copy(update={"task_id": "other"})
    elif change == "wrong_count":
        private_config = private_config.model_copy(
            update={"evaluation_seeds": list(range(100, 129))}
        )
    elif change == "outside_range":
        private_config = private_config.model_copy(
            update={"evaluation_seeds": [*range(100, 129), 2**32]}
        )
    elif change == "overlap":
        private_config = private_config.model_copy(
            update={"evaluation_seeds": [0, *range(100, 129)]}
        )

    with pytest.raises(ValueError, match=message):
        validate_private_seed_robustness_config(
            private_config,
            task_id="task",
            public_config=_public_seed_robustness_config(),
        )


def test_private_seed_file_rejects_duplicate_evaluation_seeds() -> None:
    with pytest.raises(ValueError, match="evaluation_seeds must be unique"):
        PrivateSeedRobustnessConfig.model_validate(
            {
                "task_id": "task",
                "evaluation_seeds": [100, 100],
            }
        )


def test_judge_plan_accepts_calibration_oracle_config(tmp_path: Path) -> None:
    instance = tmp_path / "instance.json"
    instance.write_text("{}")
    oracle = tmp_path / "oracle.yaml"
    oracle.write_text(
        yaml.safe_dump(
            {
                "instances": [
                    {
                        "id": "X-n101-k25",
                        "instance_uri": "private://instance.json",
                        "bks": 27591,
                    }
                ]
            }
        )
    )
    task = PitBenchTask.from_yaml(ROOT / "configs/tasks/pyvrp_v0_13_4.yaml")
    _development, judge, *_ = task.instance_sets
    task = task.model_copy(
        update={
            "instance_sets": [
                judge.model_copy(
                    update={
                        "instance_set_config": "private://oracle.yaml",
                        "instance_set_config_sha256": None,
                        "size": 1,
                    }
                ),
            ]
        }
    )

    plan = JudgePlan.from_instance_set_configs(
        task,
        PrivateAssetResolver(tmp_path),
        public_root=ROOT,
    )

    assert len(plan.cases) == 1
    assert plan.cases[0].path == instance
    assert plan.cases[0].anchor == 27591


def test_judge_plan_allows_missing_anchor_for_non_objective_tasks(
    tmp_path: Path,
) -> None:
    instance = tmp_path / "instance.json"
    instance.write_text("{}")
    instance_set_config = tmp_path / "instance-set-config.yaml"
    instance_set_config.write_text(
        yaml.safe_dump(
            {"instances": [{"id": "model-1", "uri": "private://instance.json"}]}
        )
    )
    task = PitBenchTask.from_yaml(ROOT / "configs/tasks/ortools_v9_15.yaml")
    development, judge, *_ = task.instance_sets
    task = task.model_copy(
        update={
            "instance_sets": [
                development,
                judge.model_copy(
                    update={
                        "instance_set_config": "private://instance-set-config.yaml",
                        "instance_set_config_sha256": None,
                        "size": 1,
                    }
                ),
            ]
        }
    )

    plan = JudgePlan.from_instance_set_configs(
        task,
        PrivateAssetResolver(tmp_path),
        public_root=ROOT,
    )

    assert len(plan.cases) == 1
    assert plan.cases[0].path == instance
    assert plan.cases[0].anchor is None


def test_judge_plan_requires_anchor_for_objective_tasks(tmp_path: Path) -> None:
    instance = tmp_path / "instance.json"
    instance.write_text("{}")
    instance_set_config = tmp_path / "instance-set-config.yaml"
    instance_set_config.write_text(
        yaml.safe_dump(
            {
                "instances": [
                    {"id": "routing-1", "instance_uri": "private://instance.json"}
                ]
            }
        )
    )
    task = PitBenchTask.from_yaml(ROOT / "configs/tasks/pyvrp_v0_13_4.yaml")
    _development, judge, *_ = task.instance_sets
    task = task.model_copy(
        update={
            "instance_sets": [
                judge.model_copy(
                    update={
                        "instance_set_config": "private://instance-set-config.yaml",
                        "instance_set_config_sha256": None,
                        "size": 1,
                    }
                ),
            ]
        }
    )

    with pytest.raises(ValueError, match="incomplete instance"):
        JudgePlan.from_instance_set_configs(
            task,
            PrivateAssetResolver(tmp_path),
            public_root=ROOT,
        )
