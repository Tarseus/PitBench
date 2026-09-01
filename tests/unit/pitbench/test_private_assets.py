import hashlib
from pathlib import Path

import pytest
import yaml

from pitbench.evaluator.judge import JudgePlan
from pitbench.evaluator.private_assets import PrivateAssetError, PrivateAssetResolver
from pitbench.schema.task import PitBenchTask

ROOT = Path(__file__).resolve().parents[3]


def test_private_asset_hash_is_verified(tmp_path: Path) -> None:
    asset = tmp_path / "oracle.json"
    asset.write_bytes(b"trusted")
    expected = hashlib.sha256(b"trusted").hexdigest()
    resolver = PrivateAssetResolver(tmp_path)
    assert resolver.resolve("private://oracle.json", expected) == asset
    with pytest.raises(PrivateAssetError, match="hash mismatch"):
        resolver.resolve("private://oracle.json", "0" * 64)


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
    development, judge, *_ = task.instance_sets
    task = task.model_copy(
        update={
            "instance_sets": [
                development,
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

    plan = JudgePlan.from_private_instance_set_configs(task, PrivateAssetResolver(tmp_path))

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

    plan = JudgePlan.from_private_instance_set_configs(task, PrivateAssetResolver(tmp_path))

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

    with pytest.raises(ValueError, match="incomplete instance"):
        JudgePlan.from_private_instance_set_configs(task, PrivateAssetResolver(tmp_path))
