from __future__ import annotations

import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from pitbench.drivers.common import write_result
from pitbench.evaluator.judge import FixtureJudge, JudgePlan
from pitbench.evaluator.private_assets import PrivateAssetResolver
from pitbench.families.cvrp import CVRPFamily
from pitbench.metrics.performance_report import (
    PerformanceClassification,
    compute_performance_report,
)
from pitbench.schema.task import PitBenchTask

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "task_id",
    (
        "pyvrp_v0_12_2",
        "pyvrp_v0_13_0",
        "pyvrp_v0_13_4",
        "pyvrp_v0_14_0",
    ),
)
def test_pyvrp_plan_materializes_performance_first_panel(
    tmp_path: Path, task_id: str
) -> None:
    task = PitBenchTask.from_yaml(ROOT / f"configs/tasks/{task_id}.yaml")
    resolver = PrivateAssetResolver(ROOT / "private")
    plan = JudgePlan.from_private_instance_set_configs(
        task,
        resolver,
        generated_root=tmp_path / "generated",
    )

    assert len(plan.cases) == 48
    assert sum(case.instance_set.name == "judge_shift" for case in plan.cases) == 10
    assert all(case.problem_scale is not None for case in plan.cases)
    shift_instances = [
        case for case in plan.cases if case.instance_set.name == "judge_shift"
    ]
    assert all(
        json.loads(case.path.read_text())["distance_metric"] == "EUC_2D"
        for case in shift_instances
        if case.path is not None
    )

    observations = FixtureJudge().run(plan)
    shift_observations = [
        observation
        for observation in observations
        if observation.instance_set_kind == "judge_shift"
    ]
    assert len(observations) == 1440
    assert all(observation.objective is not None for observation in shift_observations)
    assert all(
        observation.optimal_or_bks is not None
        and observation.normalized_gap is not None
        for observation in shift_observations
    )

    expected_bks = {
        "judge_shift_0000": 1331.0,
        "judge_shift_0001": 1588.0,
        "judge_shift_0002": 1782.0,
        "judge_shift_0003": 1347.0,
        "judge_shift_0004": 1498.0,
        "judge_shift_0005": 1851.0,
        "judge_shift_0006": 1747.0,
        "judge_shift_0007": 1711.0,
        "judge_shift_0008": 1683.0,
        "judge_shift_0009": 1365.0,
    }
    assert {case.instance_id: case.anchor for case in shift_instances} == expected_bks

    instance_set_config = yaml.safe_load(
        (ROOT / "private/instance_sets/pyvrp_cvrp_shift_v1.yaml").read_text()
    )
    oracle_spec = instance_set_config["oracle"]
    oracle_path = resolver.resolve(oracle_spec["uri"], oracle_spec["sha256"])
    oracle = yaml.safe_load(oracle_path.read_text())
    anchors = {item["id"]: item for item in oracle["anchors"]}
    verifier = CVRPFamily()
    for case in shift_instances:
        assert case.path is not None
        anchor = anchors[case.instance_id]
        solution = resolver.resolve(
            anchor["bks_solution_uri"], anchor["bks_solution_sha256"]
        )
        verified = verifier.verify(case.path, solution)
        assert verified.feasible
        assert verified.objective == case.anchor

    performance = compute_performance_report(
        observations,
        primary_budget_sec=task.evaluation.primary_budget_sec,
    )
    assert performance.budgets_sec == [1.0, 5.0, 10.0]
    assert set(performance.by_budget) == {"1", "5", "10"}
    assert performance.primary.paired.paired_instances == 38
    assert performance.classification is PerformanceClassification.IMPROVED
    serialized = str(performance.model_dump())
    assert "judge_shift" not in serialized
    assert "held_out" not in serialized
    assert "solver_seeds" not in serialized


def test_driver_result_includes_terminated_child_resource_usage(tmp_path: Path) -> None:
    children_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "data = bytearray(32 * 1024 * 1024); "
            "print(sum(i * i for i in range(1000000)) + data[0])",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    children_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    child_cpu_delta = (
        children_after.ru_utime
        + children_after.ru_stime
        - children_before.ru_utime
        - children_before.ru_stime
    )
    assert child_cpu_delta > 0

    self_before_write = resource.getrusage(resource.RUSAGE_SELF)
    expected_cpu_floor = (
        self_before_write.ru_utime
        + self_before_write.ru_stime
        + children_after.ru_utime
        + children_after.ru_stime
    )
    expected_rss_floor = (
        max(self_before_write.ru_maxrss, children_after.ru_maxrss) * 1024
    )
    output = tmp_path / "result.json"
    write_result(output, started=time.perf_counter(), valid=True, objective=1.0)

    payload = json.loads(output.read_text())
    assert payload["cpu_time_sec"] >= expected_cpu_floor
    assert payload["peak_rss_bytes"] >= expected_rss_floor
