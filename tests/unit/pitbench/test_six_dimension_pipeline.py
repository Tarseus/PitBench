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
from pitbench.metrics.behavior_metrics import compute_behavior_metric_report
from pitbench.metrics.decision_metrics import compute_benchmark_decision
from pitbench.metrics.outcome_metrics import compute_outcome_metrics
from pitbench.metrics.sensitivity_metrics import (
    compute_representation_stability,
    compute_sensitivity_report,
)
from pitbench.schema.observation import CodeState, RunObservation, RunStatus
from pitbench.schema.task import PitBenchTask

ROOT = Path(__file__).resolve().parents[3]


def _observation(
    state: CodeState,
    population: str,
    population_kind: str,
    instance: str,
    seed: int,
    *,
    scale: float,
    objective: float,
    gap: float | None,
    wall: float,
    cpu: float,
    rss_mb: int,
    parent: str | None = None,
    status: RunStatus = RunStatus.COMPLETED,
    valid: bool = True,
) -> RunObservation:
    return RunObservation(
        task_id="pyvrp_v0_13_4",
        code_state=state,
        population=population,
        population_kind=population_kind,
        instance_id=instance,
        instance_seed=17,
        solver_seed=seed,
        budget_sec=10.0,
        status=status,
        valid=valid,
        objective=objective,
        normalized_gap=gap,
        wall_time_sec=wall,
        cpu_time_sec=cpu,
        peak_rss_bytes=rss_mb * 1024 * 1024,
        problem_scale=scale,
        equivalence_parent_id=parent,
        equivalence_transform="customer_relabel" if parent else None,
    )


def _complete_panel() -> list[RunObservation]:
    rows: list[RunObservation] = []
    for seed in (0, 1):
        for instance, scale, base_gap, agent_gap in (
            ("id-small", 100.0, 0.10, 0.07),
            ("id-large", 500.0, 0.16, 0.11),
        ):
            rows.extend(
                [
                    _observation(
                        CodeState.BASE,
                        "judge_id",
                        "judge_id",
                        instance,
                        seed,
                        scale=scale,
                        objective=1000 * (1 + base_gap),
                        gap=base_gap,
                        wall=8.0,
                        cpu=7.0,
                        rss_mb=120,
                    ),
                    _observation(
                        CodeState.AGENT,
                        "judge_id",
                        "judge_id",
                        instance,
                        seed,
                        scale=scale,
                        objective=1000 * (1 + agent_gap),
                        gap=agent_gap,
                        wall=6.0,
                        cpu=5.0,
                        rss_mb=110,
                    ),
                ]
            )
        rows.extend(
            [
                _observation(
                    CodeState.BASE,
                    "judge_shift",
                    "judge_shift",
                    "shift-large",
                    seed,
                    scale=500.0,
                    objective=1200.0,
                    gap=None,
                    wall=9.0,
                    cpu=8.0,
                    rss_mb=125,
                ),
                _observation(
                    CodeState.AGENT,
                    "judge_shift",
                    "judge_shift",
                    "shift-large",
                    seed,
                    scale=500.0,
                    objective=1120.0,
                    gap=None,
                    wall=7.0,
                    cpu=6.0,
                    rss_mb=115,
                ),
            ]
        )
        for state, gap, transformed_gap, wall, transformed_wall in (
            (CodeState.BASE, 0.10, 0.12, 8.0, 8.5),
            (CodeState.AGENT, 0.07, 0.08, 6.0, 6.2),
        ):
            rows.append(
                _observation(
                    state,
                    "judge_id",
                    "judge_id",
                    "id-small__equiv_relabel",
                    seed,
                    scale=100.0,
                    objective=1000 * (1 + transformed_gap),
                    gap=transformed_gap,
                    wall=transformed_wall,
                    cpu=transformed_wall,
                    rss_mb=120 if state == CodeState.BASE else 110,
                    parent="id-small",
                )
            )
    return rows


def test_complete_panel_populates_every_sensitivity_cell_and_verdict() -> None:
    observations = _complete_panel()
    outcomes = compute_outcome_metrics(observations)
    sensitivity = compute_sensitivity_report(observations)
    behavior = compute_behavior_metric_report(observations)
    task = PitBenchTask.from_yaml(ROOT / "manifests/tasks/pyvrp_v0_13_4.yaml")
    decision = compute_benchmark_decision(
        outcomes,
        sensitivity,
        task.evaluation.decision,
        validity_accepted=True,
    )

    matrix = sensitivity.matrix
    assert all(
        value is not None
        for row in (matrix.equivalence, matrix.scale, matrix.population)
        for value in (row.performance, row.reliability, row.resource)
    )
    assert outcomes.base.reliability.total_runs == 4
    assert outcomes.agent.reliability.total_runs == 4
    assert {item.population for item in behavior.slices} == {
        "judge_id",
        "judge_shift",
    }
    shift_behavior = next(
        item for item in behavior.slices if item.population == "judge_shift"
    )
    assert shift_behavior.performance.distance is None
    assert "non-empty samples" in (shift_behavior.performance.reason or "")
    assert shift_behavior.reliability.distance is not None
    assert shift_behavior.resource.distance is not None
    assert decision.sensitivity_complete
    assert decision.is_resolved
    assert decision.classification == "improved"


def test_representation_reliability_keeps_invalid_runs() -> None:
    original = _observation(
        CodeState.AGENT,
        "judge_id",
        "judge_id",
        "original",
        0,
        scale=100,
        objective=100,
        gap=0.1,
        wall=1,
        cpu=1,
        rss_mb=10,
    )
    transformed = _observation(
        CodeState.AGENT,
        "judge_id",
        "judge_id",
        "transformed",
        0,
        scale=100,
        objective=0,
        gap=None,
        wall=1,
        cpu=1,
        rss_mb=10,
        parent="original",
        status=RunStatus.INVALID,
        valid=False,
    )

    result = compute_representation_stability([original, transformed])

    assert result.has_transforms
    assert result.mean_status_mismatch == 1.0


@pytest.mark.parametrize(
    "task_id",
    (
        "pyvrp_v0_12_2",
        "pyvrp_v0_13_0",
        "pyvrp_v0_13_4",
        "pyvrp_v0_14_0",
    ),
)
def test_pyvrp_plan_materializes_complete_six_dimension_panel(
    tmp_path: Path, task_id: str
) -> None:
    task = PitBenchTask.from_yaml(ROOT / f"manifests/tasks/{task_id}.yaml")
    resolver = PrivateAssetResolver(ROOT / "private")
    plan = JudgePlan.from_private_manifests(
        task,
        resolver,
        generated_root=tmp_path / "generated",
    ).with_equivalence_panel(tmp_path / "equivalence")

    originals = [case for case in plan.cases if case.equivalence_parent_id is None]
    transforms = [case for case in plan.cases if case.equivalence_parent_id is not None]
    assert len(originals) == 48
    assert len(transforms) == 8
    assert sum(case.population.name == "judge_shift" for case in originals) == 10
    assert all(case.problem_scale is not None for case in plan.cases)
    assert all(case.budgets_sec == (5.0,) for case in transforms)
    shift_instances = [
        case for case in originals if case.population.name == "judge_shift"
    ]
    assert all(
        json.loads(case.path.read_text())["distance_metric"] == "EUC_2D"
        for case in shift_instances
        if case.path is not None
    )

    observations = [
        observation.model_copy(
            update={
                "cpu_time_sec": observation.budget_sec
                * (1.0 if observation.code_state == CodeState.BASE else 0.82),
                "peak_rss_bytes": (
                    128 if observation.code_state == CodeState.BASE else 112
                )
                * 1024
                * 1024,
            }
        )
        for observation in FixtureJudge().run(plan)
    ]
    transformed_observations = [
        observation
        for observation in observations
        if observation.equivalence_parent_id is not None
    ]
    shift_observations = [
        observation
        for observation in observations
        if observation.population_kind == "judge_shift"
        and observation.equivalence_parent_id is None
    ]
    assert len(observations) == 1520
    assert {observation.budget_sec for observation in transformed_observations} == {
        5.0
    }
    assert {observation.solver_seed for observation in transformed_observations} == {
        0,
        1,
        2,
        3,
        4,
    }
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

    population_manifest = yaml.safe_load(
        (ROOT / "private/populations/pyvrp_cvrp_shift_v1.yaml").read_text()
    )
    oracle_spec = population_manifest["oracle"]
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

    outcomes = compute_outcome_metrics(observations)
    sensitivity = compute_sensitivity_report(observations)
    behavior = compute_behavior_metric_report(observations)
    decision = compute_benchmark_decision(
        outcomes,
        sensitivity,
        task.evaluation.decision,
        validity_accepted=True,
    )
    assert len(behavior.slices) == 6
    assert all(
        coordinate.distance is not None
        for item in behavior.slices
        for coordinate in (item.performance, item.reliability, item.resource)
    )
    assert all(
        value is not None
        for row in (
            sensitivity.matrix.equivalence,
            sensitivity.matrix.scale,
            sensitivity.matrix.population,
        )
        for value in (row.performance, row.reliability, row.resource)
    )
    assert decision.sensitivity_complete
    assert decision.is_resolved
    assert decision.classification == "improved"


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
