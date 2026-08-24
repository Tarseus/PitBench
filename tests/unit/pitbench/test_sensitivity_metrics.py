from __future__ import annotations

from collections.abc import Mapping

import pytest

from pitbench.metrics.sensitivity_metrics import (
    compute_cross_population_retention,
    compute_linear_slope,
    compute_pairwise_dispersion,
    compute_problem_scalability,
    compute_representation_stability,
    compute_seed_stability,
    compute_sensitivity_report,
    format_sensitivity_report_table,
)
from pitbench.schema.observation import CodeState, RunObservation, RunStatus


def _make_obs(
    code_state: CodeState,
    instance_id: str,
    solver_seed: int,
    *,
    population: str = "agent_dev",
    budget_sec: float = 10.0,
    status: RunStatus = RunStatus.COMPLETED,
    valid: bool = True,
    normalized_gap: float | None = 0.05,
    wall_time_sec: float | None = 5.0,
    cpu_time_sec: float | None = 5.0,
    peak_rss_bytes: int | None = 100 * 1024 * 1024,
    model_variables: int | None = None,
) -> RunObservation:
    return RunObservation(
        task_id="test_task",
        code_state=code_state,
        population=population,
        instance_id=instance_id,
        instance_seed=100,
        solver_seed=solver_seed,
        budget_sec=budget_sec,
        status=status,
        valid=valid,
        normalized_gap=normalized_gap,
        wall_time_sec=wall_time_sec,
        cpu_time_sec=cpu_time_sec,
        peak_rss_bytes=peak_rss_bytes,
        model_variables=model_variables,
    )


def test_helper_math_functions() -> None:
    # Pairwise dispersion
    assert compute_pairwise_dispersion([]) is None
    assert compute_pairwise_dispersion([5.0]) == 0.0
    # Values [2.0, 4.0]:
    # diffs: |2-2|=0, |2-4|=2, |4-2|=2, |4-4|=0 -> sum=4, n^2=4 -> 1.0
    assert compute_pairwise_dispersion([2.0, 4.0]) == pytest.approx(1.0)

    # Linear slope
    assert compute_linear_slope([1.0], [2.0]) is None
    assert compute_linear_slope([1.0, 1.0], [2.0, 3.0]) is None
    # y = 2x + 1 -> points (1, 3), (2, 5), (3, 7) -> slope = 2.0
    assert compute_linear_slope([1.0, 2.0, 3.0], [3.0, 5.0, 7.0]) == pytest.approx(2.0)


def test_seed_stability() -> None:
    base_obs = [
        _make_obs(CodeState.BASE, "inst_1", 0, normalized_gap=0.10, wall_time_sec=4.0),
        _make_obs(CodeState.BASE, "inst_1", 1, normalized_gap=0.20, wall_time_sec=6.0),
    ]
    agent_obs = [
        _make_obs(CodeState.AGENT, "inst_1", 0, normalized_gap=0.04, wall_time_sec=2.0),
        _make_obs(CodeState.AGENT, "inst_1", 1, normalized_gap=0.06, wall_time_sec=3.0),
    ]

    stab = compute_seed_stability(base_obs, agent_obs)
    # Base: gaps [0.10, 0.20] -> disp = 0.05
    assert stab.base.mean_gap_dispersion == pytest.approx(0.05)
    # Agent: gaps [0.04, 0.06] -> disp = 0.01
    assert stab.agent.mean_gap_dispersion == pytest.approx(0.01)
    # Delta: agent - base = 0.01 - 0.05 = -0.04 (agent is more stable)
    assert stab.delta_gap_dispersion == pytest.approx(-0.04)


def test_representation_stability_and_matrix_equivalence_none() -> None:
    observations = [
        _make_obs(CodeState.BASE, "inst_1", 0, normalized_gap=0.10),
        _make_obs(CodeState.AGENT, "inst_1", 0, normalized_gap=0.05),
    ]

    # Without transform pairs: Equivalence row MUST be None
    report_no_tf = compute_sensitivity_report(observations, transform_pairs=None)
    assert not report_no_tf.representation_stability.has_transforms
    assert report_no_tf.matrix.equivalence.performance is None
    assert report_no_tf.matrix.equivalence.reliability is None
    assert report_no_tf.matrix.equivalence.resource is None

    # With transform pairs
    tf_obs = [
        _make_obs(
            CodeState.AGENT,
            "inst_orig",
            0,
            normalized_gap=0.05,
            peak_rss_bytes=100 * 1024 * 1024,
        ),
        _make_obs(
            CodeState.AGENT,
            "inst_rot",
            0,
            normalized_gap=0.07,
            peak_rss_bytes=100 * 1024 * 1024,
        ),
    ]
    pairs = [("inst_orig", "inst_rot")]
    rep_stab = compute_representation_stability(tf_obs, transform_pairs=pairs)
    assert rep_stab.has_transforms
    assert rep_stab.pairs_evaluated == 1
    assert rep_stab.mean_gap_movement == pytest.approx(0.02)


def test_problem_scalability() -> None:
    # 1. No descriptor provided and no model_variables -> has_scale_data = False
    obs_no_scale = [
        _make_obs(CodeState.BASE, "inst_1", 0, wall_time_sec=1.0),
        _make_obs(CodeState.AGENT, "inst_1", 0, wall_time_sec=0.5),
    ]
    scal_none = compute_problem_scalability(
        obs_no_scale, obs_no_scale, scale_descriptors=None
    )
    assert not scal_none.base.has_scale_data

    # 2. With frozen scale descriptor (problem mass: 100, 200, 400 nodes)
    scale_desc: Mapping[str, float] = {
        "inst_100": 100.0,
        "inst_200": 200.0,
        "inst_400": 400.0,
    }
    base_obs = [
        _make_obs(
            CodeState.BASE, "inst_100", 0, wall_time_sec=1.0, normalized_gap=0.05
        ),
        _make_obs(
            CodeState.BASE, "inst_200", 0, wall_time_sec=2.0, normalized_gap=0.06
        ),
        _make_obs(
            CodeState.BASE, "inst_400", 0, wall_time_sec=4.0, normalized_gap=0.07
        ),
    ]
    agent_obs = [
        _make_obs(
            CodeState.AGENT, "inst_100", 0, wall_time_sec=0.5, normalized_gap=0.03
        ),
        _make_obs(
            CodeState.AGENT, "inst_200", 0, wall_time_sec=1.0, normalized_gap=0.035
        ),
        _make_obs(
            CodeState.AGENT, "inst_400", 0, wall_time_sec=2.0, normalized_gap=0.04
        ),
    ]

    scal = compute_problem_scalability(
        base_obs, agent_obs, scale_descriptors=scale_desc
    )
    assert scal.base.has_scale_data
    assert scal.base.scales_evaluated == 3
    # log(T) vs log(s) has slope = 1.0 (linear runtime scaling with scale)
    assert scal.base.runtime_scaling_slope == pytest.approx(1.0)
    assert scal.agent.runtime_scaling_slope == pytest.approx(1.0)
    assert scal.delta_runtime_slope == pytest.approx(0.0)


def test_cross_population_gain_retention() -> None:
    # agent_dev: base gap = 0.10, agent gap = 0.05 -> gain = 0.05
    # judge_id:  base gap = 0.12, agent gap = 0.08 -> gain = 0.04
    # retention = 0.04 / 0.05 = 80.0%
    base_obs = [
        _make_obs(CodeState.BASE, "d1", 0, population="agent_dev", normalized_gap=0.10),
        _make_obs(CodeState.BASE, "j1", 0, population="judge_id", normalized_gap=0.12),
    ]
    agent_obs = [
        _make_obs(
            CodeState.AGENT, "d1", 0, population="agent_dev", normalized_gap=0.05
        ),
        _make_obs(CodeState.AGENT, "j1", 0, population="judge_id", normalized_gap=0.08),
    ]

    ret = compute_cross_population_retention(base_obs, agent_obs)
    assert ret.has_multi_population
    assert ret.dev_gap_reduction == pytest.approx(0.05)
    assert ret.eval_gap_reduction == pytest.approx(0.04)
    assert ret.gain_retention == pytest.approx(0.80)
    assert ret.negative_transfer_count == 0
    assert ret.negative_transfer_fraction == pytest.approx(0.0)


def test_complete_sensitivity_report() -> None:
    obs = [
        _make_obs(
            CodeState.BASE,
            "d1",
            0,
            population="agent_dev",
            normalized_gap=0.10,
            wall_time_sec=5.0,
        ),
        _make_obs(
            CodeState.BASE,
            "d1",
            1,
            population="agent_dev",
            normalized_gap=0.12,
            wall_time_sec=6.0,
        ),
        _make_obs(
            CodeState.AGENT,
            "d1",
            0,
            population="agent_dev",
            normalized_gap=0.06,
            wall_time_sec=3.0,
        ),
        _make_obs(
            CodeState.AGENT,
            "d1",
            1,
            population="agent_dev",
            normalized_gap=0.07,
            wall_time_sec=3.5,
        ),
    ]

    report = compute_sensitivity_report(obs)
    assert report.seed_stability.base.instances_evaluated == 1
    assert report.seed_stability.delta_gap_dispersion is not None
    # No transform pairs -> equivalence is None
    assert report.matrix.equivalence.performance is None

    table_str = format_sensitivity_report_table(report)
    assert "Seed Stability" in table_str
    assert "Representation Stability" in table_str
    assert "Problem Scalability" in table_str
    assert "Cross-Population Gain Retention" in table_str
    assert "Sensitivity Summary Matrix" in table_str
