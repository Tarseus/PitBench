from __future__ import annotations

import math

import pytest

from pitbench.metrics.outcomes import (
    PerformanceOutcome,
    ReliabilityOutcome,
    ResourceOutcome,
    performance_outcome,
    performance_outcome_distance,
    reliability_outcome,
    reliability_outcome_distance,
    resource_outcome,
    resource_outcome_distance,
)
from pitbench.metrics.solver_behavior import (
    EmpiricalBehaviorKernel,
    EvaluationContext,
    compare_response_geometries,
    empirical_dispersion,
    empirical_kernel_from_observations,
    empirical_solver_distance,
    empirical_wasserstein_distance,
    induced_behavior_geometry,
    solver_lipschitz_sensitivity,
)
from pitbench.schema.observation import CodeState, RunObservation, RunStatus


def _observation(
    *,
    code_state: CodeState = CodeState.BASE,
    instance_id: str = "x",
    instance_seed: int = 1,
    solver_seed: int = 0,
    status: RunStatus = RunStatus.COMPLETED,
    valid: bool = True,
    normalized_gap: float | None = 0.0,
    wall_time_sec: float | None = 2.0,
    budget_sec: float = 10.0,
) -> RunObservation:
    return RunObservation(
        task_id="task",
        code_state=code_state,
        population="judge_id",
        instance_id=instance_id,
        instance_seed=instance_seed,
        solver_seed=solver_seed,
        budget_sec=budget_sec,
        status=status,
        valid=valid,
        normalized_gap=normalized_gap,
        wall_time_sec=wall_time_sec,
    )


def _context() -> EvaluationContext:
    return EvaluationContext(
        task_id="task", population="judge_id", budget_sec=10.0, threads=1
    )


def _scalar_kernel(
    solver_id: str, samples: dict[str, tuple[float, ...]]
) -> EmpiricalBehaviorKernel[float]:
    return EmpiricalBehaviorKernel(
        solver_id=solver_id,
        context=_context(),
        samples_by_instance=samples,
    )


def _absolute(left: float, right: float) -> float:
    return abs(left - right)


def test_outcome_geometries_are_separate_and_versionable() -> None:
    completed = _observation(normalized_gap=0.25, wall_time_sec=2.5)
    failed = _observation(
        status=RunStatus.TIMED_OUT,
        valid=False,
        normalized_gap=None,
        wall_time_sec=10.0,
    )

    assert performance_outcome(completed) == PerformanceOutcome(0.25)
    assert performance_outcome(failed) is None
    assert reliability_outcome(failed) == ReliabilityOutcome(RunStatus.TIMED_OUT, False)
    assert resource_outcome(completed) == ResourceOutcome(0.25)
    assert performance_outcome_distance(
        PerformanceOutcome(0.25), PerformanceOutcome(0.5)
    ) == pytest.approx(0.25)
    assert (
        reliability_outcome_distance(
            ReliabilityOutcome(RunStatus.COMPLETED, True),
            ReliabilityOutcome(RunStatus.TIMED_OUT, False),
        )
        == 1
    )
    assert resource_outcome_distance(
        ResourceOutcome(0.25), ResourceOutcome(0.75)
    ) == pytest.approx(0.5)


def test_empirical_wasserstein_handles_equal_and_unequal_seed_counts() -> None:
    assert empirical_wasserstein_distance([0.0, 2.0], [1.0, 3.0], _absolute) == (
        pytest.approx(1.0)
    )
    assert empirical_wasserstein_distance(
        [0.0], [0.0, 2.0], _absolute, p=2
    ) == pytest.approx(math.sqrt(2))
    assert empirical_wasserstein_distance([4.0], [9.0], _absolute, p=1) == 5


def test_empirical_wasserstein_satisfies_symmetry_and_triangle_inequality() -> None:
    first = [0.0, 2.0]
    second = [1.0]
    third = [3.0, 4.0, 5.0]

    first_second = empirical_wasserstein_distance(first, second, _absolute)
    second_first = empirical_wasserstein_distance(second, first, _absolute)
    second_third = empirical_wasserstein_distance(second, third, _absolute)
    first_third = empirical_wasserstein_distance(first, third, _absolute)

    assert first_second == pytest.approx(second_first)
    assert first_third <= first_second + second_third + 1e-12


def test_solver_distance_is_population_lp_of_conditional_wasserstein() -> None:
    left = _scalar_kernel("A", {"x": (0.0,), "y": (0.0,)})
    right = _scalar_kernel("B", {"x": (1.0,), "y": (3.0,)})

    uniform = empirical_solver_distance(left, right, lambda a, b: abs(a - b), p=2)
    weighted = empirical_solver_distance(
        left,
        right,
        lambda a, b: abs(a - b),
        p=2,
        population_weights={"x": 0.75, "y": 0.25},
    )

    assert uniform.distance == pytest.approx(math.sqrt(5))
    assert uniform.per_instance == {"x": 1.0, "y": 3.0}
    assert weighted.distance == pytest.approx(math.sqrt(3))

    with pytest.raises(ValueError, match="support weights must be finite and positive"):
        empirical_solver_distance(
            left,
            right,
            _absolute,
            population_weights={"x": 1.0, "y": 0.0},
        )


def test_solver_distance_is_zero_for_equal_empirical_kernels() -> None:
    left = _scalar_kernel("different-code-A", {"x": (0.0, 2.0)})
    right = _scalar_kernel("different-code-B", {"x": (2.0, 0.0)})

    result = empirical_solver_distance(left, right, lambda a, b: abs(a - b))

    assert result.distance == pytest.approx(0)


def test_observation_adapter_preserves_undefined_conditional_performance() -> None:
    rows = [
        _observation(instance_id="ok", normalized_gap=0.2),
        _observation(
            instance_id="failed",
            instance_seed=2,
            status=RunStatus.CRASHED,
            valid=False,
            normalized_gap=None,
        ),
    ]

    kernel = empirical_kernel_from_observations(rows, performance_outcome)

    assert kernel.samples_by_instance["ok"] == (PerformanceOutcome(0.2),)
    assert kernel.samples_by_instance["failed"] == ()
    with pytest.raises(ValueError, match="undefined at instance 'failed'"):
        empirical_solver_distance(kernel, kernel, performance_outcome_distance)


def test_reliability_distance_compares_failure_probability() -> None:
    baseline = empirical_kernel_from_observations(
        [_observation(solver_seed=0), _observation(solver_seed=1)],
        reliability_outcome,
        solver_id="baseline",
    )
    candidate = empirical_kernel_from_observations(
        [
            _observation(code_state=CodeState.AGENT, solver_seed=0),
            _observation(
                code_state=CodeState.AGENT,
                solver_seed=1,
                status=RunStatus.TIMED_OUT,
                valid=False,
                normalized_gap=None,
            ),
        ],
        reliability_outcome,
        solver_id="candidate",
    )

    result = empirical_solver_distance(
        baseline, candidate, reliability_outcome_distance, p=1
    )

    assert result.distance == pytest.approx(0.5)


def test_dispersion_behavior_geometry_and_sensitivity_are_distinct() -> None:
    kernel = _scalar_kernel("A", {"x": (0.0,), "y": (2.0,)})

    assert empirical_dispersion([0.0, 2.0], _absolute, p=2) == pytest.approx(
        math.sqrt(2)
    )
    assert induced_behavior_geometry(kernel, _absolute) == {("x", "y"): 2.0}

    sensitivity = solver_lipschitz_sensitivity(
        kernel, lambda left, right: 0.5, _absolute
    )

    assert sensitivity.lipschitz_constant == pytest.approx(4.0)
    assert sensitivity.witness == ("x", "y")


def test_zero_instance_distance_gives_infinite_sensitivity() -> None:
    kernel = _scalar_kernel("A", {"x": (0.0,), "y": (1.0,)})

    result = solver_lipschitz_sensitivity(
        kernel, lambda left, right: 0.0, lambda left, right: abs(left - right)
    )

    assert result.lipschitz_constant == math.inf


def test_response_geometry_comparison_finds_different_separation() -> None:
    left = _scalar_kernel("A", {"x": (0.0,), "y": (2.0,)})
    right = _scalar_kernel("B", {"x": (0.0,), "y": (1.0,)})

    comparison = compare_response_geometries(
        left, right, lambda first, second: abs(first - second)
    )

    assert comparison.sup_distance == pytest.approx(1.0)
    assert comparison.per_pair_difference == {("x", "y"): 1.0}


def test_kernel_builder_rejects_mixed_evaluation_contexts_and_duplicate_seeds() -> None:
    with pytest.raises(ValueError, match="share task, population, budget, and threads"):
        empirical_kernel_from_observations(
            [_observation(), _observation(instance_id="y", budget_sec=20)],
            reliability_outcome,
        )
    with pytest.raises(ValueError, match="duplicate observation"):
        empirical_kernel_from_observations(
            [_observation(), _observation()], reliability_outcome
        )
