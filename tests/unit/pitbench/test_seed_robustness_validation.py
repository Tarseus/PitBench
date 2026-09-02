from __future__ import annotations

from pitbench.metrics.seed_robustness_validation import (
    analyze_real_seed_validation,
    generate_real_validation_seeds,
)
from pitbench.schema.observation import CodeState, RunObservation, RunStatus


def _observations(reference_seeds: list[int], test_seeds: list[int]):
    observations = []
    all_seeds = [*reference_seeds, *test_seeds]
    for instance_number, instance_scale in ((1, 1.0), (2, 2.0)):
        for seed_index, seed in enumerate(all_seeds):
            gap = instance_scale * seed_index / 1000
            for code_state in CodeState:
                observations.append(
                    RunObservation(
                        task_id="pyvrp_v0_14_0",
                        code_state=code_state,
                        instance_set="agent_dev",
                        instance_set_kind="agent_dev",
                        instance_id=f"instance-{instance_number}",
                        solver_seed=seed,
                        budget_sec=10,
                        status=RunStatus.COMPLETED,
                        valid=True,
                        normalized_gap=gap,
                    )
                )
    return observations


def test_real_validation_seed_generation_is_fixed_and_disjoint() -> None:
    first = generate_real_validation_seeds(
        seed_min=0,
        seed_max=2**32 - 1,
        reference_seed_count=30,
        test_seed_count=30,
        test_list_count=2,
    )
    second = generate_real_validation_seeds(
        seed_min=0,
        seed_max=2**32 - 1,
        reference_seed_count=30,
        test_seed_count=30,
        test_list_count=2,
    )

    assert first == second
    assert len(first.reference_seeds) == 30
    assert len(first.test_seeds) == 30
    assert set(first.reference_seeds).isdisjoint(first.test_seeds)
    assert all(len(seed_list) == 30 for seed_list in first.test_seed_lists)
    assert all(set(seed_list) == set(first.test_seeds) for seed_list in first.test_seed_lists)


def test_real_validation_analyzes_no_change_solver_results() -> None:
    validation_seeds = generate_real_validation_seeds(
        seed_min=0,
        seed_max=2**32 - 1,
        reference_seed_count=30,
        test_seed_count=30,
        test_list_count=1,
    )

    summary = analyze_real_seed_validation(
        _observations(
            validation_seeds.reference_seeds,
            validation_seeds.test_seeds,
        ),
        task_id="pyvrp_v0_14_0",
        budgets_sec=[10],
        validation_seeds=validation_seeds,
    )

    assert summary.instance_count == 2
    budget = summary.by_budget["10"]
    assert budget.base.reference_value > 0
    assert budget.base.estimate_count == 1
    assert budget.base.interval_count == 1
    assert budget.agent == budget.base
    assert budget.change.reference_value == 0
    assert budget.change.mean_estimate == 0
    assert budget.change.empirical_coverage == 1
    assert budget.change.mean_interval_width == 0


def test_real_validation_allows_independent_no_change_runs_to_differ() -> None:
    validation_seeds = generate_real_validation_seeds(
        seed_min=0,
        seed_max=2**32 - 1,
        reference_seed_count=30,
        test_seed_count=30,
        test_list_count=1,
    )
    observations = _observations(
        validation_seeds.reference_seeds,
        validation_seeds.test_seeds,
    )
    observations = [
        (
            observation.model_copy(
                update={"normalized_gap": 2 * observation.normalized_gap}
            )
            if observation.code_state is CodeState.AGENT
            and observation.normalized_gap is not None
            else observation
        )
        for observation in observations
    ]

    summary = analyze_real_seed_validation(
        observations,
        task_id="pyvrp_v0_14_0",
        budgets_sec=[10],
        validation_seeds=validation_seeds,
    )

    assert summary.by_budget["10"].agent != summary.by_budget["10"].base
