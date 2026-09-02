from __future__ import annotations

from collections.abc import Iterable

import pytest

from pitbench.metrics.seed_robustness_report import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CROSSED_BOOTSTRAP_METHOD,
    SeedSelectionMetadata,
    compute_seed_robustness_details,
    compute_seed_robustness_report,
    format_seed_robustness_report,
)
from pitbench.schema.observation import CodeState, RunObservation, RunStatus

DEVELOPMENT_SEEDS = tuple(range(30))
EVALUATION_SEEDS = tuple(range(100, 130))


def _seed_selection() -> SeedSelectionMetadata:
    return SeedSelectionMetadata(
        seed_min=0,
        seed_max=2**32 - 1,
        seed_count=30,
    )


def _observation(
    code_state: CodeState,
    instance_set: str,
    instance_set_kind: str,
    instance_id: str,
    solver_seed: int,
    normalized_gap: float | None,
    *,
    budget_sec: float = 10.0,
    valid: bool = True,
    status: RunStatus = RunStatus.COMPLETED,
    equivalence_parent_id: str | None = None,
) -> RunObservation:
    return RunObservation(
        task_id="seed-robustness",
        code_state=code_state,
        instance_set=instance_set,
        instance_set_kind=instance_set_kind,
        instance_id=instance_id,
        instance_seed=17,
        solver_seed=solver_seed,
        budget_sec=budget_sec,
        status=status,
        valid=valid,
        normalized_gap=normalized_gap,
        equivalence_parent_id=equivalence_parent_id,
    )


def _complete_instance(
    instance_set: str,
    instance_set_kind: str,
    instance_id: str,
    seeds: Iterable[int],
    *,
    budget_sec: float,
    gap_scale: float,
    equivalence_parent_id: str | None = None,
) -> list[RunObservation]:
    observations: list[RunObservation] = []
    for seed_index, solver_seed in enumerate(seeds):
        base_gap = gap_scale * seed_index / 100
        observations.extend(
            [
                _observation(
                    CodeState.BASE,
                    instance_set,
                    instance_set_kind,
                    instance_id,
                    solver_seed,
                    base_gap,
                    budget_sec=budget_sec,
                    equivalence_parent_id=equivalence_parent_id,
                ),
                _observation(
                    CodeState.AGENT,
                    instance_set,
                    instance_set_kind,
                    instance_id,
                    solver_seed,
                    base_gap / 2,
                    budget_sec=budget_sec,
                    equivalence_parent_id=equivalence_parent_id,
                ),
            ]
        )
    return observations


def _report(
    observations: list[RunObservation],
    *,
    budgets_sec: tuple[float, ...] = (1.0, 10.0),
):
    return compute_seed_robustness_report(
        observations,
        task_id="seed-robustness",
        budgets_sec=budgets_sec,
        primary_budget_sec=10.0,
        seed_selection=_seed_selection(),
        development_seeds=DEVELOPMENT_SEEDS,
        evaluation_seeds=EVALUATION_SEEDS,
    )


def _details(
    observations: list[RunObservation],
    *,
    budgets_sec: tuple[float, ...] = (1.0, 10.0),
):
    return compute_seed_robustness_details(
        observations,
        task_id="seed-robustness",
        budgets_sec=budgets_sec,
        primary_budget_sec=10.0,
        seed_selection=_seed_selection(),
        development_seeds=DEVELOPMENT_SEEDS,
        evaluation_seeds=EVALUATION_SEEDS,
    )


def test_report_computes_type_7_iqr_at_every_budget_and_instance_set() -> None:
    observations: list[RunObservation] = []
    for budget_sec in (1.0, 10.0):
        observations.extend(
            _complete_instance(
                "development",
                "agent_dev",
                "dev-1",
                DEVELOPMENT_SEEDS,
                budget_sec=budget_sec,
                gap_scale=1,
            )
        )
        observations.extend(
            _complete_instance(
                "in-distribution",
                "judge_id",
                "id-1",
                EVALUATION_SEEDS,
                budget_sec=budget_sec,
                gap_scale=1,
            )
        )
        observations.extend(
            _complete_instance(
                "in-distribution",
                "judge_id",
                "id-2",
                EVALUATION_SEEDS,
                budget_sec=budget_sec,
                gap_scale=2,
                equivalence_parent_id="id-1",
            )
        )

    report = _report(observations)

    assert report.metric == "seed_robustness"
    assert report.budgets_sec == [1.0, 10.0]
    assert set(report.by_instance_set) == {"development", "in-distribution"}
    in_distribution = report.by_instance_set["in-distribution"]
    assert set(in_distribution.by_budget) == {"1", "10"}
    assert in_distribution.primary is in_distribution.by_budget["10"]
    primary = in_distribution.primary
    assert primary.instance_count == 2
    assert primary.base_complete_instance_count == 2
    assert primary.agent_complete_instance_count == 2
    assert primary.paired_complete_instance_count == 2
    assert primary.base.mean_seed_iqr == pytest.approx(0.2175)
    assert primary.agent.mean_seed_iqr == pytest.approx(0.10875)
    assert primary.change.mean_seed_iqr_change == pytest.approx(-0.10875)

    for interval in (
        primary.base.mean_seed_iqr_ci99,
        primary.agent.mean_seed_iqr_ci99,
        primary.change.mean_seed_iqr_change_ci99,
    ):
        assert interval is not None
        assert interval.level == 0.99
        assert interval.method == CROSSED_BOOTSTRAP_METHOD
        assert interval.resamples == BOOTSTRAP_RESAMPLES
        assert interval.bootstrap_seed == BOOTSTRAP_SEED


def test_incomplete_seed_list_is_counted_but_excluded_from_paired_aggregate() -> None:
    observations = _complete_instance(
        "in-distribution",
        "judge_id",
        "paired",
        EVALUATION_SEEDS,
        budget_sec=10,
        gap_scale=1,
    )
    observations.extend(
        _complete_instance(
            "in-distribution",
            "judge_id",
            "agent-incomplete",
            EVALUATION_SEEDS,
            budget_sec=10,
            gap_scale=2,
        )
    )
    failed_agent_index = next(
        index
        for index, observation in enumerate(observations)
        if observation.instance_id == "agent-incomplete"
        and observation.code_state is CodeState.AGENT
        and observation.solver_seed == EVALUATION_SEEDS[-1]
    )
    observations[failed_agent_index] = _observation(
        CodeState.AGENT,
        "in-distribution",
        "judge_id",
        "agent-incomplete",
        EVALUATION_SEEDS[-1],
        None,
        valid=False,
        status=RunStatus.TIMED_OUT,
    )

    primary = _report(observations, budgets_sec=(10.0,)).by_instance_set[
        "in-distribution"
    ].primary

    assert primary.instance_count == 2
    assert primary.base_complete_instance_count == 2
    assert primary.agent_complete_instance_count == 1
    assert primary.paired_complete_instance_count == 1
    assert primary.base.mean_seed_iqr == pytest.approx(0.145)
    assert primary.agent.mean_seed_iqr == pytest.approx(0.0725)
    assert primary.change.mean_seed_iqr_change == pytest.approx(-0.0725)
    assert primary.base.mean_seed_iqr_ci99 is None
    assert primary.agent.mean_seed_iqr_ci99 is None
    assert primary.change.mean_seed_iqr_change_ci99 is None


def test_no_paired_complete_instance_makes_all_aggregates_unavailable() -> None:
    observations = [
        observation
        for observation in _complete_instance(
            "in-distribution",
            "judge_id",
            "base-only",
            EVALUATION_SEEDS,
            budget_sec=10,
            gap_scale=1,
        )
        if observation.code_state is CodeState.BASE
    ]

    primary = _report(observations, budgets_sec=(10.0,)).by_instance_set[
        "in-distribution"
    ].primary

    assert primary.base_complete_instance_count == 1
    assert primary.agent_complete_instance_count == 0
    assert primary.paired_complete_instance_count == 0
    assert primary.base.mean_seed_iqr is None
    assert primary.agent.mean_seed_iqr is None
    assert primary.change.mean_seed_iqr_change is None


def test_crossed_bootstrap_is_independent_of_observation_order() -> None:
    observations: list[RunObservation] = []
    for instance_number, gap_scale in ((2, 2.0), (1, 1.0)):
        observations.extend(
            _complete_instance(
                "in-distribution",
                "judge_id",
                f"id-{instance_number}",
                reversed(EVALUATION_SEEDS),
                budget_sec=10,
                gap_scale=gap_scale,
            )
        )

    forward_report = _report(observations, budgets_sec=(10.0,))
    reverse_report = _report(list(reversed(observations)), budgets_sec=(10.0,))

    assert forward_report == reverse_report


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ("wrong_task", "match task_id"),
        ("undeclared_budget", "not declared"),
        ("missing_kind", "known instance_set_kind"),
        ("unexpected_seed", "not assigned"),
        ("duplicate_run", "duplicate observation"),
    ),
)
def test_report_rejects_observations_outside_the_declared_grid(
    change: str, message: str
) -> None:
    observations = _complete_instance(
        "in-distribution",
        "judge_id",
        "id-1",
        EVALUATION_SEEDS,
        budget_sec=10,
        gap_scale=1,
    )
    if change == "wrong_task":
        observations[0] = observations[0].model_copy(update={"task_id": "other"})
    elif change == "undeclared_budget":
        observations[0] = observations[0].model_copy(update={"budget_sec": 5.0})
    elif change == "missing_kind":
        observations[0] = observations[0].model_copy(
            update={"instance_set_kind": None}
        )
    elif change == "unexpected_seed":
        observations[0] = observations[0].model_copy(update={"solver_seed": 999})
    elif change == "duplicate_run":
        observations.append(observations[0])

    with pytest.raises(ValueError, match=message):
        _report(observations, budgets_sec=(10.0,))


def test_human_report_only_renders_the_primary_budget() -> None:
    observations: list[RunObservation] = []
    for budget_sec in (1.0, 10.0):
        observations.extend(
            _complete_instance(
                "in-distribution",
                "judge_id",
                "id-1",
                EVALUATION_SEEDS,
                budget_sec=budget_sec,
                gap_scale=budget_sec,
            )
        )

    rendered = format_seed_robustness_report(_report(observations))

    assert "Primary budget: 10s" in rendered
    assert "145.000%" in rendered
    assert "14.500%" not in rendered


def test_public_report_does_not_expose_seed_lists() -> None:
    observations = _complete_instance(
        "in-distribution",
        "judge_id",
        "id-1",
        EVALUATION_SEEDS,
        budget_sec=10,
        gap_scale=1,
    )

    payload = _report(observations, budgets_sec=(10.0,)).model_dump()

    assert "development_seeds" not in str(payload)
    assert "evaluation_seeds" not in str(payload)
    assert payload["seed_selection"] == {
        "seed_min": 0,
        "seed_max": 2**32 - 1,
        "seed_count": 30,
    }


def test_private_details_retain_seed_results_iqr_and_ecdf() -> None:
    observations = _complete_instance(
        "in-distribution",
        "judge_id",
        "id-1",
        EVALUATION_SEEDS,
        budget_sec=10,
        gap_scale=1,
    )

    details = _details(observations, budgets_sec=(10.0,))
    payload = details.model_dump(mode="json")
    base = payload["by_instance_set"]["in-distribution"]["by_budget"]["10"][
        "instances"
    ][0]["base"]

    assert payload["development_seeds"] == list(DEVELOPMENT_SEEDS)
    assert payload["evaluation_seeds"] == list(EVALUATION_SEEDS)
    assert base["complete"] is True
    assert base["seed_iqr"] == pytest.approx(0.145)
    assert [result["seed"] for result in base["seed_results"]] == list(
        EVALUATION_SEEDS
    )
    assert base["sorted_valid_gaps"] == sorted(base["sorted_valid_gaps"])
    assert base["ecdf"][-1]["fraction_at_or_below"] == 1


def test_private_details_retain_failed_seed_without_reporting_an_iqr() -> None:
    observations = _complete_instance(
        "in-distribution",
        "judge_id",
        "id-1",
        EVALUATION_SEEDS,
        budget_sec=10,
        gap_scale=1,
    )
    failed_index = next(
        index
        for index, observation in enumerate(observations)
        if observation.code_state is CodeState.AGENT
        and observation.solver_seed == EVALUATION_SEEDS[-1]
    )
    observations[failed_index] = observations[failed_index].model_copy(
        update={
            "status": RunStatus.TIMED_OUT,
            "valid": False,
            "normalized_gap": None,
        }
    )

    details = _details(observations, budgets_sec=(10.0,))
    agent = details.by_instance_set["in-distribution"].by_budget["10"].instances[
        0
    ].agent

    assert agent.complete is False
    assert agent.seed_iqr is None
    assert agent.seed_results[-1].seed == EVALUATION_SEEDS[-1]
    assert agent.seed_results[-1].status is RunStatus.TIMED_OUT
    assert agent.seed_results[-1].normalized_gap is None
