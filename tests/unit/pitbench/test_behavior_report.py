import pytest

from pitbench.metrics.behavior_report import (
    compute_behavior_metric_report,
    format_behavior_metric_report,
)
from pitbench.schema.observation import CodeState, RunObservation, RunStatus


def _row(state, instance, **changes):
    fields = dict(
        task_id="example",
        code_state=state,
        instance_set="judge_id",
        instance_set_kind="judge_id",
        instance_id=instance,
        solver_seed=0,
        budget_sec=10,
        status=RunStatus.COMPLETED,
        valid=True,
        normalized_gap=0.1 if state == CodeState.BASE else 0.2,
        wall_time_sec=1 if state == CodeState.BASE else 2,
    )
    fields.update(changes)
    return RunObservation(**fields)


def test_behavior_report_uses_current_observations_and_excludes_auxiliary_runs():
    rows = [_row(state, instance) for state in CodeState for instance in ("a", "b")]
    rows += [
        _row(CodeState.BASE, "a", test_suite="boundary"),
        _row(CodeState.BASE, "a", equivalence_parent_id="original"),
    ]
    report = compute_behavior_metric_report(rows)
    (item,) = report.slices
    assert item.performance.distance == pytest.approx(0.1)
    assert item.reliability.distance == 0
    assert item.resource.distance == pytest.approx(0.1)
    assert item.performance.instances_evaluated == 2
    assert "does not establish an improvement" in report.interpretation
    assert "descriptive" in format_behavior_metric_report(report)


def test_undefined_performance_retains_failed_instance_in_support():
    rows = [_row(state, instance) for state in CodeState for instance in ("a", "b")]
    rows[-1] = _row(CodeState.AGENT, "b", status=RunStatus.CRASHED, valid=False)
    (item,) = compute_behavior_metric_report(rows).slices
    assert item.performance.distance is None
    assert item.performance.reason
    assert item.reliability.instances_evaluated == 2
    assert item.reliability.distance == pytest.approx(2**-0.5)


def test_behavior_report_keeps_budgets_and_thread_counts_separate():
    rows = [
        _row(state, "a", budget_sec=budget, threads=threads)
        for state in CodeState
        for budget, threads in [(10, 1), (20, 1), (20, 2)]
    ]
    report = compute_behavior_metric_report(rows)
    assert {(item.budget_sec, item.threads) for item in report.slices} == {
        (10, 1),
        (20, 1),
        (20, 2),
    }
