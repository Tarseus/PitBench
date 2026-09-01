from __future__ import annotations

import pytest

from pitbench.evaluator.validity import evaluator_validity
from pitbench.schema.evaluation import ValidityCode
from pitbench.schema.observation import CodeState, RunObservation, RunStatus


def _observation(state: CodeState, status: RunStatus) -> RunObservation:
    return RunObservation(
        task_id="validity",
        code_state=state,
        population="judge_id",
        population_kind="judge_id",
        instance_id="instance",
        instance_seed=0,
        solver_seed=0,
        budget_sec=1.0,
        status=status,
        valid=status == RunStatus.COMPLETED,
    )


def test_confirmed_invalid_agent_solution_fails_qualification() -> None:
    result = evaluator_validity(
        patch_exists=True,
        fixture_mode=False,
        observations=[_observation(CodeState.AGENT, RunStatus.INVALID)],
    )

    assert result.accepted is False
    assert [(check.code, check.passed) for check in result.checks] == [
        (ValidityCode.PATCH_APPLY, True),
        (ValidityCode.SOLUTION, False),
    ]


@pytest.mark.parametrize("status", [RunStatus.TIMED_OUT, RunStatus.CRASHED])
def test_operational_agent_failure_does_not_fail_qualification(
    status: RunStatus,
) -> None:
    result = evaluator_validity(
        patch_exists=True,
        fixture_mode=False,
        observations=[_observation(CodeState.AGENT, status)],
    )

    assert result.accepted is True
    assert all(check.code != ValidityCode.SOLUTION for check in result.checks)


def test_invalid_base_solution_does_not_disqualify_candidate() -> None:
    result = evaluator_validity(
        patch_exists=True,
        fixture_mode=False,
        observations=[_observation(CodeState.BASE, RunStatus.INVALID)],
    )

    assert result.accepted is True
    assert all(check.code != ValidityCode.SOLUTION for check in result.checks)
