from __future__ import annotations

from pitbench.schema.evaluation import ValidityCheck, ValidityCode, ValidityResult
from pitbench.schema.observation import CodeState, RunObservation, RunStatus


def evaluator_validity(
    *,
    patch_exists: bool,
    fixture_mode: bool,
    observations: list[RunObservation] | None = None,
) -> ValidityResult:
    checks = [
        ValidityCheck(
            code=ValidityCode.PATCH_APPLY,
            passed=patch_exists,
            detail="candidate patch is available" if patch_exists else "patch missing",
        ),
    ]
    if fixture_mode:
        checks.append(
            ValidityCheck(
                code=ValidityCode.EVALUATOR,
                passed=True,
                detail="explicit deterministic fixture mode",
            )
        )
    invalid_agent_runs = sum(
        item.code_state == CodeState.AGENT and item.status == RunStatus.INVALID
        for item in observations or []
    )
    if invalid_agent_runs:
        checks.append(
            ValidityCheck(
                code=ValidityCode.SOLUTION,
                passed=False,
                detail=(
                    f"candidate produced {invalid_agent_runs} independently "
                    "verified invalid solution(s)"
                ),
            )
        )
    return ValidityResult(checks=checks)
