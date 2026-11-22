from __future__ import annotations

from pitbench.schema.evaluation import ValidityCheck, ValidityCode, ValidityResult
from pitbench.schema.observation import CodeState, RunObservation, RunStatus


def evaluator_validity(
    *,
    patch_exists: bool,
    fixture_mode: bool,
    observations: list[RunObservation] | None = None,
    exact_qualification_failures: int = 0,
    scenario_verification_failures: int = 0,
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
    invalid_agent_runs = 0
    for scope in ("primary", "extension"):
        invalid_count = sum(
            item.code_state == CodeState.AGENT
            and item.status == RunStatus.INVALID
            and (item.test_suite is not None or item.equivalence_parent_id is not None)
            == (scope == "extension")
            for item in observations or []
        )
        if not invalid_count:
            continue
        invalid_agent_runs += invalid_count
        checks.append(
            ValidityCheck(
                code=ValidityCode.SOLUTION,
                passed=False,
                scope=scope,
                detail=(
                    f"candidate produced {invalid_count} independently "
                    "verified invalid solution(s)"
                ),
            )
        )
    if scenario_verification_failures:
        checks.append(
            ValidityCheck(
                code=ValidityCode.SOLUTION,
                passed=False,
                scope="extension",
                detail=f"candidate produced {scenario_verification_failures} invalid solution(s) in independently checked metric scenarios",
            )
        )
    additional_exact_failures = max(
        0,
        exact_qualification_failures - invalid_agent_runs,
    )
    if additional_exact_failures:
        checks.append(
            ValidityCheck(
                code=ValidityCode.OBJECTIVE,
                passed=False,
                detail=(
                    f"candidate produced {additional_exact_failures} false exact "
                    "optimality claim(s)"
                ),
            )
        )
    return ValidityResult(checks=checks)
