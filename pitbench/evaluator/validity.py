from __future__ import annotations

from pitbench.schema.evaluation import ValidityCheck, ValidityCode, ValidityResult


def evaluator_validity(
    *, patch_exists: bool, policy_passed: bool, fixture_mode: bool
) -> ValidityResult:
    checks = [
        ValidityCheck(
            code=ValidityCode.PATCH_APPLY,
            passed=patch_exists,
            detail="candidate patch is available" if patch_exists else "patch missing",
        ),
        ValidityCheck(
            code=ValidityCode.PATCH_POLICY,
            passed=policy_passed,
            detail="candidate patch respects protected paths",
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
    return ValidityResult(checks=checks)
