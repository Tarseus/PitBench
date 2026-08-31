from __future__ import annotations

from pitbench.schema.evaluation import ValidityCheck, ValidityCode, ValidityResult


def evaluator_validity(
    *,
    patch_exists: bool,
    fixture_mode: bool,
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
    return ValidityResult(checks=checks)
