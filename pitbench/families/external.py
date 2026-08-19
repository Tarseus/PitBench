from __future__ import annotations

import json
import subprocess
from pathlib import Path

from pitbench.families.base import ProblemFamilyPlugin, VerificationResult


class ExternalVerifierFamily(ProblemFamilyPlugin):
    """Runs an evaluator-owned verifier executable with a fixed JSON contract."""

    name = "external"

    def __init__(self, verifier: Path | None = None) -> None:
        self.verifier = verifier

    def verify(self, instance_path: Path, solution_path: Path) -> VerificationResult:
        if self.verifier is None or not self.verifier.is_file():
            return VerificationResult(
                feasible=False,
                detail="private independent verifier is unavailable",
            )
        completed = subprocess.run(
            [str(self.verifier), str(instance_path), str(solution_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            return VerificationResult(
                feasible=False,
                detail=completed.stderr.strip() or "verifier failed",
            )
        return VerificationResult.model_validate(json.loads(completed.stdout))


class MIPFamily(ExternalVerifierFamily):
    name = "mip"


class CPFamily(ExternalVerifierFamily):
    name = "cp"
