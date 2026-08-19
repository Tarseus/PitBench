from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, Field


class PatchPolicyResult(BaseModel):
    accepted: bool
    changed_paths: list[str] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)


class PatchPolicy:
    """Reject candidate changes to evaluator-owned or hidden-judge surfaces."""

    DEFAULT_PROTECTED = (
        ".git",
        ".github",
        "private",
        "manifests",
        "pitbench/evaluator",
        "pitbench/families",
        "tests",
    )
    _DIFF_PATH = re.compile(r"^\+\+\+ b/(.+)$")

    def __init__(self, protected: tuple[str, ...] | None = None) -> None:
        self.protected = protected or self.DEFAULT_PROTECTED

    def inspect(self, patch_path: Path) -> PatchPolicyResult:
        changed: set[str] = set()
        violations: list[str] = []
        for line in patch_path.read_text(errors="replace").splitlines():
            match = self._DIFF_PATH.match(line)
            if not match:
                continue
            path = PurePosixPath(match.group(1))
            if ".." in path.parts or path.is_absolute():
                violations.append(f"unsafe path: {path}")
                continue
            normalized = str(path)
            changed.add(normalized)
            if any(
                normalized == prefix or normalized.startswith(f"{prefix}/")
                for prefix in self.protected
            ):
                violations.append(f"protected path: {normalized}")
        return PatchPolicyResult(
            accepted=not violations,
            changed_paths=sorted(changed),
            violations=violations,
        )
