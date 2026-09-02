from __future__ import annotations

import hashlib
from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator

from pitbench.schema.task import SeedRobustnessConfig


class PrivateSeedRobustnessConfig(BaseModel):
    task_id: str
    evaluation_seeds: list[int]

    @field_validator("evaluation_seeds")
    @classmethod
    def validate_unique_evaluation_seeds(cls, values: list[int]) -> list[int]:
        if len(set(values)) != len(values):
            raise ValueError("evaluation_seeds must be unique")
        return values

    @classmethod
    def from_yaml(cls, path: Path) -> "PrivateSeedRobustnessConfig":
        return cls.model_validate(yaml.safe_load(path.read_text()))


def validate_private_seed_robustness_config(
    private_config: PrivateSeedRobustnessConfig,
    *,
    task_id: str,
    public_config: SeedRobustnessConfig,
) -> None:
    if private_config.task_id != task_id:
        raise ValueError("private seed task_id does not match the task")
    if len(private_config.evaluation_seeds) != public_config.seed_selection.seed_count:
        raise ValueError("evaluation_seeds must contain seed_count values")
    if any(
        seed < public_config.seed_selection.seed_min
        or seed > public_config.seed_selection.seed_max
        for seed in private_config.evaluation_seeds
    ):
        raise ValueError("evaluation_seeds must belong to the seed range")
    if set(private_config.evaluation_seeds) & set(public_config.development_seeds):
        raise ValueError("development_seeds and evaluation_seeds must be disjoint")


class PrivateAssetError(RuntimeError):
    pass


class PrivateAssetResolver:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def resolve(self, uri: str, sha256: str | None = None) -> Path:
        if not uri.startswith("private://"):
            raise PrivateAssetError(f"not a private URI: {uri}")
        relative = Path(uri.removeprefix("private://"))
        path = (self.root / relative).resolve()
        if self.root not in path.parents and path != self.root:
            raise PrivateAssetError(f"private URI escapes root: {uri}")
        if not path.is_file():
            raise PrivateAssetError(f"missing private asset: {uri}")
        if sha256 is not None:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != sha256:
                raise PrivateAssetError(
                    f"private asset hash mismatch for {uri}: {actual} != {sha256}"
                )
        return path


def load_private_seed_robustness_config(
    resolver: PrivateAssetResolver,
    *,
    task_id: str,
    public_config: SeedRobustnessConfig,
) -> PrivateSeedRobustnessConfig:
    private_seed_file = resolver.resolve(
        public_config.evaluation_seeds_file,
        public_config.evaluation_seeds_file_sha256,
    )
    private_config = PrivateSeedRobustnessConfig.from_yaml(private_seed_file)
    validate_private_seed_robustness_config(
        private_config,
        task_id=task_id,
        public_config=public_config,
    )
    return private_config
