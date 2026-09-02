from __future__ import annotations

import secrets

from pydantic import BaseModel


class SeedLists(BaseModel):
    development_seeds: list[int]
    evaluation_seeds: list[int]


def generate_seed_lists(
    *,
    seed_min: int,
    seed_max: int,
    seed_count: int,
) -> SeedLists:
    if seed_count <= 0:
        raise ValueError("seed_count must be positive")
    if seed_min > seed_max:
        raise ValueError("seed_min must not exceed seed_max")

    available_seed_count = seed_max - seed_min + 1
    total_seed_count = 2 * seed_count
    if available_seed_count < total_seed_count:
        raise ValueError(
            "seed range must fit disjoint development and evaluation seeds"
        )

    selected_seeds = secrets.SystemRandom().sample(
        range(seed_min, seed_max + 1),
        total_seed_count,
    )
    return SeedLists(
        development_seeds=selected_seeds[:seed_count],
        evaluation_seeds=selected_seeds[seed_count:],
    )


__all__ = ["SeedLists", "generate_seed_lists"]
