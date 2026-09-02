from __future__ import annotations

import pytest

from pitbench.evaluator import seed_selection
from pitbench.evaluator.seed_selection import generate_seed_lists


class _RecordedSecureRandom:
    def __init__(self) -> None:
        self.population: range | None = None
        self.sample_size: int | None = None

    def sample(self, population: range, sample_size: int) -> list[int]:
        self.population = population
        self.sample_size = sample_size
        return list(reversed(population))[:sample_size]


def test_seed_lists_use_one_secure_sample_and_split_it_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secure_random = _RecordedSecureRandom()
    monkeypatch.setattr(
        seed_selection.secrets,
        "SystemRandom",
        lambda: secure_random,
    )

    seed_lists = generate_seed_lists(seed_min=10, seed_max=19, seed_count=3)

    assert secure_random.population == range(10, 20)
    assert secure_random.sample_size == 6
    assert seed_lists.development_seeds == [19, 18, 17]
    assert seed_lists.evaluation_seeds == [16, 15, 14]


def test_generated_seed_lists_are_disjoint_and_inside_uint32_range() -> None:
    seed_lists = generate_seed_lists(
        seed_min=0,
        seed_max=2**32 - 1,
        seed_count=30,
    )

    assert len(seed_lists.development_seeds) == 30
    assert len(seed_lists.evaluation_seeds) == 30
    assert not set(seed_lists.development_seeds) & set(seed_lists.evaluation_seeds)
    assert all(
        0 <= seed <= 2**32 - 1
        for seed in seed_lists.development_seeds + seed_lists.evaluation_seeds
    )


@pytest.mark.parametrize(
    ("parameters", "message"),
    (
        ({"seed_min": 0, "seed_max": 10, "seed_count": 0}, "must be positive"),
        ({"seed_min": 10, "seed_max": 0, "seed_count": 1}, "must not exceed"),
        ({"seed_min": 0, "seed_max": 4, "seed_count": 3}, "must fit disjoint"),
    ),
)
def test_seed_generation_rejects_invalid_parameters(
    parameters: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        generate_seed_lists(**parameters)
