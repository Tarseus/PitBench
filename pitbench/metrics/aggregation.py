from __future__ import annotations

from collections import defaultdict

from pitbench.schema.observation import RunObservation


def group_by_instance_seed(
    observations: list[RunObservation],
) -> dict[tuple[str, str, int], list[RunObservation]]:
    grouped: dict[tuple[str, str, int], list[RunObservation]] = defaultdict(list)
    for observation in observations:
        key = (
            observation.population,
            observation.instance_id,
            observation.solver_seed,
        )
        grouped[key].append(observation)
    return dict(grouped)
