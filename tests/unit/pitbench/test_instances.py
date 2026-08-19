import json
from pathlib import Path

from pitbench.families.cvrp import CVRPFamily
from pitbench.instances import materialize_population
from pitbench.tasks import TaskCatalog

ROOT = Path(__file__).resolve().parents[3]


def test_all_visible_populations_materialize(tmp_path: Path) -> None:
    for record in TaskCatalog(ROOT).validate_all():
        population = record.task.populations[0]
        paths = materialize_population(
            ROOT / population.manifest, tmp_path / record.task.task_id
        )
        assert len(paths) == population.size
        assert all(path.is_file() and path.stat().st_size > 0 for path in paths)


def test_cvrp_verifier_is_independent_and_rejects_duplicates(tmp_path: Path) -> None:
    instance = tmp_path / "instance.json"
    instance.write_text(
        json.dumps(
            {
                "depot": 0,
                "coordinates": [[0, 0], [3, 0], [0, 4]],
                "demands": [0, 1, 1],
                "capacity": 2,
            }
        )
    )
    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps({"routes": [[1, 2]]}))
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"routes": [[1, 1]]}))
    verifier = CVRPFamily()
    assert verifier.verify(instance, valid).feasible is True
    assert verifier.verify(instance, invalid).feasible is False
