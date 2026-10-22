import hashlib
import io
import json

import pytest
import yaml

from scripts.analyze_release_controls import _compare_panel


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _panel(tmp_path):
    catalog = {}
    oracle = tmp_path / "oracle.yaml"
    seeds = tmp_path / "seeds.yaml"
    seeds.write_text("evaluation_seeds: [0]\n")
    for name in ("a", "b"):
        instance = tmp_path / f"{name}.json"
        instance.write_text(
            json.dumps(
                {
                    "coordinates": [[0, 0], [1, 0], [2, 0]],
                    "demands": [0, 1, 1],
                    "capacity": 2,
                    "depot": 0,
                    "distance_metric": "EUC_2D",
                }
            )
        )
        catalog[name] = {"path": str(instance), "sha256": _sha(instance), "bks": 4}
    oracle.write_text(
        yaml.safe_dump({"instances": [{"id": name, "bks": 4} for name in catalog]})
    )
    manifest = {
        "rounds": 2,
        "budget_sec": 10,
        "seed_count": 1,
        "oracle_path": str(oracle),
        "seed_file": str(seeds),
        "old": {"commit": "base", "image": "base-image"},
        "new": {"commit": "candidate", "image": "candidate-image"},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    (tmp_path / "identity.json").write_text(
        json.dumps(
            {
                "manifest_sha256": _sha(manifest_path),
                "oracle_sha256": _sha(oracle),
                "seed_file_sha256": _sha(seeds),
                "selected_seeds": [0],
                "instance_count": 2,
                "expected_pairs": 4,
            }
        )
    )
    rows = []
    for round_id in range(2):
        for name in catalog:
            row = {
                "round": round_id,
                "instance_id": name,
                "solver_seed": 0,
                "budget_sec": 10,
                "target": 4,
                "order": ["old", "new"] if round_id == 0 else ["new", "old"],
            }
            for label in ("old", "new"):
                prefix = tmp_path / f"{round_id}-{name}-{label}"
                output = prefix.with_suffix(".json")
                solution = prefix.with_suffix(".solution.json")
                output.write_text(json.dumps({"valid": True, "objective": 4}))
                solution.write_text(json.dumps({"routes": [[1, 2]]}))
                row[label] = {
                    **manifest[label],
                    "output_path": str(output),
                    "solution_path": str(solution),
                    "status": "completed",
                    "reported_valid": True,
                    "reported_objective": 4,
                    "verified_objective": 4,
                    "normalized_gap": 0,
                }
            rows.append(row)
    (tmp_path / "pairs.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
    spec = {
        "manifest": str(manifest_path),
        "run_directory": str(tmp_path),
        "panel_id": "panel",
        "candidate_id": "candidate",
        "role": "test",
        "problem_family": "cvrp",
    }
    return spec, catalog, rows


@pytest.mark.parametrize("damage", ["missing", "duplicate"])
def test_comparison_does_not_score_incomplete_or_duplicate_grids(tmp_path, damage):
    spec, catalog, rows = _panel(tmp_path)
    rows = rows[:-1] if damage == "missing" else [*rows, rows[0]]
    (tmp_path / "pairs.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
    result = _compare_panel(spec, catalog, io.StringIO())
    assert not result["complete"]
    assert result["confirmed_quality_classification"] == "incomplete"
    assert result["rounds"] == {}
    assert result["missing_pairs"] if damage == "missing" else result["duplicate_pairs"]


def test_comparison_checks_solutions_instead_of_trusting_valid_flags(tmp_path):
    spec, catalog, rows = _panel(tmp_path)
    from pathlib import Path

    Path(rows[0]["new"]["solution_path"]).write_text('{"routes": [[1]]}')
    stream = io.StringIO()
    result = _compare_panel(spec, catalog, stream)
    assert not result["complete"]
    assert len(result["failed_runs"]) == 1
    assert result["independently_verified_solutions"] == 7
    assert len(stream.getvalue().splitlines()) == 8


def test_comparison_preserves_historical_reference_denominators(tmp_path):
    spec, catalog, _ = _panel(tmp_path)
    catalog["a"]["bks"] = 5
    stream = io.StringIO()
    result = _compare_panel(spec, catalog, stream)
    assert result["complete"]
    assert result["confirmed_quality_classification"] == "inconclusive"
    drift = [
        json.loads(line)
        for line in stream.getvalue().splitlines()
        if json.loads(line).get("reference_version_differs")
    ]
    assert len(drift) == 4
    assert all(row["retained_reference_target"] == 4 for row in drift)
