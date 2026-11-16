from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from adapters.pitbench.exact_solver_runner import solve


def test_gecode_runner_translates_bin_packing_and_fixed_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instance = tmp_path / "instance.json"
    instance.write_text(json.dumps({"capacity": 5, "weights": [4, 2, 2, 2, 2, 2]}))
    request = {
        "instance_path": str(instance),
        "solver_seed": 0,
        "budget_sec": 2.5,
        "threads": 1,
        "parameters": {"gecode.bin_packing_propagation_level": "full"},
    }
    captured = {}

    def run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps(
                {
                    "valid": True,
                    "has_solution": True,
                    "objective": 4,
                    "solver_status": "Optimal",
                    "solver_termination": "optimal",
                    "solution": {"bins": [[0], [1, 2], [3, 4], [5]]},
                }
            ),
            stderr="",
        )

    monkeypatch.setenv("PITBENCH_EXACT_SOLVER", "/opt/gecode/pitbench-exact")
    monkeypatch.setattr("adapters.pitbench.exact_solver_runner.subprocess.run", run)
    result = solve(request, engine="gecode")

    assert captured["argv"] == [
        "/opt/gecode/pitbench-exact",
        "2500",
        "full",
        "5",
        "6",
        "4",
        "2",
        "2",
        "2",
        "2",
        "2",
    ]
    assert captured["kwargs"]["timeout"] == 12.5
    assert result["objective"] == 4
    assert result["solution"]["bins"] == [[0], [1, 2], [3, 4], [5]]


def test_gecode_runner_rejects_unconfigured_parallel_search(tmp_path: Path) -> None:
    instance = tmp_path / "instance.json"
    instance.write_text(json.dumps({"capacity": 5, "weights": [4, 2, 2]}))
    request = {
        "instance_path": str(instance),
        "solver_seed": 0,
        "budget_sec": 2.5,
        "threads": 2,
        "parameters": {},
    }
    with pytest.raises(ValueError, match="require one thread"):
        solve(request, engine="gecode")
