from __future__ import annotations

from typing import Any

from pitbench.distribution import single_qoi_response
from pitbench.qoi.cvrp import extract_cvrp_instance_qoi


def test_single_qoi_pilot_keeps_generator_and_solver_pairing(monkeypatch) -> None:
    def fake_solve(
        instance: dict[str, Any], *, solver_seed: int, budget_sec: float
    ) -> dict[str, Any]:
        qoi = extract_cvrp_instance_qoi(instance, spec_version="1.0").values
        objective = 10 * qoi["pairwise_distance_median"]
        objective += 20 * qoi["demand_cv"] + solver_seed
        return {
            "feasible": True,
            "verification_detail": "test fixture",
            "solver_objective": objective,
            "verified_objective": objective,
            "wall_time_sec": budget_sec,
            "cpu_time_sec": budget_sec,
            "iterations": 100 + solver_seed,
            "trajectory": [
                {"time_sec": 0.01, "objective": objective + 10},
                {"time_sec": 0.02, "objective": objective},
            ],
        }

    monkeypatch.setattr(single_qoi_response, "_solve", fake_solve)
    monkeypatch.setattr(single_qoi_response, "version", lambda _: "test")

    report = single_qoi_response.run_pyvrp_single_qoi_pilot(
        generator_seeds=(101, 202),
        solver_seeds=(0, 1),
        budget_sec=0.1,
        reference_budget_sec=0.2,
        bootstrap_repetitions=100,
    )

    assert len(report.runs) == 2 * 2 * 2 * 2
    assert len(report.reference_runs) == 2 * 2 * 2
    assert report.solver["version"] == "test"
    assert set(report.axis_effects) == {
        "demand_cv",
        "pairwise_distance_median",
    }
    for effect in report.axis_effects.values():
        assert all(
            metric["paired_run_count"] == 4 for metric in effect["metrics"].values()
        )
