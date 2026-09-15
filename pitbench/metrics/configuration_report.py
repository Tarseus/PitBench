"""Paired configuration degradation on a complete, explicitly fixed run panel."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

from pitbench.problem_families.base import ProblemFamilyPlugin
from pitbench.repositories.base import SolverTermination


def _number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def run_feedback(
    record: dict | None,
    *,
    feedback: str,
    budget: float,
    bks: float | None,
    objective_sense: str | None,
) -> tuple[float | None, str | None]:
    if record is None:
        return None, "missing"
    if record.get("execution_status") != "returned":
        return None, record.get("execution_status", "missing_status")
    termination = record.get("termination")
    if termination == SolverTermination.ERROR:
        return None, "solver_error"
    if termination not in set(SolverTermination):
        return None, "missing_or_unknown_termination"
    verification = record.get("verification") or {}
    feasible = verification.get("feasible") is True
    if record.get("solution_value_valid") and not feasible:
        return None, "invalid_solution"
    if feedback == "normalized_gap":
        if not feasible:
            return None, "no_feasible_solution"
        objective = verification.get("objective")
        if not _number(objective) or not _number(bks):
            return None, "unavailable_objective_or_reference"
        value = ProblemFamilyPlugin.normalized_gap(
            objective,
            bks,
            objective_sense=objective_sense,
        )
        return (value, None) if _number(value) else (None, "nonfinite_gap")
    if feedback == "capped_optimal_time":
        if termination == SolverTermination.TIME_LIMIT:
            return 1.0, None
        if termination != SolverTermination.OPTIMAL:
            return None, "unexpected_termination"
        if not feasible:
            return None, "unverified_optimal_solution"
        runtime = record.get("solver_runtime_sec")
        if not _number(runtime) or runtime < 0:
            return None, "unavailable_solver_time"
        return min(runtime, budget) / budget, None
    raise ValueError(f"unknown configuration feedback: {feedback}")


def paired_panel(
    default: list[dict],
    candidate: list[dict],
    *,
    instances: list[dict],
    seeds: list[int],
    budget: float,
    feedback: str,
    objective_sense: str | None,
) -> dict:
    """Never replace absent runs or average only the successful pairs."""
    expected = {(instance["id"], seed) for instance in instances for seed in seeds}
    if not instances or not seeds or not _number(budget) or budget <= 0:
        raise ValueError(
            "a panel requires instances, seeds and a positive finite budget"
        )
    if len(expected) != len(instances) * len(seeds):
        raise ValueError("duplicate instance or solver seed in panel")

    def index(records):
        indexed = {}
        for record in records:
            key = record["instance_id"], record["solver_seed"]
            if key not in expected or record["budget_sec"] != budget:
                raise ValueError("run is outside the declared panel")
            if key in indexed:
                raise ValueError("duplicate run in configuration panel")
            indexed[key] = record
        return indexed

    defaults, candidates = index(default), index(candidate)
    pairs, by_instance = [], []
    unavailable = Counter()
    for instance in instances:
        differences = []
        for seed in seeds:
            key = instance["id"], seed
            kwargs = {
                "feedback": feedback,
                "budget": budget,
                "bks": instance.get("bks"),
                "objective_sense": objective_sense,
            }
            baseline, baseline_error = run_feedback(defaults.get(key), **kwargs)
            value, error = run_feedback(candidates.get(key), **kwargs)
            difference = (
                value - baseline if value is not None and baseline is not None else None
            )
            pairs.append(
                {
                    "instance_id": instance["id"],
                    "solver_seed": seed,
                    "default_feedback": baseline,
                    "candidate_feedback": value,
                    "degradation": difference,
                    "default_unavailable": baseline_error,
                    "candidate_unavailable": error,
                }
            )
            if difference is not None:
                differences.append(difference)
            unavailable.update(reason for reason in (baseline_error, error) if reason)
        by_instance.append(
            {
                "instance_id": instance["id"],
                "expected_pairs": len(seeds),
                "available_pairs": len(differences),
                "mean_degradation": math.fsum(differences) / len(seeds)
                if len(differences) == len(seeds)
                else None,
            }
        )
    complete = all(row["mean_degradation"] is not None for row in by_instance)
    mean_degradation = (
        math.fsum(row["mean_degradation"] for row in by_instance) / len(instances)
        if complete
        else None
    )
    return {
        "complete": complete,
        "feedback": feedback,
        "budget_sec": budget,
        "expected_pairs": len(expected),
        "available_pairs": sum(row["available_pairs"] for row in by_instance),
        "mean_degradation": mean_degradation,
        "reported_degradation": (
            mean_degradation * (100 if feedback == "normalized_gap" else 1)
            if complete
            else None
        ),
        "reported_unit": "gap_percentage_points"
        if feedback == "normalized_gap"
        else "fraction_of_budget",
        "unavailable": dict(unavailable),
        "by_instance": by_instance,
        "pairs": pairs,
    }


def write_configuration_report(output: Path) -> dict:
    """Report search and independent retest separately, including stopped searches."""
    import json

    from pitbench.evaluator.collection import write_json

    manifest = json.loads((output / "experiment.json").read_text())
    summaries = []
    for search in manifest["searches"]:
        path = output / search["id"] / "state.json"
        state = (
            json.loads(path.read_text())
            if path.exists()
            else {"status": "not_started", "candidates": []}
        )
        summaries.append(
            {
                "id": search["id"],
                "task_id": search["task_id"],
                "budget_sec": search["budget_sec"],
                **state,
            }
        )
    report = {
        "interpretation": "degradation found under the declared search budget; independent retest is reported separately",
        "searches": summaries,
    }
    write_json(output / "configuration_report.json", report)
    return report
