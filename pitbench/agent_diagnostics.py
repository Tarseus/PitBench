"""Solver-behavior diagnostics for paired public development runs."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

WORK_METRICS = (
    "iterations",
    "nodes",
    "simplex_iterations",
    "branches",
    "conflicts",
    "propagations",
)
PROGRESS_FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def finite_number(value: object) -> float | None:
    if type(value) not in {int, float}:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def trajectory_summary(path: Path, objective_sense: str) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False, "points": 0}

    records: list[dict[str, Any]] = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)

    best: float | None = None
    best_updates = 0
    observed_objective_points = 0
    best_dual: float | None = None
    dual_bound_updates = 0
    observed_dual_bound_points = 0
    time_to_first: float | None = None
    time_to_best: float | None = None
    time_to_best_dual: float | None = None
    for record in records:
        elapsed = finite_number(record.get("time_sec"))
        observed = record.get("event") != "final"
        dual_bound = finite_number(record.get("dual_bound"))
        if dual_bound is not None:
            if observed:
                observed_dual_bound_points += 1
            improved_dual = best_dual is None or (
                dual_bound > best_dual
                if objective_sense == "minimize"
                else dual_bound < best_dual
            )
            if improved_dual:
                best_dual = dual_bound
                dual_bound_updates += 1
                time_to_best_dual = elapsed if observed else None
        objective = finite_number(record.get("objective", record.get("primal_bound")))
        if objective is None:
            continue
        if observed:
            observed_objective_points += 1
        if observed and time_to_first is None:
            time_to_first = elapsed
        improved = best is None or (
            objective < best if objective_sense == "minimize" else objective > best
        )
        if improved:
            best = objective
            best_updates += 1
            time_to_best = elapsed if observed else None

    final = records[-1] if records else {}
    return {
        "available": True,
        "points": len(records),
        "best_objective": best,
        "best_updates": best_updates,
        "observed_objective_points": observed_objective_points,
        "observed_dual_bound_points": observed_dual_bound_points,
        "dual_bound_updates": dual_bound_updates,
        "best_dual_bound": best_dual,
        "time_to_best_dual_bound_sec": time_to_best_dual,
        "time_to_first_solution_sec": time_to_first,
        "time_to_best_sec": time_to_best,
        "final_primal_bound": finite_number(final.get("primal_bound")),
        "final_dual_bound": finite_number(final.get("dual_bound")),
        "final_solver_gap": finite_number(final.get("solver_gap")),
        "final_nodes": finite_number(final.get("nodes")),
        "final_simplex_iterations": finite_number(final.get("simplex_iterations")),
    }


def trajectory_checkpoints(
    path: Path, objective_sense: str, budget_sec: float
) -> list[dict[str, Any]]:
    """Sample observed search progress at fixed fractions of a declared budget.

    A final-only record is deliberately excluded: it cannot locate progress in
    time and should not be presented to the agent as an anytime observation.
    """
    times = [budget_sec * fraction for fraction in PROGRESS_FRACTIONS]
    events: list[tuple[float, float | None, float | None]] = []
    if path.is_file():
        for line in path.read_text(errors="replace").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or record.get("event") == "final":
                continue
            elapsed = finite_number(record.get("time_sec"))
            if elapsed is None or elapsed < 0 or elapsed > budget_sec:
                continue
            objective = finite_number(
                record.get("objective", record.get("primal_bound"))
            )
            dual = finite_number(record.get("dual_bound"))
            if objective is not None or dual is not None:
                events.append((elapsed, objective, dual))
    events.sort(key=lambda item: item[0])

    samples: list[dict[str, Any]] = []
    index = 0
    best_objective: float | None = None
    best_dual: float | None = None
    for fraction, checkpoint in zip(PROGRESS_FRACTIONS, times, strict=True):
        while index < len(events) and events[index][0] <= checkpoint:
            _, objective, dual = events[index]
            if objective is not None and (
                best_objective is None
                or (
                    objective < best_objective
                    if objective_sense == "minimize"
                    else objective > best_objective
                )
            ):
                best_objective = objective
            if dual is not None and (
                best_dual is None
                or (
                    dual > best_dual
                    if objective_sense == "minimize"
                    else dual < best_dual
                )
            ):
                best_dual = dual
            index += 1
        samples.append(
            {
                "budget_fraction": fraction,
                "time_sec": checkpoint,
                "best_objective": best_objective,
                "best_dual_bound": best_dual,
            }
        )
    return samples


def observed_gap_integral(
    path: Path, objective_sense: str, budget_sec: float
) -> dict[str, Any]:
    """Integrate solver-reported primal-dual gap only where both bounds exist."""
    if budget_sec <= 0:
        raise ValueError("gap integral requires a positive budget")
    events: list[tuple[float, float | None, float | None]] = []
    if path.is_file():
        for line in path.read_text(errors="replace").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or record.get("event") == "final":
                continue
            elapsed = finite_number(record.get("time_sec"))
            if elapsed is None or not 0 <= elapsed <= budget_sec:
                continue
            events.append(
                (
                    elapsed,
                    finite_number(record.get("objective", record.get("primal_bound"))),
                    finite_number(record.get("dual_bound")),
                )
            )
    events.sort(key=lambda item: item[0])
    best_primal: float | None = None
    best_dual: float | None = None
    active_gap: float | None = None
    previous_time = 0.0
    observed_time = 0.0
    area = 0.0
    inconsistent_events = 0
    for elapsed, primal, dual in events:
        if active_gap is not None:
            duration = elapsed - previous_time
            area += active_gap * duration
            observed_time += duration
        if primal is not None and (
            best_primal is None
            or (
                primal < best_primal
                if objective_sense == "minimize"
                else primal > best_primal
            )
        ):
            best_primal = primal
        if dual is not None and (
            best_dual is None
            or (dual > best_dual if objective_sense == "minimize" else dual < best_dual)
        ):
            best_dual = dual
        active_gap = None
        if best_primal is not None and best_dual is not None:
            difference = (
                best_primal - best_dual
                if objective_sense == "minimize"
                else best_dual - best_primal
            )
            if difference < 0:
                inconsistent_events += 1
            else:
                active_gap = difference / max(1.0, abs(best_primal), abs(best_dual))
        previous_time = elapsed
    if active_gap is not None:
        duration = budget_sec - previous_time
        area += active_gap * duration
        observed_time += duration
    return {
        "observed_gap_area_sec": area if observed_time > 0 else None,
        "observed_duration_sec": observed_time,
        "observed_fraction_of_budget": observed_time / budget_sec,
        "mean_gap_on_observed_window": area / observed_time
        if observed_time > 0
        else None,
        "inconsistent_bound_events": inconsistent_events,
        "full_budget_observed": observed_time == budget_sec,
        "basis": "solver_reported_bounds_diagnostic_only",
    }


def solver_behavior(
    config: dict[str, Any], result: dict[str, Any], output: Path, budget_sec: float
) -> dict[str, Any]:
    trajectory = output.with_suffix(".trajectory.jsonl")
    objective_sense = str(config.get("objective_sense", "minimize"))
    return {
        "objective_sense": objective_sense,
        "work": {
            metric: value
            for metric in WORK_METRICS
            if (value := finite_number(result.get(metric))) is not None
        },
        "trajectory": trajectory_summary(
            trajectory,
            objective_sense,
        ),
        "checkpoints": trajectory_checkpoints(trajectory, objective_sense, budget_sec),
        "observed_gap_integral": observed_gap_integral(
            trajectory, objective_sense, budget_sec
        ),
    }


def paired_behavior(current: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    before = base.get("solver_behavior", {})
    after = current.get("solver_behavior", {})
    before_work = before.get("work", {})
    after_work = after.get("work", {})
    work_delta = {
        metric: after_work[metric] - before_work[metric]
        for metric in WORK_METRICS
        if metric in before_work and metric in after_work
    }
    before_trace = before.get("trajectory", {})
    after_trace = after.get("trajectory", {})
    sense = before.get("objective_sense")
    checkpoints: list[dict[str, Any]] = []
    if sense in {"minimize", "maximize"} and sense == after.get("objective_sense"):
        for old, new in zip(
            before.get("checkpoints", []), after.get("checkpoints", []), strict=False
        ):
            if (old.get("budget_fraction"), old.get("time_sec")) != (
                new.get("budget_fraction"),
                new.get("time_sec"),
            ):
                continue
            base_objective = finite_number(old.get("best_objective"))
            candidate_objective = finite_number(new.get("best_objective"))
            base_dual = finite_number(old.get("best_dual_bound"))
            candidate_dual = finite_number(new.get("best_dual_bound"))
            checkpoints.append(
                {
                    "budget_fraction": old["budget_fraction"],
                    "time_sec": old["time_sec"],
                    "base_best_objective": base_objective,
                    "candidate_best_objective": candidate_objective,
                    "candidate_objective_improvement": (
                        (base_objective - candidate_objective)
                        * (1 if sense == "minimize" else -1)
                        if base_objective is not None
                        and candidate_objective is not None
                        else None
                    ),
                    "base_best_dual_bound": base_dual,
                    "candidate_best_dual_bound": candidate_dual,
                    "candidate_dual_bound_improvement": (
                        (candidate_dual - base_dual)
                        * (1 if sense == "minimize" else -1)
                        if base_dual is not None and candidate_dual is not None
                        else None
                    ),
                }
            )
    trajectory_available = bool(
        before_trace.get("available") and after_trace.get("available")
    )
    progress_available = bool(
        trajectory_available
        and (finite_number(before_trace.get("points")) or 0) > 1
        and (finite_number(after_trace.get("points")) or 0) > 1
    )
    objective_progress_available = bool(
        progress_available
        and (finite_number(before_trace.get("observed_objective_points")) or 0) > 0
        and (finite_number(after_trace.get("observed_objective_points")) or 0) > 0
    )
    bound_progress_available = bool(
        progress_available
        and (finite_number(before_trace.get("observed_dual_bound_points")) or 0) > 0
        and (finite_number(after_trace.get("observed_dual_bound_points")) or 0) > 0
    )

    def reduction(field: str, available: bool) -> float | None:
        base_value = finite_number(before_trace.get(field))
        agent_value = finite_number(after_trace.get(field))
        return (
            base_value - agent_value
            if available and base_value is not None and agent_value is not None
            else None
        )

    return {
        "matched_progress_checkpoints": checkpoints,
        "candidate_minus_base_work": work_delta,
        "time_to_first_solution_reduction_sec": reduction(
            "time_to_first_solution_sec", objective_progress_available
        ),
        "time_to_best_reduction_sec": reduction(
            "time_to_best_sec", objective_progress_available
        ),
        "time_to_best_dual_bound_reduction_sec": reduction(
            "time_to_best_dual_bound_sec", bound_progress_available
        ),
        "candidate_minus_base_best_updates": (
            after_trace["best_updates"] - before_trace["best_updates"]
            if "best_updates" in before_trace and "best_updates" in after_trace
            else None
        ),
        "trajectory_artifact_available_for_both": trajectory_available,
        "progress_trace_available_for_both": progress_available,
        "objective_progress_available_for_both": objective_progress_available,
        "bound_progress_available_for_both": bound_progress_available,
    }


def paired_feedback(current: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    behavior = (
        {"behavior_comparison": paired_behavior(current, base)}
        if "solver_behavior" in current and "solver_behavior" in base
        else {}
    )
    if current["performance_protocol"] == "heuristic_fixed_budget":
        before = base.get("preview_normalized_gap")
        after = current.get("preview_normalized_gap")
        return {
            "verified_objective_reduction": (
                base["verified_objective"] - current["verified_objective"]
                if base.get("verified_valid") is True
                and current.get("verified_valid") is True
                else None
            ),
            "preview_gap_reduction": before - after
            if before is not None and after is not None
            else None,
            **behavior,
        }
    before_solved = base.get("preview_solved")
    after_solved = current.get("preview_solved")
    before_runtime = base.get("preview_par2_sec")
    after_runtime = current.get("preview_par2_sec")
    return {
        "verified_objective_reduction": (
            base["verified_objective"] - current["verified_objective"]
            if base.get("verified_valid") is True
            and current.get("verified_valid") is True
            else None
        ),
        "preview_solved_change": int(after_solved) - int(before_solved)
        if before_solved is not None and after_solved is not None
        else None,
        "preview_par2_reduction_sec": before_runtime - after_runtime
        if before_runtime is not None and after_runtime is not None
        else None,
        **behavior,
    }


def matching_base_feedback(
    run_root: Path, current: dict[str, Any]
) -> tuple[Path, dict[str, Any]] | None:
    base_root = run_root.parent / "base"
    fields = (
        "instance_sha256",
        "seed",
        "budget_sec",
        "threads",
        "performance_protocol",
    )
    for path in sorted(base_root.glob("*.feedback.json")):
        try:
            base = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if all(base.get(field) == current.get(field) for field in fields):
            return path, base
    return None


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _effect_direction(value: float | None, *, tolerance: float = 1e-12) -> str:
    if value is None or abs(value) <= tolerance:
        return "equal_or_unresolved"
    return "better" if value > 0 else "worse"


def paired_behavior_summary(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for base, agent in pairs:
        instance = str(agent.get("instance_id") or agent.get("instance_sha256"))
        grouped.setdefault(instance, []).append((base, agent))

    wall_reductions: list[float] = []
    cpu_reductions: list[float] = []
    time_to_best: list[float] = []
    time_to_best_bound: list[float] = []
    work_deltas: dict[str, list[float]] = {metric: [] for metric in WORK_METRICS}
    comparisons: list[dict[str, Any]] = []
    for instance_pairs in grouped.values():
        instance_comparisons = [
            paired_behavior(agent, base) for base, agent in instance_pairs
        ]
        comparisons.extend(instance_comparisons)
        instance_wall = [
            before - after
            for base, agent in instance_pairs
            if (before := finite_number(base.get("reported_wall_time_sec"))) is not None
            and (after := finite_number(agent.get("reported_wall_time_sec")))
            is not None
        ]
        instance_cpu = [
            before - after
            for base, agent in instance_pairs
            if (before := finite_number(base.get("reported_cpu_time_sec"))) is not None
            and (after := finite_number(agent.get("reported_cpu_time_sec"))) is not None
        ]
        instance_time_to_best = [
            value
            for behavior in instance_comparisons
            if (value := finite_number(behavior.get("time_to_best_reduction_sec")))
            is not None
        ]
        instance_time_to_best_bound = [
            value
            for behavior in instance_comparisons
            if (
                value := finite_number(
                    behavior.get("time_to_best_dual_bound_reduction_sec")
                )
            )
            is not None
        ]
        if (value := _mean(instance_wall)) is not None:
            wall_reductions.append(value)
        if (value := _mean(instance_cpu)) is not None:
            cpu_reductions.append(value)
        if (value := _mean(instance_time_to_best)) is not None:
            time_to_best.append(value)
        if (value := _mean(instance_time_to_best_bound)) is not None:
            time_to_best_bound.append(value)
        for metric in WORK_METRICS:
            instance_work = [
                float(behavior["candidate_minus_base_work"][metric])
                for behavior in instance_comparisons
                if metric in behavior["candidate_minus_base_work"]
            ]
            if (value := _mean(instance_work)) is not None:
                work_deltas[metric].append(value)
    return {
        "mean_wall_time_reduction_sec": _mean(wall_reductions),
        "median_wall_time_reduction_sec": (
            statistics.median(wall_reductions) if wall_reductions else None
        ),
        "mean_cpu_time_reduction_sec": _mean(cpu_reductions),
        "trajectory_artifact_pairs": sum(
            behavior["trajectory_artifact_available_for_both"]
            for behavior in comparisons
        ),
        "progress_trace_pairs": sum(
            behavior["progress_trace_available_for_both"] for behavior in comparisons
        ),
        "mean_time_to_best_reduction_sec": _mean(time_to_best),
        "mean_time_to_best_dual_bound_reduction_sec": _mean(time_to_best_bound),
        "mean_candidate_minus_base_work": {
            metric: _mean(values) for metric, values in work_deltas.items() if values
        },
    }


def contrastive_progress_summary(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Summarize paired trace checkpoints with equal weight per instance."""
    grouped: dict[tuple[float, float], dict[str, dict[str, list[float]]]] = {}
    for base, agent in pairs:
        budget = finite_number(agent.get("budget_sec"))
        if budget is None:
            continue
        instance = str(agent.get("instance_id") or agent.get("instance_sha256"))
        for checkpoint in paired_behavior(agent, base)["matched_progress_checkpoints"]:
            key = (budget, checkpoint["budget_fraction"])
            values = grouped.setdefault(key, {}).setdefault(
                instance, {"objective": [], "dual_bound": []}
            )
            for name, field in (
                ("objective", "candidate_objective_improvement"),
                ("dual_bound", "candidate_dual_bound_improvement"),
            ):
                if (value := finite_number(checkpoint[field])) is not None:
                    values[name].append(value)

    summary: list[dict[str, Any]] = []
    for (budget, fraction), instances in sorted(grouped.items()):
        row: dict[str, Any] = {
            "budget_sec": budget,
            "budget_fraction": fraction,
            "time_sec": budget * fraction,
        }
        for name in ("objective", "dual_bound"):
            effects = [
                statistics.fmean(values[name])
                for values in instances.values()
                if values[name]
            ]
            row[name] = {
                "instances_with_paired_progress": len(effects),
                # Raw objectives and bounds can have unrelated scales across
                # instances; only their directions can be pooled here.
                "mean_candidate_improvement": _mean(effects)
                if len(instances) == 1
                else None,
                "better_instances": sum(value > 1e-12 for value in effects),
                "worse_instances": sum(value < -1e-12 for value in effects),
            }
        summary.append(row)
    return summary


def paired_gap_area_summary(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Keep observed-window gap areas and trace coverage separate by budget."""
    grouped: dict[float, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for base, agent in pairs:
        budget = finite_number(agent.get("budget_sec"))
        if budget is not None:
            grouped.setdefault(budget, []).append((base, agent))
    result: list[dict[str, Any]] = []
    for budget, budget_pairs in sorted(grouped.items()):
        observed: list[tuple[float, ...]] = []
        for base, agent in budget_pairs:
            before = base.get("solver_behavior", {}).get("observed_gap_integral", {})
            after = agent.get("solver_behavior", {}).get("observed_gap_integral", {})
            values = tuple(
                finite_number(record.get(field))
                for record, field in (
                    (before, "mean_gap_on_observed_window"),
                    (after, "mean_gap_on_observed_window"),
                    (before, "observed_gap_area_sec"),
                    (after, "observed_gap_area_sec"),
                    (before, "observed_fraction_of_budget"),
                    (after, "observed_fraction_of_budget"),
                )
            )
            if all(value is not None for value in values):
                observed.append(values)
        result.append(
            {
                "budget_sec": budget,
                "paired_runs_with_both_bounds": len(observed),
                "base_mean_observed_gap": _mean([row[0] for row in observed]),
                "candidate_mean_observed_gap": _mean([row[1] for row in observed]),
                "base_mean_observed_gap_area_sec": _mean([row[2] for row in observed]),
                "candidate_mean_observed_gap_area_sec": _mean(
                    [row[3] for row in observed]
                ),
                "base_mean_covered_fraction": _mean([row[4] for row in observed]),
                "candidate_mean_covered_fraction": _mean([row[5] for row in observed]),
            }
        )
    return result


def development_diagnosis(
    config: dict[str, Any], run_root: Path, budgets: list[float]
) -> dict[str, Any]:
    agent_root = run_root / "agent"
    agent_records: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(agent_root.glob("*.feedback.json")):
        try:
            record = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if record.get("budget_sec") in budgets:
            agent_records.append((path, record))

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    missing_base: list[str] = []
    for path, agent in agent_records:
        match = matching_base_feedback(agent_root, agent)
        if match is None:
            missing_base.append(path.name)
            continue
        pairs.append((match[1], agent))

    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for base, agent in pairs:
        instance = str(agent.get("instance_id") or agent.get("instance_sha256"))
        grouped.setdefault(instance, []).append((base, agent))

    protocol = config["performance_protocol"]
    per_instance: list[dict[str, Any]] = []
    for instance, instance_pairs in sorted(grouped.items()):
        behavior_summary = paired_behavior_summary(instance_pairs)
        if protocol == "heuristic_fixed_budget":
            effects = [
                value
                for base, agent in instance_pairs
                if (
                    value := finite_number(
                        paired_feedback(agent, base).get("preview_gap_reduction")
                    )
                )
                is not None
            ]
            effect = _mean(effects)
            per_instance.append(
                {
                    "instance_id": instance,
                    "paired_runs": len(instance_pairs),
                    "mean_normalized_gap_reduction": effect,
                    "direction": _effect_direction(effect),
                    "behavior": behavior_summary,
                    "contrastive_progress": contrastive_progress_summary(
                        instance_pairs
                    ),
                    "observed_gap_area": paired_gap_area_summary(instance_pairs),
                }
            )
            continue

        solved_changes = [
            value
            for base, agent in instance_pairs
            if (
                value := finite_number(
                    paired_feedback(agent, base).get("preview_solved_change")
                )
            )
            is not None
        ]
        par2_reductions = [
            value
            for base, agent in instance_pairs
            if (
                value := finite_number(
                    paired_feedback(agent, base).get("preview_par2_reduction_sec")
                )
            )
            is not None
        ]
        coverage_effect = _mean(solved_changes)
        par2_effect = _mean(par2_reductions)
        direction = _effect_direction(coverage_effect)
        if direction == "equal_or_unresolved":
            direction = _effect_direction(par2_effect)
        per_instance.append(
            {
                "instance_id": instance,
                "paired_runs": len(instance_pairs),
                "mean_verified_solved_change": coverage_effect,
                "mean_par2_reduction_sec": par2_effect,
                "direction": direction,
                "behavior": behavior_summary,
                "contrastive_progress": contrastive_progress_summary(instance_pairs),
                "observed_gap_area": paired_gap_area_summary(instance_pairs),
            }
        )

    direction_counts = {
        direction: sum(item["direction"] == direction for item in per_instance)
        for direction in ("better", "equal_or_unresolved", "worse")
    }
    invalid = sum(
        agent.get("verified_valid") is False
        and agent.get("verification") != "no_solution"
        for _, agent in pairs
    )
    runner_failures = sum("result_error" in agent for _, agent in pairs)
    behavior_summary = paired_behavior_summary(pairs)

    expected = {
        (target.get("sha256"), seed, budget, config["threads"])
        for target in config.get("development_targets", {}).values()
        for seed in config.get("development_seeds", [])
        for budget in budgets
    }
    observed = {
        (
            agent.get("instance_sha256"),
            agent.get("seed"),
            agent.get("budget_sec"),
            agent.get("threads"),
        )
        for _, agent in pairs
    }
    missing_declared_runs = len(expected - observed)

    better = direction_counts["better"]
    worse = direction_counts["worse"]
    if invalid or runner_failures:
        pattern = "correctness_or_execution_failure"
    elif not pairs:
        pattern = "missing_paired_evidence"
    elif better and worse:
        pattern = "mixed_instance_effect"
    elif better == 1 and not worse:
        pattern = "favorable_effect_concentrated_in_one_instance"
    elif better > 1 and not worse:
        pattern = "favorable_effect_across_multiple_instances"
    elif worse and not better:
        pattern = "unfavorable_effect"
    else:
        pattern = "no_detected_solver_outcome_change"

    possible_explanations: list[str] = []
    next_actions: list[str] = []
    if invalid:
        possible_explanations.append("candidate changes may violate solution semantics")
        next_actions.append("fix independently detected invalid solutions first")
    if runner_failures:
        possible_explanations.append(
            "candidate or runner failed before comparable output"
        )
        next_actions.append("resolve runner failures before interpreting performance")
    if missing_base:
        next_actions.append("run matching Base observations for unpaired Agent results")
    if pattern == "favorable_effect_concentrated_in_one_instance":
        possible_explanations.append(
            "the apparent gain may depend on one instance structure"
        )
        next_actions.append("test the candidate on additional independent instances")
    elif pattern == "mixed_instance_effect":
        possible_explanations.append(
            "the change trades gains on some structures for regressions"
        )
        next_actions.append("compare improving and regressing instance structures")
    elif pattern == "favorable_effect_across_multiple_instances":
        possible_explanations.append(
            "the candidate has a development-set solver outcome signal"
        )
        if missing_declared_runs:
            next_actions.append(
                "for this shortlisted candidate, expand to the complete declared "
                "grid with --all-seeds and --all-budgets"
            )
        else:
            next_actions.append(
                "request independent confirmation on untouched instances"
            )
    elif pattern == "no_detected_solver_outcome_change":
        if (behavior_summary["mean_wall_time_reduction_sec"] or 0) > 0:
            possible_explanations.append(
                "the candidate may reduce execution cost without changing solver outcome"
            )
        else:
            possible_explanations.append(
                "the measured intervention has no visible development gain"
            )
        next_actions.append(
            "inspect search behavior and choose a different modification"
        )
    if not behavior_summary["progress_trace_pairs"]:
        possible_explanations.append(
            "search-stage timing is unavailable for these paired runs"
        )

    mechanism_signals: list[str] = []
    mean_wall = behavior_summary["mean_wall_time_reduction_sec"]
    mean_time_to_best = behavior_summary["mean_time_to_best_reduction_sec"]
    mean_work = behavior_summary["mean_candidate_minus_base_work"]
    if mean_wall is not None:
        mechanism_signals.append(
            "lower_mean_wall_time"
            if mean_wall > 1e-12
            else "higher_mean_wall_time"
            if mean_wall < -1e-12
            else "equal_mean_wall_time"
        )
    if mean_time_to_best is not None:
        mechanism_signals.append(
            "earlier_best_observed"
            if mean_time_to_best > 1e-12
            else "later_best_observed"
            if mean_time_to_best < -1e-12
            else "equal_time_to_best_observed"
        )
    if any(value is not None and value > 0 for value in mean_work.values()):
        mechanism_signals.append("more_solver_work_observed")
    if any(value is not None and value < 0 for value in mean_work.values()):
        mechanism_signals.append("less_solver_work_observed")
    if better and mean_time_to_best is not None and mean_time_to_best > 0:
        mechanism_interpretation = "solver_outcome_and_search_progress_signal"
    elif better and "more_solver_work_observed" in mechanism_signals:
        mechanism_interpretation = "solver_outcome_and_throughput_signal"
    elif better:
        mechanism_interpretation = "solver_outcome_signal_mechanism_unresolved"
    elif not better and (mean_wall or 0) > 0:
        mechanism_interpretation = "runtime_only_signal"
    else:
        mechanism_interpretation = "no_positive_mechanism_signal"

    report = {
        "schema_version": 2,
        "task_id": config.get("task_id"),
        "performance_protocol": protocol,
        "budgets_sec": budgets,
        "candidate_patch_sha256": sorted(
            {
                str(agent["patch_sha256"])
                for _, agent in pairs
                if agent.get("patch_sha256")
            }
        ),
        "paired_runs": len(pairs),
        "independent_instances": len(grouped),
        "missing_base_results": missing_base,
        "missing_declared_runs": missing_declared_runs,
        "invalid_candidate_runs": invalid,
        "candidate_runner_failures": runner_failures,
        "instance_direction_counts": direction_counts,
        "observed_pattern": pattern,
        "runtime": {
            key: behavior_summary[key]
            for key in (
                "mean_wall_time_reduction_sec",
                "median_wall_time_reduction_sec",
                "mean_cpu_time_reduction_sec",
            )
        },
        "solver_behavior": {
            key: behavior_summary[key]
            for key in (
                "trajectory_artifact_pairs",
                "progress_trace_pairs",
                "mean_time_to_best_reduction_sec",
                "mean_time_to_best_dual_bound_reduction_sec",
                "mean_candidate_minus_base_work",
            )
        },
        "contrastive_progress": contrastive_progress_summary(pairs),
        "progress_interpretation": (
            "Observed trace checkpoints are diagnostic only. Positive paired "
            "objective or dual-bound differences mean better search progress at "
            "that time. Raw differences are only averaged within an instance; "
            "across instances only direction counts are pooled. These are not "
            "verified final scores or proof certificates."
        ),
        "gap_area_interpretation": (
            "Gap area uses solver-reported primal and dual bounds only where both "
            "were observed. Compare coverage before interpreting areas; unequal "
            "observed windows cannot establish a better full-budget integral."
        ),
        "mechanism_signals": mechanism_signals,
        "mechanism_interpretation": mechanism_interpretation,
        "possible_explanations": possible_explanations,
        "recommended_next_actions": list(dict.fromkeys(next_actions)),
        "per_instance": per_instance,
        "claim_boundary": (
            "Development diagnosis only. It does not establish hidden-instance "
            "generalization or a confirmed improvement."
        ),
    }
    if config.get("agent_feedback_mode") == "outcome_only":
        for key in (
            "solver_behavior",
            "contrastive_progress",
            "progress_interpretation",
            "gap_area_interpretation",
            "mechanism_signals",
            "mechanism_interpretation",
        ):
            report.pop(key, None)
        report["possible_explanations"] = [
            explanation
            for explanation in report["possible_explanations"]
            if "search-stage timing" not in explanation
        ]
        for instance in report["per_instance"]:
            for key in ("behavior", "contrastive_progress", "observed_gap_area"):
                instance.pop(key, None)
    return report
