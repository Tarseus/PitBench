"""Metrics recomputed from complete, independently checked scenario evidence."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from pitbench.evaluator.representations import representation_type
from pitbench.metrics.performance_report import BOOTSTRAP_SEED, _cluster_mean_ci
from pitbench.problem_families.verification import NumericModelFamily
from pitbench.schema.scenarios import ScenarioCampaign, ScenarioProtocol, ScenarioRun


def sequence_key(
    instance_set: str, instance_id: str, scenario_id: str | None = None
) -> str:
    return json.dumps(
        [scenario_id, instance_set, instance_id]
        if scenario_id is not None
        else [instance_set, instance_id],
        separators=(",", ":"),
    )


class ScenarioMetricResult(BaseModel):
    scenario_id: str
    protocol_sha256: str
    metric_id: str
    instance_set: str
    direction: str
    complete: bool = False
    quality_qualified: bool = False
    classification: str = "incomplete"
    base: float | None = None
    agent: float | None = None
    degradation: float | None = None
    lower: float | None = None
    upper: float | None = None
    confidence_level: float | None = None
    expected_runs: int
    observed_runs: int
    by_instance: dict[str, dict[str, Any]] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)


class ScenarioReport(BaseModel):
    protocol_version: str = "1.0"
    task_id: str
    task_sha256: str
    candidate_sha256: str
    order: str
    runtime_identities: dict[str, str]
    hardware_identity: dict
    protocol_hashes: dict[str, str]
    results: list[ScenarioMetricResult]
    invalid_runs: list[dict] = Field(default_factory=list)
    missing_run_count: int
    complete: bool
    design_sha256: str
    collection_id: str
    started_at: float | None
    finished_at: float | None


def _reduce(values, method):
    if not values:
        raise ValueError("a declared reduction has no observations")
    return (
        statistics.fmean(values)
        if method == "mean"
        else max(values)
        if method == "maximum"
        else values[-1]
    )


def _first_passage(events, target, sense):
    for time, value in events:
        if value <= target if sense == "minimize" else value >= target:
            return time
    return None


def _integral(events, *, horizon, reference, scale, missing_loss, sign):
    last_time, best, area = 0.0, None, 0.0
    for time, value in events:
        loss = missing_loss if best is None else sign * (best - reference) / scale
        area += (time - last_time) * loss
        best = (
            value
            if best is None
            else min(best, value)
            if sign == 1
            else max(best, value)
        )
        last_time = time
    loss = missing_loss if best is None else sign * (best - reference) / scale
    return area + (horizon - last_time) * loss


def _check_solution(path, model, protocol, values, payload=None):
    if isinstance(model, dict) and "matrix" in model:
        return NumericModelFamily.check_model(
            model,
            values if values is not None else (payload or {}).get("values"),
            protocol.feasibility_tolerance,
        )
    if payload is None:
        return {
            "feasible": False,
            "reason": "the representation requires a structured solution",
        }
    implementation = representation_type(protocol.representation)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".solution.json") as candidate:
        json.dump(payload, candidate, allow_nan=False)
        candidate.flush()
        return (
            implementation.verifier(
                protocol.feasibility_tolerance
                if implementation.requires_tolerance
                else None
            )
            .verify(path, Path(candidate.name))
            .model_dump(mode="json")
        )


def _checked_run(
    run: ScenarioRun, protocol: ScenarioProtocol, expected_models: list[dict]
) -> list[dict]:
    """Ignore claimed objective/validity fields; recompute from original models."""
    if run.status != "returned":
        raise ValueError(f"worker status: {run.status}; {run.error or ''}")
    if len(run.steps) != len(expected_models) or not run.steps:
        raise ValueError("the recorded model sequence is incomplete")
    if protocol.model_sequences_sha256.get(run.key.instance_set, {}).get(
        run.key.instance_id
    ) != [item["sha256"] for item in expected_models]:
        raise ValueError(
            "the model sequence is not preregistered in the scenario protocol"
        )
    if (
        protocol.kind in {"trajectory", "scaling", "numerical", "replay", "setup"}
        and len(run.steps) != 1
    ):
        raise ValueError("this scenario requires one solve per run")
    identities = {(step.process_boot_id, step.process_id) for step in run.steps}
    if run.key.mode == "warm" and len(identities) != 1:
        raise ValueError("warm sequence changed process identity")
    result = []
    for step, expected in zip(run.steps, expected_models, strict=True):
        path = Path(step.instance_path)
        if (
            step.instance_sha256 != expected["sha256"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]
        ):
            raise ValueError("a scenario model hash changed")
        model = representation_type(protocol.representation).read_input(path)
        numeric = isinstance(model, dict) and "matrix" in model
        if numeric:
            sense = "minimize" if model["sense"] in {1, "minimize"} else "maximize"
            if sense != protocol.objective_sense:
                raise ValueError(
                    "scenario objective sense differs from the original model"
                )
        if step.termination not in {
            "optimal",
            "time_limit",
            "infeasible",
            "other",
        } and not (protocol.kind == "numerical" and step.termination == "solver_error"):
            raise ValueError(
                "solver failure cannot supply successful scenario measurements"
            )
        times = [e.time_sec for e in step.events]
        if times != sorted(times) or any(t > step.solve_time_sec for t in times):
            raise ValueError(
                "trajectory timestamps are not in the measured solve interval"
            )
        primal, dual, invalid = [], [], False
        if protocol.kind == "numerical":
            witness = protocol.reference_solution_by_instance.get(run.key.instance_id)
            if (
                witness is None
                or not _check_solution(
                    path,
                    model,
                    protocol,
                    witness if isinstance(witness, list) else None,
                    witness if isinstance(witness, dict) else None,
                )["feasible"]
            ):
                raise ValueError(
                    "numerical stress reference witness does not verify on the original model"
                )
        for event in step.events:
            if event.solution_values is not None or event.solution is not None:
                checked = _check_solution(
                    path, model, protocol, event.solution_values, event.solution
                )
                if checked.get("infrastructure_error"):
                    raise ValueError(
                        checked.get(
                            "reason",
                            checked.get("detail", "independent checker unavailable"),
                        )
                    )
                if not checked["feasible"]:
                    invalid = True
                elif event.time_sec <= protocol.budget_sec:
                    primal.append((event.time_sec, checked["objective"]))
            if (
                event.row_multipliers is not None
                and event.time_sec <= protocol.budget_sec
                and numeric
            ):
                bound = NumericModelFamily.certified_relaxation_bound(
                    model, event.row_multipliers
                )
                if bound is not None:
                    dual.append((event.time_sec, bound))
        final = None
        if step.final_solution_values is not None or step.final_solution is not None:
            final = _check_solution(
                path, model, protocol, step.final_solution_values, step.final_solution
            )
            if final.get("infrastructure_error"):
                raise ValueError(
                    final.get(
                        "reason", final.get("detail", "independent checker unavailable")
                    )
                )
            invalid |= not final["feasible"]
        if step.termination == "optimal" and not (final or {}).get("feasible"):
            invalid = True
        result.append(
            {
                "step": step,
                "primal": primal,
                "dual": dual,
                "final": final,
                "invalid": invalid,
                "quality_qualified": bool(
                    final
                    and final.get("feasible")
                    and step.instance_sha256
                    in protocol.qualification_target_by_model_sha256
                    and (
                        final["objective"]
                        <= protocol.qualification_target_by_model_sha256[
                            step.instance_sha256
                        ]
                        if protocol.objective_sense == "minimize"
                        else final["objective"]
                        >= protocol.qualification_target_by_model_sha256[
                            step.instance_sha256
                        ]
                    )
                ),
            }
        )
    return result


def _single_measurements(
    run: ScenarioRun, p: ScenarioProtocol, checked: list[dict]
) -> dict[str, Any]:
    first = checked[0]
    step = first["step"]
    iid = run.key.instance_id
    values = {}
    censor = p.budget_sec * (2 if p.censoring == "par2" else 1)
    if p.kind == "trajectory":
        first_time = first["primal"][0][0] if first["primal"] else None
        values["time_to_first_verified_solution_sec"] = {
            "value": first_time if first_time is not None else censor,
            "censored": first_time is None,
        }
        if "time_to_target_sec" in p.metrics:
            time = _first_passage(
                first["primal"], p.target_by_instance[iid], p.objective_sense
            )
            values["time_to_target_sec"] = {
                "value": time if time is not None else censor,
                "censored": time is None,
            }
        for metric, events, sign in (
            (
                "primal_integral",
                first["primal"],
                1 if p.objective_sense == "minimize" else -1,
            ),
            (
                "dual_progress_integral",
                first["dual"],
                -1 if p.objective_sense == "minimize" else 1,
            ),
        ):
            if metric not in p.metrics:
                continue
            if metric == "dual_progress_integral" and not events:
                values[metric] = {
                    "value": None,
                    "reason": "no independently evaluated bound certificate is available",
                }
                continue
            values[metric] = {
                "value": _integral(
                    events,
                    horizon=p.budget_sec,
                    reference=p.reference_by_instance[iid],
                    scale=p.loss_scale_by_instance[iid],
                    missing_loss=p.missing_initial_loss,
                    sign=sign,
                ),
                "events": events,
            }
    if p.kind == "lifecycle":
        series = [item["step"].retained_rss_bytes for item in checked][p.warmup_steps :]
        if len(series) < 2:
            raise ValueError("memory growth needs at least two post-warm-up steps")
        center = (len(series) - 1) / 2
        slope = sum((i - center) * value for i, value in enumerate(series)) / sum(
            (i - center) ** 2 for i in range(len(series))
        )
        values["retained_memory_series"] = {
            "value": _reduce(series, p.series_reduction),
            "series": series,
        }
        values["memory_growth_per_solve"] = {
            "value": slope,
            "series": series,
            "estimator": "ols",
        }
    if p.kind == "sequence":
        values["sequence_solve_time_sec"] = {
            "value": sum(item["step"].solve_time_sec for item in checked)
        }
        values["update_overhead_sec"] = {
            "value": sum(item["step"].update_time_sec for item in checked)
        }
    if p.kind == "setup":
        values["setup_time_sec"] = {"value": step.setup_time_sec}
        values["setup_peak_memory"] = {
            "value": step.setup_peak_rss_bytes,
            "sample_interval_sec": p.memory_sample_interval_sec,
        }
    if p.kind == "numerical":
        final = first["final"]
        residuals = (
            [
                final.get(k)
                for k in (
                    "max_bound_violation",
                    "max_row_violation",
                    "max_integrality_violation",
                )
            ]
            if final
            else []
        )
        values["independent_violation"] = {
            "value": max(residuals)
            if residuals and all(v is not None for v in residuals)
            else None,
            "residuals": final,
        }
        values["numerical_failure_counts"] = {
            "value": int(
                first["invalid"] or step.termination in {"solver_error", "infeasible"}
            )
        }
    if p.kind == "scaling":
        time = _first_passage(
            first["primal"], p.target_by_instance[iid], p.objective_sense
        )
        values["target_time"] = {"value": time, "censored": time is None}
    if p.kind == "replay":
        final = first["final"]
        if not (final or {}).get("feasible"):
            raise ValueError(
                "replay requires a verified final solution in every repeat"
            )
        values["replay"] = {
            "objective": final["objective"],
            "termination": step.termination,
            "solution_hash": hashlib.sha256(
                json.dumps(
                    step.final_solution_values
                    if step.final_solution_values is not None
                    else step.final_solution,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        }
    return values


def _replay_agreement(records, p):
    agree, pairs = 0, list(itertools.combinations(records, 2))
    if not pairs:
        raise ValueError("replay comparison needs at least two repeats")
    for left, right in pairs:
        same = True
        for field in p.replay_fields:
            a, b = left[field], right[field]
            same &= (
                math.isclose(
                    a,
                    b,
                    rel_tol=p.replay_relative_tolerance,
                    abs_tol=p.replay_absolute_tolerance,
                )
                if field == "objective"
                else a == b
            )
        agree += same
    return {
        "value": agree / len(pairs),
        "matching_pairs": agree,
        "expected_pairs": len(pairs),
    }


def compute_scenario_report(campaign: ScenarioCampaign) -> ScenarioReport:
    indexed = {run.key: run for run in campaign.runs}
    measured, issues, invalid_runs = {}, {}, []
    for p in campaign.protocols:
        for key in [k for k in campaign.expected_runs if k.scenario_id == p.id]:
            try:
                if key not in indexed:
                    raise ValueError("missing declared scenario run")
                models = campaign.model_sequences.get(
                    sequence_key(key.instance_set, key.instance_id, p.id)
                )
                if models is None:
                    models = campaign.model_sequences[
                        sequence_key(key.instance_set, key.instance_id)
                    ]
                checked = _checked_run(indexed[key], p, models)
                if any(item["invalid"] for item in checked):
                    invalid_runs.append(key.model_dump(mode="json"))
                measured[key] = _single_measurements(indexed[key], p, checked)
                measured[key]["__quality_qualified"] = all(
                    item["quality_qualified"] for item in checked
                )
            except (KeyError, OSError, ValueError, TypeError) as error:
                issues[key] = str(error)
    results = []
    for p in campaign.protocols:
        for panel in p.instance_sets:
            keys = [
                key
                for key in campaign.expected_runs
                if key.scenario_id == p.id and key.instance_set == panel
            ]
            for metric in p.metrics:
                higher = metric in {
                    "parallel_efficiency",
                    "same_seed_replay_consistency",
                } or (
                    metric == "thread_scaling_curve"
                    and p.curve_reduction == "mean_efficiency"
                )
                result = ScenarioMetricResult(
                    scenario_id=p.id,
                    protocol_sha256=p.digest(),
                    metric_id=metric,
                    instance_set=panel,
                    direction="higher" if higher else "lower",
                    expected_runs=len(keys),
                    observed_runs=sum(k in indexed for k in keys),
                )
                result.issues = sorted({issues[k] for k in keys if k in issues})
                result.quality_qualified = all(
                    k in measured and measured[k].get("__quality_qualified")
                    for k in keys
                )
                cells = defaultdict(dict)
                for key in keys:
                    if key in measured:
                        cells[
                            (
                                key.instance_id,
                                key.solver_seed,
                                key.repeat,
                                key.mode,
                                key.code_state,
                            )
                        ][key.threads] = measured[key]
                per_instance = defaultdict(lambda: defaultdict(list))
                for (iid, seed, repeat, mode, state), threads in cells.items():
                    try:
                        if p.kind == "sequence" and mode != p.sequence_mode:
                            continue
                        if p.kind == "scaling":
                            times = {
                                n: threads[n]["target_time"]["value"]
                                for n in p.thread_counts
                            }
                            if any(
                                value is None or value <= 0 for value in times.values()
                            ):
                                raise ValueError(
                                    "parallel efficiency cannot use censored or zero target times"
                                )
                            efficiencies = {
                                n: times[p.reference_threads]
                                / value
                                * p.reference_threads
                                / n
                                for n, value in times.items()
                            }
                            scalar = (
                                statistics.fmean(efficiencies.values())
                                if metric == "parallel_efficiency"
                                or p.curve_reduction == "mean_efficiency"
                                else statistics.fmean(times.values())
                            )
                            value = {
                                "value": scalar,
                                "target_times": times,
                                "efficiencies": efficiencies,
                            }
                            per_instance[iid][state].append(value)
                        elif p.kind != "replay":
                            for measures in threads.values():
                                value = measures[metric]
                                if value["value"] is None or not math.isfinite(
                                    value["value"]
                                ):
                                    raise ValueError(
                                        "a required metric value is unavailable"
                                    )
                                per_instance[iid][state].append(value)
                    except (KeyError, ValueError) as error:
                        result.issues.append(str(error))
                if p.kind == "replay":
                    for iid in campaign.instances_by_set[panel]:
                        for state, seed, threads, mode in itertools.product(
                            ("base", "agent"), p.solver_seeds, p.thread_counts, p.modes
                        ):
                            try:
                                values = [
                                    measured[
                                        next(
                                            k
                                            for k in keys
                                            if (
                                                k.instance_id,
                                                k.code_state,
                                                k.solver_seed,
                                                k.threads,
                                                k.mode,
                                                k.repeat,
                                            )
                                            == (iid, state, seed, threads, mode, repeat)
                                        )
                                    ]["replay"]
                                    for repeat in range(p.repeats)
                                ]
                                per_instance[iid][state].append(
                                    _replay_agreement(values, p)
                                )
                            except (StopIteration, KeyError, ValueError) as error:
                                result.issues.append("incomplete replay: " + str(error))
                paired = {}
                base, agent = [], []
                for iid in campaign.instances_by_set[panel]:
                    sides = per_instance[iid]
                    if (
                        not sides["base"]
                        or not sides["agent"]
                        or len(sides["base"]) != len(sides["agent"])
                    ):
                        result.issues.append(f"{iid}: incomplete paired conditions")
                        continue
                    b, a = (
                        statistics.fmean(v["value"] for v in sides[state])
                        for state in ("base", "agent")
                    )
                    result.by_instance[iid] = {
                        "base": b,
                        "agent": a,
                        "records": dict(sides),
                    }
                    base.append(b)
                    agent.append(a)
                    paired[(panel, iid)] = [(b - a) if higher else (a - b)]
                if not result.issues and len(paired) == len(
                    campaign.instances_by_set[panel]
                ):
                    result.base, result.agent = (
                        statistics.fmean(base),
                        statistics.fmean(agent),
                    )
                    result.degradation = statistics.fmean(v[0] for v in paired.values())
                    ci = _cluster_mean_ci(paired, seed=BOOTSTRAP_SEED)
                    result.complete = True
                    if ci is not None and len(paired) >= 2:
                        result.lower, result.upper, result.confidence_level = (
                            ci.lower,
                            ci.upper,
                            ci.level,
                        )
                        result.classification = (
                            "improved"
                            if ci.upper < 0
                            else "regressed"
                            if ci.lower > 0
                            else "inconclusive"
                        )
                    else:
                        result.classification = "inconclusive"
                result.issues = sorted(set(result.issues))
                results.append(result)
    missing = len(set(campaign.expected_runs) - set(indexed))
    design = {
        "task_sha256": campaign.task_sha256,
        "candidate_sha256": campaign.candidate_sha256,
        "protocols": {p.id: p.digest() for p in campaign.protocols},
        "models": {
            k: [s["sha256"] for s in v] for k, v in campaign.model_sequences.items()
        },
        "grid": sorted(
            (k.model_dump(mode="json") for k in campaign.expected_runs),
            key=lambda k: json.dumps(k, sort_keys=True),
        ),
        "runtime_identities": campaign.runtime_identities,
        "hardware_identity": campaign.hardware_identity,
    }
    design_hash = hashlib.sha256(
        json.dumps(design, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return ScenarioReport(
        task_id=campaign.task_id,
        task_sha256=campaign.task_sha256,
        candidate_sha256=campaign.candidate_sha256,
        order=campaign.order,
        runtime_identities=campaign.runtime_identities,
        hardware_identity=campaign.hardware_identity,
        protocol_hashes={p.id: p.digest() for p in campaign.protocols},
        results=results,
        invalid_runs=invalid_runs,
        missing_run_count=missing,
        complete=not missing
        and bool(results)
        and all(item.complete for item in results),
        design_sha256=design_hash,
        collection_id=campaign.collection_id,
        started_at=campaign.started_at,
        finished_at=campaign.finished_at,
    )
