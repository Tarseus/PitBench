"""Paired resource ratios: seed means within instances, geometric means across them.

Memory is the primary observation and CPU is auxiliary. These are resource-use
ratios, not quality-adjusted scores or acceptance criteria. Only original inputs
are included; each instance set and time budget is reported separately.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, Field

from pitbench.schema.observation import CodeState, RunObservation, RunStatus


class ResourceRatio(BaseModel):
    paired_seed_count: int = 0
    base_mean: float | None = None
    agent_mean: float | None = None
    agent_base_ratio: float | None = None
    excluded_pairs: dict[str, int] = Field(default_factory=dict)
    measurement_scope: str | None = None


class InstanceResources(BaseModel):
    instance_id: str
    observed_seed_count: int
    missing_seed_count: int
    base_status_counts: dict[str, int]
    agent_status_counts: dict[str, int]
    memory: ResourceRatio
    cpu: ResourceRatio
    paired_quality_seed_count: int = 0
    mean_gap_change: float | None = None


class AggregateResources(BaseModel):
    geometric_mean_ratio: float | None = None
    paired_instance_count: int = 0
    complete_instance_count: int = 0
    paired_seed_count: int = 0


class BudgetResources(BaseModel):
    budget_sec: float
    instance_count: int
    expected_instance_count: int
    memory: AggregateResources
    cpu: AggregateResources


class ResourceReport(BaseModel):
    task_id: str
    primary_resource: str = "peak_rss_bytes"
    primary_budget_sec: float
    expected_seed_count: int = Field(gt=0)
    by_instance_set: dict[str, dict[str, BudgetResources]]


class ResourceDetails(BaseModel):
    task_id: str
    expected_seed_count: int
    memory_unit: str = "bytes"
    cpu_unit: str = "seconds"
    gap_change_direction: str = "agent_minus_base"
    by_instance_set: dict[str, dict[str, list[InstanceResources]]]


def _pair_problem(
    base: RunObservation | None, agent: RunObservation | None
) -> str | None:
    if base is None and agent is None:
        return "missing_both"
    for name, observation in (("base", base), ("agent", agent)):
        if observation is None:
            return f"missing_{name}"
        if not observation.valid or observation.status != RunStatus.COMPLETED:
            return f"invalid_{name}"
    assert base is not None and agent is not None
    if base.threads != agent.threads:
        return "thread_count_mismatch"
    return None


def _resource_ratio(
    pairs: Sequence[tuple[RunObservation | None, RunObservation | None]],
    field: str,
    missing_seed_count: int,
) -> ResourceRatio:
    excluded = Counter()
    if missing_seed_count:
        excluded["missing_both"] = missing_seed_count
    values: list[tuple[float, float]] = []
    scopes = set()
    for base, agent in pairs:
        problem = _pair_problem(base, agent)
        if problem is not None:
            excluded[problem] += 1
            continue
        assert base is not None and agent is not None
        if any(
            not item.resource_scope or item.resource_scope.startswith("legacy_")
            for item in (base, agent)
        ):
            excluded["unqualified_measurement_scope"] += 1
            continue
        if base.resource_scope != agent.resource_scope:
            excluded["measurement_scope_mismatch"] += 1
            continue
        base_value = getattr(base, field)
        agent_value = getattr(agent, field)
        if any(
            value is None or not math.isfinite(value) or value <= 0
            for value in (base_value, agent_value)
        ):
            excluded["missing_or_nonpositive_measurement"] += 1
            continue
        scopes.add(base.resource_scope)
        values.append((base_value, agent_value))
    if len(scopes) > 1:
        excluded["mixed_measurement_scopes"] += len(values)
        values = []
    if not values:
        return ResourceRatio(excluded_pairs=dict(excluded))
    base_mean = statistics.fmean(value[0] for value in values)
    agent_mean = statistics.fmean(value[1] for value in values)
    return ResourceRatio(
        paired_seed_count=len(values),
        base_mean=base_mean,
        agent_mean=agent_mean,
        agent_base_ratio=agent_mean / base_mean,
        measurement_scope=next(iter(scopes)),
        excluded_pairs=dict(excluded),
    )


def _aggregate(
    instances: list[InstanceResources], field: str, expected: int
) -> AggregateResources:
    values = [getattr(instance, field) for instance in instances]
    ratios = [
        value.agent_base_ratio for value in values if value.agent_base_ratio is not None
    ]
    return AggregateResources(
        geometric_mean_ratio=(
            math.exp(statistics.fmean(math.log(ratio) for ratio in ratios))
            if ratios
            else None
        ),
        paired_instance_count=len(ratios),
        complete_instance_count=sum(
            value.paired_seed_count == expected for value in values
        ),
        paired_seed_count=sum(value.paired_seed_count for value in values),
    )


def compute_resource_reports(
    observations: Sequence[RunObservation],
    *,
    primary_budget_sec: float,
    expected_seed_count: int = 30,
    budgets_sec: Sequence[float] | None = None,
    expected_instance_counts: Mapping[str, int] | None = None,
) -> tuple[ResourceReport, ResourceDetails]:
    """Return a public aggregate and private per-instance resource details.

    Means use the same eligible paired seeds on both sides, even if solution
    quality worsens. Incomplete results remain explicitly conditional on the
    available pairs; missing resources never become zero or a saving. Raw run
    statuses, solutions and solver bounds remain in the observation artifact.
    """
    if expected_seed_count < 1:
        raise ValueError("expected_seed_count must be positive")
    selected = [
        item
        for item in observations
        if item.equivalence_parent_id is None and item.test_suite is None
    ]
    task_ids = {item.task_id for item in selected}
    if len(task_ids) != 1:
        raise ValueError(
            "resource report requires original observations from exactly one task"
        )
    budgets = sorted(
        set(budgets_sec)
        if budgets_sec is not None
        else {item.budget_sec for item in selected}
    )
    if primary_budget_sec not in budgets or any(
        not math.isfinite(value) or value <= 0 for value in budgets
    ):
        raise ValueError("primary budget must belong to positive finite report budgets")
    if any(item.budget_sec not in budgets for item in selected):
        raise ValueError("observations contain an undeclared budget")
    keyed = {}
    instance_sets: dict[str, set[str]] = defaultdict(set)
    seeds_by_instance: dict[tuple[str, str], set[int]] = defaultdict(set)
    for item in selected:
        key = (
            item.instance_set,
            item.instance_id,
            item.budget_sec,
            item.solver_seed,
            item.code_state,
        )
        if key in keyed:
            raise ValueError(f"duplicate resource observation: {key}")
        keyed[key] = item
        instance_sets[item.instance_set].add(item.instance_id)
        seeds_by_instance[(item.instance_set, item.instance_id)].add(item.solver_seed)
    if any(len(seeds) > expected_seed_count for seeds in seeds_by_instance.values()):
        raise ValueError("observed seed count exceeds the declared seed count")
    if expected_instance_counts is not None:
        if set(instance_sets) - set(expected_instance_counts):
            raise ValueError("observations contain an undeclared instance set")
        for instance_set, count in expected_instance_counts.items():
            if count < 1 or len(instance_sets[instance_set]) > count:
                raise ValueError(
                    "observed instance count exceeds the declared instance count"
                )
    aggregate_sets = {}
    detailed_sets = {}
    for instance_set, instance_ids in sorted(instance_sets.items()):
        aggregate_budgets = {}
        detailed_budgets = {}
        for budget in budgets:
            instances = []
            for instance_id in sorted(instance_ids):
                seeds = sorted(seeds_by_instance[(instance_set, instance_id)])
                pairs = [
                    (
                        keyed.get(
                            (instance_set, instance_id, budget, seed, CodeState.BASE)
                        ),
                        keyed.get(
                            (instance_set, instance_id, budget, seed, CodeState.AGENT)
                        ),
                    )
                    for seed in seeds
                ]
                missing = expected_seed_count - len(seeds)
                quality_changes = [
                    agent.normalized_gap - base.normalized_gap
                    for base, agent in pairs
                    if _pair_problem(base, agent) is None
                    and base.normalized_gap is not None
                    and agent.normalized_gap is not None
                    and math.isfinite(base.normalized_gap)
                    and math.isfinite(agent.normalized_gap)
                ]
                states = []
                for index in (0, 1):
                    counts = Counter(
                        "missing"
                        if pair[index] is None
                        else (
                            "invalid"
                            if not pair[index].valid
                            and pair[index].status == RunStatus.COMPLETED
                            else pair[index].status.value
                        )
                        for pair in pairs
                    )
                    if missing:
                        counts["missing"] += missing
                    states.append(dict(counts))
                instances.append(
                    InstanceResources(
                        instance_id=instance_id,
                        observed_seed_count=sum(
                            base is not None or agent is not None
                            for base, agent in pairs
                        ),
                        missing_seed_count=missing
                        + sum(base is None and agent is None for base, agent in pairs),
                        base_status_counts=states[0],
                        agent_status_counts=states[1],
                        memory=_resource_ratio(pairs, "peak_rss_bytes", missing),
                        cpu=_resource_ratio(pairs, "cpu_time_sec", missing),
                        paired_quality_seed_count=len(quality_changes),
                        mean_gap_change=statistics.fmean(quality_changes)
                        if quality_changes
                        else None,
                    )
                )
            detailed_budgets[f"{budget:g}"] = instances
            aggregate_budgets[f"{budget:g}"] = BudgetResources(
                budget_sec=budget,
                instance_count=len(instances),
                expected_instance_count=(
                    expected_instance_counts[instance_set]
                    if expected_instance_counts is not None
                    else len(instances)
                ),
                memory=_aggregate(instances, "memory", expected_seed_count),
                cpu=_aggregate(instances, "cpu", expected_seed_count),
            )
        aggregate_sets[instance_set] = aggregate_budgets
        detailed_sets[instance_set] = detailed_budgets
    task_id = next(iter(task_ids))
    return (
        ResourceReport(
            task_id=task_id,
            primary_budget_sec=primary_budget_sec,
            expected_seed_count=expected_seed_count,
            by_instance_set=aggregate_sets,
        ),
        ResourceDetails(
            task_id=task_id,
            expected_seed_count=expected_seed_count,
            by_instance_set=detailed_sets,
        ),
    )


def compute_resource_report(
    observations: Sequence[RunObservation], **kwargs
) -> ResourceReport:
    return compute_resource_reports(observations, **kwargs)[0]


def format_resource_report(report: ResourceReport) -> str:
    lines = [
        "Resource usage (Agent/Base; lower uses less resource)",
        f"Primary: peak RSS. Expected seeds per instance: {report.expected_seed_count}.",
    ]
    for instance_set, budgets in report.by_instance_set.items():
        for cell in budgets.values():
            memory = cell.memory
            cpu = cell.cpu
            memory_ratio = (
                "-"
                if memory.geometric_mean_ratio is None
                else f"{memory.geometric_mean_ratio:.4f}"
            )
            cpu_ratio = (
                "-"
                if cpu.geometric_mean_ratio is None
                else f"{cpu.geometric_mean_ratio:.4f}"
            )
            lines.append(
                f"{instance_set} / {cell.budget_sec:g}s: memory {memory_ratio}, CPU {cpu_ratio}; "
                f"memory pairs {memory.paired_seed_count}/{cell.expected_instance_count * report.expected_seed_count}, "
                f"complete instances {memory.complete_instance_count}/{cell.expected_instance_count}; "
                f"CPU pairs {cpu.paired_seed_count}/{cell.expected_instance_count * report.expected_seed_count}."
            )
    lines.append(
        "Ratios divide paired seed means within each instance, then take an equal-instance geometric mean."
    )
    lines.append(
        "Incomplete ratios describe available pairs only. Interpret with quality and failure records; no combined score."
    )
    return "\n".join(lines)
