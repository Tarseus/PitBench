"""Descriptive pass counts on a declared boundary test grid, without imputation."""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, Field

from pitbench.instances.boundary import boundary_suite
from pitbench.schema.observation import CodeState, RunObservation, RunStatus
from pitbench.schema.task import PitBenchTask


class ReliabilityCounts(BaseModel):
    expected_runs: int
    observed_runs: int
    passed_runs: int
    failed_runs: int
    missing_runs: int
    pass_rate: float | None
    status_counts: dict[str, int]
    invalid_solutions: int


class ReliabilityBudget(BaseModel):
    budget_sec: float
    by_state: dict[str, ReliabilityCounts]
    paired_pass_to_fail: int = 0
    paired_fail_to_pass: int = 0


class ReliabilityReport(BaseModel):
    task_id: str
    test_suite: str = "operational_reliability"
    interpretation: str = "pass rate on the declared boundary test grid"
    instance_count: int
    solver_seeds: list[int]
    complete: bool
    qualification_passed: bool
    by_budget: dict[str, ReliabilityBudget]


class ReliabilityCaseDetail(BaseModel):
    instance_id: str
    description: str
    budget_sec: float
    by_state: dict[str, ReliabilityCounts]
    failed_runs: list[RunObservation] = Field(default_factory=list)
    missing_runs: list[dict] = Field(default_factory=list)
    changed_seeds: list[dict] = Field(default_factory=list)


class ReliabilityDetails(BaseModel):
    task_id: str
    cases: list[ReliabilityCaseDetail]


def _passed(observation: RunObservation) -> bool:
    return observation.status == RunStatus.COMPLETED and observation.valid


def _counts(records: list[RunObservation], expected: int) -> ReliabilityCounts:
    passed = sum(_passed(item) for item in records)
    return ReliabilityCounts(
        expected_runs=expected,
        observed_runs=len(records),
        passed_runs=passed,
        failed_runs=len(records) - passed,
        missing_runs=expected - len(records),
        pass_rate=passed / expected if len(records) == expected else None,
        status_counts=dict(
            sorted(Counter(item.status.value for item in records).items())
        ),
        invalid_solutions=sum(item.status == RunStatus.INVALID for item in records),
    )


def compute_reliability_reports(
    observations: list[RunObservation],
    *,
    task: PitBenchTask,
    code_states: tuple[CodeState, ...] = tuple(CodeState),
) -> tuple[ReliabilityReport, ReliabilityDetails]:
    examples = boundary_suite(task.problem_family).cases()
    seeds = (
        task.evaluation.seed_robustness.development_seeds
        if task.evaluation.seed_robustness is not None
        else task.evaluation.solver_seeds
    )
    assert seeds is not None
    budgets = task.evaluation.budgets_sec
    names = {case.name for case in examples}
    if not code_states or len(set(code_states)) != len(code_states):
        raise ValueError("reliability report requires distinct code states")
    indexed = {}
    for item in observations:
        if item.test_suite != "operational_reliability":
            continue
        if item.task_id != task.task_id:
            raise ValueError("reliability observation belongs to another task")
        if (
            item.instance_id not in names
            or item.instance_set != "boundary_cases"
            or item.solver_seed not in seeds
            or item.budget_sec not in budgets
            or item.code_state not in code_states
            or item.threads != task.evaluation.threads
            or item.equivalence_parent_id is not None
        ):
            raise ValueError("reliability observation is outside the declared grid")
        key = (item.instance_id, item.budget_sec, item.solver_seed, item.code_state)
        if key in indexed:
            raise ValueError("duplicate reliability observation")
        indexed[key] = item
    by_budget = {}
    details = []
    for budget in budgets:
        summary = ReliabilityBudget(
            budget_sec=budget,
            by_state={
                state.value: _counts(
                    [
                        item
                        for item in indexed.values()
                        if item.budget_sec == budget and item.code_state == state
                    ],
                    len(examples) * len(seeds),
                )
                for state in code_states
            },
        )
        for example in examples:
            records = [
                item
                for item in indexed.values()
                if item.instance_id == example.name and item.budget_sec == budget
            ]
            detail = ReliabilityCaseDetail(
                instance_id=example.name,
                description=example.description,
                budget_sec=budget,
                by_state={
                    state.value: _counts(
                        [item for item in records if item.code_state == state],
                        len(seeds),
                    )
                    for state in code_states
                },
                failed_runs=[item for item in records if not _passed(item)],
            )
            for seed in seeds:
                for state in code_states:
                    if (example.name, budget, seed, state) not in indexed:
                        detail.missing_runs.append(
                            {"solver_seed": seed, "code_state": state.value}
                        )
                base = indexed.get((example.name, budget, seed, CodeState.BASE))
                agent = indexed.get((example.name, budget, seed, CodeState.AGENT))
                if base is None or agent is None:
                    continue
                if _passed(base) and not _passed(agent):
                    summary.paired_pass_to_fail += 1
                if not _passed(base) and _passed(agent):
                    summary.paired_fail_to_pass += 1
                if (base.status, base.valid) != (agent.status, agent.valid):
                    detail.changed_seeds.append(
                        {
                            "solver_seed": seed,
                            "base_status": base.status.value,
                            "agent_status": agent.status.value,
                        }
                    )
            details.append(detail)
        by_budget[f"{budget:g}"] = summary
    return ReliabilityReport(
        task_id=task.task_id,
        instance_count=len(examples),
        solver_seeds=list(seeds),
        complete=all(
            counts.missing_runs == 0
            for budget in by_budget.values()
            for counts in budget.by_state.values()
        ),
        qualification_passed=not any(
            item.status == RunStatus.INVALID for item in indexed.values()
        ),
        by_budget=by_budget,
    ), ReliabilityDetails(task_id=task.task_id, cases=details)


def format_reliability_report(report: ReliabilityReport) -> str:
    lines = [
        "Operational reliability — boundary test suite",
        f"Cases: {report.instance_count}; seeds per case: {len(report.solver_seeds)}",
    ]
    for budget in report.by_budget.values():
        lines.append(f"Budget {budget.budget_sec:g}s:")
        for state, counts in budget.by_state.items():
            rate = (
                f"{counts.pass_rate:.1%}"
                if counts.pass_rate is not None
                else "incomplete"
            )
            lines.append(
                f"  {state}: {counts.passed_runs}/{counts.expected_runs} passed ({rate}); {counts.failed_runs} failed; {counts.missing_runs} missing"
            )
            failures = {
                status: count
                for status, count in counts.status_counts.items()
                if status != "completed"
            }
            if failures:
                lines.append(f"    {failures}")
        if {"base", "agent"} <= budget.by_state.keys():
            lines.append(
                f"  Paired pass → fail: {budget.paired_pass_to_fail}; fail → pass: {budget.paired_fail_to_pass}"
            )
    if not report.qualification_passed:
        lines.append(
            "Independent verification found invalid solutions; qualification failed."
        )
    return "\n".join(lines)
