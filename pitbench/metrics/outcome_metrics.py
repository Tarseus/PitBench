from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Sequence

from pydantic import BaseModel, Field

from pitbench.schema.observation import CodeState, RunObservation, RunStatus

OUTCOME_REPORT_VERSION = "1.0"


def shifted_geometric_mean(values: Sequence[float], shift: float = 1.0) -> float | None:
    """Compute Shifted Geometric Mean (SGM): exp(mean(log(x + shift))) - shift."""
    if not values:
        return None
    if shift < 0:
        raise ValueError(f"shift must be non-negative, got {shift}")
    non_negative = [max(0.0, float(v)) for v in values]
    log_sum = sum(math.log(v + shift) for v in non_negative)
    return math.exp(log_sum / len(non_negative)) - shift


class PerformanceMetrics(BaseModel):
    """Performance coordinates for runs."""

    valid_runs: int = Field(default=0, ge=0)
    mean_normalized_gap: float | None = None
    median_normalized_gap: float | None = None
    operational_capped_sgm_sec: float | None = None
    common_success_sgm_sec: float | None = None
    sgm_runtime_sec: float | None = None
    mean_wall_time_sec: float | None = None
    min_normalized_gap: float | None = None
    max_normalized_gap: float | None = None


class ReliabilityMetrics(BaseModel):
    """Reliability coordinates and failure taxonomy for all trials."""

    total_runs: int = Field(default=0, ge=0)
    valid_runs: int = Field(default=0, ge=0)
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    failure_counts: dict[str, int] = Field(default_factory=dict)


class ResourceMetrics(BaseModel):
    """Resource consumption coordinates across all runs with telemetry."""

    telemetry_runs: int = Field(default=0, ge=0)
    mean_peak_rss_mb: float | None = None
    median_peak_rss_mb: float | None = None
    mean_cpu_time_sec: float | None = None
    mean_budget_utilization: float | None = None


class OutcomeMetrics(BaseModel):
    """3D Outcome summary for one code state."""

    code_state: CodeState
    performance: PerformanceMetrics
    reliability: ReliabilityMetrics
    resource: ResourceMetrics


class OutcomeComparison(BaseModel):
    """Relative improvement/trade-off comparison of Agent vs Base."""

    operational_speedup: float | None = None
    common_success_speedup: float | None = None
    speedup: float | None = None
    common_success_runs: int = 0
    gap_reduction: float | None = None
    delta_success_rate: float = 0.0
    peak_rss_ratio: float | None = None
    cpu_speedup: float | None = None
    paired_runs_evaluated: int = 0


class OutcomeReport(BaseModel):
    """Complete Outcome 3D report containing Base, Agent, and Comparison."""

    report_version: str = OUTCOME_REPORT_VERSION
    base: OutcomeMetrics
    agent: OutcomeMetrics
    comparison: OutcomeComparison
    by_population: dict[str, OutcomeReport] = Field(default_factory=dict)


def _compute_single_state_metrics(
    observations: Sequence[RunObservation], state: CodeState
) -> OutcomeMetrics:
    total = len(observations)
    valid_obs = [obs for obs in observations if obs.valid]
    valid_count = len(valid_obs)

    # Reliability
    status_counts = Counter(obs.status.value for obs in observations)
    failures = {
        k: v for k, v in status_counts.items() if k != RunStatus.COMPLETED.value
    }
    success_rate = (valid_count / total) if total > 0 else 0.0
    reliability = ReliabilityMetrics(
        total_runs=total,
        valid_runs=valid_count,
        success_rate=success_rate,
        failure_counts=failures,
    )

    # Performance: Normalized gaps on valid runs
    gaps = [obs.normalized_gap for obs in valid_obs if obs.normalized_gap is not None]
    wall_times_valid = [
        obs.wall_time_sec for obs in valid_obs if obs.wall_time_sec is not None
    ]

    mean_gap = statistics.fmean(gaps) if gaps else None
    median_gap = statistics.median(gaps) if gaps else None
    mean_wall_time = statistics.fmean(wall_times_valid) if wall_times_valid else None
    min_gap = min(gaps) if gaps else None
    max_gap = max(gaps) if gaps else None

    # Performance: Operational Capped SGM across all runs
    # (timed out or failed runs are capped at budget_sec)
    capped_times = [
        min(obs.wall_time_sec, obs.budget_sec)
        if (obs.valid and obs.wall_time_sec is not None)
        else obs.budget_sec
        for obs in observations
    ]
    operational_sgm = (
        shifted_geometric_mean(capped_times, shift=1.0) if capped_times else None
    )

    performance = PerformanceMetrics(
        valid_runs=valid_count,
        mean_normalized_gap=mean_gap,
        median_normalized_gap=median_gap,
        operational_capped_sgm_sec=operational_sgm,
        common_success_sgm_sec=None,  # computed during comparison pairing
        sgm_runtime_sec=operational_sgm,
        mean_wall_time_sec=mean_wall_time,
        min_normalized_gap=min_gap,
        max_normalized_gap=max_gap,
    )

    # Resource: Across ALL runs that have telemetry recorded (not just valid == True)
    rss_mb = [
        obs.peak_rss_bytes / (1024 * 1024)
        for obs in observations
        if obs.peak_rss_bytes is not None
    ]
    cpu_times = [
        obs.cpu_time_sec for obs in observations if obs.cpu_time_sec is not None
    ]
    budget_fractions = [
        obs.wall_time_sec / obs.budget_sec
        for obs in observations
        if obs.wall_time_sec is not None and obs.budget_sec > 0
    ]

    mean_rss = statistics.fmean(rss_mb) if rss_mb else None
    median_rss = statistics.median(rss_mb) if rss_mb else None
    mean_cpu = statistics.fmean(cpu_times) if cpu_times else None
    mean_budget = statistics.fmean(budget_fractions) if budget_fractions else None
    telemetry_runs = max(len(rss_mb), len(cpu_times), len(budget_fractions))

    resource = ResourceMetrics(
        telemetry_runs=telemetry_runs,
        mean_peak_rss_mb=mean_rss,
        median_peak_rss_mb=median_rss,
        mean_cpu_time_sec=mean_cpu,
        mean_budget_utilization=mean_budget,
    )

    return OutcomeMetrics(
        code_state=state,
        performance=performance,
        reliability=reliability,
        resource=resource,
    )


def compute_outcome_metrics(
    observations: Sequence[RunObservation],
    *,
    compute_subpopulations: bool = True,
    primary_population_only: bool = True,
) -> OutcomeReport:
    """Compute Outcome 3D without letting diagnostic panels bias the main score."""
    original_observations = [
        obs for obs in observations if obs.equivalence_parent_id is None
    ]
    aggregate_observations = original_observations
    if primary_population_only:
        primary = [
            obs for obs in original_observations if obs.population_kind == "judge_id"
        ]
        if primary:
            aggregate_observations = primary
    base_obs = [
        obs for obs in aggregate_observations if obs.code_state == CodeState.BASE
    ]
    agent_obs = [
        obs for obs in aggregate_observations if obs.code_state == CodeState.AGENT
    ]

    base_metrics = _compute_single_state_metrics(base_obs, CodeState.BASE)
    agent_metrics = _compute_single_state_metrics(agent_obs, CodeState.AGENT)

    # Match paired keys to count common trials and compute common-success SGM
    base_by_key = {
        (obs.population, obs.instance_id, obs.solver_seed, obs.budget_sec): obs
        for obs in base_obs
    }
    agent_by_key = {
        (obs.population, obs.instance_id, obs.solver_seed, obs.budget_sec): obs
        for obs in agent_obs
    }
    common_keys = sorted(set(base_by_key) & set(agent_by_key))
    paired_count = len(common_keys)

    base_common_times: list[float] = []
    agent_common_times: list[float] = []
    for key in common_keys:
        b_obs = base_by_key[key]
        a_obs = agent_by_key[key]
        if (
            b_obs.valid
            and a_obs.valid
            and b_obs.wall_time_sec is not None
            and a_obs.wall_time_sec is not None
        ):
            base_common_times.append(b_obs.wall_time_sec)
            agent_common_times.append(a_obs.wall_time_sec)

    common_success_runs = len(base_common_times)
    base_common_sgm = (
        shifted_geometric_mean(base_common_times, shift=1.0)
        if base_common_times
        else None
    )
    agent_common_sgm = (
        shifted_geometric_mean(agent_common_times, shift=1.0)
        if agent_common_times
        else None
    )

    base_metrics.performance.common_success_sgm_sec = base_common_sgm
    agent_metrics.performance.common_success_sgm_sec = agent_common_sgm

    # Comparison metrics
    operational_speedup: float | None = None
    if (
        base_metrics.performance.operational_capped_sgm_sec is not None
        and agent_metrics.performance.operational_capped_sgm_sec is not None
        and agent_metrics.performance.operational_capped_sgm_sec > 0
    ):
        operational_speedup = (
            base_metrics.performance.operational_capped_sgm_sec
            / agent_metrics.performance.operational_capped_sgm_sec
        )

    common_success_speedup: float | None = None
    if (
        base_common_sgm is not None
        and agent_common_sgm is not None
        and agent_common_sgm > 0
    ):
        common_success_speedup = base_common_sgm / agent_common_sgm

    gap_reduction: float | None = None
    if (
        base_metrics.performance.mean_normalized_gap is not None
        and agent_metrics.performance.mean_normalized_gap is not None
    ):
        gap_reduction = (
            base_metrics.performance.mean_normalized_gap
            - agent_metrics.performance.mean_normalized_gap
        )

    delta_success = (
        agent_metrics.reliability.success_rate - base_metrics.reliability.success_rate
    )

    peak_rss_ratio: float | None = None
    if (
        base_metrics.resource.mean_peak_rss_mb is not None
        and agent_metrics.resource.mean_peak_rss_mb is not None
        and base_metrics.resource.mean_peak_rss_mb > 0
    ):
        peak_rss_ratio = (
            agent_metrics.resource.mean_peak_rss_mb
            / base_metrics.resource.mean_peak_rss_mb
        )

    cpu_speedup: float | None = None
    if (
        base_metrics.resource.mean_cpu_time_sec is not None
        and agent_metrics.resource.mean_cpu_time_sec is not None
        and agent_metrics.resource.mean_cpu_time_sec > 0
    ):
        cpu_speedup = (
            base_metrics.resource.mean_cpu_time_sec
            / agent_metrics.resource.mean_cpu_time_sec
        )

    comparison = OutcomeComparison(
        operational_speedup=operational_speedup,
        common_success_speedup=common_success_speedup,
        speedup=operational_speedup,
        common_success_runs=common_success_runs,
        gap_reduction=gap_reduction,
        delta_success_rate=delta_success,
        peak_rss_ratio=peak_rss_ratio,
        cpu_speedup=cpu_speedup,
        paired_runs_evaluated=paired_count,
    )

    by_population: dict[str, OutcomeReport] = {}
    if compute_subpopulations:
        populations = sorted({obs.population for obs in original_observations})
        if len(populations) > 1:
            for pop in populations:
                pop_obs = [
                    obs for obs in original_observations if obs.population == pop
                ]
                by_population[pop] = compute_outcome_metrics(
                    pop_obs,
                    compute_subpopulations=False,
                    primary_population_only=False,
                )

    return OutcomeReport(
        base=base_metrics,
        agent=agent_metrics,
        comparison=comparison,
        by_population=by_population,
    )


def format_outcome_report_table(report: OutcomeReport) -> str:
    """Format OutcomeReport into a human-readable comparison table."""
    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None  # type: ignore

    rows: list[list[str]] = []

    # Performance
    rows.append(["[Performance]", "", "", ""])
    base_p, agent_p = report.base.performance, report.agent.performance
    rows.append(
        [
            "  Operational Capped SGM (s)",
            (
                f"{base_p.operational_capped_sgm_sec:.3f}"
                if base_p.operational_capped_sgm_sec is not None
                else "-"
            ),
            (
                f"{agent_p.operational_capped_sgm_sec:.3f}"
                if agent_p.operational_capped_sgm_sec is not None
                else "-"
            ),
            (
                f"{report.comparison.operational_speedup:.2f}x speedup"
                if report.comparison.operational_speedup is not None
                else "-"
            ),
        ]
    )
    cs_label = f"  Common-Success SGM (s) [N={report.comparison.common_success_runs}]"
    rows.append(
        [
            cs_label,
            (
                f"{base_p.common_success_sgm_sec:.3f}"
                if base_p.common_success_sgm_sec is not None
                else "-"
            ),
            (
                f"{agent_p.common_success_sgm_sec:.3f}"
                if agent_p.common_success_sgm_sec is not None
                else "-"
            ),
            (
                f"{report.comparison.common_success_speedup:.2f}x speedup"
                if report.comparison.common_success_speedup is not None
                else "-"
            ),
        ]
    )
    rows.append(
        [
            "  Mean Normalized Gap",
            (
                f"{base_p.mean_normalized_gap:.4f}"
                if base_p.mean_normalized_gap is not None
                else "-"
            ),
            (
                f"{agent_p.mean_normalized_gap:.4f}"
                if agent_p.mean_normalized_gap is not None
                else "-"
            ),
            (
                f"Δ {report.comparison.gap_reduction:+.4f}"
                if report.comparison.gap_reduction is not None
                else "-"
            ),
        ]
    )
    rows.append(
        [
            "  Median Normalized Gap",
            (
                f"{base_p.median_normalized_gap:.4f}"
                if base_p.median_normalized_gap is not None
                else "-"
            ),
            (
                f"{agent_p.median_normalized_gap:.4f}"
                if agent_p.median_normalized_gap is not None
                else "-"
            ),
            "-",
        ]
    )

    # Reliability
    rows.append(["[Reliability]", "", "", ""])
    base_r, agent_r = report.base.reliability, report.agent.reliability
    base_sr = (
        f"{base_r.success_rate * 100:.1f}% ({base_r.valid_runs}/{base_r.total_runs})"
    )
    agent_sr = (
        f"{agent_r.success_rate * 100:.1f}% ({agent_r.valid_runs}/{agent_r.total_runs})"
    )
    rows.append(
        [
            "  Success Rate",
            base_sr,
            agent_sr,
            f"Δ {report.comparison.delta_success_rate * 100:+.1f}%",
        ]
    )
    if base_r.failure_counts or agent_r.failure_counts:
        all_failures = sorted(set(base_r.failure_counts) | set(agent_r.failure_counts))
        for fail_type in all_failures:
            rows.append(
                [
                    f"    - Fail: {fail_type}",
                    str(base_r.failure_counts.get(fail_type, 0)),
                    str(agent_r.failure_counts.get(fail_type, 0)),
                    "-",
                ]
            )

    # Resource Efficiency
    rows.append(["[Resource Efficiency]", "", "", ""])
    base_res, agent_res = report.base.resource, report.agent.resource
    rows.append(
        [
            "  Mean Peak RSS (MB)",
            (
                f"{base_res.mean_peak_rss_mb:.1f}"
                if base_res.mean_peak_rss_mb is not None
                else "-"
            ),
            (
                f"{agent_res.mean_peak_rss_mb:.1f}"
                if agent_res.mean_peak_rss_mb is not None
                else "-"
            ),
            (
                f"{report.comparison.peak_rss_ratio:.2f}x"
                if report.comparison.peak_rss_ratio is not None
                else "-"
            ),
        ]
    )
    rows.append(
        [
            "  Mean CPU Time (s)",
            (
                f"{base_res.mean_cpu_time_sec:.3f}"
                if base_res.mean_cpu_time_sec is not None
                else "-"
            ),
            (
                f"{agent_res.mean_cpu_time_sec:.3f}"
                if agent_res.mean_cpu_time_sec is not None
                else "-"
            ),
            (
                f"{report.comparison.cpu_speedup:.2f}x"
                if report.comparison.cpu_speedup is not None
                else "-"
            ),
        ]
    )
    rows.append(
        [
            "  Budget Utilization",
            (
                f"{base_res.mean_budget_utilization * 100:.1f}%"
                if base_res.mean_budget_utilization is not None
                else "-"
            ),
            (
                f"{agent_res.mean_budget_utilization * 100:.1f}%"
                if agent_res.mean_budget_utilization is not None
                else "-"
            ),
            "-",
        ]
    )

    if tabulate:
        return tabulate(
            rows,
            headers=["Coordinate / Metric", "Base", "Agent", "Comparison / Gain"],
            tablefmt="github",
        )
    # Simple fallback without tabulate
    header = (
        f"{'Coordinate / Metric':<30} | {'Base':<15} | "
        f"{'Agent':<15} | {'Comparison':<15}"
    )
    lines = [header, "-" * 85]
    for r in rows:
        lines.append(f"{r[0]:<30} | {r[1]:<15} | {r[2]:<15} | {r[3]:<15}")
    return "\n".join(lines)
