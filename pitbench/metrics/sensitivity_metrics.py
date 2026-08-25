from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, Field

from pitbench.schema.observation import CodeState, RunObservation

SENSITIVITY_REPORT_VERSION = "1.0"


def compute_pairwise_dispersion(values: Sequence[float]) -> float | None:
    """Compute normalized pairwise dispersion: (1 / N^2) * sum_{i, j} |v_i - v_j|."""
    if not values:
        return None
    n = len(values)
    if n == 1:
        return 0.0
    total_diff = sum(abs(a - b) for a in values for b in values)
    return total_diff / (n * n)


def compute_linear_slope(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Compute OLS slope for 1D regression y = slope * x + intercept."""
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x == 0:
        return None
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return cov_xy / var_x


# ---------------------------------------------------------------------------
# 1. Seed Stability (Intra-Kernel Dispersion, Standalone Profile)
# ---------------------------------------------------------------------------


class SeedStabilityProfile(BaseModel):
    """Multi-seed output dispersion for a single state on identical representation."""

    instances_evaluated: int = Field(default=0, ge=0)
    mean_gap_dispersion: float | None = None
    median_gap_dispersion: float | None = None
    mean_runtime_dispersion_sec: float | None = None


class SeedStabilityComparison(BaseModel):
    """Seed stability comparison between Base and Agent."""

    base: SeedStabilityProfile
    agent: SeedStabilityProfile
    delta_gap_dispersion: float | None = None
    delta_runtime_dispersion_sec: float | None = None


# ---------------------------------------------------------------------------
# 2. Representation Stability (Semantic Equivalence Transforms)
# ---------------------------------------------------------------------------


class RepresentationStabilityMetrics(BaseModel):
    """Behavior movement across semantic-preserving instance transforms."""

    has_transforms: bool = False
    pairs_evaluated: int = Field(default=0, ge=0)
    mean_gap_movement: float | None = None
    mean_status_mismatch: float | None = None
    mean_resource_movement: float | None = None
    mean_rss_ratio_movement: float | None = None
    base_gap_movement: float | None = None
    agent_gap_movement: float | None = None
    delta_gap_movement: float | None = None
    base_status_mismatch: float | None = None
    agent_status_mismatch: float | None = None
    delta_status_mismatch: float | None = None
    base_resource_movement: float | None = None
    agent_resource_movement: float | None = None
    delta_resource_movement: float | None = None


# ---------------------------------------------------------------------------
# 3. Problem Scalability (Scale Descriptor vs Problem Mass)
# ---------------------------------------------------------------------------


class ProblemScaleProfile(BaseModel):
    """Scaling behavior over frozen problem scale descriptors s(x)."""

    has_scale_data: bool = False
    scales_evaluated: int = Field(default=0, ge=0)
    runtime_scaling_slope: float | None = None
    gap_scaling_slope: float | None = None
    reliability_scaling_slope: float | None = None
    gap_scaling_slopes_by_budget: dict[float, float] = Field(default_factory=dict)
    reliability_scaling_slopes_by_budget: dict[float, float] = Field(
        default_factory=dict
    )
    runtime_scaling_slopes_by_budget: dict[float, float] = Field(default_factory=dict)


class ProblemScalabilityComparison(BaseModel):
    """Problem scalability comparison across problem scale."""

    base: ProblemScaleProfile
    agent: ProblemScaleProfile
    delta_runtime_slope: float | None = None
    delta_gap_slope: float | None = None
    delta_reliability_slope: float | None = None
    delta_gap_slopes_by_budget: dict[float, float] = Field(default_factory=dict)
    delta_reliability_slopes_by_budget: dict[float, float] = Field(default_factory=dict)
    delta_runtime_slopes_by_budget: dict[float, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# 4. Cross-Population Gain Retention
# ---------------------------------------------------------------------------


class CrossPopulationRetentionMetrics(BaseModel):
    """Gain retention and negative transfer across hidden populations (judge_id vs judge_shift)."""

    has_multi_population: bool = False
    populations_evaluated: list[str] = Field(default_factory=list)
    id_population: str | None = None
    shift_population: str | None = None
    id_gap_reduction: float | None = None
    shift_gap_reduction: float | None = None
    gain_retention: float | None = None
    id_reliability_gain: float | None = None
    shift_reliability_gain: float | None = None
    reliability_gain_retention: float | None = None
    id_resource_gain: float | None = None
    shift_resource_gain: float | None = None
    resource_gain_retention: float | None = None
    delta_performance_gain: float | None = None
    delta_reliability_gain: float | None = None
    delta_resource_gain: float | None = None
    negative_transfer_count: int = Field(default=0, ge=0)
    negative_transfer_fraction: float = Field(default=0.0, ge=0.0, le=1.0)

    # Backward-compatible aliases
    dev_population: str | None = None
    dev_gap_reduction: float | None = None
    eval_gap_reduction: float | None = None
    dev_reliability_gain: float | None = None
    eval_reliability_gain: float | None = None
    dev_resource_gain: float | None = None
    eval_resource_gain: float | None = None


# ---------------------------------------------------------------------------
# 5. Sensitivity Summary Matrix (3 x 3)
# ---------------------------------------------------------------------------


class MatrixRow(BaseModel):
    """One perturbation row in the Sensitivity Summary Matrix."""

    performance: float | None = None
    reliability: float | None = None
    resource: float | None = None


class SensitivitySummaryMatrix(BaseModel):
    """3x3 Sensitivity Summary Matrix (Equiv, Scale, Pop x Perf, Rel, Res).

    Note: The Equivalence row strictly represents transforms (None if absent).
    Scale strictly represents frozen problem mass scaling (None if absent).
    """

    equivalence: MatrixRow = Field(default_factory=MatrixRow)
    scale: MatrixRow = Field(default_factory=MatrixRow)
    population: MatrixRow = Field(default_factory=MatrixRow)


# ---------------------------------------------------------------------------
# 6. Complete Sensitivity Report
# ---------------------------------------------------------------------------


class SensitivityReport(BaseModel):
    """Complete Sensitivity evaluation report."""

    report_version: str = SENSITIVITY_REPORT_VERSION
    seed_stability: SeedStabilityComparison
    representation_stability: RepresentationStabilityMetrics
    problem_scalability: ProblemScalabilityComparison
    cross_population_retention: CrossPopulationRetentionMetrics
    matrix: SensitivitySummaryMatrix


# ---------------------------------------------------------------------------
# Computation Functions
# ---------------------------------------------------------------------------


def _compute_seed_dispersion_for_state(
    observations: Sequence[RunObservation],
) -> SeedStabilityProfile:
    """Group by (pop, instance, budget) and compute seed dispersion."""
    groups: dict[tuple[str, str, float], list[RunObservation]] = defaultdict(list)
    for obs in observations:
        groups[(obs.population, obs.instance_id, obs.budget_sec)].append(obs)

    instance_gap_disps: list[float] = []
    instance_time_disps: list[float] = []

    for group in groups.values():
        valid_gaps = [
            obs.normalized_gap
            for obs in group
            if obs.valid and obs.normalized_gap is not None
        ]
        if len(valid_gaps) >= 2:
            disp = compute_pairwise_dispersion(valid_gaps)
            if disp is not None:
                instance_gap_disps.append(disp)

        valid_times = [
            obs.wall_time_sec
            for obs in group
            if obs.valid and obs.wall_time_sec is not None
        ]
        if len(valid_times) >= 2:
            time_disp = compute_pairwise_dispersion(valid_times)
            if time_disp is not None:
                instance_time_disps.append(time_disp)

    mean_gap_disp = statistics.fmean(instance_gap_disps) if instance_gap_disps else None
    median_gap_disp = (
        statistics.median(instance_gap_disps) if instance_gap_disps else None
    )
    mean_time_disp = (
        statistics.fmean(instance_time_disps) if instance_time_disps else None
    )

    return SeedStabilityProfile(
        instances_evaluated=len(groups),
        mean_gap_dispersion=mean_gap_disp,
        median_gap_dispersion=median_gap_disp,
        mean_runtime_dispersion_sec=mean_time_disp,
    )


def compute_seed_stability(
    base_obs: Sequence[RunObservation], agent_obs: Sequence[RunObservation]
) -> SeedStabilityComparison:
    base_profile = _compute_seed_dispersion_for_state(base_obs)
    agent_profile = _compute_seed_dispersion_for_state(agent_obs)

    delta_gap = None
    if (
        base_profile.mean_gap_dispersion is not None
        and agent_profile.mean_gap_dispersion is not None
    ):
        delta_gap = agent_profile.mean_gap_dispersion - base_profile.mean_gap_dispersion

    delta_time = None
    if (
        base_profile.mean_runtime_dispersion_sec is not None
        and agent_profile.mean_runtime_dispersion_sec is not None
    ):
        delta_time = (
            agent_profile.mean_runtime_dispersion_sec
            - base_profile.mean_runtime_dispersion_sec
        )

    return SeedStabilityComparison(
        base=base_profile,
        agent=agent_profile,
        delta_gap_dispersion=delta_gap,
        delta_runtime_dispersion_sec=delta_time,
    )


def compute_representation_stability(
    observations: Sequence[RunObservation],
    transform_pairs: Sequence[tuple[str, str]] | None = None,
) -> RepresentationStabilityMetrics:
    """Compute representation stability across certified semantic equivalent pairs."""
    if transform_pairs is None:
        transform_pairs = sorted(
            {
                (obs.equivalence_parent_id, obs.instance_id)
                for obs in observations
                if obs.equivalence_parent_id is not None
            }
        )
    if not transform_pairs:
        return RepresentationStabilityMetrics(has_transforms=False)

    obs_by_key = {
        (
            obs.instance_id,
            obs.code_state,
            obs.population,
            obs.solver_seed,
            obs.budget_sec,
        ): obs
        for obs in observations
    }

    gap_diffs: dict[CodeState, list[float]] = defaultdict(list)
    status_diffs: dict[CodeState, list[float]] = defaultdict(list)
    resource_diffs: dict[CodeState, list[float]] = defaultdict(list)
    rss_diffs: dict[CodeState, list[float]] = defaultdict(list)

    for orig_id, tf_id in transform_pairs:
        transformed = [obs for obs in observations if obs.instance_id == tf_id]
        for tf_obs in transformed:
            orig_obs = obs_by_key.get(
                (
                    orig_id,
                    tf_obs.code_state,
                    tf_obs.population,
                    tf_obs.solver_seed,
                    tf_obs.budget_sec,
                )
            )
            if orig_obs is None:
                continue
            if (
                orig_obs.valid
                and tf_obs.valid
                and orig_obs.normalized_gap is not None
                and tf_obs.normalized_gap is not None
            ):
                gap_diffs[tf_obs.code_state].append(
                    abs(orig_obs.normalized_gap - tf_obs.normalized_gap)
                )
            status_diffs[tf_obs.code_state].append(
                0.0
                if (orig_obs.status, orig_obs.valid) == (tf_obs.status, tf_obs.valid)
                else 1.0
            )
            if orig_obs.wall_time_sec is not None and tf_obs.wall_time_sec is not None:
                resource_diffs[tf_obs.code_state].append(
                    abs(
                        orig_obs.wall_time_sec / orig_obs.budget_sec
                        - tf_obs.wall_time_sec / tf_obs.budget_sec
                    )
                )
            if (
                orig_obs.peak_rss_bytes is not None
                and tf_obs.peak_rss_bytes is not None
                and orig_obs.peak_rss_bytes > 0
                and tf_obs.peak_rss_bytes > 0
            ):
                rss_diffs[tf_obs.code_state].append(
                    abs(
                        math.log(orig_obs.peak_rss_bytes)
                        - math.log(tf_obs.peak_rss_bytes)
                    )
                )

    all_status = [value for values in status_diffs.values() for value in values]
    if not all_status:
        return RepresentationStabilityMetrics(has_transforms=False)

    def _mean(values: dict[CodeState, list[float]], state: CodeState) -> float | None:
        state_values = values[state]
        return statistics.fmean(state_values) if state_values else None

    def _delta(base: float | None, agent: float | None) -> float | None:
        if base is None or agent is None:
            return None
        return agent - base

    base_gap = _mean(gap_diffs, CodeState.BASE)
    agent_gap = _mean(gap_diffs, CodeState.AGENT)
    base_status = _mean(status_diffs, CodeState.BASE)
    agent_status = _mean(status_diffs, CodeState.AGENT)
    base_resource = _mean(resource_diffs, CodeState.BASE)
    agent_resource = _mean(resource_diffs, CodeState.AGENT)
    all_gaps = [value for values in gap_diffs.values() for value in values]
    all_resources = [value for values in resource_diffs.values() for value in values]
    all_rss = [value for values in rss_diffs.values() for value in values]

    return RepresentationStabilityMetrics(
        has_transforms=True,
        pairs_evaluated=len(all_status),
        mean_gap_movement=statistics.fmean(all_gaps) if all_gaps else None,
        mean_status_mismatch=statistics.fmean(all_status),
        mean_resource_movement=(
            statistics.fmean(all_resources) if all_resources else None
        ),
        mean_rss_ratio_movement=statistics.fmean(all_rss) if all_rss else None,
        base_gap_movement=base_gap,
        agent_gap_movement=agent_gap,
        delta_gap_movement=_delta(base_gap, agent_gap),
        base_status_mismatch=base_status,
        agent_status_mismatch=agent_status,
        delta_status_mismatch=_delta(base_status, agent_status),
        base_resource_movement=base_resource,
        agent_resource_movement=agent_resource,
        delta_resource_movement=_delta(base_resource, agent_resource),
    )


def compute_problem_scalability(
    base_obs: Sequence[RunObservation],
    agent_obs: Sequence[RunObservation],
    scale_descriptors: Mapping[str, float] | None = None,
) -> ProblemScalabilityComparison:
    """Compute scalability strictly against frozen problem scale descriptors s(x)."""
    if not scale_descriptors:
        # Prefer an evaluator-frozen problem descriptor carried by every run.
        scale_map: dict[str, float] = {}
        for obs in (*base_obs, *agent_obs):
            if obs.problem_scale is not None:
                scale_map[obs.instance_id] = obs.problem_scale
            elif obs.model_variables is not None and obs.model_variables > 0:
                scale_map[obs.instance_id] = float(obs.model_variables)
        if scale_map:
            scale_descriptors = scale_map
        else:
            return ProblemScalabilityComparison(
                base=ProblemScaleProfile(has_scale_data=False),
                agent=ProblemScaleProfile(has_scale_data=False),
            )

    def _profile(obs_list: Sequence[RunObservation]) -> ProblemScaleProfile:
        scale_obs = [o for o in obs_list if o.instance_id in scale_descriptors]
        if not scale_obs:
            return ProblemScaleProfile(has_scale_data=False)

        scales = [
            scale_descriptors[o.instance_id]
            for o in scale_obs
            if scale_descriptors[o.instance_id] > 0
        ]
        unique_scales = len(set(scales))
        if unique_scales < 2:
            return ProblemScaleProfile(
                has_scale_data=False,
                scales_evaluated=unique_scales,
            )

        budgets = sorted({o.budget_sec for o in scale_obs if o.budget_sec is not None})
        if not budgets:
            budgets = [1.0]

        gap_slopes_by_budget: dict[float, float] = {}
        reliability_slopes_by_budget: dict[float, float] = {}
        runtime_slopes_by_budget: dict[float, float] = {}

        for b in budgets:
            b_runs = [o for o in scale_obs if o.budget_sec == b]
            if not b_runs:
                continue
            runs_by_inst: dict[str, list[RunObservation]] = defaultdict(list)
            for o in b_runs:
                runs_by_inst[o.instance_id].append(o)

            log_scales_b: list[float] = []
            log_runtimes_b: list[float] = []
            gap_log_scales_b: list[float] = []
            mean_gaps_b: list[float] = []
            reliabilities_b: list[float] = []

            for inst_id, runs in runs_by_inst.items():
                scale = scale_descriptors[inst_id]
                if scale <= 0:
                    continue
                log_s = math.log(scale)
                log_scales_b.append(log_s)

                # Reliability (fraction of valid runs for this instance and budget)
                valid_runs = [r for r in runs if r.valid]
                reliabilities_b.append(len(valid_runs) / len(runs))

                # Runtime (mean wall time across runs, or budget fallback)
                budget_fallback = max(1e-6, b)
                wall_times = [
                    (
                        r.wall_time_sec
                        if r.wall_time_sec is not None and r.wall_time_sec > 0
                        else budget_fallback
                    )
                    for r in runs
                ]
                mean_runtime = statistics.mean(wall_times)
                log_runtimes_b.append(math.log(mean_runtime))

                # Gap (mean normalized gap over valid runs with gap data)
                valid_gaps = [
                    r.normalized_gap for r in valid_runs if r.normalized_gap is not None
                ]
                if valid_gaps:
                    gap_log_scales_b.append(log_s)
                    mean_gaps_b.append(statistics.mean(valid_gaps))

            r_slope = compute_linear_slope(log_scales_b, log_runtimes_b)
            if r_slope is not None:
                runtime_slopes_by_budget[b] = r_slope

            rel_slope = compute_linear_slope(log_scales_b, reliabilities_b)
            if rel_slope is not None:
                reliability_slopes_by_budget[b] = rel_slope

            g_slope = compute_linear_slope(gap_log_scales_b, mean_gaps_b)
            if g_slope is not None:
                gap_slopes_by_budget[b] = g_slope

        has_estimable_slice = bool(
            gap_slopes_by_budget
            or reliability_slopes_by_budget
            or runtime_slopes_by_budget
        )

        return ProblemScaleProfile(
            has_scale_data=has_estimable_slice,
            scales_evaluated=unique_scales,
            gap_scaling_slopes_by_budget=gap_slopes_by_budget,
            reliability_scaling_slopes_by_budget=reliability_slopes_by_budget,
            runtime_scaling_slopes_by_budget=runtime_slopes_by_budget,
        )

    base_p = _profile(base_obs)
    agent_p = _profile(agent_obs)

    all_budgets = sorted(
        set(base_p.gap_scaling_slopes_by_budget)
        | set(agent_p.gap_scaling_slopes_by_budget)
        | set(base_p.reliability_scaling_slopes_by_budget)
        | set(agent_p.reliability_scaling_slopes_by_budget)
        | set(base_p.runtime_scaling_slopes_by_budget)
        | set(agent_p.runtime_scaling_slopes_by_budget)
    )
    delta_gap_slopes_by_budget: dict[float, float] = {}
    delta_rel_slopes_by_budget: dict[float, float] = {}
    delta_runtime_slopes_by_budget: dict[float, float] = {}

    for b in all_budgets:
        base_g = base_p.gap_scaling_slopes_by_budget.get(b)
        agent_g = agent_p.gap_scaling_slopes_by_budget.get(b)
        if base_g is not None and agent_g is not None:
            delta_gap_slopes_by_budget[b] = agent_g - base_g

        base_r = base_p.reliability_scaling_slopes_by_budget.get(b)
        agent_r = agent_p.reliability_scaling_slopes_by_budget.get(b)
        if base_r is not None and agent_r is not None:
            delta_rel_slopes_by_budget[b] = agent_r - base_r

        base_t = base_p.runtime_scaling_slopes_by_budget.get(b)
        agent_t = agent_p.runtime_scaling_slopes_by_budget.get(b)
        if base_t is not None and agent_t is not None:
            delta_runtime_slopes_by_budget[b] = agent_t - base_t

    def _primary_common_slopes(
        base_slopes: Mapping[float, float],
        agent_slopes: Mapping[float, float],
    ) -> tuple[float | None, float | None, float | None]:
        common_budgets = set(base_slopes) & set(agent_slopes)
        if not common_budgets:
            return None, None, None
        primary_budget = max(common_budgets)
        base_slope = base_slopes[primary_budget]
        agent_slope = agent_slopes[primary_budget]
        return base_slope, agent_slope, agent_slope - base_slope

    base_runtime, agent_runtime, delta_runtime_slope = _primary_common_slopes(
        base_p.runtime_scaling_slopes_by_budget,
        agent_p.runtime_scaling_slopes_by_budget,
    )
    base_gap, agent_gap, delta_gap_slope = _primary_common_slopes(
        base_p.gap_scaling_slopes_by_budget,
        agent_p.gap_scaling_slopes_by_budget,
    )
    base_reliability, agent_reliability, delta_reliability_slope = (
        _primary_common_slopes(
            base_p.reliability_scaling_slopes_by_budget,
            agent_p.reliability_scaling_slopes_by_budget,
        )
    )
    base_p = base_p.model_copy(
        update={
            "runtime_scaling_slope": base_runtime,
            "gap_scaling_slope": base_gap,
            "reliability_scaling_slope": base_reliability,
        }
    )
    agent_p = agent_p.model_copy(
        update={
            "runtime_scaling_slope": agent_runtime,
            "gap_scaling_slope": agent_gap,
            "reliability_scaling_slope": agent_reliability,
        }
    )

    return ProblemScalabilityComparison(
        base=base_p,
        agent=agent_p,
        delta_runtime_slope=delta_runtime_slope,
        delta_gap_slope=delta_gap_slope,
        delta_reliability_slope=delta_reliability_slope,
        delta_gap_slopes_by_budget=delta_gap_slopes_by_budget,
        delta_reliability_slopes_by_budget=delta_rel_slopes_by_budget,
        delta_runtime_slopes_by_budget=delta_runtime_slopes_by_budget,
    )


def compute_cross_population_retention(
    base_obs: Sequence[RunObservation],
    agent_obs: Sequence[RunObservation],
) -> CrossPopulationRetentionMetrics:
    """Compute gain retention across hidden populations (judge_id vs judge_shift)."""
    # Strictly exclude agent_dev (training / development set) from generalization
    valid_base = [
        o
        for o in base_obs
        if o.population_kind != "agent_dev" and o.population != "agent_dev"
    ]
    valid_agent = [
        o
        for o in agent_obs
        if o.population_kind != "agent_dev" and o.population != "agent_dev"
    ]

    populations = sorted({o.population for o in (*valid_base, *valid_agent)})
    if len(populations) < 2:
        return CrossPopulationRetentionMetrics(
            has_multi_population=False, populations_evaluated=populations
        )

    id_populations = sorted(
        {
            obs.population
            for obs in (*valid_base, *valid_agent)
            if obs.population_kind == "judge_id"
        }
    )
    shift_populations = sorted(
        {
            obs.population
            for obs in (*valid_base, *valid_agent)
            if obs.population_kind == "judge_shift"
        }
    )
    if (
        len(id_populations) != 1
        or len(shift_populations) != 1
        or id_populations[0] == shift_populations[0]
    ):
        return CrossPopulationRetentionMetrics(
            has_multi_population=False, populations_evaluated=populations
        )
    id_name = id_populations[0]
    shift_name = shift_populations[0]

    def _paired_gains(pop: str) -> tuple[float | None, float | None, float | None]:
        base_by_key = {
            (o.instance_id, o.solver_seed, o.budget_sec): o
            for o in valid_base
            if o.population == pop and o.equivalence_parent_id is None
        }
        agent_by_key = {
            (o.instance_id, o.solver_seed, o.budget_sec): o
            for o in valid_agent
            if o.population == pop and o.equivalence_parent_id is None
        }
        performance: list[float] = []
        reliability: list[float] = []
        resource: list[float] = []
        for key in set(base_by_key) & set(agent_by_key):
            base = base_by_key[key]
            agent = agent_by_key[key]
            reliability.append(float(agent.valid) - float(base.valid))
            if base.wall_time_sec is not None and agent.wall_time_sec is not None:
                resource.append(
                    base.wall_time_sec / base.budget_sec
                    - agent.wall_time_sec / agent.budget_sec
                )
            if base.valid and agent.valid:
                if (
                    base.objective is not None
                    and agent.objective is not None
                    and abs(base.objective) > 1e-12
                ):
                    performance.append(
                        (base.objective - agent.objective) / abs(base.objective)
                    )
                elif (
                    base.normalized_gap is not None and agent.normalized_gap is not None
                ):
                    performance.append(base.normalized_gap - agent.normalized_gap)
        return (
            statistics.fmean(performance) if performance else None,
            statistics.fmean(reliability) if reliability else None,
            statistics.fmean(resource) if resource else None,
        )

    id_reduction, id_reliability, id_resource = _paired_gains(id_name)
    shift_reduction, shift_reliability, shift_resource = _paired_gains(shift_name)

    gain_retention = None
    if (
        id_reduction is not None
        and shift_reduction is not None
        and abs(id_reduction) > 1e-9
    ):
        gain_retention = shift_reduction / id_reduction

    reliability_retention = None
    if (
        id_reliability is not None
        and shift_reliability is not None
        and abs(id_reliability) > 1e-9
    ):
        reliability_retention = shift_reliability / id_reliability

    resource_retention = None
    if (
        id_resource is not None
        and shift_resource is not None
        and abs(id_resource) > 1e-9
    ):
        resource_retention = shift_resource / id_resource

    def _difference(reference: float | None, shifted: float | None) -> float | None:
        if reference is None or shifted is None:
            return None
        return shifted - reference

    # Negative transfer instances on the shifted population
    base_by_key = {
        (o.instance_id, o.solver_seed, o.budget_sec): o
        for o in valid_base
        if o.population == shift_name and o.equivalence_parent_id is None
    }
    agent_by_key = {
        (o.instance_id, o.solver_seed, o.budget_sec): o
        for o in valid_agent
        if o.population == shift_name and o.equivalence_parent_id is None
    }
    paired_performance: list[float] = []
    for key in set(base_by_key) & set(agent_by_key):
        base, agent = base_by_key[key], agent_by_key[key]
        if not (base.valid and agent.valid):
            continue
        if (
            base.objective is not None
            and agent.objective is not None
            and abs(base.objective) > 1e-12
        ):
            paired_performance.append(
                (base.objective - agent.objective) / abs(base.objective)
            )
        elif base.normalized_gap is not None and agent.normalized_gap is not None:
            paired_performance.append(base.normalized_gap - agent.normalized_gap)
    neg_count = sum(value < -1e-9 for value in paired_performance)
    neg_fraction = neg_count / len(paired_performance) if paired_performance else 0.0

    return CrossPopulationRetentionMetrics(
        has_multi_population=True,
        populations_evaluated=[id_name, shift_name],
        id_population=id_name,
        shift_population=shift_name,
        id_gap_reduction=id_reduction,
        shift_gap_reduction=shift_reduction,
        gain_retention=gain_retention,
        id_reliability_gain=id_reliability,
        shift_reliability_gain=shift_reliability,
        reliability_gain_retention=reliability_retention,
        id_resource_gain=id_resource,
        shift_resource_gain=shift_resource,
        resource_gain_retention=resource_retention,
        delta_performance_gain=_difference(id_reduction, shift_reduction),
        delta_reliability_gain=_difference(id_reliability, shift_reliability),
        delta_resource_gain=_difference(id_resource, shift_resource),
        negative_transfer_count=neg_count,
        negative_transfer_fraction=neg_fraction,
        # Backward-compatible aliases
        dev_population=id_name,
        dev_gap_reduction=id_reduction,
        eval_gap_reduction=shift_reduction,
        dev_reliability_gain=id_reliability,
        eval_reliability_gain=shift_reliability,
        dev_resource_gain=id_resource,
        eval_resource_gain=shift_resource,
    )


def compute_sensitivity_report(
    observations: Sequence[RunObservation],
    scale_descriptors: Mapping[str, float] | None = None,
    transform_pairs: Sequence[tuple[str, str]] | None = None,
) -> SensitivityReport:
    """Compute complete Sensitivity evaluation report and SensitivitySummaryMatrix."""
    base_obs = [o for o in observations if o.code_state == CodeState.BASE]
    agent_obs = [o for o in observations if o.code_state == CodeState.AGENT]

    seed_stability = compute_seed_stability(base_obs, agent_obs)
    representation_stability = compute_representation_stability(
        observations, transform_pairs
    )
    primary_base = [
        obs
        for obs in base_obs
        if obs.population_kind == "judge_id" and obs.equivalence_parent_id is None
    ]
    primary_agent = [
        obs
        for obs in agent_obs
        if obs.population_kind == "judge_id" and obs.equivalence_parent_id is None
    ]
    scalability = compute_problem_scalability(
        primary_base or base_obs,
        primary_agent or agent_obs,
        scale_descriptors,
    )
    retention = compute_cross_population_retention(base_obs, agent_obs)

    # Construct 3x3 SensitivitySummaryMatrix
    # Row 1: Equivalence (populated ONLY from Representation Stability, None if absent)
    if representation_stability.has_transforms:
        equiv_row = MatrixRow(
            performance=representation_stability.delta_gap_movement,
            reliability=representation_stability.delta_status_mismatch,
            resource=representation_stability.delta_resource_movement,
        )
    else:
        equiv_row = MatrixRow(performance=None, reliability=None, resource=None)

    # Row 2: Scale (populated ONLY from Problem Scalability, None if absent)
    if scalability.base.has_scale_data and scalability.agent.has_scale_data:
        scale_row = MatrixRow(
            performance=scalability.delta_gap_slope,
            reliability=scalability.delta_reliability_slope,
            resource=scalability.delta_runtime_slope,
        )
    else:
        scale_row = MatrixRow(performance=None, reliability=None, resource=None)

    # Row 3: Population (Cross-population shift)
    if retention.has_multi_population:
        pop_row = MatrixRow(
            performance=retention.delta_performance_gain,
            reliability=retention.delta_reliability_gain,
            resource=retention.delta_resource_gain,
        )
    else:
        pop_row = MatrixRow(performance=None, reliability=None, resource=None)

    matrix = SensitivitySummaryMatrix(
        equivalence=equiv_row,
        scale=scale_row,
        population=pop_row,
    )

    return SensitivityReport(
        seed_stability=seed_stability,
        representation_stability=representation_stability,
        problem_scalability=scalability,
        cross_population_retention=retention,
        matrix=matrix,
    )


def format_sensitivity_report_table(report: SensitivityReport) -> str:
    """Format SensitivityReport and matrix into readable GitHub table."""
    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None  # type: ignore

    rows: list[list[str]] = []

    # Section: Seed Stability (Standalone Profile)
    rows.append(["[1. Seed Stability (Intra-Kernel)]", "", "", ""])
    base_s, agent_s = report.seed_stability.base, report.seed_stability.agent
    rows.append(
        [
            "  Mean Gap Dispersion (1/R^2 sum |g_r-g_s|)",
            f"{base_s.mean_gap_dispersion:.4f}"
            if base_s.mean_gap_dispersion is not None
            else "-",
            f"{agent_s.mean_gap_dispersion:.4f}"
            if agent_s.mean_gap_dispersion is not None
            else "-",
            (
                f"Δ {report.seed_stability.delta_gap_dispersion:+.4f}"
                if report.seed_stability.delta_gap_dispersion is not None
                else "-"
            ),
        ]
    )
    rows.append(
        [
            "  Mean Runtime Dispersion (s)",
            (
                f"{base_s.mean_runtime_dispersion_sec:.3f}"
                if base_s.mean_runtime_dispersion_sec is not None
                else "-"
            ),
            (
                f"{agent_s.mean_runtime_dispersion_sec:.3f}"
                if agent_s.mean_runtime_dispersion_sec is not None
                else "-"
            ),
            (
                f"Δ {report.seed_stability.delta_runtime_dispersion_sec:+.3f}s"
                if report.seed_stability.delta_runtime_dispersion_sec is not None
                else "-"
            ),
        ]
    )

    # Section: Representation Stability (Equivalence Orbit)
    rows.append(["[2. Representation Stability (Equiv Orbit)]", "", "", ""])
    if report.representation_stability.has_transforms:
        rs = report.representation_stability
        rows.append(
            [
                f"  Gap Movement across transforms [N={rs.pairs_evaluated}]",
                f"{rs.base_gap_movement:.4f}"
                if rs.base_gap_movement is not None
                else "-",
                f"{rs.agent_gap_movement:.4f}"
                if rs.agent_gap_movement is not None
                else "-",
                f"Δ {rs.delta_gap_movement:+.4f}"
                if rs.delta_gap_movement is not None
                else "-",
            ]
        )
        rows.extend(
            [
                [
                    "  Status/validity mismatch",
                    f"{rs.base_status_mismatch:.4f}"
                    if rs.base_status_mismatch is not None
                    else "-",
                    f"{rs.agent_status_mismatch:.4f}"
                    if rs.agent_status_mismatch is not None
                    else "-",
                    f"Δ {rs.delta_status_mismatch:+.4f}"
                    if rs.delta_status_mismatch is not None
                    else "-",
                ],
                [
                    "  Wall-time fraction movement",
                    f"{rs.base_resource_movement:.4f}"
                    if rs.base_resource_movement is not None
                    else "-",
                    f"{rs.agent_resource_movement:.4f}"
                    if rs.agent_resource_movement is not None
                    else "-",
                    f"Δ {rs.delta_resource_movement:+.4f}"
                    if rs.delta_resource_movement is not None
                    else "-",
                ],
            ]
        )
    else:
        rows.append(
            [
                "  Semantic Equivalence Transforms",
                "Not Tested",
                "Not Tested",
                "None (No transform panel)",
            ]
        )

    # Section: Problem Scalability
    rows.append(["[3. Problem Scalability (Scale s(x))]", "", "", ""])
    if report.problem_scalability.base.has_scale_data:
        base_sc = report.problem_scalability.base
        agent_sc = report.problem_scalability.agent

        def _append_budget_slopes(
            label: str,
            derivative: str,
            base_slopes: Mapping[float, float],
            agent_slopes: Mapping[float, float],
            delta_slopes: Mapping[float, float],
        ) -> None:
            budgets = sorted(set(base_slopes) | set(agent_slopes))
            if not budgets:
                rows.append([f"  {label} ({derivative})", "-", "-", "-"])
                return
            for budget in budgets:
                base_slope = base_slopes.get(budget)
                agent_slope = agent_slopes.get(budget)
                delta_slope = delta_slopes.get(budget)
                rows.append(
                    [
                        f"  {label} @ {budget:.1f}s ({derivative})",
                        f"{base_slope:.3f}" if base_slope is not None else "-",
                        f"{agent_slope:.3f}" if agent_slope is not None else "-",
                        f"Δ {delta_slope:+.3f}" if delta_slope is not None else "-",
                    ]
                )

        _append_budget_slopes(
            "Runtime Scaling Slope",
            "d log T / d log s",
            base_sc.runtime_scaling_slopes_by_budget,
            agent_sc.runtime_scaling_slopes_by_budget,
            report.problem_scalability.delta_runtime_slopes_by_budget,
        )
        _append_budget_slopes(
            "Gap Scaling Slope",
            "d gap / d log s",
            base_sc.gap_scaling_slopes_by_budget,
            agent_sc.gap_scaling_slopes_by_budget,
            report.problem_scalability.delta_gap_slopes_by_budget,
        )
        _append_budget_slopes(
            "Reliability Scaling Slope",
            "d success / d log s",
            base_sc.reliability_scaling_slopes_by_budget,
            agent_sc.reliability_scaling_slopes_by_budget,
            report.problem_scalability.delta_reliability_slopes_by_budget,
        )
    else:
        has_multiple_scales = (
            report.problem_scalability.base.scales_evaluated >= 2
            or report.problem_scalability.agent.scales_evaluated >= 2
        )
        if has_multiple_scales:
            rows.append(
                [
                    "  Problem Scale Scaling",
                    "Not Estimable",
                    "Not Estimable",
                    "None (No fixed-budget regression)",
                ]
            )
        else:
            rows.append(
                [
                    "  Problem Scale Scaling",
                    "No Descriptor",
                    "No Descriptor",
                    "None (Scale descriptor required)",
                ]
            )

    # Section: Cross-Population Generalization
    rows.append(["[4. Cross-Population Generalization (judge_id -> judge_shift)]", "", "", ""])
    if report.cross_population_retention.has_multi_population:
        cpr = report.cross_population_retention
        id_pop = cpr.id_population or cpr.dev_population or "judge_id"
        shift_pop = cpr.shift_population or "judge_shift"
        id_gap = (
            cpr.id_gap_reduction
            if cpr.id_gap_reduction is not None
            else cpr.dev_gap_reduction
        )
        shift_gap = (
            cpr.shift_gap_reduction
            if cpr.shift_gap_reduction is not None
            else cpr.eval_gap_reduction
        )
        id_rel = (
            cpr.id_reliability_gain
            if cpr.id_reliability_gain is not None
            else cpr.dev_reliability_gain
        )
        shift_rel = (
            cpr.shift_reliability_gain
            if cpr.shift_reliability_gain is not None
            else cpr.eval_reliability_gain
        )
        id_res = (
            cpr.id_resource_gain
            if cpr.id_resource_gain is not None
            else cpr.dev_resource_gain
        )
        shift_res = (
            cpr.shift_resource_gain
            if cpr.shift_resource_gain is not None
            else cpr.eval_resource_gain
        )
        rows.append(
            [
                f"  In-Distribution Performance Gain ({id_pop})",
                "-",
                "-",
                f"{id_gap:.4f}"
                if id_gap is not None
                else "-",
            ]
        )
        rows.append(
            [
                f"  Distribution Shift Performance Gain ({shift_pop})",
                "-",
                "-",
                f"{shift_gap:.4f}"
                if shift_gap is not None
                else "-",
            ]
        )
        rows.append(
            [
                "  Generalization Gain Retention (Shift / ID)",
                "-",
                "-",
                f"{cpr.gain_retention * 100:.1f}%"
                if cpr.gain_retention is not None
                else "-",
            ]
        )
        for label, reference, shifted, delta in (
            (
                "  Reliability Gain (Agent - Base)",
                id_rel,
                shift_rel,
                cpr.delta_reliability_gain,
            ),
            (
                "  Resource Gain (Base - Agent)",
                id_res,
                shift_res,
                cpr.delta_resource_gain,
            ),
        ):
            rows.append(
                [
                    label,
                    f"{reference:.4f}" if reference is not None else "-",
                    f"{shifted:.4f}" if shifted is not None else "-",
                    f"Δ {delta:+.4f}" if delta is not None else "-",
                ]
            )
        neg_msg = (
            f"{cpr.negative_transfer_fraction * 100:.1f}% "
            f"({cpr.negative_transfer_count} cases)"
        )
        rows.append(
            [
                f"  Negative Transfer Fraction ({shift_pop})",
                "-",
                "-",
                neg_msg,
            ]
        )
    else:
        rows.append(
            [
                "  Population Generalization",
                "Single Pop",
                "Single Pop",
                "None (judge_id + judge_shift required)",
            ]
        )

    # Section: 3x3 Sensitivity Summary Matrix
    rows.append(
        [
            "[5. Sensitivity Summary Matrix (3x3)]",
            "Performance",
            "Reliability",
            "Resource",
        ]
    )
    m = report.matrix

    def _val(v: float | None) -> str:
        return f"{v:.4f}" if v is not None else "None"

    rows.append(
        [
            "  Row 1: Equivalence (Representation Shift)",
            _val(m.equivalence.performance),
            _val(m.equivalence.reliability),
            _val(m.equivalence.resource),
        ]
    )
    rows.append(
        [
            "  Row 2: Scale (Problem Mass s(x))",
            _val(m.scale.performance),
            _val(m.scale.reliability),
            _val(m.scale.resource),
        ]
    )
    rows.append(
        [
            "  Row 3: Population (Cross-Population)",
            _val(m.population.performance),
            _val(m.population.reliability),
            _val(m.population.resource),
        ]
    )

    if tabulate:
        return tabulate(
            rows,
            headers=[
                "Sensitivity Dimension / Matrix",
                "Base",
                "Agent",
                "Comparison / Matrix Value",
            ],
            tablefmt="github",
        )
    lines = [
        f"{'Sensitivity Dimension':<38} | {'Base':<12} | "
        f"{'Agent':<12} | {'Comparison':<20}"
    ]
    lines.append("-" * 90)
    for r in rows:
        lines.append(f"{r[0]:<38} | {r[1]:<12} | {r[2]:<12} | {r[3]:<20}")
    return "\n".join(lines)
