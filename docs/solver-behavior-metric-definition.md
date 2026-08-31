# Solver behavior metrics induced by outcome geometry

Status: definition 1.0 and finite empirical reference implementation.

## 1. Three spaces and one stochastic map

PitBench separates the instance space, one-run outcome space, and solver behavior
space:

\[
(\mathcal X,\delta_X)
\xrightarrow{K_A(dy\mid x)}
(\mathcal Y,d_Y),
\]

where a solver is a stochastic kernel rather than one point. The implementation
does not identify solver similarity with solver quality. Oracle-relative loss still
answers which solver is better; the definitions below answer how differently two
solvers behave.

No universal outcome metric is declared. Definition 1.0 exposes three independent
projections of `RunObservation`.

### Conditional performance

For a completed, independently valid run with normalized gap `g`,

\[
y_{\mathrm{Perf}}=g,
\qquad
d_Y^{\mathrm{Perf}}(g,g')=|g-g'|.
\]

Performance is conditional on validity. A failed or gap-less run has no performance
outcome; it is not assigned an arbitrary worst gap. Reliability must be reported
beside conditional performance.

### Reliability

For every run,

\[
y_{\mathrm{Rel}}=(\mathrm{status},\mathrm{valid}),
\]

with the discrete metric

\[
d_Y^{\mathrm{Rel}}(y,y')=\mathbf 1[y\ne y'].
\]

This preserves failure categories without inventing an ordering between timeout,
crash, invalid solution, and build failure.

### Resource consumption

Definition 1.0 pins one resource coordinate:

\[
y_{\mathrm{Res}}=\frac{\mathrm{wall\ time}}{\mathrm{declared\ budget}},
\qquad
d_Y^{\mathrm{Res}}(y,y')=|y-y'|.
\]

CPU time, memory, nodes, and iterations are intentionally not mixed into this
coordinate. Each can later receive its own versioned metric or enter an explicitly
declared product metric. PitBench does not silently weight unlike units.

## 2. Empirical stochastic kernels

For solver `A`, instance `x`, and observed solver seeds `r=1,...,R`, define

\[
\widehat K_A(\cdot\mid x)
=\nu_A^x
=\frac1R\sum_{r=1}^R\delta_{y^A_{x,r}}.
\]

`EmpiricalBehaviorKernel` fixes task, population, budget, and thread count. These
conditions are not outcome randomness and may not be mixed in one kernel. Instance
support is retained even when a conditional projector yields no outcome, so failed
runs cannot silently alter the benchmark population.

The reference implementation computes exact finite uniform optimal transport for
equal or unequal sample counts. Given a declared ground metric `d_Y`, it returns

\[
W_{p,d_Y}(\nu_A^x,\nu_B^x).
\]

## 3. Population-conditional solver distance

For a fixed population `P`, define

\[
\boxed{
D_{P,p}(A,B)
=
\left[
\int_{\mathcal X}
W_{p,d_Y}^p\left(K_A(\cdot\mid x),K_B(\cdot\mid x)\right)
P(dx)
\right]^{1/p}.
}
\]

For a finite benchmark, `P` is an explicit set of normalized instance weights. The
default is uniform. `empirical_solver_distance` returns both the aggregate and every
per-instance Wasserstein distance.

Because Wasserstein distance and finite weighted `L_p` aggregation are metrics,
`D_{P,p}` is a pseudometric on solver implementations. It is zero exactly when the
two conditional behavior distributions agree on every positive-weight empirical
instance. On the quotient

\[
A\sim_P B
\iff
K_A(\cdot\mid x)=K_B(\cdot\mid x)
\quad P\text{-almost everywhere},
\]

it is a metric. With one outcome per instance it reduces to the ordinary function
space `L_p` metric.

Conditional performance distance is undefined when either solver has no valid
performance sample at an instance. This is a deliberate result, not missing-data
imputation. The reliability metric remains defined and exposes why performance is
unavailable.

## 4. Stochasticity and solver-induced instance geometry

For one empirical conditional kernel, the reference stochasticity observable is
the pairwise `p`-dispersion

\[
\operatorname{Disp}_p(\nu)
=
\left[
\frac1{R^2}\sum_{r,s}d_Y(y_r,y_s)^p
\right]^{1/p}.
\]

It is zero for deterministic or empirically identical outcomes. It is not a quality
score.

Each solver also induces a behavioral pseudometric on the observed instances:

\[
\rho_A(x,x')
=W_{p,d_Y}\left(K_A(\cdot\mid x),K_A(\cdot\mid x')\right).
\]

`induced_behavior_geometry` computes this finite matrix. Given an instance metric,
`solver_lipschitz_sensitivity` computes

\[
\widehat{\operatorname{Lip}}(A)
=\max_{x\ne x'}\frac{\rho_A(x,x')}{\delta_X(x,x')}
\]

and returns a maximizing witness. If `delta_X` identifies two inputs but their
behavior distributions differ, sensitivity is infinite. Two solvers' response
geometries can be compared through

\[
\max_{x<x'}|\rho_A(x,x')-\rho_B(x,x')|.
\]

## 5. Reference API

The implementation is split between:

- `pitbench.metrics.outcomes`: versioned one-run projections and ground metrics;
- `pitbench.metrics.solver_behavior`: empirical kernels, exact finite Wasserstein,
  solver distance, dispersion, response geometry, and sensitivity.

Typical usage is:

```python
base = empirical_kernel_from_observations(
    base_rows, performance_outcome, solver_id="base"
)
agent = empirical_kernel_from_observations(
    agent_rows, performance_outcome, solver_id="agent"
)
result = empirical_solver_distance(
    base, agent, performance_outcome_distance, p=2
)
```

The same orchestration is repeated independently with `reliability_outcome` and
`resource_outcome`. The API does not produce a universal scalar across the three.

The production evaluator performs this orchestration for every fixed
`(population, budget, threads)` slice and stores it in `EvaluationSummary.behavior`.
Diagnostic equivalence variants are excluded from population support so they cannot
reweight the main solver distance.

## 6. Production six-dimensional evaluation

PitBench keeps the three outcome coordinates independent and evaluates their
response along three frozen input directions:

1. semantic equivalence, using deterministic customer relabeling and matched
   state/seed/budget runs;
2. problem scale, using evaluator-supplied customer count rather than a
   solver-reported proxy;
3. population shift, using paired Base-versus-Agent performance, reliability, and
   resource gains on hash-pinned `judge_id` and `judge_shift` populations.

This produces the complete `equivalence/scale/population ×
performance/reliability/resource` response matrix. Multi-seed dispersion is
reported separately as an intra-kernel stochasticity profile. Main outcome quality
uses only the public-source BKS-anchored `judge_id` instances, so the shift panel
cannot reweight the primary aggregate. The frozen shift population has its own
versioned empirical BKS oracle, enabling population-conditional performance geometry
while retaining its diagnostic role.

The manifest declares a versioned Pareto-style decision policy. It requires the
configured response matrix and CPU/RSS telemetry, accepts an improvement in at
least one performance/resource coordinate, and rejects reliability, quality, or
resource regressions beyond declared tolerances. The generic harness receives only
that evaluator-owned boolean verdict and does not interpret metric payloads.

## 7. Scope of the claim

The implementation gives exact optimal transport between finite seed-empirical
outcome measures. It does not claim that a small seed panel identifies the true
kernel, provide confidence intervals, rank solver quality, or validate a chosen
outcome geometry. Those are statistical and domain-validation layers above the
metric definition.

Sensitivity is computed only on the supplied finite instance support. When
`delta_X` is the exact CVRP definition oracle, its current computational limit also
limits direct sensitivity experiments; a separately validated scalable instance
distance can be passed through the same API.
