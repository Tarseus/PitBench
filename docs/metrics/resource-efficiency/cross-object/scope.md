# Resource Efficiency cross-object scope

## M0: shared resource observation contract

Resource Efficiency measures paired resource use for the same original
optimization input, budget, thread count, solver seed, configuration, and
code-state comparison. It reports resource ratios, not a quality-adjusted score,
performance classification, or deployment cost guarantee.

The current shared report records:

- peak RSS as the primary resource observation;
- CPU time as an auxiliary resource observation;
- the declared measurement scope for each observation;
- Base/Agent pairing by instance, budget, seed, and code state; and
- failures, missing resources, invalid runs, and incompatible scopes without
  converting them into savings.

## Solver and object coverage

The same observation contract applies to the current solver set:

- PyVRP and VROOM routing drivers;
- HiGHS MIP driver;
- Choco exact runner;
- OR-Tools CP-SAT exact runner; and

Native solver processes use the shared resource snapshot contract. External and
JVM runners report the lifetime peak and CPU use of the single native child
process through the same normalized fields. The evaluator computes the resource
report automatically for original observations; no solver-specific metric flag
is required.

## Aggregation and exclusions

Resource ratios use the same eligible Base/Agent seed pairs within each instance.
Ratios are aggregated with an equal-instance geometric mean and reported
separately by instance set and budget. The expected seed count comes from the
task's active seed protocol: the 30-seed protocol for stochastic tasks and the
declared solver-seed count for deterministic or non-Seed-Robustness tasks.

Equivalent-representation runs, operational-reliability cases, and other test
suites are excluded from the ordinary resource report. Their resource records
remain retained in their own artifacts where collected.

Missing or non-positive resource observations, incompatible measurement scopes,
thread mismatches, and failed runs remain explicit exclusions. A missing memory
or CPU value is never interpreted as zero usage.

## Lifecycle boundary

This scope fixes a shared cross-object observation contract. It does not select a
resource threshold, a combined resource-quality score, a hardware-normalized
ranking, or a deployment cost model. Those decisions require later M1-M3
evidence and specification work.
