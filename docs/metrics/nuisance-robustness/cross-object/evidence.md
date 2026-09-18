# Nuisance Robustness cross-object evidence

## M1: HiGHS MIP Seed Robustness candidate

This document records repository and experiment evidence for the MIP Seed
Robustness candidate in the Nuisance Robustness `0.0.2` cycle. It does not
advance the candidate to M2, select an estimator, or claim that MIP Seed
Robustness is already supported.

## Repository evidence

The current HiGHS task is pinned to source commit
`04024d701f79feb8e2f18bc3df0dffc04ef05088` and task version `1.15.1`, as
recorded in [the task configuration](../../../../configs/tasks/highs_v1_15_1.yaml).

The normalized HiGHS driver accepts a solver seed and passes it to the solver
through the `--random_seed` option. The seed is therefore an explicit execution
input in the shared run contract. The task configuration also declares two
solver budgets, 10 and 30 seconds, and one solver thread.

This establishes that the current HiGHS integration has a controllable seed
input. It does not establish how much the seed changes the solver's outcome
distribution, nor that the seed identifiers form a target probability domain.

## Existing MIP nuisance panel

The retained panel is:

```text
private/highs_nuisance/20260909/
```

Its experiment manifest records ten MIP instances, two budgets, thirty explicit
solver seeds for the seed axis, row/column representation transformations, and
control runs. The planned grid contains:

- 600 seed-axis runs;
- 600 representation-axis runs;
- 60 control runs; and
- 1,260 runs in total.

The saved collection summary records 1,260 expected and 1,260 completed runs,
with all runs returning an execution result. The summary was produced as a
generic nuisance exploration and retains `statistics: deferred`; it is not a
formal Seed Robustness report.

The panel uses a single recorded code state and therefore does not provide the
paired Base/Agent observations required by the current Seed Robustness report.
Its seed-axis results establish that a complete MIP seed panel can be collected,
not that the approved Seed Robustness estimand has been computed or validated
for MIP.

## Methodological evidence and boundary

The existing [Seed Robustness evidence](../evidence.md) records the broader
methodological evidence: stochastic-algorithm benchmarking retains repeated
outcomes as an empirical distribution, and published MIP methodology supports
using multiple random seeds when randomized search behavior matters. That
evidence does not select a universal MIP Seed Robustness formula, IQR, seed
count, confidence interval, or cross-instance aggregation.

The current MIP evidence therefore supports only these M1 claims:

1. HiGHS exposes an explicit solver-seed input through the PitBench driver.
2. The repository has a fixed-version, 30-seed MIP experiment with two budgets.
3. The existing panel has complete execution records for its declared grid.
4. The existing panel is exploratory evidence for seed-axis collection, not
   formal Seed Robustness validation.

The following remain open for M2 and later stages:

- the admissible MIP seed domain and target seed distribution;
- whether the HiGHS seed effect is scientifically meaningful for the selected
  task and solver mode;
- the seed-list size and sampling protocol;
- Base/Agent pairing and code-state requirements;
- the estimand, aggregation, confidence interval, and missingness rules; and
- the synthetic and real-panel acceptance criteria under the shared M5 gate.

No MIP Seed Robustness support claim is made until these decisions are recorded
and the resulting protocol passes its own M3-M5 lifecycle gates.
