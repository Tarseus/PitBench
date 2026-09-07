# Seed Robustness changelog

## 0.0.1 — 2026-09-07

Stage: M7 / Frozen, with explicit user approval.
Implementation commit: `cede287d6d7f5151f870efb53654b47f228dcbe6`.
Commit subject: `metric: freeze seed robustness v0.0.1`.

The frozen definition and implementation are specified in [spec.md](spec.md).
The implementation commit includes the metric code, four PyVRP task configurations,
validation scripts, related tests, and the M5/M6 evidence. These final freeze
records were filled in after that commit to reference its actual SHA.

### Frozen protocol

- 30 development seeds and 30 disjoint hidden evaluation seeds per task.
- Type 7 IQR of normalized gap per instance, averaged with equal weights over
  the fixed paired-complete instance set.
- Base/Agent pairing by instance, budget, and seed; change is Agent IQR minus
  Base IQR, with negative values indicating reduced central dispersion.
- Fixed-instance, shared-seed-column paired percentile bootstrap: 5000 resamples,
  bootstrap seed `20260824`, nominal two-sided 99% confidence intervals.
- PyVRP evaluation budgets of 5 and 10 seconds, with 10 seconds as primary.
- Descriptive bias, coverage, and interval-width results; no hard statistical
  acceptance threshold or categorical robustness verdict.

### Evidence and accepted boundaries

[M5 validation](validation-results.md) uses the completed PyVRP v0.12.2, v0.13.0,
and v0.13.4 experiments and their fixed-instance reanalysis. The original 1-second
results remain historical diagnostics. [M6 audit](audit.md) records the review and
resolution of the budget and seed-count documentation inconsistencies.

Recorded validation includes 29 tests for the fixed-instance implementation,
27 point-estimate/interval comparisons with saved reanalysis, and task-catalog
validation plus 12 tests after the budget update. The full task CLI test module
could not be collected in the available image because `typer` was absent; the
catalog was checked directly through its API. No additional solver experiments
were required for M7.

The fourth-version experiment remains supplementary. Nonzero robustness-difference
detection validation is deferred. IQR measures central dispersion, and the finite
reference-seed results do not establish universal sufficiency of 30 seeds or exact
99% coverage. These accepted boundaries do not introduce new completion gates.

After freezing, changes to metric semantics or protocol require a user-approved
version bump under the metric-development contract.
