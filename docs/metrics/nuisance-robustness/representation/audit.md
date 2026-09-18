# Representation Robustness M6 freeze-candidate audit

Date: 2026-09-18.

Metric: Nuisance Robustness / Representation Robustness.
Stage: M6.
Outcome: freeze candidate; reviewed consistency blockers are resolved.

This audit does not freeze the protocol, assign a frozen version, or record a
frozen commit. M7 remains a separate user-approved lifecycle stage.

## Candidate protocol

- Equivalent customer relabeling is the in-scope representation transformation.
- The first protocol uses PyVRP 0.14.0, the ten-instance development panel,
  solver seed `0`, generation seed `20260907`, and 30 distinct non-identity
  relabelings per instance.
- Base and Agent use the same relabelings, budgets, and solver seed.
- The headline statistic is Type 7 IQR of normalized gap within each instance,
  followed by an equal-weight mean over the ten instances.
- The reported change is `Agent - Base`.
- The 5-second and 10-second budgets are reported separately.
- An instance requires all 30 valid independently verified relabeling outcomes
  to produce an IQR. An overall state mean requires all ten instance IQRs;
  missing values are not imputed.
- The protocol has no confidence interval, robustness pass threshold, or claim
  that 30 relabelings are sufficient for every solver or instance.

## M3 and M5 evidence

The seven approved deterministic M3 falsification cases pass:

- Type 7 quantile exactness;
- relabeling-order invariance;
- constant-shift invariance;
- single-tail-outlier behavior;
- equal-weight instance aggregation;
- Base/Agent change direction and antisymmetry; and
- incomplete-data propagation.

Existing representation tests cover transformation legality, including
reproducible non-identity mappings, fixed depot and bijective customer mapping,
synchronized coordinates and demands, mapped feasibility and objective
preservation, failure retention, and the 1200-run grid.

The real PyVRP panel at
`private/representation_robustness/pyvrp_v0_14_0_20260907/results/` contains
1,200 planned and 1,200 completed runs. All 1,200 observations are valid, pass
mapped feasibility, and preserve the mapped objective. The formal report is
available in the `report/` subdirectory and the canonical collection summary
now records `type_7_iqr_equal_instance_mean`.

## Resolved M6 consistency blockers

1. The representation scope had retained an obsolete statement that statistic
   selection was deferred. It now points to the approved M2 Type 7 IQR,
   equal-instance-weight aggregation, and validation artifacts.
2. The canonical collection summary had been generated before the M4 estimator
   existed and still reported `statistics: deferred`. It was regenerated from
   the unchanged raw results and now contains the formal Representation
   Robustness report.
3. Directory-format nuisance results did not propagate
   `observation.normalized_gap` to the metric estimator. The shared report
   loader now preserves that field, with a regression test.

These fixes change no raw solver result, transformation mapping, seed, budget,
or estimator definition.

## Implementation checks

The reviewed implementation and regression checks passed:

```text
tests/unit/pitbench/test_metric_reports.py: 72 passed
tests/unit/pitbench/test_representation_robustness.py: 10 passed, 15 skipped
tests/unit/pitbench/test_evaluation.py -k 'nuisance or representation': 2 passed, 73 deselected
Ruff: passed
git diff --check: passed
```

## M7 handoff

No unresolved scientific or implementation blocker remains in this M6 review.
M7 still requires explicit user approval of the freeze, the `0.0.1` frozen
version record, and the final implementation commit SHA. The registry has not
been updated and no frozen status is recorded by this audit.
