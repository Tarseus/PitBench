# Seed Robustness M5 validation results

M5 closed with explicit user approval on 2026-09-07. The completed PyVRP v0.12.2,
v0.13.0, and v0.13.4 experiments and their fixed-instance reanalysis are the main
evidence for this cycle. PyVRP v0.14.0 is a supplementary experiment and is not a
condition for M5 closure. This decision does not advance to M6 or M7.

## Current decision

The user retained 30 seeds and Type 7 IQR. Instance IQRs are averaged with equal
weights over the fixed paired-complete instance set. The paired percentile
bootstrap resamples shared seed columns only, with 5000 replicates, seed
`20260824`, and nominal 99% intervals. There is no hard acceptance threshold.
Bias, empirical coverage, and interval width remain descriptive diagnostics.
Nonzero robustness-difference detection validation is deferred for this cycle.

The user accepted the existing descriptive results as sufficient to close M5,
without requiring the fourth version to finish. Closure records that decision;
it does not assert universal sufficiency of 30 seeds or calibrated 99% coverage.

## Existing real-solver batch

Batch: `private/seed_robustness/four_versions_20260905_local`.
Each version uses the same 10 fixed instances, 700 reference seeds, a disjoint
pool of 300 test seeds, and 1000 test lists containing 30 seeds each. The
Base/Agent comparison uses an empty patch. Each version has 60000 solver runs
across the original 1, 5, and 10 second budgets. The current discussion focuses
on 5 and 10 seconds; the existing 1-second results remain available as diagnostics.

The first three versions completed all 60000 runs with valid observations.
Their fixed-instance reanalysis reused the saved observations and test lists;
no solver runs were repeated for this reanalysis.

## Fixed-instance results

Coverage is the fraction of the 1000 intervals containing the corresponding
700-seed reference estimate. Width is the mean upper-minus-lower endpoint, in
normalized-gap units. Change means Agent IQR minus Base IQR.

| Version | Budget (s) | Base coverage | Agent coverage | Change coverage | Base width | Agent width | Change width |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pyvrp_v0_12_2 | 1 | 99.2% | 99.2% | 100.0% | 0.0023306 | 0.0023348 | 0.0010012 |
| pyvrp_v0_12_2 | 5 | 99.3% | 99.3% | 99.9% | 0.0016578 | 0.0016407 | 0.0006095 |
| pyvrp_v0_12_2 | 10 | 99.9% | 100.0% | 100.0% | 0.0016291 | 0.0016222 | 0.0007018 |
| pyvrp_v0_13_0 | 1 | 97.5% | 97.8% | 90.7% | 0.0041309 | 0.0041332 | 0.0006728 |
| pyvrp_v0_13_0 | 5 | 98.1% | 97.5% | 100.0% | 0.0022354 | 0.0022105 | 0.0006849 |
| pyvrp_v0_13_0 | 10 | 99.5% | 99.5% | 100.0% | 0.0014763 | 0.0014869 | 0.0004897 |
| pyvrp_v0_13_4 | 1 | 99.9% | 99.9% | 98.2% | 0.0047827 | 0.0047731 | 0.0005786 |
| pyvrp_v0_13_4 | 5 | 99.2% | 99.3% | 100.0% | 0.0021572 | 0.0021512 | 0.0006030 |
| pyvrp_v0_13_4 | 10 | 98.0% | 97.6% | 100.0% | 0.0014332 | 0.0014141 | 0.0005110 |

For Base and Agent at 5 and 10 seconds, mean interval widths decreased by
50.72% to 66.04% compared with the original crossed bootstrap. Point estimates
and reference estimates match the original summaries to numerical precision.
The original crossed results remain preserved beside the new outputs.

Per-version artifacts:

- pyvrp_v0_12_2: [summary](../../../private/seed_robustness/four_versions_20260905_local/pyvrp_v0_12_2/fixed_instance_intervals/validation_summary.json), [all 1000 interval sets](../../../private/seed_robustness/four_versions_20260905_local/pyvrp_v0_12_2/fixed_instance_intervals/intervals.json).
- pyvrp_v0_13_0: [summary](../../../private/seed_robustness/four_versions_20260905_local/pyvrp_v0_13_0/fixed_instance_intervals/validation_summary.json), [all 1000 interval sets](../../../private/seed_robustness/four_versions_20260905_local/pyvrp_v0_13_0/fixed_instance_intervals/intervals.json).
- pyvrp_v0_13_4: [summary](../../../private/seed_robustness/four_versions_20260905_local/pyvrp_v0_13_4/fixed_instance_intervals/validation_summary.json), [all 1000 interval sets](../../../private/seed_robustness/four_versions_20260905_local/pyvrp_v0_13_4/fixed_instance_intervals/intervals.json).

## Supplementary fourth-version experiment

At the earlier implementation-check snapshot, v0.14.0 had not produced
`validation_summary.json`. Its recorded progress was:

> PITBENCH_PROGRESS Judge progress: instances 6/10, seed groups 12514/20000, solver runs 37550/60000, valid 37550

Its completed observations and fixed-instance intervals may be added as
supplementary evidence later. They are not outstanding requirements for this M5
cycle. Closing M5 does not stop the running experiment.

## Implementation checks

The formal report now uses `paired seed-list bootstrap with fixed instances`.
The following tests passed in the existing solver image:

```sh
python -m pytest -q -p no:cacheprovider \
  tests/unit/pitbench/test_seed_robustness_report.py \
  tests/unit/pitbench/test_seed_robustness_validation.py \
  tests/unit/pitbench/test_seed_robustness_validation_runner.py \
  tests/unit/pitbench/test_performance_pipeline.py
```

Result: 29 passed. A regression check verifies that repeating every fixed
instance row equally does not change the seed intervals. Existing checks cover
observation ordering, incomplete lists, paired no-change results, and integration.

The formal public report interface was also compared with the saved first test
list for each of the three completed versions at all three budgets. All 27
point estimates and intervals matched the reanalysis (relative tolerance
`1e-10`, absolute tolerance `1e-14`; numerical checks, not statistical gates).

## Interpretation and handoff

- The 700-seed values are finite reference estimates, not known population truths.
- The 1000 test lists reuse the 300-seed pool; they do not provide new solver runs.
- The results concern the fixed instance panel and the empty-patch comparison.
- Historical synthetic results in `validation-plan.md` used crossed resampling.
  They are retained as M3 evidence, not relabeled as seed-only coverage results.
- M5 is complete on the basis of the three finished versions, the implementation
  checks, and the user's explicit acceptance of the descriptive evidence.
- The fourth-version experiment and nonzero-effect detection study are not M5
  closure requirements. Their deferral does not imply they have been validated.
- The subsequent M6 freeze-candidate review is recorded in [audit.md](audit.md).
  No version or commit has been marked frozen.
