# Performance validation results

## CP-SAT exact-solving M5 control

Control artifact: `private/performance_validation/ortools_cp_sat_solve_v9_15_m5_noop_full_no_base_cache/`.

The no-op candidate patch was empty and BASE caching was disabled. The judge produced
all 360 expected observations; every observation completed and was valid. There were
no Qualification failures, infrastructure-missing observations, or BKS discrepancies.

At the primary 30-second budget, BASE solved 39/90 runs (coverage 0.4333333) with
PAR-2 mean 36.8595862 seconds; Agent solved 39/90 runs (coverage 0.4333333) with
PAR-2 mean 37.0067114 seconds. The paired mean penalized-runtime reduction was
-0.1471252 seconds, with 95% percentile bootstrap interval [-1.53583565,
1.05933792], so the control comparison is inconclusive.

At 10 seconds, the one-run, eight-worker control produced 27/90 BASE and 28/90
Agent solves; its paired confidence interval includes zero. This is recorded as the
parallel CP-SAT control's one-run noise boundary, not as an improvement.

Conclusion: the M5 control evidence passes. This result does not advance the
protocol to M6 or M7.
