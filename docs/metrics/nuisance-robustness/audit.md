# Seed Robustness M6 freeze-candidate audit

Date: 2026-09-07.

Metric: Nuisance Robustness / Seed Robustness.
Stage: M6.
Outcome: freeze candidate; the two reviewed consistency issues are resolved.
This audit records the current working tree, not a frozen commit or version.

## Candidate protocol

- Each task has 30 development seeds and 30 disjoint hidden evaluation seeds.
- Per-instance central seed dispersion is the Type 7 IQR of normalized gap.
- Base and Agent use the same instance, budget, and seed conditions.
- Aggregates give equal weight to the fixed paired-complete instance set.
- Nominal 99% paired percentile intervals resample shared seed columns only,
  with 5000 replicates and bootstrap seed `20260824`.
- Current PyVRP task budgets are `[5.0, 10.0]` seconds; the primary budget is
  `10.0` seconds.
- Bias, empirical coverage, and interval width are descriptive results. There
  is no hard statistical acceptance threshold for this cycle.
- Robustness change is Agent IQR minus Base IQR. Negative values indicate
  reduced central dispersion. The report does not add a categorical verdict.

The formal definition is in [spec.md](spec.md). This audit introduces no new
metric semantics or validation criteria.

## Resolved consistency issues

1. The four PyVRP task configurations still included a 1-second budget after the
   user chose 5 and 10 seconds. All four now use `[5.0, 10.0]`. The integration
   test expects two budget cells and 6960 observations across its existing grid.
2. [evidence.md](evidence.md) still described the current setting as five seeds.
   It now records the M5 decision of 30 seeds per list and points to the current
   specification and descriptive validation evidence. The actual task seed
   counts were already 30 and did not require regeneration.

No unresolved blocker remains among these reviewed implementation/document
consistency issues.

## Validation evidence

M5 was explicitly closed using PyVRP v0.12.2, v0.13.0, and v0.13.4. Each completed
60000 valid solver observations on the 10-instance panel under the original
three-budget experiment. The original 1-second results remain historical
diagnostics; changing future task budgets does not rewrite that experiment.

[validation-results.md](validation-results.md) records the original evidence and
the fixed-instance interval reanalysis. The formal report interface matched the
saved reanalysis for 27 point estimates and intervals across three versions and
three budgets, using the first stored test list for each version.

The fixed-instance implementation previously passed 29 relevant tests. After the
budget update, `TaskCatalog.validate_all()` succeeded, and explicit checks
confirmed all four tasks use `[5.0, 10.0]`, a 10-second primary budget, and 30 seeds
per list. The following tests passed in the existing solver image:

```sh
python -m pytest -q -p no:cacheprovider \
  tests/unit/pitbench/test_performance_pipeline.py \
  tests/unit/pitbench/test_seed_robustness_validation_runner.py
```

Result: 12 passed. The full `test_tasks.py` suite could not be collected in that
image because `typer` is absent; the task catalog was instead validated through
its API. This is the recorded test scope, not a claim that the full suite ran.

## Accepted boundaries and deferred work

- The result measures central seed dispersion, not all tail risks or performance
  quality. The 700-seed references are finite estimates, not exact truths.
- The user accepted the descriptive M5 results without requiring a hard
  coverage threshold or a universal sufficiency claim for 30 seeds.
- Nonzero robustness-difference detection validation is deferred and is not a
  requirement for closing this cycle.
- PyVRP v0.14.0 is supplementary evidence and is not a condition for M5 closure
  or this M6 audit. This audit does not stop or modify the existing run.
- Original synthetic results used crossed instance/seed resampling. They remain
  historical M3 evidence, not coverage results for the new fixed-instance method.

## Git and M7 handoff

The implementation, configuration fixes, supporting scripts/tests, and documents
are still working-tree changes. No code commit was created by this documentation
handoff. The registry records M6 / `freeze_candidate`; M7 has not started.

Formal freezing still requires the user's explicit M7 approval and choice of
version, plus a traceable implementation commit and the authorized freeze records.
Those are the next-stage actions, not additional M5 experimental requirements.
