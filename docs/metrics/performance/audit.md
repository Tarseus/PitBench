# Performance M6 freeze-candidate audit

Metric: Performance.
Stage: M6.
Outcome: freeze candidate, not frozen.

This audit covers the full Performance scope: the `exact_verified_solve` protocol
for HiGHS and Choco, the CP-SAT exact-solving task, and the
`verified_cp_sat_model_construction` protocol. It introduces no metric semantics,
validation criteria, or freeze decision.

## Available M5 evidence

The CP-SAT exact-solving control is recorded in
[validation-results.md](validation-results.md). It has a complete 360-observation
no-op control panel with no Qualification failure, infrastructure missingness, or
BKS discrepancy.

## M5 evidence and blocker resolution

The lightweight HiGHS control is recorded in
`private/performance_validation/highs_v1_15_1_m5_noop_reduced_judge_id_5_retry_timeout_solution/`.
It contains 60 complete observations with no Qualification failure or infrastructure
missingness; timeout runs are classified as solver-origin unsolved.

The lightweight Choco control is recorded in
`private/performance_validation/choco_v6_0_1_m5_noop_reduced_judge_id_5_retry_python3/`.
It contains 60 complete, valid, independently verified optimal observations.

The CP-SAT exact-solving control remains recorded in
`validation-results.md` with 360 complete valid observations. The lightweight
CP-SAT model-construction dimensioning/M5 control is recorded in
`private/performance_validation/ortools_v9_15_m5_model_build_reduced_judge_id_5/`;
it contains 10 valid Base/Agent samples and 10/10 independent structure-verifier
passes. These controls close the M6 implementation and M5-validation blockers for
the lightweight scope.

The full JMH parameter freeze remains a later M7 protocol decision; this audit does
not freeze it or introduce new metric semantics.

## Registry and M7 handoff

The registry records M6 / `freeze_candidate` with no remaining lightweight-control
blocker. No version, frozen commit, or freeze status is recorded. M7 still requires
explicit user approval and a version/commit decision.
