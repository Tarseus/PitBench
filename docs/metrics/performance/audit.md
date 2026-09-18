# Performance M6 freeze-candidate audit

Metric: Performance.
Stage: M6.
Outcome: freeze candidate, not frozen.

This audit covers the Performance 0.0.2 scope: the heuristic fixed-budget and
`exact_verified_solve` protocols, including the CP-SAT exact-solving task. The
separate model-construction task has been removed at the user's direction. This
audit introduces no new metric semantics or validation criteria.

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
`validation-results.md` with 360 complete valid observations. Model-construction
artifacts are no longer part of the active Performance scope.

The Performance 0.0.2 protocol removal remains a freeze-candidate change until its
implementation commit and lifecycle handoff are recorded.

## Registry and M7 handoff

The registry records M6 / `freeze_candidate` with no remaining lightweight-control
blocker. No version, frozen commit, or freeze status is recorded. M7 still requires
explicit user approval and a version/commit decision.
