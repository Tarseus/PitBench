# Performance M6 freeze-candidate audit

## 2026-10-15 calibration addendum

The earlier audit below records its historical evidence boundary. Complete
same-code panels have now run on the revised Choco and HiGHS instances, so the
two `revised_panel_real_validation` blockers are superseded by the full-panel
record in [validation-results.md](validation-results.md). The CP-SAT and HiGHS
knapsack empty patches each lost one verified solve and exposed the former
raw-coverage rule's false-regression behavior. A full PyVRP empty patch was
falsely `improved` by the one-run heuristic interval; a fresh reverse-order
repeat was inconclusive. Version 0.0.3 introduces instance-supported exact
coverage directions and independent confirmation of every full real
directional result. A VROOM positive and negative patch each retained its
direction in two complete opposite-order runs.

The metric remains an M6 freeze candidate. One null panel per task does not
bound the long-run false-positive rate, exact PAR-2 showed directional
intervals on two identical-code HiGHS controls, and Choco's external optimum
evidence boundary still needs review. A later full, opposite-order HiGHS
job-shop pair with matching harness fingerprints was `inconclusive` in both
rounds; this narrows the time-noise concern but does not establish a universal
false-positive rate. No M7 freeze or frozen commit is recorded here.

Metric: Performance.
Stage: M6.
Outcome: freeze candidate, not frozen.

This audit covers the Performance 0.0.2 scope: the heuristic fixed-budget and
`exact_verified_solve` protocols, including the CP-SAT exact-solving task. The
separate model-construction task has been removed at the user's direction. This
audit introduces no new metric semantics or validation criteria.

The 2026-10-13 single-worker CP-SAT task setting postdates this audit. Its
eight-worker CP-SAT validation evidence and freeze-candidate assessment do not
validate the new execution setting; a separate real-panel validation is still
needed before a new formal result is reported.

The 2026-10-14 Choco instance replacement also postdates this audit. The old
lightweight Choco control validates only the former capacity-lower-bound panel.
The new panel uses retained independent CP-SAT optimal-result records and needs a
new evaluator-owned real-panel validation. The revised oracle trust boundary also
needs review before a freeze decision; the standalone one-seed screening is not an
M5 validation result.

The 2026-10-14 HiGHS panel replacement also postdates the old HiGHS control.
The old 20/40/80-job panel had zero verified optimal coverage at both budgets.
The replacement uses 9/10/11 jobs and retains the same solver budgets and seed
grid. Its first end-to-end run exposed a verifier error: tiny numerical output
differences were rejected as nonintegers. The verifier now rounds only within
the pinned HiGHS default MIP feasibility tolerance, then checks the integer
schedule exactly. A new complete real panel remains necessary before freezing.

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

The registry remains at M6 / `freeze_candidate`. The revised Choco and HiGHS
panels have not passed new complete real-panel validation. No new frozen
version, frozen commit, or freeze status is recorded.
