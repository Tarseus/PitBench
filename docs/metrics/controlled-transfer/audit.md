# Controlled Transfer M6 freeze-candidate audit

## Decision

Controlled Transfer is an M6 freeze candidate with unresolved blockers. It is not
frozen. The implementation and its current validation evidence are retained for
review, but neither the protocol nor its task panels are fixed together.

## Evidence reviewed

- M0 scope, M1 evidence, M2 specification, and M3 falsification plan are present in
  this metric directory.
- The M4 implementation provides a common task schema, condition metadata propagated
  to expected runs and observations, report computation in the normal evaluator, and
  `pitbench report` output.
- The M4 unit suite has ten tests covering no-op behavior, strict 80% retention,
  primary-budget selection, negative target benefit, incomplete data, exact
  coverage-first classification including solver-origin unsolved handling, schema
  propagation, report integration, interrupted continuous scale support, and shared
  Bonferroni bound counts.
- The M5 replay in [validation-results.md](validation-results.md) has complete real
  Base/Agent data and an expected grid. Replaying the panel reproduces report SHA-256
  `e299747d208a526bc0881a32b9898e5a4d03b9323d685a58d0777399fa38124b`.

## Freeze blockers

1. No configured task currently opts into `evaluation.controlled_transfer`; therefore
   a normal evaluator run cannot yet produce a prospective Controlled Transfer result.
2. The only real panel is a retrospective PyVRP structural-view replay. It does not
   validate a prospectively declared reference condition, any scale path, or a joint
   condition.
3. Its frozen Agent patch has negative reference benefit. The real report correctly
   emits `reference_improvement_unestablished`, but no real panel exercises a defined
   retention ratio or 80% retention margin.
4. The current metric-program contract still names Scalability and Distributional
   Generalization as separate dimensions, while the approved Controlled Transfer
   scope replaces them. The project-level ontology record has not been reconciled.
5. Exact-solver behavior is covered by synthetic cases only; a complete real exact
   panel has not exercised the coverage-first Controlled Transfer integration.

## Non-blocking test context

The Controlled Transfer suite, ruff, compile checks, and targeted evaluation/report
tests pass. Two unrelated full-suite tests remain excluded because the current dirty
worktree changes their task-config prerequisites: the static exact judge test uses a
missing public development file, and the exact-report CLI test supplies both legacy
seed mechanisms.

## Required path to freeze

Resolve every blocker with a prospectively declared evaluator-owned panel, a retained
real validation result for every required view, and an explicit ontology decision.
Only then can a version, changelog, commit, and M7 freeze decision be considered.
