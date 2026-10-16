# Performance changelog

## v0.0.3 — 2026-10-15 (freeze candidate)

Calibrates public directions against full empty-patch controls on the seven
current tasks. Exact solved-coverage differences now need a paired
instance-bootstrap interval excluding zero; unresolved coverage differences
remain visible and cannot fall through to PAR-2. Full real directional results
require a fresh, complete, opposite-order Base/Agent confirmation with the
same direction. Measured Base observations are no longer cached by default.
The mounted judge harness is fingerprinted in run identity and opt-in cache
keys. Prior one-run labels must not be read as confirmed v0.0.3 labels.

This is not an M7 freeze or a claim of a measured universal false-positive
rate. The implementation commit and further exact-time null evidence are
pending.

## v0.0.2 — 2026-09-18

Removes the separate OR-Tools 9.15 model-construction task and protocol from
Performance. OR-Tools CP-SAT solving remains covered by `exact_verified_solve`.
This change is a freeze-candidate scope update; the implementation commit is
recorded when the change is committed.

## v0.0.1 — 2026-09-18

Performance v0.0.1 froze the approved heuristic fixed-budget, exact verified
solve, and verified CP-SAT model-construction protocols together with their
implementation and lightweight M5 validation evidence.

Implementation commit: `57780dc1`.

The release does not claim universal solver quality, reliability, scalability,
generalization, resource efficiency, or tunability. Those dimensions remain
separate evaluation programs.
