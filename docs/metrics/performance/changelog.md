# Performance changelog

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
