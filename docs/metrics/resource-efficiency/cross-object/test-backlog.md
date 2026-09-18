# Resource Efficiency cross-object test backlog

Tests are deferred while the shared cross-object observation contract is being
used by all current tasks.

## Observation coverage

- Verify native Python/C++ drivers record CPU time, peak RSS, and the declared
  worker-process measurement scope.
- Verify JVM Choco and OR-Tools runners record the single native child scope.
- Verify VROOM and all exact tasks use the task's declared thread count and seed
  count in the expected grid.

## Aggregation invariants

- Base/Agent pairing is preserved by instance, budget, seed, and code state.
- Reordering observations does not change ratios or aggregates.
- Equal-instance geometric means do not weight an instance by its seed count.
- Missing, failed, invalid, zero, negative, and nonfinite values are excluded
  with explicit reasons and never become resource savings.
- Measurement-scope mismatch and thread-count mismatch remain visible.
- Boundary and representation observations do not contaminate ordinary resource
  reports.

## M5 validation

- Run synthetic resource invariants for each normalized resource scope.
- Run complete Base/Agent panels for CVRP, MIP, CP exact, and deterministic VROOM
  capabilities.
- Recompute reports from retained observations and verify identical summaries.
- Apply the shared project-wide M5 gate separately to each solver capability.
