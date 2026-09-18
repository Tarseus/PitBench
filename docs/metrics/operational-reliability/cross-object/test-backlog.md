# Operational Reliability cross-object test backlog

Tests are intentionally deferred while the cross-object configuration is being
enabled.

## Configuration and grid

- Validate all current task YAML files with the task catalog.
- Confirm every current task enables Operational Reliability.
- Confirm each prepared reliability manifest uses the task's budgets, threads,
  code states, and development seed configuration.
- Confirm CP adapters preserve the original task verifier and instance paths.
- Confirm deterministic VROOM execution is reported without seed-robustness
  interpretation.

## Boundary and adapter behavior

- CVRP boundary cases retain existing route feasibility and objective checks.
- MIP boundary cases retain exact integer feasibility and false-optimality checks.
- Choco bin-packing task panels preserve trusted-optimum verification.
- OR-Tools CP-SAT task panels preserve exact-target verification.
- Missing, timeout, crash, OOM, solver-error, invalid, output-error, and
  infrastructure statuses remain distinct.

## Report behavior

- Complete grids produce pass rates and paired Base/Agent status changes.
- Missing runs produce incomplete reports rather than perfect pass rates.
- CP task-panel cases receive stable descriptions and retained identifiers.
- Reliability output never becomes a deployment success probability or a
  performance classification.

## M5 validation

- Add synthetic status and missingness falsification cases for every adapter.
- Run a complete Base/Agent real panel for each declared solver capability.
- Recompute reports from retained observations and verify deterministic output.
- Apply the shared project-wide M5 gate separately to CVRP, MIP, CP exact, and
  deterministic VROOM capabilities.
