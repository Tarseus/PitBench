# Cross-solver Nuisance Robustness test backlog

The current cross-solver seed configuration and exact-solver report adapter are
being enabled before their full test suite is executed. This file records the
tests that must be added and run before M3-M5 support claims.

## Configuration and seed assets

- Load every public task configuration through `TaskCatalog.validate_all()`.
- Verify each stochastic task has 30 unique development seeds and a disjoint
  private 30-seed evaluation list.
- Verify every private seed file hash, task identity, domain bound, and seed
  order.
- Verify deterministic VROOM and OR-Tools model-build tasks remain unsupported
  for Seed Robustness.
- Verify seed lists are reused identically across Base, Agent, instances, and
  budgets.

## Exact outcome adapter

- Capped optimal time uses independently verified optimal runs only.
- Normal time-limit runs receive the budget as a censored primary observation.
- Crash, OOM, infrastructure failure, invalid solution, and invalid optimality
  claims remain missing/failure records.
- Final normalized-gap outcomes require finite independently verified feasible
  results and never impute missing values.
- Verified-optimal coverage is retained per instance, state, and budget.
- Type 7 IQR, equal-instance aggregation, Base/Agent antisymmetry, and the
  fixed-instance shared-seed-column bootstrap match the approved protocol.

## Solver integration

- HiGHS exact task produces the exact Seed Robustness report from 30+30 seeds.
- OR-Tools CP-SAT exact task produces the same report shape with eight workers.
- Choco exact task produces the same report shape with one worker.
- Heuristic PyVRP tasks continue using the frozen normalized-gap report without
  semantic changes.
- The evaluator dispatches heuristic and exact outcome adapters correctly.

## M3-M5 validation

- Add solver-class-specific falsification cases for capped time, censoring,
  invalid optimality, final gap, coverage, and missingness.
- Run a complete Base/Agent real panel for each supported exact solver.
- Recompute reports from retained observations and verify deterministic output.
- Verify the shared M5 gate separately for every declared solver capability.
