# Code-structure cleanup record

## Active rule

PitBench owns one shared execution, verification, collection, and reporting flow.
Solver, problem-family, and agent differences belong to the relevant plugin or
adapter capability, not in generic CLI, scheduler, or evaluator branches.

## Default decisions applied

- Keep repository-specific build, runtime, and container details in
  `pitbench/repositories/plugins.py`; Dockerfiles remain task environment assets.
- Keep solver invocation adapters together in `pitbench/solver_drivers/run.py` and
  JVM adapter support together in `adapters/jvm/runner.py`; do not split either into
  new solver-specific Python modules.
- Keep verifier differences in `pitbench/problem_families/verification.py` and use
  problem-family capabilities from generic reliability and fixture paths.
- Keep native-agent launch integrations in their existing adapter locations. They
  contain provider credential and CLI contracts, while the benchmark execution flow
  remains shared in the harness.
- Move generic entrypoint behavior that had solver or problem names into the existing
  repository, driver, or problem-family plugin contracts. This preserves every
  configured task, seed, budget, and metric definition.

## Completed consolidation

- The shared configuration-collection Docker command obtains extra environment
  variables from the selected repository plugin. VROOM's binary path is now a
  VROOM plugin declaration rather than a global collector assumption.
- The shared doctor command obtains judge-image environment requirements from the
  selected repository plugin. Choco's Maven cache requirement is now a Choco plugin
  declaration rather than a CLI branch.
- Solver-driver trajectory support is declared by the shared driver interface, so
  repository invocation and configuration collection use the same capability.
- Reliability-case sourcing and fixture search-node reporting are declared by the
  relevant problem-family plugin instead of evaluator branches on family names.

## Open decisions

None. The two uncommitted Configuration Robustness files are user-owned and excluded
from this cleanup.
