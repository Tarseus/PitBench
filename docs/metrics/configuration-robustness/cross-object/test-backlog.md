# Configuration Robustness & Tunability cross-object test backlog

Tests are deferred while the cross-object capability boundary is being recorded.

## Capability and parameter contract

- Verify every declared parameter has a type, default, legal domain, and legal
  combination constraint.
- Verify defaults can be applied and reproduced for every supported collector.
- Verify parameter rejection is preserved as parameter rejection, not converted
  into numeric feedback.
- Verify effective parameters are retained with every run.
- Verify collector identity, solver source commit, and parameter application
  environment are retained.

## Panel and feedback behavior

- Default reference panels must be complete before search starts.
- Candidate panels must use the full declared instance/seed/budget grid.
- Missing feedback must stop the affected search path without imputation.
- Solver failures, infrastructure failures, timeouts, and invalid solutions must
  remain distinguishable.
- Search-selected configurations must be retested on the declared independent
  seed panel without reselecting after retest.
- Quality feedback and exact capped-time feedback must preserve their separate
  semantics.

## Solver capability coverage

- Revalidate existing PyVRP and HiGHS collectors under the shared M5 gate.
- Add collector and parameter-space tests for VROOM only after its M1 evidence
  selects legal parameters and feedback.
- Add JVM collector tests for Choco with independent solution verification.
- Add separate CP-SAT solve and model-build collector tests for OR-Tools.
- Confirm deterministic solvers do not receive artificial seed variation.

## M5 validation

- Run approved synthetic falsification cases for each capability.
- Run complete default, search, and independent-retest panels for every declared
  supported solver.
- Recompute reports from retained manifests and raw records.
- Apply the shared project-wide M5 gate separately; passing PyVRP or HiGHS does
  not establish support for another solver.
