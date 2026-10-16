# Operational Reliability M5 validation results

Historical validation record. The current five-task benchmark retired this
dimension on 2026-10-14 after its completed Base/Agent panels showed no
discriminatory variation. The observations and failure classifications below
remain available for audit and explicit diagnostics.

The decision reviewed the four current non-HiGHS tasks under
`codex-gpt-6-luna-xhigh-four-20261012`,
`codex-gpt-6-luna-medium-four-20261012`, and
`codex-gpt-6-sol-xhigh-four-20261013`. Across all 12 task/model reports and
their declared budgets, both Base and Agent pass rates were 1.0 with no
paired pass-to-fail or fail-to-pass changes. This is a lack of discrimination
on the observed panels, not evidence that future solver changes cannot fail.

After retirement, the VROOM empty-patch normal dry run at
`runs/reliability-retired-vroom-20261014/dry-run/vroom_v1_15_0/` produced six
ordinary observations and no reliability report. The explicit
`--reliability-only` run at `runs/reliability-retired-vroom-20261014/diagnostic/`
completed all 24 declared boundary observations (four cases, three budgets,
Base and Agent), all valid, and produced a complete reliability report. This
checks that the retired dimension is absent from normal scoring while the
diagnostic remains callable.

## M5 outcome

Operational Reliability M5 passes for the current declared solver capabilities:

- PyVRP v0.14.0 (CVRP)
- VROOM v1.15.0 (deterministic CVRP)
- HiGHS v1.15.1 (MIP)
- Choco v6.0.1 (CP exact)
- OR-Tools CP-SAT v9.15 (CP exact)

The result covers observed execution coverage and preserved failure modes on each
task's declared reliability grid. It is not a deployment success probability,
performance classification, or claim about unsupported solver capabilities.

## Synthetic falsification

`tests/unit/pitbench/test_operational_reliability.py` constructs the declared
reliability grid for each capability and verifies that timeout, crash, OOM,
solver error, invalid solution, output error, infrastructure error, and a
missing run remain distinct. A missing run makes the affected report incomplete
and removes its pass rate rather than being imputed as success. The test also
recomputes the report and details from the same synthetic observations and
requires identical serialized output.

Result: 14 passed.

## Real Base/Agent panels

The following retained `trials.parquet` artifacts each produced a complete
Operational Reliability report with `qualification_passed=true`. Recomputing
each report and detail artifact twice from the retained observations produced
identical serialized output.

- PyVRP: `runs/codex-gpt-5.6-luna-xhigh-20260920-retry5/pyvrp_v0_14_0/pyvrp_v0_14_0.1-of-1.codex-gpt-5.6-luna-xhigh-20260920-retry5/evaluation/trials.parquet`
- VROOM: `runs/codex-gpt-5.6-luna-xhigh-20260920-retry5/vroom_v1_15_0/vroom_v1_15_0.1-of-1.codex-gpt-5.6-luna-xhigh-20260920-retry5/evaluation/trials.parquet`
- HiGHS: `runs/repairs/codex-gpt-5.6-luna-xhigh-20260921-no-shift/highs_v1_15_1/trials.parquet`
- Choco: `runs/m5/operational-reliability/choco_v6_0_1/trials.parquet`
- OR-Tools CP-SAT: `runs/repairs/codex-gpt-5.6-luna-xhigh-20260921-parallel/ortools_cp_sat_solve_v9_15/trials.parquet`

The Choco repair panel contains 600 independently verified completed runs and
no infrastructure error after the CP public-panel verifier repair.

## M5 gate

The approved synthetic falsification cases pass. Each declared capability has
a complete retained Base/Agent real panel, with raw observations, expected grid,
failure records, and independently verified outcomes retained by the evaluator.
Reports reproduce from those artifacts, and no remaining status in these panels
is unclassified.
