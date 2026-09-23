# Resource Efficiency M5 validation results

## M5 outcome

Resource Efficiency M5 passes for the current declared solver capabilities:

- PyVRP v0.14.0 (CVRP)
- VROOM v1.15.0 (deterministic CVRP)
- HiGHS v1.15.1 (MIP)
- Choco v6.0.1 (CP exact)
- OR-Tools CP-SAT v9.15 (CP exact)

The result validates paired resource observations, explicit exclusions, and
report reproducibility. It does not select a resource threshold, combined
quality-resource score, hardware-normalized ranking, or deployment-cost claim.

## Synthetic resource contract

The relevant resource tests cover pairing, reordering invariance, equal-instance
aggregation, boundary exclusion, missing/non-positive measurements,
measurement-scope mismatch, thread-count mismatch, native child measurement,
VROOM child measurement, and retention of resource fields on a parsed failed
run.

Result: 9 passed.

## Real Base/Agent panels

The following retained `trials.parquet` artifacts each recompute identical
Resource Efficiency report and detail output on repeated calculation:

- PyVRP: `runs/codex-gpt-5.6-luna-xhigh-20260920-retry5/pyvrp_v0_14_0/pyvrp_v0_14_0.1-of-1.codex-gpt-5.6-luna-xhigh-20260920-retry5/evaluation/trials.parquet`
- VROOM: `runs/m5/resource-efficiency/vroom_v1_15_0-fixed/trials.parquet`
- HiGHS: `runs/m5/resource-efficiency/highs_v1_15_1/trials.parquet`
- Choco: `runs/m5/resource-efficiency/choco_v6_0_1/trials.parquet`
- OR-Tools CP-SAT: `runs/repairs/codex-gpt-5.6-luna-xhigh-20260921-parallel/ortools_cp_sat_solve_v9_15/trials.parquet`

PyVRP records the declared
`worker_process_start_through_solver_return` scope. VROOM, HiGHS, Choco, and
OR-Tools CP-SAT record the declared `single_native_child_lifetime` scope.

The HiGHS panel retains CPU, peak RSS, and measurement scope on all 4,200
`no_solution` observations. Those failures remain explicit resource exclusions;
they are not converted into resource savings. The VROOM panel retains its child
resource fields and classifies a native timeout as `timed_out` rather than
`crashed`.

## M5 gate

The approved synthetic resource contract passes. Each declared capability has a
complete retained Base/Agent real panel with its expected grid, raw observations,
failure records, and resource measurements. Reports reproduce from the retained
artifacts, and no resource observation in these panels is silently imputed or
treated as a saving when unavailable.
