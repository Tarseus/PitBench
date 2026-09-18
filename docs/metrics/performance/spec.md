# Performance specification

## M2: heuristic fixed-budget protocol

The `heuristic_fixed_budget` protocol compares independently verified normalized
objective gap under each declared solver time budget. Performance uses only original
`judge_id` observations and reports every budget separately.

### Normalized objective gap

Let `objective_sense_direction` be `+1` for minimization and `-1` for maximization.
For a finite independently verified objective and a finite nonzero anchor fixed before
candidate evaluation, define:

```text
normalized_gap =
    objective_sense_direction
    * (verified_objective - fixed_anchor)
    / abs(fixed_anchor)
```

`objective_sense` must be explicitly declared. A zero anchor is outside the
`heuristic_fixed_budget` protocol; it is not replaced by an epsilon or another implicit
scale. A missing or nonfinite objective or anchor does not produce a normalized gap.
The evaluator treats a task configuration without the required sense or with an invalid
anchor as invalid for this protocol rather than silently passing the run into the
aggregate.

A solution better than the fixed anchor has a negative normalized gap. Negative gaps,
including a result that beats a best-known-solution anchor, are retained without
clipping. The anchor is not recomputed from the current Base or Agent results.

Base and Agent are paired by:

```text
(instance_set, instance_id, solver_seed, budget_sec)
```

For each code state and budget, average valid finite normalized-gap observations across
solver seeds within each instance. Give every instance with at least one valid seed mean
equal weight when computing the overall mean, median, p95, and confidence interval.

For paired gain, subtract Agent normalized gap from Base normalized gap within each
matching solver seed. Average valid paired gains within each instance and give every
instance with at least one valid paired gain equal weight:

```text
G(T) = mean_instance(mean_valid_paired_seed(
    normalized_gap_base(x, seed; T)
    - normalized_gap_agent(x, seed; T)
))
```

A positive `G(T)` means that Agent has lower normalized gap than Base.

An instance with no valid seed mean is excluded from the corresponding gap aggregate.
An instance with no valid paired-seed mean is excluded from paired gain. Invalid,
unsuccessful, and missing runs remain in the detailed observations and completeness
records; they are not imputed as a finite normalized gap.

For Base, Agent, and paired gain, use an instance bootstrap over per-instance seed
means. Resample instances with replacement after the within-instance seed mean is
formed. Use a 95% percentile interval with 5000 resamples and fixed bootstrap seed
`20260824`. Solver seeds are not resampled as a second bootstrap level; seed sensitivity
belongs to Nuisance Robustness.

Report all declared budgets separately. `primary_budget_sec` must belong to the declared
budget list and selects the public heuristic classification:

- `improved` when the primary paired-gain interval lower bound is greater than zero;
- `regressed` when the primary paired-gain interval upper bound is less than zero;
- `inconclusive` when the interval includes zero;
- `incomplete` when the primary paired estimate or interval is unavailable.

Qualification failures disqualify the candidate independently of the heuristic
classification. Operational failures remain owned by Operational Reliability even when
they prevent a run from contributing a valid gap.

## M2: exact optimization single-run outcome

This item defines the single-run penalized-runtime outcome for the
`exact_verified_solve` protocol. It does not define aggregation across runs,
pairing, confidence intervals, classifications, or the relationship between multiple
time budgets.

## Inputs

For each exact-optimization run, let:

- `budget_sec` be the declared solver time limit `T`;
- `cpu_time_sec` be the solver CPU time;
- `wall_time_sec` be the retained wall-clock diagnostic;
- `solver_termination` be the normalized solver termination;
- `solution_verification` be the independent feasibility and objective result;
- `optimality_verification` be either an independently checked certificate result or
  comparison with a fixed trusted optimum;
- `certificate_check_budget_sec` be the separately declared checker time limit when
  certificate checking is required.

`certificate_check_budget_sec` is not derived from `budget_sec`. Certificate-checker
time is not added to the solver Performance outcome.

### Declared timing basis

Current sequential exact tasks use solver CPU time as their Performance time
observation. The parallel CP-SAT exact-solving task instead uses the wall-clock
time reported from within the CP-SAT `Solve()` call. Its timing interval excludes
the fixed `CpModelProto` read and Java runner startup. Model construction is not
evaluated by Performance 0.0.2.

The parallel CP-SAT task fixes eight CP-SAT workers on eight allocated CPU cores.
It does not permit a candidate-created external multiprocess portfolio. CPU time
and memory remain Resource Efficiency observations for this task; they are not
substituted for its wall-clock Performance observation.

## Solved exact run

An exact-optimization run is solved only when all applicable conditions hold:

- `solver_termination` reports optimal termination;
- the returned solution passes independent feasibility and objective verification;
- an independently checked optimality certificate is accepted, or the independently
  verified objective equals the task's fixed exact target.

The single-run Performance outcome for a solved run is the task's declared timing
basis. For current sequential exact tasks this is `cpu_time_sec`; for the parallel
CP-SAT exact-solving task this is the wall-clock time reported from within
`Solve()`. The other timing observation remains available for diagnostics and
Resource Efficiency.

## Fixed exact-target protocol v1

The first `exact_verified_solve` protocol supports two fixed exact-target kinds:
independently checkable trusted optima for the current HiGHS and Choco tasks, and a
published-BKS target for the parallel CP-SAT task. VIPR and other certificate-native
protocols are outside this version.

The v1 panel contains only feasible instances with integer, unambiguous objective
semantics. An infeasible instance is outside v1 until an independently checked
infeasibility-certificate protocol is approved.

Every fixed exact-target record is fixed before candidate evaluation and contains:

- the instance identity and instance-content hash;
- the objective sense and fixed integer target value;
- source and verification provenance.

A trusted-optimum record additionally contains a reference solution and its content
hash, plus an independently checkable basis for the optimum. A published-BKS target
record instead contains the published source artifact hash and path that bind its BKS
value to the instance; it does not require a reference schedule.

The candidate solver under evaluation is not the sole source of a fixed exact target.
The evaluator resolves the target from private, evaluator-owned storage and verifies
that the instance content matches its record before running the candidate.

For v1, objective comparison is exact integer equality. A run is solved only when the
solver reports optimal termination, the evaluator-owned verifier accepts the returned
solution, the independently recomputed objective equals the fixed exact target, and the
reported objective agrees with that recomputed value. Finding a fixed target as an
incumbent without reporting optimal termination remains unsolved.

### HiGHS scheduling oracle

The current generated HiGHS `judge_id` panel uses the single-machine scheduling
structure and sum-of-start-times objective declared by its generator. Oracle generation
retains the processing times, constructs a shortest-processing-time schedule, records
its exact objective and reference solution, and binds them to the generated instance
hash.

The evaluator-owned verifier checks the returned schedule, non-overlap, model-variable
values, and objective using integer arithmetic. It does not trust the HiGHS status text,
reported objective, or the current regular-expression LP verifier as independent
evidence.

### Choco bin-packing oracle

The first Choco panel uses instances with a reference packing whose bin count matches an
independently checkable lower-bound certificate. The oracle records the matching lower
bound, exact optimum, reference packing, instance hash, and provenance.

The evaluator-owned verifier checks that every item appears exactly once, every bin
respects capacity, and the recomputed bin count equals both the reported objective and
the trusted optimum. Instances without an approved independently checkable optimum are
outside the v1 panel.

### Parallel CP-SAT published-BKS target

The parallel CP-SAT judge panel binds each JSSP instance and its fixed integer BKS to
the published source artifact and its content hash. The judge does not require or retain
a reference schedule. The evaluator independently verifies the candidate-returned
schedule and recomputes its integer makespan.

For this target kind, a run is solved only when CP-SAT reports optimal termination, the
candidate schedule passes independent verification, and the recomputed makespan equals
the fixed published BKS. A verified makespan above the BKS is solver-origin unsolved and
receives the ordinary `2T` outcome. A verified makespan below the BKS is an oracle
discrepancy, not a Qualification failure: retain the validated result, pause ordinary
aggregation for that instance, and recheck the published BKS before resuming it.

### Certificate-native future protocol

A future certificate-native protocol may admit independently checked VIPR or other
optimality and infeasibility certificates. It requires its own approved certificate
formats, checker implementations, checker limits, instance scope, and validation. A
separate certificate generator or solver is not treated as neutral evidence about the
candidate solver's own proof time.

## Solver-origin unsolved run

An executed run is unsolved when solver behavior does not establish a verified exact
result without triggering a Qualification failure. This includes:

- solver timeout;
- out-of-memory termination;
- crash or solver error;
- no optimal result;
- an incumbent without an optimality result;
- certificate checking that exceeds `certificate_check_budget_sec`.

The single-run Performance outcome for a solver-origin unsolved run is `2T`, where
`T` is `budget_sec`.

## Qualification failures

A wrong solution, independently rejected certificate, or false optimality claim is a
Qualification failure. It does not produce an ordinary penalized-runtime outcome.
The candidate evaluation remains disqualified by the Qualification layer.

## Infrastructure missingness

A run that was not started, a collection or evaluator infrastructure failure, or a
certificate-checker infrastructure error is missing rather than unsolved. It has no
penalized-runtime outcome and makes any report requiring that run incomplete.

Certificate outcomes are classified as follows:

- `certificate_accepted` can satisfy the optimality condition for a solved run;
- `certificate_rejected` is a Qualification failure;
- `certificate_timeout` is solver-origin unsolved and receives `2T`;
- `certificate_checker_error` is infrastructure missingness.

All solver failures, checker outcomes, Qualification failures, and infrastructure
missing records remain in the evaluation artifacts.

## Exact-optimization aggregation

Compute each declared solver budget `T` separately. Do not pool observations from
different budgets.

Base and Agent runs are paired by:

```text
(instance_set, instance_id, solver_seed, budget_sec)
```

Performance uses only original `judge_id` observations, so `instance_set` is retained
in the pairing identity without expanding the aggregate to other instance-set kinds.

Let `Y_state(x, seed; T)` be the single-run penalized-runtime outcome for code state
`state`, instance `x`, solver seed `seed`, and budget `T`.

Let `S_state(x, seed; T)` be one when that run is a verified solved exact result and
zero when it is solver-origin unsolved. Infrastructure-missing and
Qualification-failure runs have no ordinary `S_state` value.

Verified-solved coverage is the primary exact-optimization result. Average the solved
indicator over the declared seeds within each instance and then give every instance
equal weight:

```text
C_state(T) = mean_instance(mean_seed(S_state(x, seed; T)))
```

Penalized average runtime is the secondary exact-optimization result. For each code
state, average the declared seed outcomes within each instance and then give every
instance equal weight:

```text
P_state(T) = mean_instance(mean_seed(Y_state(x, seed; T)))
```

For paired improvement, subtract Agent from Base within each paired seed before
averaging seeds within an instance and instances across the panel:

```text
G(T) = mean_instance(mean_seed(
    Y_base(x, seed; T) - Y_agent(x, seed; T)
))
```

A positive `G(T)` means that Agent has lower penalized runtime than Base.

Solver-origin unsolved runs have the defined `2T` outcome and remain in every
applicable mean. They are not excluded as invalid or unsuccessful observations.

## Exact-optimization completeness and counts

The expected grid is the Cartesian product of the declared original `judge_id`
instances, solver seeds, code states, and budgets. A required infrastructure-missing
run makes the corresponding aggregate and classification incomplete. The report does
not reduce the instance or seed panel to obtain a complete-looking aggregate.

For each budget and code state, report:

- verified-solved coverage;
- penalized average runtime;
- `solved_run_count`;
- `expected_run_count`;
- whether the declared grid is complete.

Retain per-run solved, unsolved, Qualification-failure, and infrastructure-missing
records in the detailed artifacts.

## Exact-optimization uncertainty and classification

For Base and Agent penalized runtime and the paired penalized-runtime improvement, use
an instance bootstrap over per-instance seed means. Resample instances with
replacement, preserving each sampled instance's declared seed mean. Use a 95%
percentile interval with 5000 resamples and fixed bootstrap seed `20260824`.

Report every declared budget separately. Only `primary_budget_sec` determines the
public classification. Apply the following order:

1. Return `incomplete` when the required primary grid, coverage, penalized-runtime
   aggregate, or paired penalized-runtime interval is unavailable.
2. Return `improved` when Agent verified-solved coverage is greater than Base coverage.
3. Return `regressed` when Agent verified-solved coverage is less than Base coverage.
4. When the coverage values are equal, return `improved` when the paired
   penalized-runtime interval lower bound is greater than zero, `regressed` when its
   upper bound is less than zero, and `inconclusive` when it includes zero.

A Qualification failure disqualifies the candidate independently of this
classification. The PAR-2 classification does not replace Qualification.

## Remaining M2 decisions

The checker limits, accepted formats, and independent checker implementations for a
future certificate-native protocol remain open. They do not block the trusted-optimum
v1 protocol.
