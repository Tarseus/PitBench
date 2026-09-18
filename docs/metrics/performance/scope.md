# Performance scope

## M0: supported task classes

Performance measures the result delivered by Base and Agent under a declared
evaluation budget, after the result passes the applicable independent qualification
checks. The dimension covers heuristic and exact optimization through separate
protocol variants.

The variants share the candidate-evaluation entry point and preserve all run outcomes,
but they do not force objective quality and time to a verified exact result into one
common scalar.

## Heuristic optimization

For heuristic solvers, Performance measures independently verified solution quality
under fixed time budgets. The comparison uses normalized objective gap for the
declared objective sense and compares Base with Agent on the same evaluation cases.

This variant does not require an optimality proof. It does require an independently
verified feasible solution and objective value for a run to contribute a gap.

## Exact optimization

For exact optimization solvers, Performance follows the SAT Competition model: it
measures time to a verified final result under fixed time and resource limits and uses
a penalized-runtime protocol for unsolved cases.

An exact run is solved only when all applicable conditions hold:

- the solver reports an optimal termination;
- the returned solution passes independent feasibility and objective verification;
- optimality is established by an independently checked certificate or by agreement
  with a fixed trusted optimum.

A timeout, out-of-memory result, crash, missing optimal result, incumbent without an
optimality result, or failed or timed-out certificate check is unsolved. A wrong
solution, wrong certificate, or false optimality claim is a Qualification failure,
not merely an unsolved Performance run.

The exact penalized-runtime definition, aggregation, pairing, and uncertainty
procedure are deferred to M2.

### OR-Tools CP-SAT exact solving

The OR-Tools exact-solving task consumes an evaluator-owned `CpModelProto` whose
content and hash are fixed before candidate evaluation. Base and Agent receive the
same proto. The timed and editable solving scope may include model loading, presolve,
propagation, search, cuts, and solving heuristics. Model construction is outside the
Performance 0.0.2 scope.

The exact-solving task evaluates the CP-SAT solving implementation. It does not combine
model-construction time with solving time or allow construction improvements to offset
solving regressions.

## Evaluation population and boundaries

Performance uses original `judge_id` observations. Development observations,
equivalent-representation transformations, operational-reliability boundary cases,
and `judge_shift` observations do not contribute to the Performance estimand or
classification. Their complete run records remain available to their owning
evaluation layers and dimensions.

The following concerns remain outside Performance:

- semantic validity and independent verification, which belong to Qualification;
- crashes, timeouts, out-of-memory results, and other run-failure classification,
  which belong to Operational Reliability even when a Performance protocol treats
  the corresponding run as unsolved or unavailable;
- solver-seed and equivalent-representation sensitivity, which belong to Nuisance
  Robustness;
- memory and CPU consumption, which belong to Resource Efficiency;
- sensitivity to configuration choices and the ability to tune them, which belong to
  Configuration Robustness & Tunability;
- behavior across problem scale, which belongs to Scalability;
- behavior under population or structural shift, which belongs to Distributional
  Generalization.

Performance reports results only for its declared task panel and protocol. It does
not claim universal solver quality, reliability, robustness, scalability, or
generalization.
