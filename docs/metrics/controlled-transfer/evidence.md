# Controlled Transfer evidence

## M1 evidence boundary

This document records approved literature and repository conclusions for Controlled
Transfer. It does not choose thresholds, scale levels, held-out domains, weights, or
an aggregation formula.

## Scale and structural conditions are problem-family specific

The reviewed CVRP, scheduling, bin-packing, and Job-Shop sources support describing
scale through each problem family's generation rules or shape, rather than ordering
all instances by one raw count. They also support defining structural transfer through
an explicit generation mechanism or source-domain meaning.

- Uchoa et al., [New benchmark instances for the Capacitated Vehicle Routing Problem](https://doi.org/10.1016/j.ejor.2016.08.012), distinguishes CVRP instance-generation factors.
- Queiroga et al., [The XL Instances and the CVRPLib Best Known Solution Challenge](https://arxiv.org/abs/2601.11467), provides large CVRP instances whose scale and structural settings require separate interpretation.
- [MIPLIB 2017](https://doi.org/10.1007/s12532-020-00194-3) documents feature-aware selection of a diverse MIP collection.
- [BPPLIB](https://site.unibo.it/operations-research/en/research/bpplib-a-bin-packing-problem-library) documents bin-packing families with distinct construction mechanisms.
- [OR-Library weighted-tardiness information](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/wtinfo.html) documents parameterized single-machine scheduling generation.
- The [OR-Tools Job-Shop documentation](https://developers.google.com/optimization/scheduling/job_shop) specifies the Job-Shop semantics; Taillard's [Benchmarks for basic scheduling problems](https://www.sciencedirect.com/science/article/pii/037722179390182M) is a scheduling benchmark source.

These sources are evidence for panel design only. They do not prescribe a PitBench
threshold, scale path, held-out domain, or estimator.

## Task-native Performance remains the comparison basis

The repository contains task-native Performance reporting for fixed-budget heuristic
quality and exact verified solving. Controlled Transfer compares an Agent with its
unmodified Base under the applicable task-native Performance protocol; it does not
replace that protocol with a universal raw performance quantity.

For PyVRP and VROOM, the applicable basis is fixed-budget, independently verified
quality. For HiGHS, Choco, and OR-Tools CP-SAT, the applicable basis is exact
Performance: solved coverage is compared first, then paired penalized runtime when
coverage is equal. The exact Performance distinction is documented in
[Performance specification](../performance/spec.md), and the task drivers and report
implementation retain the corresponding solver-specific execution and timing data.

This evidence supports a shared transfer-evaluation framework across these tasks. It
does not select a new Performance protocol or replace the exact coverage-first rule.
