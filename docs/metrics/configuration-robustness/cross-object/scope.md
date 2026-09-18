# Configuration Robustness & Tunability cross-object scope

## M0: parameter-space, collector, and feedback capability boundary

Configuration Robustness & Tunability studies whether a solver can be made to
degrade relative to its declared default configuration within an explicitly
approved legal parameter space. It does not treat arbitrary changes to budgets,
threads, input instances, numerical correctness requirements, or runtime
environment as configuration perturbations.

The shared search structure is:

- a declared parameter space with defaults, types, bounds, choices, and legal
  combinations;
- fixed instances, budgets, threads, input representation, verification, and
  runtime environment;
- paired solver seeds where the solver is stochastic;
- a complete default reference panel;
- complete candidate panels before feedback is sent to the searcher;
- separate handling of solver failures, parameter rejection, infrastructure
  failure, and missing feedback;
- a fixed search budget and an independent retest of the selected candidate; and
- the shared project-wide M5 validation gate.

The estimator and feedback are capability-specific. No common parameter names,
parameter ranges, or solver cost units are assumed across repositories.

## Current capability matrix

### PyVRP

PyVRP has a working collection backend, nested parameter application through
SolveParams, independent CVRP verification, normalized-gap feedback, and the
approved first parameter space recorded in the existing M2 specification.

PyVRP is supported by the existing Configuration Robustness protocol. Its
current M2 artifacts remain authoritative.

### HiGHS

HiGHS has a working collection backend, explicit option application, numerical
model and solution verification, capped time-to-optimal feedback, final-gap
retention, and the approved first parameter space recorded in the existing M2
specification.

HiGHS is supported by the existing first-iteration configuration protocol. Its
current M2 artifacts remain authoritative.

### VROOM

VROOM exposes a deterministic search configuration through its v1.15.0
`exploration_level` option. The first candidate space keeps the fixed instance,
budget, thread count, routing environment, and solver seed while varying only
that search parameter. Determinism does not replace candidate panels or an
independent retest.

VROOM is an M4 implementation candidate pending the shared M5 validation gate.

### Choco

Choco has a shared solver runner but no configuration collection backend or
approved parameter-space declaration. Its JVM options and search parameters
must be identified from the pinned source and applied through a reproducible
collector before search can begin.

Choco is a configuration capability candidate.

### OR-Tools CP-SAT

OR-Tools CP-SAT has an exact-solve configuration collector and a solver-parameter
space. Its feedback uses capped time-to-optimal with independent verification.

OR-Tools model construction is outside the active Performance and Configuration
Robustness scopes.

## Shared exclusions

This scope excludes:

- changing the optimization instance or distribution;
- changing the solver budget or declared thread count;
- changing hardware, runtime environment, or numerical verification tolerances;
- changing the problem representation unless it is separately declared as a
  Representation Robustness transformation;
- using a parameter outside its independently verified legal domain;
- treating parameter rejection as a numeric performance outcome; and
- silently assigning a penalty to a failed or unavailable candidate panel.

## Lifecycle boundary

M0 fixes the capability boundary only. It does not choose new solver parameters,
ranges, feedback formulas, search budgets, failure costs, thresholds, or
acceptance criteria for VROOM, Choco, or OR-Tools. Those choices require
solver-specific M1 evidence and M2 approval.

The existing PyVRP and HiGHS M2 protocols are not changed by this cross-object
scope. VROOM, Choco, and OR-Tools CP-SAT must pass the shared M5 validation
protocol independently before being reported as supported.
