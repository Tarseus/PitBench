# Cross-solver Seed Robustness evidence

## M1: solver capability and seed-domain matrix

This document records repository evidence for the cross-solver Seed Robustness
protocol in the Nuisance Robustness 0.0.2 cycle. It distinguishes an explicit
seed input from evidence that changing the seed changes solver behavior. A
solver is not considered Seed Robustness-supported until both capability and
stochastic-sensitivity evidence are recorded and the solver passes its own
M2-M5 gates.

## PyVRP

### Evidence

- The task configuration declares a seed-robustness protocol for the current
  PyVRP tasks.
- The frozen protocol declares the unsigned 32-bit domain
  \(0 \le \xi \le 2^{32}-1\).
- Each task has 30 development seeds and 30 disjoint hidden evaluation seeds.
- The evaluator starts a fresh solver process for each
  code-state/instance/seed/budget combination.
- The existing real-solver validation and frozen report provide evidence for
  repeated seed outcomes, Type 7 IQR, and fixed-instance seed-column bootstrap.

### Status

PyVRP Seed Robustness is supported under frozen protocol version 0.0.1. Its
existing frozen artifacts remain authoritative and are not changed by the
cross-solver 0.0.2 cycle.

## HiGHS

### Evidence

- The task is pinned to HiGHS 1.15.1 and source commit
  04024d701f79feb8e2f18bc3df0dffc04ef05088.
- The shared HiGHS driver passes the seed through the command-line
  '--random_seed' option.
- The current task configuration declares 30 development solver seeds
  [0, ..., 29].
- The retained MIP nuisance panel uses an explicit 30-seed axis, two budgets,
  ten instances, and 1,260 total nuisance runs.
- The current panel proves that a complete seed-axis collection is executable.
  It does not provide paired Base/Agent Seed Robustness observations.

### Domain and status

The cross-solver protocol records [0, 2^31-1] as a project-declared candidate
domain because the current configuration and experiments use nonnegative
signed-32-bit seed values. Repository evidence has not yet established that
this is the complete native HiGHS domain.

HiGHS is a Seed Robustness candidate. It is not yet a supported MIP Seed
Robustness protocol.

## OR-Tools CP-SAT exact solving

### Evidence

- The CP-SAT Java adapter parses the solver seed as a Java long.
- It converts the value with Math.toIntExact and passes it to CpSolver through
  setRandomSeed.
- The current exact CP-SAT task now declares 30 development seeds, a private
  30-seed evaluation list, two budgets, and eight solver workers.
- This establishes the shared seed-panel configuration; it does not establish
  a real Base/Agent Seed Robustness panel or meaningful seed sensitivity.

### Domain and status

The initial candidate domain is the nonnegative signed-32-bit range
[0, 2^31-1], subject to M1 confirmation of the solver's accepted range and
meaningful seed sensitivity. The current repository evidence does not establish
that range as a complete native CP-SAT domain.

CP-SAT exact solving is a Seed Robustness candidate. It requires a new M2
panel decision and a 30-seed M5 panel before support can be claimed.

## Choco exact solving

### Evidence

- The shared Choco driver passes the seed to the JVM runner.
- The current exact Choco runner parses the value as a Java long and calls
  model.setSeed(solverSeed).
- The local Choco 6.0.1 source documents Model.setSeed(long), stores the model
  seed as a long, and uses model seeds to initialize Java random generators in
  random search components.
- Some Choco search components use negative seeds as special sentinel values;
  the cross-solver candidate therefore restricts the project protocol to
  nonnegative values.
- The current Choco task now declares 30 development seeds, a private 30-seed
  evaluation list, and two budgets.

### Domain and status

The cross-solver candidate domain is the nonnegative signed-64-bit range
[0, 2^63-1]. The Java long and local source evidence support the transport and
random-generator path, but do not by themselves establish the magnitude of
seed-induced variation on the selected task.

Choco exact solving is a Seed Robustness candidate. Stochastic-sensitivity
evidence and a complete Base/Agent M5 panel remain required.

## VROOM

### Evidence

- The VROOM repository plugin is marked deterministic = True.
- The VROOM driver accepts the shared seed argument through the common command
  contract but does not use it when invoking VROOM.
- The VROOM task currently declares only solver seed 0.

### Status

VROOM is unsupported for Seed Robustness. Repeating the deterministic run with
different labels would not establish a solver-seed nuisance axis. VROOM may
enter Representation Robustness only through an independently verified
equivalent-transformation adapter.

## OR-Tools model construction

### Evidence

- The model-building adapter accepts the shared solver-seed argument in its
  command contract.
- The model builder constructs and serializes the CP model without using the
  seed to vary model construction.
- The model-construction task currently declares only solver seed 0.
- The repository plugin marks the model-building protocol deterministic.

### Status

OR-Tools model construction is unsupported for Seed Robustness. Its seed input
is a common interface field, not evidence of stochastic model-construction
behavior.

## Cross-solver conclusions

The evidence supports a shared sampling and reporting structure, not one shared
numeric seed domain or universal stochastic capability. The current matrix is:

- PyVRP: supported under frozen 0.0.1;
- HiGHS: explicit seed input, domain and sensitivity candidate;
- OR-Tools CP-SAT exact solving: explicit seed input, domain and sensitivity
  candidate;
- Choco: explicit seed input, domain and sensitivity candidate;
- VROOM: deterministic, Seed Robustness unsupported; and
- OR-Tools model construction: deterministic, Seed Robustness unsupported.

No current evidence justifies adding VROOM or deterministic model construction to
Seed Robustness. No candidate solver may enter M5 until its seed domain,
seed-list configuration, outcome adapter, independent verification, and
solver-specific M3 cases are approved.
