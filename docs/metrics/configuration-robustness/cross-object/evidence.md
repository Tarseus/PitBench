# Configuration Robustness & Tunability cross-object evidence

## M1: collector and parameter-space capability matrix

This document records repository evidence for the cross-object Configuration
Robustness capability boundary. It does not select new parameter ranges or claim
that every current solver is configuration-supported.

## PyVRP

The repository provides:

- a PyVRP collection backend;
- nested parameter application through the existing SolveParams interface;
- effective-parameter retention;
- independent CVRP verification;
- normalized-gap feedback; and
- an approved first parameter space for PyVRP 0.14.0.

PyVRP remains supported by the existing Configuration Robustness M2 protocol.

## HiGHS

The repository provides:

- a HiGHS collection backend;
- explicit option application through the numeric-model collector;
- effective-option retention;
- independent numeric-model and solution checks;
- capped time-to-optimal feedback with final-gap retention; and
- an approved first parameter space for HiGHS 1.15.1.

HiGHS remains supported by the existing Configuration Robustness M2 protocol.

## VROOM

The pinned VROOM v1.15.0 Usage documentation exposes `--explore` with legal
range `0..5` and default `5`. The pinned source maps this value to search depth
and the number of searches. The source also exposes an experimental TSPFix debug
switch, but the approved first space keeps that switch fixed and varies only
`exploration_level`.

The PitBench driver applies `exploration_level`, passes the fixed budget through
VROOM's `--limit`, and independently verifies the returned CVRP solution. VROOM
is an M4 implementation candidate pending M5.

## Choco

Choco has a shared JVM execution runner, but the current repository has no
configuration collection backend, effective-parameter capture contract, or
approved solver-parameter space for the pinned Choco task.

Choco is a configuration capability candidate. Its JVM and search parameters
require source evidence, legal-domain decisions, an application adapter, and an
independent retest contract before search can begin.

## OR-Tools CP-SAT

OR-Tools CP-SAT exposes an exact-solve collector and approved parameter space.
Its feedback and verification remain separate from the removed model-construction
task.

OR-Tools model construction is outside the active metric scope.

## Cross-object conclusions

The existing implementation supports formal Configuration Robustness for PyVRP,
HiGHS, VROOM, Choco, and OR-Tools CP-SAT at different lifecycle stages. The
shared search protocol—SMAC3 ask/tell, complete default and
candidate panels, explicit failure handling, independent retest, and retained
raw records—can be reused by new collectors.

The following are not established by this M1 evidence:

- that parameters with similar names have comparable meanings across solvers;
- that a PyVRP or HiGHS parameter range transfers to another solver;
- that deterministic execution eliminates the need for a solver-specific
  configuration panel;
- that missing collectors can be replaced by a numeric penalty.

Each candidate solver requires its own M2 parameter space, feedback, legal-value
rules, collection backend, independent verification, M3 falsification cases,
and M5 real panel before it can be reported as supported.
