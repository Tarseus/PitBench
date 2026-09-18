# Operational Reliability cross-object scope

## M0: current solver capability boundary

Operational Reliability measures observed execution coverage and failure modes on
the declared reliability grid. It is a qualification-adjacent operational
observation, not a deployment success probability and not a replacement for
independent solution verification.

The reliability grid holds the task's instance, code state, budget, thread
count, configuration, and runtime environment fixed. It preserves every
requested run, including missing, timeout, crash, OOM, solver error, invalid
solution, output error, and infrastructure failure records.

## Boundary capability by object

### CVRP and MIP

CVRP uses the existing evaluator-owned routing boundary suite. MIP uses the
existing evaluator-owned integer boundary suite. These suites provide small
inputs, explicit expected solution contracts, and independent verification.

### CP exact tasks

Choco and OR-Tools CP-SAT use their visible task `agent_dev` instances and the
existing task-specific independent verifiers as the first reliability panel.
The prepared cases are exposed under the common `boundary_cases` test-suite
identity, with the task's declared budgets, threads, and development seeds.
This is a task-panel adapter, not a claim that the current development instances
are universal CP boundary cases. Tiny synthetic CP boundary suites may be added
later under a new approved protocol decision.

### Deterministic solvers

Determinism excludes a solver from Seed Robustness, but not from Operational
Reliability. VROOM therefore uses its declared reliability grid with its existing
single solver seed and budgets.

## Shared configuration

Every current task enables the operational-reliability flag. The task's own
budget and thread configuration is retained. Stochastic tasks use their declared
development seed list; deterministic tasks retain their declared execution seed
without interpreting it as a randomization axis.

Reliability reports preserve Base/Agent pairing when both states are evaluated,
but do not convert a paired change into a performance or robustness score.

Operational failures remain owned by Operational Reliability. Semantic invalidity
remains owned by Qualification, and a timeout is not silently classified as a
semantic invalidity.

## Lifecycle boundary

This M0 scope fixes the current object and adapter boundary. It does not select
new reliability thresholds, deployment probabilities, synthetic CP case counts,
or M5 completion claims. The shared project-wide M5 validation protocol applies
to each declared solver capability separately.
