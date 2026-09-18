# Cross-solver Seed Robustness protocol

## M2: shared sampling and inference structure for protocol version 0.0.2

This document defines the shared Seed Robustness protocol structure for the
current solver set. It does not claim that every solver is stochastic or that
every solver is currently supported. Each solver must provide a capability
adapter and an evidence-backed seed domain before it can enter a formal
validation panel.

The protocol extends the frozen CVRP Seed Robustness 0.0.1 structure without
changing that frozen version. The shared structure is:

- a solver-declared finite seed domain;
- uniform sampling without replacement from that domain;
- 30 development seeds and 30 disjoint evaluation seeds;
- the exact stored seed lists retained with the panel;
- identical seed lists for Base and Agent;
- reuse of each list across instances and budgets;
- a fresh process for each code-state, instance, seed, and budget combination;
- fixed-instance, shared-seed-column paired percentile bootstrap;
- 5,000 bootstrap resamples;
- bootstrap seed 20260824; and
- nominal two-sided 99% intervals.

The same sampled seed indices are applied to Base and Agent while instances are
held fixed. Missing, invalid, operational, and infrastructure records remain
explicit and are never imputed.

## Current task-panel targets

The shared 30-plus-30 seed structure is applied within each solver's existing
task budget and thread contract; budgets and threads are not pooled across
solvers:

- PyVRP heuristic tasks: 5 and 10 seconds, one thread, using the frozen PyVRP
  seed protocol;
- HiGHS exact MIP: 10 and 30 seconds, one thread;
- OR-Tools CP-SAT exact solving: 10 and 30 seconds, eight workers;
- Choco exact solving: 5 and 20 seconds, one thread;
- VROOM: no Seed Robustness panel because the solver is deterministic; and
- OR-Tools model construction: no Seed Robustness panel because the model
  builder is deterministic.

The HiGHS, CP-SAT, and Choco task files now declare 30 development seeds and
private 30-seed evaluation lists under this candidate protocol. Their existing
solver panels still require real Base/Agent execution and M5 validation before
they become supported Seed Robustness results.

## Solver capability boundary

### PyVRP

PyVRP has an approved stochastic seed capability and an evidence-backed unsigned
32-bit candidate domain:

\[
D_{\mathrm{PyVRP}}
=
\{d \in \mathbb{Z} \mid 0 \le d \le 2^{32}-1\}.
\]

The existing frozen Seed Robustness 0.0.1 protocol remains authoritative for
the currently supported PyVRP tasks.

### HiGHS

HiGHS exposes an explicit 'random_seed' input through the PitBench driver. The
current cross-object candidate uses the nonnegative signed-32-bit range:

\[
D_{\mathrm{HiGHS}}^{\mathrm{candidate}}
=
\{d \in \mathbb{Z} \mid 0 \le d \le 2^{31}-1\}.
\]

This is a project-declared candidate domain. Native-domain completeness still
requires M1 evidence. HiGHS exact-solver outcomes use the exact-solver adapter:
capped time-to-optimal as primary, final normalized-gap dispersion as secondary,
and verified-optimal coverage as a companion observation.

### OR-Tools CP-SAT

The CP-SAT adapter passes the solver seed through Java's integer conversion and
sets CP-SAT's random seed. A nonnegative signed-32-bit range is therefore the
initial candidate domain:

\[
D_{\mathrm{CP\text{-}SAT}}^{\mathrm{candidate}}
=
\{d \in \mathbb{Z} \mid 0 \le d \le 2^{31}-1\}.
\]

The candidate requires M1 evidence that changing this seed produces meaningful
solver variation for the selected CP-SAT task. It does not apply to the
deterministic model-construction protocol, whose model builder does not use the
solver seed to vary the constructed model.

### Choco

The Choco adapter passes the seed to the JVM runner, and the current exact
runner sets the Choco model seed. The accepted numeric seed domain is not yet
selected; the local Choco 6.0.1 source records the model seed as a Java long
and the random search components construct Java random generators from it. Some
Choco search components use negative seed values as special sentinels, so the
initial project candidate domain is the nonnegative signed-64-bit range:

\[
D_{\mathrm{Choco}}^{\mathrm{candidate}}
=
\{d \in \mathbb{Z} \mid 0 \le d \le 2^{63}-1\}.
\]

M1 must still verify meaningful seed sensitivity for the selected task before
the domain is treated as supported.

When supported, Choco exact-solver outcomes use the same exact-solver outcome
adapter as HiGHS and CP-SAT.

### VROOM

The VROOM repository plugin is marked deterministic, and the VROOM driver does
not use the shared solver-seed argument to vary its execution. VROOM is
unsupported for Seed Robustness under this protocol. It may enter a different
Nuisance axis only after an approved equivalent-transformation adapter exists.

## Outcome adapters

The sampling and inference structure is shared, but the raw outcome is
solver-class-specific.

### Heuristic solvers

Use the approved normalized-gap seed outcome and Type 7 IQR structure from the
frozen PyVRP protocol. Any new heuristic solver requires its own M1 evidence and
M2 verification of normalized-gap semantics.

### Exact solvers

For each seed, retain:

- capped time-to-optimal as the primary outcome;
- final normalized gap as a secondary outcome; and
- independently verified optimal coverage as a companion observation.

An optimal, independently verified run contributes its runtime capped at the
budget. A normal time-limit run without verified optimality contributes the
budget as a censored primary observation. Crashes, OOM, infrastructure failures,
and invalid optimality claims remain missing or failure records.

An IQR of zero caused by every seed being censored at the budget must not be
interpreted as good absolute solving performance. Coverage remains visible and
is not combined into a new score.

## Cross-solver exclusions

The shared protocol does not make these changes equivalent to solver-seed
variation:

- instance-generation randomness;
- hardware, platform, or runtime-environment changes;
- thread-count changes;
- configuration changes;
- operational failures;
- arbitrary input transformations; or
- deterministic repeated runs.

The protocol does not rank raw IQR values across different solvers without
separate object-specific interpretation. It also does not replace Performance,
Operational Reliability, Configuration Robustness, or Representation
Robustness.

## Remaining M1-M2 work

Before a solver enters a formal cross-object M5 panel, its artifacts must record:

- evidence for the solver's seed capability and meaningful seed sensitivity;
- an accepted numeric seed domain and sampling source;
- the applicable heuristic or exact outcome adapter;
- independent verification and failure ownership;
- solver-specific M3 falsification cases; and
- a real panel satisfying the shared project-wide M5 protocol.

Unsupported solvers remain explicitly unsupported; they do not count as passed
cross-solver coverage.
