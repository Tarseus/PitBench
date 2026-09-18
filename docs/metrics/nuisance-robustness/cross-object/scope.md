# Nuisance Robustness cross-object scope

## M0: capability matrix for protocol version 0.0.2

This scope starts a new Nuisance Robustness protocol cycle. It does not modify
the frozen Seed Robustness or Representation Robustness `0.0.1` protocols.

Nuisance Robustness measures sensitivity to a controlled nuisance axis while
holding the underlying optimization object, code state, solver configuration,
budget, thread count, and runtime environment fixed. The nuisance axis is
capability-specific: different optimization families may require different
equivalent transformations or stochastic controls. A common estimator is not
assumed across all axes or problem families.

## Capability boundary

### Seed Robustness

Seed Robustness is eligible when the solver declares a meaningful stochastic
`solver_seed` capability and repeated seeds can be assigned under a declared
seed-domain and sampling protocol.

The protocol may cover heuristic, MIP, or CP solvers when their seed behavior,
seed domain, and independent repeated-run semantics are established in later
M1-M2 artifacts. Instance-generation randomness, hardware or platform changes,
thread-count changes, configuration changes, and operational failures remain
outside this axis.

Deterministic solvers are unsupported for Seed Robustness. A repeated run with
the same deterministic behavior is not silently treated as seed variation.

### Representation Robustness

Representation Robustness is eligible only when a family-specific equivalent
transformation adapter exists. The adapter must generate the declared
transformation, preserve the intended optimization object, and independently
verify feasibility and objective preservation after mapping a solution back to
the original representation.

The metric does not treat arbitrary input changes, changed feasible sets, unit
changes, hardware changes, or unverified format edits as equivalent
representations.

## Current object and capability matrix

The following matrix records the approved direction for the `0.0.2` research
cycle. `Candidate` means that M1 evidence and M2 protocol decisions are still
required. `Unsupported` is not a validation pass.

### CVRP

- stochastic solver: Seed Robustness candidate;
- customer relabeling with independent route mapping: Representation Robustness
  existing `0.0.1` capability; and
- deterministic solver: Seed Robustness unsupported unless a future protocol
  changes the nuisance question.

### MIP

- stochastic solver: Seed Robustness candidate when the solver's seed behavior
  is established;
- row/column permutation with independent model and solution checks:
  Representation Robustness candidate; and
- deterministic solver: Seed Robustness unsupported.

### CP

- stochastic solver: Seed Robustness candidate when the solver's seed behavior
  is established;
- equivalent representation: unsupported until a family-specific independent
  transformation adapter exists; and
- deterministic solver: Seed Robustness unsupported.

The current repository has formal evidence and frozen implementation only for
the CVRP Seed Robustness and CVRP customer-relabeling protocols. The MIP and CP
entries above are research candidates, not completed support claims.

## Shared exclusions

This scope excludes:

- instance-generation randomness as a substitute for solver-seed variation;
- hardware, platform, or runtime-environment changes;
- thread-count changes;
- configuration changes, which belong to Configuration Robustness & Tunability;
- timeouts, crashes, OOM, numerical failures, and infrastructure missingness,
  which belong to Operational Reliability;
- arbitrary changes to the underlying optimization problem; and
- any transformation without an independently verified equivalence contract.

## Lifecycle boundary

M0 fixes the object and capability boundary only. It does not select seed counts,
seed distributions, transformation counts, estimators, aggregations, confidence
intervals, thresholds, or M5 acceptance details beyond the shared project-wide
M5 validation protocol.

Each candidate capability must provide its own M1 evidence, M2 formal
specification, M3 falsification plan, and M4-M5 implementation and validation
artifacts before it can be reported as supported.
