# Configuration Robustness & Tunability cross-object candidate specification

## M2: approved first parameter spaces

This document records the approved cross-object M2 candidate spaces. The
existing PyVRP and HiGHS M2 specifications remain authoritative and are not
changed here.

The shared fixed conditions remain those in the existing Configuration
Robustness protocol: fixed instances, input representation, budgets, threads,
verification, numerical requirements, runtime environment, solver seed panels,
complete default reference panels, complete candidate panels, and independent
retest.

## VROOM 1.15.0

The first VROOM space varies one solver search parameter:

```yaml
exploration_level:
  type: integer
  bounds: [0, 5]
  default: 5
```

The pinned source maps this option to search depth and search count. Threads,
budget, routing environment, input representation, and the deterministic seed
label remain fixed. The experimental `apply_tsp_fix` debug switch is outside the
first space.

VROOM uses normalized-gap feedback against the fixed BKS and an independent
retest in a fresh process with the same deterministic seed label.

## Choco 6.0.1

The pinned Choco source exposes multiple integer search strategies applicable to
the current bin-packing runner. The first candidate space is one categorical
parameter:

```yaml
search_strategy:
  type: categorical
  choices:
    - input_order_lb
    - min_dom_lb
    - dom_over_wdeg
    - activity_based
  default: input_order_lb
```

The choices map to the pinned Choco Search API as follows:

- `input_order_lb`: current runner behavior;
- `min_dom_lb`: smallest-domain variable selection with lower-bound values;
- `dom_over_wdeg`: domain-over-weighted-degree selection with lower-bound values;
- `activity_based`: activity-based variable selection.

The first space keeps restart policy and random search,
threads, budget, and solver seed fixed. Random search is not included in this
first space because it would confound Configuration Robustness with Seed
Robustness. Restart policy is reserved for a later candidate because the
current runner does not expose it.

The intended exact-solver feedback is the existing capped time-to-optimal
feedback contract, with final gap and verification records retained as
auxiliary observations. A Choco collector and parameter application adapter
are required before this space can be executed.

## OR-Tools CP-SAT 9.15

The pinned CP-SAT source exposes the following first candidate spaces:

```yaml
cp_model_presolve:
  type: categorical
  choices: [true, false]
  default: true

cp_model_probing_level:
  type: integer
  bounds: [0, 2]
  default: 2

linearization_level:
  type: integer
  bounds: [0, 2]
  default: 1

symmetry_level:
  type: integer
  bounds: [0, 4]
  default: 2

search_branching:
  type: categorical
  choices:
    - automatic_search
    - lp_search
    - pseudo_cost_search
    - portfolio_search
  default: automatic_search
```

The source comments support the stated ranges and enum values. The first space
holds constant:

- `num_workers` and the task's eight-worker contract;
- time limits;
- solver seed;
- numerical correctness settings; and
- logging and output settings.

`randomize_search`, `use_lns`, and additional worker controls are not in the
first space because they introduce additional stochastic or parallel-search
interactions. They remain candidates for a separately approved space.

The intended exact-solver feedback is capped time-to-optimal, with final gap,
independent verification, and coverage retained separately. An OR-Tools
collector and parameter application adapter are required before execution.

## Scope exclusion

OR-Tools model construction has no active solver-configuration parameter space in
this cycle. Its instance-generation fields are not silently reclassified as
solver parameters.

## M4-M5 boundary

The approved spaces do not imply implementation or validation completion. Each
new collector must retain effective parameters, legal-parameter rejections,
complete default and candidate panels, raw failures, independent verification,
and independent retest results. The shared M5 protocol is the stopping point;
M6 and M7 are not part of this cycle.
