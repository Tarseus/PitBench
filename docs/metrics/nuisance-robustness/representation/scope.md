# Representation Robustness scope

## M0: equivalent customer relabeling, scope reviewed on 2026-09-08

Representation Robustness belongs to Nuisance Robustness. The initial study covers
customer ordering and relabeling for PyVRP 0.14.0 on the existing CVRP instances,
under the experimental settings recorded in [spec.md](spec.md).

With the underlying instance, solver seed, code state, budget, thread count,
configuration, and runtime environment fixed, the study measures the sensitivity
of solution quality to equivalent customer labelings.

Customer data, including coordinates and demands, are permuted together while the
depot remains fixed. Distances between corresponding nodes and vehicle capacity
must be preserved. Routes must map back to the original instance with unchanged
feasibility and objective value.

The evaluation covers Base, Agent, and the change in representation sensitivity
induced by the patch. Base and Agent use the same fixed solver seed and
transformations. The initial collection uses an empty patch, so its two code
states provide repeated-run observations rather than a patch improvement.

## First-iteration transformation scope

The user approved the following scope after reviewing practical input scenarios
and the current solver implementation.

### Include customer ordering and relabeling

The same customer set can acquire different input orders through sorting, data
export, or model construction. The experiment changes the customer order that
reaches the solver, with coordinates and demands permuted together.

There is a concrete mechanism in the pinned PyVRP implementation: its
[neighbourhood construction](https://github.com/PyVRP/PyVRP/blob/5d9776a954b810bdb3fe71d47b1e7de7cffe90d2/pyvrp/cpp/search/neighbourhood.cpp#L215)
starts from index order and uses a stable sort, so equal proximity scores can
retain an ordering affected by relabeling. The traditional-solver precedent for
permutation experiments is recorded in [evidence.md](evidence.md).

The 30 random permutations per instance are controlled probes of order
sensitivity. They do not represent the frequency of different customer orders
in a production system or estimate a production failure probability.

### Do not add separate experiments for superficial format changes

Changing only external customer IDs, JSON field order, or whitespace does not
create a new customer order in the current solver input. The
[PyVRP driver](../../../../pitbench/solver_drivers/pyvrp.py) reads the specified
arrays and assigns node indices from their positions when writing VRPLIB.
These changes are not separate solver-performance experiments in this iteration.

### Exclude demand and capacity unit changes from planned experiments

The user decided not to pursue demand and capacity unit changes as a robustness
experiment. These representations can be standardized explicitly at the input
boundary. Unit scaling is therefore outside the planned experimental scope,
replacing its earlier status as a deferred candidate.

### Defer translations, rotations, and reflections

These transformations preserve the Euclidean problem when all locations,
including the depot, are transformed together and the parsed distance matrix is
preserved. Static inspection of the pinned PyVRP 0.14.0 search did not find direct
use of location x/y coordinates. The first iteration therefore defers these
experiments. This is a priority decision for the current implementation, not an
empirical invariance claim about every solver or future patch.

### Keep changes to the underlying problem outside this scope

Demand fluctuations and coordinate errors can change the feasible set or route
costs. They require a separate experimental question and are not treated as
equivalent representations in this study.

## Experimental observations and deferred statistic selection

The first experiment holds the ten development instances, solver seed `0`, and
each budget fixed while varying customer order. It retains the outcomes of all
30 relabelings per instance, separately for Base and Agent and for the 5-second
and 10-second budgets. Complete outcomes, mappings, solutions, verification
results, and failure records are retained under [spec.md](spec.md).

Statistic selection and its related M3 validation are deferred until all
robustness experiments have been developed. IQR and its associated aggregation
and completeness rules remain a candidate in the specification, not a required
statistical report. This replaces the earlier scope direction to require IQR and
repeat the representation experiment over multiple solver seeds.

The experiment examines representation sensitivity conditional on the fixed
solver seed. Seed Robustness instead measures dispersion across solver seeds
with the representation fixed.

Using the same seed controls and pairs runs; it does not guarantee corresponding
random choices or identical search trajectories after relabeling. The study does
not require individual runs to return the same mapped route. Different routes or
search paths alone do not establish poor robustness or a correctness failure.

This scope does not select a final statistic or complete the deferred formal
specification and statistical validation.

The frozen Seed Robustness 0.0.1 protocol remains unchanged.
