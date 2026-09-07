# Representation Robustness scope

## M0: equivalent customer relabeling

Representation Robustness belongs to Nuisance Robustness. The initial study covers
customer relabeling for the existing PyVRP/CVRP tasks.

With the underlying instance, solver seed, code state, budget, thread count,
configuration, and runtime environment fixed, the study measures the sensitivity
of solution quality to equivalent customer labelings.

Customer data, including coordinates and demands, are permuted together while the
depot remains fixed. Distances between corresponding nodes and vehicle capacity
must be preserved. Routes must map back to the original instance with unchanged
feasibility and objective value.

The evaluation covers Base, Agent, and the change in representation sensitivity
induced by the patch. Base and Agent use the same solver seeds and transformations.

## Approved measurement direction

For each fixed instance and solver seed, run multiple equivalent customer
labelings and compute the IQR of their gap values. Repeat this assessment over
multiple solver seeds and fixed instances, then summarize the results using a
method to be specified later.

This direction measures central dispersion across representations conditional on
the solver seed. Seed Robustness instead measures dispersion across solver seeds
with the representation fixed.

Using the same seed controls and pairs runs; it does not guarantee corresponding
random choices or identical search trajectories after relabeling. The study does
not require individual runs to return the same mapped route. Different routes or
search paths alone do not establish poor robustness or a correctness failure.

The proposed IQR is not a distance between marginal output distributions. This
scope does not adopt a two-distribution comparison metric.

## Decisions reserved for the experimental protocol

The number of customer relabelings and the method of aggregating results across
solver seeds and instances remain to be decided. This M0 document records the
approved research scope and IQR direction, not a complete estimator or execution
protocol.

The frozen Seed Robustness 0.0.1 protocol remains unchanged.
