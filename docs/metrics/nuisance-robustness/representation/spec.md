# Representation Robustness specification draft

## M2: first iteration approved on 2026-09-07

This document records the user's approved simplified design for equivalent
customer relabeling in PyVRP/CVRP. The first experiment uses only PyVRP 0.14.0
(`pyvrp_v0_14_0`), with the source revision pinned by the existing
[task configuration](../../../../configs/tasks/pyvrp_v0_14_0.yaml).
It is an experimental protocol draft, not a frozen metric specification.

The current priority is to collect and retain complete experimental results.
Statistics will be selected after inspecting those results. IQR remains a
candidate rather than a required estimator.

The user selected one fixed solver seed and 30 equivalent customer relabelings
per instance. This decision supersedes the earlier multiple-seed direction in
[scope.md](scope.md). The proposed 30-seed by 30-relabeling experiment is not part
of this first iteration.

## Controlled runs

The solver seed is fixed at `0`. Use that same seed for all instances, customer
relabelings, budgets, and both Base and Agent. Each run initializes the solver
with this fixed seed.

Use an empty candidate patch for the first experiment. Base and Agent therefore
execute the same code in separate runs. Their results form a repeated-run
baseline and must not be interpreted as evidence of a patch improvement.

Use a separate relabeling generator with generation seed `20260907`. Generate
relabelings in ascending original instance-ID order. For each instance, generate
30 distinct customer permutations other than the original ordering, rejecting
duplicates and the original ordering. Save the actual mappings for reproduction
and reuse them across budgets and both code states. The original representation
is not one of these 30 experimental inputs.

For each instance, use the same 30 customer relabelings for Base and Agent.
Keep the depot fixed and permute customer coordinates and demands together.
Distances between corresponding nodes, capacity, feasibility, and the objective
value of a mapped route must be preserved, as required by the M0 scope.

The underlying instances, thread count, configuration, and runtime environment
remain fixed. Evaluate at 5 and 10 seconds and retain separate results for each
budget.

## Complete experimental results

Retain the following information for every run:

- The original instance identifier and the customer-relabeling mapping.
- The fixed solver seed, code version, and Base or Agent code state.
- The time budget and actual elapsed time.
- The objective value and normalized gap, when available.
- The output routes and feasibility-verification results, when available.
- The execution status and failure reason for unsuccessful runs.

Retain unsuccessful runs as well as successful ones. Complete result retention
does not require a failed run to have an objective value or a solution.

## Statistics after result inspection

Inspect the per-instance outcome distributions and Base/Agent differences before
selecting a summary statistic. The same fixed seed and customer relabelings
provide the paired experimental conditions.

The user deferred the statistical choices. IQR is a candidate; the earlier
proposal to require per-instance IQR and equal-weight averaging is not the
current reporting requirement. Aggregation formulas, confidence intervals, and
statistical handling of missing or invalid outcomes remain to be decided after
inspecting the retained results.

This decision supersedes the mandatory IQR direction recorded in the earlier M0
scope and discussed in the M1 evidence document. The literature evidence remains
relevant to the experiment's motivation.

## Interpretation and separation from Seed Robustness

The experiment examines representation sensitivity conditional on the declared
solver seed. It does not average over the solver's seed domain. No distribution
distance or other final robustness statistic is selected at this stage.

Using the same solver seed does not require identical search trajectories or
identical mapped routes after relabeling.

Seed Robustness remains a separate evaluation: it uses 30 solver seeds with the
representation fixed. Its frozen 0.0.1 protocol is unchanged.

## Initial experiment size

For the existing fixed panel of 10 instances, 30 customer relabelings, two code
states, and two budgets, the PyVRP 0.14.0 experiment requires 1,200 relabeling runs.
The sum of the configured solver budgets is 9,000 seconds, or 2.5 serial hours.
This is a budget calculation, not a wall-clock completion estimate; it excludes
process overhead and any separate original-representation control runs.

## Decision status

The solver version, fixed solver seed, relabeling count and generation seed,
exclusion of the original ordering, and empty-patch baseline are approved for
the first experiment.

Statistical choices are deliberately deferred until results can be inspected;
they are not prerequisites for collecting the experimental outcomes. M2 is not
a completed formal metric specification.
