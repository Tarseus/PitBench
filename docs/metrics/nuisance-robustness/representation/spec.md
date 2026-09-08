# Representation Robustness specification draft

## M2: experimental protocol, statistic selection deferred

This document records the user's approved simplified design for equivalent
customer relabeling in PyVRP/CVRP. The first experiment uses only PyVRP 0.14.0
(`pyvrp_v0_14_0`), with the source revision pinned by the existing
[task configuration](../../../../configs/tasks/pyvrp_v0_14_0.yaml).
It is an experimental protocol draft, not a frozen metric specification.

On 2026-09-08, the user deferred the choice of Representation Robustness
statistic and its related M3 validation until all robustness experiments have
been developed. IQR remains a candidate rather than a required estimator. The
approved experimental protocol and complete result retention remain in place.
Confidence intervals and hard acceptance thresholds are not included in this
iteration.

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

## Deferred IQR candidate

The following scheme was approved earlier and is retained for reconsideration.
It is not the current required reporting scheme; the statistic and its
associated aggregation will be revisited after the robustness experiments.

Under this scheme, each original instance, code state, and budget would have an
IQR computed from the 30 normalized gap outcomes of its customer relabelings
under solver seed `0`.
The IQR is the 75th percentile minus the 25th percentile, using the Hyndman–Fan
Type 7 sample-quantile convention: linear interpolation at zero-based position
`(sample_count - 1) * probability` in the sorted outcomes. This is the convention
used by `numpy.quantile(..., method="linear")`.

For each code state and budget, the scheme would take the arithmetic mean of
the per-instance IQR values over the fixed panel of ten original instances,
giving each instance equal weight. The IQR would be computed within each
instance before averaging, without pooling gap outcomes from different
instances or budgets. There would be no averaging across solver seeds.

The scheme would report the 5-second and 10-second budgets separately. Base and
Agent use the same solver seed and customer relabelings. Complete outcomes and
mappings remain retained regardless of the eventual statistic.

The user has paused the proposed M3 checks associated with statistic selection.
Neither the candidate scheme nor the collected results establish an ability to
detect real patch improvements.

## Outcome retention and deferred IQR completeness rules

The completeness rules below were approved for the IQR scheme. They remain
part of that deferred candidate, rather than a final rule for an as-yet
unselected statistic.

A run contributes a valid statistical outcome only when its solution passes the
required verification and its normalized gap is a finite number. A normal stop
at the 5-second or 10-second budget that returns such a solution and gap is a
valid outcome.

For each original instance, code state, and budget, compute the IQR only when
all 30 predefined customer relabelings have valid outcomes. Otherwise, report
the per-instance IQR as `null`, together with the valid-outcome count out of 30
and the reasons for missing or invalid outcomes.

For each code state and budget, compute the equal-weight mean only when all ten
fixed instances have an available IQR. Otherwise, report the overall mean as
`null` while still displaying the available per-instance IQR values. Do not
reduce the instance panel to obtain an overall mean.

Determine completeness separately for Base, Agent, and each budget. A Base/Agent
comparison requires both corresponding summaries to be available.

Raw-result retention continues while statistic selection is deferred. Retain
all raw results and failure records. Do not impute missing or invalid gap values,
including with zero or infinity, and do not replace a failed relabeling with a
different relabeling. These rules keep the declared set of observations fixed.

## Interpretation and separation from Seed Robustness

The experiment examines representation sensitivity conditional on the declared
solver seed. The IQR candidate describes central dispersion across the sampled
relabelings; a smaller IQR means less central dispersion. That candidate does
not average over the solver's seed domain or measure a distance between marginal
output distributions. No final statistic is selected at this stage.

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

The user's decision on 2026-09-08 supersedes the earlier decision to fix IQR as
the M2 statistic. Type 7 IQR, the equal-weight instance mean, and their
complete-data rules are retained as the earlier candidate for later review.
Statistical selection and its related M3 validation are deferred until all
robustness experiments have been developed; they do not block the experimental
work. The approved experimental settings and complete result retention remain
in effect.

This deferral does not complete the formal metric specification or M3
validation, and it does not mark the metric frozen.
