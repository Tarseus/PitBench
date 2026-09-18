# Representation Robustness formal specification

## M2: equivalent customer relabeling protocol

This document records the user's approved `0.0.1` protocol for equivalent
customer relabeling in PyVRP/CVRP. The first experiment uses only PyVRP 0.14.0
(`pyvrp_v0_14_0`), with the source revision pinned by the existing
[task configuration](../../../../configs/tasks/pyvrp_v0_14_0.yaml).

On 2026-09-18, the user confirmed the headline statistic and aggregation for
M2. This is the formal M2 specification for the `0.0.1` protocol; it is not a
freeze record. M3 falsification and later lifecycle stages remain separate.
Confidence intervals and hard acceptance thresholds are not part of this
protocol.

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

## Headline estimand

For original instance \(x\), code state \(c\), and budget \(T\), let

\[
R(c,x,T)
=
Q_{0.75}\left(\{g(c,x,\rho_j;T)\}_{j=1}^{30}\right)
-
Q_{0.25}\left(\{g(c,x,\rho_j;T)\}_{j=1}^{30}\right),
\]

where \(g\) is the normalized gap from a valid, independently verified run and
\(\rho_1,\ldots,\rho_{30}\) are the declared customer relabelings for \(x\).
The sample quantiles use the Hyndman–Fan Type 7 convention. Smaller
\(R(c,x,T)\) means less central dispersion across equivalent representations.

The point estimate is the equal-weight mean over the ten-instance panel:

\[
\overline{R}_{c,T}
=
\frac{1}{10}\sum_{x \in \mathcal{X}} R(c,x,T),
\]

where \(\mathcal{X}\) is the fixed panel of ten original instances. Gap
outcomes from different instances are not pooled, and the fixed solver seed is
not averaged over.

The representation-robustness change induced by the patch is

\[
\Delta R_T
=
\overline{R}_{\mathrm{Agent},T}
-
\overline{R}_{\mathrm{Base},T}.
\]

Negative \(\Delta R_T\) means that Agent has lower central representation
dispersion; positive \(\Delta R_T\) means higher central representation
dispersion.

## Outcome retention and completeness rules

The following rules apply to the formal IQR estimand.

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
comparison and its \(\Delta R_T\) require both corresponding summaries to be
available.

Retain all raw results and failure records. Do not impute missing or invalid gap
values, including with zero or infinity, and do not replace a failed relabeling
with a different relabeling. These rules keep the declared set of observations
fixed.

## Interpretation and separation from Seed Robustness

The experiment examines representation sensitivity conditional on the declared
solver seed. The IQR describes central dispersion across the sampled
relabelings; a smaller IQR means less central dispersion. It does not average
over the solver's seed domain or measure a distance between marginal output
distributions.

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

On 2026-09-18, the user confirmed Type 7 IQR as the headline statistic, the
equal-weight instance mean, the complete-data rules, and
\(\Delta R_T = \overline{R}_{\mathrm{Agent},T} -
\overline{R}_{\mathrm{Base},T}\) as the patch comparison.

M2 is complete for this protocol. M3 falsification, implementation validation,
freeze-candidate review, and protocol freeze remain outstanding and are not
implied by this specification.
