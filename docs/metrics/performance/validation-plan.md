# Performance falsification plan

## M3 status

This plan records the approved minimum counterexamples for the
`heuristic_fixed_budget` and `exact_verified_solve` protocols. Each case is intended to expose an
attractive but incorrect implementation or interpretation.

## Heuristic fixed-budget cases

### Budget direction reversal

Construct a complete paired panel in which Agent has positive paired gap reduction at
a shorter diagnostic budget and negative paired gap reduction at the declared primary
budget.

Expected result: retain both budget reports and classify the candidate as `regressed`
from the primary-budget interval. Pooling budgets, selecting the most favorable budget,
or automatically selecting the largest budget fails the case.

### Base/Agent pairing antisymmetry

Construct a complete paired panel with a nonzero paired gap reduction, then exchange
the Base and Agent identities without changing any other observation.

Expected result: Base and Agent summaries exchange positions, the paired gain changes
sign with unchanged magnitude, and `improved` and `regressed` classifications exchange.
Failure to pair on the same instance and solver seed fails the case.

### Unequal valid-seed counts

Construct two instances with different counts of valid solver-seed observations. Give
the instances distinct seed means so that pooling every seed observation would differ
from the equal-weight mean of the two instance means.

Expected result: each instance contributes one equal-weight instance mean. The instance
with more valid seeds must not receive more weight. Invalid and missing seeds remain in
the retained completeness records.

### Bootstrap target validation

Construct a panel whose per-instance valid-seed means are distinct and whose raw seed
observations are unbalanced across instances.

Expected result: every bootstrap draw resamples the fixed per-instance seed means. A
bootstrap that resamples raw instance-seed rows or introduces a second seed-resampling
level fails the case.

### Semantic-invalid adversarial case

Construct an Agent observation with a numerically better objective and normalized gap
than Base, but whose returned solution fails independent semantic verification.

Expected result: Qualification fails. The invalid Agent observation cannot support an
`improved` Performance classification.

### BKS-beating case

Construct a valid minimizing observation whose independently verified objective is
strictly better than its fixed best-known-solution anchor.

Expected result: normalized gap is negative and is retained without clipping. Replacing
the negative gap with zero or rejecting it solely because it beats the anchor fails the
case.

### Primary-budget versioning

Use one fixed set of budget-stratified observations for two task configurations whose
explicit primary budgets select opposite paired-gain directions.

Expected result: each configuration classifies from its declared primary budget and
retains the other budget as diagnostic. Changing the primary budget without changing
the recorded task protocol identity fails the case.

## Exact-optimization cases

### Solved coverage dominates speed

Construct a complete panel in which Agent has lower penalized runtime on its verified
solved runs but lower verified-solved coverage than Base.

Expected result: classify the candidate as `regressed` from solved coverage. Faster
solved runs or a favorable PAR-2 aggregate cannot mask the lost solved coverage.

### Equal coverage uses penalized runtime

Construct a complete panel with equal Base and Agent verified-solved coverage and a
paired penalized-runtime interval that is entirely positive in the Base-minus-Agent
direction.

Expected result: classify the candidate as `improved` from the secondary PAR-2 rule.
Applying the PAR-2 rule before verifying equal coverage fails the case.

### Optimum incumbent without optimal termination

Construct an executed Agent run whose independently verified incumbent equals the
trusted optimum but whose normalized termination is not optimal.

Expected result: the run is solver-origin unsolved, its solved indicator is zero, and
its penalized-runtime outcome is `2T`.

### False optimality claim

Construct an Agent run that reports optimal termination but returns an invalid solution,
an objective inconsistent with the solution, or a verified objective different from the
trusted optimum.

Expected result: Qualification fails. Converting the result to an ordinary unsolved
`2T` outcome fails the case.

### Published BKS above target

Construct a published-BKS exact run that reports optimal termination and has an
independently verified objective strictly above its fixed BKS.

Expected result: the run is solver-origin unsolved and receives `2T`. Classifying it as
a Qualification failure fails the case.

### Published BKS below target

Construct a published-BKS exact run that reports optimal termination and has an
independently verified objective strictly below its fixed BKS.

Expected result: retain the verified result as `oracle_discrepancy` and make the
corresponding exact aggregate `incomplete`. Treating it as a Qualification failure or
ordinary `2T` outcome fails the case.

### Solver timeout versus infrastructure missingness

Construct one launched run that reaches its solver time limit and one required run that
was not launched or whose evaluator infrastructure failed.

Expected result: the solver timeout receives `2T`; the infrastructure failure has no
ordinary outcome and makes the required aggregate `incomplete`.

## Required retention

Every validation case retains its complete synthetic inputs, expected grid, computed
details, failure or missingness records, and random seeds. Passing these cases does not
authorize a lifecycle transition without the separately approved real-panel validation.
