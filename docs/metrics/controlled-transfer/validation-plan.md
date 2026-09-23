# Controlled Transfer falsification plan

## M3 status

This plan records the approved synthetic counterexamples for Controlled Transfer.
Each case tests a definition in [spec.md](spec.md). A passing synthetic case does not
validate a real panel, select a future panel, or advance the metric lifecycle.

## 1. No-op reference benefit

Construct complete paired reference and target conditions in which every Base-Agent
paired difference is zero.

Expected result: \(G_h(c_0)=0\); the reference improvement is unestablished and the
retention ratio is `null`. Treating no change as full retention fails the case.

## 2. Strict 80% retention boundary

Use complete, zero-variance synthetic conditions with \(G_h(c_0)=1\). First set the
target benefit to \(0.8\), then set it to \(0.81\).

Expected result: the first condition has \(M_h=0\) and does not support retention
because the lower bound must be strictly positive. The second has \(M_h=0.01\) and
supports retention. Replacing the strict inequality with \(M_h\geq0\), or applying a
retention fraction other than 0.8, fails the case.

## 3. Positive target benefit with a zero reference benefit

Construct a complete reference condition with \(G_h(c_0)=0\) and a target condition
with positive benefit.

Expected result: report the target benefit, but report
`reference_improvement_unestablished` and a `null` retention ratio. Adding an
implicit denominator epsilon fails the case.

## 4. A single structural-domain regression remains visible

Construct three complete matching structural target conditions with a positive
reference benefit. Give two target conditions positive benefits and one target
condition a negative benefit.

Expected result: all three condition reports are retained and the negative-benefit
domain is reported as a regression relative to Base. Averaging domains into a positive
summary that removes the negative condition fails the case.

## 5. Continuous scale support stops at the first failure

Construct an ordered scale path with a positive reference benefit and three complete
target conditions. Let the first and third have positive simultaneous retention
margins, and let the second not support retention.

Expected result: the largest continuously supported tested scale is the first target
scale. A later recovered target condition cannot bridge the failed intermediate scale.

## 6. Exact solved coverage precedes penalized runtime

Construct one complete exact panel where Agent has lower solved coverage than Base but
a faster penalized-runtime result, and another where Base and Agent have equal solved
coverage with a positive paired penalized-runtime interval.

Expected result: the first panel is `regressed` from coverage; the faster time cannot
override it. The second is `improved` from paired penalized runtime. Applying the
time rule before the coverage comparison fails the case.

## 7. Completeness and exact solver-origin unsolved outcomes

Construct a condition with a required missing run, an infrastructure-error run, or a
run without a usable task-native Performance result. Separately construct an exact
solver-origin unsolved run.

Expected result: each of the first group makes its Controlled Transfer condition
`incomplete`; none may be silently dropped. The exact solver-origin unsolved run
remains an ordinary \(2b\) outcome, not `incomplete`. Converting it to missingness
fails the case.

## 8. Joint resampling and Bonferroni bounds

Construct a claim family with one reference benefit and multiple target conditions.
Retain the bootstrap replicates needed to recompute the shared reference benefit and
every target margin in each replicate.

Expected result: the joint calculation recomputes the shared reference benefit in
every replicate and uses one-sided lower-tail probability \(0.05/m\) for a family of
\(m\) required bounds. Reusing a fixed point reference estimate, or substituting
ordinary per-condition 95% lower bounds for the Bonferroni bounds, fails the case.

## Retained synthetic evidence

Every case retains its complete planned run grid, synthetic observations, task-native
outcomes, bootstrap inputs, and expected result. Qualification failures and oracle
discrepancies retain their existing task-native meanings; this plan does not convert
them into ordinary finite benefits.
