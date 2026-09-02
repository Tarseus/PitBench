# Nuisance Robustness evidence

## Seed Robustness v1

The literature does not define a single accepted formula for Seed Robustness.
Instead, stochastic-algorithm benchmarking treats repeated independent outcomes as
samples from a distribution and reports summaries or empirical distributions chosen
for the scientific question.

## Domain practice

PyVRP runs each benchmark instance ten times with different seeds, averages the ten
objectives for the instance, and compares the result with the best-known solution.
Latorre's Hybrid Genetic Search study likewise runs each instance ten times with
varying random seeds and reports the average solution value, best solution, and BKS.
These protocols estimate expected performance; they do not define a robustness
scalar.

Birattari and Dorigo reject best-of-N as a replacement for a central statistic when
assessing a stochastic algorithm. Their result does not select variance, IQR, MAD, or
another statistic as a Seed Robustness definition.

Hoos and Stützle study stochastic-search behaviour through empirical run-time
distributions. Their results, together with Hoos's report of heavy-tailed or
multimodal behaviour, support retaining the complete empirical distribution instead
of assuming that a single spread statistic captures distribution shape.

COCO reports empirical cumulative distribution functions for target-attainment
runtimes. IOHprofiler also provides ECDF analysis of fixed-budget outcome values.
These systems provide direct methodological support for retaining and reporting
empirical distributions, while COCO's primary estimand remains fixed-target runtime
rather than fixed-budget VRP gap.

Ivković, Kudelić, and Črepinšek recommend quantiles for peak, typical, and bad-case
performance and discuss bootstrap confidence intervals. This supports quantile-based
reporting but does not establish IQR as the definition of Seed Robustness.

Published stochastic-optimization experiments use IQR as a dispersion or consistency
summary over independent runs. This is application precedent, not a VRP benchmarking
standard.

Lodi and Tramontani recommend multiple random seeds and robust indicators such as
truncated averages and rank statistics when random perturbations are material. This
supports avoiding reliance on mean or variance alone, but does not specifically
select IQR or MAD. Their MIP setting is methodological supporting evidence only and
is outside the v1 solver scope.

Campelo and Wanner formulate algorithm comparison over a population of problem
instances. Repeated runs improve estimation on an instance, while cross-instance
inference and sample-size design remain centred on the number of instances.

## Candidate evidence assessment

### Empirical distribution and ECDF

The complete per-instance empirical seed distribution has the strongest support as
the retained evidence object. An ECDF preserves distribution shape and can expose
tails or multiple search regimes, but it is not a single headline scalar.

### Interquartile range

IQR directly represents the spread of the central half of the seed outcomes and has
application precedent in stochastic-optimization experiments. It is invariant to an
overall shift in gap and therefore describes dispersion separately from performance
level. It is not an established VRP Seed Robustness convention and does not represent
the outer tails.

### Median absolute deviation

MAD is also invariant to an overall shift in gap, but the reviewed domain evidence
does not directly support it as a stochastic-optimization benchmark convention. It
can be zero when a majority of seeds share one outcome regime even if a minority of
seeds has substantially worse outcomes, so it may miss a bad-seed regime.

### Tail probability

A probability of exceeding a quality threshold has strong support from
target-attainment and fixed-probability benchmarking. It measures threshold risk,
not pure seed-induced dispersion: a solver that is stable but consistently worse
than the threshold has high tail probability. Similarly, the probability that Agent
is worse than Base measures patch-regression risk rather than the spread of either
code state.

## M1 conclusions and boundaries

1. There is no accepted single Seed Robustness formula in the reviewed literature.
2. The complete per-instance empirical seed distribution or ECDF is the
   best-supported foundational evidence object.
3. IQR has stochastic-optimization dispersion precedent but is not a VRP standard.
4. MAD lacks direct benchmark support in the reviewed domain and may miss a minority
   bad-seed regime.
5. Tail probability has strong benchmark support but is a threshold-risk estimand,
   not a spread estimand.
6. M1 does not select a headline estimator, threshold, confidence interval, seed
   count, or cross-instance aggregation.
7. PitBench currently configures five seeds for PyVRP, fewer than the ten seeds used
   by PyVRP's public benchmark. A formal seed count must be chosen later based on the
   required estimation precision.

## Sources

- PyVRP, "Benchmarking": https://pyvrp.org/dev/benchmarking.html
- V. Latorre, "A hybrid genetic search based approach for the generalized vehicle
  routing problem," *Soft Computing* 29 (2025):
  https://doi.org/10.1007/s00500-025-10507-0
- M. Birattari and M. Dorigo, "How to assess and report the performance of a
  stochastic algorithm on a benchmark problem: mean or best result on a number of
  runs?" *Optimization Letters* 1 (2007):
  https://doi.org/10.1007/s11590-006-0011-8
- H. H. Hoos and T. Stützle, "Towards a characterisation of the behaviour of
  stochastic local search algorithms for SAT," *Artificial Intelligence* 112
  (1999): https://doi.org/10.1016/S0004-3702(99)00048-X
- H. H. Hoos, "Heavy-Tailed Behaviour in Randomised Systematic Search Algorithms for
  SAT?" UBC Technical Report TR-99-16 (1999):
  https://www.cs.ubc.ca/tr/1999/tr-99-16
- COCO, "Performance Assessment":
  https://numbbo.github.io/coco-doc/perf-assessment/
- IOHprofiler, "Real-valued Black-Box Optimization":
  https://iohprofiler.github.io/Background
- N. Ivković, R. Kudelić, and M. Črepinšek, "Probability and Certainty in the
  Performance of Evolutionary and Swarm Optimization Algorithms," *Mathematics* 10
  (2022): https://doi.org/10.3390/math10224364
- A. Lodi and A. Tramontani, "Performance Variability in Mixed-Integer Programming,"
  *TutORials in Operations Research* (2013):
  https://doi.org/10.1287/educ.2013.0112
- F. Campelo and E. F. Wanner, "Sample size calculations for the experimental
  comparison of multiple algorithms on multiple problem instances," *Journal of
  Heuristics* 26 (2020): https://doi.org/10.1007/s10732-020-09454-w
- "Self-learning salp swarm algorithm for global optimization and its application in
  multi-layer perceptron model training," *Scientific Reports* (2024):
  https://doi.org/10.1038/s41598-024-77440-4
