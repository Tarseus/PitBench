# Nuisance Robustness falsification plan

## Current M5 decision (2026-09-07)

The user retained 30 seeds and Type 7 IQR, with an equal-weight mean over the fixed
paired-complete instances. Confidence intervals now resample shared seed columns
only, preserving Base/Agent pairing and holding the instances fixed. The nominal
level remains 99%, with 5000 resamples and bootstrap seed `20260824`.

M5 records bias, empirical coverage, and mean interval width descriptively. There is
no hard acceptance threshold for this cycle. The former 95% minimum below belongs
to the historical M3 design and does not gate the current M5 results. Validation of
nonzero robustness-difference detection is deferred, not a completion requirement.

The sections below retain the original M3 model, parameters, and results for
traceability. Their crossed-bootstrap results do not validate the new seed-only
intervals. Current real-solver results are recorded in `validation-results.md`.

The user approved M5 closure on 2026-09-07 using the completed PyVRP v0.12.2,
v0.13.0, and v0.13.4 experiments and their fixed-instance reanalysis. The v0.14.0
run remains supplementary evidence and is not required for closure. No additional
statistical acceptance gate is introduced by this decision.

## Historical M3 validation target

Test whether the crossed `instance_set` and `seed_list` percentile bootstrap in
the formal specification provides adequate coverage for the Base mean Seed
Robustness, Agent mean Seed Robustness, and their paired change.

The interval is a nominal 99% interval. The minimum accepted empirical coverage is

```yaml
minimum_empirical_coverage: 0.95
```

Every approved synthetic case must meet this minimum for Base, Agent, and the paired
change. Report empirical coverage, estimator bias, and mean interval width for every
case. Bias and interval width are diagnostics, not separate acceptance gates.

## Fixed simulation parameters

```yaml
instance_count: 30
seed_count: 30
simulation_count: 2000
bootstrap_count: 5000
simulation_seed: 20260902
bootstrap_seed: 20260824
nominal_ci_level: 0.99
ci_lower_quantile: 0.005
ci_upper_quantile: 0.995
```

Each simulation draws one shared `seed_list` and one synthetic set of instance
rows. Within each bootstrap replicate, seed indices are sampled once and applied to
all instance rows. Instance indices are sampled separately. The same crossed indices
are used for Base and Agent. Individual `(instance, seed)` outcomes are never treated
as iid observations.

## Shared synthetic model

For instance \(i\) and seed \(r\), draw

\[
\sigma_i \sim \operatorname{Uniform}(0.005,0.02),
\]

\[
Z_r \sim N(0,1),
\qquad
\varepsilon_{i,r} \sim N(0,1),
\]

and define

\[
Y_{i,r}
=
\frac{Z_r+\varepsilon_{i,r}}{\sqrt{2}}.
\]

The shared \(Z_r\) makes the seed a blocking factor across all instance rows. For
case-specific outcome \(X_{i,r}\), define the paired gaps as

\[
g_{\mathrm{Base},i,r}
=
0.05+\sigma_i X_{i,r},
\]

\[
g_{\mathrm{Agent},i,r}
=
0.04+0.8\sigma_i X_{i,r}.
\]

The Base and Agent outcomes are exactly paired. The Base target is the exact
case-specific IQR multiplied by \(E[\sigma_i]=0.0125\). The Agent target is 0.8 times
the Base target, and the paired-change target is -0.2 times the Base target. Their
coverage results are therefore identical by construction and are not independent
validation observations.

## Approved synthetic cases

### Normal

\[
X_{i,r}=Y_{i,r}.
\]

This is the smooth symmetric baseline.

### Skewed

\[
X_{i,r}=\exp(Y_{i,r}).
\]

This tests a continuous asymmetric seed-outcome distribution.

### Two regime

For every seed, draw one shared

\[
B_r \sim \operatorname{Bernoulli}(0.5),
\]

then define

\[
X_{i,r}=Y_{i,r}+3B_r.
\]

This tests seeds entering two separated search regimes. The same \(B_r\) applies to
all instance rows for seed \(r\).

### Outlier

For every seed, draw one shared \(C_r\) with

\[
P(C_r=-8)=0.05,
\qquad
P(C_r=0)=0.90,
\qquad
P(C_r=8)=0.05,
\]

then define

\[
X_{i,r}=Y_{i,r}+C_r.
\]

This tests rare lucky and unlucky seeds. The same \(C_r\) applies to all instance
rows for seed \(r\).

The exact normal and skewed targets use their analytic quantiles. The two-regime and
outlier targets use numerical inversion of their exact mixture CDFs rather than a
large simulated reference sample.

## Exploratory falsification evidence

These results were obtained from temporary simulations using the approved formulas
and fixed seeds. They are retained as M3 exploratory evidence, not as M5 validation
results.

### Seed-count sensitivity with nominal 95% intervals

The normal case was used to compare candidate seed counts before fixing the formal
seed-list size.

| `seed_count` | relative bias | empirical coverage | mean interval width |
| ---: | ---: | ---: | ---: |
| 10 | -13.39% | 89.75% | 0.01307 |
| 20 | -6.98% | 92.30% | 0.00880 |
| 30 | -4.48% | 94.30% | 0.00756 |

This falsified `seed_count: 10` for the approved IQR and crossed percentile method
and selected `seed_count: 30` for the remaining cases.

### Distribution sensitivity with 30 seeds and nominal 95% intervals

| case | relative bias | empirical coverage | mean interval width |
| --- | ---: | ---: | ---: |
| normal | -4.48% | 94.30% | 0.00756 |
| skewed | -2.31% | 95.75% | 0.01438 |
| two_regime | -5.41% | 92.00% | 0.02044 |
| outlier | -3.13% | 96.00% | 0.01807 |

The two-regime case falsified the nominal 95% interval against the approved 95%
minimum empirical coverage.

### Nominal 99% interval checks

With 30 seeds, the normal case reached 98.90% empirical coverage with a mean interval
width of 0.01007. The two-regime case reached 97.95% empirical coverage with a mean
interval width of 0.02722. The two-regime interval was 33.15% wider than its nominal
95% interval.

The skewed and outlier cases already exceeded 95% empirical coverage with the
narrower nominal 95% intervals. Their nominal 99% intervals contain their nominal
95% intervals, so their empirical coverage cannot fall below the recorded 95.75%
and 96.00%, respectively.

All four approved cases therefore meet `minimum_empirical_coverage: 0.95` under the
formal `seed_count: 30` and nominal 99% interval. This is evidence against the
approved falsification cases; it is not a claim that every possible solver-outcome
distribution has at least 95% coverage.

## M5 real-solver validation

M5 may compute empirical Seed Robustness from fixed real PyVRP instances with known
BKS values. A larger real-solver reference seed list may approximate each
`seed_domain` IQR, after which repeated 30-seed lists can test the formal interval
against that reference.

The existing four-version batch uses PyVRP v0.12.2, v0.13.0, v0.13.4, and v0.14.0,
with 10 fixed instances, 700 reference seeds, and a disjoint pool of 300 test seeds.
It samples 1000 lists of 30 seeds from that test pool. The recorded batch includes
1, 5, and 10 second budgets. Existing 1-second results are retained as historical
diagnostics; the current discussion focuses on 5 and 10 seconds, with 10 seconds as
the primary budget.

The fixed-instance reanalysis reuses those observations and seed lists. It holds
the 10 instance rows fixed and resamples shared seed columns only. Both code states
use the same sampled columns. Each list and budget resets `random.Random(20260824)`
and draws only seed indices for its 5000 replicates.

Real-solver reference results cannot establish exact coverage because the reference
seed list is finite. They complement rather than replace the synthetic cases with known
targets.
