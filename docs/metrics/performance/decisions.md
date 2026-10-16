# Performance decisions

## Instance bootstrap over per-instance seed means

- The Performance confidence interval expresses uncertainty across instances only.
  Seed sensitivity belongs to Nuisance Robustness.
- For Base and Agent gap summaries, average valid seed results within each instance.
  Compute mean, median, p95, and the confidence interval from those per-instance seed
  means, with every included instance receiving equal weight.
- For paired gain, pair Base and Agent by solver seed, average paired gains within each
  instance, and give every included instance equal weight.
- Exclude an instance with no valid seed mean from gap aggregation while retaining its
  failure records.
- Use a 95% percentile interval with 5000 resamples and fixed bootstrap seed
  `20260824`.
- Report the method as `instance bootstrap over per-instance seed means`.

## Generalization boundary

- Performance uses only original `judge_id` observations: observations with no
  equivalence parent and no separate test-suite identity.
- `judge_shift` observations remain retained in the evaluator output, but they do
  not contribute to the Performance estimand or classification.
- Distributional Generalization may consume `judge_shift` observations when that
  dimension is developed. It is not a dependency of the Performance protocol.

## 2026-10-15: noise-calibrated public directions

- A one-run exact coverage difference is reported with a paired instance
  interval. It has a public direction only when that interval excludes zero;
  otherwise the public result is `inconclusive`, and PAR-2 does not override
  unresolved coverage.
- A directional result from any full real Performance panel is provisional
  until a fresh, complete Base/Agent repeat in reverse build and job order has
  the same direction. Discordant directions or an inconclusive repeat produce
  an `inconclusive` final result. An incomplete repeat produces `incomplete`.
- Independent repeats are preserved as separate observations and reports;
  they are not treated as additional solver seeds in one instance bootstrap.
- Final evaluations measure Base afresh by default. Explicit Base-cache reuse
  can produce exploratory estimates but does not support an unconfirmed
  directional public label. Source repository and image build caches remain
  available.

These decisions follow the seven-task empty-patch audit and the real PyVRP
same-code false positive documented in
[validation-results.md](validation-results.md). The protocol remains a freeze
candidate pending further repeated-null evidence, especially for the exact
secondary PAR-2 branch.
