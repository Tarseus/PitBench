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

## Deferred Generalization replacement

- Keep the hidden-shift acceptance logic in `PerformanceDecision` temporarily.
- Do not treat that logic as part of Frozen Performance semantics.
- Distributional Generalization must formally replace and take ownership of the
  hidden-shift acceptance logic when that metric is developed.
