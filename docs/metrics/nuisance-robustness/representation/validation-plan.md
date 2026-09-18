# Representation Robustness falsification plan

## M3 status

This plan covers the approved `0.0.1` Representation Robustness protocol. The
user approved these seven deterministic synthetic cases and an absolute and
relative floating-point tolerance of `1e-12` on 2026-09-18.

The cases test the formal Type 7 IQR estimand, equal-weight instance
aggregation, and the Base/Agent change. They are falsification cases: a failure
means that the implementation does not match the approved specification. They
do not establish that 30 relabelings are universally sufficient, create a
robustness threshold, or validate a confidence interval.

## Acceptance rules

All seven deterministic cases must pass. A mismatch in any expected value,
invariance, sign, or missing-data result keeps M3 incomplete.

Use exact comparison where the construction is exactly representable. Where
the implementation performs floating-point interpolation or aggregation, use
`math.isclose`-equivalent comparison with:

```yaml
relative_tolerance: 1e-12
absolute_tolerance: 1e-12
```

The real PyVRP panel is not an M3 acceptance gate. Its later validation record
must retain planned runs, completed runs, valid normalized gaps, missing and
failed runs, per-instance IQR availability, and Base/Agent/ΔR availability
without introducing a statistical pass threshold.

## Synthetic falsification cases

### 1. Type 7 quantile exactness

Input the sorted normalized-gap outcomes:

```text
0, 1, 2, ..., 29
```

Expected Type 7 values are:

```yaml
q25: 7.25
q75: 21.75
iqr: 14.5
```

This catches nearest-rank quantiles, off-by-one indexing, and incorrect
interpolation.

### 2. Relabeling-order invariance

Compute the same IQR from the same 30 outcomes in original order, reverse order,
and a deterministic random order.

Expected result:

```text
all three IQR values are equal
```

The result must depend on the empirical outcome values, not on generation order
or file order.

### 3. Constant-shift invariance

Compare the outcomes `0, 1, ..., 29` with the same outcomes after adding `5` to
every value.

Expected result:

```yaml
original_iqr: 14.5
shifted_iqr: 14.5
```

This verifies that the headline statistic measures central dispersion rather
than the absolute performance level.

### 4. Single-tail-outlier behavior

Replace the largest value in `0, 1, ..., 29` with `10000`, producing:

```text
0, 1, 2, ..., 28, 10000
```

Expected result:

```yaml
iqr: 14.5
```

The single extreme tail value must not turn the IQR into a max-min or other
worst-case statistic.

### 5. Equal-weight instance aggregation

At the aggregation layer, provide ten already-computed per-instance IQRs:

```text
1, 1, 1, 1, 1, 1, 1, 1, 1, 11
```

Expected result:

```yaml
equal_weight_instance_mean: 2.0
```

The implementation must average the ten instance IQRs with equal weights. It
must not pool the 300 relabeling outcomes and compute one pooled IQR.

### 6. Base/Agent change direction and antisymmetry

Provide ten Base instance IQRs equal to `1` and ten Agent instance IQRs equal to
`3`.

Expected results:

```yaml
base_mean: 1.0
agent_mean: 3.0
delta_representation_robustness: 2.0
```

Swapping Base and Agent must produce `-2.0`. Identical Base and Agent values
must produce `0.0`. The implementation must use
`Agent - Base`, preserve the sign, and not take an absolute value.

### 7. Incomplete-data propagation

Check all of the following cases:

1. One instance has only 29 valid relabeling outcomes: that instance's IQR is
   `null`.
2. Nine instances are complete and one instance is incomplete: the overall
   ten-instance mean is `null`.
3. Agent is complete while Base is incomplete: Agent's own mean remains
   available, Base's mean is unavailable, and `ΔR` is unavailable.

No missing or invalid outcome may be imputed with zero, infinity, or the mean of
the available outcomes. No replacement relabeling may be introduced.

## Existing transformation-validity evidence

Transformation legality is a prerequisite for interpreting the statistic. The
existing representation tests provide evidence for this boundary and should be
referenced by the M3 validation record rather than duplicated as new estimator
cases. The evidence covers:

- reproducible, distinct, non-identity customer permutations;
- a fixed depot and bijective customer mapping;
- coordinates and demands permuted together;
- mapped solutions retaining feasibility and objective value;
- retention of failed runs and the configured 1200-run relabeling grid.

These checks do not replace the seven synthetic estimator cases. They establish
that the collected inputs satisfy the approved equivalent-representation
contract before the IQR is interpreted.

## Explicit non-gates

M3 does not require:

- a proof that 30 relabelings are sufficient for every solver or instance;
- a robustness pass threshold for `R` or `ΔR`;
- confidence-interval coverage or bootstrap calibration;
- exact equality of Base and Agent on the real empty-patch panel; or
- a worst-case, max-min, or tail-probability replacement for the IQR.

Those choices are outside the approved `0.0.1` protocol or belong to later
validation evidence.
