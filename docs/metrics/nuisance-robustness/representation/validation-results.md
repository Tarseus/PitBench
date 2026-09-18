# Representation Robustness M5 validation results

## M5 outcome

M5 validation was completed with user approval on 2026-09-18. The approved
synthetic falsification cases pass, and the retained PyVRP panel is complete
under the `0.0.1` protocol. This closes M5 validation for Representation
Robustness; it does not freeze the metric or create a statistical pass
threshold.

## Synthetic validation

The seven approved deterministic falsification cases pass:

- Type 7 quantile exactness;
- relabeling-order invariance;
- constant-shift invariance;
- single-tail-outlier behavior;
- equal-weight instance aggregation;
- Base/Agent change direction and antisymmetry; and
- incomplete-data propagation.

The transformation-validity evidence also passes through the existing
representation tests. It covers reproducible non-identity permutations, fixed
depot and bijective mappings, synchronized coordinates and demands, mapped
solution feasibility and objective preservation, failed-run retention, and the
configured 1200-run grid.

The implementation test evidence was:

```text
tests/unit/pitbench/test_metric_reports.py: 72 passed
tests/unit/pitbench/test_representation_robustness.py: 10 passed, 15 skipped
tests/unit/pitbench/test_evaluation.py -k 'nuisance or representation': 2 passed, 73 deselected
```

Ruff and `git diff --check` also passed.

## Real PyVRP panel

Primary artifact:

```text
private/representation_robustness/pyvrp_v0_14_0_20260907/results/
```

Protocol identity:

- task: `pyvrp_v0_14_0`;
- source commit: `5d9776a954b810bdb3fe71d47b1e7de7cffe90d2`;
- ten fixed development instances;
- solver seed `0`;
- generation seed `20260907`;
- 30 distinct non-identity customer relabelings per instance;
- budgets of 5 and 10 seconds;
- Base and Agent code states; and
- empty candidate patch.

The panel contains 1,200 planned relabeling runs. All 1,200 completed, all
1,200 produced valid observations, all 1,200 passed mapped feasibility, and all
1,200 preserved the mapped objective. The raw `results.jsonl` was not changed;
the formal report was recomputed with the M4 estimator at:

```text
private/representation_robustness/pyvrp_v0_14_0_20260907/results/report/
```

## Descriptive estimates

At 5 seconds:

```yaml
base_mean_iqr: 0.003240069156058914
agent_mean_iqr: 0.0029563106735945185
agent_minus_base: -0.00028375848246439553
base_complete_instances: 10
agent_complete_instances: 10
```

At 10 seconds:

```yaml
base_mean_iqr: 0.002710013392580523
agent_mean_iqr: 0.002787703661442877
agent_minus_base: 0.00007769026886235387
base_complete_instances: 10
agent_complete_instances: 10
```

The 5-second result has a lower Agent IQR and the 10-second result has a higher
Agent IQR. Because both code states used an empty patch, these values are
descriptive repeated-run observations, not evidence that a patch improved or
regressed representation robustness. The protocol does not include confidence
intervals or a hard robustness threshold, so this validation does not claim
that either observed difference is statistically significant.

## Accepted boundaries

This validation concerns the fixed PyVRP 0.14.0 source, ten-instance panel,
solver seed `0`, 30 generated relabelings, and 5/10-second budgets. It does not
establish that 30 relabelings are sufficient for every solver or instance, does
not generalize to other representation transformations, and does not validate
a patch effect.

The M6 freeze-candidate review and M7 protocol freeze are recorded in
[audit.md](audit.md) and [freeze.md](freeze.md). The frozen implementation
commit is `22d19065`.
