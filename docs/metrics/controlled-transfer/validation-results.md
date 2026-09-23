# Controlled Transfer M5 validation results

## Scope

This M5 result validates the Controlled Transfer report against an immutable real
PyVRP 0.14.0 Base/Agent trial. It covers the reference and structural views at the
10-second primary budget. The panel is a retrospective condition mapping of a trial
whose instance sets, anchors, expected grid, and candidate patch were fixed before
this validation artifact; the Controlled Transfer manifest itself was added later.
It is therefore implementation validation, not a prospectively declared benchmark
claim. It does not validate the scale or joint views on a real panel.

The full panel identity and condition mapping are in
`private/controlled_transfer/pyvrp_v0_14_0_retry5_m5/panel.json`:

- source task config SHA-256: `6f29c4e8c7bd8df230f57fca4e2e3423edf6599bc1a82ea9c7ea6f7874b35cdf`;
- observations SHA-256: `834f3c5819d94fdec27db8e84933ff9e26bb5876d53530657b625ae86138413b`;
- expected-grid SHA-256: `63655554b2feeda3651f01929f17d712e5917a213e858e55e612ec209aa1fa6a`;
- candidate-patch SHA-256: `f8b92329431bdad0de9c992e4acec6281ea14f4eaa65e45a8b3cefae2314bd5d`;
- report SHA-256: `e299747d208a526bc0881a32b9898e5a4d03b9323d685a58d0777399fa38124b`.

## Execution

The report was generated with:

```text
.venv/bin/python scripts/report_controlled_transfer.py \
  --task-config configs/tasks/pyvrp_v0_14_0.yaml \
  --panel-config private/controlled_transfer/pyvrp_v0_14_0_retry5_m5/panel.json \
  --observations runs/codex-gpt-5.6-luna-xhigh-20260920-retry5/pyvrp_v0_14_0/pyvrp_v0_14_0.1-of-1.codex-gpt-5.6-luna-xhigh-20260920-retry5/evaluation/trials.parquet \
  --expected-run-grid runs/codex-gpt-5.6-luna-xhigh-20260920-retry5/pyvrp_v0_14_0/pyvrp_v0_14_0.1-of-1.codex-gpt-5.6-luna-xhigh-20260920-retry5/evaluation/expected-run-grid.json \
  --output private/controlled_transfer/pyvrp_v0_14_0_retry5_m5/report.json
```

## Complete-grid check

At the primary 10-second budget, every planned key was observed exactly once and was
valid:

- reference ID: 2,280 observations = 38 instances × 30 seeds × 2 code states;
- structural shift: 600 observations = 10 instances × 30 seeds × 2 code states.

Both conditions are `complete`; neither has missing, infrastructure, qualification,
or oracle-discrepancy records.

## Report result

The report uses the fixed 95% percentile instance bootstrap with 5,000 resamples and
seed `20260824`.

- Reference ID benefit: `-0.003029429526016964`, 95% interval
  `[-0.0042056480250881175, -0.00196435904081776]`.
- Structural-shift benefit: `-0.0014061812841903905`, 95% interval
  `[-0.0022314636546373313, -0.0006737866814495806]`.

The reference lower bound is not greater than zero. The required outcome is therefore
`reference_improvement_unestablished`: the structural condition retains its actual
negative benefit, while retention ratio and margin are `null`. This is correct
handling of a patch that did not establish reference improvement; it is not evidence
of successful transfer.

## M5 conclusion

The synthetic M3 cases and this complete real Base/Agent panel exercise the schema,
expected-grid completeness gate, task-native paired benefit, fixed bootstrap, and
reference-benefit guard. The real panel did not establish candidate benefit and does
not cover the scale or joint views, so it is not a broad benchmark-performance claim.
