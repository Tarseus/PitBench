# Performance validation results

The dated sections below preserve the evidence available at each stage. The
2026-10-15 full-panel calibration at the end supersedes earlier statements
that a revised panel had not yet run.

The CP-SAT control below used eight workers in one solver process. The current
single-worker task configuration was introduced later and has not been validated
by this control. Its Base/Agent results require a new real panel before they can
be compared under the revised execution setting.

## CP-SAT exact-solving M5 control

Control artifact: `private/performance_validation/ortools_cp_sat_solve_v9_15_m5_noop_full_no_base_cache/`.

The no-op candidate patch was empty and BASE caching was disabled. The judge produced
all 360 expected observations; every observation completed and was valid. There were
no Qualification failures, infrastructure-missing observations, or BKS discrepancies.

At the primary 30-second budget, BASE solved 39/90 runs (coverage 0.4333333) with
PAR-2 mean 36.8595862 seconds; Agent solved 39/90 runs (coverage 0.4333333) with
PAR-2 mean 37.0067114 seconds. The paired mean penalized-runtime reduction was
-0.1471252 seconds, with 95% percentile bootstrap interval [-1.53583565,
1.05933792], so the control comparison is inconclusive.

At 10 seconds, the one-run, eight-worker control produced 27/90 BASE and 28/90
Agent solves; its paired confidence interval includes zero. This is recorded as the
parallel CP-SAT control's one-run noise boundary, not as an improvement.

Conclusion: the M5 control evidence passes. This result does not advance the
protocol to M6 or M7.

## Choco replacement panel integration check, 2026-10-14

The replacement retains 5- and 20-second solver budgets, 30 declared solver
seeds, one solver worker per run, five public development instances, and 30
hidden judge instances. The selection record is retained at
`private/performance_validation/choco_panel_20261014_screening/manifest.json`.
Standalone BASE screening used seed 0 only: 3/5 development instances and 18/30
judge instances reported optimal termination at each budget. This shows that
neither budget has full optimal coverage in the screened seed; it is not a
full-seed coverage estimate.

All 35 fixed targets were regenerated with a separate OR-Tools CP-SAT 9.15
process configured for one worker and seed 0. Each run returned `OPTIMAL` with
an objective equal to its reported bound within a 10-second proof-preparation
budget. The recorded reference packings passed the independent bin-packing
feasibility and objective check. This evidence trusts the separate solver's
optimal-bound claim; it is not a replayable mathematical certificate.

The empty-patch `dry-run` at `runs/choco-panel-20261014-smoke/dry-run/choco_v6_0_1/`
completed 840/840 observations with valid solutions and no recorded errors. It
included one development instance, one judge instance, and the configured
reliability cases, all with the declared 30-seed and two-budget grid. A separate
one-seed `DockerJudge` run at `runs/choco-panel-20261014-hard-check/` completed
32 observations and checked hidden `judge_id_0005`: at both budgets its solution
was independently valid, while termination remained `time_limit`. This is an
unsolved Performance outcome, not a verifier or infrastructure failure.

After rebuilding the pinned JVM images for the changed adapter source, both
Choco and OR-Tools Configuration Robustness collector probes passed. The new
Choco judge image `sha256:c5d1c94253debad339469eb538275e1700d2ab408575cd28896545368a4a4dff`
completed a separate six-observation, one-seed Base-only integration check with
all solutions valid and no errors. Directly running that final Choco image on
hidden `judge_id_0005` with evaluation seed `407464839` returned `time_limit`
at both 5 and 20 seconds, so both budgets retain at least one unsolved run.
The new OR-Tools image completed its public
`la01` solve with one worker and a valid optimal result. Replaying all 35
proof-preparation runs in that final OR-Tools image reproduced the retained
proof records byte-for-byte. These checks do not
replace a complete new-image Performance panel.

These are integration and screening checks. A complete 30-instance, 30-seed
Base/Agent real panel under the replacement oracle has not run. The revised
oracle evidence boundary and full-panel result remain freeze blockers.

## HiGHS replacement panel integration check, 2026-10-14

The former 20/40/80-job judge panel produced zero verified optimal runs at both
10 and 30 seconds in the retained full evaluations. The replacement keeps those
budgets, the 30 solver seeds, one worker per process, five public development
instances, and 30 hidden judge instances. It generates 9/10/11-job scheduling
models from consecutive instance seeds. Trusted optima still follow the exact
shortest-processing-time argument and are bound to every generated instance
hash. The 35 source models, screening results, and raw solver logs are retained
under `private/performance_validation/highs_panel_20261014_screening/`.

Standalone BASE screening with solver seed 0 found 9/30 judge optima at 10
seconds and 13/30 at 30 seconds. The public development panel had 2/5 and
3/5 respectively. These are selection diagnostics, not full-seed performance
estimates.

The first empty-patch end-to-end run exposed 56 false invalid classifications:
HiGHS reported optimal schedules with integer-variable text deviations up to
`3.8e-12`, while the verifier required exact decimal
integers. The pinned HiGHS 1.15.1 source declares a default MIP feasibility
tolerance of `1e-6` in `highs/lp_data/HConst.h`. The verifier now accepts raw
integer variables and raw objective reports only within that tolerance of the
nearest exact integer values, then recomputes all scheduling constraints and
the objective with integers. Replaying all 56 previously invalid solution files
with the corrected verifier produced 56 valid results.

The corrected empty-patch run at
`runs/highs-panel-20261014-smoke-v2/dry-run/highs_v1_15_1/` completed all
720 observations: 706 completed with valid solutions, 14 had ordinary
solver-origin `no_solution` outcomes, and none had semantic-invalid or
infrastructure-missing status. Its Performance report was complete. A further
one-seed evaluator-owned judge check at `runs/highs-panel-20261014-hard-check/`
covered the first 9/10/11-job cases and produced nine completed and nine
ordinary unsolved observations, with no verification errors.

The formal `pitbench evaluate ... --check-only` preflight found a digest-only
HiGHS agent image reference that could not be used in a generated Dockerfile
`FROM`. The fixed source image was tagged for agent use; the judge remains
pinned to its image ID. Repeating the preflight then passed the private assets,
seed protocol, agent and judge images, all configured collectors, and Codex
runtime. In the prepared agent container, `pitbench inspect` exposed all five
public instances and their exact targets. A real `pitbench bench` run returned
independently verified solved feedback for `dev_0000` and an ordinary unsolved
PAR-2 preview for `dev_0001` at 10 seconds.

The complete replacement 30-instance, 30-seed Base/Agent panel has not run.
Automatic BASE cache paths now include the task-configuration and effective
judge-image hashes; the former HiGHS panel's cache cannot be reused by the new
task configuration. The HiGHS revised-panel freeze blocker remains open.

## Seven-task same-code noise calibration, 2026-10-15

The [calibration record](../../evaluation-noise-calibration.md) and
`report/noise-calibration-20261015/analysis.json` retain the complete raw-run
provenance. One full empty-patch Base/Agent panel was run for each current task:
PyVRP 0.14 (8,160 observations), VROOM 1.15 (228), Choco bin packing (4,200),
Choco graph coloring (120), HiGHS job-shop (4,200), HiGHS multidimensional
knapsack (600), and CP-SAT (4,200). All **21,708** expected observations arrived.
There were no semantic-invalid or infrastructure-error observations. The
HiGHS job-shop panel had 2,839 ordinary `no_solution` outcomes, and knapsack
had 419; they remain solver-origin unsolved outcomes, not missing data.

The former raw-coverage rule would have called three identical-code exact
controls `regressed`: CP-SAT 125/900 versus 124/900, HiGHS job-shop 358/900
versus 357/900, and knapsack 81/150 versus 80/150. Each net loss was confined
to one instance/seed pair. Their paired instance-coverage intervals include
zero, so the revised coverage-first rule reports all three as `inconclusive`.
Both Choco controls have equal Base/Agent coverage and inconclusive PAR-2.

PyVRP exposed a separate heuristic failure: its full empty-patch first run
had paired normalized-gap reduction `0.0000259325` with 95% instance interval
`[0.00000174794, 0.0000577537]` and was falsely called `improved`. A complete
fresh run with Base/Agent build and job order reversed had reduction
`0.00000508319`, interval `[-0.0000180945, 0.0000267946]`, and was
`inconclusive`. The between-order interval includes zero, so this evidence does
not establish order as the sole cause. It does establish that the one-run
positive direction failed an independent repeat. The new full-evaluation rule
requires both complete rounds to agree on direction and therefore makes this
public result `inconclusive`. The two runs are retained separately rather than
counted as extra seeds.

As a power check, two nonempty historical VROOM patches were replayed under
the newly aligned source image. The GPT-6 Sol xhigh patch improved paired gap
by `0.00110990` and `0.00195580` in its two opposite-order full rounds, with
both 95% intervals entirely positive; its final label remains `improved`.
The GPT-6 Luna medium patch changed paired gap by `-0.0162755` and
`-0.0161837`, with both intervals entirely negative; its final label remains
`regressed`. All four 228-observation rounds completed with valid solutions.
This checks that confirmation does not erase these clear real directions.
The two-round checkpoint identities, intervals, and artifact paths are retained
in `report/noise-calibration-20261015/signal-controls.json`.

The two HiGHS no-op controls had negative PAR-2 intervals excluding zero
despite identical code. Their unresolved coverage differences prevent the
secondary time rule from issuing a label in these runs. A full reverse-order
knapsack no-op panel changed the one-run coverage difference from -1/150 to
+1/150 on the same instance and changed mean paired PAR-2 from -0.145383 to
+0.141812 seconds. Its first round lacks a mounted-harness fingerprint, so
order is a plausible but not uniquely established cause. In the retained
job-shop no-op data, excluding the sole coverage-changing instance leaves
equal coverage but a negative paired PAR-2 interval
`[-0.0147025, -0.00287467]` seconds. That 29-instance subset is a diagnostic,
not an official panel; it demonstrates that the one-run secondary time rule
can falsely direct identical code under the older harness. A separate pair of
full job-shop empty-patch runs now has an identical recorded harness SHA and
opposite Base/Agent order. The forward run had 357/900 versus 358/900 and
paired PAR-2 interval `[-0.0140556, +0.0929509]` seconds; the reverse run had
357/900 on both sides and interval `[-0.00249949, +0.0119136]` seconds. Both
were `inconclusive`, with no false public direction. The two-round data are
retained in
`report/noise-calibration-20261015/highs-jobshop-matched-order-diagnostic.json`.
The new confirmation gate protects future public time-only directions. A few
null panels cannot estimate a universal false-positive probability or freeze
the metric.
