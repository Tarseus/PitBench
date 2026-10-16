# Evaluation noise calibration

This calibration checks whether identical solver code can receive a directional
PitBench conclusion. It follows the [benchmark motivation](benchmark-motivation.md):
the evaluator must distinguish a reliable solver improvement from measurement
variation before comparing its conclusions with ordinary coding-agent tests.

Each task gives a fixed solve budget per instance. For that reason, elapsed
wall time is retained here as a secondary diagnostic for timing and order
effects. The central distinct-value comparison is ordinary build/test outcomes
against independently verified solution quality or optimality at the declared
budget.

## Scope and controls

The active task set contains PyVRP 0.14, VROOM 1.15, Choco bin packing and graph
coloring, HiGHS job-shop scheduling and multidimensional knapsack, and OR-Tools
CP-SAT solving. Historical PyVRP release tasks are excluded. Every control uses
the task's declared hidden instances, budgets, seeds, single-solver-process thread
setting, pinned source and judge image. An empty candidate patch makes Base and
Agent code identical. Automatic Base-cache reuse is disabled so both states are
actually run. The full expected grid, run statuses, solution verification, build
logs, task/image/patch hashes, CPU quota, and judge parallelism are retained.

The primary operational resource setting for the new controls is 12 judge CPUs
and 12 parallel one-thread solver processes, with no other solver evaluation
running concurrently. Existing 10-CPU/10-slot controls are analyzed as a separate
resource stratum. No result from one stratum is pooled into another without
showing the stratum distinction.

The machine-local `config/evaluate.local.yaml` now selects this 12/12 stratum
and disables measured-Base reuse, so a single ordinary evaluation command on
this host uses the calibrated execution setting. Results from another host or
resource stratum must retain their effective CPU and parallel-run identities
and cannot inherit this calibration by name alone.

Before comparing controls, resolve the **effective** judge image, including any
machine-local override, and compare it to the task declaration. During this
calibration, VROOM's task-declared local image ID was absent while the local
evaluation config selected an older image. The older-image control is retained
but excluded from the current-task null analysis. The fixed-commit VROOM
Dockerfile was rebuilt and the task, local evaluator, and configuration search
were aligned to its resulting image ID; this is a new protocol identity. PyVRP's
task declaration named its source image as the judge even though the local
evaluator used a distinct published judge image. The task declaration now uses
that digest-pinned judge image. These corrections do not change solver budgets,
seeds, instance sets, or patch contents.
The newly built VROOM image ID is currently local to this host. Before another
host can claim the same image protocol, the image must be distributed by an
immutable registry digest or rebuilt and separately identified; the Dockerfile's
unversioned apt packages do not guarantee an identical image ID from a later
build.

For every complete no-change comparison, record:

- the number and identity of changed independently verified solutions and
  optimal proofs, grouped by instance before interpreting seed counts;
- the paired heuristic-gap, exact coverage, and PAR-2 estimates and their
  instance-level intervals at every declared budget;
- the current public classification, including false directional labels for
  identical code;
- missing, solver-origin unsolved, qualification-failure, oracle-discrepancy,
  and infrastructure-error counts without treating them as interchangeable;
- repeat-to-repeat variation and run-order/resource sensitivity where repeated
  observations exist.

As an exploratory check, swap the Base/Agent sign of each complete instance
cluster together, repeat 50,000 times with fixed seed `20261015`, and record the
two-sided fraction whose absolute mean is at least the observed absolute mean.
This paired-instance randomization diagnostic is kept separate from the current
public 95% percentile-bootstrap classification until the null controls and
run-order check support a protocol decision.

Any revised decision rule must retain the raw effect and coverage-first priority
for exact tasks. Effects within the observed no-change variation remain
`inconclusive`; incomplete grids remain `incomplete`, and Qualification failures
remain separate. The rule must be generic across solvers and tested against both
no-change controls and known substantive changes. Calibration data may be
task-specific, but the code path and evidence requirements must not be.

When a full same-code control receives a directional label, repeat that task
under the same resource, image, instance, seed, and budget contract with the
Base/Agent job order reversed. A sign change would implicate run-order effects;
repeated directions would instead require a stronger null-bias or repeated-run
analysis. The reversed control is a diagnostic stratum and is not pooled with
the primary schedule without showing the order distinction.

## Evidence before new controls

The full current single-worker CP-SAT empty-patch run at
`runs/evaluation-disagreement-current-20261014/luna-medium/ortools_cp_sat_solve_v9_15/`
has 4,200 complete observations. At the 30-second primary budget, Base proves
125/900 optima and identical Agent code proves 124/900. The present exact rule
calls this `regressed`; the one-run difference occurs in one instance/seed pair.

Three GPT-6 PyVRP 0.14 and three GPT-6 VROOM 1.15 full runs from 2026-10-12/13
each contain an independently executed Base state under the same task hash,
judge image, primary hidden grid, and 10-CPU/10-slot resources within that
solver. Comparing unchanged Base states gives three exploratory pairs per
task. Two of three PyVRP pairs receive a positive heuristic-gap interval under
the current one-run analysis even though the solver code is identical; none of
the three VROOM pairs receives a directional quality label. A fourth,
2026-10-10 run for each solver is excluded from these pairs because its harness
date differs and old checkpoints did not fingerprint the mounted
PitBench/adapter source. Equal task and image hashes alone cannot prove an
unchanged harness across dates. Cross-run pairs diagnose possible batch effects;
they do not replace same-run empty-patch controls. New judge checkpoints
fingerprint the mounted Python code; opt-in Base-cache keys include that fingerprint.
The detailed paired values and provenance are retained in the calibration
report under `report/noise-calibration-20261015/`.

Because reused Base observations are measured in a different evaluation run,
Base caching is now opt-in for final judgments. The ordinary single-command
evaluation measures Base and Agent in the same judge job; image and repository
source caches remain available. The standalone `pitbench judge` command also
lets the evaluator choose a CPU quota matching its automatically selected
parallelism when no CPU override is supplied, avoiding its former 8-CPU quota
with 10 auto-selected one-thread solver slots on a 12-CPU host.

## Completion criteria

The calibration is complete when the seven current tasks have retained null
observations or an explicit unavailable reason, all available independent
repeats have been audited, the public categorical rules have been checked for
false directions against those controls, necessary rule changes and tests are
recorded, and remaining false-positive uncertainty is stated. A single no-op
pass does not establish a universal false-positive rate.

## First full same-code panels, 2026-10-15

The first complete empty-patch panel is available for each of the seven current
tasks. All declared observations arrived; none had a semantic-invalid or
infrastructure-error status. HiGHS job-shop produced 1,361 completed valid
observations and 2,839 ordinary `no_solution` outcomes. Multidimensional
knapsack produced 181 completed valid observations and 419 ordinary
`no_solution` outcomes. These are complete panels, not 100%-solved panels.

- Choco graph coloring: Base and Agent each proved 12/30 optima at 20 seconds;
  `inconclusive`.
- Choco bin packing: both proved 540/900 at 20 seconds; `inconclusive`.
- CP-SAT: Base proved 125/900 and identical Agent code 124/900 at 30 seconds;
  one instance/seed pair changed. The old raw-coverage rule said `regressed`;
  the paired instance interval includes zero and the revised rule says
  `inconclusive`.
- HiGHS multidimensional knapsack: 81/150 versus 80/150 at 20 seconds, with
  one changed instance/seed pair. The revised rule says `inconclusive`.
- HiGHS job-shop: 358/900 versus 357/900 at 30 seconds, with one changed
  instance/seed pair. The revised rule says `inconclusive`.
- VROOM: all 228 observations completed; 8/38 hidden primary-budget pairs
  returned different objective values, but the paired gap interval includes
  zero and the label is `inconclusive`. The earlier control on a different
  image is excluded.
- PyVRP: all 8,160 observations completed; 186/1,140 hidden primary-budget
  pairs returned different objective values. The paired gap reduction is
  `0.0000259325`, with 95% instance-bootstrap interval
  `[0.00000174794, 0.0000577537]`. The current one-run heuristic rule
  **falsely says `improved`** for the empty patch.

The two HiGHS empty-patch panels also have negative paired PAR-2 intervals
that exclude zero despite identical code. Their one-run coverage differences
currently keep the public label `inconclusive`; this is evidence that the
secondary exact-time branch needs its own noise check. A single complete
control per task cannot estimate a false-positive probability.

A full reverse-order knapsack control also completed all 600 observations.
Its one-run coverage changed from the first round's Base 81/150 versus Agent
80/150 to Base 80/150 versus Agent 81/150, on the same hidden instance.
Paired PAR-2 changed from `-0.145383` to `+0.141812` seconds. Both complete
rounds remain `inconclusive` under the revised coverage rule. The first run
predates harness fingerprinting, so task/image/resource identity is verified
but equal mounted harness code cannot be proved retrospectively; order is a
plausible contributor, not a proven sole cause. The detailed comparison is in
`report/noise-calibration-20261015/highs-knapsack-order-diagnostic.json`.

The exact-time risk is not limited to that one changed coverage run. If the
single affected hidden instance is excluded from the retained HiGHS job-shop
no-op data as a diagnostic 29-instance subset, Base and Agent have equal
verified-solved coverage, but the remaining paired PAR-2 mean is still
`-0.00839545` seconds with 95% interval
`[-0.0147025, -0.00287467]`. The old one-run secondary rule would falsely
call identical code `regressed` on that subset. This is a diagnostic subset,
not a new official panel. Its first full run predates the current harness
fingerprint, so a matched-harness order comparison was completed separately.

Under the same recorded harness SHA, the fresh full job-shop forward-order
control had Base 357/900 and Agent 358/900 at 30 seconds, with the one gained
solve on `judge_id_0004`; the reverse-order control had 357/900 on both sides.
The paired PAR-2 reductions were `+0.0240852` seconds with interval
`[-0.0140556, +0.0929509]` and `+0.00388619` seconds with interval
`[-0.00249949, +0.0119136]`, respectively. Both complete panels are
`inconclusive`. Their task, image, patch, harness, CPU quota, parallelism,
and expected-grid identities match, with only Base/Agent order reversed.
The structured comparison is retained at
`report/noise-calibration-20261015/highs-jobshop-matched-order-diagnostic.json`.
This closes the specific unmatched-harness order check; it does not estimate
a long-run false-positive rate from two null runs.

The full reversed-order PyVRP control also completed 8,160/8,160 valid
observations with the same patch, task, image, harness, resources, and expected
grid. Its paired gap reduction was `0.00000508319`, interval
`[-0.0000180945, 0.0000267946]`, and single-run label `inconclusive`.
The first-run minus reversed-run effect was `0.0000208493`, interval
`[-0.00000809114, 0.0000530173]`; order alone is **not** established as the
cause. Averaging the two per-instance effects gives `0.0000155079`, interval
`[-0.00000290651, 0.0000376296]`. The first-run positive label is therefore
not repeatable under an otherwise identical resource contract.

For full real evaluations, a one-run directional Performance label now triggers
one fresh, complete confirmation run with Base/Agent order reversed. The public
label keeps the direction only if the second single-run label agrees. A
non-directional repeat makes the final label `inconclusive`; an incomplete
repeat makes it `incomplete`. Both raw rounds and their separate reports are
retained, rather than treating the repeat as extra solver seeds. This policy
addresses the observed PyVRP false positive and provides a confirmation gate
for small exact PAR-2 directions. It does not claim a precisely estimated
false-positive probability from two controls.

## Real-patch Choco order diagnostic, 2026-10-16

The archived GPT-5.6 Luna xhigh Choco 6.0.1 patch
(`03d346229d66e49bd49a3a1cca46d0d58f9e5cc279111984fd6c100e5e61fd79`) produced
a significant time-only slowdown in the Base-then-Agent run. A complete
order-reversed replay was then run with the same task hash, harness SHA,
judge-image ID, 12 CPU quota, 12 parallel one-thread workers, 8 GB memory,
patch hash, and expected grid. Only `code_states` order changed. Both panels
completed 4,200/4,200 valid observations, and their expected grids and
observation identity sets match exactly.

At the 20-second primary budget, both orders verified 540/900 optima for Base
and Agent. Both exact-task Performance labels are `inconclusive`: forward
paired PAR-2 reduction was −0.00251 seconds
`[−0.00937, +0.00147]`; reversed was −0.00188 seconds
`[−0.00702, +0.00112]`. No paired primary-budget objective values changed.

The forward time-only Base-minus-Agent difference was −8.59 ms
`[−16.50, −3.53]`, indicating Agent slower in that order. Reversed, it was
+1.96 ms `[−6.32, +10.86]`, which is inconclusive. The forward-minus-reversed
order effect was −10.55 ms `[−19.65, −3.32]` across instance clusters. This
supports an order-sensitive timing artifact for this patch and shows why one
directional time estimate should not be treated as a solver-performance
conclusion. Since this is one candidate patch with one run per order, it is a
diagnostic example, not an estimate of the benchmark-wide noise rate. Raw
observations and identity checks are retained under
`report/evaluation-disagreement-calibrated-20261016/choco-gpt5_6-luna-xhigh/`.

## Third PyVRP patch replay, 2026-10-16

The GPT-6 Luna medium PyVRP patch was also replayed under the same current
task, judge image, harness, resources, seeds, budgets, and hidden grid as the
two xhigh PyVRP patches. Forward and reversed Base/Agent orders each completed
8,160/8,160 valid observations; both Base and Agent passed the PyVRP
validation test build in each round.

At the 10-second primary budget, the forward normalized-gap reduction was
`+0.00217` percentage points with interval `[-0.00093, +0.00524]`, so the
one-run classification was `inconclusive`. The reversed reduction was
`+0.00951` points with interval `[+0.00255, +0.01726]`, classified `improved`.
Time-only intervals crossed zero in both rounds: `+2.24 ms
[-4.18, +10.04]` and `+7.24 ms [-4.94, +20.43]`. Because the first round was
already `inconclusive`, the public confirmation flow would stop at that label;
the reversal was run as an additional research diagnostic. The difference
between the two quality classifications shows that a single positive repeat
does not make the result stable. Raw paired observations and identities are
under `report/evaluation-disagreement-calibrated-20261016/gpt-6-luna-medium-pyvrp/`.

## HiGHS matched tests and same-patch performance, 2026-10-16

The GPT-5.6 Luna xhigh HiGHS patch
(`27e6ed2e801d114ae8d596d80070a400af0bacddb9188e2e384e4daba2a45344`) was
checked on Base and Agent trees at source commit
`04024d701f79feb8e2f18bc3df0dffc04ef05088`. Applying the archived patch to a
clean Base worktree produced byte-identical contents for all five changed
files in the Agent test tree.

Both trees passed the full C++ unit test binary: 328 cases and 1,259,125
assertions. Both passed the targeted MIP suite: 37 cases and 303 assertions.
Both also passed a 21-test CTest MIP subset. The first CTest attempt failed
because the `highs-bin` target had not been built; after building that target,
the same 21 tests passed for both trees. These tests ran in local worktrees;
the solver panel ran in the pinned judge image.

The complete forward-order solver panel contains 4,200/4,200 observations.
At the 30-second primary budget, Base verified 349/900 optima and Agent
350/900. The single additional solve came from one hidden instance. The
instance-bootstrap coverage interval includes zero, so Performance is
`inconclusive`; paired PAR-2 is also `inconclusive`. Base-minus-Agent wall time
was −23.24 ms `[−42.46, −6.75]`, indicating slower Agent execution in that
order. This is a test-passing patch for which the matched solver panel does
not establish a generalizable performance improvement.

The order-reversed panel completed 4,200/4,200 observations. Its expected grid
matches the forward grid, all observations match their expected identities,
and there are no missing or duplicate runs. The patch, task, image, harness,
CPU, memory, and parallel-run identities match; Base/Agent order is reversed.
The panel recorded 1,368 `completed` and 2,832 `no_solution` observations,
with no infrastructure-error status.

At 30 seconds, reverse-order Base verified 355/900 optima and Agent 358/900.
The three-solve net difference came from two hidden instances. Coverage was
`+0.00333` with 95% interval `[0, +0.00889]`; paired PAR-2 reduction was
`+0.11801` seconds `[−0.00011, +0.31439]`. Performance remained
`inconclusive`, as it did in the forward order (349/900 versus 350/900).
The time-only Base-minus-Agent difference changed sign: forward was −23.24 ms
`[−42.46, −6.75]`, while reversed was +59.89 ms `[+22.06, +103.29]`.

A duplicate container was briefly started and stopped during validation after
the original live container was found. It wrote no solver observations, but
its one-thread build may have added CPU contention on the 12-CPU host during a
short portion of the reverse panel. Treat this HiGHS order comparison as a
diagnostic, not a clean independent confirmation. The solver-aware label was
inconclusive in both orders; the wall-clock direction flipped, but the
temporary load means this run cannot attribute that flip solely to execution
order. Raw artifacts are under
`report/evaluation-disagreement-calibrated-20261016/highs-gpt5_6-luna-xhigh-current-harness/`.
