# Audit of PitBench's distinct-value claim

## Reported premise

The project report already shows that ordinary correctness has the weakest
discrimination among its current evaluation dimensions. This audit treats that
as an established result for the present task suite; it does not attempt to
prove it again or generalize it to every coding benchmark.

## Claim under test

The research discussion's strongest proposal is to apply conventional coding-
agent evaluation and solver-aware evaluation to the **same agent patches**,
then test whether they reach materially different, reproducible conclusions.
For PitBench, the target capability is a patch that preserves correctness and
reliably improves solver outcomes on unseen instances under declared budgets.

This proposal is not established by showing that two metrics differ. It
requires evidence that the solver-aware result adds a useful decision signal
to what a conventional agent evaluation would report. The relevant questions
are:

1. **Same-patch comparison:** do both evaluation approaches assess identical
   patches, and do their conclusions differ in a consequential way?
2. **Validity of the disagreement:** does solver-aware evaluation's direction
   repeat on hidden instances and exceed no-change, seed, and run-order noise?
3. **Decision value:** is the difference large and general enough to change
   which patches or agents would be considered useful?

The declared per-instance solver budget is fixed. Therefore elapsed wall time
is a secondary timing diagnostic, not an adequate stand-alone conventional
comparator for PitBench's distinct value. The report's low-discrimination
correctness result is relevant background, but the strongest evidence still
requires a fair conventional-agent score and a solver-aware score on the same
patches.

“Conventional coding-agent score” also needs an explicit operational
definition: coding benchmarks do not share one universal score. The available
PitBench archive has patch acceptance and build/test outcomes, but test
commands vary by solver family. Its wall-clock-only comparison is a separate
exploratory proxy, not a substitute for a standardized conventional score.
Consequently, the full conventional-versus-solver-aware comparison has not yet
been run on a uniform comparator.

The existing experiments are retrospective and exploratory. They test these
parts separately; they do not provide a pre-registered estimate of a general
benchmark advantage.

## What the benchmark-level claim requires

Showing that two metric values disagree is not enough: they measure different
things, so disagreement can be expected by construction. The benchmark-level
claim is that a conventional coding-agent evaluation can miss or misjudge a
solver improvement that solver-aware evaluation identifies correctly, and
that this difference survives repeated complete runs and no-change controls.
To establish practical benchmark value, the effect must also meet a
practical-importance criterion declared before a new confirmatory cohort. The
current data do not establish that full claim.

## Evidence for additional signal

The strongest direct comparison is the three GPT-6 PyVRP 0.14 patches replayed
under the current task, image, harness, budgets, hidden grid, and resources.
Base and Agent passed the evaluator's PyVRP validation suite in every complete
round. For the Luna xhigh and Sol xhigh patches, the solver-aware Performance
label was `improved` in both execution orders. Their normalized-gap gains were
0.00612 percentage points (95% interval [0.00373, 0.00904]) and 0.00841
points ([0.00533, 0.01179]) for Luna xhigh, and 0.00520 points ([0.00113,
0.00912]) and 0.00486 points ([0.00171, 0.00791]) for Sol xhigh. All four
corresponding wall-clock-only intervals crossed zero; the Luna medium patch's
two additional intervals also crossed zero. A test-pass result would tie all
three patches, while solver-aware Performance distinguished two repeatable
quality improvements. The wall-clock results are secondary because the task
budget was fixed.

The Luna medium patch limits that interpretation: its forward result was
`inconclusive` and its extra reversed-order diagnostic was `improved`. That
order-specific label change shows that one positive repeat is not enough to
claim a stable public result. The post-hoc order-averaged estimates for the
three PyVRP patches were positive, but those averages were not pre-registered
or used as public scores.

Two cross-family examples clarify what the extra signal means. A VROOM patch
was faster in both orders, but its solver-quality result changed from
`improved` to `inconclusive` after reversal. A Choco patch had unchanged
verified-optimum coverage and inconclusive solver-aware results in both
orders; its one-run wall-clock slowdown disappeared after reversing Base and
Agent order. These examples show that runtime direction, order sensitivity,
and fixed-budget solver quality are distinct observations. They do not show
that a stable speedup without a quality change is useless.

## Evidence against stronger claims

- **Reliable ranking is not established.** The paired comparison of the two
  xhigh PyVRP patches had an interval spanning zero. The three-patch
  order-averaged pairwise intervals also crossed zero. Current evidence
  separates some patches from Base more clearly than it separates agents from
  each other.
- **Practical importance is unknown.** The two repeated PyVRP gains are below
  0.009 percentage points. No minimum practically meaningful effect was
  registered before these replays, so a positive interval alone cannot show
  that the gains matter in use.
- **The conventional comparator is incomplete.** Unit-test execution differs
  across solver families: some panels run tests, while others only build or
  disable tests. The cross-family comparison therefore is not yet against one
  standardized conventional coding-benchmark score.
- **The noise rate is not estimated precisely.** Empty-patch and repeated-run
  controls have exposed false directions and order sensitivity, and the
  current confirmation policy suppresses several fragile labels. The number
  of complete no-change controls remains too small to estimate a benchmark-wide
  false-positive rate.
- **The raw 10/16 disagreement count is not confirmatory.** It came from
  single-run historical labels. Current-rule rescoring changes the 16-patch
  cohort to 5 `improved`, 2 `regressed`, and 9 `inconclusive`; repeated
  evaluations also changed several directions to `inconclusive`. Some older
  panels lack a current harness fingerprint, so this re-score is exploratory.
  The raw count must not be presented as a calibrated value estimate.

This does not mean that only PyVRP ever received a positive solver label. The
16-patch current-rule re-score includes single-round `improved` labels for
non-PyVRP patches as well. Those labels were not all confirmed by a matched
order-reversed replay, and some historical panels have identity gaps. Among
the newer cross-family replays, VROOM's solver-quality direction did not
survive reversal, while Choco and HiGHS remained `inconclusive`. The two
clear, repeated solver-quality gains in the current matched replay set are
both PyVRP patches.

The HiGHS matched-test follow-up adds a test-pass case, but its solver results
do not establish an improvement. Base and Agent both passed 328 unit-test
cases (1,259,125 assertions), 37 targeted MIP cases (303 assertions), and a
21-test CTest subset after the missing `highs-bin` target was built. The
complete forward solver panel found 349 versus 350 verified optima at the
primary budget; the complete reverse-order panel found 355 versus 358. Both
Performance labels were `inconclusive`: forward coverage gain was `+0.00111`
with interval `[0, +0.00333]` and PAR-2 reduction was `+0.03194` seconds
`[−0.01816, +0.12239]`; reversed coverage gain was `+0.00333`
`[0, +0.00889]` and PAR-2 reduction was `+0.11801` seconds
`[−0.00011, +0.31439]`. The time-only difference changed sign across orders:
forward Base-minus-Agent was −23.24 ms `[−42.46, −6.75]`; in reverse order it
was +59.89 ms `[+22.06, +103.29]`. A duplicate reverse-order
container was briefly started and stopped during its validation build after
the original live container was found. It wrote no solver observations, but
the one-thread build may have added short CPU contention to the 12-CPU host.
The reverse panel therefore remains diagnostic: it confirms no directional
solver-performance result, but it cannot attribute the time-direction flip
solely to Base/Agent order.

## Verdict

The evidence **partly supports solver-aware measurement**: two test-passing
PyVRP patches had small solver-quality gains that repeated in both orders.
This supports the possibility that solver-aware evaluation adds information
among patches with the same correctness outcome. It does not yet demonstrate
the strongest benchmark-level claim, because the experiments did not compare
against a standardized conventional coding-agent score across the same patch
cohort. The time-only intervals are secondary under fixed solver budgets; the
order replays show why single-run timing should be treated cautiously.

The evidence does **not establish PitBench's broader or unique value** as a
benchmark. It does not show that the measured gains are practically large,
that they generalize across solver families, that they reliably rank agents,
or that conventional coding-agent evaluation reaches different useful
conclusions on a matched patch cohort. The claim remains an empirically
supported research hypothesis, not a demonstrated benchmark-wide advantage.
A stronger result requires a standardized conventional-agent comparator,
more independent patches across problem families, and a predeclared
practical-effect criterion evaluated against complete no-change and
order-reversed controls. Wall-clock measurements can remain a secondary
diagnostic for order and resource sensitivity.

## Evidence index

- [Benchmark motivation and full evidence discussion](benchmark-motivation.md)
- [Noise-control protocol and calibration evidence](evaluation-noise-calibration.md)
- [Solver-release positive-control validation](release-control-validation.md)

The release-control study found that the existing Performance rule detects a
repeatable PyVRP v0.13.4 → v0.14.0 average quality gain on an untouched
38-instance panel under a fixed 10-second budget. That supports metric
sensitivity. Because it compares complete release images rather than the same
Agent patch under two evaluation protocols, it does not close the distinct-value
claim audited above.

The local raw replay tables, historical cohort re-score, matched-test summary,
and order-reversed HiGHS observations are retained under
`report/evaluation-disagreement-calibrated-20261016/`. The repository's
`.gitignore` excludes `report/`; the findings and test summary needed to
understand the claim are also recorded in the linked documents above.
