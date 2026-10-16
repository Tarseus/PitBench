# Why PitBench needs solver-aware evaluation

The [shared research discussion](https://chatgpt.com/share/6acede63-7c28-83ea-a4aa-49361ec6255a)
recommends making evaluation methodology the main research question: **can
ordinary coding-agent evaluation mischaracterize an agent's ability to improve
optimization software?** Its strongest proposed experiment applies conventional
and solver-aware evaluations to the **same agent patches**, then checks whether
their conclusions differ in a material, reproducible way.

PitBench's target capability is a coding agent's ability to modify a real solver
repository and produce a **correct, repeatable improvement on unseen optimization
instances under declared computational budgets**. Test passing and a single
runtime measurement are useful observations, but neither establishes that
capability. PitBench fixes the solve budget per task, so elapsed runtime is a
secondary diagnostic and may be censored by that budget. The central question
is whether fixed-budget solver outcomes can reliably distinguish real solver
improvements among correctness-passing patches. The current report already
shows that ordinary correctness has the weakest discrimination among its
evaluation dimensions. A search change can appear beneficial for one seed
while failing on other instances; an exact solver can return a feasible
solution without a trustworthy optimality result.

For each fixed patch, the benchmark should retain enough evidence to compare:

1. Ordinary software signals: build and unit-test results. A single-run speed
   measurement may be retained as a secondary diagnostic.
2. Solver outcomes: independently checked feasibility and objective value;
   independently supported optimality for exact tasks; solution quality or
   verified-solve coverage at each declared budget.
3. Reliability: paired Base/Agent observations across independent instances and
   declared seeds, repeats of the complete evaluation, and results on unseen
   instances or input representations where the task provides them.
4. A calibrated conclusion of improvement, regression, inconclusive evidence,
   or incomplete evaluation, with the raw results and uncertainty retained.

“Conventional coding-agent evaluation” is not one universal score. A study
must name the comparator in advance, such as task completion under hidden
correctness tests or the benchmark's existing task-success score. In the
available PitBench cohort, patch acceptance and build/test gates vary across
solver families; the wall-clock-only proxy is exploratory and cannot stand in
for a standardized conventional-agent score. That matched comparison remains
open.

The motivating demonstration is a patch that passes ordinary correctness
tests while solver-aware evaluation **reliably** finds a hidden-instance
regression or improvement, or the reverse. A difference in labels alone is
insufficient: it must exceed the disagreement produced by empty patches and
repeated evaluations of identical code. Because the solver budget is fixed,
elapsed runtime is useful for diagnosing timing noise but is not the main
conventional comparator. Noise controls are part of the benchmark's validity.

The first comparison covered 16 nonempty patches. Its early scoring reported
10 cases where wall-clock and solver-aware labels differed, but an empty
CP-SAT patch was falsely marked `regressed` from one changed solve out of 900,
and small VROOM labels changed across repeats. Re-scoring those observations
with the current exact-task rule leaves 5 improved, 2 regressed, and 9
inconclusive solver results. Five former directional labels become
`inconclusive`, including all three Choco patches whose extra solves occurred
on one instance. Thus the early 10/16 figure is exploratory, not a calibrated
estimate of useful disagreement. All 16 saved patch hashes matched the cohort
manifest and their expected solver grids were complete; the HiGHS panel's
2,824 `no_solution` outcomes are ordinary budget-limited solver results, not
missing runs. The original evaluation envelopes for all 16 patches also
record `completed=true` and `validity.accepted=true`. That only means each
patch was accepted and the evaluation completed, not that one shared unit-test
suite passed; it shows that this basic acceptance gate alone cannot rank them.

The [seven-task noise calibration](evaluation-noise-calibration.md) found a
full empty PyVRP patch that a single-run solver-aware interval falsely called
`improved`, as well as one-run exact coverage differences on identical code.
PitBench now requires independent opposite-order confirmation for directional
full evaluations and instance-supported exact coverage directions. This makes
the comparison with ordinary coding-agent evaluation more credible; it does
not by itself prove that the two methods reach materially different conclusions
on real Agent patches.

## Evidence update: calibrated PyVRP patch replays

On 2026-10-16, three completed GPT-6 PyVRP patches were replayed under the same
current harness, judge image, task, resources, hidden grid, budgets, and seeds.
Each forward and reversed round contains 8,160 valid observations, and the
PyVRP validation build ran its test suite on Base and Agent. Two patches had
recorded agent-side test results of 952 passed and 2 skipped (plotting tests
excluded).

- GPT-6 Luna xhigh was `improved` in both orders, with gap gains of 0.00612 and
  0.00841 percentage points.
- GPT-6 Sol xhigh was `improved` in both orders, with gains of 0.00520 and
  0.00486 percentage points.
- GPT-6 Luna medium was `inconclusive` forward (+0.00217 pp
  [−0.00093, +0.00524]) and `improved` reversed (+0.00951 pp
  [+0.00255, +0.01726]). Its first public result therefore remains
  `inconclusive`; the extra reversed run was a research diagnostic. Averaging
  per-instance effects across both orders gives +0.00584 pp
  [+0.00213, +0.00986], an exploratory estimate that is not the public score.

All six time-only intervals crossed zero. This is initial evidence that the
solver-aware metric can show a quality change among test-passing patches when
a time-only signal cannot establish one. Per-patch order-averaged estimates are
positive in all three cases: Luna medium +0.00584 pp
[+0.00213, +0.00986], Luna xhigh +0.00727 pp [+0.00508, +0.00967], and Sol
xhigh +0.00503 pp [+0.00245, +0.00780]. These are exploratory averages over
the two order-specific per-instance means, not the public single-run labels;
Luna medium's single-run classification changes with order. A post-hoc paired
comparison across the three patches leaves all order-averaged between-patch
intervals crossing zero, so the experiment does not reliably rank these
agents. Effects are small, and all three patches are from one problem family.

The evaluator's validation stage is not a uniform test gate across solvers.
Its PyVRP plugin runs pytest on Base and Agent; VROOM and HiGHS validation
commands build the solver; Choco invokes Maven with `-DskipTests`; OR-Tools
configures CMake with `BUILD_TESTING=OFF`. These commands are defined in
[`plugins.py`](../pitbench/repositories/plugins.py), and the judge runs them
for each code state before performance evaluation. So the PyVRP examples give
a genuine test-pass comparison, while the wider cohort mainly compares solver
performance against patch/build acceptance. That limitation needs to be fixed
before drawing a general conclusion about ordinary coding-agent benchmarks.

The same follow-up also replayed a VROOM patch in both execution orders. Its
time-only result remained faster in both rounds (+41.78 ms [15.19, 71.33] and
+45.36 ms [16.05, 81.05]), while its solver-quality result became
`inconclusive` after the reversed-order run. A GPT-5.6 Luna xhigh Choco patch
provided an order-sensitivity example. Base and Agent both verified 540/900
optima at 20 seconds in both rounds; solver-aware Performance was
`inconclusive` both times. The forward-order time-only result said Agent was
8.59 ms slower [3.53, 16.50], but after reversing order it was 1.96 ms faster
with an interval crossing zero. The forward-minus-reverse order effect was
−10.55 ms [−19.65, −3.32]. All 4,200 observations in each round were valid,
and only Base/Agent execution order changed.

The historical GPT-6 Luna xhigh Choco patch's agent trace also records 13/13
targeted TestNG tests passing through direct execution of compiled tests. The
normal Maven test command could not resolve the Surefire TestNG provider
without network access, so this is not a full-suite pass. Its hidden re-judge
had 570/900 versus 540/900 verified optima, but all 30 extra solves came from
one instance; the current instance-supported rule calls that result
`inconclusive`. Passing targeted correctness tests and seeing one favorable
solve count still do not establish a generalizable solver improvement.

The historical traces do not yet provide one standardized ordinary test score
across all solver families: several VROOM runs recorded no executed unit test
command, Choco's Maven tests were blocked by dependencies that required network
access. An older HiGHS agent trace recorded direct unit-test failures. A later
matched Base/Agent rerun with the same commit and patch passed 328 cases with
1,259,125 assertions, 37 targeted MIP cases with 303 assertions, and 21 CTest
cases after the `highs-bin` target was built. The earlier failures did not
recur in that rerun; their cause remains unknown. Thus the current evidence
compares solver metrics mainly with test-gating examples and a time-only proxy,
not a complete, uniform coding-agent benchmark outcome.

Together, these results support a limited claim: solver-aware evaluation can
identify test-passing patches with a positive solver-quality signal. The
direct comparison between the two PyVRP patches did not establish a reliable
ranking between them. The order-sensitive runtime result is a timing-noise
diagnostic, not the central evidence for distinct benchmark value under fixed
solver budgets. These results do not prove that the quality signal is usually
useful, practically large, or general across tasks. The no-op and repeat
protocols are described in
[evaluation noise calibration](evaluation-noise-calibration.md). The research
question remains open: compare both methods on a larger set of identical
patches and test whether their disagreements exceed calibrated no-change and
repeat-run variation. The current claim and counter-evidence are summarized in
the [distinct-value claim audit](benchmark-distinct-value-audit.md).
