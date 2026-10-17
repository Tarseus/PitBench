# Solver-release controls for Performance sensitivity

PyVRP release differences provide a positive control for the question “can
PitBench's solver-aware Performance metric detect a substantive change under a
fixed solve budget?” They are calibration data, not additional Agent tasks or
evidence that a conventional Agent benchmark would select a different patch.

## Candidate search on public development instances

The retained four-release archive contains valid no-change Base observations
for v0.12.2, v0.13.0, v0.13.4, and v0.14.0 on the same ten development
instances, same 1,000 solver seeds, and 1/5/10-second budgets. Each release has
30,000 Base observations. The archived Agent copies were not treated as
independent release comparisons.

The [v0.13.0 release](https://github.com/PyVRP/PyVRP/releases/tag/v0.13.0)
replaced hybrid genetic search with iterated local search.
Its result relative to v0.12.2 changes with the budget: mean normalized gap
gets 0.716 percentage points worse at 1 second, 0.270 points worse at 5
seconds, and 0.146 points better at 10 seconds. The ten-instance interval at
10 seconds crosses zero. Thus a large algorithm rewrite is not automatically a
positive control under this task's time limits.

The strongest development candidate was v0.13.4 → v0.14.0. At the current
10-second primary budget, v0.14.0 reduced mean normalized gap by 0.545
percentage points; eight of ten independent development instances improved.
The [v0.14.0 release notes](https://github.com/PyVRP/PyVRP/releases/tag/v0.14.0)
also describe substantial changes in search representation, neighbourhood
setup, and penalties. The version pair and confirmation rule
were frozen before inspecting the hidden panel. Detailed selection evidence
and the frozen protocol are retained under `report/version-control-20261017/`.

## Untouched 38-instance confirmation

The confirmation used the 38 hidden CVRPLIB X instances, which have zero ID
overlap with the ten development instances. It used the first five fixed
hidden seeds, the current 10-second budget, one solver thread per one-CPU
container, and 12 concurrent pairs on a 12-CPU host. Each instance/seed pair
ran both releases in two rounds, reversing execution order in the second.
Both releases used the same current PitBench driver and independent CVRP
verifier. The 760 planned solver runs all produced valid solutions; there were
no missing or failed runs and no reported-versus-verified objective mismatch.

| Measurement | Round 1 | Reversed-order round 2 |
| :-- | --: | --: |
| Mean old-minus-new gap reduction | +0.1884 percentage points | +0.1958 percentage points |
| 95% instance-bootstrap interval | [+0.0329, +0.3454] points | [+0.0382, +0.3552] points |
| Median instance effect | +0.0826 points | +0.1058 points |
| Worst instance effect | -0.6777 points | -0.6752 points |
| Better / equal / worse instances | 24 / 0 / 14 | 24 / 0 / 14 |
| Mean after removing largest absolute instance effect | +0.1575 points | +0.1653 points |
| Existing PitBench Performance label | `improved` | `improved` |

All 38 instances kept the same effect direction across the two rounds. The
same-version round-to-round mean drift was -0.0015 points for v0.13.4 and
+0.0059 points for v0.14.0, with both null intervals spanning zero. The
smaller version effect was about 32 times the larger absolute null mean drift.
The existing opposite-order confirmation rule therefore retains `improved` on
this reduced grid. Independent recomputation directly from verified objectives
agreed with the saved analysis.

The improvement is an average, not an all-instance guarantee: 14 of 38
instances became worse in both rounds. The one-sided sign test for 24 better
versus 14 worse instances has `p = 0.0717`. The positive mean and bootstrap
interval are supported by improvement magnitudes as well as counts; outlier
removal still leaves a positive mean. This is a strong release-level positive
control for metric sensitivity under the declared reduced regime.

## Boundary of the claim

The version images are pinned by full local SHA-256 IDs, but only seven of
their thirteen filesystem layers are identical. The remaining differences
include the released code. A read-only check found the same Python 3.13.15,
NumPy 2.5.2, Pybind11 3.1.0, Meson 1.12.0, and key installed system packages
in both images; not every installed dependency or build condition was proved
identical. The experiment therefore measures the complete release-image
transition and does not isolate one algorithm commit. The five-seed
confirmation also does not substitute for the task's full 30-seed Agent
evaluation.

Most importantly, observing a real release improvement does not show that
PitBench adds decision value over an ordinary Agent benchmark. That stronger
claim still needs identical Agent patches evaluated by a declared conventional
comparator and by PitBench, with both judgments confirmed beyond their noise
and a predeclared practical-effect criterion. The release pair can serve as a
positive sensitivity control in that study.

The local raw pair records, exact image and code hashes, independent solutions,
frozen protocol, and analysis are in the ignored
`report/version-control-20261017/` directory. The reusable orchestration and
analysis commands are in `scripts/calibrate_release_controls.py` and
`scripts/analyze_release_controls.py`.
