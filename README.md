# PitBench

PitBench is a benchmark and evaluation framework for measuring how coding
agents improve real combinatorial-optimization solvers.

The project separates solver validity from performance and records evaluation
at the level of code state, instance population, instance, solver seed, and
budget. Higher-level quality, stability, patch-gain, human-relative, and
distributional-robustness metrics are derived from those observations.

## Repository status

The existing `fc-eval` codebase is preserved under [`upstream/`](upstream/) as
execution-harness legacy and implementation reference. PitBench will inherit
generic agent, terminal, container, remote-execution, and provenance
infrastructure without inheriting FormulaCode's ASV/workload evaluation model.

PitBench's native data model and evaluator have not yet been implemented.

