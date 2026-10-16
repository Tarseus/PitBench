# Trace-guided development feedback

The shared research discussion proposes a solver-behavior-guided Agent loop:
compare Base and candidate search behavior, turn the contrast into a testable
hypothesis, and confirm shortlisted patches independently. PitBench implements
the first development-facing part of that loop without treating it as a
validated method contribution.

## Purpose

Raw per-run outputs make an Agent reconstruct the same questions repeatedly:

- Did solution quality or verified solve coverage change?
- Is the apparent gain spread across independent instances or concentrated in
  one case and many seeds?
- Did the candidate reach useful solutions earlier, perform more solver work,
  or merely report a smaller runtime?
- Is the development grid complete enough to interpret the result?

`pitbench diagnose` answers these questions from paired, public development
runs. It never reads hidden instances and does not promote a candidate.

## Workflow

From a clean task repository:

```console
pitbench inspect
pitbench bench --label base
# edit the solver
pitbench bench --label agent --build
pitbench diagnose
# Only for a promising shortlist:
pitbench bench --label base --all-seeds --all-budgets
pitbench bench --label agent --all-seeds --all-budgets
pitbench diagnose --all-budgets
```

Every run retains independently checked feasibility where a public verifier is
available, the declared solver metric, runtime, common work counters, and a
summary of the solver trajectory. Missing trajectory fields remain explicitly
unavailable. They are not replaced with a solver-specific guess.

The diagnosis groups seeds within each independent development instance before
counting better, equal/unresolved, and worse instances. For exact tasks it
compares verified solved coverage first and PAR-2 only when coverage ties. For
heuristic tasks it compares paired normalized-gap reduction. It separately
reports runtime and work deltas so higher throughput is not mistaken for a
fixed-budget solver improvement.

## Diagnostic patterns

The output uses descriptive patterns rather than significance claims:

- `favorable_effect_across_multiple_instances`: a development signal appears
  on more than one independent instance and no instance regresses;
- `favorable_effect_concentrated_in_one_instance`: additional independent
  instances are needed before the mechanism is credible;
- `mixed_instance_effect`: inspect structural differences between improving
  and regressing cases;
- `no_detected_solver_outcome_change`: a runtime-only change may still be an
  implementation optimization, but it has not improved the declared solver
  outcome;
- `correctness_or_execution_failure`: repair validity or execution before
  interpreting performance;
- `missing_paired_evidence`: collect matching Base and candidate observations.

The saved `diagnosis.json` includes the patch hash, grid completeness, paired
trajectory coverage, per-instance effects, diagnostic hypotheses, and the next
bounded experiment.

The initial comparison intentionally uses the default development slice. The
diagnosis requests the complete seed and budget grid only after a candidate has
a favorable signal across multiple independent instances. Concentrated, mixed,
invalid, and unchanged candidates receive cheaper falsification steps instead
of the same fixed repetition budget.

## Evidence and claim boundary

An audit of the existing Agent cohort found that most final patches changed
caching, allocation, or repeated computation, while relatively few directly
changed search policy. It also found repeated cases where the default public
result directory was read-only. The latter is now repaired by automatically
using a task-specific temporary directory while keeping Base and candidate
results paired.

These observations motivate structured solver feedback, but do not prove that
it improves Agent patches. A confirmatory intervention must hold the model,
starting repository, tool permissions, and total development budget fixed, and
compare at least:

1. final solver metrics only;
2. final metrics plus generic profiling;
3. trace-guided diagnosis.

The outcome must be measured on untouched instances with correctness gates and
the benchmark's noise-confirmation protocol. Search-policy, implementation, and
configuration changes must be reported separately. Until that intervention is
complete, this feature is development tooling and the proposed TRACE method
remains unvalidated.
