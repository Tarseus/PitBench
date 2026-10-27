# Extended metric scenarios

Qualification protocol **1.1** extends the existing protocol without changing the
frozen protocol **1.0** file or its hash. It supplies twenty additional metric
adapters. The three ordinary performance metrics and the two paired resource
ratios keep their existing estimators; search-work counters stay diagnostic.

This is an implemented collection/reporting/qualification path. It is not evidence
that every solver release supports every API scenario, or that the sixteen release
combinations have completed their real benchmark runs.

## Declare the experiment before collection

`PitBenchTask.metric_scenarios` contains `ScenarioProtocol` declarations. Each
declaration fixes its ID, kind, metric IDs, task panels, budget, solver seeds,
thread counts, repeat count, cold/warm modes, feasibility tolerance, memory sampling
interval, search-reset policy and garbage-collection policy. The protocol hash
also binds every original model/update sequence through
`model_sequences_sha256[panel][instance_id]`. Model sequences have the same hashes
for Base and Agent and for their cold/warm runs.

Additional choices are explicit:

- First-passage times require `censoring: budget` or `par2` and a quality target
  per instance where applicable.
- Integrals require an independently established reference objective, a positive
  loss scale, and the initial loss assigned before the first observation.
- Lifecycle experiments require a warm-up count, a series reduction (`mean`,
  `maximum` or `last`) and `growth_estimator: ols`.
- Scaling requires a reference thread count, at least two thread counts, a common
  quality target and a declared curve reduction.
- Update experiments require both cold and warm modes and the scored mode.
- Replay requires at least two repeats, compared fields and explicit absolute and
  relative tolerances.
- Numerical stress experiments bind feasible reference assignments, which the
  evaluator independently checks on each original stress model.
- Formal setup, memory, sequence and replay qualification also binds
  `qualification_target_by_model_sha256` for every model step. Measured time or
  memory remains reportable, but a candidate returning worse final solutions
  cannot qualify by reducing work. Those targets are chosen before collection.

No task receives an inferred target, missing-value penalty, warm-up convention or
acceptance tolerance. Existing tasks with no scenario declaration keep their
existing run grids and scoring.

## Collect and retain evidence

The collector reads a YAML file with `task_config`, `candidate_patch`, `cpus`,
`watchdog_grace_sec`, `order`, `runtimes` and `instances`. An optional `protocols`
list constructs a new frozen `task.yaml` before measurement; an existing frozen
declaration cannot be changed during resume. Each panel inventory contains its
full declared number of cases. Cases have an `id` and `models`, or
`models_by_scenario` for scenario-specific sequences.

Each Base/Agent runtime supplies `python`, `session_plugin`, `source_commit`,
`patch_sha256`, `expected_binding_sha256` and optional solver `options`. The empty
patch hash identifies Base. Agent must bind the actual selected patch. Probe the
built binding before filling its fingerprint:

```bash
python -m scripts.collect_metric_scenarios probe \
  --session-plugin pitbench.solver_drivers.run:HighsScenarioSession
python -m scripts.collect_metric_scenarios prepare \
  --config /path/to/frozen-experiment.yaml --output /path/to/experiment
python -m scripts.collect_metric_scenarios run --output /path/to/experiment
python -m scripts.collect_metric_scenarios report --output /path/to/experiment
```

Each scenario selects an existing input-representation plugin and independent
problem-family verifier. Structured routing/CP assignments use those verifiers;
numeric assignments additionally expose residuals and relaxation certificates.

`ScenarioSession` is the common driver interface. The supplied
`HighsScenarioSession` verifies its binding version, Git hash, binary and Python-code fingerprints
against the task and runtime. It records improving MIP assignments and final LP
solutions/row multipliers. Stateful updates support fixed dimensions and fixed
integrality, with changed costs, bounds and matrix coefficients. Other APIs can
implement the same interface in the existing driver module. Missing stateful or
reset capabilities are reported as unsupported; launching another process cannot
substitute for retained-memory or warm-update evidence.

Every job gets a fresh worker and the exact allocated CPU affinity. Warm sequences
retain one solver object and one process; cold sequences construct a new solver
object for each model. The common bounded scheduler and process watchdog are
reused. Models are copied to content-addressed input files. Every callback,
completed step, exception, process log and attempt is retained. Resuming keeps
completed attempts; `--retry-errors` explicitly starts new attempts after errors.

The worker exports assignments rather than trusted objective/validity claims.
The reporting process rereads the original hash-bound models and checks row,
bound, integrality and objective arithmetic independently. A missing run or model
step makes the metric incomplete; invalid intermediate solutions remain visible.

## Measurement definitions

**First solution/target time.** Use the first timestamp of an independently verified
assignment satisfying the declared target. A target is shared across thread
conditions. Incumbents arriving after the budget do not count. The selected budget
or PAR2 value is reported with an explicit censoring flag when the target is not
reached.

**Primal integral.** Integrate the right-continuous best verified loss to the fixed
budget. Loss is `(objective - reference) / scale` for minimization and its negative
for maximization. The declared initial loss applies until the first solution;
signed improvements below a BKS are retained. After solver return the final state
is held through the remaining budget.

**Bound integral.** Apply the corresponding signed bound loss. A bound is accepted
only after independent weak-duality evaluation. The supplied verifier computes LP
relaxation certificates using exact rational arithmetic on stored coefficients,
row multipliers, row bounds and reduced-cost support over variable bounds, then
rounds outward. These bounds remain valid for integer restrictions. A logged MIP
search bound without a certificate is unavailable. This does not hide independently
available primal measurements.

**Retained memory/growth.** Read current resident pages after the declared reset and
cleanup operation. Remove exactly the declared warm-up prefix, retain the series,
and apply its preregistered reduction. OLS growth regresses these bytes on
post-warm-up solve index. A process-lifetime maximum cannot populate the series.
The scope is `binding_worker_after_record_flush`, including the binding's resident
process rather than only its native heap. Completed observation payloads are
flushed and released before sampling; the full history is reconstructed afterward.

**Thread scaling/efficiency.** Keep per-thread target times and compute
`speedup(n) = T(reference) / T(n)` and
`efficiency(n) = speedup(n) / (n / reference_threads)`.
Censored or zero target times cannot establish efficiency. Raw curves are
retained along with their declared scalar reduction.

**Incremental sequences.** Cold and warm modes execute identical model sequences.
Sum solve times and separately sum in-place update times for the declared scored
mode. Missing control-mode observations prevent a complete comparison.

**Numerical stress.** Recompute maximum original-model row, bound and integrality
violations. Count invalid assignments, explicit solver errors and false infeasibility
claims on the independently witnessed feasible corpus. An ordinary time limit
without a solution is not automatically a numerical failure. Nonfinite independent
arithmetic is an evaluator limitation and cannot establish candidate correctness
or a numerical improvement.

**Replay.** Compare every pair of identical input/seed/thread/mode repeats on the
declared fields (`objective`, `termination`, `solution_hash`). Apply the declared
numeric tolerances to objectives; hash/status equality is exact. Consistency is
matching pairs divided by the complete declared pair count.

**Setup.** Measure binding API construction and model loading including marshalling,
with Python startup outside the stage. Record a sampled stage RSS peak and its
declared sample interval. It is not a read-time-only or process-lifetime measurement.

**Native robustness.** The native seed adapter preserves its existing dispersion
coordinate and 99% paired seed-list bootstrap. Representation qualification reuses
Type-7 per-instance IQR and its full declared relabeling grid; protocol 1.1 can bind
an explicitly declared hidden qualification panel on supported Judge backends.
Boundary counts retain the declared case/seed denominator. Controlled retention
uses all declared conditions and the existing simultaneous support decision.

## Reporting, qualification and confirmation

Select `protocol_version: "1.1"` in the objective scoring plan and use
`qualification_protocols_sha256("1.1")`. Freeze the scenario hashes in
`scoring.scenario_protocols_sha256`. Seed qualification additionally declares its
scored native dispersion coordinate. Existing quality, coverage, compute and
memory guardrails remain mandatory and their tolerances remain task-specific.

Scenario scalars use equal-instance paired means and the existing 95% instance
bootstrap, with at least two complete independent instances. Higher-is-better
metrics reverse the signed degradation. Raw vectors, censoring, missing records,
native decisions and per-instance results remain available. Distance and curve
diagnostics do not silently choose a primary metric.

The design hash also binds the host fingerprint, architecture, CPU model/pool,
total memory and cgroup resource limits. Resume validates that hardware contract.
A nonempty candidate patch with identical Base/Agent API fingerprints is rejected
instead of measuring an unchanged installed binding as the candidate.

The normal independent Judge grid is still required. A scenario primary can request
confirmation, but acceptance additionally needs a new full Judge run and a fresh
scenario collection with the same task, candidate, runtime fingerprints, model
sequences and grid, a distinct collection ID, later start than the initial finish,
and reversed Base/Agent order. Reusing, reducing or changing a collection cannot
confirm an improvement. Invalid scenario solutions invalidate the candidate.

For unified evaluation, map each applicable task to its machine-local runtime
configuration under `metric_scenario_configs` in the evaluation YAML. Preflight
checks the explicit configuration; the evaluator collects initial and confirmation
scenarios and retains their campaigns as artifacts. The API runtimes must be built
from the pinned Base and selected candidate source; an installed release is not a
substitute for the candidate binding. Build support remains a per-release capability.

`pitbench report` discovers sibling `metric-scenarios/campaign.json` and confirmation
campaigns, or accepts `--scenario-campaign`, `--repeat-scenario-campaign` and
`--candidate-patch`. It recomputes from raw evidence and includes `metric_scenarios`
in JSON. PR qualification exports and revalidates those campaign artifacts.

## Verification scope

The regression suite checks every extended adapter, malformed/missing evidence,
normalization, late solutions, cold/warm controls, resource semantics, invalid
solutions, protocol hashes and independent confirmation. The native collection
test executes seven scenario kinds and forty isolated worker jobs on tiny LP
fixtures using HiGHS 1.15.1, including stateful updates and resume. This validates
the implementation path; it is not a performance result for the benchmark corpus
or the other selected releases.
