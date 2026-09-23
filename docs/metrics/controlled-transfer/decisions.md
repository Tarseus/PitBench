# Controlled Transfer implementation decisions

This record distinguishes user-confirmed metric semantics from implementation defaults
selected under the user's 2026-09-22 instruction to continue with documented defaults.
It does not freeze the metric or select a real evaluation panel.

## Confirmed metric decisions

- Controlled Transfer replaces separate Scalability and Distributional Generalization
  metrics; scale and structural changes are reporting views.
- Exact Performance remains coverage-first. Solver-origin unsolved outcomes retain
  their existing `2T` treatment.
- The retention fraction is `0.8`, the reference-benefit floor is `0`, individual
  intervals are 95% percentile instance bootstraps with 5,000 resamples and seed
  `20260824`, and joint claims use Bonferroni bounds.
- Missing, infrastructure-failed, or otherwise unusable task-native results make the
  relevant Controlled Transfer condition `incomplete`.

## Implementation defaults selected

- `PitBenchTask` owns the task-context labels `problem_variant` and `formulation_id`.
- `EvaluationProtocol.controlled_transfer.conditions` owns condition metadata. Every
  condition maps to one instance set and declares its view, reference condition,
  scale metadata, structural domain, generator version, and an optional evaluation
  budget. A missing condition budget defaults to the task's `primary_budget_sec`;
  this prevents diagnostic budgets from being pooled into the transfer benefit.
- `InstanceCase`, `ExpectedRun`, and `RunObservation` carry resolved `condition_id`
  and optional `template_id`. Static instance-set entries may provide `template_id`.
  A missing template ID means bootstrap at the individual-instance level.
- The declaration order of conditions within the same `scale_path_id` defines the
  ordered scale path used for continuous-support reporting.
- Controlled Transfer is computed in `controlled_transfer_report.py`, is exposed by
  `EvaluationSummary`, and is rendered by the `pitbench report` command. Existing
  tasks remain unchanged unless they opt into `evaluation.controlled_transfer`.

## Open questions for user decision

- Which task context and evaluator-owned scale, structural, and joint panels should
  be the first real validation panel.
- Which existing or new task configurations should opt into Controlled Transfer after
  their panel metadata, instance generators, and oracle evidence are approved.
- Whether the project-level metric-program ontology document should be revised to
  replace its separate Scalability and Distributional Generalization entries.

## M5 candidate-panel audit

The first default M5 candidates were the retained PyVRP 0.14.0 Base/Agent trial
artifacts and the `results/scale-structure-20260910` routing exploration. Neither is
accepted as a Controlled Transfer real-panel validation result.

- The retained trial artifacts contain real Base and Agent observations, but do not
  contain a predeclared Controlled Transfer condition manifest or the matching
  expected-run grid.
- The scale-and-structure exploration has a complete, independently feasibility-checked
  installed-baseline run set, but it has no Agent-versus-Base comparison. Its routing
  reference is a longer-budget feasible result, explicitly not a proven optimum or
  frozen BKS anchor.

The default M5 decision is therefore to retain these artifacts as implementation and
panel-design evidence only. A real M5 result requires an evaluator-owned panel that
declares the Controlled Transfer conditions before execution, retains the expected
run grid, uses an approved task-native reference/oracle, and evaluates both Base and
the same frozen Agent patch.

## M5 default execution selection

The retry5 PyVRP 0.14.0 trial is the default real-panel implementation validation.
It contains the immutable, complete Base/Agent observations and expected run grid for
38 `judge_id` instances and 10 `judge_shift` instances at 5 and 10 seconds with 30
solver seeds. The `judge_shift` generator records clustered coordinates, bimodal
demands, a corner depot, and capacity ratio 0.12, so the replay labels it as the
structural view at the task primary 10-second budget.

The replay is retrospective: the original trial predates the Controlled Transfer
manifest. It validates report computation, task-native BKS-relative gaps, paired
Base/Agent data, and complete-grid handling on real runs. It does not establish a
prospectively predeclared benchmark panel, nor does it validate the scale or joint
views. The panel identity, input hashes, candidate-patch hash, and condition mapping
are stored in `private/controlled_transfer/pyvrp_v0_14_0_retry5_m5/panel.json`.
