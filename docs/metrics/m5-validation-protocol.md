# Project-wide M5 validation protocol

## Purpose

This protocol fixes the validation evidence and lifecycle gate shared by the
PitBench metrics. It follows the validation discipline established by the
Performance M5 controls while preserving each metric's approved M2 estimand,
aggregation, confidence interval, bootstrap, and threshold decisions.

The protocol applies to every declared object or capability that a metric claims
to support. Passing one object panel does not establish support for another
problem family, solver class, or execution mode.

## Required M5 evidence

### Synthetic falsification cases

Every approved M3 synthetic falsification case must pass with its approved
expected values, invariances, missingness behavior, and numerical tolerance.
Missing or incomplete synthetic evidence does not count as a pass.

### Declared real panel

For every object or capability declared in scope, retain a real validation panel
with an explicit expected grid. The manifest must identify the task, problem
family, solver or repository version, source commit, code states, instance set,
budgets, seeds or repetitions, thread count, and metric-specific protocol
parameters.

The panel must retain raw observations, successful outcomes, independent
verification results, failure records, missing required runs, and the computed
metric details. Runs may not be silently removed, imputed, replaced, or pooled
across undeclared strata.

### Reproducible report

The report must recompute the approved estimator from the retained artifact and
report every declared budget, code state, instance set, and supported object
stratum separately. A second report from the same manifest and raw observations
must reproduce the recorded point estimates and any metric-approved interval
outputs within the approved numerical tolerance.

The report must preserve the distinction between:

- semantic qualification failure;
- operational failure or infrastructure missingness;
- unavailable metric observations; and
- a valid metric outcome.

## M5 gate

An M5 validation item passes only when all of the following hold:

1. Every approved M3 falsification case passes.
2. Every declared supported object or capability has a complete real panel, or
   the validation record explicitly marks that object as `incomplete` and keeps
   M5 open for it.
3. The expected grid, raw records, failures, missingness, and independent
   verification evidence are complete and auditable.
4. The report output matches the approved M2 estimator and reproduces from the
   retained artifacts.
5. No qualification, operational, infrastructure, or reporting blocker remains
   unclassified.

An object without an approved adapter is `unsupported`; it is not a successful
validation result and does not count toward coverage.

## Interpretation boundaries

An empty patch or no-op control can validate the execution and reporting
protocol, but it is not evidence that a patch improves or regresses the metric.
M5 results do not establish universal sample-size sufficiency, cross-object
generalization, statistical significance, or a deployment success probability
unless those claims are explicitly part of the metric's approved M2 protocol.

Passing M5 does not advance a metric to M6 or M7 automatically. Freeze-candidate
review, version selection, commit recording, and freeze status still require
explicit user approval.

## Metric-specific decisions retained

This shared protocol does not select or override any metric-specific:

- scientific question or exclusion;
- random variable or estimand;
- aggregation rule;
- confidence interval or bootstrap method;
- numerical threshold or classification gate; or
- object-specific transformation, parameter, or feedback definition.

Those decisions remain in the metric's approved M0-M3 artifacts.
