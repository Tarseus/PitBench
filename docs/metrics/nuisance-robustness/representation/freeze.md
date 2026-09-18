# Representation Robustness M7 freeze record

Metric: Nuisance Robustness / Representation Robustness.
Version: 0.0.1.
Stage: M7.
Status: frozen.

Frozen on: 2026-09-18.
Frozen implementation commit: `22d19065`.
Branch: `metric/representation-robustness-v1`.

The frozen protocol is the approved equivalent customer-relabeling protocol for
PyVRP 0.14.0. It uses a fixed solver seed of `0`, 30 distinct non-identity
customer relabelings per instance, Type 7 IQR within each instance, equal-weight
aggregation over ten instances, separate 5-second and 10-second budgets, and
the signed change `Agent - Base`.

The freeze is backed by the M3 falsification plan and results, the complete
1,200-run PyVRP panel, the M6 freeze-candidate audit, and the implementation
tests recorded in the metric artifacts. Missing or invalid relabeling outcomes
remain explicit and are not imputed.

This freeze does not claim that 30 relabelings are sufficient for every solver
or instance, does not add confidence intervals or a robustness threshold, and
does not generalize beyond the approved customer-relabeling transformation.

After freezing, changes to the metric semantics or protocol require a
user-approved version bump under the metric-development contract.
