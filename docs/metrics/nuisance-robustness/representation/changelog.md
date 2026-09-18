# Representation Robustness changelog

## 0.0.1 — 2026-09-18

Stage: M7 / Frozen, with explicit user approval.
Implementation commit: `22d19065`.
Commit subject: `freeze: representation robustness v0.0.1`.

The frozen definition is specified in [spec.md](spec.md), the falsification
cases are recorded in [validation-plan.md](validation-plan.md), and the M5/M6
evidence is recorded in [validation-results.md](validation-results.md) and
[audit.md](audit.md).

### Frozen protocol

- Equivalent customer relabeling for the fixed PyVRP 0.14.0 protocol.
- Solver seed `0` and 30 distinct non-identity relabelings per instance.
- Type 7 IQR of normalized gap within each instance.
- Equal-weight mean across the ten-instance panel.
- Separate 5-second and 10-second budget results.
- Signed change `Agent - Base`.
- Complete-data requirements with no imputation.
- No confidence interval or robustness pass threshold.

The M5 panel contains 1,200 planned and 1,200 completed runs. All observations
are valid, mapped-feasible, and objective-preserving. The M6 review found and
resolved the scope, canonical-summary, and directory-format gap-propagation
consistency blockers.

After freezing, semantic or protocol changes require a user-approved version
bump.
