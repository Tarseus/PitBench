# Superseded MIP Seed Robustness draft

> The current cross-solver M2 authority is [seed-protocol.md](seed-protocol.md).
> This HiGHS-specific draft is retained for traceability and is not an
> independent protocol.

## M2: HiGHS capped optimal time and final-gap seed sensitivity

This document records the approved M2 candidate for MIP Seed Robustness in the
Nuisance Robustness 0.0.2 cycle. It extends the existing seed-panel structure
to the HiGHS MIP task without changing the frozen CVRP 0.0.1 protocols.

The candidate uses the existing HiGHS source snapshot, ten-instance MIP panel,
two budgets, one thread, and the same domain-and-list structure as the frozen
Seed Robustness protocol. It declares the candidate seed domain

\[
D_{\mathrm{HiGHS}}
=
\{d \in \mathbb{Z} \mid 0 \le d \le 2^{31}-1\}.
\]

The target seed distribution is uniform over this declared domain. Each task
uses a 30-seed development list and a disjoint 30-seed evaluation list, both
sampled without replacement from the domain. The actual lists are fixed before
the Agent runs, retained with the panel, and reused across instances, budgets,
and Base/Agent code states. A fresh solver process is used for each
code-state/instance/seed/budget combination.

This is a project-declared candidate domain for the MIP protocol. It is not a
claim that the current repository evidence has proved the complete native HiGHS
seed range.

## Controlled conditions

Hold the following fixed within a panel:

- HiGHS source commit and repository environment;
- original MIP instance and representation;
- solver configuration and numerical tolerances;
- one solver thread;
- budget, reported separately at 10 and 30 seconds; and
- the declared HiGHS seed domain and its two disjoint 30-seed lists.

Base and Agent use identical instance, budget, and solver-seed conditions. The
same seed list is paired between code states. The existing MIP nuisance panel
provides the ten-instance and 30-seed starting artifact; a formal Base/Agent
panel must retain both code states.

Operational failures, invalid solutions, invalid optimality claims, and
infrastructure missingness remain failure or missingness records. They are not
converted into timeouts or ordinary metric outcomes.

## Primary result: capped time-to-optimal

For code state \(c\), instance \(x\), budget \(T\), and solver seed \(\xi\), define
the capped optimal time \(C(c,x,T,\xi)\) in seconds:

- if the solver reports 'Optimal' and the returned solution passes independent
  verification, use the solver runtime capped at \(T\);
- if the run normally reaches the declared time limit without an independently
  verified optimal result, use \(T\) as a censored observation; and
- if the run crashes, runs out of memory, has an infrastructure failure, or
  makes an invalid optimality claim, no \(C\) outcome is available.

The per-instance primary seed-sensitivity estimator is the Type 7 IQR of the
available capped-time outcomes over the applicable declared 30-seed list:

\[
S_C(c,x,T)
=
Q_{0.75}\left(\{C(c,x,T,\xi_r)\}_{r=1}^{30}\right)
-
Q_{0.25}\left(\{C(c,x,T,\xi_r)\}_{r=1}^{30}\right).
\]

The per-instance result is unavailable when the declared complete-data rule is
not met. The cross-instance primary result is the equal-weight mean of the
available per-instance IQRs only when all ten declared instances are complete.
The Base/Agent change is:

\[
\Delta S_C(T)
=
\overline{S}_{C,\mathrm{Agent},T}
-
\overline{S}_{C,\mathrm{Base},T}.
\]

Lower \(S_C\) means less seed-induced dispersion in time-to-optimal. Negative
\(\Delta S_C\) means lower Agent dispersion.

The report retains both capped seconds and the deterministic budget-normalized
form \(C/T\). The budget-normalized form is a reporting representation of the
same capped-time observation, not a second headline estimator.

## Secondary result: final normalized-gap dispersion

For every seed with a finite normalized gap from an independently verified
feasible result, retain the final normalized gap \(g(c,x,T,\xi)\). The secondary
per-instance estimator is:

\[
S_G(c,x,T)
=
Q_{0.75}\left(\{g(c,x,T,\xi_r)\}_{r=1}^{30}\right)
-
Q_{0.25}\left(\{g(c,x,T,\xi_r)\}_{r=1}^{30}\right).
\]

The same declared seed-list completeness, equal-instance-weight aggregation, and
Agent - Base change structure applies to \(S_G\). A seed without a finite,
independently verified normalized gap is retained as missing for this secondary
result and is not imputed with zero, infinity, a timeout penalty, or another
seed's value.

The final-gap result is secondary because exact MIP seed variation can primarily
change whether and when optimality is established, while final objective values
may be identical across successful optimal runs.

## Inference

For the primary result, secondary result, and their paired Base/Agent changes,
use the same inference structure as the frozen Seed Robustness protocol:

- fixed-instance, shared-seed-column paired percentile bootstrap;
- 5,000 bootstrap resamples;
- bootstrap seed `20260824`;
- nominal two-sided 99% intervals; and
- the same sampled seed columns for Base and Agent while holding instances fixed.

This reuses the approved bootstrap method, not the PyVRP seed domain. The
synthetic M3 cases and M5 real panel must still validate the method for the MIP
outcome distributions.

## Optimal coverage companion observation

For each code state, instance, budget, and seed list, report:

~~~text
verified-optimal count / 30
~~~

Coverage is a companion observation, not a component of either IQR and not a
new combined score. It has no M2 pass threshold in this candidate. The report
must make clear that an IQR of zero caused by every seed being censored at the
budget does not indicate good absolute solving performance.

## Retained records and interpretation boundaries

Retain solver status, termination, independent verification, objective, final
normalized gap, solver runtime, capped-time classification, seed, instance,
budget, code state, and every failure or missingness reason.

This candidate does not claim that the declared candidate range is the full
native HiGHS seed domain. It does not combine MIP Seed Robustness with Performance,
Operational Reliability, Configuration Robustness, or Representation
Robustness. Timeout remains a finite censored primary observation but is not a
semantic-validity pass.

## Open M2 decisions

The following choices are not selected by this candidate and require explicit
approval before M2 is closed:

- any MIP-specific M3 acceptance construction beyond the shared project-wide
  M5 validation gate.

The bootstrap method and the domain-and-list sampling structure were confirmed
for reuse on 2026-09-18. The declared range remains a project protocol choice,
not a native-domain completeness claim.
