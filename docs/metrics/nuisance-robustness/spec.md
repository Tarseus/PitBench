# Nuisance Robustness formal specification

## Per-instance Seed Robustness headline

For code state \(c\), fixed instance \(x\), and fixed budget \(T\), define the
target Seed Robustness estimand as

\[
S(c,x,T)
=
Q_{0.75}\!\left(g(c,x,\xi;T)\right)
-
Q_{0.25}\!\left(g(c,x,\xi;T)\right),
\]

where:

- \(c \in \{\mathrm{Base},\mathrm{Agent}\}\);
- \(g\) is normalized gap from a valid run; and
- \(\xi\) is solver seed under a target seed distribution and sampling protocol that
  remain to be specified.

Smaller \(S\) means that the central seed outcomes are more stable.

The per-instance robustness change induced by the patch is

\[
\Delta S(x,T)
=
S(\mathrm{Agent},x,T)
-
S(\mathrm{Base},x,T).
\]

The direction semantics are:

- \(\Delta S(x,T) < 0\): Agent is more stable;
- \(\Delta S(x,T) > 0\): Agent is less stable; and
- \(\Delta S(x,T) = 0\): central spread is unchanged.

## Target seed distribution and panel sampling

Each stochastic VRP task \(\tau\) declares a finite admissible seed domain
\(D_\tau\) supported by its native solver. The target seed distribution is

\[
\xi_\tau \sim \operatorname{Uniform}(D_\tau).
\]

This distribution is uniform over seed identifiers. It does not assert that the
solver's internal RNG states are uniformly distributed.

Each task uses two disjoint seed panels sampled without replacement from \(D_\tau\)
and fixed before the agent starts:

- a visible development panel for `agent_dev`; and
- a hidden evaluation panel shared by `judge_id` and `judge_shift`.

The hidden evaluation panel is not included in agent-visible configuration. The
panel assigned to an instance-set role is reused across all corresponding instances
and budgets. Base and Agent use identical `(instance, budget, seed)` conditions.

Every `(code_state, instance, seed, budget)` combination starts a fresh process and
reinitializes the solver RNG. Runs at different budgets do not share a trajectory or
solver state.

## Retained evidence object

The complete per-instance empirical seed distribution and its ECDF are retained as
evidence objects. They are not the headline scalar.

MAD and tail probability are not headline estimands. Whether either is reported as
a diagnostic remains undecided.

## Open formal-specification items

- deterministic panel sampler and master seed;
- exact task-specific admissible seed domains;
- sample-quantile convention;
- seed count;
- missing-seed rules;
- cross-instance aggregation;
- confidence interval; and
- report schema.
