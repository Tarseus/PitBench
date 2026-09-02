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

## Retained evidence object

The complete per-instance empirical seed distribution and its ECDF are retained as
evidence objects. They are not the headline scalar.

MAD and tail probability are not headline estimands. Whether either is reported as
a diagnostic remains undecided.

## Open formal-specification items

- target seed distribution and sampling protocol;
- sample-quantile convention;
- seed count;
- missing-seed rules;
- cross-instance aggregation;
- confidence interval; and
- report schema.
