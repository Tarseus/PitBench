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
- \(\xi\) is solver seed under the target seed distribution and sampling protocol
  defined below.

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

### Sample-quantile convention

PitBench uses the Hyndman–Fan Type 7 sample quantile. For \(n \ge 1\) sorted
valid gap outcomes

\[
g_{(1)} \le \cdots \le g_{(n)},
\]

define

\[
h = 1 + (n - 1)p,
\qquad
j = \lfloor h \rfloor,
\qquad
\lambda = h - j,
\]

and

\[
Q_p
=
(1 - \lambda)g_{(j)}
+
\lambda g_{(\lceil h \rceil)}.
\]

The headline IQR uses \(p=0.25\) and \(p=0.75\). This is the convention
implemented by NumPy's `quantile(..., method="linear")` and R's default
`quantile(..., type=7)`.

## Target seed distribution and panel sampling

Each stochastic VRP task \(\tau\) declares a finite admissible seed domain
\(D_\tau\) supported by its native solver. The target seed distribution is

\[
\xi_\tau \sim \operatorname{Uniform}(D_\tau).
\]

This distribution is uniform over seed identifiers. It does not assert that the
solver's internal RNG states are uniformly distributed.

### Current task domains

The current in-scope PyVRP tasks use the complete unsigned 32-bit seed range
accepted by both the native `RandomNumberGenerator` constructor and its Python
binding:

\[
D_\tau
=
\{d \in \mathbb{Z} \mid 0 \le d \le 2^{32} - 1\}.
\]

This domain applies to:

- `pyvrp_v0_12_2`;
- `pyvrp_v0_13_0`;
- `pyvrp_v0_13_4`; and
- `pyvrp_v0_14_0`.

Each task uses two disjoint seed panels sampled without replacement from \(D_\tau\)
and fixed before the agent starts:

- a visible development panel for `agent_dev`; and
- a hidden evaluation panel shared by `judge_id` and `judge_shift`.

The public field `seed_count` gives the number of seed identifiers in each panel.
For v1,

```yaml
seed_count: 10
```

Thus, each task has 10 development seeds and 10 evaluation seeds. Because the
panels are disjoint, they contain 20 distinct seed identifiers in total.

The hidden evaluation panel is not included in agent-visible configuration. The
panel assigned to an instance-set role is reused across all corresponding instances
and budgets. Base and Agent use identical `(instance, budget, seed)` conditions.

Every `(code_state, instance, seed, budget)` combination starts a fresh process and
reinitializes the solver RNG. Runs at different budgets do not share a trajectory or
solver state.

### Incomplete seed panels

For a fixed `(code_state, instance, budget)`, \(S(c,x,T)\) is available only when
all 10 seeds in the assigned panel produce a valid normalized gap. If the panel is
incomplete, \(S(c,x,T)\) is unavailable. Missing outcomes are not imputed, assigned
a penalty gap, or replaced with different seeds.

Base and Agent completeness are evaluated separately. If only one code state has a
complete panel, its own \(S(c,x,T)\) remains available, while the other code state's
\(S(c,x,T)\) and \(\Delta S(x,T)\) are unavailable. The valid outcomes and failure
records from an incomplete panel remain in the evaluation record. Failure records
are not Seed Robustness outcomes, and the panel does not produce a headline IQR.

An execution protocol may retry the same `(code_state, instance, seed, budget)`
tuple after an infrastructure failure. Such a retry does not change the assigned
seed panel. The Seed Robustness protocol never substitutes a different seed.

### Cross-instance aggregation

Aggregation is performed separately for each task \(\tau\), `instance_set` \(k\),
and budget \(T\). Define the paired-complete instances as

\[
E_{\tau,k,T}
=
\left\{
x \in k
\;\middle|\;
S(\mathrm{Base},x,T)
\text{ and }
S(\mathrm{Agent},x,T)
\text{ are available}
\right\}.
\]

For code state \(c\), the paired `instance_set` mean Seed Robustness is

\[
\overline{S}_{c,\tau,k,T}
=
\frac{1}{|E_{\tau,k,T}|}
\sum_{x \in E_{\tau,k,T}} S(c,x,T).
\]

The paired `instance_set` mean robustness change is

\[
\overline{\Delta S}_{\tau,k,T}
=
\frac{1}{|E_{\tau,k,T}|}
\sum_{x \in E_{\tau,k,T}} \Delta S(x,T)
=
\overline{S}_{\mathrm{Agent},\tau,k,T}
-
\overline{S}_{\mathrm{Base},\tau,k,T}.
\]

Each instance receives equal weight. Instances are not weighted by customer count,
gap, seed count, or runtime. A per-instance \(S(c,x,T)\) that is available for only
one code state remains retained but does not enter a paired aggregate. If
\(|E_{\tau,k,T}|=0\), all three paired aggregates are unavailable. The size
\(|E_{\tau,k,T}|\) is retained to audit aggregate coverage and is not a Seed
Robustness score.

The `agent_dev`, `judge_id`, and `judge_shift` `instance_set` values are aggregated
separately. Different tasks, `instance_set` values, and budgets are not pooled.
Aggregates are computed and retained at every configured budget; current primary
reporting selects `primary_budget_sec`.

### Confidence intervals

PitBench computes two-sided 95% paired percentile bootstrap confidence intervals
for \(\overline{S}_{\mathrm{Base},\tau,k,T}\),
\(\overline{S}_{\mathrm{Agent},\tau,k,T}\), and
\(\overline{\Delta S}_{\tau,k,T}\). The resampling unit is one paired-complete
instance from \(E_{\tau,k,T}\).

Let \(N=|E_{\tau,k,T}|\). The instances are ordered by ascending `instance_id`.
For each of 5000 bootstrap replicates, PitBench samples \(N\) indices independently
with replacement and computes all three aggregate statistics from the same sampled
indices. This preserves Base–Agent pairing within every replicate.

Each `(task_id, instance_set, budget_sec)` group initializes a separate
`random.Random(20260824)` generator. For every sampled index, the protocol obtains
\(u\) from `Random.random()` and selects the zero-based index
\(\lfloor Nu\rfloor\). Resetting the generator for each group makes the resampling
independent of group traversal order.

For each aggregate statistic, the confidence interval endpoints are the
Hyndman–Fan Type 7 \(Q_{0.025}\) and \(Q_{0.975}\) of its 5000 bootstrap values.
If \(N=0\), the aggregate and confidence interval are unavailable. If \(N=1\), the
point aggregate remains available but its confidence interval is unavailable. If
\(N\ge2\), the interval is computed; an identical lower and upper endpoint is a
valid zero-width interval.

Confidence intervals are computed and retained at every configured budget; current
primary reporting selects `primary_budget_sec`. These intervals express instance
resampling uncertainty only. They do not include seed-panel sampling uncertainty.
BCa intervals are not part of the v1 headline protocol; M3 may evaluate them as a
sensitivity analysis.

### Deterministic seed selection

The seed selection `method` is `hmac_sha256_rank_v1`. Each task \(\tau\) has a
distinct 32-byte seed selection key \(K_\tau\), generated by the operating system's
cryptographic RNG and fixed before the agent starts. A key must not be reused across
tasks or panel protocol versions. The evaluator-private field
`seed_selection.key_hex` stores the exact key as 64 lowercase hexadecimal
characters.

For admissible seed identifier \(d\) and panel label \(l\), the selection method
computes

\[
h_{\tau,l}(d)
=
\operatorname{HMAC-SHA256}
\left(
K_\tau,
\operatorname{encode}
\left(
\texttt{pitbench-seed-panel-v1},
\texttt{task_id},
l,
d
\right)
\right).
\]

The encoded fields are UTF-8 and separated by a single NUL byte. A seed identifier
is encoded as decimal ASCII without leading zeroes. The admissible seed domain is
normalized in ascending integer order. Digest ties are broken by ascending seed
identifier.

The development panel consists of the first `seed_count` seeds ranked with the
`development` label. The evaluation panel consists of the first `seed_count`
remaining seeds ranked with the `evaluation` label.

The public key verification digest is

\[
V_\tau
=
\operatorname{SHA256}
\left(
\texttt{pitbench-seed-panel-key-v1}
\mathbin\Vert \texttt{NUL}
\mathbin\Vert \operatorname{UTF8}(\texttt{task_id})
\mathbin\Vert \texttt{NUL}
\mathbin\Vert K_\tau
\right).
\]

The public field `key_verification_sha256` stores \(V_\tau\) as 64 lowercase
hexadecimal characters. Active-task metadata sets `key_visibility` to `private` and
does not contain the exact key. Public metadata also stores `method`, `seed_min`,
`seed_max`, and `seed_count`. Changing any of those fields or the key requires a new
panel protocol version.

## Report schema

The structured evaluation summary stores Seed Robustness under the separate
`nuisance_robustness` field. Its public shape is

```yaml
nuisance_robustness:
  task_id: <task ID>
  metric: seed_robustness
  primary_budget_sec: <declared primary budget>
  budgets_sec: <all configured budgets>
  seed_selection:
    method: hmac_sha256_rank_v1
    seed_min: 0
    seed_max: 4294967295
    seed_count: 10
    key_visibility: private
    key_verification_sha256: <64 lowercase hexadecimal characters>
  by_instance_set: <one entry per instance_set>
```

Each `by_instance_set` entry stores its `instance_set_kind`, a `primary` budget
cell, and a `by_budget` mapping containing every configured budget. A budget cell
stores:

- `budget_sec`;
- `instance_count`;
- `base_complete_instance_count`;
- `agent_complete_instance_count`;
- `paired_complete_instance_count`;
- Base `mean_seed_iqr` and `mean_seed_iqr_ci95`;
- Agent `mean_seed_iqr` and `mean_seed_iqr_ci95`; and
- `mean_seed_iqr_change` and `mean_seed_iqr_change_ci95`.

Every confidence interval stores `lower`, `upper`, `level`, `method`, `resamples`,
and `seed`. Unavailable estimates or intervals are encoded as null. Numeric gap and
IQR values remain proportions rather than formatted percentages.

The structured report retains all budgets. The current human-readable primary
report renders only `primary_budget_sec` for each `instance_set`. The report does
not emit a categorical robustness classification and does not combine Robustness
with Performance.

The public report contains aggregate evidence only. A separate evaluator-private
`seed_robustness_audit_evidence` artifact retains the complete seed identifiers,
seed-to-gap outcomes, panel completeness, per-instance IQR values, and empirical
distributions and ECDFs required to reproduce the aggregates. Its artifact reference
must set `private: true`. Active-task public or agent-facing outputs must not expose
hidden seed identifiers, seed-to-gap mappings, detailed hidden ECDFs, or raw hidden
observations.

Seed median, MAD, and tail probability are not fields in the v1 public report. MAD
and tail probability are not v1 diagnostics. They may be studied later from the
retained audit evidence under a new protocol decision.

## Retired-task key publication

A task is retired for this protocol only when it permanently stops accepting scored
submissions and its seed panels will not be used in any future evaluation. At that
point, PitBench must publish the seed selection key and the previously private
`seed_robustness_audit_evidence` artifact.

The publication record stores:

```yaml
task_id: <task ID>
retired_at: <timestamp>
seed_selection:
  method: hmac_sha256_rank_v1
  seed_min: 0
  seed_max: 4294967295
  seed_count: 10
  key_visibility: published
  key_verification_sha256: <64 lowercase hexadecimal characters>
  key_hex: <64 lowercase hexadecimal characters>
  key_published_at: <timestamp>
```

Before publication, PitBench recomputes `key_verification_sha256` from `key_hex` and
rejects a record that does not match the value fixed before the agent started. The
published key and audit evidence allow third parties to reconstruct both panels and
recompute the report.

A retired task cannot resume under the disclosed key or panels. Any later
reactivation requires a new task or panel protocol version, a new seed selection
key, and new panels. A published key is never reused.

## Retained evidence object

The complete per-instance empirical seed distribution and its ECDF are retained in
the evaluator-private audit evidence. They are not the headline scalar.

## Open formal-specification items

None.
