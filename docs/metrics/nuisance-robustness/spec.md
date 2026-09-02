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

### Deterministic panel sampler

The panel sampler identifier is `hmac_sha256_rank_v1`. Each task \(\tau\) has a
distinct 32-byte `panel_master_key` \(K_\tau\), generated by the operating system's
cryptographic RNG and fixed before the agent starts. A key must not be reused across
tasks or panel protocol versions. The exact key is stored only in an
evaluator-private audit record.

For admissible seed identifier \(d\) and panel label \(l\), the sampler computes

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

The public key commitment is

\[
C_\tau
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

The public field `panel_key_commitment_sha256` stores \(C_\tau\) as 64 lowercase
hexadecimal characters. Public metadata separately stores the sampler identifier,
task identifier, domain declaration, and `seed_count`. Changing the domain,
`seed_count`, or the key requires a new panel protocol version.

## Retained evidence object

The complete per-instance empirical seed distribution and its ECDF are retained as
evidence objects. They are not the headline scalar.

MAD and tail probability are not headline estimands. Whether either is reported as
a diagnostic remains undecided.

## Open formal-specification items

- confidence interval;
- report schema; and
- retired-task key disclosure policy.
