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

This \(S(c,x,T)\) is the target over the complete `seed_domain`; it is not
computed directly. Let

\[
L_{\tau,s}
=
\{\xi_1,\ldots,\xi_R\},
\qquad
R=\texttt{seed_count},
\]

be the assigned development or evaluation `seed_list`. The point estimator is

\[
\widehat S(c,x,T;L_{\tau,s})
=
\widehat Q_{0.75}
\left(\{g(c,x,\xi_r;T)\}_{r=1}^{R}\right)
-
\widehat Q_{0.25}
\left(\{g(c,x,\xi_r;T)\}_{r=1}^{R}\right).
\]

Thus, \(S\) denotes the `seed_domain` estimand and \(\widehat S\) denotes its
`seed_list` estimate.

The per-instance robustness change induced by the patch is

\[
\Delta S(x,T)
=
S(\mathrm{Agent},x,T)
-
S(\mathrm{Base},x,T).
\]

Its `seed_list` estimate is

\[
\widehat{\Delta S}(x,T;L_{\tau,s})
=
\widehat S(\mathrm{Agent},x,T;L_{\tau,s})
-
\widehat S(\mathrm{Base},x,T;L_{\tau,s}).
\]

The direction semantics are:

- \(\Delta S(x,T) < 0\): Agent is more stable;
- \(\Delta S(x,T) > 0\): Agent is less stable; and
- \(\Delta S(x,T) = 0\): central spread is unchanged.

The same direction interpretation applies to \(\widehat{\Delta S}\).

### Sample-quantile convention

PitBench computes \(\widehat S\) with the Hyndman–Fan Type 7 sample quantile. For
\(n \ge 1\) sorted valid gap outcomes

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

The `seed_list` IQR estimate uses \(p=0.25\) and \(p=0.75\). This is the convention
implemented by NumPy's `quantile(..., method="linear")` and R's default
`quantile(..., type=7)`.

## Target seed distribution and seed-list sampling

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

Each task uses two disjoint seed lists sampled without replacement from \(D_\tau\)
and fixed before the agent starts:

- a visible development seed list for `agent_dev`; and
- a hidden evaluation seed list shared by `judge_id` and `judge_shift`.

The public field `seed_count` gives the number of seed identifiers in each list.
For v1,

```yaml
seed_count: 30
```

Thus, each task has 30 development seeds and 30 evaluation seeds. Because the
lists are disjoint, they contain 60 distinct seed identifiers in total.

The hidden evaluation seed list is not included in agent-visible configuration. The
seed list assigned to an instance-set role is reused across all corresponding
instances and budgets. Base and Agent use identical `(instance, budget, seed)`
conditions.

Every `(code_state, instance, seed, budget)` combination starts a fresh process and
reinitializes the solver RNG. Runs at different budgets do not share a trajectory or
solver state.

### Incomplete seed lists

For a fixed `(code_state, instance, budget)`, \(\widehat S(c,x,T;L_{\tau,s})\) is
available only when all 30 seeds in the assigned list produce a valid normalized
gap. If the list is incomplete, \(\widehat S(c,x,T;L_{\tau,s})\) is unavailable.
Missing outcomes are not imputed, assigned a penalty gap, or replaced with different
seeds.

Base and Agent completeness are evaluated separately. If only one code state has a
complete list, its own \(\widehat S(c,x,T;L_{\tau,s})\) remains available, while the
other code state's \(\widehat S(c,x,T;L_{\tau,s})\) and
\(\widehat{\Delta S}(x,T;L_{\tau,s})\) are unavailable. The valid outcomes and
failure records from an incomplete list remain in the evaluation record. Failure
records are not Seed Robustness outcomes, and the list does not produce a headline
IQR estimate.

An execution protocol may retry the same `(code_state, instance, seed, budget)`
tuple after an infrastructure failure. Such a retry does not change the assigned
seed list. The Seed Robustness protocol never substitutes a different seed.

### Cross-instance aggregation

Aggregation is performed separately for each task \(\tau\), `instance_set` \(k\),
and budget \(T\). Define the paired-complete instances as

\[
E_{\tau,k,T}
=
\left\{
x \in k
\;\middle|\;
\widehat S(\mathrm{Base},x,T;L_{\tau,s})
\text{ and }
\widehat S(\mathrm{Agent},x,T;L_{\tau,s})
\text{ are available}
\right\}.
\]

For code state \(c\), the paired `instance_set` mean Seed Robustness target is

\[
\overline{S}_{c,\tau,k,T}
=
\frac{1}{|E_{\tau,k,T}|}
\sum_{x \in E_{\tau,k,T}} S(c,x,T).
\]

The paired `instance_set` mean robustness-change target is

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

Their `seed_list` estimates are

\[
\widehat{\overline{S}}_{c,\tau,k,T}
=
\frac{1}{|E_{\tau,k,T}|}
\sum_{x \in E_{\tau,k,T}} \widehat S(c,x,T;L_{\tau,s}).
\]

\[
\widehat{\overline{\Delta S}}_{\tau,k,T}
=
\frac{1}{|E_{\tau,k,T}|}
\sum_{x \in E_{\tau,k,T}} \widehat{\Delta S}(x,T;L_{\tau,s})
=
\widehat{\overline{S}}_{\mathrm{Agent},\tau,k,T}
-
\widehat{\overline{S}}_{\mathrm{Base},\tau,k,T}.
\]

Each instance receives equal weight. Instances are not weighted by customer count,
gap, seed count, or runtime. A per-instance \(\widehat S(c,x,T;L_{\tau,s})\) that is
available for only one code state remains retained but does not enter a paired
aggregate. If \(|E_{\tau,k,T}|=0\), all three paired aggregates are unavailable. The
size \(|E_{\tau,k,T}|\) is retained to check aggregate coverage and is not a Seed
Robustness score.

The `agent_dev`, `judge_id`, and `judge_shift` `instance_set` values are aggregated
separately. Different tasks, `instance_set` values, and budgets are not pooled.
Aggregates are computed and retained at every configured budget; current primary
reporting selects `primary_budget_sec`.

### Confidence intervals

PitBench computes two-sided 99% paired percentile bootstrap confidence intervals
for \(\overline{S}_{\mathrm{Base},\tau,k,T}\),
\(\overline{S}_{\mathrm{Agent},\tau,k,T}\), and
\(\overline{\Delta S}_{\tau,k,T}\) by resampling their `seed_list` estimates with
crossed `instance_set` and `seed_list` resampling. Individual `(instance, seed)`
runs are not treated as iid observations.

Let \(N=|E_{\tau,k,T}|\) and \(R=\texttt{seed_count}\). The instances are ordered by
ascending `instance_id`, and the seeds retain their stored order in the assigned list.
For each of 5000 bootstrap replicates, PitBench:

- samples \(R\) seed indices independently with replacement from
  \(\{0,\ldots,R-1\}\) once for the whole replicate;
- samples \(N\) instance indices independently with replacement from
  \(\{0,\ldots,N-1\}\);
- applies the same sampled seed indices to every sampled instance and to both Base
  and Agent;
- recomputes each sampled instance's Type 7 IQR from its resampled seed outcomes;
  and
- computes all three aggregate statistics from the same crossed instance and seed
  indices.

Sampling the seed indices once per replicate preserves the assigned seeds as a
shared blocking factor across instances. Using the same crossed indices for Base and
Agent preserves their pairing.

Each `(task_id, instance_set, budget_sec)` group initializes a separate
`random.Random(20260824)` generator. Within each replicate, the protocol draws all
seed indices first and then all instance indices. For a seed index, it obtains \(u\)
from `Random.random()` and selects \(\lfloor Ru\rfloor\); for an instance index, it
selects \(\lfloor Nu\rfloor\). Resetting the generator for each group makes the
resampling independent of group traversal order.

For each aggregate statistic, the confidence interval endpoints are the
Hyndman–Fan Type 7 \(Q_{0.005}\) and \(Q_{0.995}\) of its 5000 bootstrap values.
If \(N=0\), the aggregate and confidence interval are unavailable. If \(N=1\), the
point aggregate remains available but its confidence interval is unavailable. If
\(N\ge2\), the interval is computed; an identical lower and upper endpoint is a
valid zero-width interval.

Confidence intervals are computed and retained at every configured budget; current
primary reporting selects `primary_budget_sec`. These intervals include
`instance_set` resampling uncertainty and uncertainty from estimating the
`seed_domain` IQR with the sampled `seed_list`. BCa intervals are not part of the v1
headline protocol; M3 may evaluate them as a sensitivity analysis. M3 must also
evaluate the coverage of the crossed percentile interval with `seed_count: 30`.

### Seed selection

Before the agent starts, PitBench uses the operating system's cryptographically secure
random source to sample an ordered list of `2 * seed_count` distinct identifiers
uniformly without replacement from the declared inclusive range from `seed_min` to
`seed_max`.

The first `seed_count` identifiers become `development_seeds`. The remaining
`seed_count` identifiers become `evaluation_seeds`. The two stored lists therefore
have equal size and are disjoint by construction. Seed order is retained because the
crossed bootstrap treats each shared seed as the same column across all instances and
both code states.

The public task configuration stores `seed_min`, `seed_max`, `seed_count`,
`development_seeds`, and `evaluation_seeds_file_sha256`. The evaluator-private seed
file stores `task_id` and `evaluation_seeds`. The file is fixed before the agent starts,
and its public SHA-256 digest commits the evaluator to that exact hidden list without
revealing it.

Changing the declared range, count, either stored list, or the private file requires a
new task or seed-list protocol version.

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
    seed_min: 0
    seed_max: 4294967295
    seed_count: 30
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
- Base `mean_seed_iqr` and `mean_seed_iqr_ci99`;
- Agent `mean_seed_iqr` and `mean_seed_iqr_ci99`; and
- `mean_seed_iqr_change` and `mean_seed_iqr_change_ci99`.

Every confidence interval stores `lower`, `upper`, `level`, `method`, `resamples`,
and `bootstrap_seed`. Unavailable estimates or intervals are encoded as null.
Numeric gap and IQR values remain proportions rather than formatted percentages.

The structured report retains all budgets. The current human-readable primary
report renders only `primary_budget_sec` for each `instance_set`. The report does
not emit a categorical robustness classification and does not combine Robustness
with Performance.

The public report contains aggregate evidence only. A separate evaluator-private
`seed_robustness_details` artifact retains the complete seed identifiers,
seed-to-gap outcomes, seed-list completeness, per-instance IQR values, and empirical
distributions and ECDFs required to reproduce the aggregates. Its artifact reference
must set `private: true`. Active-task public or agent-facing outputs must not expose
hidden seed identifiers, seed-to-gap mappings, detailed hidden ECDFs, or raw hidden
observations.

Seed median, MAD, and tail probability are not fields in the v1 public report. MAD
and tail probability are not v1 diagnostics. They may be studied later from the
retained detailed results under a new protocol decision.

## Retired-task seed publication

A task is retired for this protocol only when it permanently stops accepting scored
submissions and its seed lists will not be used in any future evaluation. At that
point, PitBench must publish the exact evaluation-seeds file and the previously private
`seed_robustness_details` artifact.

Before publication, PitBench recomputes the file SHA-256 and requires it to match
`evaluation_seeds_file_sha256` from the task configuration fixed before the agent
started. The published development and evaluation lists and detailed results allow third
parties to recompute the report.

A retired task cannot resume under the disclosed seed lists. Any later reactivation
requires a new task or seed-list protocol version and fresh seed lists.

## Retained evidence object

The complete per-instance empirical seed distribution and its ECDF are retained in
the evaluator-private detailed results. They are not the headline scalar.

## Open formal-specification items

None.
