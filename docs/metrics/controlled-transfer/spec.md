# Controlled Transfer specification

## M2: task-native benefit under controlled conditions

Controlled Transfer evaluates one frozen Agent modification against the unmodified
Base for one declared task context:

$$
h=(\text{solver version},\text{problem variant},\text{formulation},
\text{Performance protocol},\text{resource contract}).
$$

The metric does not compare benefit values across task contexts. Each evaluation
condition \(c\) belongs to a declared panel and identifies its reference condition,
scale setting, structural-domain setting, joint-condition status, declared budget, and
complete planned Base/Agent run grid.

Scale and structural-domain changes are reporting views of the same metric. A scale
condition changes scale along a declared path. A structural condition changes the
declared structural domain at its matching scale. A joint condition changes both. The
panel declaration supplies the concrete scale values and domain identities; this
specification does not choose them.

## Task-native paired benefit

For a fixed-budget heuristic task, let \(\ell_a(x,r;b)\) be the task's existing
independently verified normalized objective gap for code state \(a\), instance \(x\),
solver seed \(r\), and budget \(b\). It retains the task's objective-sense, anchor,
and validity rules.

For an exact task, let \(P_a(x,r;b)\) be the existing `exact_verified_solve`
penalized runtime: the task-declared time measurement for a verified solved run and
\(2b\) for a solver-origin unsolved run. For reporting a budget-normalized time
effect, define:

$$
\ell_a(x,r;b)=\frac{P_a(x,r;b)}{b}.
$$

This normalization is not an unbounded speedup measure. It does not change the
task-native exact Performance classification.

Base and Agent are paired by the task-native run key:

```text
(instance_set, instance_id, solver_seed, budget_sec)
```

For each instance, form the mean of its valid paired seed differences. The condition
benefit is the equal-instance-weight mean:

$$
G_h(c)=
\operatorname{mean}_{x\mid c}\left[
  \operatorname{mean}_{r\mid x,c}
  \bigl(\ell_B(x,r;b)-\ell_A(x,r;b)\bigr)
\right].
$$

A positive \(G_h(c)\) is a task-native time or quality benefit for Agent over Base.

For exact tasks, each condition follows the coverage-first Performance conclusion.
Report paired Agent-minus-Base verified-solved coverage and an instance-bootstrap
interval, clustering declared template siblings together. A coverage difference is
`improved` or `regressed` only when its interval excludes zero in that direction;
an unresolved difference is `inconclusive`. Only exactly equal coverage is then
classified by the paired penalized-runtime interval. The normalized time benefit
\(G_h(c)\) is reported for retention analysis and cannot override an unresolved
coverage difference.

## Retention reporting

For a target condition \(c\) and its declared reference condition \(c_0\), define
the raw benefit-retention ratio:

$$
R_h(c)=\frac{G_h(c)}{G_h(c_0)}.
$$

Report \(R_h(c)\) only when the lower confidence bound for
\(G_h(c_0)\) is strictly greater than \(\gamma=0\). Otherwise report
`reference_improvement_unestablished` and `null` for the ratio. A negative raw ratio
is retained; it is not clipped.

The required retention fraction is \(\rho=0.8\). Its margin is:

$$
M_h(c)=G_h(c)-0.8G_h(c_0).
$$

The scale view reports each declared scale condition's benefit, ratio when defined,
and retention margin. Its largest continuously supported tested scale is the greatest
declared target scale for which every preceding target condition has a positive
simultaneous lower bound for \(M_h\) and has no `incomplete` result. This is limited
to the declared test points and is not a claim about untested intermediate scales.

The structural-domain and joint-condition views report the same quantities for every
declared matching condition. They do not combine solver or task contexts into a
single score.

## Inference

For every condition estimate, first average paired solver-seed differences within an
instance and then resample instances with replacement. Solver seeds are not resampled
as a second level. Use 5,000 bootstrap resamples, a fixed bootstrap seed of
`20260824`, and 95% percentile intervals for individual-condition reporting.

Every bootstrap replicate jointly recomputes the declared reference benefit and all
target benefits in its reported claim family before recomputing retention ratios and
margins. When instances share a declared template, resampling occurs by template
cluster rather than by individual instance.

For a joint conclusion about \(m\) bounds, including the reference-benefit bound when
that claim depends on it, use Bonferroni bounds with family-wise \(\alpha=0.05\).
Each two-sided interval uses tail probability \(0.05/(2m)\); each one-sided lower
bound uses tail probability \(0.05/m\). A target condition supports 80% retention
only when its simultaneous lower bound for \(M_h(c)\) is strictly greater than zero.

## Completeness and existing qualification semantics

A Controlled Transfer condition is `incomplete` when any planned Base or Agent run
has an infrastructure error, is missing, or lacks a usable task-native Performance
result. Such runs are not silently dropped to form a favorable condition estimate.

Existing exact Performance semantics are preserved: solver-origin timeout, out of
memory, crash, solver error, or other unsolved outcome remains a valid \(2b\)
outcome, not an `incomplete` result. Qualification failures remain Qualification
failures; they are not converted into finite benefits or `incomplete` observations.
An oracle discrepancy remains outside ordinary aggregation under the existing exact
Performance protocol.

This specification does not select instance generators, scale levels, structural
domains, budget policies, or an implementation schema. Those must be declared by a
future panel protocol before a Controlled Transfer result is produced.
