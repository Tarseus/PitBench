# A provable metric for standard metric CVRP instances

Status: definition 1.0, complete metric proof, exact finite reference, and
independently verifiable map-pair upper-bound certificates.

This document defines a solver-independent metric on a declared quotient of
standard symmetric metric CVRP instances. It does not use QoIs, population-fitted
normalization, solver outcomes, or experiment-conditioned feature selection.

## 1. Scope and equivalence convention

An instance is

\[
I=(X_I,d_I,q_I,Q_I,o_I),
\]

where `X_I` is a finite node set, `o_I` is the depot, `d_I` is a strict metric,
`Q_I>0`, the depot demand is zero, and every customer demand lies in `(0,Q_I]`.
The current implementation constructs `d_I` from distinct two-dimensional
Euclidean coordinates. It covers the standard unlimited-fleet CVRP objective; a
fixed fleet size, time windows, service times, asymmetric costs, or other side
constraints are outside this domain and would have to be added to the object.

Let

\[
\operatorname{diam}(I)=\max_{x,x'\in X_I}d_I(x,x'),\qquad
\bar d_I=d_I/\operatorname{diam}(I),\qquad
a_I(x)=q_I(x)/Q_I.
\]

Two instances are declared equivalent, written `I ~ J`, if there are a bijection
`f:X_I -> X_J` and a scalar `c>0` such that

\[
f(o_I)=o_J,\qquad
d_J(f(x),f(x'))=c\,d_I(x,x'),\qquad
a_J(f(x))=a_I(x).
\]

Thus customer names, customer order, Euclidean pose, routing-distance units, and
common demand/capacity units have no semantic effect. Scaling all routing costs is
explicitly treated as objective-unit equivalence: it preserves feasible routes and
their ordering, although it rescales every objective value. If absolute objective
scale is later declared semantic, diameter normalization must be removed and that
would define a different metric version.

The demand marks are complete for standard capacity feasibility because a route
`S` is feasible exactly when

\[
\sum_{x\in S}a_I(x)\le 1.
\]

## 2. Anchored marked correspondences

An anchored correspondence between `I` and `J` is a relation

\[
R\subseteq X_I\times X_J
\]

whose projections cover both node sets and which respects the depot partition:

\[
(o_I,o_J)\in R,
\]

\[
(x,y)\in R\Longrightarrow
\bigl(x=o_I\iff y=o_J\bigr).
\]

Unlike a bijection, a correspondence exists when the customer counts differ. Its
relational distortion and mark discrepancy are

\[
\operatorname{dis}(R)=
\max_{(x,y),(x',y')\in R}
\left|\bar d_I(x,x')-\bar d_J(y,y')\right|,
\]

\[
\operatorname{mark}(R)=
\max_{(x,y)\in R}|a_I(x)-a_J(y)|.
\]

Define the anchored marked CVRP metric

\[
\boxed{
\delta_{\mathrm{CVRP}}(I,J)
=
\min_{R\in\operatorname{Corr}_o(I,J)}
\max\{\operatorname{dis}(R),\operatorname{mark}(R)\}.
}
\]

This is an anchored, marked variant of the correspondence formulation underlying
Gromov--Hausdorff distance. The conventional factor `1/2` is omitted; multiplying a
metric by a fixed positive constant does not affect its metric properties. Because
both normalized distances and demand fractions lie in `[0,1]`, so does
`delta_CVRP`. There is no fitted scale and no geometry-versus-demand weight.

## 3. Metric theorem

**Theorem.** `delta_CVRP` is a metric on the quotient space of finite standard
metric CVRP instances under `~`.

### 3.1 Existence and non-negativity

Finite node sets have finitely many relations, and at least one anchored
correspondence always exists: pair the depots and use the complete relation between
the two customer sets. The displayed minimum is therefore attained. Every term is
an absolute value, so `delta_CVRP >= 0`.

### 3.2 Symmetry

If `R` is an anchored correspondence from `I` to `J`, then

\[
R^{-1}=\{(y,x):(x,y)\in R\}
\]

is anchored from `J` to `I`. Absolute differences give

\[
\operatorname{dis}(R^{-1})=\operatorname{dis}(R),\qquad
\operatorname{mark}(R^{-1})=\operatorname{mark}(R).
\]

Taking minima in both directions proves symmetry.

### 3.3 Identity of indiscernibles on the quotient

If `I ~ J`, the graph of the witnessing bijection is an anchored correspondence
with zero normalized-distance distortion and zero mark discrepancy. Hence
`delta_CVRP(I,J)=0`.

Conversely, suppose `delta_CVRP(I,J)=0`. Finiteness gives an attained zero-cost
correspondence `R`. If one node `x` were related to two nodes `y` and `y'`, zero
distortion would imply

\[
d_J(y,y')/\operatorname{diam}(J)
=d_I(x,x)/\operatorname{diam}(I)=0.
\]

Strictness of `d_J` forces `y=y'`. The symmetric argument shows that each `y` is
related to only one `x`. Surjectivity of both projections therefore turns `R` into a
bijection. Anchoring preserves the depot, zero mark cost preserves every `q/Q`, and
zero distortion gives

\[
d_J(f(x),f(x'))
=
\frac{\operatorname{diam}(J)}{\operatorname{diam}(I)}d_I(x,x').
\]

Thus `I ~ J`. Therefore zero distance identifies exactly one quotient element.

### 3.4 Triangle inequality

Let `R` correspond `I` with `J`, and let `S` correspond `J` with `K`. Their
relational composition

\[
T=S\circ R
=\{(x,z):\exists y,\;(x,y)\in R,\;(y,z)\in S\}
\]

is an anchored correspondence from `I` to `K`. For any two pairs in `T`, choose
their intermediate nodes `y,y'`. The ordinary triangle inequality on real numbers
gives

\[
|\bar d_I(x,x')-\bar d_K(z,z')|
\le
|\bar d_I(x,x')-\bar d_J(y,y')|
+|\bar d_J(y,y')-\bar d_K(z,z')|.
\]

Consequently,

\[
\operatorname{dis}(T)
\le \operatorname{dis}(R)+\operatorname{dis}(S).
\]

The demand marks obey the same argument:

\[
\operatorname{mark}(T)
\le \operatorname{mark}(R)+\operatorname{mark}(S).
\]

Using

\[
\max(a+c,b+d)\le\max(a,b)+\max(c,d)
\]

and then minimizing over `R` and `S` proves

\[
\delta_{\mathrm{CVRP}}(I,K)
\le
\delta_{\mathrm{CVRP}}(I,J)
+\delta_{\mathrm{CVRP}}(J,K).
\]

This completes the metric proof.

## 4. Exact reference computation

The implementation is in `pitbench/metrics/cvrp_problem.py`. For `n` left and `m`
right customers, it enumerates unions of two graphs:

\[
f:X_I\setminus\{o_I\}\to X_J\setminus\{o_J\},\qquad
g:X_J\setminus\{o_J\}\to X_I\setminus\{o_I\}.
\]

There are `m^n n^m` such configurations. This enumeration is exact. To see why,
take any anchored correspondence `R`, select one partner `f(x)` for every left
customer and one partner `g(y)` for every right customer, and call the union of
those selected pairs `R'`. Then `R'` remains an anchored correspondence,
`R' subseteq R`, and deleting pairs cannot increase either maximum in the cost.
Therefore some optimum is present in the enumerated family.

The algorithm deliberately raises `ExactMetricLimitError` above a caller-selected
configuration limit. It is a small-instance definition oracle, not a scalable
solver. An entropic, greedy, or learned approximation must use a different API and
must not be advertised as satisfying the exact metric theorem unless separately
proved.

## 5. Certified map-pair upper bounds

Exact enumeration is not needed to verify an upper bound. Let `f:X_I -> X_J` and
`g:X_J -> X_I` preserve the depot partition, and define

\[
D_f=\max_{x,x'}
|\bar d_I(x,x')-\bar d_J(f(x),f(x'))|,
\]

\[
D_g=\max_{y,y'}
|\bar d_I(g(y),g(y'))-\bar d_J(y,y')|,
\]

\[
C_{f,g}=\max_{x,y}
|\bar d_I(x,g(y))-\bar d_J(f(x),y)|.
\]

The last quantity is the codistortion. It is required: the two maps may each be
low-distortion while being mutually incompatible. For the demand marks, define

\[
Q_f=\max_x|a_I(x)-a_J(f(x))|,
\qquad
Q_g=\max_y|a_I(g(y))-a_J(y)|.
\]

Under PitBench definition 1.0, a map pair certifies

\[
\boxed{
U(f,g)=\max\{D_f,D_g,C_{f,g},Q_f,Q_g\}.
}
\]

This formula deliberately has no factor `1/2` and no fitted demand weight, matching
the convention fixed in Section 2. Under the conventional GH normalization, all
travel-distortion terms would instead receive the same factor `1/2`; that is not the
version-1.0 PitBench metric.

Construct the anchored correspondence

\[
R_{f,g}=\{(x,f(x)):x\in X_I\}\cup\{(g(y),y):y\in X_J\}.
\]

Its travel distortion is exactly `max(D_f,D_g,C_f,g)`, and its mark discrepancy is
exactly `max(Q_f,Q_g)`. Therefore

\[
\delta_{\mathrm{CVRP}}(I,J)\le U(f,g).
\]

Conversely, selecting one partner in each direction from any anchored
correspondence produces a map pair whose union is a subrelation and cannot have a
larger cost. Consequently,

\[
\boxed{
\delta_{\mathrm{CVRP}}(I,J)=\inf_{f,g}U(f,g).
}
\]

### 5.1 Proposer--verifier API

`pitbench.metrics.cvrp_certificate` separates untrusted search from deterministic
verification:

- `CVRPMapCertificate` stores the two maps as `O(n+m)` integers and supports a
  JSON-compatible dictionary representation;
- `verify_cvrp_map_certificate` validates depot preservation and independently
  recomputes all five maxima in `O(n²+m²+nm)` time;
- `signature_cvrp_map_certificate` proposes a deterministic initial witness using
  normalized demand, depot distance, and distance-distribution quantiles;
- `search_cvrp_upper_bound` applies budgeted coordinated reassignment/swap moves.
  Every accepted move is reverified, and `upper_bound_history` is monotonically
  decreasing. Evaluation-count and wall-time budgets make the search anytime.

The proposer has no trusted role. A certificate produced by a heuristic, a learned
model, MILP, CP-SAT, or an external program has the same status after the verifier
recomputes its bound.

### 5.2 Interpretation and propagation

`U(f,g)` is a certified upper approximation to `delta_CVRP`; it is not itself a new
metric and is not claimed to satisfy the triangle inequality. The exact reference
can measure certificate tightness on small instances. A nontrivial lower-bound
relaxation is not part of definition 1.0, so the only universally available interval
without additional machinery is `[0,U]`.

If every ground cost used by an outer population transport problem is replaced by
a certified upper bound `U_ij >= delta_ij`, minimizing the transport objective with
those costs yields an upper bound on the true population Wasserstein distance. Thus
the certificate property propagates through the outer optimal-transport layer.

## 6. What the theorem does and does not claim

The theorem proves that the declared formula is a legal metric for the declared
CVRP semantics. It does not prove that this is the unique possible CVRP metric, that
its magnitude predicts solver behavior, or that every domain expert must accept
objective-scale equivalence. Those are not metric axioms.

The current QoI feature vector remains useful for interpretation and controlled
interventions, but it is not part of this definition. Population Wasserstein distance
can subsequently use `delta_CVRP` as its fixed ground metric; solver robustness is a
separate layer and cannot validate or invalidate the theorem above.
