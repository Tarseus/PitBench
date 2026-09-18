# Performance evidence

## M1: exact-optimization evaluation and verification

This evidence item records the approved conclusions from a targeted primary-source
review of SAT, MaxSAT, constraint-programming, and mixed-integer-programming
competitions and verification research. It distinguishes competition practice from a
formal proof of optimality and does not define the remaining PitBench estimators.

## SAT Competition: verified decision results and PAR-2

The SAT Competition 2025 Main Track rules were inspected from the official
`satcompetition/2025` repository at commit
`5f286f4443ab2e1f3c1deded3255d8500ae46e2e`.

The official
[call for solvers and benchmarks](https://github.com/satcompetition/2025/blob/5f286f4443ab2e1f3c1deded3255d8500ae46e2e/README.md)
states that solvers are ranked by penalized average runtime, PAR-2. A solved instance
contributes its runtime and an unsolved instance contributes twice the time limit.

The official
[output requirements](https://github.com/satcompetition/2025/blob/5f286f4443ab2e1f3c1deded3255d8500ae46e2e/output.html)
require a satisfying assignment for a SAT result and an UNSAT proof for Main Track
unsatisfiability results. The proof is passed to a selected proof checker.

The official
[competition rules](https://github.com/satcompetition/2025/blob/5f286f4443ab2e1f3c1deded3255d8500ae46e2e/rules.html)
state that a solver producing a wrong answer or wrong certificate is disqualified. An
incorrect SAT model or an UNSAT proof rejected by the selected checker is not merely an
unsolved run with a runtime penalty.

The official
[competition presentation](https://github.com/satcompetition/2025/blob/5f286f4443ab2e1f3c1deded3255d8500ae46e2e/slides/satcomp25slides.tex)
states that an instance is not solved when its proof checker times out. It records a
5000-second solver limit and a separate 45000-second checker-toolchain limit. The
sequential Main Track plots use CPU time.

SAT is a decision problem with a binary final result. Its rules provide a strong
precedent for a correctness qualification, explicit unsolved outcomes, and penalized
runtime. They do not by themselves establish that PAR-2 should be the sole primary
measure for exact optimization.

## MaxSAT Evaluation: optimal-solution coverage first

The official
[MaxSAT Evaluation 2024 rules](https://github.com/maxsat-evaluations/maxsat-evaluations.github.io/blob/65ff14b9719c68d861c196d1d98c46ca98d24603/2024/contents/contents_rules.html)
rank exact solvers by the number of instances solved within predefined resource limits.
For the exact track, solving means finding an optimal solution. A nonoptimal incumbent
has no exact-track score.

The rules use returned assignments and known solutions to identify incorrect results.
A solver is buggy when its assignment violates hard clauses, its reported cost differs
from the assignment cost, or it reports `OPTIMUM FOUND` with a cost higher than another
known solution. Submitted solvers must also pass a regression suite containing prior
crashes and incorrect results.

The
[MaxSAT Evaluation 2026 rules page](https://github.com/maxsat-evaluations/maxsat-evaluations.github.io/blob/65ff14b9719c68d861c196d1d98c46ca98d24603/2026/contents/contents_rules.html)
continues to describe solved-instance count as the primary exact-solver ranking and adds
optional lower-bound reporting. That page warns that its rules were subject to organizer
approval, so it is supporting trend evidence rather than the primary fixed source for
this item.

MaxSAT therefore provides a closer optimization precedent than SAT: verified optimal
coverage is primary, while runtime and PAR-2 can be reported without replacing the
solved-result distinction.

## XCSP3 Competition: checked best answers and time tie-breaking

The official
[XCSP3 Competition 2025 call](https://www.cril.univ-artois.fr/~lecoutre/compets/callXCSP25.pdf)
covers both constraint satisfaction and constrained optimization. For a constrained
optimization problem, `OPTIMUM FOUND` means that the solver claims no better solution
exists. The evaluation checks returned assignments and objective values.

Solvers are ranked by the number of times they give the best answer obtained in the
competition. Ties are broken by cumulative CPU or wall-clock time to those answers. A
claim of `OPTIMUM FOUND` is wrong if a better feasible assignment exists. A crash or no
solution is treated as `UNKNOWN`; wrong answers have stronger consequences, including
discarding results for the corresponding instance series.

The competition supplies a solution-and-cost checker but does not require Choco or
other complete CP solvers to emit a general, independently replayable proof of
optimality. This supports an oracle-backed solution-checking route for the current Choco
task.

## MIPLIB: time to optimality, primal checking, and consistency

Gleixner et al.,
[MIPLIB 2017: Data-Driven Compilation of the 6th Mixed-Integer Programming Library](https://doi.org/10.1007/s12532-020-00194-3),
report time to optimality, solved count, and branch-and-bound nodes when assessing MIP
solver performance. Shifted geometric mean runtime is used for aggregate comparisons.
In one performance matrix, a solver that does not solve an instance is assigned the
four-hour time limit rather than a proof-derived value.

MIPLIB independently checks every returned primal solution against the original model.
The checker uses arbitrary-precision arithmetic with declared feasibility, integrality,
and objective tolerances. The paper explicitly states that this solution checker cannot
verify optimality in general.

Reported optimal values and primal and dual bounds are also compared across solvers to
identify inconsistencies. Instances with ambiguous numerical behavior are unsuitable
for the benchmark set; 328 submitted instances, about five percent, were excluded for
such inconsistencies unless a solver bug clearly explained them.

This is direct evidence that conventional floating-point MIP benchmarking relies on
independent primal checking, trusted or cross-checked objective information, and
exclusion of numerically ambiguous cases. It does not treat a solver's status string as
an independent proof.

The reviewed local manuscript is stored at
[gleixner-et-al-2021-model-library.pdf](../../../papers/resource-efficiency/sources/gleixner-et-al-2021-model-library.pdf).

## General MILP certificates

Cheung, Gleixner, and Steffy,
[Verifying Integer Programming Results](https://doi.org/10.1007/978-3-319-59250-3_13),
introduce the VIPR certificate format for LP-based branch-and-cut results. The
[VIPR project](https://github.com/scipopt/vipr) checks certificates using exact rational
arithmetic and documents an exact-rational SCIP branch capable of producing them.

Eifler and Gleixner,
[A Computational Status Update for Exact Rational Mixed Integer Programming](https://doi.org/10.1007/978-3-030-73879-2_12),
evaluate exact SCIP and VIPR-style certificate production. Certificate-enabled solving
increased solving time by approximately 101.2% and 51.4% on the two reported test sets.
Checker time was substantially lower than solving time, but certificate production was
not negligible and depended on supported presolve and dual-bounding methods. The local
manuscript is stored at
[eifler-gleixner-2021-exact-rational.pdf](../../../papers/operational-reliability/sources/eifler-gleixner-2021-exact-rational.pdf).

Eifler and Gleixner's
[Safe and Verified Gomory Mixed-Integer Cuts in a Rational Mixed-Integer Program Framework](https://doi.org/10.1137/23M156046X)
extends the exact framework with cuts that can be verified under the VIPR standard. It
is evidence of progress in an exact-rational solver framework, not evidence that ordinary
floating-point MIP solvers emit a complete certificate by default.

Wood et al.,
[Satisfiability Modulo Theories for Verifying MILP Certificates](https://arxiv.org/abs/2312.10420v4),
formalize VIPR's inference rules in Why3 and implement an SMT-based checker. This
strengthens the checker trust story but still requires a valid VIPR certificate as input.

Szeider's 2026
[VORC project](https://github.com/szeider/vorc) constructs VIPR certificates through a
black-box ILP oracle and exact rationalization. Its current implementation depends on
Gurobi and effectively provides a separate certificate-construction solve. It is relevant
to offline oracle construction but is not a replay of the current HiGHS candidate's own
search.

These sources establish that VIPR is a viable future certificate-native protocol. They
also show why adding VIPR as a mandatory postprocessor can change the algorithm and
performance being measured.

## Current solver compatibility

The public
[HiGHS 1.15.1 option definitions](https://github.com/ERGO-Code/HiGHS/blob/v1.15.1/highs/lp_data/HighsOptions.h)
provide solution, model, and basis outputs but no VIPR or complete external MIP
optimality-certificate option. Internal HiGHS code uses proof-like data for algorithmic
decisions and rays, but these are not a complete independent branch-and-cut certificate.

The public Choco 6.0.1
[solver](https://github.com/chocoteam/choco-solver/blob/v6.0.1/solver/src/main/java/org/chocosolver/solver/Solver.java)
has internal explanation and solution-checking mechanisms. Its
[XCSP integration](https://github.com/chocoteam/choco-solver/blob/v6.0.1/parsers/src/main/java/org/chocosolver/parser/xcsp/XCSP.java)
can invoke the XCSP solution checker. No stable general external optimality-proof output
contract was found for this release.

Requiring VIPR or another general proof format for the current HiGHS or Choco candidate
would therefore require another solver, a new certificate generator, or restrictions on
the candidate solver's algorithms. It would not be a neutral checker addition.

## Approved conclusions for PitBench

The competition and literature evidence support the following conclusions.

### Exact Performance outputs

- Verified solved coverage is the primary exact-optimization result.
- PAR-2 is a time-oriented secondary result rather than the sole primary exact result.
- CPU time is the primary time observation for the current sequential exact protocols;
  wall-clock time remains a retained diagnostic.
- A nonoptimal incumbent is not an exact solved result even if it matches a trusted
  optimum before the solver establishes optimal termination.
- Wrong solutions, wrong objective values, and false optimality claims belong to the
  Qualification hard gate.

The evidence does not itself select the final rule that combines solved coverage and
PAR-2 into a PitBench classification. That remains an explicit M2 decision.

### Current optimality-verification route

- Current HiGHS and Choco use fixed trusted optima plus evaluator-owned independent
  primal verification for the first exact-optimization protocol.
- The exact v1 panel is limited to feasible instances with integer or otherwise
  unambiguous objective semantics and a strong fixed oracle.
- Infeasible instances are not included until an independent infeasibility-certificate
  protocol exists.
- VIPR remains a future certificate-native protocol rather than a mandatory component
  of the current HiGHS and Choco evaluation.

For the generated HiGHS single-machine scheduling task, the oracle can be derived from
the shortest-processing-time ordering for the sum-of-start-times objective. The classic
single-machine ordering result is due to Smith,
[Various Optimizers for Single-Stage Production](https://doi.org/10.1002/nav.3800030106).
A problem-specific verifier can check the schedule and objective using integer
arithmetic without trusting HiGHS.

For the Choco bin-packing task, the evaluator requires a fixed optimum and independent
packing verification. The first panel should prefer instances whose reference packing
matches an independently checkable lower bound. Other instances require separately
approved oracle provenance and verification evidence.

### Checker accounting

- Every checker protocol declares its own time limit; no universal ratio to the solver
  budget is inferred from the SAT Competition setting.
- Checker time is recorded separately and is not added to solver Performance time.
- An explicitly rejected result is a Qualification failure.
- Checker timeout makes a certificate-dependent result unsolved.
- Checker or evaluator infrastructure error remains missing and makes the report
  incomplete.

## Current repository gaps found during the review

The current evaluator does not satisfy these conclusions yet:

- `private/oracles/verifiers/mip_solution` uses a partial regular-expression LP parser,
  can skip unsupported input, and has a fallback that accepts the reported objective
  without complete independent model verification.
- The HiGHS `judge_id` generator configuration has no optimum anchors.
- The task-level `private/oracles/miplib.json` reference does not supply the generated
  HiGHS scheduling `judge_id` optima.
- `horizon_factor` is configured for the scheduling generator but is not consumed by
  the generator implementation.
- HiGHS and Choco do not yet provide a common normalized optimal-termination field to
  the Performance estimator.
- The current workspace lacks the complete Choco private panel, oracle, and verifier.
- The OR-Tools model-construction task requires its separate verified-construction
  protocol and must not be treated as an exact-solve normalized-gap task.

These are implementation and protocol blockers. This evidence item does not authorize
their fixes or change the existing M2 specification by itself.
