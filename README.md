# PitBench: Benchmarking Coding Agents on Real-World Solver Improvement

*A repository-level benchmark for combinatorial optimization solvers.*

PitBench evaluates whether coding agents can make correct, repeatable, and
generalizable performance improvements to mature combinatorial optimization
solver repositories. Directional conclusions require calibrated evidence from
the solver's actual outputs.

**The task in one sentence:** An agent edits a mature solver's source code, and
the patched solver must find better solutions or prove optimality sooner on
unseen optimization instances without violating the original mathematical
problem. PitBench measures the full solve: making a search routine faster does
not help if the resulting search makes worse decisions.

<p align="center">
  <img src="https://img.shields.io/badge/PitBench-Benchmark-8A2BE2?style=for-the-badge&logo=github&logoColor=white" alt="PitBench Benchmark">
  <img src="https://img.shields.io/badge/Domain-Combinatorial%20Optimization%20%26%20OR-0A7A5E?style=for-the-badge&logo=target&logoColor=white" alt="Combinatorial Optimization">
  <img src="https://img.shields.io/badge/Current%20Track-PyVRP-1F6FEB?style=for-the-badge" alt="Current track: PyVRP">
  <img src="https://img.shields.io/badge/Evaluation-Performance--First-EA580C?style=for-the-badge&logo=speedtest&logoColor=white" alt="Performance First">
  <img src="https://img.shields.io/badge/Sandbox-Docker%20Isolated-3776AB?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/License-BSD%203--Clause-gray?style=for-the-badge" alt="License">
</p>

<p align="center">
  <a href="#-quickstart">
    <img src="https://img.shields.io/badge/%F0%9F%9A%80%20Quickstart-0A7A5E?style=flat-square" alt="Quickstart">
  </a>
  <a href="#-current-combinatorial-optimization-tasks">
    <img src="https://img.shields.io/badge/%F0%9F%93%A6%20Task%20Snapshots-1F6FEB?style=flat-square" alt="Task snapshots">
  </a>
  <a href="#%EF%B8%8F-the-three-pillars-of-pitbench">
    <img src="https://img.shields.io/badge/%F0%9F%8F%9B%EF%B8%8F%20Three%20Pillars-7C3AED?style=flat-square" alt="Pillars">
  </a>
  <a href="#-roadmap--future-directions">
    <img src="https://img.shields.io/badge/%F0%9F%A7%AD%20Roadmap-EA580C?style=flat-square" alt="Roadmap">
  </a>
  <a href="#-citation">
    <img src="https://img.shields.io/badge/%F0%9F%93%9D%20Citation-475569?style=flat-square" alt="Citation">
  </a>
</p>

---

## 🎯 Why Combinatorial Optimization (CO)?

**Combinatorial Optimization (CO)** is the algorithmic backbone of modern logistics, supply chains, manufacturing, and scheduling. Developing and accelerating high-performance solvers requires mathematical insight, intricate search heuristics, and low-level systems programming. PitBench has eight solver repositories and forty-eight configured tasks, including thirty-eight validated historical PR starting snapshots. It asks whether coding agents, given a solver repository and an improvement goal but no prescribed code change, can investigate the solver, develop and test an optimization idea, and deliver correct, repeatable gains on unseen instances under fixed computational budgets.

Unlike general repository-level coding benchmarks, solver optimization must preserve problem semantics while evaluating a randomized quality-time process:

* 🧠 **Complex Algorithmic Search Spaces**: PyVRP relies on Hybrid Genetic Search, local search neighborhoods, population management, and adaptive control.
* ⚡ **High-Performance Hybrid Architectures**: Solvers heavily utilize C++/Python hybrid designs (e.g., C++ computational kernels exposed via Pybind11 / Cython) where minor changes can break memory invariants or cause severe algorithmic regressions.
* ⏱️ **Stochastic Quality-Time Tradeoffs**: Combinatorial search is inherently randomized. Speedup is meaningless if solution quality (optimality gap) degrades. Benchmarking must evaluate the quality-time Pareto frontier under fixed computational budgets.

**PitBench provides a problem-aware testbed for agentic performance engineering on real CO solver repositories.**

---

## 🌟 Key Highlights

1. **Repository-Level CO Solver Optimization**
   Agents modify real solver repositories rather than isolated algorithm exercises. The current catalog covers eight solvers and thirty-eight validated historical snapshots across PyVRP, VROOM, HiGHS, Choco, and Gecode before human optimization PRs.

2. **Eight Solver Repositories, Forty-Eight Configured Tasks**
   PyVRP, VROOM, HiGHS, Choco, Gecode, OR-Tools CP-SAT, SCIP, and CBC cover heuristic and exact solving across routing, packing, coloring, scheduling, and arithmetic CP problems. Seventeen historical PyVRP tasks start before PRs #232, #511, #701, #783, #845, #865, #919, #962, #966, #967, #974, #988, #998, #1146, #1155, #1156, and #1190; VROOM starts before #112, #146, #978, and #1222, HiGHS before #577, #618, #651, #685, #690, #693, #1992, #2027, #2522, #2861, #3213, and #3286; Choco before #493, #1152, #1154, and #1178, and Gecode before #224. Historical source patches and their measured reference status are retained per task; open-opportunity tasks do not require a positive human-PR effect. SCIP and CBC are configured MIP source-code tasks; full agent comparisons on them are pending.

3. **CO-Specific Quality-Time Performance-First Protocol**
   Heuristic tasks measure paired normalized optimality-gap reductions against best-known solutions; exact tasks measure independently verified optimal coverage and penalized solve time. Each task declares its own budgets and seeds, with instance-cluster bootstrap confidence intervals for paired comparisons.
   A directional result is confirmed with a fresh, opposite-order Base/Agent
   run before it becomes a public improvement or regression label.

4. **Prepared Docker Environments**
   PyVRP native extensions are compiled in prepared, commit-tagged images.
   Agent execution is network-disabled, while evaluator-owned hidden assets
   are supplied only to a separate judge environment. The current Judge mounts
   its private asset root into the same container that runs candidate solver
   code; read-only mounting prevents modification but does not isolate hidden
   oracles from that code. A claim of leakage-resistant hidden judging therefore
   needs a separate solver-process access boundary.

---

## 🏛️ The Three Pillars of PitBench

```
                       ┌──────────────────────────────────────────────────────────┐
                       │               PitBench Evaluation Pipeline               │
                       └────────────────────────────┬─────────────────────────────┘
                                                    │
        ┌───────────────────────────────────────────┼───────────────────────────────────────────┐
        ▼                                           ▼                                           ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐   ┌───────────────────────────────┐
│   1. Quality-Time Contract    │   │  2. Randomized Repeatability  │   │  3. Held-Out Generalization   │
├───────────────────────────────┤   ├───────────────────────────────┤   ├───────────────────────────────┤
│ • Task-specific fixed budgets│   │ • Multi-seed paired trials    │   │ • Evaluate on unseen instances│
│ • Anchored to optimum / BKS   │   │ • Instance-cluster bootstrap  │   │ • Retention on hidden shifts  │
│ • True Gap Reduction:         │   │ • Strict 95% confidence bounds│   │ • Check retention beyond dev  │
│   Δgap = gap_base - gap_agent │   │ • Paired seed evidence        │   │   on judge_shift              │
└───────────────────────────────┘   └───────────────────────────────┘   └───────────────────────────────┘
```

### 1. Quality-Time Performance First
- **Quality and time stay coupled**: Faster execution is not treated as an improvement when independently verified solution quality regresses.
- **Independent Solution Verifiers**: Evaluator-owned verifiers check feasibility, integrality, and objective values against original instances. Exact tasks compare solver optimality claims with independently established optimum targets; the judge does not reconstruct a general branch-and-bound proof certificate.
- **Search-progress diagnostics**: Public paired traces expose incumbent and bound progress where the solver emits it. A normalized gap area is reported only over the time with both bounds observed, together with its covered fraction of the budget. It is not part of the frozen score: intermediate bounds are solver-reported, and a full-budget penalty for missing incumbents or bounds has not been fixed across backends.
- **Zero optimum targets**: The signed normalized gap divides by `abs(target)` for nonzero targets and by `1` when the independently established optimum is zero. Existing nonzero-target scores retain their prior definition.

### 2. Randomized Repeatability & Statistical Rigor
- Metaheuristics and randomized search algorithms exhibit high variance across seeds.
- Stochastic evaluations pair Base and Agent across the task's declared solver seeds and compute **instance-cluster bootstrap 95% confidence intervals (CI95)**. An improvement claim is accepted only when the lower bound is strictly positive ($CI_{\text{lower}} > 0$).

### 3. Generalization across Problem Topologies (Held-Out Retention)
- Where configured, evaluations test both calibration instances (`judge_id`) and unseen structural shift instances (`judge_shift`, e.g., different routing distributions or a held-out MIP family).
- The report shows whether the measured improvement is retained on the declared
  hidden-shift instance set; it does not claim universal generalization.

---

## PR types and objective-specific measurement

PR type is recorded along three axes: change kind, solver phase, and intended
benefit. One PR may have several labels. The versioned
[catalog](pitbench/schema/objective_catalog.json) currently maps 13 benefit
categories to 26 measurement items; this is extensible and does not impose
equal quotas or create a combined score.

| Intended benefit | Measurement and required scenario | Current support |
| --- | --- | --- |
| Solve efficiency | Verified solve coverage, penalized runtime; paired solves | Existing exact protocol |
| Fixed-budget quality | Independently checked normalized gap | Existing heuristic protocol |
| Earlier good solutions | First verified incumbent, time to target, primal integral | Requires verified incumbent trajectories and a qualification protocol |
| Bound and proof progress | Solve coverage/time, certified bound trajectory, search-work diagnostics | Final exact outcomes supported; trajectory qualification incomplete |
| Memory footprint | Scope-matched paired peak RSS and auxiliary CPU ratios | Formal paired-resource qualification with predeclared tradeoff limits |
| Memory lifecycle | Retained-memory series and growth across repeated solves in one process | Requires a long-lived-process scenario |
| Parallel/accelerator efficiency | Thread scaling, efficiency, CPU and memory under a declared hardware grid | Requires multiple resource conditions |
| Incremental/reoptimization efficiency | Update overhead and total time across equivalent model-update sequences | Requires a sequence scenario |
| Numerical robustness | Independent original-model violations and numerical failures on declared stress cases | Checkers and raw evidence exist; dedicated stress protocol incomplete |
| Operational reliability | Boundary-suite pass, crash, OOM, output-error and missing-run counts | Existing descriptive reliability report |
| Repeatability | Existing seed/representation dispersion; separately, same-seed replay consistency | Seed and representation reports exist; replay protocol pending |
| Scale/structural generalization | Improvement retention on declared scale and structural conditions | Existing controlled-transfer protocol where configured |
| Solver-owned setup | Read/initialization time and stage-qualified memory | Requires stage-specific collection |

The research basis includes [SCIP testing](https://www.scipopt.org/scip/doc/html/TEST.php),
[primal-integral research](https://www.zib.de/userpage//hendel/publication/rounding-and-propagation-heuristics-for-mip/),
[MiniZinc 2026 scoring](https://www.minizinc.org/challenge/2026/rules/),
[SCIP reoptimization](https://www.scipopt.org/doc/html/REOPT.php), and actual PRs.
For example, [HiGHS #2981](https://github.com/ERGO-Code/HiGHS/pull/2981)
concerns memory retained across repeated solves, while
[HiGHS #3192](https://github.com/ERGO-Code/HiGHS/pull/3192) reports different
effects under one and eight threads. These motivate different experiments;
author-reported effects are not independently confirmed PitBench results.
The catalog retains all source links and implementation limitations.

Tasks may declare an optional `optimization_profile` with `objectives`,
`change_kinds`, `solver_phases`, classification provenance, and a
`primary_metric_id`. Title hints never choose a qualification primary metric.
The evaluator and `pitbench report` expose `objective_measurements`, reusing
native measurements and showing partial, unavailable, raw-only, and
inapplicable evidence. New scenarios are not fabricated from a single solve;
unmeasured objectives are not zero gains or rejections. The optional formal
qualification contract is defined below. Frozen tasks without a profile retain
their existing scoring; no task budgets or seeds are changed by annotation.

For example, an annotated task can request resource evidence with:

```yaml
optimization_profile:
  objectives: [peak_memory, memory_lifecycle]
  change_kinds: [resource_management]
  classification_basis: declared
```

This requests measurements only. Admission additionally requires the scoring
plan, registration, complete runs, independent correctness and confirmation.

### Formal objective qualification protocol 1.0

The versioned [qualification registry](pitbench/schema/qualification_protocols.json)
covers all 13 objective categories and 26 measurement items. Every item has an
explicit implementation status. Exact coverage/runtime and fixed-budget gap
reuse their existing definitions; paired peak RSS and CPU ratios can now qualify
resource optimizations. Dedicated lifecycle, trajectory, scaling, sequence,
numerical-stress and replay scenarios remain pending where their collection or
decision protocol is missing. Existing seed (99% paired seed bootstrap) and
controlled-transfer statistics remain unchanged; their descriptive availability
alone does not define a new primary admission decision.

One primary metric is frozen before qualification outcomes. Every declared
qualification panel must improve; the program does not select the best metric
or panel after seeing results. Exact qualification keeps the coverage-first
decision and PAR2 treatment. Gap qualification keeps the native paired 95%
instance-bootstrap decision. Resource qualification keeps the existing ratio of
paired seed means per instance and geometric mean across instances; the native
95% instance bootstrap is applied to log ratios and exponentiated. Its upper
bound on Agent/Base must be below 1, with at least two complete instances.
There is no universal 5% minimum improvement.

**Allowed degradation is declared per task.** The primary metric and explicit
guardrails jointly cover quality/exact solved coverage, compute cost/exact PAR2,
and peak RSS on every qualification panel. Relative limits use Agent/Base minus
1; absolute limits use signed deterioration in the metric's own units (coverage
uses Base minus Agent). The adverse confidence bound must stay within the
declared limit in both runs. `guardrail_failed` means this requirement was not
established, including uncertain bounds; it is not an automatic permanent PR
rejection. No declaration is replaced with an implicit zero or percentage.

The task's `optimization_profile.scoring` contains:

| Field | Required decision |
| --- | --- |
| `protocol_version` | `"1.0"` |
| `protocols_sha256` | Hash returned by `qualification_protocols_sha256()` |
| `qualification_instance_sets` | Frozen hidden panels used for selection |
| `reporting_instance_sets` | Separate hidden panels for final effect estimates |
| `guardrails` | Explicit `{metric_id, comparison, max_degradation, instance_sets}` entries; `null` remains `needs_protocol` |

`exact_solved_coverage`, `exact_penalized_runtime_sec`, and
`fixed_budget_normalized_gap` use absolute guards; resource ratios use relative
guards. Final reporting panels cannot act as qualification or control panels.
Correctness failures are never an allowed tradeoff. All declared pairs, failures
and missing observations are retained. Missing protocols yield `needs_protocol`,
missing measurements yield `incomplete`, and uncertain primary directions yield
`inconclusive`. A fresh full Judge repeat must independently improve and satisfy
all guards, with identical identities/resources/grid and reversed Base/Agent
order. Cached Base observations cannot provide this confirmation.

For each indexed PR with a pinned task configuration, use the existing curator:

```bash
# First freeze the profile, primary metric, panels and per-task guardrails in YAML.
python scripts/curate_solver_prs.py objective-register \
  --catalog configs/pr_curation/repositories.json --registry REGISTRY.jsonl \
  --repository OWNER/REPO --pr-number N --output report/REGISTRATION.json

# Run the normal full Judge, then recompute from its original artifacts.
python scripts/curate_solver_prs.py objective-qualify \
  --catalog configs/pr_curation/repositories.json --registry REGISTRY.jsonl \
  --repository OWNER/REPO --pr-number N --run-directory report/JUDGE_RUN \
  --output report/QUALIFICATION.json
```

Registration binds the task, source patch and protocol hashes before measured
runs. Judge checkpoints retain measurement start times. These are audit records
from the evaluator, not an external timestamp attestation. Revisions retain the
previous registration and require fresh runs. Qualification re-reads hash-bound
observations and grids; an `improved` JSON flag cannot promote a task. Only a
confirmed result within the declared limits advances an `expert_reference` to
`performance_confirmed`. The existing source/build/workload/independent-checker
gates remain mandatory, and `integrate` requires a real task run before
`task_ready`. `open_opportunity` retains its Base-opportunity route without
requiring the historical human implementation to improve.

`scripts/curate_solver_prs.py type-audit --catalog ... --registry ...
--output-dir ...` produces a JSONL hint inventory
and hash-bound summary without modifying the source registry. Its title
matches are provisional, overlap, and do not constitute semantic review or
task admission. Correctness-only, feature, wrapper, parameter and maintenance
changes remain separately classified under the existing research scope.

## 📦 Current Combinatorial Optimization Tasks

| Task ID | Solver & Version | Problem Family | Architecture | Optimization Scope |
| :--- | :--- | :--- | :--- | :--- |
| `pyvrp_v0_14_0` | **PyVRP** v0.14.0 | CVRP | C++ / Python (Pybind11) | Search and local-search efficiency |
| `pyvrp_default_operators_20240305` | **PyVRP** 0.8.0a0 historical snapshot | CVRP; held-out clustered routing | C++ / Python (Pybind11) | Default local-search operator portfolio |
| `pyvrp_route_distance_storage_20241117` | **PyVRP** 0.11.0a0 historical snapshot | CVRP | C++ / Python (Pybind11) | Local-search route-segment distance storage and query efficiency |
| `pyvrp_search_20250717` | **PyVRP** 0.12.0a0 historical snapshot | CVRP | C++ / Python (Pybind11) | Open solver improvement with a source-only human PR reference |
| `pyvrp_search_20251020` | **PyVRP** 0.12.0 historical snapshot | CVRP | C++ / Python (Pybind11) | Open solver improvement with a source-only human PR reference |
| `pyvrp_search_20251219` | **PyVRP** 0.13.0a0 historical snapshot | CVRP | C++ / Python (Pybind11) | Open optimization opportunity; human PR reference inconclusive |
| `pyvrp_search_20260112` | **PyVRP** 0.13.0a0 historical snapshot | CVRP | C++ / Python (Pybind11) | Open ILS policy optimization opportunity; human PR hidden effect not claimed |
| `pyvrp_search_20260728` | **PyVRP** 0.14.0a0 historical snapshot | CVRP | C++ / Python (Pybind11) | Open solver improvement with a source-only human PR reference |
| `pyvrp_search_20260806` | **PyVRP** 0.14.0a0 historical snapshot | CVRP | C++ / Python (Pybind11) | Open solver improvement with a source-only human PR reference |
| `vroom_v1_15_0` | **VROOM** v1.15.0 | CVRP | C++ | Route quality under fixed budgets |
| `vroom_tspfix_20230918` | **VROOM** 1.14.0-dev historical snapshot | CVRP; held-out clustered routing | C++ | Route-level TSP local search with an open improvement opportunity |
| `vroom_cvrp_regret_insertion_20180813` | **VROOM** 1.3.0-dev historical snapshot | CVRP; held-out clustered routing | C++ | Regret insertion and capacity arithmetic efficiency under fixed budgets |
| `vroom_heuristic_20250313` | **VROOM** 1.15.0-dev historical snapshot | CVRP | C++ | Open insertion-pruning opportunity with independent bound checks |
| `highs_v1_15_1` | **HiGHS** v1.15.1 | Scheduling MIP | C++ | Verified optimal coverage and solve time |
| `highs_multidimensional_knapsack_v1_15_1` | **HiGHS** v1.15.1 | Multidimensional 0–1 knapsack | C++ | Verified optimal coverage and solve time |
| `highs_transformed_integer_repair_20241031` | **HiGHS** 1.8.0 historical snapshot | Multidimensional knapsack; held-out scheduling MIP | C++ | Open transformed integer solution repair opportunity; human PR effect not claimed |
| `scip_v9_2_4` | **SCIP** v9.2.4 | Multidimensional 0–1 knapsack; held-out scheduling | C / C++ | MIP source-code optimization |
| `cbc_v2_10_12` | **CBC** v2.10.12 | Multidimensional 0–1 knapsack; held-out scheduling | C++ | MIP source-code optimization |
| `choco_v6_0_1` | **Choco** v6.0.1 | Bin packing | Java | Verified optimal coverage and solve time |
| `choco_graph_coloring_v6_0_1` | **Choco** v6.0.1 | Graph coloring | Java | Verified optimal coverage and solve time |
| `choco_first_fail_selector_20170117` | **Choco** 4.0.2-SNAPSHOT historical snapshot | Balanced subset-sum CP | Java | Open FirstFail domain-scan efficiency opportunity; human PR effect not claimed |
| `choco_decision_path_20250326` | **Choco** 5.0.0-beta.2 historical snapshot | Bin packing; held-out graph coloring | Java | Search decision construction with an open exact-solve opportunity |
| `choco_arithmetic_propagation_20260115` | **Choco** 5.0.0-beta.2 historical snapshot | Balanced subset sum CP | Java | Arithmetic propagation with independent DP optima |
| `ortools_cp_sat_solve_v9_15` | **OR-Tools CP-SAT** v9.15 | Job-shop scheduling | C++ / Python | Verified optimal coverage and solve time |
| `gecode_binpacking_dff_20260711` | **Gecode** 6.3.0 historical snapshot | Bin packing | C++ | Dual-feasible-function propagation with exact baseline opportunity |

The current PyVRP release task remains at v0.14.0. Four positive-reference
historical tasks use earlier source commits; their expert source patches are hash-pinned in
`configs/references/`. The patches are evaluator-owned references, not
prescribed edits shown to the agent. Each PR completed a 38-instance,
5-hidden-seed, opposite-order human-reference control and a limited
132-observation Judge integration check. All four formally improved 10-second
solution quality; PR #919 has a small gain below the separate strong-reference
minimum effect size. PR #865 uses its upstream test adjustment, hash-pinned and
applied only by the evaluator to both validation builds; performance builds
still use the source-only patch. All four share one solver and hidden panel.
Two additional historical tasks use the open-opportunity track. For
`pyvrp_search_20251219`, the human PR was inconclusive on the hidden panel,
but the pinned base had a 1.403% mean independently verified normalized gap. It retains the full
negative reference result and is not counted as a positive expert patch. For
`pyvrp_search_20260112`, the human PR hidden effect has not been claimed; the
pinned base completed all 38 hidden ID instances at one fixed seed with a
1.541% mean independently verified normalized gap.
One independent Codex attempt on `pyvrp_search_20260806` has
completed its 30-hidden-seed grid: all 8,160 Base/Agent observations were valid,
and its 10-second mean normalized gap reduction was 0.0077 percentage points
(95% CI: 0.0038–0.0122 percentage points). The opposite execution order also
classified the patch as improved. On the same five hidden seeds used for PR
#1156, the human patch had a larger gain in this one exploratory comparison.
Repeated Agent starts and cross-solver expert controls are still needed to
characterize a general gap.

The published [FormulaCode study](https://proceedings.mlr.press/v306/sehgal26a.html)
reports 957 expert-patch tasks. PitBench's scale objective is **more than 1,000
fully validated historical PR tasks** under its own solver-specific admission
rules. The current count is **38/1,001 minimum**, leaving 963 tasks:
four positive expert references and thirty-four open opportunities.
Only the expert-reference subtotal is directly comparable with a collection of
positive expert patches; the total measures distinct solver optimization
challenges under PitBench's stated task definition.
Seventeen current PR tasks use PyVRP, four use VROOM, twelve use HiGHS, four use Choco, and one uses Gecode. The structured reviews
cover search-state construction, ILS policy, insertion-bound pruning, MIP cut aging,
branching, and presolve; the
four earlier tasks retain their original review records and are labeled as
legacy in mechanism-stratified counts.
HiGHS #685 adds a distinct dual steepest-edge weight-maintenance task. Its 60 hidden
base runs were hash-bound to the frozen MPS inputs and independently checked: 25
proved optimal and 35 reached the 20-second time limit. The empty-patch Judge run
checked the 40-observation evaluation path; no coding agent was run and no Agent
performance result informed task admission.
HiGHS #3286 adds a binary clique-row presolve task. Its source-only patch matches
the pinned PR head; base and PR builds, public activation probes, and independent
checks of original MPS solutions passed. The frozen 60-instance base qualification
retained all outcomes: 38 proved optimal and 22 reached the normal time limit. A
16-cell no-op Judge integration run had eight independently valid solutions and
eight normal time limits, with no crashes. No coding agent or candidate performance
result informed admission.
VROOM #146 adds a CVRP regret-insertion and capacity-arithmetic task. The reviewed
path was activated on all ten frozen public cases; 20 debug-build routes passed the
independent original-instance checker. A one-seed, base-only 30-second qualification
completed all 38 CVRPLIB X instances with a 4.58% mean normalized gap. A separate
10-second pilot is retained with 37 completed runs and one normal timeout on
X-n957-k87; the benchmark keeps that large instance and uses the 30-second primary
budget. Its 12-cell no-op Judge integration had eight valid solutions and four
normal timeouts, with no crashes. No coding agent or PR-patch performance result
informed admission.
The local discovery registry currently retains 26,817 distinct merged PRs from
69 PR source repositories, with pinned base/head commits. **This is not 69
different solvers:** the sources include solver components, modeling and
algorithm frameworks, and related projects such as the OptaPlanner/Timefold
lineage. Only eight solver repositories currently have configured PitBench tasks;
nine further solver-source repositories have passed domain review but do not yet
have configured tasks. Timefold and OptaPlanner still share one project lineage.
Seventeen historical PR tasks come from PyVRP, four from VROOM, twelve from HiGHS, four from Choco, and one from Gecode. The tracked
[snapshot](configs/pr_curation/snapshot.json) records the repository-role
review and PR stage counts. The number of independent solvers represented by the
discovery pool has not yet been established. **Thirty-eight PR-sourced tasks** have passed
the source, build, workload, and independent correctness gates plus their
track-specific hidden evidence. Two further PRs have paired builds, and two
have verified source patches; these four records are not validated tasks. Two
retained public screens were demoted to `needs_evidence` after their changed code
paths proved inactive on the CVRP panel. The local audit has 12,960 complete
PR-detail records and a separate manual-diff queue for 42 incomplete large PRs;
these metadata counts are not benchmark tasks.
VROOM #112 adds a 2018 CVRP clustering candidate-selection task. Its changed path
was activated on all ten public probe cases, all 48 hidden base runs had
independently verified solutions, and the base had a 15.23% mean normalized gap.
The historical PR's hidden effect was inconclusive under a two-round paired
comparison. No coding agent was run; admission used the independently verified
baseline opportunity and an empty-patch Judge integration check.
PyVRP #845 adds a duration-proposal shortcut task. On ten public distance-only
CVRP cases, temporary counters observed 116.9 million base duration proposals
and 275.2 million base segment merges; the PR bypassed duration evaluation on
145.9 million proposals. A separate time-window control kept duration evaluation
enabled. Counters establish activation only; instrumented timings were excluded.
The 48-case base-only hidden panel had a 1.34% mean normalized gap, and all
public solutions passed independent checks. No coding agent was run.
PyVRP #998 records the seeded randomized-initial-solution path on all ten public
CVRP cases; its 48-case base-only panel retained a 0.80% mean normalized gap.
Two shifted-instance anchors were refreshed in a task-specific oracle version
after the base produced independently checked better incumbents. The prior
oracle remains unchanged, and no coding agent was run.
PyVRP #962 records the nonpromising-node guard on all ten public CVRP cases.
Temporary counters show fewer optional-client, group, and reload-depot helper
calls in the PR state; they establish activation, not a timing result. Its
base-only 48-case panel had a 0.66% mean normalized gap. The task explicitly
does not claim activation of optional-client or reload-depot feature branches
in this all-required CVRP panel, and no coding agent was run.
HiGHS #2027 adds four bound-derived primal candidates before root processing.
All five public MKnapsack models emitted PR trivial-heuristic source messages;
the base emitted none. The 30-case exact hidden base solved 25 optima and
normally timed out on five. An independent fractional-bound regression also
confirmed the PR's infeasibility status; no coding agent was run.
HiGHS #577 adds MIP row aggregation, Mod-K separation, and clique-table limits
at the 1.0.0 source snapshot. Mod-K and clique-merge paths activated on public
knapsack and scheduling probes; the path separator did not, and is not claimed
as covered. Across ten public MIPs, 12 of 20 paired source runs proved optimal
and eight hit the time limit; all 20 returned incumbents passed independent
original-model feasibility checks, and all reported dual bounds remained valid.
The paired MIP/GF(k) suites passed 24,977 assertions per source state. On 60
base-only hidden instances, 26 proved optimal and 34 normally timed out. A
40-run empty-patch Judge integration passed. No coding agent was run, and no
agent outcome or human PR speedup was used for admission.
Discovery entries and public-only screens
are not counted as benchmark tasks. The generic curation commands are in
`scripts/curate_solver_prs.py`; other discovered repositories need domain
review, and approved sources still need adapters and workload validation.

### Historical PR selection method

The unit is one merged PR's pinned **pre-PR solver snapshot** plus an open
improvement goal, not a repository or a raw PR-search hit. New reviews retain
the PR and its decision so accepted and rejected cases remain auditable; the
four earlier historical tasks retain their original gate evidence. Repository
review first establishes that the project exposes a combinatorial optimization
solver and that the candidate modifies its engine, not only a modeling API,
dependency, benchmark, or interface. A general SAT/SMT or constraint project
does not enter an optimization task merely because it has merged PRs: the task
must identify an objective or exact-solve target and an independently checkable
problem family. The first queue interleaves repositories for metadata retrieval.
A second queue uses changed source paths, solver-phase path hints, and title
terms to order human review across repositories and mechanisms. Reverts and
branch merges receive a lineage-review flag, while a solver operation that
merges routes remains in the mechanism lane. A fixed 10%
share of each full review batch samples low-signal PRs by stable hash and
round-robin repository coverage to expose missed candidates. Missing or
rate-limited metadata is reported separately;
oversized PRs with incomplete file listings enter a separate `manual_full_diff`
queue. `triage` accepts repeated `--details` inputs so each fetch batch remains
an intact audit file. PRs marked `needs_evidence` leave the fresh-review queue
until their named missing evidence is supplied; they remain in the registry.
No metadata signal rejects a PR. New structured decisions
and their file hashes
are recorded before hidden performance results are run. Hashes backfilled for
earlier reviews detect later edits but do not establish prospective registration.
The `screening-audit` command reports reviewed and unreviewed PRs by metadata
signal band, repository, and solver-phase path hint, with hashes of each input
batch. This makes the screen's blind spots measurable. In the first two detail
batches, a low-signal PR, SHOT #9, advanced to source verification even though
its title lacked the usual performance terms. Only two low-signal PRs had a
recorded structured decision at that point, so this is a discovered miss of the
priority heuristic, not an estimate of its recall.
Later detail batches and decisions must be passed together to
`screening-audit`; its output is the current count of low-signal reviews and
advances. The sampled reviews include solver correctness fixes and build/API
changes that metadata hints alone could mis-rank. Most of the low-signal pool
remains unreviewed, so recall is still unknown.

The default bulk path now accepts schema 3 `pr_metadata` prechecks. It extracts
PR title/body, the complete file list, links, fenced materials and pinned-diff
context without a model. Its `advance` schedules measurement; mechanism labels
may remain unknown. These records pass through `screen` and `source-verify`
without a mandatory causal-chain essay. Actual source reproduction, builds,
workload activation, independent correctness and the frozen scoring contract
still decide qualification. For metadata candidates, workload selection is
bound to the task configuration hash. Exact candidates with unknown mechanisms
conservatively retain bound-validity, status-consistency and optimality evidence
obligations; missing checkers remain pending. Schema 3 cannot enter `task_ready`
without the explicit objective scoring contract.

The optional schema 2 mechanism-review `advance` needs source-level evidence that the PR changes a solver
engine, a concrete algorithmic-search or computational-efficiency mechanism,
and a hypothesis about how that change affects end-to-end solving. The review
requires the repository to be task-backed or approved after domain review. It
names the actual solver entry points and phase, distinguishes per-event cost
from changes in search tree, primal or dual progress, proof time, and memory,
and defines a structural trigger plus an activation probe. It also records
specific mathematical failure modes, an independent checker plan, the
exact/anytime metric family, a predeclared instance-selection rule, hidden
generalization, resource risk, and overlap with neighboring PRs. At the
workload gate, hash-pinned instance and activation records must show that the
reviewed code path actually ran on the selected solver workloads.
When a mechanism requires a non-default solver strategy, the task fixes that
strategy in `evaluation.fixed_solver_parameters`. The same public setting is
passed to Agent development runs, paired source checks, and hidden Judge runs;
the correctness and integration gates verify the retained per-run parameter
files. Choco #1154 uses this rule for weighted-degree search, with an
input-order control that does not reach the changed selector.
Selection separates **mechanism plausibility** from **task suitability**.
No coding agent is run to decide whether a historical PR task is admitted, and
no Agent score or improvement is an admission condition. The expert-reference
track uses independently measured human-PR effects; the open-opportunity track
uses a verified baseline improvement gap without requiring a positive human PR
effect. A limited empty-patch Judge run checks that the task harness works; it
does not measure an Agent or affect inclusion. Agent outcomes are reported only
after the task set is fixed.
A reached function is only an exposure observation: the reviewer identifies
which search decisions, per-event costs, primal or dual trajectory, proof time,
or memory could change, then compares those observations on matched base and PR
builds. The selected panel includes structurally triggered cases and, where a
non-triggering case exists, a control. Mixed PRs remain one task; separate
correctness fixes or unrelated code paths are recorded so a total speedup is
not attributed to the wrong mechanism. Choco #1152 is counted once for its
cached decision path: debugger traces reach `IntStrategy.computeDecision` in
both pinned states, while the separate conflict-strategy changes are reported
as inactive. Choco #1178 is counted once for arithmetic propagation: both
changed propagators are reached on balanced subset-sum cases and neither is
reached on the bin-packing control. Both tasks passed independent original
problem checks and hidden exact-solve baselines; no human PR gain is claimed.
The review schema requires a failure mode appropriate to the named solver
phase: presolve and cuts address bound or status validity, while local search
addresses primal validity or search completeness.
The reviewer identifies families where the mechanism may not activate;
end-to-end results are reported by family so a gain on one structure cannot
hide a severe regression elsewhere.
For PRs that may invalidate bounds, infeasibility, or optimality claims, the
correctness gate additionally requires hash-pinned independent bound, status,
or certificate checks. Checking a returned primal solution alone does not
discharge those obligations.

Before a new task is used for a correctness or capability claim, its problem
family and evaluation mode need a recorded verification contract. The contract
names every solver result state used by the score and the evidence that
supports it:

| Solver result | Required independent evidence |
| --- | --- |
| Feasible solution | Original constraints, integrality, and recomputed objective. |
| Bound | Validity for the original problem; a solver-reported bound alone is insufficient. |
| Optimal | A verified feasible solution plus an independent optimum target or checkable proof. |
| Infeasible | A checkable certificate or independently established infeasibility. |
| Time limit | Validity of any incumbent and bound, with the termination state retained. |
| Crash, invalid output, or build failure | A separate failure record, never a normal solver time limit. |

The current exact Judge compares optimality claims with trusted targets rather
than constructing general proof certificates. The existing selection review
records failure modes and semantic checks, but it does not yet store this full
status-by-status contract as a machine-checked field. Tasks lacking evidence
for a scored status need that evidence or a score restricted to independently
verifiable outcomes. Exact reports now count solver failures separately from
normal unsolved outcomes while retaining the existing PAR2 penalty.
Previously saved reports keep their original categories until their raw ledgers
are reprocessed.

`needs_evidence` and `reject` remain in the
registry and do not count as tasks. A bug fix, test change, dependency update,
modeling-only change, or faster local routine is not accepted on its title or
local timing alone.

An advanced PR still needs a source-only patch that reproduces the PR head,
reproducible builds, exposed solver workloads, independent original-problem
correctness checks, and a fixed-budget paired hidden performance result.
Exact-solving tasks inspect solve/proof outcomes and time; anytime tasks inspect
verified solution and bound trajectories at declared budgets. Instance-family
holdouts, missing runs, regressions, and PR overlap are reported rather than
silently filtered. The `expert_reference` track requires a confirmed positive
human PR on the hidden panel. The `open_opportunity` track instead requires a
complete independently verified hidden baseline with nonzero room under its
declared score; the human PR result is retained and cannot be called a positive
expert reference. A unique historical PR counts once across these tracks. Only a PR
passing its full track gate becomes one `task_ready` task.

For `open_opportunity`, an unsolved exact run or a positive baseline gap is an
observable baseline state, not evidence that an Agent can improve it. The
selection record must name the frozen score and budget, the independently
valid baseline outcomes, and how the Judge can distinguish a change from
baseline variation. A stability or detectability threshold must be fixed
before inspecting qualification outcomes; none is chosen here. The current
`baseline-evidence` gate checks complete, valid runs and nonzero room but does
not yet enforce a separate stability criterion.

For `expert_reference`, the qualification panel used to select a positive
human PR and any final panel used to estimate or compare its effect have
different roles. The final effect estimate must use a panel and analysis rule
not repeatedly searched for admission; otherwise it is reported as
selection-conditioned. Budgets, seeds, aggregation, missing runs, regressions,
and retry rules are declared before the qualification run. The current
`performance_confirmed` gate accepts a recorded positive paired
classification; the report pipeline supports reverse-order confirmation, but
the gate does not encode a distinct final reporting panel.

The Agent receives the pinned pre-PR source, the public task goal, and
development instances. Its source snapshot is shallow and has no named refs
or remote; Agent execution has no internet access. Reference patches,
selection reviews, and hidden data are evaluator-owned. In the real Judge,
candidate patches are checked before build: changed files must stay within
configured editable paths, generated build paths are excluded, and a nonempty
patch must change a recognized solver source file. Candidate patch and Judge
image identities are retained. This controls submitted files; the remaining
private-oracle process boundary above must be resolved before claiming full
protection against runtime inspection of hidden assets.

Structured tasks from the same repository and overlap group cannot both enter
the ready count; a reviewer must establish distinct mechanisms and workloads
before assigning separate groups. PRs sharing the same pre-PR commit cannot
count twice with an identical Agent instruction, editable scope, workload,
budget, and seed protocol; task IDs and image tags do not create distinct
inputs. For every same-snapshot pair, at least one hash-bound selection review
must link the other PR and state the different improvement target; this check
does not depend on PR merge order. The snapshot reports repository, track,
mechanism, solver phase, and overlap-group counts so scale does not hide
concentration in one solver or PR series.

Capability reports should additionally show problem family, source-version
period, task difficulty, and source-change complexity, grouping near-identical
snapshots and workloads rather than treating them as independent samples. No
equal mechanism quota is assumed. These additional distributions are not yet
part of the generated snapshot.

For scalable open-opportunity qualification, the evaluator can run the full
hidden ID instance panel at one seed selected before outcomes, using only the
pinned base image. This establishes baseline validity and remaining gap without
claiming a human PR effect; final Agent judging still uses the task's full
declared seed and budget grid. `scripts/calibrate_release_controls.py` and
`scripts/analyze_release_controls.py --baseline-only` retain the raw run ledger,
failures, independent checks, and baseline analysis.
On #966, this mode completed 38/38 independent base checks with a 1.200% mean
normalized gap; its more expensive two-order human comparison remains a
separate, inconclusive reference record.
This solver-behavior and mathematical-validity screen is PitBench's selection
criterion; FormulaCode supplies a scale comparison, not the admission rule.

`measurement_ready` remains a proposed queue state, not an implemented stage or
an admission count. Its pre-queue build should be a lightweight smoke check;
the queued HPC job should retain a full fixed-identity build and public
correctness ledger. A representative, preregistered cross-solver pilot should
record human review time, stage failures, build cost, and ready yield before
extrapolating to 1,001 tasks. The pilot size and any admission thresholds still
need a protocol decision.

Build a review order, then apply a structured JSON decision before source
verification:

```bash
uv run python -m scripts.curate_solver_prs queue \
  --catalog configs/pr_curation/repositories.json \
  --registry report/pr-scale-20261018/pr_registry.jsonl \
  --limit 100 --output report/pr-scale-20261018/review_queue.json
uv run python -m scripts.curate_solver_prs enrich \
  --catalog configs/pr_curation/repositories.json \
  --registry report/pr-scale-20261018/pr_registry.jsonl \
  --queue report/pr-scale-20261018/review_queue.json \
  --limit 100 --output report/pr-scale-20261018/pr_detail_review.jsonl
uv run python -m scripts.curate_solver_prs triage \
  --catalog configs/pr_curation/repositories.json \
  --registry report/pr-scale-20261018/pr_registry.jsonl \
  --details report/pr-scale-20261018/pr_detail_review.jsonl \
  --limit 100 --audit-fraction 0.1 \
  --output report/pr-scale-20261018/mechanism_review_queue.json
uv run python -m scripts.curate_solver_prs screening-audit \
  --catalog configs/pr_curation/repositories.json \
  --registry report/pr-scale-20261018/pr_registry.jsonl \
  --details report/pr-scale-20261018/pr_detail_review.jsonl \
  --output report/pr-scale-20261018/screening_audit.json
uv run python -m scripts.curate_solver_prs prepare-review-batch \
  --catalog configs/pr_curation/repositories.json \
  --registry report/pr-scale-20261018/pr_registry.jsonl \
  --details report/pr-scale-20261018/pr_detail_review.jsonl \
  --queue report/pr-scale-20261018/mechanism_review_queue.json \
  --limit 100 --output-dir report/pr-scale-20261018/review_batch_01
uv run python -m scripts.curate_solver_prs screen \
  --catalog configs/pr_curation/repositories.json \
  --registry report/pr-scale-20261018/pr_registry.jsonl \
  --review-file REVIEW.json
```

`enrich` caches public PR descriptions and complete changed-file lists without
making an admission decision; oversized merges are marked for manual diff
inspection and rate-limit failures remain in the audit log. Where GitHub
supplies a textual source-file patch, the cache retains it for the review card;
omitted or truncated patches still require pinned-source inspection. Each fetched
detail is durably appended, so a resumed batch skips completed PRs without
rewriting the full detail cache. `triage` is an
outcome-blind review order: phase hints can be wrong and low-signal PRs remain
eligible for later batches. The review schema
and allowed decisions are checked by
`pitbench.repositories.pr_curation`.
`prepare-review-batch` freezes the chosen triage cohort and input hashes,
writes a linked `index.md`, public PR review cards, and editable JSON drafts containing
only pinned PR facts. In this optional schema 2 path, a reviewer fills the decision, rationale, mechanism, and
correctness fields; `review_duration_sec` may record active review time. Drafts
do not count as decisions. Place completed reviews in a new tracked directory
under `configs/pr_curation/reviews/`, then pass that directory to `screen` with
`--review-dir`. The entire batch is validated before the registry is written
once; the existing `--review-file` option also accepts repeated files.
Re-running `prepare-review-batch` with the same queue and detail inputs keeps
edited drafts and the frozen manifest. After screening or later evidence
stages, run:

```bash
uv run python -m scripts.curate_solver_prs screen \
  --catalog configs/pr_curation/repositories.json \
  --registry report/pr-scale-20261018/pr_registry.jsonl \
  --review-dir configs/pr_curation/reviews/pilot_01
uv run python -m scripts.curate_solver_prs review-batch-status \
  --catalog configs/pr_curation/repositories.json \
  --registry report/pr-scale-20261018/pr_registry.jsonl \
  --manifest report/pr-scale-20261018/review_batch_01/manifest.json \
  --output report/pr-scale-20261018/review_batch_01/status.json
```

The batch status
reports the cohort's decisions, evidence stages, ready count, and recorded
review hours. This measures throughput for that selected queue; it is not an
estimate of readiness among unreviewed PRs. The raw registry and detail batches
must be retained alongside their manifest hashes to reproduce the cohort.
Use repeated `--details` arguments for all fetched batches. The `triage`
options `--scope-status task_backed --review-order throughput` schedule approved,
already task-backed solver sources first and puts explicit optimization terms
ahead of weaker title signals; it retains the low-signal audit lane. The
default `coverage` order spreads review across the full catalog. Both orders
only schedule source review and do not admit or reject PRs.

Bulk acquisition and screening are also available in the standalone
[pitbench-curation](https://github.com/Tarseus/pitbench-curation) project. It
provides PR-link-list intake, paced GitHub collection, concurrent Git source
checks, durable per-PR attempts, resume, and hash-bound exports. Its default
source/prepare path schedules measurement without a prose or model review,
retaining solve-time, solution-quality, and peak-memory objectives. Memory-only
candidates stay in that queue when time/quality gains are inconclusive; the
formal resource protocol now allows qualification after a primary metric,
per-task tradeoff limits and independent confirmation have been declared.
Command or HTTP mechanism review remains optional. Default measurement exports
also contain automatically extracted schema 3 prechecks in `reviews/`; the same
`screen --review-dir` entry accepts them without a model or mechanism narrative.
Their `advance` is scheduling approval. PitBench still validates the
solver build, workload, mathematical correctness, performance protocol, and
real Judge integration before counting a task as ready.

The optional `semantic` stage in pitbench-curation classifies PR title/body and
the pinned source-analysis diff through a local System One compatible model or Ollama
structured output. It retains multiple labels, unknowns, material ranges and raw
model responses; long inputs are processed in explicit chunks. Exported labels
use `classification_basis: semantic_model` and propose objective-specific
scenarios. They cannot declare a primary metric or satisfy task admission;
confirm the objective and freeze its scoring contract before qualification.

For older detail caches that contain patch hashes but no patch text, clone the
public solver repository with Git and run `cache-git-diffs` on a frozen review
manifest. Supply one `--repository-checkout REPOSITORY=PATH` mapping per Git
mirror and `--fetch-missing` to request the exact registry base, head, and merge
SHAs from the mirror's checked GitHub origin. The command compares the full
integrated Git file list with the cached PR detail list. For a merge or squash
commit whose first parent is the pinned base, it extracts the integrated diff.
If the integration base advanced, it accepts the pinned branch-base diff only
when its file list and complete binary patch equal the actual integrated diff byte for
byte; other base drift and unavailable merge commits receive separate statuses.
The command writes full-index source diffs, hashes, per-PR status, and a linked
diff index after each PR. `--limit` and `--git-timeout-sec` bound a fetch batch;
reruns keep verified cached patches and unvisited status records if interrupted.
An unchanged merge SHA rejected by remote Git is retained without repeated
fetches; `--retry-unavailable` explicitly retries it. Source diffs are review material, while
`source-verify` separately checks patch application and source-byte identity.
Screening and stage transitions validate their changed records and summarize
the registry without replaying all historical run artifacts. `audit` remains
the full evidence-file and hash validation command.
`workload-evidence` binds a frozen instance manifest and activation record to
the reviewed selection rule, verifies per-case solver-path counts and trace
hashes, then advances a build-verified PR only when at least one case exercises
the changed path.
`public-correctness` converts a complete paired public run ledger into
hash-bound Base/PR observations, reruns the independent original-problem
verifier on each returned solution, retains normal exact-solver timeouts, and
requires the review's bound/status/certificate checks. `baseline-evidence`
then derives and validates an open-opportunity record from the full hidden
baseline ledger.
The command copies the independent checker into an evidence file named by its
content hash before validation, so later verifier edits cannot silently change an
earlier PR's checker. Eight earlier correctness records were redirected to
byte-identical frozen copies with their original JSON retained in the local
migration audit.
`integrate` checks the pinned Git-backed Agent image, task configuration,
complete real Judge grid, valid solutions, and normal exact-solver timeouts
before moving that PR to `task_ready`. Both commands write new evidence files
and retain the original run records.
VROOM #142 remains `needs_evidence`: its TSP speedup has no demonstrated call
path in the current CVRP workload. OR-Tools #1496 is rejected as a distinct
task after its source diff was checked against the reverted #1493 patch and
the later #1498 reapplication. VROOM #146 is now a task-ready open opportunity. Its source patch and paired
historical builds passed; debug counters activated the reviewed path on ten
public cases, and all twenty returned routes passed independent checks. The
base-only 38-case hidden qualification completed with a 4.58% mean normalized
gap at the 30-second primary budget. A 10-second pilot timeout on X-n957-k87 is
retained; the instance remains in the task. The historical CLI returns only its
final route, so the task evaluates verified gap at fixed budgets and records
completion time; it makes no claim about partial incumbent trajectories or PDI.
The 12-cell no-op Judge integration passed without crashes. No coding agent or
PR-patch performance result informed admission.
PyVRP #1170 and #1164 remain `needs_evidence` despite earlier public-validity
screens: the former changes shipment insertion, and the latter caches operators
that require shipments, mutually exclusive groups, or reload depots. The CVRP
panel does not exercise those changed mechanisms. The low-signal audit recovered
SHOT #9's bilinear auxiliary-variable deduplication; its pinned source patch is
verified, while its MINLP build and independent semantic checks remain pending.
The low-signal review also put PyVRP #843 in `needs_evidence`: it skips route
distance evaluation only when distance is absent from both the objective and
maximum-distance constraints, a structure the current CVRP driver does not
create. A title or changed solver path alone does not prove workload activation.
PyVRP #376 passed source-only patch reproduction and paired historical builds,
but its time-window and fleet-cost move-evaluation branches still need a frozen
activated panel; a capacity-only CVRP smoke run does not make it a task.
HiGHS #3213 reduces the early strong-branching LP budget at a pinned 1.15.1
snapshot. Source probes recorded the changed budget on a public knapsack MIP
and no calls on a scheduling control. Its 10-case paired public panel checked
20 valid bounds and statuses, with 18 independently certified optimal solves
and two normal timeouts. All 60 hidden optima and reference solutions were
rechecked against their original models; the pinned base proved 48 optimal and
normally timed out on 12. Its Git-backed Agent image completed 40/40 limited
real Judge observations, with 18 valid optimal solves and 22 normal timeouts.
It is an open opportunity; no hidden human PR gain is claimed. HiGHS #651
changes RENS/RINS effort accounting and heuristic LP behavior at a pinned
1.1.1 snapshot. Public probes reached both heuristic paths on knapsack and
scheduling MIPs, while a trivial binary control reached neither. Its paired
public panel independently checked 20 bounds and statuses, with 12 valid
optimal solves and eight normal timeouts. The pinned base proved 19 of 60
hidden optima and normally timed out on 41; its Git-backed Agent image passed
40/40 limited real Judge observations under the established eight-CPU,
eight-worker allocation, with ten valid optimal solves and 30 normal timeouts.
The human PR's hidden gain is not claimed. The first Judge attempt was rejected
after a harness code change, and a separate one-CPU/eight-worker attempt is
retained as a resource-mismatched diagnostic, not as task integration evidence.
PyVRP #232 changes CVRP crossover repair to choose a nearby route by centroid.
Pinned probes reached greedy repair on a multi-route case and recorded no
centroid choices on a single-route control; the hidden task scope records the
separate time-window cost branch as unmeasured. Twenty paired public solutions
passed independent route checks. Its 48-case pinned-base pilot had a 26.66%
mean normalized gap at ten seconds, establishing substantial improvement room;
the Git-backed Agent image completed 40/40 valid limited Judge observations.
No hidden human PR gain is claimed. VROOM #383 has passed paired historical
builds but still needs priority-triggered workload and mathematical evidence.
Choco #1154 changes weighted CP
variable selection. Source probes reached the new selector and conflict-update
path on public subset-sum and graph-coloring cases, and neither path on an
input-order control. The task fixes `dom_over_wdeg` search for Agent development,
public comparison, and hidden judging. Twenty paired public optimality claims
passed independent DP or clique certificates; both source states passed the
same BlackBoxTest suite with the PR's exact one-line obsolete-test adjustment.
The pinned base proved 59/60 hidden optima and normally timed out on one; its
Git-backed Agent image completed 40/40 valid limited Judge observations with
the fixed strategy verified in every process record. No hidden human PR gain
is claimed.
PyVRP #783 replaces per-location visit counts with a packed bitmap in each
candidate `Solution` object. Bounded source probes observed the constructor on
242-location and three-location CVRP cases in both source states, and all four
public outputs passed independent route checks. A 48-case pinned-base pilot at
10 seconds had a 1.81% mean normalized gap, establishing verified improvement
room; the Git-backed Agent image completed 40/40 valid limited Judge runs. This
measures an open memory and construction-efficiency opportunity; no hidden
human PR gain is claimed.
The first structured reviews rejected HiGHS #140 as a numerical-correctness
fix. HiGHS #2861 passed pinned builds, public
correctness, a verified hidden exact-solve baseline, and a complete 20-run
limited Judge integration; it is an open-opportunity task. HiGHS #693 changes
binary-implication cache storage and shares its pre-PR commit with #690, but
uses a distinct implication-focused Agent goal and workload. Its public checks
covered 20 valid bounds, 22 statuses, and six optimality claims. The hidden
base completed 60/60 runs, with one verified optimum and 59 normal timeouts.
The real limited Judge run completed 400/400 observations, with two verified
optimal solves and 398 normal timeouts across base and empty-patch Agent states.
It is an open-opportunity task; no human PR gain is claimed.
HiGHS #690 changes MIP cut aging. Its pinned base solved 15 of 60 hidden
knapsack ID and scheduling OOD cases at the declared 20-second budget; the
other 45 were normal timeouts. Public checks covered 20 valid bounds, 22 status
claims, and six optimality claims, with independent SCIP proofs on three cases.
Its limited real Judge run completed 40/40 observations, including eight valid
optimal solves and 32 normal timeouts. The human PR gain is not claimed.
HiGHS #618 changes directional reliability in strong branching. A public MIP
trace reached the one-direction reliability condition in both source states;
its base solved 14 of 60 hidden ID/OOD cases and normally timed out on 46.
Independent public checks covered 20 bounds, 22 statuses, and four optimality
claims. Its real Judge integration completed 40/40 observations, with 12 valid
optimal solves and 28 normal timeouts. It is a separate algorithmic-search
open challenge, not a confirmed positive human reference.
Choco #1152 targets search-decision construction in a pinned 5.0.0-beta.2
snapshot. Its public panel independently checked ten returned solutions, six
optimality claims, and four feasible-incumbent timeouts; both source states
passed 33 targeted tests. The final pinned base completed 60/60 hidden runs:
32 proved optimal and 28 normally timed out with valid incumbents. A Git-backed
Agent image passed 40/40 real limited Judge observations. The separate
conflict-strategy changes were inactive on this task's basic-search panel.
Choco #1178 targets integer arithmetic propagation on a balanced subset-sum CP
panel. Fixed-cardinality dynamic programming independently proves all 60 hidden
optima, including zero targets. Public traces reached both changed propagators
in both source states and neither on a bin-packing control. Twelve public
solutions and optimality claims were independently checked; both source states
passed 84 targeted tests. The hidden base proved 35/60 optimal and normally
timed out with valid incumbents on 25; its Git-backed Agent image passed 40/40
limited Judge observations. Neither Choco task claims a hidden human PR gain.
PyVRP #511 uses a 0.8.0a0 snapshot to test the default local-search operator
portfolio. A runtime profile observed 15 operator registrations in Base and 12
in the source-only PR build; a symmetric control disabled the same three
operators in both states and returned independently valid routes. Ten public
solutions and 108 tests per source state passed. The final pinned base completed
48/48 hidden runs with a 31.05% mean normalized gap to independently checked
best-known routes. Its Git-backed Agent image passed 40/40 limited real Judge
observations. The human PR hidden effect is not claimed.
VROOM #978 adds a route-level TSPFix local-search move. Matched source
instrumentation observed 5,282 attempts and 208 applications on a visible
multi-job CVRP instance, and zero attempts on a one-job control. The production
images completed five paired public runs with independently valid routes. The
pinned base completed 48/48 hidden ID and structural-shift runs with a 2.59%
mean gap to independently checked best-known routes. Its Git-backed Agent
image passed 60/60 limited real Judge observations; no hidden human PR gain is
claimed.
The first newly domain-approved source, Pumpkin #212, also passed structured
review and source verification; its build and solver evaluation remain pending.
PyVRP #720 remains `needs_evidence`: its reported large speedup concerns ILS,
while the pinned base's `Model.solve` runs a genetic algorithm. PyVRP #966
passed source, pinned build, runtime code-path activation, and public
independent correctness checks. Its frozen 380-pair hidden human-reference
comparison completed without missing or failed pairs but was inconclusive in
both execution orders. Its independently verified baseline gap qualifies the
snapshot as one open-opportunity task, with the human PR explicitly recorded as
inconclusive. Its Git-backed Agent image completed a 480/480 valid limited
Judge integration.
PyVRP #988 changes ILS search policy. A public Python coverage trace reached
its new exhaustive-search branch, and a symmetric evaluator-owned test
adjustment passed in both validation states. Its one-seed full hidden ID
baseline qualified a second open-opportunity task; no positive human PR effect
is claimed. Its Git-backed Agent image completed a 480/480 valid limited
Judge integration.
PyVRP #967 and #1190 were admitted as distinct load and unload route-reuse
tasks only after their respective fast paths executed 131,625 and 465,801 times
in separate public probes. #974's changed neighbourhood lines and #1155's
promising-state bitset were likewise traced on real CVRP solves. Each of these
four open-opportunity tasks passed a complete 480/480 valid limited Judge run;
none is labeled a positive human reference without hidden paired confirmation.
VROOM #1222 adds an insertion-cost lower bound. Its evaluator-only validation
build recomputed 1,958,953 pruned single-job insertion scores across ten public
CVRP instances with zero violations. The historical base had a 2.693% mean
verified gap across all 38 hidden ID instances at one fixed seed, qualifying a
third open-opportunity task without claiming a hidden human-PR gain. VROOM
driver reports the independently computed original-model objective and keeps
the solver's integer-matrix cost separately for audit.

The SCIP and CBC tasks reuse hash-pinned MIP instance generators for direct
cross-solver comparison. Their new hidden oracles stay in evaluator-owned
private storage. Local pinned-source image builds, all five public development
instances, and five instances from each hidden panel per solver have been
checked. Both tasks also passed a 40-run limited empty-patch dry-run. Its
one-instance directional labels are sampling noise, not performance evidence;
full hidden-grid calibration and agent runs are still needed before reporting
comparative results for these tasks.

---

## 🚀 Quickstart

PitBench requires Linux x86-64, Python 3.12+, [uv](https://github.com/astral-sh/uv),
Docker Engine with Compose, and an installed and authenticated Codex or
Antigravity CLI. Formal runs also require the evaluator-provided private bundle
and configured agent and judge images.

With those prerequisites provisioned, install PitBench, connect the native CLI
login, and start a Codex run:

```bash
git clone https://github.com/Tarseus/PitBench.git && cd PitBench
uv sync
cp config/evaluate.example.yaml config/evaluate.local.yaml
uv run pitbench auth login codex
uv run pitbench auth status codex
uv run pitbench evaluate pyvrp_v0_14_0 \
  --agent codex \
  --model gpt-5.6-sol \
  --agent-kwarg reasoning_effort=xhigh
```

To connect and use Antigravity:

```bash
uv run pitbench auth login agy
uv run pitbench auth status agy
uv run pitbench evaluate pyvrp_v0_14_0 \
  --agent antigravity \
  --model gemini-3.1-pro-high
```

`auth login` first reuses a configured or native credential file. For agy on
Linux, it also imports the existing desktop keyring login into
`.pitbench/credentials/antigravity-oauth-token` with permissions `0600` and a local
ignore rule. Keyring import uses `secret-tool` or `/usr/bin/python3` with
`python3-gi`; a locked or inaccessible keyring produces a diagnostic instead of
being treated as a missing login. If no credential exists, the command opens the
native CLI. Complete its sign-in and exit the CLI to return to PitBench.
The separate Gemini CLI `oauth_creds.json` is not an agy credential.

The command creates or updates `config/evaluate.local.yaml`, preserving other
agents and task settings. `--config` selects another untracked local YAML.
`--proxy-url URL` records the proxy used for native login and evaluation;
`--model MODEL_ID` optionally records a default model. Existing settings are kept
when those options are omitted. Do not reuse another machine's proxy port.

On a server without a browser, Codex supports device authorization:

```bash
uv run pitbench auth login codex --device-auth
```

Codex native login requests file credential storage for that invocation so
PitBench can read `auth.json` under the native `CODEX_HOME` (default `~/.codex`).
It does not rewrite the user's global Codex configuration. If existing Codex
credentials are available only in an OS keyring, complete this native login or
provide an exported `auth.json`. See the
[official Codex authentication documentation](https://developers.openai.com/codex/auth)
for browser/device login and managed storage restrictions.

For an exported credential or unattended setup, use:

```bash
uv run pitbench auth login agy --auth-file /private/exported-agy.json --non-interactive
uv run pitbench auth login codex --auth-file /private/auth.json --non-interactive
```

`--non-interactive` never opens a login UI. `--force-login` opens the native login
CLI even when a credential is already configured. Credential contents are never
printed. Authentication state is machine-local; users authorize their own accounts.

`auth status` requires no solver task or private test data. It checks the
configured Codex credential through native login status, or queries agy models
inside the configured container, and checks the runner. It does not start model
inference or benchmark trials. Docker or runner-image problems remain separate
from login failures; a failed model request is not proof that an account signed out.

To evaluate several configured tasks with the same agent and model, pass their IDs
to the same command (replace `MODEL_ID` with an ID from `agy models`):

```bash
uv run pitbench evaluate pyvrp_v0_14_0 highs_v1_15_1 --agent antigravity --model MODEL_ID
```

Configure each task's images under `tasks` in the local YAML. The
command checks every task and the selected agent before starting any agent work.
Add `--check-only` to inspect prerequisites and collection coverage without
materializing tasks or running agents. Tasks share one run directory and run
serially by default; `--n-concurrent` controls trial concurrency.
When Configuration Robustness is enabled, the check also probes each configured
collector image and rejects stale compiled JVM adapters before agent work.

PitBench fetches each pinned source release once into
`.pitbench/cache/source-snapshots` and reuses the validated snapshot for later
runs. No manually downloaded solver repository or local source path is required.
The first run requires access to the declared Git source when the cache is
empty. Generated task images copy pinned submodules from that validated cache,
initialize them without network access, and check the full recursive submodule
state before starting an agent. This also repairs older prepared solver images
that omitted submodules.
Prepared task images are reused when their source is pinned by
digest and their revision, tooling, and copied development data match; mutable
image tags are rebuilt. BASE solver observations use the separate cache under
`.pitbench/cache/base`, keyed by the task configuration and effective judge image
so replacing an instance panel cannot reuse observations from the old panel.

By default, the task image includes visible development instances and the
`pitbench inspect`, `pitbench workspace`, `pitbench bench`, and
`pitbench diagnose` commands. Inside
the agent task, `pitbench inspect` shows the protocol, primary budget, seeds,
runner, build commands, and public targets. Run `pitbench bench --label base`
before editing, then run `pitbench bench --label agent --build` on the
same instances. The bench command defaults to the task's primary budget and
first development seed. It writes per-run results and paired development
feedback under `/pitbench/runs`; `--build` refreshes the current solver and the
evaluator still judges independently. `--instance` selects a public ID/name or
any readable custom instance file; `--seed`, `--budget`, and `--threads` control
each development run. `--all-seeds` and `--all-budgets` optionally run the full
declared development grid; use the same selection for Base and Agent. Runner
timeouts and start failures also leave per-run feedback and available logs.
If the configured results directory is read-only, `pitbench bench` automatically
uses a task-specific directory under the container's temporary directory and
prints the selected path.
After paired Base and Agent runs, `pitbench diagnose` aggregates results by
independent development instance. It reports better/equal/worse instance
counts, whether an apparent gain is concentrated in one instance, runtime and
available solver-work changes, matched Base/Agent objective and bound progress
at 25%, 50%, 75%, and 100% of each budget, missing grid cells, and a bounded
next experiment. Each checkpoint uses the best observed value by that time;
missing or final-only traces are reported as unavailable. These traces diagnose
search behavior but do not change the benchmark score.
`pitbench diagnose --all-budgets` includes every declared development budget.
The generated diagnosis is development evidence;
it does not claim hidden-instance generalization or promote a patch by itself.
For a matched feedback experiment, `agent_feedback_mode: outcome_only` keeps
the same visible instances, helper commands, task instruction, and final judge,
but omits structured solver-behavior summaries from public feedback.
`solver_behavior` is the default. Raw solver logs remain available in both
conditions for audit, so this comparison isolates the supplied structured
feedback rather than all possible solver profiling.
For HiGHS, independent development verification applies to the hash-bound
provided LP instances; custom LP files still run but are marked unverified.
Feedback records the input hash and runner artifact hash
so Base/Agent comparisons expose stale builds. Set `agent_tools: []` explicitly
in the local config to omit these tools and instances.

The normal candidate judge collects Performance and Resource Efficiency data,
plus the seed and representation runs enabled by each task config. Solver
failures, missing runs, and independent solution checks remain in the raw
evaluation records. The current benchmark does not include an
Operational Reliability score because its completed panels did not distinguish
Base from Agent.
Configuration Robustness still uses the separate release collector and is not
part of candidate judging. The task IDs alone do not imply complete coverage of
all active measures; in particular, the current HiGHS task does not enable
representation runs. Provision its private assets and judge image before launch.

The default Codex configuration uses `runner_backend: workspace`. Codex and its
native shell/file tools run inside the solver container, while a separate relay
frontend sidecar is its only permitted network peer. That sidecar has no internet
route or credential; it bridges over a per-run Unix socket to a host relay that
holds OAuth, accepts only Responses requests for the assigned model, rejects web
search, and applies per-trial request/concurrency admission limits, a
response-accounted token cutoff, and a request-boundary duration cutoff.
Codex's network proxy allowlists only the frontend sidecar IP plus container-local
loopback.
On hosts with `kernel.apparmor_restrict_unprivileged_userns=1`, workspace
containers use the distribution's `bwrap` AppArmor profile, which permits the
nested user namespaces. The profile must be loaded on the Docker host (provided
by the distribution's bubblewrap package). An explicitly `unconfined` container
is still affected by this kernel restriction and can fail with
`bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`.
The run starts only after a real Codex shell probe demonstrates that the relay is
reachable, the public internet is not, and no relay credential exists in the shell.
A `profile_path` overlay may include user-installed
plugins, skills, hooks, and local MCP servers; server-backed connectors remain
offline unless PitBench is explicitly extended with a separately isolated
capability relay.

`runner_backend: container` remains available as the compatibility fallback.
It runs the locally installed CLI in a short-lived control-plane container and
uses the bounded PitBench MCP tool surface to operate on the offline solver
container. Neither backend mounts the host home or Docker socket into the agent.
Remote builds currently require `runner_backend: container`; PitBench rejects the
workspace backend before allocating remote resources.
Set `proxy_url` only on machines that require one for model access.

If the run cannot start, diagnose the current shell and machine with
`uv run pitbench doctor pyvrp_v0_14_0`. Custom Codex and Antigravity profiles can be
created with `pitbench profiles init` and selected through `profile_path` in the
local configuration. PitBench records the runner image ID and profile hash with
every trial.

Agent execution evidence is recorded automatically for every agent through one
harness entry, including custom imported agents. See [Agent execution
traces](AGENT_TRACING.md) for the event format, reconstruction, and observation
boundaries.

### Schedule multiple agents externally

Each `pitbench evaluate` invocation evaluates one externally selected agent on
one or more configured snapshot tasks. Agent/model selection and batch scheduling stay outside
the benchmark task definition. The bundled shell script is an example serial
scheduler whose agent/model combinations are read from an experiment YAML:

```bash
scripts/run-model-matrix.sh pyvrp_v0_14_0 configs/experiments/model_matrix.yaml
```

The script makes independent `pitbench evaluate` calls in sequence and records
their command logs plus `status.tsv`. Other schedulers can invoke the same
single-agent command concurrently or serially using the same evaluation interface.

### Generate a Performance-First Report

Each trial stores observations under `runs/<run_id>/<task_id>/<trial_name>/evaluation/trials.parquet`. The report command accepts either that file or its containing `evaluation/` directory:

```bash
uv run pitbench report \
  runs/<run_id>/<task_id>/<trial_name>/evaluation \
  --task-config configs/tasks/<task_id>.yaml

uv run pitbench report \
  runs/<run_id>/<task_id>/<trial_name>/evaluation \
  --task-config configs/tasks/<task_id>.yaml --json
```

The public boundary suite remains available as an explicit diagnostic. It is
not run by the current benchmark's normal `evaluate` or `judge`
commands. To run only these cases on an already captured candidate patch:

```bash
uv run pitbench judge pyvrp_v0_14_0 candidate.patch \
  --expected-patch-sha256 <captured-patch-sha256> \
  --reliability-only --output-dir runs/boundary-check
```

This uses the same repository snapshot and isolated solver image configuration as
the regular judge. `reliability/manifest.json` records inputs and expected behavior;
`reliability/report.json` reports pass counts for each budget and code state;
`reliability/details.json` lists failures, missing runs, and paired changes. All
observations remain in `trials.parquet`, with process logs and solutions alongside.

A passing run must end normally and return a feasible solution that passes the
independent verifier. A native budget stop with a feasible incumbent passes; an
external watchdog timeout does not. Crash, timeout, explicit out-of-memory,
solver error, missing or malformed output, no solution, and invalid solution are
reported separately. Invalid solutions still fail the existing qualification
check. The integral MIP cases also check reported objective values and false
optimality claims against their known optima.

The diagnostic pass rate counts instance–seed runs within each budget, with the
whole declared grid as its denominator. It is left unset when runs are missing.
It describes this test suite, not a deployment success probability. Boundary
observations are excluded from performance, resource, and seed-IQR aggregation.

---

## 🛠️ Verification & Developer Tooling

The 2026-10-21 curation review fixes are published on
`curation/pr-acquisition-review-20261021`. This branch includes the objective
schemas, metadata screening interface, and source recovery changes used by
the independent curation project; they are not asserted to be on `main`.
Squash Base drift requires byte-identical complete patches, renames include
both paths, and source-only and complete integrated patches have separate
hashes. Unknown source formats remain candidates for review.

For this snapshot, the curation project's full suite passed 41 tests. PitBench's
PR curation, source cache, and repository plugin suites passed 71 tests. The
expanded `tests/unit/pitbench` plus exact runner check passed 446, skipped 17,
and failed 6 because the local private oracle/seed assets were absent. No
solver qualification or new `task_ready` is claimed by these checks.

Repository code organization and cleanup rules are in [AGENTS.md](AGENTS.md).

Experiment tools select implementations through task configurations and shared
modules. To generate empirical BKS anchors, run inside the appropriate solver
judge environment with a clean source snapshot:

```bash
python -m scripts.generate_shift_bks \
  --task-config configs/tasks/<task_id>.yaml --repository /input/base \
  --instance-set-config /private/hidden_instances/<panel>.yaml \
  --private-root /private --output-dir /private/oracles/<panel> \
  --budget-sec 30 --seeds 0 1 2 3 4
```

`--task-config` and `--repository` replace the former `--solver-python` option.
The shared judge builds the configured repository and independently verifies each
solution. Budget and seeds are explicit anchor-generation inputs. Inputs, logs,
solutions, and observations are retained; an incomplete or invalid run grid does
not produce a new oracle. Solution references follow the actual output location.

Seed validation requires an explicit `--task-config` defining
`evaluation.seed_robustness`; there is no solver or release whitelist. A prepared
batch can run selected tasks sequentially:

```bash
bash scripts/run-seed-validation-matrix.sh /absolute/batch /absolute/private <task_id>...
```

The batch must contain `harness/` and `sources/<task_id>/`. Each task supplies its
`repository.judge_image`, or `PITBENCH_JUDGE_IMAGE` supplies an explicit override.
Use `PITBENCH_PYTHON` to select the host Python environment. The batch reuses the
single-task validation runner and its existing sampling settings. Fixed-instance
seed intervals are computed by the shared seed report; the old batch-specific
interval-recomputation implementation has been removed.

Nuisance collection uses one configured entry point. The supplied panels retain
all original releases, budgets, seed IDs and transformation counts. Select a panel
and run preparation in its solver environment (numeric input parsing needs the
installed binding). Source-based execution reuses the normal repository judge;
installed execution reuses the parameter-search worker and scheduler.

```bash
python -m scripts.collect_nuisance_results prepare \
  --config configs/experiments/nuisance_robustness.yaml --panel routing \
  --repository /path/to/source --output results/nuisance-routing
python -m scripts.collect_nuisance_results run \
  --output results/nuisance-routing --cpus <available-cpu-ids>
```

Use `--panel linear` without `--repository` in the numeric solver environment.
The configured input index can reference the retained experiment via an explicit
path template; it can also be any fixed instance index with `id` and `path` fields.
New panels save a common per-run result format, including failures and unrun jobs.
Previously retained experiment formats remain readable by the same report command:


```bash
python -m scripts.report_nuisance_results --source /path/to/experiment --output /path/to/report
```

The report reads a job manifest with per-run results, or judge checkpoints with
their transformation manifest. It derives group sizes and budgets from those
records, separates code states, and retains missing groups. CSV, JSON summaries,
and HTML share the same observations; no robustness statistic is selected here.

Parameter robustness uses a fixed number of SMAC3 configuration evaluations. Install
the controller's optional search dependencies with `uv sync --extra configuration`.
Each worker Python environment must contain its configured solver and the PitBench
dependencies needed by its collector and independent verifier. The experiment YAML
selects the collection backends, parameter domains, instance sources and seed lists;
task files supply solver releases and per-run budgets.

```bash
uv run --extra configuration python -m scripts.collect_configuration_results prepare \
  --config configs/experiments/configuration_robustness.yaml \
  --output results/configuration-robustness \
  --solver-python pyvrp=/path/to/routing-environment/bin/python \
  --solver-python highs=/path/to/exact-environment/bin/python \
  --cpus <available-cpu-ids> --search-seed <searcher-seed>

uv run --extra configuration python -m scripts.collect_configuration_results run \
  --output results/configuration-robustness

uv run python -m scripts.collect_configuration_results report \
  --output results/configuration-robustness
```

Preparation copies the fixed inputs and records disjoint retest seeds, runtime
identities and CPU assignments without solving. Each search trial evaluates the
whole instance/seed panel before feedback; the supplied protocol allows 16 new
configurations per solver/budget. Search and retest results, effective parameters,
solutions, trajectories and every run attempt are retained. The JSON report keeps
paired degradation and unavailable feedback separate. Normal native time limits
use the approved capped-time feedback; unavailable feedback stops that search and
selects the offending configuration for retest. Collection faults pause instead;
after fixing the cause, `run --retry-collection-errors` retains failed attempts and
retries only collection errors. Re-running a completed search reuses its results.
The supplied full panel can consume about 87 solver-budget core-hours; smoke checks
with reduced temporary fixtures do not constitute the full robustness experiment.

```bash
# Validate task configs and contracts
uv run pitbench tasks validate

# Run the deterministic evaluator fixture
uv run pitbench tasks fixture --instances-per-instance-set 1

# Run the unit suite
ALL_PROXY= all_proxy= uv run pytest -q tests/unit
```

---

## 🧭 Roadmap & Future Directions

- [ ] **Broader Combinatorial Optimization Problem Domains**
- [ ] **Advanced Sensitivity & Distributional Metrics**
- [ ] **Hardware-Isolated Distributed Cloud Infrastructure**

---

## 📂 Repository Layout

```
PitBench/
├── pitbench/
│   ├── harness/         # Agent execution sandbox, loopback MCP terminal & container orchestration
│   ├── evaluator/       # Isolated read-only judge and evaluation input validation
│   ├── metrics/         # PerformanceReport, Cluster Bootstrap 95% CI & decision policies
│   ├── schema/          # Pydantic data schemas (RunObservation, EvaluationResult, PitBenchTask)
│   ├── repositories/    # Solver build and execution plugins
│   ├── solver_drivers/  # Solver CLI adapters and normalized output handling
│   ├── problem_families/ # Problem semantics and independent verification
│   ├── instances/       # CO instance generation and materialization tooling
│   └── cli/             # Unified CLI (`run/evaluate/judge/report/tasks`)
├── scripts/             # Optional external scheduling and maintenance scripts
├── configs/tasks/        # Public task configs
├── configs/instance_sets/ # Public instance-set configs
├── adapters/pitbench/    # Prepared PyVRP Docker adapter
├── private/              # Maintainer-provisioned hidden evaluation assets
└── tests/unit/           # Unit test suite
```

---

## 🏷️ Keywords & Topics

`combinatorial-optimization` • `operations-research` • `llm-agents` • `code-optimization` • `pyvrp` • `vehicle-routing-problem` • `performance-engineering`

---

## 📝 Citation

If you use PitBench in your research, please cite our work:

```bibtex
@misc{pitbench2026,
  title={PitBench: Benchmarking Coding Agents on Real-World Solver Improvement},
  author={Tarseus Team and Contributors},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/Tarseus/PitBench}}
}
```

---

## License

PitBench is distributed under the [BSD 3-Clause License](LICENSE). The root
license retains the FormulaCode Developers notice and adds the PitBench
Contributors notice for subsequent work. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for provenance; the FormulaCode
snapshot under `upstream/` retains its original license unchanged.
