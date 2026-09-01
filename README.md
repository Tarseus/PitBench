# PitBench: Benchmarking Coding Agents on Real-World Solver Improvement

*A repository-level benchmark for combinatorial optimization solvers.*

PitBench is a benchmark for evaluating whether coding agents can make
statistically significant, generalizable performance improvements to mature
combinatorial optimization solver repositories.

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

**Combinatorial Optimization (CO)** is the algorithmic backbone of modern logistics, supply chains, manufacturing, and scheduling. Developing and accelerating high-performance solvers requires mathematical insight, intricate search heuristics, and low-level systems programming. PitBench is designed for repository-level evaluation of CO solvers; **PyVRP is the first implemented track**, currently represented by four release snapshots, rather than the boundary of the benchmark.

Unlike general repository-level coding benchmarks, solver optimization must preserve problem semantics while evaluating a randomized quality-time process:

* 🧠 **Complex Algorithmic Search Spaces**: PyVRP relies on Hybrid Genetic Search, local search neighborhoods, population management, and adaptive control.
* ⚡ **High-Performance Hybrid Architectures**: Solvers heavily utilize C++/Python hybrid designs (e.g., C++ computational kernels exposed via Pybind11 / Cython) where minor changes can break memory invariants or cause severe algorithmic regressions.
* ⏱️ **Stochastic Quality-Time Tradeoffs**: Combinatorial search is inherently randomized. Speedup is meaningless if solution quality (optimality gap) degrades. Benchmarking must evaluate the quality-time Pareto frontier under fixed computational budgets.

**PitBench provides a problem-aware testbed for agentic performance engineering on real CO solver repositories.**

---

## 🌟 Key Highlights

1. **Repository-Level CO Solver Optimization**
   Agents modify real solver repositories rather than isolated algorithm exercises. The current implementation uses PyVRP release snapshots.

2. **Four PyVRP Release Snapshots**
   The supported tasks cover PyVRP v0.12.2, v0.13.0, v0.13.4, and v0.14.0 using common instances, seeds, and budgets.

3. **CO-Specific Quality-Time Performance-First Protocol**
   Evaluates true algorithmic improvements by measuring **normalized optimality gap reductions ($\Delta \text{gap} = \text{gap}_{\text{base}} - \text{gap}_{\text{agent}}$)** against official **Best-Known-Solutions (CVRPLIB-X BKS)** under **fixed time budgets (`budget_sec` = 1s, 5s, 10s)**, paired across identical solver seeds with **5,000 instance-cluster bootstrap 95% confidence intervals**.

4. **Prepared Docker Environments**
   PyVRP native extensions are compiled in prepared, commit-tagged images. Agent execution is network-disabled, while evaluator-owned hidden assets are supplied only to a separate judge environment.

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
│ • Fixed budgets (1s/5s/10s)  │   │ • Multi-seed paired trials    │   │ • Evaluate on unseen instances│
│ • Anchored to optimum / BKS   │   │ • Instance-cluster bootstrap  │   │ • Retention on hidden shifts  │
│ • True Gap Reduction:         │   │ • Strict 95% confidence bounds│   │ • Check retention beyond dev  │
│   Δgap = gap_base - gap_agent │   │ • Paired seed evidence        │   │   on judge_shift              │
└───────────────────────────────┘   └───────────────────────────────┘   └───────────────────────────────┘
```

### 1. Quality-Time Performance First
- **Quality and time stay coupled**: Faster execution is not treated as an improvement when independently verified solution quality regresses.
- **Independent Solution Verifiers**: Evaluator-owned verifiers check route feasibility and capacity constraints, and recompute objective values. BKS anchors then define normalized gaps.

### 2. Randomized Repeatability & Statistical Rigor
- Metaheuristics and randomized search algorithms exhibit high variance across seeds.
- PyVRP evaluations pair Base and Agent across five solver seeds and compute **instance-cluster bootstrap 95% confidence intervals (CI95)**. An improvement claim is accepted only when the lower bound is strictly positive ($CI_{\text{lower}} > 0$).

### 3. Generalization across Problem Topologies (Held-Out Retention)
- Evaluations test both calibration instances (`judge_id`) and unseen structural shift instances (`judge_shift`, e.g., clustered customer distributions, skewed demand profiles).
- The report shows whether the measured improvement is retained on the declared
  hidden-shift instance set; it does not claim universal generalization.

---

## 📦 Current Combinatorial Optimization Tasks

| Task ID | Solver & Version | Problem Family | Architecture | Optimization Scope |
| :--- | :--- | :--- | :--- | :--- |
| `pyvrp_v0_12_2` | **PyVRP** v0.12.2 | CVRP | C++ / Python (Pybind11) | Search and local-search efficiency |
| `pyvrp_v0_13_0` | **PyVRP** v0.13.0 | CVRP | C++ / Python (Pybind11) | Search and local-search efficiency |
| `pyvrp_v0_13_4` | **PyVRP** v0.13.4 | CVRP | C++ / Python (Pybind11) | Search and local-search efficiency |
| `pyvrp_v0_14_0` | **PyVRP** v0.14.0 | CVRP | C++ / Python (Pybind11) | Search and local-search efficiency |

---

## 🚀 Quickstart

PitBench requires Linux x86-64, Python 3.12+, [uv](https://github.com/astral-sh/uv),
Docker Engine with Compose, and an installed and authenticated Codex or
Antigravity CLI. Formal runs also require the evaluator-provided private bundle
and configured agent and judge images.

With those prerequisites provisioned, install PitBench and start a Codex run in
four commands:

```bash
git clone https://github.com/Tarseus/PitBench.git && cd PitBench
uv sync
cp config/evaluate.example.yaml config/evaluate.local.yaml
uv run pitbench evaluate pyvrp_v0_14_0 \
  --agent codex \
  --model gpt-5.6-sol \
  --agent-kwarg reasoning_effort=xhigh
```

Use Antigravity by replacing the final command with:

```bash
uv run pitbench evaluate pyvrp_v0_14_0 \
  --agent antigravity \
  --model gemini-3.1-pro-high
```

The default Codex configuration uses `runner_backend: workspace`. Codex and its
native shell/file tools run inside the solver container, while a separate relay
frontend sidecar is its only permitted network peer. That sidecar has no internet
route or credential; it bridges over a per-run Unix socket to a host relay that
holds OAuth, accepts only Responses requests for the assigned model, rejects web
search, and applies per-trial request/concurrency admission limits, a
response-accounted token cutoff, and a request-boundary duration cutoff.
Codex's network proxy allowlists only the frontend sidecar IP plus container-local
loopback.
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
`uv run pitbench doctor pyvrp`. Custom Codex and Antigravity profiles can be
created with `pitbench profiles init` and selected through `profile_path` in the
local configuration. PitBench records the runner image ID and profile hash with
every trial.

### Schedule multiple agents externally

Each `pitbench evaluate` invocation evaluates one externally selected agent on
one PyVRP snapshot task. Agent/model selection and batch scheduling stay outside
the benchmark task definition. The bundled shell script is an example serial
scheduler whose list of agents can be edited by the caller:

```bash
scripts/run-pyvrp-model-matrix.sh pyvrp_v0_14_0
```

The script makes independent `pitbench evaluate` calls in sequence and records
their command logs plus `status.tsv`. Other schedulers can invoke the same
single-agent command concurrently or serially without a PitBench-specific
matrix schema.

### Generate a Performance-First Report

Each trial stores observations under `runs/<run_id>/<task_id>/<trial_name>/evaluation/trials.parquet`. The report command accepts either that file or its containing `evaluation/` directory:

```bash
uv run pitbench report \
  runs/<run_id>/<task_id>/<trial_name>/evaluation

uv run pitbench report \
  runs/<run_id>/<task_id>/<trial_name>/evaluation --json
```

---

## 🛠️ Verification & Developer Tooling

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
│   ├── families/        # Independent CVRP verification
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
