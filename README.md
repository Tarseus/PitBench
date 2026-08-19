# PitBench

PitBench evaluates coding agents that improve real combinatorial-optimization
solvers. It inherits the agent, terminal, Docker, remote-execution, and provenance
infrastructure from `fc-eval`, while defining a separate solver-evaluation model.
The original project is preserved unchanged under [`upstream/`](upstream/).

## Evaluation contract

The generic `fceval` Harness only creates an environment, runs an agent, captures a
binary candidate patch, calls an evaluator plugin, and stores its opaque result plus
cost/trajectory metadata. It does not know about solver gaps, speedups, workloads,
ASV or distribution metrics.

PitBench records the raw grid

```
task × code_state × population × instance × solver_seed × budget
```

where `code_state` is `base` or `agent`. Validity is separate from performance.
Problem-level optimum/BKS oracles anchor solution quality; they are not code
references.

A production evaluation destroys the agent environment and launches a fresh,
network-disabled judge container from a digest-pinned image. Hidden instances,
independent verifiers, and oracle data are mounted only there.
`fixture_mode` is explicit, deterministic, and cannot be reported as a real solver
result.

## Implemented release snapshots

- `pyvrp_v0_13_4`: PyVRP v0.13.4, CVRP search track.
- `vroom_v1_15_0`: VROOM v1.15.0, heuristic routing track.
- `highs_v1_15_1`: HiGHS v1.15.1, exact MIP track.
- `choco_v6_0_1`: Choco v6.0.1, CP track.
- `ortools_v9_15`: OR-Tools v9.15, model-build auxiliary track.

Every task fixes a public release tag to its full commit SHA and declares a
snapshot-only information regime, repository/family plugins, build protocols,
explicit RNG dimensions, and agent-dev/hidden population definitions.

## Commands

```bash
uv sync --group dev
uv run pitbench tasks validate
uv run pitbench tasks materialize-dev --output dataset/dev
uv run pitbench tasks smoke --instances-per-population 1
uv run pytest -q tests/unit/pitbench
uv run pytest -q tests/unit -m 'not docker'
```

Materialized tasks expose the same agent-side developer interface:

```bash
pitbench inspect
pitbench bench --split dev
pitbench bench --split dev --instance dev_0000
pitbench profile --instance dev_0000
pitbench verify
pitbench diff
```

`tasks smoke` validates the full evaluator/storage contract with synthetic fixture
observations. It does not execute the public solvers. Real evaluation additionally
requires time-censored repository snapshots, private hidden assets, immutable Java
runners
for the Choco/OR-Tools tracks, and digest-pinned repository judge images.

## Layout

```
fceval/                 generic inherited execution harness
adapters/pitbench/      time-censored Git snapshot tooling
pitbench/schema/        task, validity, observation, and result contracts
pitbench/evaluator/     patch policy, isolated judge, artifact/Parquet storage
pitbench/repositories/  build/run plugins for the five solver repositories
pitbench/families/      independent CVRP and private MIP/CP verifier contracts
pitbench/metrics/       quality, stability, gain, and aggregation
pitbench/distribution/  research-only population discrepancy layer
manifests/              public task and agent-development population definitions
private/                hidden instances, verifiers, and oracle data
upstream/               untouched fc-eval legacy/reference implementation
```

`pitbench/evaluator` is benchmark truth. `pitbench/distribution` is an analysis
layer: Wasserstein, MMD, energy distance, or another discrepancy can change without
rerunning agents or changing judge validity.
