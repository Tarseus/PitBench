# PitBench

PitBench evaluates coding agents that improve real combinatorial-optimization
solvers, providing full execution harness, isolated Docker judging, and
6-dimensional (Outcome 3D + Sensitivity 3D) evaluation metrics.

## Evaluation contract

The PitBench execution harness creates the task environment, runs an agent, captures a
binary candidate patch, executes isolated judging, and stores the evaluation results.

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

Production PyVRP evaluation populates two independent three-dimensional views:

- outcome coordinates: performance, reliability, and resource consumption;
- input sensitivity directions: certified representation equivalence, frozen
  customer-count scale, and an independently seeded shifted population.

The evaluator also publishes the three population-conditional empirical Wasserstein
solver distances instead of collapsing unlike outcome units into one scalar. The
main quality result uses only the BKS-anchored `judge_id` population; equivalence and
shift panels remain diagnostics and cannot bias the primary gap or runtime summary.
CPU time and peak RSS are captured for every driver process. A versioned,
manifest-declared Pareto gate turns the evaluator-owned report into the generic
resolved verdict and fails closed when a required six-dimensional panel is missing.

## Implemented release snapshots

- `pyvrp_v0_12_2`: PyVRP v0.12.2, CVRP search track.
- `pyvrp_v0_13_0`: PyVRP v0.13.0, CVRP search track.
- `pyvrp_v0_13_4`: PyVRP v0.13.4, CVRP search track.
- `pyvrp_v0_14_0`: PyVRP v0.14.0, CVRP search track.
- `vroom_v1_15_0`: VROOM v1.15.0, heuristic routing track.
- `highs_v1_15_1`: HiGHS v1.15.1, exact MIP track.
- `choco_v6_0_1`: Choco v6.0.1, CP track.
- `ortools_v9_15`: OR-Tools v9.15, model-build auxiliary track.

Every task fixes a public release tag to its full commit SHA and declares a
snapshot-only information regime, repository/family plugins, build protocols,
explicit RNG dimensions, and agent-dev/hidden population definitions.
The four PyVRP snapshots share the same generated development population and the
same 38-instance CVRPLIB-X calibration population with published BKS anchors, so
cross-version comparisons use common instances, seeds, and budgets.
They also share a hash-pinned 10-instance hidden structural-shift generator and a
bounded customer-relabel equivalence panel.

## Commands

```bash
uv sync --group dev
uv run pitbench tasks validate
uv run pitbench tasks materialize-dev --output dataset/dev
uv run pitbench tasks smoke --instances-per-population 1
uv run pytest -q tests/unit/pitbench
uv run pytest -q tests/unit -m 'not docker'
```

The outcome geometries, population-conditional solver pseudometric, empirical
stochasticity, and instance-space sensitivity definitions are specified in
[`docs/solver-behavior-metric-definition.md`](docs/solver-behavior-metric-definition.md).
The provable instance-space metric and upper-bound certificates are specified in
[`docs/cvrp-problem-metric-definition.md`](docs/cvrp-problem-metric-definition.md).

## Codex agent with a ChatGPT subscription

PitBench runs Codex on the host while exposing only a loopback MCP command surface
for the assigned, network-disabled task container. Log in and install the isolated
runner once:

```bash
codex login
sudo scripts/install-codex-runner.sh
```

Re-run the installer after upgrading Codex because it copies the pinned executables.

```bash
uv run pitbench tasks materialize-dev --output dataset/dev
uv run pitbench run \
  --dataset-path dataset/dev \
  --agent codex \
  --model gpt-5.6-terra \
  --n-concurrent 1
```

The runner passes subscription authentication through standard input into a
temporary directory, removes it after the run, and executes as `pitbench-codex`, a
dedicated system user without Docker socket access. No API key or auth file is
mounted into the task container. Before exposing the task MCP endpoint, the agent
runs a bounded control-plane preflight with the selected model. Backend or proxy
failures therefore stop within 120 seconds and are recorded as
`codex-preflight.jsonl` and `codex-preflight.stderr.log`; the solver container stays
network-disabled throughout.

When the host requires an outbound proxy, pass it explicitly to the isolated runner:

```bash
uv run pitbench run \
  --dataset-path dataset/dev \
  --agent codex \
  --model gpt-5.6-terra \
  --agent-kwarg proxy_url=http://127.0.0.1:7897
```

The proxy applies only to host Codex. The task container remains offline, and the
loopback MCP endpoint remains in `NO_PROXY`.

## Antigravity agent with a Google subscription

PitBench can also run the host Antigravity CLI (`agy`) through the same bounded
loopback MCP surface. Complete the normal interactive sign-in and install the
isolated runner once:

```bash
agy
sudo scripts/install-antigravity-runner.sh
```

Re-run the installer after upgrading `agy` because it copies the executable used by
the dedicated runner account.

```bash
uv run pitbench tasks materialize-dev --output dataset/dev
uv run pitbench run \
  --dataset-path dataset/dev \
  --agent antigravity \
  --model gemini-3.1-pro-high \
  --n-concurrent 1
```

The runner copies only Antigravity's OAuth token and authentication mode through
standard input into temporary files with mode `0600`. It runs as
`pitbench-agy`, which has no Docker socket access, and removes the temporary HOME
after every trial. The task container remains network-disabled and receives no
credential files. PitBench deliberately does not enable Antigravity's
`--dangerously-skip-permissions` mode: the temporary settings allow only
`mcp(pitbench/run_command)`.

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
pitbench/harness/       execution harness, agent loopback, Docker sandboxes
pitbench/evaluator/     patch policy, isolated judge, artifact/Parquet storage
pitbench/repositories/  build/run plugins for solver repositories
pitbench/families/      independent CVRP and private MIP/CP verifier contracts
pitbench/metrics/       6D outcome & sensitivity metrics, summary matrix
pitbench/distribution/  research-only population discrepancy layer
pitbench/schema/        task, validity, observation, and result contracts
adapters/pitbench/      time-censored Git snapshot tooling
manifests/              public task and agent-development population definitions
private/                hidden instances, verifiers, and oracle data
upstream/               untouched reference implementation
```

`pitbench/evaluator` is benchmark truth. `pitbench/distribution` is an analysis
layer: Wasserstein, MMD, energy distance, or another discrepancy can change without
rerunning agents or changing judge validity.
