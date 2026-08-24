# PitBench

PitBench evaluates coding agents that modify real combinatorial-optimization
solvers. Its benchmark contract is performance-first: independently certify every
solution, measure quality at fixed time budgets, quantify randomized repeatability,
and check whether improvements remain on a hidden instance population.

## Evaluation contract

The PitBench execution harness creates the task environment, runs an agent, captures a
binary candidate patch, executes isolated judging, and stores the evaluation results.

PitBench records the canonical experimental grid

```
task × code_state × population × instance × solver_seed × budget
```

where `code_state` is `base` or `agent`. The experimental unit is
`instance × solver_seed × budget`, not an unstructured timing repeat. Base and Agent
runs are paired on that unit.

The primary report has four parts:

1. **Independent validity.** Evaluator-owned problem verifiers check feasibility and
   recompute objectives; solver-reported validity and objectives are not trusted.
2. **Quality-time performance.** At each fixed budget, feasible objective values are
   anchored to an optimum or best-known solution. Runtime alone is never called a
   speedup when solution quality changes.
3. **Randomized repeatability.** Solver seeds remain first-class observations. The
   report gives paired better/equal/worse counts, per-seed gap reductions, and an
   instance-cluster bootstrap 95% confidence interval. Repeatability is evidence for
   a performance conclusion, not a separate Stability score.
4. **Held-out population retention.** Judge-ID and hidden-shift improvements are
   reported directly, together with `shift improvement - ID improvement`. No
   instance-distance metric is required for this claim.

Problem-level optimum/BKS oracles anchor solution quality; they are not human code
references. This makes the benchmark about a randomized optimization process
`A(instance; seed, budget)`, rather than runtime of an equivalent deterministic
computation.

A production evaluation destroys the agent environment and launches a fresh,
network-disabled judge container from a digest-pinned image. Hidden instances,
independent verifiers, and oracle data are mounted only there.
`fixture_mode` is explicit, deterministic, and cannot be reported as a real solver
result.

The primary conclusion uses the largest declared fixed budget on the BKS-anchored
`judge_id` population. All budgets remain visible as a quality-time curve. A result
is classified as improved only when the paired judge-ID 95% interval excludes zero,
independent validity does not regress, and the point improvement remains non-negative
on `judge_shift`. A positive mean whose interval crosses zero is inconclusive.

The six-dimensional experimental metric definitions remain on the independent
`metric/definition` branch. They are not part of this performance-first branch or
its evaluator verdict.

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
They also share a fixed 10-instance hidden structural-shift population.

## Commands

Start a complete evaluation with one command. PitBench materializes the selected
release task, runs the agent, and invokes the independent judge:

```bash
uv run pitbench evaluate pyvrp_v0_14_0 \
  --agent codex \
  --model gpt-5.6-terra
```

Materialized tasks are retained under `.pitbench/tasks/<run-id>/`; results are
written under `runs/<run-id>/`. PyVRP release images are built ahead of time and
published to GHCR under commit tags. A new machine pulls the prepared PyVRP image
and builds only a thin PitBench tooling layer; later runs reuse the local thin
image. Pass `--rebuild` to force a refresh.

The publication matrix and recipes live under `.github/workflows/` and
`docker/solver-images/pyvrp/`. Each prepared image contains the frozen PyVRP
checkout, build toolchain, dependencies, and initial release build. The materialized task
Dockerfile adds only agent-visible tooling and development instances; its
`.dockerignore` excludes the locally cloned solver snapshot from the Docker context.
The PyVRP image build imports both native extension modules before publication, so a
missing or ABI-incompatible build fails CI instead of becoming a downloadable image.

Lower-level task and reporting commands remain available for custom workflows:

```bash
uv sync --group dev
uv run pitbench tasks validate
uv run pitbench tasks materialize-dev --output dataset/dev
uv run pitbench tasks smoke --instances-per-population 1
uv run pitbench report path/to/trials.parquet
uv run pytest -q tests/unit/pitbench
uv run pytest -q tests/unit -m 'not docker'
```

The `pitbench report` command emits only the performance-first report.

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
uv run pitbench evaluate pyvrp_v0_14_0 \
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
uv run pitbench evaluate pyvrp_v0_14_0 \
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
uv run pitbench evaluate pyvrp_v0_14_0 \
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
pitbench/metrics/       canonical performance report and decision policy
pitbench/schema/        task, validity, observation, and result contracts
adapters/pitbench/      time-censored Git snapshot tooling
manifests/              public task and agent-development population definitions
private/                hidden instances, verifiers, and oracle data
upstream/               untouched reference implementation
```

`pitbench/evaluator` is benchmark truth. Independent problem verifiers live under
`pitbench/families`; performance reporting and its decision policy live under
`pitbench/metrics`.
