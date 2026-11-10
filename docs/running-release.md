# Running a pinned benchmark release

`configs/benchmarks/0.0.1.json` binds the sixteen release tasks to eight solver
repositories, two official releases per repository, with their source commit and
tree hashes. Each task records its historical design template. Budgets, seeds,
instance panels, transformations and metric definitions are inherited unchanged.
Private seed files and exact-oracle inventories have new task ownership IDs and
hashes; their seed lists and reference records remain unchanged.

PyVRP 0.14.1 uses the existing `PyVRPDebugOptimizedRepositoryPlugin` for
validation: its debug logging cannot format `Measure` with fmt 10. ASan/UBSan,
O1 optimization and debug assertions remain enabled. The performance build and
scored protocol are unchanged. Build stdout, stderr, commands and return codes
are retained under the Judge output `build-logs/` directory.

Use a separate checkout and provision `private/` before running. Machine settings
belong in an ignored `config/*.local.yaml` file. The source-image builder accepts
`solver_image_build.base_images`, keyed by Docker build argument names such as
`UBUNTU_BASE_IMAGE`, `JAVA_BASE_IMAGE` and `PYTHON_BASE_IMAGE`. These permit an
offline import when the Docker daemon cannot reach the original registry. Record
and check the imported image identities before building. Default Dockerfiles keep
their original fixed upstream base-image references.

Keep one campaign under one `--output` directory. With default evaluation paths,
the release runner stores model records in `<output>/runs` and materialized
workspaces in `<output>/workspaces`, alongside baseline and model-status records.
Explicit custom paths in the local evaluation config are preserved. The local
workspace index is documented in `gushengda/README.md`; campaign inputs use their
organized paths directly.

Agent development storage follows the trial's run directory. The generated
Compose configuration uses local bind-backed volumes for `/workspace/repo` and
`/home/pitbench-agent`. Docker populates these from the task image on first use,
preserving prepared build artifacts, source permissions and user ownership.
The backing directories live at `<trial>/development-storage/{workspace,home}`;
`/tmp` and `/var/tmp` bind to sibling `tmp` and `var-tmp` directories. They remain
after container cleanup. Public tool output stays in `agent-logs/development-runs`,
and the trace collector does not ingest the entire backing repository as logs.
Docker image-only snapshots exclude mounted volumes; retain development storage
alongside the captured candidate patch and traces when restoring a trial.

For the g49 experiments, keep the checkout, configured run/workspace/cache paths
and campaign output physically under `/data3`, and set host temporary directories
before launching Python:

```bash
export TMPDIR=/data3/tmp/$USER
export TMP=$TMPDIR TEMP=$TMPDIR
mkdir -p "$TMPDIR"
```

Container caches and temporary files then use the run's data-disk mounts too.
Terminal recording explicitly uses `/agent-logs/.asciinema`, so recorder config
does not depend on an image's default HOME. Recorder startup failures preserve
the actual exit code and error output instead of claiming the tool is absent.

Docker image layers and build-time `RUN` commands still use the daemon's
`DockerRootDir`; changing the campaign directory cannot relocate them. Doctor
checks that filesystem separately using the existing 10/20-GiB space thresholds,
and invokes the terminal tools rather than only looking up their executables.
Moving the daemon storage requires host administration and coordination with
other containers. Bind-backed development storage does not restart that daemon.

```bash
python -m scripts.run_benchmark_release \
  --manifest configs/benchmarks/0.0.1.json \
  --config config/evaluate.g52-release.local.yaml \
  --output gushengda/experiments/release/0.0.1-20261028 \
  --phase build --workers 4

python -m scripts.run_benchmark_release \
  --manifest configs/benchmarks/0.0.1.json \
  --config config/evaluate.g52-release.local.yaml \
  --output gushengda/experiments/release/0.0.1-20261028 \
  --phase baseline --workers 4 \
  --cpu-pools 0-15 --cpu-pools 16-31 \
  --cpu-pools 32-47 --cpu-pools 48-63

python -m scripts.run_benchmark_release \
  --manifest configs/benchmarks/0.0.1.json \
  --config config/evaluate.g52-release.local.yaml \
  --output gushengda/experiments/release/0.0.1-20261028 \
  --phase models --workers 4 \
  --cpu-pools 0-15 --cpu-pools 16-31 \
  --cpu-pools 32-47 --cpu-pools 48-63 \
  --models configs/experiments/gpt_6_1_matrix.yaml
```

Choose CPU pools from the actual physical-core topology. Each model attempt uses
the same reserved pool for its Agent and final Judge. Build work finishes
before baseline measurements start. A baseline runs the full declared
instance/seed/budget grid using an empty source patch, independent solution
verification, and the existing evaluator and reports. Model execution requires
all sixteen baseline runs to pass, with unchanged task configurations. The model
matrix runs one attempt per task at `gpt-6.1-sol` with `xhigh` and one with `high`.
Both settings share one queue, so free slots admit the next independent trial.
It preserves the configured research budget. Parameter robustness defaults to
`configuration_robustness: {enabled: true, scope: candidate}`. Both the regular
launcher and the deferred measurement queue run it after native source evaluation;
the study launcher no longer adds an implicit opt-out. An explicit
`--no-configuration-robustness` or local `enabled: false` remains available.
This single-attempt run does not
provide the separately recommended three-attempt comparison.

Candidate parameter evaluation defaults to the eight-repository panel in
`configs/research/configuration_robustness_native_v2.yaml`. The original five
parameter domains and fidelity declarations are retained unchanged. New native-only
panels cover CBC's presolve/cut/heuristic switches, SCIP's official default/off/fast/
aggressive presets, and Gecode's existing propagation levels. CBC and SCIP now
accept parameter files through the same native driver interface; budget and seed
controls are applied after these settings, and rejected options remain failures.
SCIP presets use its `emphasis` submenu; unavailable commands are errors even when
SCIP exits with code zero. The native CBC build is serial, so its driver does not
send the unsupported `-threads` option and continues to require one solver thread.
The approved release-only five-repository experiment remains unchanged in
`configs/experiments/configuration_robustness.yaml`.

Parameter evaluation freezes the
declared number of inputs from the current task's primary hidden population and
its native hidden seeds, independently of the historical collector input files.
This is a new, explicitly recorded native-source protocol, not a reinterpretation
of the old parameter-search results. Both sources receive their own bounded
search, and the union of their selected configurations plus defaults is compared
on the same independent holdout seeds. No parameter results are sent to the Agent.
The three new panels keep their task's frozen native seed list, including existing
CBC/SCIP study views with 30 seeds. For a smaller native list, fidelity stages stop
at that list instead of inventing additional seeds. The Gecode native adapter has
fixed branch selectors and no stochastic seed input; its existing seed labels are
retained, and independent repeated measurements remain separate from search data.

The native Judge applies the actual captured patch and uses its existing
performance build cache and independent verifier. Parameters do not change the
build recipe, so each source builds once. The HiGHS driver directs its native
log to each solve's `.solver.log` output; its default `Highs.log` would otherwise
mutate the cached build tree and force a rebuild at every parameter panel.
Source/patch, complete input bytes,
parameters, budget, threads, pinned image, harness code, CPU quota, worker count,
memory, host boot and SMT profile identify measurement banks. Compatible Base
measurements are shared across candidates; nested stages and resumed runs execute
only missing cells. Raw observations have immutable snapshots with hashes and
reuse receipts. Solver failures are retained; retrying infrastructure errors is
explicit. Search and independent-retest banks are separate, including for a
deterministic solver whose protocol uses the same fixed seed in both phases.
A repository without an approved parameter panel or native parameter
driver is reported as unavailable, rather than passed or silently disabled.

The standalone release-only collector remains available through `scope: release`,
the original experiment config and `scripts.collect_configuration_results`. It now reuses one container per
image/environment/CPU, while starting a fresh process for every solver run.
Temporary paths follow the data-disk output directory. Candidate evaluation
launches one native Judge per new panel, rather than one Docker container per
instance and seed. These changes preserve the full default fidelity schedule,
search count, seed counts and independent retest. Explicit smoke limits reduce
only the validation run and are recorded in its manifest.

Previously captured patches can receive parameter-only supplementation without
another Agent or a repeat of the native solve grid. Use a new output directory
to retain the old campaign unchanged:

```bash
pitbench judge TASK_ID /data3/path/to/candidate.patch \
  --expected-patch-sha256 RECORDED_SHA256 \
  --config /data3/path/to/evaluate.local.yaml \
  --output-dir /data3/path/to/new-parameter-evaluation \
  --configuration-only
```

For a complete retained cohort, `scripts.run_benchmark_release --phase
study-parameters --source-cohort ORIGINAL_COHORT --output NEW_OUTPUT` uses the
same per-candidate evaluator with disjoint `--evaluation-cpu-pools`. It records
every source condition, including missing candidates, and never starts an Agent
or rewrites the original cohort.

Per-task status files and command logs remain under the output directory.
Completed stages are reused. Failed/interrupted stages require `--retry-errors`;
prior statuses and appended logs are retained. For a changed experimental design,
use a new output directory.

Generalization studies can set `evaluation_run_target: 100000` in the study plan.
The launcher first selects supplemental panel instances within this solve-count
ceiling, counting Base, Agent and configured representation transformations.
Selection follows the frozen input inventory order and retains every condition.
It then applies `supplemental_seed_count: 5` and
`supplemental_primary_budget_only: true`, as enabled in the latest-release plan.
Each supplemental panel uses the first five frozen hidden seeds and its calibrated
primary solve budget. Selected instances are not expanded after this reduction.
Original primary panels, development seeds, representation transformations and API
scenario schedules retain their declared settings. The ceiling excludes API
scenario calls and build/verification costs; it is not a wall-time estimate.
The evaluator runs each declared grid and scenario campaign once, without an
automatic confirmation repeat. Use a new output directory for the changed design.

`configs/research/latest_high_replicates_v3.json` selects `model_repeats`: one
standard condition with the full visible development population, no ABC intervention,
and three fresh independent starts per task/model. The existing high-effort matrix
contains Sol and Terra, giving six starts per solver. Each repeat gets its own
workspace, run ID and usage record; no best-of-three selection is applied.

`configs/research/latest_high_algorithm_space_v4.json` extends this to all eight
approved latest releases. Each task now explicitly names its target problem;
the unchanged native `judge_id` performance contract defines success for that
target, while cross-problem performance is reported separately. Current targets
are CVRP (PyVRP/VROOM), single-machine scheduling (HiGHS), bin packing
(Choco/Gecode), job-shop scheduling (OR-Tools CP-SAT), and multidimensional
knapsack (SCIP/CBC). This does not rewrite older experiments.

The `agent_augmentation` comparison has two conditions with identical public
task instructions and data. The ordinary condition uses Codex; the enhanced
condition uses the external package through the existing `--agent-import-path`
interface. Algorithm-Space planning and implementation stay in the same coding
session and four-hour, 8-vCPU, 32-GiB budget. Each model gets three independent
starts per condition: eight tasks × two models × two conditions × three attempts
is 96 planned starts. There is no extra planner-model allowance or additional
PitBench hypothesis-registration intervention.

The supplied `algorithm-space-agent` 0.1.0 source has no Git commit, so its source
file hashes and wheel hash are frozen in the condition's provenance. The adapter
remains an external distribution, rather than a solver-specific implementation
inside PitBench. Each condition receives the same model endpoint and runtime
options. Task-specific seed policies retain the agreed CBC/SCIP hidden lists;
other tasks keep their native seed protocols, including deterministic VROOM.
API scenario dimensions without implemented source-bound adapters remain
explicitly unavailable; a passing native test does not claim those APIs passed.

The study launcher enables campaign-local Base and build caches under
`<output>/shared-cache/`. A single Base-only producer measures each complete grid
(including representation transformations); candidates then run concurrently and
reuse its raw observations and artifacts. Cache identities bind the frozen task, actual planned instance bytes and reference
values, image/harness, hardware/SMT profile, thread count, CPU quota, memory and worker
count. API scenarios also retain shared Base provenance. Shared measurements are
correlated evidence across conditions, not additional independent repetitions.
The HTML/JSON report identifies the shared Base source.

Build caches retain successful validation, performance and API workspaces at their
original absolute paths. Source/patch, recipe, toolchain/image and environment
identities select entries; changed products cause a new build without relocating
or deleting the prior workspace. Build failures and build-cache receipts remain
in the retained logs. These caches do not reuse candidate solve observations.

The Judge records its launch identity before starting. Re-running an interrupted
candidate automatically resumes only missing instance/state/seed/budget cells,
including representation runs. Existing solver failures remain terminal records;
identity mismatches preserve the old output and require a separate directory.
Setup metrics reuse a compatible cold scenario measurement when input hashes,
seed, threads, budget, runtime and execution options match. Raw source observations
and explicit reuse references remain available.

Default Judge concurrency uses all slots in the declared CPU allocation; explicit
`judge_parallel_runs` still takes precedence within that allocation. Long-budget
jobs run first with adjacent Base/Agent pairs. `judge-scheduling.json` records the
planned execution order, actual worker count and observed solver-stage throughput,
so worker-count comparisons can use the same frozen grid. No new solver budget,
seed selection or scoring threshold is inferred from scheduling.

To prepare reduced evaluation views for an existing frozen study:

```bash
python -m scripts.prepare_research_cases balance-views \
  --views /path/to/cohort/frozen-seed-views/task-views.json \
  --private-root /path/to/private \
  --output-dir /path/to/new-evaluation-views --target-runs 100000 \
  --supplemental-seed-count 5 --supplemental-primary-budget-only
```

PyVRP drivers use the direct `pyvrp.solve(ProblemData, ...)` API when available.
This avoids creating a complete Python edge graph and converting it back to
distance matrices before each solve. Older APIs retain their existing adapter.

These tasks run their declared metrics. Extended stateful/API scenarios still
require explicit per-task protocols and source-built runtime capabilities as
described in [metrics/scenarios.md](metrics/scenarios.md). Passing these baselines
does not establish that every extended metric is supported on all sixteen releases.

Read all retained measurements through the same report command:

```bash
pitbench report /path/to/trial/evaluation \
  --task-config /path/to/frozen-task.yaml --json --all-metrics
```

The JSON includes the frozen primary performance result, supplementary native
estimates for each hidden instance set, declared controlled retention, seed IQR,
equivalent-representation IQR, resource ratios and eligible-pair counts, boundary
reliability, declared API scenarios, and a status/reason for every catalog metric.
Seed reporting preserves the saved seed order when available and reuses retained
exact seed reports. Representation details must match the task and candidate.
Each panel uses its own declared seed and budget grid; panel estimates do not
create a retention protocol. Resource estimates based on partial eligible pairs
remain explicitly partial. The command reads saved evidence; it does not collect
missing scenarios or launch another Agent or solver run.

Configuration search remains a separate campaign with its own search/retest
records. Reference-solver campaigns provide instance anchors; completing them
does not establish that candidate patches were evaluated on those populations.

The public runtime helpers declared by each repository plugin are readable by
`pitbench-agent` and remain evaluator-owned. Prepared image revision 17 includes
this permission repair, covering the JVM launchers and Gecode's wrapper and C++
adapter source. Rebuild source and prepared images after changing these helpers;
existing campaign receipts and performance results describe the old environment.

The CP-SAT adapter records intermediate bounds through its solution callback
and the final response. It avoids the pinned releases' unsafe Java best-bound
`Consumer<Double>` binding. There is no separate bound-only event stream; the
ordinary final coverage/runtime metrics keep their existing definitions.
SWIG `.i` and `.swig` edits qualify as solver source within the declared editable
roots, while generated build paths remain excluded.

OR-Tools source images prefetch and run the upstream Java CP-SAT test
dependencies, then restore `BUILD_TESTING=OFF` for the existing performance
configuration. The generated upstream test POMs remain available for offline
development checks. For example, inside the prepared image:

```bash
mkdir -p /tmp/ortools-maven
cp -a /root/.m2/repository/. /tmp/ortools-maven/
MAVEN_OPTS=-Dmaven.repo.local=/tmp/ortools-maven \
  mvn --offline --batch-mode \
  --file build/java/sat/CpSolverTest/pom.xml test
```

The same command applies to `CpModelTest` and `LinearExprTest`. Maven's entrypoint
uses the writable `MAVEN_CONFIG=/tmp/ortools-maven` directory. Test versions and
dependencies come from the pinned upstream POMs. The adapter replaces the
generated test POMs' open OR-Tools dependency range with the exact version of
the locally built Java package, so testing an older checkout cannot select a
newer published release. This changes generated build files only.
For upstream test sources using JUnit 4, the generated POM adds the Vintage
engine at the already declared JUnit platform version. Surefire is configured
to fail when it discovers no tests.

## Automatic reports and confidence-interval figures

Install the host-side reporting extra with `uv sync --extra reports`.
The `models` and `all` phases automatically write `<output>/report/` after the
model queue finishes, including failed conditions. A campaign with model errors
still exits unsuccessfully after generating its report.

The report contains offline HTML, `report.json`, `conditions.csv`, `effects.csv`,
and separate SVG, PNG and PDF figures. Native gap, verified coverage and PAR-2
effects use their saved 95% intervals. Native seed IQR uses its saved 99%
intervals; transfer benefits retain their original normalization and intervals.
Units, panels and confidence levels are never mixed into one axis or score.
Exact primary figures follow coverage-first and runtime-only-on-ties. A hollow
point denotes a paired evaluation estimate; filled points mark retained historical
opposite-order repeats. Current evaluation does not require a repeat. Missing intervals
do not become zero-width error bars. Resource ratios without saved intervals
remain descriptive values with their paired-run counts.

Regenerate a report without executing Agents or solvers:

```bash
python -m scripts.run_benchmark_release \
  --manifest configs/benchmarks/0.0.1.json \
  --config /path/to/evaluate.local.yaml \
  --output /path/to/campaign --phase report

pitbench report-campaign /path/to/campaign --root /path/to/PitBench
```

`report-campaign` also accepts `--output`, `--runs-root` for custom run storage,
and `--index` for a saved JSON condition list or `{conditions: [...]}` audit.
It reads the normal run hierarchy or retained per-condition snapshots. All
indexed failures and missing results remain visible. Frozen task configurations
and available raw observations provide the catalog coverage inventory.
Supplemental hidden-shift gap plots reuse the existing native estimator only
when the retained panel grid is complete and valid. They are labeled separately
and do not create a retention protocol or change the saved primary result.
When a complete heuristic shift panel has multiple recorded problem sizes, the
report also plots its observed size strata, with Base/Agent gap and paired gap
reduction at each declared budget. These are exploratory slices within that
panel's recipe, with independent-instance counts shown. They do not rank
different shift types by severity, replace a dedicated scale/structure matrix,
or qualify a scalability score. Single-instance strata show means without an
across-instance interval. Native primary estimates remain unchanged.

## Current g52 scheduling

The ongoing full baseline keeps its original eight-worker allocation. The next
model stage uses twelve independent eight-core pools across four CPU packages,
with 32 GiB and 14,400 seconds per Agent. This reserves sixteen physical cores
and about 119 GiB of the 503 GiB host memory. Fourteen workers remain an upper
capacity estimate; they are not the active setting.

The machine-local `resource-policy.json` under the campaign output delays new
commands below 64 GiB available memory, at CPU temperatures of 86 C or more, or
when one-minute host load reaches 112. The temperature admission limit leaves
5 C below the hardware-reported 91 C maximum. These are dispatch controls; they
do not change solver budgets, running trials, or metric qualification rules.
`host-health.jsonl` records memory, swap, load and CPU temperatures every fifteen
seconds. The final CPU pools and model config are recorded in `parallelism.json`.
