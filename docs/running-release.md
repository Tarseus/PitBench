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
It preserves the configured research budget. The separate global configuration
search is not repeated inside each source-editing attempt; its historical
five-solver experiment is outside this sixteen-release native-metric run. This single-attempt run does not
provide the separately recommended three-attempt comparison.

Per-task status files and command logs remain under the output directory.
Completed stages are reused. Failed/interrupted stages require `--retry-errors`;
prior statuses and appended logs are retained. For a changed experimental design,
use a new output directory.

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
Panel estimates do not inherit the primary result's confirmation and do not
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
point does not claim primary opposite-order confirmation. Missing intervals
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
