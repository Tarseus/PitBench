# Project maintenance and ownership

PitBench owns the benchmark interfaces. Solver and problem differences are
selected through task configuration, repository plugins and problem-family
plugins. Experiments keep their configuration and evidence outside Git.

Grouped command entry points are documented in [scripts](../scripts/README.md).
Functional test suites and shared sample builders are documented in
[tests](../tests/README.md). Individual collection/report/preparation wrappers
have been removed; maintained callers use the grouped entry points.

## Maintained interfaces

- `pitbench.evaluator.collection` owns collected execution, process-group
  watchdogs, reference collection and release-control calibration.
  `execute_collection_command` is the common command/log/timeout interface.
- `pitbench.evaluator.remote` owns SSH deployment, collection, progress and
  targeted reference retries. Credentials use the configured private credential
  file and existing SSH interface; campaign scripts do not parse passwords.
- `pitbench.metrics.performance_report` owns native performance and declared
  control/noise statistics. Control bootstrap seed 20261017 and 10,000 draws
  remain separate parameters of the shared bootstrap implementation.
- `pitbench.metrics.campaign_report` owns complete condition selection and
  read-only saved-run diagnostics, including failed and missing conditions.
- `pitbench.harness.utils.agent_trace.read_trace` is the shared event/payload
  reader. `artifact_storage.create_verified_archive` is the shared tar/zstd
  writer/comparer for compaction and non-destructive exports.
- `pitbench.repositories.pr_curation` owns discovery, cached review operations,
  patch coding and evidence construction. CLI files parse arguments and dispatch
  to these interfaces.
- `scripts.run_benchmark_release` owns campaign scheduling. Its configuration
  controls resources and the study plan controls solver/problem/model/method
  selection. Temporary launchers are source records, not another implementation.

## Records and reference operations

```bash
python -m scripts.compact_records diagnose --project PROJECT --campaign NAME \
  --runs-directory RUNS --campaign-directory CAMPAIGN --output OUTPUT
python -m scripts.compact_records export --project PROJECT --campaign NAME \
  --runs-directory RUNS --campaign-directory CAMPAIGN --output OUTPUT
python -m scripts.collect reference --campaign-config CONFIG \
  --output-dir CAMPAIGN --job-id JOB --retry-status process_error \
  --require-queue-drained
python -m scripts.collect reference --remote-config CONFIG \
  --output-dir DISTRIBUTION --progress
```

An export selects all recorded/planned conditions and every retained run
directory, including terminal failures and earlier infrastructure attempts.
Missing run IDs remain in its manifest. Source records are not deleted.
Transient workspaces/build caches, Docker images and credentials are excluded.
Use a separate output directory; active changing files cause verification to
fail rather than producing a falsely complete archive. `archive` compacts and
removes a verified loose directory; `export` is non-destructive.

Targeted retries select jobs from the complete prepared manifest. They do not
rewrite it or change its instance, seed, solve time or thread allocation. The
controller lock and live-worker recovery checks replace task-specific queue
proof files and manifest narrowing. Failed attempts remain available. Remote
retry dispatch requires deployment of the maintained collector version; progress
uses the same read-only collector through the configured SSH connection.

Install the existing optional SSH dependency with `uv sync --extra remote`.
Record compaction/export requires the system `tar` and `zstd` commands.

## Independent Agent package

`algorithm-space-agent` is an independent project, with canonical source in the
separate `../algorithm-space-agent` repository for this workspace. PitBench does
not import its experiment snapshots. The current implementation is 0.2.0; local
installation was aligned to that version during this cleanup. Source and tests
are registered in that repository's Git index; no initial commit or published
remote URL is invented by this maintenance change.

Install the canonical source or a frozen wheel after preparing PitBench:

```bash
uv sync --extra remote
uv pip install --python .venv/bin/python /path/to/algorithm-space-agent
uv run python -c 'import importlib.metadata as m; print(m.version("algorithm-space-agent"))'
```

Run the benchmark through its prepared environment. A later exact `uv sync`
without this external plugin removes packages outside the main lock, so install
the selected Agent again afterward, or use `uv sync --inexact`. The ordinary
PitBench checkout remains installable without the Agent source directory.

Study arms use the existing `agent_import_path` and `agent_provenance` fields.
Freeze distribution version, source content identity and wheel identity for each
new method version; retain old wheels/source snapshots for old cohorts. A source
tree without a commit is identified as such, not assigned a fictitious revision.
The `pitbench-curation` project keeps its own repository and maintenance boundary.

## Local material and historical sources

Keep bulk inputs, generated task packages, native run records and original
failure/missing evidence in ignored data/experiment directories. Historical
source snapshots are evidence and are not imported as current implementations.
Superseded operational scripts live under `source-records/`; their old budgets
and fixed paths are not defaults for new campaigns. Machine credentials and
editor settings remain local. Tracked conversation exports are removed from the
index while the local files remain available. Bytecode and test caches are
generated material. `upstream/`, solver working copies and Dockerfiles are not
deleted as part of organizational cleanup.

Frozen driver overlays and retry sources embedded in replay/staging records may
be byte-identical across datasets. They remain build/replay inputs, not another
maintained implementation. Do not replace them with links to mutable current
source or delete an entire dataset on the strength of matching script files.
