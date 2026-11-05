# Public benchmark result interface

`pitbench bench --result /path/result.json` exports version 1.0 public development
evidence. Without `--result`, it writes a unique record under
`$PITBENCH_RUNS/results/`. This interface is independent of Agent strategy.

The JSON object contains:

- `schema_version: "1.0"`, `kind: "pitbench.development_result"`, unique `id`, `task_id`.
- `source`: Git `base_commit` and `patch_sha256`, including tracked binary changes
  and untracked source files, excluding `.pitbench/` metadata.
- `stage`: `build` or `solve`; `attempt_status`: `running` or `finished`.
- `expected_runs`, raw `observations`, `artifact_paths`, and the requested `workload`
  (instance paths, seeds, budgets, threads, label and optional experiment IDs).
- `execution_status`: `pass`, `fail`, or `missing`; `solution_validity`: `valid`,
  `invalid`, or `unverified`; `failure_stage`, `started_at`, and `finished_at`.

The record is atomically saved before building and updated at execution boundaries.
Build failures retain source identity and logs. Missing feedback remains missing;
only a full set of independently verified valid observations is marked valid.
A caller should inspect raw records and expected counts when consuming evidence.
Nonzero build/solver exits or missing feedback produce a nonzero command exit.
An abruptly killed process may leave a `running` record; it does not imply success.
Pre-execution argument or workspace validation errors return nonzero before a run
record is created.

Each attempt retains separate output files under `$PITBENCH_RUNS/attempts/`.
The top-level `base` and `agent` links point to the latest respective feedback for
existing pairing/diagnosis tools. Generated tasks bind only the public output
subdirectory into `/workspace/repo/.pitbench/runs`, so the native coding sandbox
can write results and the host retains them after container teardown. Agent trace
and credential directories remain outside this writable workspace mount.

External Agent packages use the existing `--agent-import-path` extension. They
consume this result contract and own their strategy-specific state; PitBench
does not impose an algorithm model, proposal schema or planning policy.
