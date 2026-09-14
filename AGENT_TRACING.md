# Agent execution traces

Every agent dispatched by `pitbench run` goes through
`BaseAgent.execute_task`, called explicitly by the harness. This includes all
registered agents, `--agent-import-path` agents, and each agent in multi-agent
trials. Logging is enabled automatically. Adapters implement `perform_task`;
they do not select a trace format or open the unified event file.

Each invocation writes to:

```text
<trial>/agent-trace/<execution_id>/events.jsonl
<trial>/agent-trace/<execution_id>/objects/<sha256>
<trial>/agent-trace/<execution_id>/coverage.json
```

The canonical index and objects stay on the host. Only the execution's dedicated
`native-inbox` subdirectory is mounted into isolated CLI runners, for hook output.
An execution gets a fresh ID, so retries of the harness entry do not
overwrite earlier evidence. Resuming an interrupted task archives its previous
directory under `<run>/interrupted/<archive_id>/<task_id>/` before rerunning it.
Existing provider logs and terminal recordings remain available.

## Recorded evidence

- `execution.*`, `environment`, `instruction.rendered`, `context.profile`:
  original and rendered instructions, configured agent/model identity and
  configuration, validated profile files, container image and resource settings,
  result, exceptions, and harness timeouts. A configured version such as `latest`
  is not proof of the installed binary version; provider artifacts may supply
  that information.
- `process.*`, `agent.event`: CLI stdout/stderr as they arrive, each subprocess
  invocation and exit, and JSON events exposed by the CLI. Original provider
  event IDs, sub-agent relationships, usage and compaction events remain in the
  `native` object when the provider emits them. Container-based output is also
  forwarded during execution, with combined stdout/stderr labeled explicitly.
- `tool.*`, `terminal.*`: arguments, returned feedback, failures and call timing
  at the shared MCP and tmux boundaries. Patch arguments are preserved. Raw MCP
  command output is stored before truncation, separately from returned output.
  Async command chunks retain the originating call ID and job handle, including
  output the agent never polled. Terminal keystroke completion alone is not a
  shell exit status. `terminal.command_result` separately records the actual
  status of a blocking tmux shell command, or explicitly marks it unavailable.
- `native.hook`, `agent.child`: native pre/post-tool and supported child-agent
  events. Recording hooks are installed into the disposable Codex and
  Antigravity runtimes; the installed-agent entry also supports Claude Code,
  Gemini CLI and Cursor hook configuration. Existing user hooks are preserved.
  Antigravity's passive hook response is empty stdout; returning `{}` would
  deny its tool calls. Other supported hook protocols use an empty JSON object.
- `mcp.*`: requests/results at the MCP request handler, including requests
  rejected before reaching a tool. MCP `_meta.pitbench` carries backend call and
  state IDs without changing the tool's text or structured content. Native hooks
  can link those IDs to the provider call. A unique active pre-hook with exactly
  matching MCP arguments can also be bound at request receipt. Ambiguous matches
  are recorded without assigning a backend. Antigravity tool results are recovered
  by its authoritative step index from later transcript snapshots; an empty hook
  `error` field is not treated as a result.
- `model.*`: actual model requests and responses at the shared completion
  boundary. The workspace model relay additionally distinguishes received
  requests, transformed requests, upstream response bytes, and bytes written to
  the agent connection. These records do not claim access to hidden reasoning.
- `source.*`: provider files and session recordings collected every 250 ms.
  File generations distinguish appends from observed replacements and rewrites.
  Remote Docker log directories are preserved as tar archives during execution
  and before the agent returns, since their normal host copy happens later.
- `repository.checkpoint`: source snapshots at execution start/end, instrumented
  tool boundaries, and approximately once a second during execution. Each points
  to an archive of the Git HEAD blobs and a binary patch covering tracked and non-ignored
  untracked files, including additions, deletions and reversions. `.pitbench` is
  excluded, consistent with candidate capture. A state ID identifies the archive
  and patch; call IDs associate boundary observations with operations.
  Source archives preserve tracked files even when `.gitattributes` marks them
  `export-ignore` or `export-subst`. Submodules are reported as a collection gap.
- `workspace.snapshot`: full file manifests at instrumented boundaries, including
  ignored build outputs, `.pitbench` data and initialized submodule files. Git
  administration directories are excluded. Files are stored by content hash;
  unchanged inode/size/mtime/ctime signatures reuse prior objects. Concurrently
  modified files are marked unstable. `before_state_id` and `after_state_id` on
  operation completion refer to the captured versions, not whichever snapshot
  happens to be newest when logs are read.
- `tool.target_snapshot`: explicit file targets of native tools, such as a CLI's
  own tool-schema files outside the task repository. These IDs describe those
  targets only, not the entire repository.
- `filesystem.mutation`: Linux inotify notifications for creation, deletion,
  moves, attribute changes and write-close events. Available file versions are
  preserved after write-close notifications. Queue overflow, unavailable versions
  and watcher failures produce collection gaps. The journal can expose a file
  created and deleted entirely between periodic source snapshots.
- `collection.gap`: failures to collect a source. The final execution event
  reports the number of distinct collection gaps. An agent completing its task
  does not imply that every source was observable.

Each index row has a schema version, sequence, UTC timestamp, elapsed time,
execution ID, run/task/trial identity, event kind and optional call ID. The
`data` field references the complete event payload. Payloads, output chunks and
patches use SHA-256-addressed objects; the collector does not truncate them.
Raw subprocess bytes are written without waiting for a newline. Objects and
index entries are flushed and fsynced, including during a failed execution.
An abruptly killed execution can lack its closing event or have a partial final
index line. Native hooks write durable records before returning; their outboxes
and available transcripts are collected before disposable runtimes are removed.
Unacknowledged provider data and storage failure remain outside this guarantee.

## Reading and reconstructing

The shared reader resolves event payloads and verifies their hashes. It ignores
an unfinished final index line and rejects corrupt complete records or missing
objects. Nested references to raw output or archives remain references.

```python
from pathlib import Path
from pitbench.harness.utils.agent_trace import read_trace

path = Path("runs/<run>/<task>/<trial>/agent-trace/<execution_id>/events.jsonl")
for event in read_trace(path):
    print(event["sequence"], event["event"], event["call_id"], event["data"])
```

To reconstruct an observed source state, unpack its `base_archive` into a fresh
directory, initialize Git there, and apply its `patch` with `git apply --binary`.
An empty patch needs no application. Extract archives as untrusted data, without
following external paths or links. Do not apply a reconstruction over a live
working repository.

Join execution evidence to `pipeline_trace.jsonl` using `run_id`, `task_id` and
`trial_name`. The pipeline's completed agent stage links `agent_trace_dir`;
subsequent candidate capture and evaluation events describe the submitted patch
and its results. Evaluation after the agent finishes was not feedback available
to the agent. Instrumented test commands have before/after source checkpoints;
this does not by itself establish which compiled binary a test loaded.
Full workspace manifests preserve the actual ignored binaries present at tool
boundaries. They do not establish every library or memory page a process loaded.

`coverage.json` is generated for every execution. Its
`observed_operations_paired` field checks recorded requests, results, completion
and state references. It does **not** certify that a provider exposed all actions.
Missing provider IDs, ambiguous bindings, unfinished calls and collection gaps
remain visible. The same check is available as
`pitbench.harness.utils.trace_validation.inspect_trace(path)`.

## Real runtime checks

Run a configured agent on an isolated tiny repository using its actual executable
and model. The check requests reading, editing, compilation, execution and one
deliberately failing command, retains its evidence and independently runs the
produced binary. It uses the existing agent factory, execution entry and terminal.

```bash
uv run python scripts/check-agent-tracing.py \
  --agent codex --model MODEL_FROM_YOUR_CONFIGURATION \
  --image YOUR_PREPARED_TASK_IMAGE \
  --output runs/trace-check-unique-name --timeout 180
```

The output directory must be new. `--agent-kwarg key=value` selects existing
runner options for that check; it does not edit machine configuration.
`validation.json` separates agent failure, build verification and trace inspection.
This check makes real model calls. Registry-dispatch tests and hook-schema tests
alone are not live verification of an agent or a model. The installed-agent hook
configurations still require verification against the versions used in an experiment.

## Observation boundaries

The same entry and storage policy apply to every agent. The amount of internal
detail still depends on what that agent exposes. A closed CLI's internal tool
calls, model context or child agents cannot be inferred from terminal output
alone. Raw provider evidence is retained for later parsers rather than replaced
with guessed behavior labels. Custom adapters can publish additional evidence
with `current_trace().record(...)` while running through the harness.

Source and file-manifest walks are non-atomic observations. Native hooks improve
the boundary timing; concurrently running tools are not serialized by the logger.
Inotify can coalesce events and does not cover every mmap or remote filesystem
write; recursively added directories have a watch-registration window. A version
read after a notification is labeled as such, not claimed to be an atomic snapshot
at the instant of that mutation. External services are not snapshotted.
The trace explicitly records its scope and sampling policy. It is a source and
feedback reconstruction record, not deterministic re-execution of a model.
For native tools without an instrumented boundary, a sampled checkpoint must
not be treated as the exact state used by an individual command.

Collection uses the existing execution budget. Source capture duration is
recorded on each checkpoint; collection also adds disk and Docker I/O. Measure
this overhead for the intended repository before comparing timed experiments.
No solver budgets, seeds, transformations or evaluation criteria are changed.

The collector does not store runner credential payloads or model authentication
headers. Raw prompts, source code, provider files and tool output can nevertheless
contain sensitive task data. Keep trace directories private and inspect them
before sharing. Treat an absent usage field or unavailable source as missing
information, not a measured zero.
