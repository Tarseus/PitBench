# Algorithm-space global planning

The `global-plan` controller keeps a versioned HATS model, opportunities, decisions,
experiment evidence, budget clock, and append-only event history in one state
directory. The planner model and coding agent are configured separately. Codex,
Claude Code, and Antigravity CLI adapters are available for local runs.

```bash
global-plan run \
  --agent codex \
  --task task.md \
  --task-config configs/tasks/release_scip_v10_1_0.yaml \
  --repository /path/to/solver \
  --dir runs/scip/global-plan \
  --total-budget-sec 14400
```

Each Agent turn receives the current decision and compact context. It must run the
visible development benchmark with the supplied `--global-plan-dir` and experiment
ID. `pitbench bench` records execution, independent validity, performance change,
patch identity, seeds, budgets, run count, and feedback paths. The outer controller
then reloads that persisted boundary, updates the model, and decides whether to
continue, switch, combine, or stop. If no benchmark boundary was recorded, the
turn is stored as missing evidence. The controller resumes from the same directory
after process restart; planning and Agent time share the persistent wall-clock
budget.

Persistence here covers the planning model, decisions, evidence, and budget. Each
coding turn starts a new native Agent session and injects the saved planning context.
Codex uses `exec --ephemeral`; native Agent conversations are not restored.

A benchmark saved before an Agent timeout remains valid evidence. The timeout is
recorded separately as `agent.turn_failed`. Incomplete run grids are stored as
`missing / unverified / unknown`. Previously recorded partial evidence is corrected
with an `evidence.corrected` event containing the original and corrected records;
the experiment count does not increase.

The latest invocation exposes `session_status` in state, `global-plan status`, and
the `session.finished` event. `completed` (exit 0) means all attempted turns have
complete, verified execution evidence, or the planner explicitly stopped.
`incomplete` (exit 2) means measurements or verification are missing, or the budget
prevented a scheduled turn. `failed` (exit 1) means an Agent/controller error,
execution failure, or invalid solution occurred, including an Agent timeout after
a successful benchmark. Performance regression alone does not fail the session.
These statuses describe execution of this invocation, not optimization benefit.

The coding Agent needs the normal PitBench task environment (`PITBENCH_AGENT_CONFIG`,
development instances, solver runner, and writable results directory) available in
its process environment. The task YAML supplies model bootstrapping metadata; `--task`
supplies the coding objective. Hidden Judge data is not passed to the planner.

Adapters can write and consume hook events without implementing planning logic:

```bash
global-plan record-event --dir runs/scip/global-plan \
  --event-type hook.session-restored --payload '{"client":"codex"}'
global-plan events --dir runs/scip/global-plan --since 10
```

The JSON state and revision snapshots are stored in `plan_state.json` and `history/`.
Only the controller decides replanning boundaries; a hook may report task, resume,
or completion events but should not independently invoke another planning loop.
