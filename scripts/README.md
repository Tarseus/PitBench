# Command entry points

Use the grouped module commands from the repository checkout:

- `python -m scripts.prepare {instances,optima,research} ...`
- `python -m scripts.collect {reference,configuration,scenarios,nuisance,anchors,controls,seeds} ...`
- `python -m scripts.report {controls,agent-reference,noise,patches,nuisance,transfer} ...`
- `python -m scripts.check {feedback,tracing} ...`

Each selected command retains its own arguments and `--help`. Backends are loaded
only after selection, so an unselected solver/method does not add dependencies.
Collection workers inside containers and distributed deployments use the same
entry point. Solver, problem-family and transformation choices remain in task
and experiment configuration, not this dispatch table.

Campaign scheduling remains `scripts.run_benchmark_release`, PR workflow remains
`scripts.curate_solver_prs`, and record maintenance remains
`scripts.compact_records`. Their execution, verification and metrics interfaces
are maintained in the corresponding `pitbench` modules. Shell matrix commands
are thin batch coordinators over those interfaces.
Seed-validation batches created by older versions retain their frozen entry
point; the matrix coordinator invokes that snapshot without modifying it.

`install-runner.sh AGENT [SOURCE_BINARY]` shares runner installation logic.
The two `pitbench-*-runner` executables remain separate because they are the
root-owned constrained-sudo protocol interfaces for their respective agents.
This directory is source checkout/deployment tooling; it is not an alternate
solver implementation or a store for experiment-specific Python code.
