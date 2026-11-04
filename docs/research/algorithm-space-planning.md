# Algorithm-Space Global Planning

The harness helps the existing coding Agent choose changes to algorithm mechanisms.
Its model is a partial hierarchical transition system, with Level 0 goals, Level 1
subsystems, Level 2 mechanisms, and Level 3 source references. State variables S,
actions U, transitions T, resource costs C, progress signals R, and correctness
invariants I are explicit. Hidden state, stochastic behavior, and unknown relations
are allowed. Model claims are hypotheses or evidence-backed observations, not proofs.

Enable `plan` alongside `bench` in the normal evaluation configuration:

```yaml
agent_tools: [inspect, workspace, bench, plan]
```

PitBench bundles the model tool and adds its instructions to the existing Agent
session. Model construction, planning, implementation, and model revision use that
Agent and the normal development budget. Models come from the Agent's source and
public-trace inspection; the harness does not infer semantics from filenames or
supply canned optimization opportunities.

## The development loop

1. Run the public baseline with `pitbench bench --label base`.
2. Inspect `pitbench plan --schema`. Keep planning input JSON files in the public
   runs directory, outside implementation source. Register source/trace observations with
   `pitbench plan --evidence evidence.json`, then submit a partial model with
   `pitbench plan --model model.json`.
3. Submit competing mechanisms with `pitbench plan --proposal proposal.json`.
   Each proposal identifies existing transitions and their replacements, any state
   or action changes, expected global effects, required invariants, concrete source
   changes, and a controlled experiment with observations that would falsify it.
4. Compare the known proposals with `pitbench plan --decision decision.json`.
   A decision selects one mechanism or a compatible combination, explains deferred
   alternatives, and addresses pending experimental evidence. An empty selection
   records a reasoned stop. Changing multiple modules is not a success criterion.
5. Implement the mechanism, then submit
   `pitbench plan --implementation implementation.json`. Reference the recorded
   `decision_index`, actual source version, and proposal IDs; declare each proposal
   as `implemented`, `partial`, or `abandoned`. Give source-file references for the
   implementation and explain deviations from the planned model with their own
   source references. A deviation may explicitly identify a previously deferred
   proposal that was implemented instead. The tool checks references and versions;
   it does not prove semantic equivalence between source and model.
6. Run the ordinary `pitbench bench --label agent --build --implementation-id ID`
   with matched instances, seeds, and budgets. Editing source again requires a new
   implementation record bound to that patch. Baseline runs need no such record.
7. Read `pitbench plan`. Benchmark evidence identifies affected implementation and
   mechanism nodes and their subsystem/global ancestors. Update model claims from
   those observations, compare the remaining opportunities, and decide the next
   experiment. A new source version requires model review before the next decision.

The model, proposed changes, decisions, and implementation records are structured JSON; `--schema` supplies
the exact contracts. A transition declares its action, state reads/writes, relation,
resource costs, progress, invariant references, and Level 3 implementation nodes.
For example, a local-to-compound transition proposal may introduce a hidden search
state variable, replace a single-variable action, retain feasibility obligations,
and compare the added per-action work with the resulting global search reduction.
The problem-specific feasibility and cost semantics still have to be instantiated.

## Evidence and source versions

`algorithm_plan.json` in the existing public runs directory retains model revisions,
all proposals, decisions, implementation records, source/trace evidence, and raw per-run benchmark
feedback. Source identity uses repository HEAD and the same candidate patch hash as
bench, including untracked source. Each experiment retains its instances, seeds,
budgets, threads, verification status, failures, missing results, and artifact paths.
Generated tasks set `PITBENCH_RUNS=/agent-logs/development-runs`, under the retained
Agent log volume, so development evidence survives task-container cleanup.

Each decision retains its planned candidate model A'. Experiments reference the
implementation record and its actual patch, rather than assuming selected proposals
were fully implemented. Direct scope includes original and replacement transition
owners, implementation nodes, declared deviation files, and their ancestors.
`potential_affected_nodes` separately lists possible influence inferred through
model dependencies; it does not establish a causal observation.

An attempt enters the evidence history and pending-review list before the build
starts. Build failure, compiler absence, or timeout records `failure_stage: build`,
the implementation binding, and logs in a unique attempt directory. A successful
build advances the same record to solver measurement. Failed attempts remain pending
until reviewed; an attempt still running cannot be marked reviewed.

Data/control dependencies and performance influence have separate edge types.
An observed performance influence requires complete, independently valid experiment
evidence. A speedup does not automatically validate a causal mechanism: the Agent
must revise the relevant claims and cite its evidence. Observations from another
source version cannot silently become current facts. The harness consumes only
public development data and reuses PitBench's execution and verification paths.
It introduces no performance thresholds, averaging rule, or separate scoring metric.

## Initial comparison on SCIP 10.1.0

Use three representation conditions: original source and tools; source plus an
Agent-built call/dependence graph; and the partial algorithm model with dynamic
mechanism planning. The graph condition describes software structure; the algorithm
condition additionally records state transitions, costs, progress, and invariants.
Keep the foundation model, development information, total budget, solver seeds,
transformations, and final scoring protocol fixed, with independent Agent repeats.
Study whether the representation changes opportunity comparisons and experiments;
module count is not an optimization-quality metric.
