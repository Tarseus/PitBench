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
5. Implement the selected mechanism and run the ordinary
   `pitbench bench --label agent --build` with matched instances, seeds, and budgets.
6. Read `pitbench plan`. Benchmark evidence identifies affected implementation and
   mechanism nodes and their subsystem/global ancestors. Update model claims from
   those observations, compare the remaining opportunities, and decide the next
   experiment. A new source version requires model review before the next decision.

The model, proposed changes, and decisions are structured JSON; `--schema` supplies
the exact contracts. A transition declares its action, state reads/writes, relation,
resource costs, progress, invariant references, and Level 3 implementation nodes.
For example, a local-to-compound transition proposal may introduce a hidden search
state variable, replace a single-variable action, retain feasibility obligations,
and compare the added per-action work with the resulting global search reduction.
The problem-specific feasibility and cost semantics still have to be instantiated.

## Evidence and source versions

`algorithm_plan.json` in the existing public runs directory retains model revisions,
all proposals and decisions, source/trace evidence, and raw per-run benchmark
feedback. Source identity uses repository HEAD and the same candidate patch hash as
bench, including untracked source. Each experiment retains its instances, seeds,
budgets, threads, verification status, failures, missing results, and artifact paths.

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
