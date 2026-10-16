# Task expansion: additional problem types

The active benchmark uses one pinned release per solver. The two added tasks
reuse the existing HiGHS and Choco repositories, build plugins, isolated judge,
paired Performance report, Resource Efficiency report, and development tools.
They add independent optimization objectives and instance panels rather than
counting more seeds or older releases as new tasks.

Development instances, reference solutions, and generation settings are in the
tracked `configs/instance_sets/` tree. Hidden seeds, instances, and oracle
records remain in the evaluator's ignored `private/` bundle, with their hashes
declared in the task configurations. The standard Choco Dockerfile was rebuilt
from its pinned Git commit and submodule; no local solver checkout is needed to
build the task image.

## HiGHS: multidimensional 0–1 knapsack

`highs_multidimensional_knapsack_v1_15_1` uses 5 visible and 30 hidden LP
instances, one worker per process, solver seeds 0–4, and 5- and 20-second
budgets. For each item, its profit is the sum of its weights across all
dimensions. Every feasible assignment therefore has profit at most the sum of
the dimension capacities. The generator plants an assignment that fills every
capacity exactly. This matching primal solution and dual bound prove the saved
optimum without relying on HiGHS's own optimal-status claim.

The independent verifier parses only the canonical generated LP, checks every
binary value and capacity, and recomputes the objective from the assignment.
Visible development feedback runs the same check for hash-bound public inputs.
The hidden oracle records the input hash, bound construction, and reference
assignment. The local HiGHS 1.15.1 seed-0 screening selected a panel with
partial optimal coverage: 1/5 and 3/5 visible instances at 5 and 20 seconds,
and 21/30 hidden instances at 20 seconds. These are screening results, not a
full multi-seed Base/Agent evaluation.

## Choco: graph coloring

`choco_graph_coloring_v6_0_1` uses 5 visible and 30 hidden graphs, one worker
and one solver seed, with 5- and 20-second budgets. Its base graph is the
Mycielski construction applied to `K2` until it has chromatic number six.
Vertex relabeling and extra edges between differently colored vertices vary
the inputs. The verifier reconstructs the six-chromatic subgraph and checks a
valid six-color reference assignment, establishing matching lower and upper
bounds. The lower bound follows inductively: if the Mycielski graph had a
coloring with no more colors than its parent, recoloring each original vertex
that uses the new root's color with its shadow vertex's color would give a
coloring of the parent with one fewer color. It separately verifies every
candidate's edge constraints and color count. The Choco adapter selects graph
coloring from the instance format and continues to support bin packing.

Initial planted-clique graphs all solved too quickly. The first Mycielski
panel had 0/30 verified optimal runs at 5 seconds, so it was replaced with a
mixed-density panel. Final single-seed hidden screening found 6/30 optimal
runs at 5 seconds and 14/30 at 20 seconds. The raw screening records are kept
in `private/performance_validation/choco_graph_coloring_screening/`. The full
development panel solved 3/5 at each budget under seed 0. The full Base/Agent
evaluation remains separate from these selection results.

Both tasks keep raw failures and missing runs. The independent check verifies
solution feasibility and objective; the stated mathematical constructions
support the trusted optimum values. A solver's claim of optimal termination is
recorded separately from those checks.

## End-to-end validation

`pitbench tasks validate` accepted both new task configs and their visible
assets. `pitbench tasks materialize-dev` exposed five inputs and targets for
each task. The HiGHS empty-patch dry run at
`runs/task-expansion-20261014/dry-run/highs_multidimensional_knapsack_v1_15_1/`
completed all 20 planned hidden observations for one instance, five seeds, two
budgets, and Base/Agent. Six were valid optimal solves at 20 seconds; the other
14 were ordinary solver-origin no-solution outcomes. Its Performance report was
complete, with no infrastructure or semantic-invalid observations. This is an
integration check, not the complete 30-instance Base/Agent panel.

The Choco graph-coloring empty-patch dry run in the same `run-id` also passed
its limited-panel judge: all four observations carried independently valid
colorings, with ordinary time-limit termination on the selected hard instance
at both budgets. Its Performance report was complete. The combined command
then reached the separately configured Configuration Robustness collector,
where its old Choco image
failed the adapter-source fingerprint check. The original Choco task and that
collector now use the standard Git-built image with the new shared adapter.
The collector's `prepare` command subsequently accepted all 11 configured
searches, and the original bin-packing runner solved a development instance
on the updated image. The complete Configuration Robustness search and the
full new-task Base/Agent panels were not run by these checks.

A second combined dry run with the independent configuration search disabled
completed both new tasks and exited successfully at
`runs/task-expansion-20261014-clean/dry-run/`. Prepared agent images with
`inspect`, `workspace`, and `bench` exposed all five public instances and
targets for each task. Real `pitbench bench` calls on HiGHS `dev_0000` and
Choco `dev_0004` returned independently verified optimal feedback. The first
HiGHS agent-tooling probe exposed a missing public-verifier dependency; the
canonical LP formatter was moved into the already bundled public verifier,
and both in-container bench calls passed after rebuilding the task images.
