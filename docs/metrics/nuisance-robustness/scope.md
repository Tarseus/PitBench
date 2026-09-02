# Nuisance Robustness scope

## Seed Robustness v1

Seed Robustness v1 applies to all stochastic VRP heuristic solvers.

With the instance, code state, budget, thread count, configuration, and runtime
environment fixed, Seed Robustness measures the sensitivity of solution quality to
`solver_seed`.

The evaluation covers Base, Agent, and the change in robustness induced by the patch.
It covers the `agent_dev`, `judge_id`, and `judge_shift` instance sets. Within each
set, Seed Robustness studies solver-seed variation within each fixed instance.

Every evaluation budget produces an equal-status Seed Robustness evaluation, and all
results are retained. For now, only the result at `primary_budget_sec` is reported.

## Exclusions

Seed Robustness v1 excludes:

- instance-generation randomness, including `instance_seed`, `coordinate_seed`, and
  `demand_seed`;
- hardware or platform changes;
- thread-count changes;
- configuration changes;
- deterministic solvers;
- MIP and CP solvers; and
- run failures, which remain owned by Qualification or Operational Reliability.

Performance and Seed Robustness are presented as separate evaluation dimensions. No
combined relationship between them is defined in v1.

This M0 scope does not select IQR, MAD, tail probability, or any other estimator.
