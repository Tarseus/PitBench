# Private evaluator storage

This directory is never mounted into an agent container. Local installations place
hidden populations, independent verifiers, and oracle data here using
the paths referenced by `private://` task URIs.

Everything below this README is gitignored. A production judge must verify hashes and
fail closed when an asset is missing; fixture mode is explicit and cannot substitute
for a real evaluation result.

The PyVRP six-dimensional protocol additionally expects
`populations/pyvrp_cvrp_shift_v1.yaml`. It is a hash-pinned, judge-only deterministic
generator manifest; generated instances never enter the agent container. Its chained
`oracles/pyvrp_cvrp_shift_v1/oracle.yaml` contains independently verified empirical
BKS anchors and solution hashes. Refreshing these anchors requires a new benchmark
oracle version rather than silently replacing the frozen files.
