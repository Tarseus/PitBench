# Private evaluator storage

This directory is never mounted into an agent container. Local installations place
hidden instance sets, independent verifiers, and oracle data here using
the paths referenced by `private://` task URIs.

Everything below this README is gitignored. A production judge must verify hashes and
fail closed when an asset is missing; fixture mode is explicit and cannot substitute
for a real evaluation result.

The directory also contains maintainer experiment outputs. Do not delete it as a
unit: task configurations currently require assets under `hidden_instances/`,
`oracles/`, `calibration/`, `instance_sets/`, the seed YAML files directly under
`seed_robustness/`, and `validation_tests/`. Follow private URIs transitively when
moving these assets. The evaluator's `private_root` setting can point outside the
repository if local private assets should live elsewhere.

Historical validation subdirectories under `seed_robustness/` and
`performance_validation/` are experiment evidence rather than task inputs. They
can be archived with `python -m scripts.compact_records archive DIRECTORY` and
restored with its `restore` command. Keep the required seed YAML files available
without extraction. Existing `agent-records/*.tar.zst` files are already compressed
evidence exports. See [trace storage](../AGENT_TRACING.md#compacting-completed-records)
for verification and restoration details.

The PyVRP six-dimensional protocol additionally expects
`instance_sets/pyvrp_cvrp_shift_v1.yaml`. It is a hash-pinned, judge-only deterministic
instance-set config; generated instances never enter the agent container. Its chained
`oracles/pyvrp_cvrp_shift_v1/oracle.yaml` contains independently verified empirical
BKS anchors and solution hashes. Refreshing these anchors requires a new benchmark
oracle version rather than silently replacing the frozen files.
