# Private evaluator storage

This directory is never mounted into an agent container. Local installations place
hidden populations, independent verifiers, and oracle data here using
the paths referenced by `private://` task URIs.

Everything below this README is gitignored. A production judge must verify hashes and
fail closed when an asset is missing; fixture mode is explicit and cannot substitute
for a real evaluation result.
