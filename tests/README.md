# Test organization

`tests/unit/` is a flat set of functional unit/component suites. Related public
input/image/source provisioning, geometry, behavior, and retained-record cases
share suites instead of mirroring package directories or creating one file per
change. Large independent evaluator and metric domains remain focused suites.

`tests/support.py` holds the repository root and shared scenario/transfer sample
builders. Tests import these builders directly, not other test modules. Their
sample budgets, seeds and protocol settings remain explicit and unchanged.
`tests/fixtures/` retains static input fixtures.

Run local non-solver coverage with:

```bash
uv run python -m pytest tests/unit -m 'not docker and not external'
```

Docker/external tests keep their existing markers and require their actual
environments. Test collection during this consolidation was compared by test
name and parameter identity, independently of changed file paths, to detect
missing or accidentally overridden tests.
