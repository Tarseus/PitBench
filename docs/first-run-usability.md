# First-run usability findings

This note records the first external-machine PyVRP run on `g53` on 2026-08-25. It
separates observed setup failures from proposed product changes so that the
public quickstart can eventually become a reliable one-command workflow.

## What worked

- Both published PyVRP images were anonymously accessible with a clean Docker
  client configuration.
- The prepared agent image contained the expected censored PyVRP v0.14.0 tree.
- The transferred private assets passed `pitbench tasks validate`.
- A real end-to-end smoke run completed with the independent verifier and two
  valid observations: one base and one no-op agent run on a real CVRPLIB case.

## Friction observed

1. Docker group membership only took effect after starting a new login session.
   The resulting socket error looked unrelated to PitBench.
2. The server could not reach GHCR directly. Docker depended on a user-session
   proxy at `127.0.0.1:17898`, which intermittently dropped TLS connections.
3. There was no documented offline image transfer path. The pipeline
   `docker save IMAGE | ssh HOST docker load` worked, but loading a digest-only
   archive produced an unnamed image that needed a local tag or image ID.
4. A local judge image ID works because the judge invokes `docker run` directly.
   A local agent image ID does not work in the generated Dockerfile: BuildKit
   treats `FROM sha256:...` as a remote image name. The agent image needs a tag.
5. `pitbench tasks smoke` says that production smoke tests use `--real`, but the
   option does not exist. The only built-in smoke is a synthetic contract test.
6. The real judge previously applied `--cpus 1.0` to its entire container. That
   correctly constrained solver work but also throttled C++ builds and pytest.
7. A no-op smoke still builds and tests separate base and agent validation trees,
   then builds two more performance trees. The fixed setup cost dominates the
   tiny solver grid.
8. `pitbench run --help` still refers to the default `fc-eval` registry, and one
   unit test still hard-codes the `fc-eval` distribution name.
9. `pitbench tasks validate` lists task families whose public images are not yet
   provisioned, although the public quickstart currently supports only PyVRP.
10. Users must manually copy evaluator-owned `private/` assets and discover that
    `runs/` is intentionally local output.
11. The isolated-agent installers originally inferred the target user only from
    `SUDO_USER`. An administrator installing on behalf of a different benchmark
    user had no explicit target-user interface.

## Recommended changes

### First priority

- Add `pitbench doctor pyvrp`. It should check Docker socket access, architecture,
  free disk and memory, private-asset checksums, required images, anonymous GHCR
  access, and whether a proxy is configured on the daemon rather than only in the
  shell. Errors should include the exact recovery command.
- Implement a real `pitbench tasks smoke pyvrp` path using one real instance per
  judge population, one seed, and the shortest budget while retaining the full
  independent correctness gate.
- Cache validated base builds by task commit, judge image digest, build kind, and
  compiler configuration. Candidate builds remain patch-specific. A no-op smoke
  should reuse the base tree rather than compiling the same tree four times.
- Keep build resources separate from solver resources. Build/test receives an
  eight-CPU container budget by default; each solver process is bound with
  `taskset` to the manifest's declared `threads` count.
- Pin the prepared agent and judge images in the PyVRP manifest and resolve them
  automatically. Users should not have to paste two long digest overrides into
  every command.
- Provide an offline image command that exports named archives and prints the
  exact local references accepted by `--agent-image` and `--judge-image`.

### Follow-up

- Remove remaining FormulaCode names from CLI help, tests, logs, and registry
  defaults.
- Avoid forcing a task-wrapper rebuild solely because `--agent-image` was passed;
  reuse it when the source and image-revision labels match.
- Show judge phases (`validation/base`, `validation/agent`, `performance/base`,
  `performance/agent`, solver grid), elapsed time, estimated remaining time, and
  resume instructions in the main run log.
- Package private evaluator assets as a checksum-verified bundle with an explicit
  provisioning command instead of asking users to copy an internal directory.
- Scope the public task listing to provisioned solvers, while keeping the plugin
  interfaces available for future CO solver repositories.

## Target workflow

The intended public experience is:

```bash
uv run pitbench doctor pyvrp
uv run pitbench evaluate pyvrp_v0_14_0 --agent codex --model <model>
```

The first command should either finish with a clear readiness confirmation or
provide bounded remediation. The second command should reuse prepared images and
cached base builds, then start the agent and real evaluation without additional
environment setup.
