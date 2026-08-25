# Reproducible Codex runner

PitBench can run Codex without installing a root-owned launcher. The evaluator
starts an ephemeral runner container, read-only mounts the host's Codex and
`codex-code-mode-host` executables, and sends the existing ChatGPT OAuth login to
the runner over stdin. It does not mount the Docker socket, host home, benchmark
task directory, or evaluator assets.

Docker access is still required. This removes the Codex-specific `sudo`
installation step; it does not remove Docker from PitBench. If `docker version`
fails in a tmux session after an administrator added the account to the Docker
group, leave that tmux server and start it from a fresh SSH login (or use
`newgrp docker`) before running PitBench.

## Configure the container backend

Copy `config/evaluate.example.yaml` to `config/evaluate.local.yaml`. Its Codex
section defaults to:

```yaml
agents:
  codex:
    codex_binary: codex
    codex_auth_path: ~/.codex/auth.json
    runner_backend: container
    container_runner_image: python:3.13-slim-bookworm
    profile_path: null
    proxy_url: null
```

The image is pulled automatically before the first evaluation if it is absent.
`pitbench doctor pyvrp` only diagnoses state, so it reports the exact `docker
pull` command instead of changing the machine. For strict reruns, replace the
image tag with an immutable digest. Every trial writes `codex-runner.json` with
the resolved image ID even when a tag is used.

## Create a custom profile

A profile is a versioned, non-secret `CODEX_HOME` overlay:

```text
agent-profiles/my-codex/
├── profile.yaml
└── codex-home/
    ├── config.toml
    ├── plugins/
    └── skills/
```

Create and validate one with:

```bash
uv run pitbench profiles init my-codex --allow-hooks
uv run pitbench profiles validate agent-profiles/my-codex
```

Then select it in the local evaluation configuration:

```yaml
agents:
  codex:
    runner_backend: container
    profile_path: agent-profiles/my-codex
```

Relative profile paths are resolved from the PitBench repository root. To use
Codex's plugin manager while preparing the overlay, point `CODEX_HOME` at it:

```bash
CODEX_HOME="$PWD/agent-profiles/my-codex/codex-home" \
  codex plugin marketplace add OWNER/REPOSITORY --ref COMMIT
CODEX_HOME="$PWD/agent-profiles/my-codex/codex-home" \
  codex plugin add PLUGIN@MARKETPLACE
uv run pitbench profiles validate agent-profiles/my-codex
```

The validator rejects symbolic links, special files, path escapes, and
`auth.json`. Its SHA-256 covers the normalized manifest, every relative file
path and file content, and whether each file is executable. PitBench copies the
validated overlay into a new temporary `CODEX_HOME` for every invocation, then
writes a temporary mode-`0600` `auth.json` there. The host login file is never
added to the profile.

Set `allow_hooks: true` only for a profile whose hooks you trust. In that case
PitBench explicitly enables hook execution for the automated Codex invocation.
Plugins and hooks run in the Codex runner and can therefore read its temporary
OAuth file, use its outbound network access, and invoke the same loopback MCP
interface as the agent. Treat the whole profile as trusted experiment code. It
receives no task filesystem mount; all task access goes through PitBench MCP,
while judging remains isolated from the runner.

## Legacy host backend

The existing root-owned dedicated-user runner remains available for compatibility:

```yaml
agents:
  codex:
    runner_backend: host
    runner_path: /usr/local/libexec/pitbench-codex-runner
    runner_user: pitbench-codex
    profile_path: null
```

Install that backend with `sudo scripts/install-codex-runner.sh`. Custom profiles
are supported only by the container backend.
