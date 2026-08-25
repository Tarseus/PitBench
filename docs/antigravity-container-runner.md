# Reproducible Antigravity runner

PitBench can run Antigravity without installing a root-owned launcher. The
evaluator starts an ephemeral container, read-only mounts the host's `agy`
executable, and sends the existing Google OAuth token and authentication
selection to the runner over stdin. It does not mount the Docker socket, host
home, benchmark task directory, or evaluator assets.

Docker access is still required because PitBench itself is containerized. If a
new Docker group membership is not visible inside tmux, start the tmux server
from a fresh SSH login or run `newgrp docker` first.

## Configure the container backend

The example evaluation configuration defaults to:

```yaml
agents:
  antigravity:
    agy_binary: agy
    auth_token_path: ~/.gemini/antigravity-cli/antigravity-oauth-token
    settings_path: ~/.gemini/antigravity-cli/settings.json
    runner_backend: container
    container_runner_image: python:3.13-slim-bookworm
    profile_path: null
    proxy_url: null
```

The runner image is pulled automatically before an evaluation if absent.
`pitbench doctor pyvrp` remains diagnostic-only and tells the user which image
to pull. Every trial writes `antigravity-runner.json` containing the resolved
image ID and selected profile hash. Use an immutable image digest for strict
reruns.

## Create a custom profile

Antigravity documents `~/.gemini/config/` as its global customization root.
PitBench profiles exactly that non-secret subtree instead of copying the whole
`~/.gemini` directory, which also contains OAuth, history, logs, and sessions.

```bash
uv run pitbench profiles init my-agy --agent antigravity --allow-hooks
uv run pitbench profiles validate agent-profiles/my-agy
```

The generated layout is:

```text
agent-profiles/my-agy/
├── profile.yaml
└── gemini-config/
    ├── config.json
    ├── hooks.json
    ├── skills/
    ├── rules/
    └── plugins/
```

Select it in `config/evaluate.local.yaml`:

```yaml
agents:
  antigravity:
    runner_backend: container
    profile_path: agent-profiles/my-agy
```

Relative paths are resolved from the PitBench repository root. The profile may
also provide `skills.json`, `plugins.json`, and `mcp_config.json`. PitBench
preserves those files, injects or replaces only the dynamic `pitbench` MCP
entry, and appends the exact `mcp(pitbench/run_command)` permission required by
the benchmark.

Without a profile, PitBench retains Antigravity's
`--disable-slash-commands` restriction. Selecting a profile intentionally
removes that flag so its skills and plugin customizations can participate in the
run; the selected profile hash makes that experimental condition explicit.

The validator rejects symbolic links, special files, path escapes, and known
Antigravity credential filenames. Its SHA-256 covers the normalized manifest,
relative paths, contents, and executable bits. OAuth and minimal runtime
settings are written with mode `0600` inside a fresh temporary HOME for every
invocation.

Antigravity hooks execute automatically when discovered. A profile containing
any `hooks.json` is therefore rejected unless its manifest explicitly sets
`allow_hooks: true`. Plugins, hooks, skills, rules, and extra MCP servers are
trusted experiment code: they run with the agent's temporary OAuth and network
access and may invoke the same PitBench MCP interface. They receive no task
filesystem mount, and the independent judge remains isolated.

## Legacy host backend

The root-owned dedicated-user runner remains available for compatibility:

```yaml
agents:
  antigravity:
    runner_backend: host
    runner_path: /usr/local/libexec/pitbench-antigravity-runner
    runner_user: pitbench-agy
    profile_path: null
```

Install it with `sudo scripts/install-antigravity-runner.sh`. Custom profiles
are supported only by the container backend.
