#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  printf 'Usage: %s PROFILE [OUTPUT_DIR]\n' "$0" >&2
  exit 2
fi

profile=$1
shift
args=(
  --profile "$profile"
  --phase study-status
  --watch
  --watch-interval-sec 10
  --live-stream
)
if [[ $# -eq 1 ]]; then
  args+=(--output "$1")
fi

exec uv run python -m scripts.run_benchmark_release "${args[@]}"
