#!/usr/bin/env bash
set -uo pipefail

# Run the requested PyVRP model matrix sequentially. Usage:
#   scripts/run-pyvrp-model-matrix.sh [TASK_ID]
#
# Optional environment variables:
#   PITBENCH_CONFIG     Evaluation config (default: config/evaluate.local.yaml)
#   PITBENCH_BATCH_ID   Stable prefix for run IDs (default: current UTC time)
#   PITBENCH_LOG_DIR    Per-model command logs and status summary

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname -- "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"

TASK_ID="${1:-pyvrp_v0_14_0}"
CONFIG_PATH="${PITBENCH_CONFIG:-config/evaluate.local.yaml}"
BATCH_ID="${PITBENCH_BATCH_ID:-$(date -u +%Y-%m-%d__%H-%M-%S)}"
LOG_DIR="${PITBENCH_LOG_DIR:-runs/model-matrix-${BATCH_ID}}"
STATUS_PATH="${LOG_DIR}/status.tsv"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "PitBench evaluation config not found: ${CONFIG_PATH}" >&2
  exit 2
fi

if ! docker version >/dev/null 2>&1; then
  echo "The current process cannot access the Docker API." >&2
  echo "Start this script from a fresh login process that includes the docker group." >&2
  echo "Creating a new tmux socket inside an old tmux session does not refresh groups." >&2
  id >&2
  stat -c 'Docker socket: %A owner=%U group=%G' /var/run/docker.sock >&2 || true
  exit 2
fi

if ! sudo -n -u pitbench-agy -- \
  /usr/local/libexec/pitbench-antigravity-runner --self-test >/dev/null; then
  echo "The installed Antigravity runner is unavailable or not isolated." >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"
printf 'order\tagent\tmodel\tsetting\trun_id\tstatus\texit_code\n' >"${STATUS_PATH}"

failures=0

run_one() {
  local order="$1"
  local agent="$2"
  local model="$3"
  local setting="$4"
  local label="$5"
  local run_id="${BATCH_ID}__${order}-${label}"
  local log_path="${LOG_DIR}/${order}-${label}.log"
  local -a command=(
    uv run pitbench evaluate "${TASK_ID}"
    --config "${CONFIG_PATH}"
    --agent "${agent}"
    --model "${model}"
    --run-id "${run_id}"
  )

  if [[ -n "${setting}" ]]; then
    command+=(--agent-kwarg "${setting}")
  fi

  echo
  echo "[${order}/5] ${agent} ${model} ${setting:-default}"
  echo "run_id=${run_id}"
  echo "log=${log_path}"

  "${command[@]}" 2>&1 | tee "${log_path}"
  local exit_code="${PIPESTATUS[0]}"
  local status="command_completed"
  if (( exit_code != 0 )); then
    status="command_failed"
    failures=$((failures + 1))
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${order}" "${agent}" "${model}" "${setting:-model-default}" \
    "${run_id}" "${status}" "${exit_code}" >>"${STATUS_PATH}"
  echo "status=${status} exit_code=${exit_code}"
}

run_one 1 codex gpt-5.6-sol reasoning_effort=xhigh codex-gpt-5-6-sol-xhigh
run_one 2 codex gpt-5.4-mini reasoning_effort=medium codex-gpt-5-4-mini-medium
run_one 3 antigravity gemini-3.7-flash-high "" antigravity-gemini-3-7-flash-high
run_one 4 antigravity gemini-3.1-pro-high "" antigravity-gemini-3-1-pro-high
run_one 5 antigravity gemini-3.5-flash-low "" antigravity-gemini-3-5-flash-low

echo
echo "Model matrix complete: ${STATUS_PATH}"
if (( failures != 0 )); then
  echo "${failures} evaluation(s) failed." >&2
  exit 1
fi
