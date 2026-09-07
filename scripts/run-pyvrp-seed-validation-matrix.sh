#!/usr/bin/env bash
set -euo pipefail

# Run four fresh M5 panels sequentially using the original local experiment settings.
# Usage: bash scripts/run-pyvrp-seed-validation-matrix.sh /absolute/prepared/batch [private-root]
# The batch must contain a frozen harness/ and sources/<task_id>/ for each release.
BATCH_DIR="${1:?Provide the absolute prepared batch directory}"
if [[ "$BATCH_DIR" != /* || ! -d "$BATCH_DIR/harness" ]]; then
  echo "Expected an absolute batch directory containing harness/" >&2
  exit 2
fi
JUDGE_IMAGE="sha256:461a73388c878eb3ae0f83278a15ca225de673670598c7961d11de0eaa9ff84b"
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PRIVATE_ROOT="${2:-$PROJECT_ROOT/private}"
test -d "$PRIVATE_ROOT"
TASK_IDS=(pyvrp_v0_12_2 pyvrp_v0_13_0 pyvrp_v0_13_4 pyvrp_v0_14_0)

exec 9>"$BATCH_DIR/runner.lock"
if ! flock -n 9; then
  echo "This batch already has an active runner" >&2
  exit 2
fi
docker image inspect "$JUDGE_IMAGE" >/dev/null
for task_id in "${TASK_IDS[@]}"; do
  test -d "$BATCH_DIR/sources/$task_id/.git"
  test -f "$BATCH_DIR/harness/configs/tasks/$task_id.yaml"
done

for task_id in "${TASK_IDS[@]}"; do
  output_dir="$BATCH_DIR/$task_id"
  mkdir -p -- "$output_dir"
  if [[ -f "$output_dir/validation_summary.json" ]]; then
    echo "$task_id already completed in this batch; preserving its results"
    continue
  fi
  available_memory_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
  if (( available_memory_kib < 8 * 1024 * 1024 )); then
    echo "Insufficient available memory for the original 8 GiB settings: ${available_memory_kib} KiB" >&2
    exit 2
  fi
  echo "Starting $task_id at $(date -Is)"
  docker run --rm --pull=never \
    --name "pitbench-m5-four-versions-$task_id" \
    --network none --read-only --cpus 8 --memory 8g --pids-limit 2048 \
    --tmpfs /tmp:rw,exec,nosuid,size=4g \
    --env PYTHONPATH=/opt/pitbench \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --env GIT_CONFIG_COUNT=1 \
    --env GIT_CONFIG_KEY_0=safe.directory \
    --env GIT_CONFIG_VALUE_0=/input/base \
    --volume "$BATCH_DIR/harness:/opt/pitbench:ro" \
    --volume "$BATCH_DIR/harness/configs/tasks/$task_id.yaml:/input/task.yaml:ro" \
    --volume "$BATCH_DIR/sources/$task_id:/input/base:ro" \
    --volume "$PRIVATE_ROOT:/private:ro" \
    --volume "$output_dir:/output:rw" \
    "$JUDGE_IMAGE" python /opt/pitbench/scripts/validate_seed_robustness_real_solver.py \
    --inside-judge --task-config /input/task.yaml --repository /input/base \
    --private-root /private --output-dir /output --instance-limit 10 \
    --parallel-runs 8 --reference-seed-count 700 --test-seed-count 300 \
    --test-list-count 1000 --generation-seed 20260902 \
    2>&1 | tee -a "$output_dir/validation.log"
  echo "Completed $task_id at $(date -Is)"
done
echo "Four-version M5 batch complete: $BATCH_DIR"
