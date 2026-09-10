#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/run-seed-validation-matrix.sh BATCH_DIR PRIVATE_ROOT TASK_ID...
# The prepared batch contains harness/ and sources/<task_id>/ checkouts.
# Each task supplies its judge image, or PITBENCH_JUDGE_IMAGE overrides it.
BATCH_DIR="${1:?Provide the absolute prepared batch directory}"
PRIVATE_ROOT="${2:?Provide the absolute private asset directory}"
shift 2
if [[ "$BATCH_DIR" != /* || ! -d "$BATCH_DIR/harness" ]]; then
  echo "Expected an absolute batch directory containing harness/" >&2
  exit 2
fi
if [[ "$PRIVATE_ROOT" != /* || ! -d "$PRIVATE_ROOT" || "$#" -eq 0 ]]; then
  echo "Expected an absolute private asset directory and at least one task ID" >&2
  exit 2
fi

exec 9>"$BATCH_DIR/runner.lock"
if ! flock -n 9; then
  echo "This batch already has an active runner" >&2
  exit 2
fi
for task_id in "$@"; do
  test -e "$BATCH_DIR/sources/$task_id/.git"
  test -f "$BATCH_DIR/harness/configs/tasks/$task_id.yaml"
done

for task_id in "$@"; do
  output_dir="$BATCH_DIR/$task_id"
  mkdir -p -- "$output_dir"
  if [[ -f "$output_dir/validation_summary.json" ]]; then
    echo "$task_id already completed in this batch; preserving its results"
    continue
  fi
  command=(
    "${PITBENCH_PYTHON:-python}"
    "$BATCH_DIR/harness/scripts/validate_seed_robustness_real_solver.py"
    --task-config "$BATCH_DIR/harness/configs/tasks/$task_id.yaml"
    --repository "$BATCH_DIR/sources/$task_id"
    --private-root "$PRIVATE_ROOT"
    --output-dir "$output_dir"
  )
  judge_image="${PITBENCH_JUDGE_IMAGE:-}"
  if [[ -z "$judge_image" ]]; then
    judge_image="$("${PITBENCH_PYTHON:-python}" -c \
      'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["repository"].get("judge_image") or "")' \
      "$BATCH_DIR/harness/configs/tasks/$task_id.yaml")"
  fi
  if [[ -z "$judge_image" ]]; then
    echo "$task_id requires repository.judge_image or PITBENCH_JUDGE_IMAGE" >&2
    exit 2
  fi
  command+=(--judge-image "$judge_image")
  echo "Starting $task_id at $(date -Is)"
  "${command[@]}" 2>&1 | tee -a "$output_dir/validation.log"
  echo "Completed $task_id at $(date -Is)"
done
echo "Seed validation batch complete: $BATCH_DIR"
