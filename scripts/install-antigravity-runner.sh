#!/bin/sh
set -eu

if [ "$(id -u)" -ne 0 ]; then
  echo "run this installer with sudo" >&2
  exit 1
fi

TARGET_USER="${SUDO_USER:-}"
if [ -z "$TARGET_USER" ] || [ "$TARGET_USER" = "root" ]; then
  echo "SUDO_USER must identify the desktop user running PitBench" >&2
  exit 1
fi

case "$TARGET_USER" in
  *[!A-Za-z0-9_-]*)
    echo "unsupported user name: $TARGET_USER" >&2
    exit 1
    ;;
esac

RUNNER_USER="pitbench-agy"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
AGY_SOURCE="${1:-/home/$TARGET_USER/.local/bin/agy}"
AGY_SOURCE=$(readlink -f "$AGY_SOURCE")

if [ ! -x "$AGY_SOURCE" ]; then
  echo "agy executable not found: $AGY_SOURCE" >&2
  exit 1
fi

if ! id "$RUNNER_USER" >/dev/null 2>&1; then
  useradd --system --user-group --no-create-home \
    --home-dir /nonexistent --shell /usr/sbin/nologin "$RUNNER_USER"
fi

if id -nG "$RUNNER_USER" | tr ' ' '\n' | grep -qx docker; then
  echo "$RUNNER_USER must not belong to the docker group" >&2
  exit 1
fi

install -d -o root -g root -m 0755 /usr/local/libexec
install -o root -g root -m 0755 \
  "$SCRIPT_DIR/pitbench-antigravity-runner" \
  /usr/local/libexec/pitbench-antigravity-runner
install -o root -g root -m 0755 \
  "$AGY_SOURCE" \
  /usr/local/libexec/pitbench-agy-bin

SUDOERS_TMP=$(mktemp)
trap 'rm -f "$SUDOERS_TMP"' EXIT
printf '%s ALL=(%s) NOPASSWD: /usr/local/libexec/pitbench-antigravity-runner *\n' \
  "$TARGET_USER" "$RUNNER_USER" >"$SUDOERS_TMP"
chmod 0440 "$SUDOERS_TMP"
visudo -cf "$SUDOERS_TMP" >/dev/null
install -o root -g root -m 0440 \
  "$SUDOERS_TMP" /etc/sudoers.d/pitbench-antigravity

echo "installed isolated Antigravity runner for $TARGET_USER"
