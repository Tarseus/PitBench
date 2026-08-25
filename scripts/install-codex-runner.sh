#!/bin/sh
set -eu

if [ "$(id -u)" -ne 0 ]; then
  echo "run this installer with sudo" >&2
  exit 1
fi

TARGET_USER="${PITBENCH_TARGET_USER:-${SUDO_USER:-}}"
if [ -z "$TARGET_USER" ] || [ "$TARGET_USER" = "root" ]; then
  echo "PITBENCH_TARGET_USER or SUDO_USER must identify the PitBench user" >&2
  exit 1
fi

case "$TARGET_USER" in
  *[!A-Za-z0-9_-]*)
    echo "unsupported user name: $TARGET_USER" >&2
    exit 1
    ;;
esac

RUNNER_USER="pitbench-codex"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
CODEX_SOURCE="${1:-/home/$TARGET_USER/.local/bin/codex}"
CODEX_SOURCE=$(readlink -f "$CODEX_SOURCE")
CODEX_SOURCE_DIR=$(dirname "$CODEX_SOURCE")
CODE_MODE_HOST_SOURCE="$CODEX_SOURCE_DIR/codex-code-mode-host"

if [ ! -x "$CODEX_SOURCE" ]; then
  echo "Codex executable not found: $CODEX_SOURCE" >&2
  exit 1
fi

if [ ! -x "$CODE_MODE_HOST_SOURCE" ]; then
  echo "Codex code-mode host not found: $CODE_MODE_HOST_SOURCE" >&2
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
  "$SCRIPT_DIR/pitbench-codex-runner" \
  /usr/local/libexec/pitbench-codex-runner
install -o root -g root -m 0755 \
  "$CODEX_SOURCE" \
  /usr/local/libexec/pitbench-codex-bin
install -o root -g root -m 0755 \
  "$CODE_MODE_HOST_SOURCE" \
  /usr/local/libexec/codex-code-mode-host

SUDOERS_TMP=$(mktemp)
trap 'rm -f "$SUDOERS_TMP"' EXIT
printf '%s ALL=(%s) NOPASSWD: /usr/local/libexec/pitbench-codex-runner *\n' \
  "$TARGET_USER" "$RUNNER_USER" >"$SUDOERS_TMP"
chmod 0440 "$SUDOERS_TMP"
visudo -cf "$SUDOERS_TMP" >/dev/null
install -o root -g root -m 0440 \
  "$SUDOERS_TMP" /etc/sudoers.d/pitbench-codex

echo "installed isolated Codex runner for $TARGET_USER"
