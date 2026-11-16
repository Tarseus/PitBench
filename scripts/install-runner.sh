#!/bin/sh
set -eu

usage() {
  echo "Usage: sudo scripts/install-runner.sh {codex|antigravity} [SOURCE_BINARY]"
}
case "${1:-}" in
  -h|--help) usage; exit 0 ;;
  codex) AGENT=codex; RUNNER_USER=pitbench-codex; BINARY=codex ;;
  antigravity) AGENT=antigravity; RUNNER_USER=pitbench-agy; BINARY=agy ;;
  *) usage >&2; exit 2 ;;
esac
shift

if [ "$(id -u)" -ne 0 ]; then
  echo "Run this installer with sudo" >&2
  exit 1
fi
TARGET_USER="${PITBENCH_TARGET_USER:-${SUDO_USER:-}}"
if [ -z "$TARGET_USER" ] || [ "$TARGET_USER" = root ]; then
  echo "PITBENCH_TARGET_USER or SUDO_USER must identify the PitBench user" >&2
  exit 1
fi
case "$TARGET_USER" in
  *[!A-Za-z0-9_-]*) echo "unsupported user name: $TARGET_USER" >&2; exit 1 ;;
esac

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SOURCE="${1:-/home/$TARGET_USER/.local/bin/$BINARY}"
SOURCE=$(readlink -f "$SOURCE")
if [ ! -x "$SOURCE" ]; then
  echo "$BINARY executable not found: $SOURCE" >&2
  exit 1
fi
if [ "$AGENT" = codex ]; then
  CODE_MODE_HOST_SOURCE="$(dirname "$SOURCE")/codex-code-mode-host"
  if [ ! -x "$CODE_MODE_HOST_SOURCE" ]; then
    echo "Codex code-mode host not found: $CODE_MODE_HOST_SOURCE" >&2
    exit 1
  fi
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
install -o root -g root -m 0755 "$SCRIPT_DIR/pitbench-$AGENT-runner" \
  "/usr/local/libexec/pitbench-$AGENT-runner"
install -o root -g root -m 0755 "$SOURCE" "/usr/local/libexec/pitbench-$BINARY-bin"
if [ "$AGENT" = codex ]; then
  install -o root -g root -m 0755 "$CODE_MODE_HOST_SOURCE" /usr/local/libexec/codex-code-mode-host
fi

SUDOERS_TMP=$(mktemp)
trap 'rm -f "$SUDOERS_TMP"' EXIT
printf '%s ALL=(%s) NOPASSWD: /usr/local/libexec/pitbench-%s-runner *\n' \
  "$TARGET_USER" "$RUNNER_USER" "$AGENT" >"$SUDOERS_TMP"
chmod 0440 "$SUDOERS_TMP"
visudo -cf "$SUDOERS_TMP" >/dev/null
install -o root -g root -m 0440 "$SUDOERS_TMP" "/etc/sudoers.d/pitbench-$AGENT"
echo "installed isolated $AGENT runner for $TARGET_USER"
