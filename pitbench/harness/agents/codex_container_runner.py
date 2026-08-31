#!/usr/bin/python3
"""Ephemeral Codex launcher used inside a Docker container without its socket."""

from __future__ import annotations

import grp
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CODEX_BINARY = Path("/opt/pitbench/bin/codex")
CODE_MODE_HOST = Path("/opt/pitbench/bin/codex-code-mode-host")
PROFILE_ROOT = Path("/opt/pitbench/profile")
PROXY_KEYS = {
    "ALL_PROXY",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "all_proxy",
    "http_proxy",
    "https_proxy",
    "no_proxy",
}


def _self_test() -> int:
    group_names: list[str] = []
    for group_id in os.getgroups():
        try:
            group_names.append(grp.getgrgid(group_id).gr_name)
        except KeyError:
            group_names.append(str(group_id))
    docker_socket_access = os.access("/var/run/docker.sock", os.R_OK | os.W_OK)
    print(
        json.dumps(
            {
                "uid": os.getuid(),
                "groups": sorted(group_names),
                "docker_socket_present": Path("/var/run/docker.sock").exists(),
                "docker_socket_access": docker_socket_access,
                "profile_mounted": PROFILE_ROOT.is_dir(),
            },
            sort_keys=True,
        )
    )
    return int(docker_socket_access or "docker" in group_names)


def _copy_profile(codex_home: Path) -> None:
    if not PROFILE_ROOT.is_dir():
        return
    for entry in PROFILE_ROOT.rglob("*"):
        if entry.is_symlink():
            raise ValueError(f"profile contains a symbolic link: {entry}")
    shutil.copytree(PROFILE_ROOT, codex_home, dirs_exist_ok=True)


def _run_codex(arguments: list[str]) -> int:
    for binary in (CODEX_BINARY, CODE_MODE_HOST):
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise RuntimeError(f"mounted Codex executable is unavailable: {binary}")
    payload = json.load(sys.stdin)
    if not isinstance(payload, dict):
        raise ValueError("runner payload must be an object")
    auth_json = payload.get("auth_json")
    if not isinstance(auth_json, str) or not auth_json.strip():
        raise ValueError("auth_json must be a non-empty string")
    json.loads(auth_json)
    requested_proxy_env = payload.get("proxy_env") or {}
    if not isinstance(requested_proxy_env, dict):
        raise ValueError("proxy_env must be an object")
    proxy_env = {
        key: str(value)
        for key, value in requested_proxy_env.items()
        if key in PROXY_KEYS and value
    }
    allow_hooks = payload.get("allow_hooks", False)
    if not isinstance(allow_hooks, bool):
        raise ValueError("allow_hooks must be true or false")

    with tempfile.TemporaryDirectory(prefix="pitbench-codex-container-") as root_str:
        root = Path(root_str)
        codex_home = root / "codex-home"
        workspace = root / "workspace"
        codex_home.mkdir(mode=0o700)
        workspace.mkdir(mode=0o700)
        _copy_profile(codex_home)
        auth_path = codex_home / "auth.json"
        if auth_path.exists():
            raise ValueError("profile attempted to provide reserved auth.json")
        auth_path.write_text(auth_json)
        auth_path.chmod(0o600)
        env = os.environ.copy()
        env.update(proxy_env)
        env.pop("OPENAI_API_KEY", None)
        env["CODEX_HOME"] = str(codex_home)
        env["HOME"] = str(root)
        command = [str(CODEX_BINARY), "--ask-for-approval", "never"]
        if allow_hooks:
            command.append("--dangerously-bypass-hook-trust")
        command.extend(arguments)
        result = subprocess.run(
            command,
            cwd=workspace,
            env=env,
            stdin=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        return _self_test()
    if not sys.argv[1:] or sys.argv[1] != "exec":
        raise ValueError("runner only accepts Codex exec arguments")
    return _run_codex(sys.argv[1:])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"pitbench-codex-container-runner: {error}", file=sys.stderr)
        raise SystemExit(2)
