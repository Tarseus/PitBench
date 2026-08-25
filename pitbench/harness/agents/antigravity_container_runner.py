#!/usr/bin/python3
"""Ephemeral Antigravity launcher used in a Docker container without its socket."""

from __future__ import annotations

import argparse
import grp
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

AGY_BINARY = Path("/opt/pitbench/bin/agy")
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


def _validate_mcp_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.port is None
        or parsed.path != "/mcp"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("mcp URL must be http://127.0.0.1:<port>/mcp")
    return value


def _parse_agy_arguments(arguments: list[str]) -> list[str]:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--print", required=True)
    parser.add_argument("--output-format", choices=["stream-json"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=["accept-edits"], required=True)
    parser.add_argument("--sandbox", action="store_true", required=True)
    parser.add_argument("--disable-slash-commands", action="store_true")
    parser.add_argument("--print-timeout", required=True)
    parser.add_argument("--effort", choices=["low", "medium", "high"])
    parsed = parser.parse_args(arguments)
    result = [
        "--print",
        parsed.print,
        "--output-format",
        parsed.output_format,
        "--model",
        parsed.model,
        "--mode",
        parsed.mode,
        "--sandbox",
        "--print-timeout",
        parsed.print_timeout,
    ]
    if parsed.disable_slash_commands:
        result.append("--disable-slash-commands")
    if parsed.effort is not None:
        result.extend(["--effort", parsed.effort])
    return result


def _parse_command(arguments: list[str]) -> tuple[str, list[str]]:
    if not arguments or arguments[0] != "run":
        raise ValueError("runner only accepts the run operation")
    try:
        separator = arguments.index("--")
    except ValueError as error:
        raise ValueError("runner arguments must contain --") from error
    prefix = arguments[1:separator]
    if len(prefix) != 2 or prefix[0] != "--mcp-url":
        raise ValueError("run requires exactly one --mcp-url")
    return _validate_mcp_url(prefix[1]), _parse_agy_arguments(
        arguments[separator + 1 :]
    )


def _json_document(payload: dict, key: str) -> object:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty JSON string")
    return json.loads(value)


def _write_private_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(json.dumps(value, separators=(",", ":")))
    path.chmod(0o600)


def _copy_profile(gemini_config: Path, *, allow_hooks: bool) -> None:
    if not PROFILE_ROOT.is_dir():
        return
    for entry in PROFILE_ROOT.rglob("*"):
        if entry.is_symlink():
            raise ValueError(f"profile contains a symbolic link: {entry}")
        if entry.name == "hooks.json" and not allow_hooks:
            raise ValueError("profile contains hooks.json but allow_hooks is false")
    shutil.copytree(PROFILE_ROOT, gemini_config, dirs_exist_ok=True)


def _read_json_object(path: Path, *, label: str) -> dict:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"profile {label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"profile {label} must contain a JSON object: {path}")
    return value


def _configure_pitbench(gemini_config: Path, mcp_url: str) -> None:
    mcp_path = gemini_config / "mcp_config.json"
    mcp_config = _read_json_object(mcp_path, label="mcp_config.json")
    servers = mcp_config.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        raise ValueError("profile mcp_config.json mcpServers must be an object")
    servers["pitbench"] = {"url": mcp_url}
    _write_private_json(mcp_path, mcp_config)

    config_path = gemini_config / "config.json"
    config = _read_json_object(config_path, label="config.json")
    permissions = config.setdefault("permissions", {})
    if not isinstance(permissions, dict):
        raise ValueError("profile config.json permissions must be an object")
    allowed = permissions.setdefault("allow", [])
    if not isinstance(allowed, list) or not all(
        isinstance(value, str) for value in allowed
    ):
        raise ValueError("profile config.json permissions.allow must be a string list")
    required = "mcp(pitbench/run_command)"
    if required not in allowed:
        allowed.append(required)
    _write_private_json(config_path, config)


def _run_agy(mcp_url: str, arguments: list[str]) -> int:
    if not AGY_BINARY.is_file() or not os.access(AGY_BINARY, os.X_OK):
        raise RuntimeError(f"mounted agy binary is unavailable: {AGY_BINARY}")
    payload = json.load(sys.stdin)
    if not isinstance(payload, dict):
        raise ValueError("runner payload must be an object")
    auth_token = _json_document(payload, "auth_token_json")
    if (
        not isinstance(auth_token, dict)
        or not isinstance(auth_token.get("auth_method"), str)
        or not isinstance(auth_token.get("token"), dict)
    ):
        raise ValueError("auth_token_json has an invalid Antigravity token schema")
    settings = _json_document(payload, "settings_json")
    if not isinstance(settings, dict):
        raise ValueError("settings_json must contain an object")
    auth = (settings.get("security") or {}).get("auth") or {}
    selected_type = auth.get("selectedType")
    if not isinstance(selected_type, str) or not selected_type:
        raise ValueError("settings_json has no authentication selection")
    minimal_settings = {"security": {"auth": {"selectedType": selected_type}}}
    allow_hooks = payload.get("allow_hooks", False)
    if not isinstance(allow_hooks, bool):
        raise ValueError("allow_hooks must be true or false")
    requested_proxy_env = payload.get("proxy_env") or {}
    if not isinstance(requested_proxy_env, dict):
        raise ValueError("proxy_env must be an object")
    proxy_env = {
        key: str(value)
        for key, value in requested_proxy_env.items()
        if key in PROXY_KEYS and value
    }

    with tempfile.TemporaryDirectory(
        prefix="pitbench-antigravity-container-"
    ) as root_str:
        root = Path(root_str)
        workspace = root / "workspace"
        workspace.mkdir(mode=0o700)
        gemini_home = root / ".gemini"
        gemini_config = gemini_home / "config"
        gemini_config.mkdir(parents=True, mode=0o700)
        _copy_profile(gemini_config, allow_hooks=allow_hooks)
        _configure_pitbench(gemini_config, mcp_url)
        _write_private_json(gemini_home / "settings.json", minimal_settings)
        _write_private_json(
            gemini_home / "antigravity-cli" / "antigravity-oauth-token",
            auth_token,
        )
        _write_private_json(
            gemini_home / "antigravity-cli" / "settings.json",
            {
                "trustedWorkspaces": [str(workspace)],
                "toolPermission": "request-review",
            },
        )
        env = os.environ.copy()
        env.update(proxy_env)
        for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            env.pop(key, None)
        env["HOME"] = str(root)
        env["XDG_CACHE_HOME"] = str(root / "cache")
        env["XDG_CONFIG_HOME"] = str(root / "config")
        env["XDG_DATA_HOME"] = str(root / "data")
        env.pop("DBUS_SESSION_BUS_ADDRESS", None)
        env.pop("XDG_RUNTIME_DIR", None)
        result = subprocess.run(
            [str(AGY_BINARY), *arguments],
            cwd=workspace,
            env=env,
            stdin=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        return _self_test()
    mcp_url, agy_arguments = _parse_command(sys.argv[1:])
    return _run_agy(mcp_url, agy_arguments)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"pitbench-antigravity-container-runner: {error}", file=sys.stderr)
        raise SystemExit(2)
