#!/usr/bin/python3
"""Ephemeral entry point for isolated coding-agent CLI containers."""

from __future__ import annotations

import argparse
import grp
import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import urlsplit

PROFILE_ROOT = Path("/opt/pitbench/profile")
RECORDING_SCRIPT = Path("/opt/pitbench/recording.py")
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


class ContainerEntrypoint(ABC):
    """Common isolated workspace, profile, proxy, and recording lifecycle."""

    provider: str
    binary: Path
    credential_environment_keys: tuple[str, ...] = ()
    clear_inherited_proxy = False

    @abstractmethod
    def run(self, arguments: list[str], payload: dict) -> int: ...

    def validate_binaries(self) -> None:
        if not self.binary.is_file() or not os.access(self.binary, os.X_OK):
            raise RuntimeError(
                f"mounted {self.provider} executable is unavailable: {self.binary}"
            )

    def environment(self, root: Path, proxy_env: dict[str, str]) -> dict[str, str]:
        env = os.environ.copy()
        if self.clear_inherited_proxy:
            for key in PROXY_KEYS:
                env.pop(key, None)
        env.update(proxy_env)
        for key in self.credential_environment_keys:
            env.pop(key, None)
        env["HOME"] = str(root)
        return env

    def copy_profile(self, destination: Path, *, allow_hooks: bool) -> None:
        if not PROFILE_ROOT.is_dir():
            return
        for entry in PROFILE_ROOT.rglob("*"):
            if entry.is_symlink():
                raise ValueError(f"profile contains a symbolic link: {entry}")
            if entry.name == "hooks.json" and not allow_hooks:
                raise ValueError("profile contains hooks.json but allow_hooks is false")
        shutil.copytree(PROFILE_ROOT, destination, dirs_exist_ok=True)

    def configure_recording(self, hook_path: Path) -> bool:
        if not RECORDING_SCRIPT.is_file():
            return False
        return bool(
            runpy.run_path(str(RECORDING_SCRIPT))["configure_mounted_hooks"](
                hook_path,
                self.provider,
            )
        )

    def finalize_recording(self) -> None:
        if RECORDING_SCRIPT.is_file():
            runpy.run_path(str(RECORDING_SCRIPT))["finalize_mounted_recording"](
                self.provider
            )

    @staticmethod
    def payload() -> dict:
        value = json.load(sys.stdin)
        if not isinstance(value, dict):
            raise ValueError("runner payload must be an object")
        return value

    @staticmethod
    def proxy_environment(payload: dict) -> dict[str, str]:
        requested = payload.get("proxy_env") or {}
        if not isinstance(requested, dict):
            raise ValueError("proxy_env must be an object")
        return {
            key: str(value)
            for key, value in requested.items()
            if key in PROXY_KEYS and value
        }

    @staticmethod
    def allow_hooks(payload: dict) -> bool:
        value = payload.get("allow_hooks", False)
        if not isinstance(value, bool):
            raise ValueError("allow_hooks must be true or false")
        return value

    @staticmethod
    def json_document(payload: dict, key: str) -> object:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a non-empty JSON string")
        return json.loads(value)

    @staticmethod
    def write_private_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.write_text(json.dumps(value, separators=(",", ":")))
        path.chmod(0o600)

    @staticmethod
    def self_test() -> int:
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


class CodexEntrypoint(ContainerEntrypoint):
    provider = "codex"
    binary = Path("/opt/pitbench/bin/codex")
    code_mode_host = Path("/opt/pitbench/bin/codex-code-mode-host")
    credential_environment_keys = ("OPENAI_API_KEY",)

    def validate_binaries(self) -> None:
        super().validate_binaries()
        if not self.code_mode_host.is_file() or not os.access(
            self.code_mode_host, os.X_OK
        ):
            raise RuntimeError(
                f"mounted Codex executable is unavailable: {self.code_mode_host}"
            )

    def run(self, arguments: list[str], payload: dict) -> int:
        if not arguments or arguments[0] != "exec":
            raise ValueError("runner only accepts Codex exec arguments")
        self.validate_binaries()
        auth_json = payload.get("auth_json")
        if not isinstance(auth_json, str) or not auth_json.strip():
            raise ValueError("auth_json must be a non-empty string")
        json.loads(auth_json)
        allow_hooks = self.allow_hooks(payload)
        proxy_env = self.proxy_environment(payload)

        with tempfile.TemporaryDirectory(
            prefix="pitbench-codex-container-"
        ) as root_str:
            root = Path(root_str)
            codex_home = root / "codex-home"
            workspace = root / "workspace"
            codex_home.mkdir(mode=0o700)
            workspace.mkdir(mode=0o700)
            # Codex applies its own hook trust gate at launch time. Preserve the
            # overlay here even when hooks were not explicitly enabled.
            self.copy_profile(codex_home, allow_hooks=True)
            recording_hooks = self.configure_recording(codex_home / "hooks.json")
            auth_path = codex_home / "auth.json"
            if auth_path.exists():
                raise ValueError("profile attempted to provide reserved auth.json")
            auth_path.write_text(auth_json)
            auth_path.chmod(0o600)
            env = self.environment(root, proxy_env)
            env["CODEX_HOME"] = str(codex_home)
            command = [str(self.binary), "--ask-for-approval", "never"]
            if allow_hooks or recording_hooks:
                command.append("--dangerously-bypass-hook-trust")
            command.extend(arguments)
            result = subprocess.run(
                command,
                cwd=workspace,
                env=env,
                stdin=subprocess.DEVNULL,
                check=False,
            )
            if recording_hooks:
                self.finalize_recording()
            return result.returncode


class AntigravityEntrypoint(ContainerEntrypoint):
    provider = "antigravity"
    binary = Path("/opt/pitbench/bin/agy")
    credential_environment_keys = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    clear_inherited_proxy = True

    @staticmethod
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

    @staticmethod
    def _parse_arguments(arguments: list[str]) -> list[str]:
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

    def _parse_command(self, arguments: list[str]) -> tuple[str | None, list[str]]:
        if arguments == ["models"]:
            return None, ["models"]
        if not arguments or arguments[0] != "run":
            raise ValueError("runner only accepts run or models")
        try:
            separator = arguments.index("--")
        except ValueError as error:
            raise ValueError("runner arguments must contain --") from error
        prefix = arguments[1:separator]
        if len(prefix) != 2 or prefix[0] != "--mcp-url":
            raise ValueError("run requires exactly one --mcp-url")
        return self._validate_mcp_url(prefix[1]), self._parse_arguments(
            arguments[separator + 1 :]
        )

    @classmethod
    def _read_json_object(cls, path: Path, *, label: str) -> dict:
        if not path.exists():
            return {}
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"profile {label} is not valid JSON: {path}") from error
        if not isinstance(value, dict):
            raise ValueError(f"profile {label} must contain a JSON object: {path}")
        return value

    def _configure_pitbench(self, gemini_config: Path, mcp_url: str) -> None:
        mcp_path = gemini_config / "mcp_config.json"
        mcp_config = self._read_json_object(mcp_path, label="mcp_config.json")
        servers = mcp_config.setdefault("mcpServers", {})
        if not isinstance(servers, dict):
            raise ValueError("profile mcp_config.json mcpServers must be an object")
        servers["pitbench"] = {"url": mcp_url}
        self.write_private_json(mcp_path, mcp_config)

        config_path = gemini_config / "config.json"
        config = self._read_json_object(config_path, label="config.json")
        permissions = config.setdefault("permissions", {})
        if not isinstance(permissions, dict):
            raise ValueError("profile config.json permissions must be an object")
        allowed = permissions.setdefault("allow", [])
        if not isinstance(allowed, list) or not all(
            isinstance(value, str) for value in allowed
        ):
            raise ValueError(
                "profile config.json permissions.allow must be a string list"
            )
        required = "mcp(pitbench/run_command)"
        if required not in allowed:
            allowed.append(required)
        self.write_private_json(config_path, config)

    def run(self, arguments: list[str], payload: dict) -> int:
        self.validate_binaries()
        mcp_url, agy_arguments = self._parse_command(arguments)
        auth_token = self.json_document(payload, "auth_token_json")
        if (
            not isinstance(auth_token, dict)
            or not isinstance(auth_token.get("auth_method"), str)
            or not isinstance(auth_token.get("token"), dict)
        ):
            raise ValueError("auth_token_json has an invalid Antigravity token schema")
        settings = self.json_document(payload, "settings_json")
        if not isinstance(settings, dict):
            raise ValueError("settings_json must contain an object")
        auth = (settings.get("security") or {}).get("auth") or {}
        selected_type = auth.get("selectedType")
        if not isinstance(selected_type, str) or not selected_type:
            raise ValueError("settings_json has no authentication selection")
        minimal_settings = {"security": {"auth": {"selectedType": selected_type}}}
        allow_hooks = self.allow_hooks(payload)
        proxy_env = self.proxy_environment(payload)

        with tempfile.TemporaryDirectory(
            prefix="pitbench-antigravity-container-"
        ) as root_str:
            root = Path(root_str)
            workspace = root / "workspace"
            workspace.mkdir(mode=0o700)
            gemini_home = root / ".gemini"
            gemini_config = gemini_home / "config"
            gemini_config.mkdir(parents=True, mode=0o700)
            self.copy_profile(gemini_config, allow_hooks=allow_hooks)
            recording_hooks = mcp_url is not None and self.configure_recording(
                gemini_config / "hooks.json"
            )
            if mcp_url is not None:
                self._configure_pitbench(gemini_config, mcp_url)
            self.write_private_json(gemini_home / "settings.json", minimal_settings)
            self.write_private_json(
                gemini_home / "antigravity-cli" / "antigravity-oauth-token",
                auth_token,
            )
            self.write_private_json(
                gemini_home / "antigravity-cli" / "settings.json",
                {
                    "trustedWorkspaces": [str(workspace)],
                    "toolPermission": "request-review",
                    "permissions": {"allow": ["mcp(pitbench/run_command)"]},
                },
            )
            env = self.environment(root, proxy_env)
            env["XDG_CACHE_HOME"] = str(root / "cache")
            env["XDG_CONFIG_HOME"] = str(root / "config")
            env["XDG_DATA_HOME"] = str(root / "data")
            env.pop("DBUS_SESSION_BUS_ADDRESS", None)
            env.pop("XDG_RUNTIME_DIR", None)
            result = subprocess.run(
                [str(self.binary), *agy_arguments],
                cwd=workspace,
                env=env,
                stdin=subprocess.DEVNULL,
                check=False,
                **({"timeout": 25} if mcp_url is None else {}),
            )
            if recording_hooks:
                self.finalize_recording()
            return result.returncode


ENTRYPOINTS: dict[str, type[ContainerEntrypoint]] = {
    entrypoint.provider: entrypoint
    for entrypoint in (CodexEntrypoint, AntigravityEntrypoint)
}


def main(arguments: list[str] | None = None) -> int:
    values = sys.argv[1:] if arguments is None else arguments
    if not values or values[0] not in ENTRYPOINTS:
        raise ValueError("runner requires a supported provider")
    entrypoint = ENTRYPOINTS[values[0]]()
    provider_arguments = values[1:]
    if provider_arguments == ["--self-test"]:
        return entrypoint.self_test()
    return entrypoint.run(provider_arguments, entrypoint.payload())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"pitbench-container-runner: {error}", file=sys.stderr)
        raise SystemExit(2)
