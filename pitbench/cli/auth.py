"""Provision coding-agent credentials through their native CLIs and local stores."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated

import typer
import yaml

from pitbench.cli.doctor import CheckStatus, _credential_check, _runner_check
from pitbench.cli.evaluate_config import EvaluationConfig, resolve_repository_path

auth_app = typer.Typer(
    help="Connect native agent logins to PitBench without running tasks."
)

# Use the desktop's Python/Gio, which need not be installed in PitBench's venv.
# Only the agy item is searched. Its secret travels over a captured pipe.
_READ_KEYRING = """
import json
from gi.repository import Gio, GLib
bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
def call(path, interface, method, args):
    return bus.call_sync('org.freedesktop.secrets', path, interface, method,
        args, None, Gio.DBusCallFlags.NONE, 10000, None).unpack()
unlocked, locked = call('/org/freedesktop/secrets',
    'org.freedesktop.Secret.Service', 'SearchItems',
    GLib.Variant('(a{ss})', ({'service': 'gemini', 'username': 'antigravity'},)))
if locked:
    raise RuntimeError('Unlock the desktop keyring and retry.')
if not unlocked:
    print('null')
else:
    if len(unlocked) != 1:
        raise RuntimeError('Multiple agy credentials; use --auth-file.')
    _, session = call('/org/freedesktop/secrets',
        'org.freedesktop.Secret.Service', 'OpenSession',
        GLib.Variant('(sv)', ('plain', GLib.Variant('s', ''))))
    try:
        secret, = call(unlocked[0], 'org.freedesktop.Secret.Item', 'GetSecret',
            GLib.Variant('(o)', (session,)))
        print(json.dumps(json.loads(bytes(secret[2]))))
    finally:
        call(session, 'org.freedesktop.Secret.Session', 'Close', None)
"""


def read_agy_keyring() -> dict | None:
    """Read the existing Linux desktop login without prompting for another login."""
    if not sys.platform.startswith("linux"):
        raise ValueError(
            "Automatic keyring import supports Linux; use --auth-file on this host."
        )
    if (
        not os.environ.get("DBUS_SESSION_BUS_ADDRESS")
        and not Path(f"/run/user/{os.getuid()}/bus").exists()
    ):
        return None
    secret_tool = shutil.which("secret-tool")
    command = (
        [secret_tool, "lookup", "service", "gemini", "username", "antigravity"]
        if secret_tool
        else ["/usr/bin/python3", "-c", _READ_KEYRING]
    )
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=35)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError(
            "Could not access the agy keyring; unlock it or use --auth-file."
        ) from error
    if (
        secret_tool
        and result.returncode == 1
        and not result.stdout
        and not result.stderr.strip()
    ):
        return None
    if result.returncode:
        # Do not echo a credential helper's stdout/stderr into logs.
        raise ValueError(
            "Could not read the agy keyring. Unlock it and ensure secret-tool or "
            "/usr/bin/python3 with python3-gi is available, or use --auth-file."
        )
    try:
        value = json.loads(result.stdout)
    except ValueError as error:
        raise ValueError("The agy keyring did not return a JSON credential.") from error
    return value


def _read_credential(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read a JSON credential from {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Credential must contain a JSON object: {path}")
    return value


def _write_private(path: Path, contents: str) -> None:
    """Publish a private file atomically, without exposing a partially written token."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError(f"Private output must not be a symbolic link: {path}")
    if (
        shutil.which("git")
        and subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", str(path.resolve())],
            cwd=path.parent,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    ):
        raise ValueError(
            f"Refusing to write private settings or credentials to a tracked file: {path}"
        )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            os.fchmod(stream.fileno(), 0o600)
            stream.write(contents)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class CodexAuth:
    name = "codex"
    binary_key = "codex_binary"
    auth_key = "codex_auth_path"
    default_binary = "codex"
    default_backend = "workspace"

    @staticmethod
    def default_auth_file() -> Path:
        return (
            Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))).expanduser()
            / "auth.json"
        )

    @staticmethod
    def validate(value: dict) -> None:
        tokens = value.get("tokens") if isinstance(value, dict) else None
        if not isinstance(tokens, dict) or not all(
            isinstance(tokens.get(key), str) and tokens[key]
            for key in ("access_token", "refresh_token", "account_id")
        ):
            raise ValueError(
                "Expected a Codex ChatGPT auth.json credential, including refresh_token and account_id."
            )

    @staticmethod
    def import_keyring(root: Path) -> Path | None:
        # Official Codex can provision file storage during native login. No
        # undocumented OS-keyring identifiers are assumed here.
        return None

    @staticmethod
    def login_command(binary: str, device_auth: bool) -> list[str]:
        command = [binary, "-c", 'cli_auth_credentials_store="file"', "login"]
        return [*command, "--device-auth"] if device_auth else command


class AntigravityAuth:
    name = "antigravity"
    binary_key = "agy_binary"
    auth_key = "auth_token_path"
    default_binary = "agy"
    default_backend = "container"

    @staticmethod
    def default_auth_file() -> Path:
        return Path.home() / ".gemini/antigravity-cli/antigravity-oauth-token"

    @staticmethod
    def validate(value: dict) -> None:
        token = value.get("token") if isinstance(value, dict) else None
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("auth_method"), str)
            or not value["auth_method"]
            or not isinstance(token, dict)
            or not all(
                isinstance(token.get(key), str) and token[key]
                for key in ("access_token", "refresh_token")
            )
        ):
            raise ValueError(
                "Expected an agy credential with auth_method and token; Gemini CLI oauth_creds.json is a different format."
            )

    @staticmethod
    def import_keyring(root: Path) -> Path | None:
        credential = read_agy_keyring()
        if credential is None:
            return None
        AntigravityAuth.validate(credential)
        destination = root / ".pitbench/credentials/antigravity-oauth-token"
        _write_private(destination.parent / ".gitignore", "*\n")
        _write_private(destination, json.dumps(credential) + "\n")
        return destination

    @staticmethod
    def login_command(binary: str, device_auth: bool) -> list[str]:
        if device_auth:
            raise ValueError("--device-auth is supported only for Codex.")
        return [binary]


AUTH_PROVIDERS = {
    "codex": CodexAuth,
    "agy": AntigravityAuth,
    "antigravity": AntigravityAuth,
}


def _context(agent: str, root: Path | None, config: Path | None):
    if agent not in AUTH_PROVIDERS:
        raise ValueError("Agent must be codex, agy, or antigravity.")
    repository_root = (root or Path.cwd()).resolve()
    config_path = resolve_repository_path(
        repository_root, config or Path("config/evaluate.local.yaml")
    )
    payload = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
    payload = payload or {}
    EvaluationConfig.model_validate(payload)
    provider = AUTH_PROVIDERS[agent]
    values = payload.get("agents", {}).get(provider.name, {}).copy()
    values.setdefault("runner_backend", provider.default_backend)
    for key in (provider.auth_key, "profile_path", "settings_path"):
        if values.get(key):
            values[key] = str(
                resolve_repository_path(repository_root, Path(values[key]))
            )
    return repository_root, config_path, payload, provider, values


@auth_app.command("login")
def login(
    agent: Annotated[str, typer.Argument(help="codex, agy, or antigravity")],
    auth_file: Annotated[
        Path | None, typer.Option(help="Use an existing exported credential")
    ] = None,
    binary: Annotated[str | None, typer.Option(help="Native CLI executable")] = None,
    proxy_url: Annotated[
        str | None, typer.Option(help="Proxy used by login and evaluation")
    ] = None,
    model: Annotated[
        str | None, typer.Option(help="Default model for this agent")
    ] = None,
    device_auth: Annotated[
        bool, typer.Option(help="Use Codex device login when login is needed")
    ] = False,
    force_login: Annotated[
        bool, typer.Option(help="Start native login even if a credential exists")
    ] = False,
    non_interactive: Annotated[
        bool, typer.Option(help="Only connect an existing login; never open a login UI")
    ] = False,
    config: Annotated[
        Path | None, typer.Option(help="Machine-local evaluation YAML")
    ] = None,
    root: Annotated[Path | None, typer.Option(help="PitBench repository root")] = None,
) -> None:
    """Reuse/import a login, or invoke the native CLI, and save local connection settings."""
    try:
        repository_root, config_path, payload, provider, values = _context(
            agent, root, config
        )
        if device_auth and provider is not CodexAuth:
            raise ValueError("--device-auth is supported only for Codex.")
        if force_login and (non_interactive or auth_file):
            raise ValueError(
                "--force-login cannot be combined with --non-interactive or --auth-file."
            )
        requested_binary = (
            binary or values.get(provider.binary_key) or provider.default_binary
        )
        resolved_binary = shutil.which(requested_binary)
        if resolved_binary is None and binary is None:
            resolved_binary = shutil.which(provider.default_binary)
        if resolved_binary is None:
            raise ValueError(
                f"Install the native {provider.default_binary} CLI or pass --binary."
            )
        values[provider.binary_key] = resolved_binary
        values.setdefault("runner_backend", provider.default_backend)
        if proxy_url is not None:
            values["proxy_url"] = proxy_url
        if model is not None:
            values["model_name"] = model

        def discover() -> Path | None:
            if auth_file is not None:
                path = resolve_repository_path(repository_root, auth_file)
                provider.validate(_read_credential(path))
                return path
            candidates = [provider.default_auth_file()]
            if values.get(provider.auth_key):
                candidates.insert(0, Path(values[provider.auth_key]))
            for path in candidates:
                if path.is_file():
                    provider.validate(_read_credential(path))
                    return path
            return provider.import_keyring(repository_root)

        source = None if force_login else discover()
        if source is None:
            if non_interactive or not sys.stdin.isatty():
                raise ValueError(
                    f"No readable {provider.default_binary} credential. Run this command in a terminal "
                    "to complete native login, or supply --auth-file."
                )
            command = provider.login_command(resolved_binary, device_auth)
            typer.echo(
                f"Opening native {provider.default_binary} login; finish sign-in and exit its CLI to return here."
            )
            environment = os.environ.copy()
            if values.get("proxy_url"):
                for key in (
                    "HTTP_PROXY",
                    "HTTPS_PROXY",
                    "ALL_PROXY",
                    "http_proxy",
                    "https_proxy",
                    "all_proxy",
                ):
                    environment[key] = str(values["proxy_url"])
            result = subprocess.run(command, env=environment, check=False)
            if result.returncode:
                raise ValueError(
                    f"Native {provider.default_binary} login exited with code {result.returncode}."
                )
            # A forced native login must use its newly provisioned credential.
            if force_login:
                values.pop(provider.auth_key, None)
            source = discover()
            if source is None:
                raise ValueError(
                    "Native login did not provide a readable credential; use --auth-file."
                )
        values[provider.auth_key] = str(source.resolve())
        if provider is AntigravityAuth:
            settings = Path(
                values.get("settings_path")
                or Path.home() / ".gemini/antigravity-cli/settings.json"
            )
            if not settings.is_file():
                settings = (
                    repository_root / ".pitbench/credentials/antigravity-settings.json"
                )
                _write_private(settings, "{}\n")
            values["settings_path"] = str(settings.resolve())
        payload.setdefault("agents", {})[provider.name] = values
        EvaluationConfig.model_validate(payload)
        _write_private(config_path, yaml.safe_dump(payload, sort_keys=False))
    except (ValueError, OSError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"Connected {provider.name} credential: {source}")
    typer.echo(f"Saved local settings: {config_path}")
    typer.echo(
        f"Check the connection with: pitbench auth status {agent} --config {config_path}"
    )


@auth_app.command("status")
def status(
    agent: Annotated[str, typer.Argument(help="codex, agy, or antigravity")],
    config: Annotated[
        Path | None, typer.Option(help="Machine-local evaluation YAML")
    ] = None,
    root: Annotated[Path | None, typer.Option(help="PitBench repository root")] = None,
) -> None:
    """Check configured authentication and the runner without requiring task assets."""
    try:
        _, _, _, provider, values = _context(agent, root, config)
        credential, credentials_ready = _credential_check(provider.name, values)
        runner, runner_ready = _runner_check(provider.name, values)
    except (ValueError, OSError) as error:
        raise typer.BadParameter(str(error)) from error
    for check in (credential, runner):
        typer.echo(f"{check.status.value} {check.name}: {check.detail}")
        if check.recovery:
            typer.echo(f"     {check.recovery}")
    if not credentials_ready or not runner_ready:
        raise typer.Exit(1)
    typer.echo(
        f"{CheckStatus.PASS.value} {provider.name}: authentication and runner checks passed; no task was started."
    )
