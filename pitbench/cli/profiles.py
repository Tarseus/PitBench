from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated

import typer
import yaml

from pitbench.harness.agents.antigravity_profile import AntigravityProfile
from pitbench.harness.agents.codex_profile import CodexProfile

profiles_app = typer.Typer(help="Create and validate reproducible agent profiles.")


def _profile_kind(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"codex", "antigravity"}:
        raise typer.BadParameter(
            "must be 'codex' or 'antigravity'", param_hint="--agent"
        )
    return normalized


@profiles_app.command("init")
def init_agent_profile(
    name: Annotated[str, typer.Argument(help="Profile name")],
    agent: Annotated[
        str,
        typer.Option(help="Agent profile type: codex or antigravity"),
    ] = "codex",
    output: Annotated[
        Path | None,
        typer.Option(help="Destination; defaults to agent-profiles/<name>"),
    ] = None,
    allow_hooks: Annotated[
        bool,
        typer.Option(help="Allow this trusted profile to run lifecycle hooks"),
    ] = False,
) -> None:
    """Create an empty agent profile without copying credentials or sessions."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise typer.BadParameter(
            "use only letters, digits, '.', '_', or '-'", param_hint="NAME"
        )
    destination = (output or Path("agent-profiles") / name).expanduser().resolve()
    if destination.exists():
        raise typer.BadParameter(f"destination already exists: {destination}")
    kind = _profile_kind(agent)
    if kind == "codex":
        overlay = destination / "codex-home"
        overlay.mkdir(parents=True)
        (destination / "profile.yaml").write_text(
            "\n".join(
                [
                    'schema_version: "1.0"',
                    f'name: "{name}"',
                    "codex_home: codex-home",
                    f"allow_hooks: {str(allow_hooks).lower()}",
                    "",
                ]
            )
        )
        (overlay / "config.toml").write_text(
            "# Put reproducible Codex configuration in this CODEX_HOME overlay.\n"
            "# Installed plugins, skills, hooks, and local MCP config may live here.\n"
            "# OAuth auth.json is reserved; PitBench supplies model access separately.\n"
        )
        profile = CodexProfile.load(destination)
    else:
        overlay = destination / "gemini-config"
        overlay.mkdir(parents=True)
        (destination / "profile.yaml").write_text(
            "\n".join(
                [
                    'schema_version: "1.0"',
                    f'name: "{name}"',
                    "gemini_config: gemini-config",
                    f"allow_hooks: {str(allow_hooks).lower()}",
                    "",
                ]
            )
        )
        (overlay / "config.json").write_text("{}\n")
        profile = AntigravityProfile.load(destination)
    typer.echo(f"Created {kind} profile {profile.name} at {destination}")
    typer.echo(f"sha256={profile.sha256}")


@profiles_app.command("validate")
def validate_agent_profile(
    path: Annotated[Path, typer.Argument(help="Agent profile directory")],
    agent: Annotated[
        str | None,
        typer.Option(help="Agent profile type; inferred from profile.yaml by default"),
    ] = None,
) -> None:
    kind = _profile_kind(agent) if agent is not None else None
    if kind is None:
        try:
            manifest = yaml.safe_load((path.expanduser() / "profile.yaml").read_text())
        except Exception as error:
            raise typer.BadParameter(
                f"cannot read profile.yaml: {error}", param_hint="PATH"
            ) from error
        if not isinstance(manifest, dict):
            raise typer.BadParameter(
                "profile.yaml must contain a mapping", param_hint="PATH"
            )
        if "codex_home" in manifest:
            kind = "codex"
        elif "gemini_config" in manifest:
            kind = "antigravity"
        else:
            raise typer.BadParameter(
                "cannot infer profile type; pass --agent", param_hint="PATH"
            )
    profile = (
        CodexProfile.load(path) if kind == "codex" else AntigravityProfile.load(path)
    )
    typer.echo(
        f"OK {kind} {profile.name} sha256={profile.sha256} "
        f"files={profile.file_count} bytes={profile.size_bytes}"
    )
