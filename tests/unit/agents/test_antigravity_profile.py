from pathlib import Path

import pytest
from typer.testing import CliRunner

from pitbench.cli.main import app
from pitbench.harness.agents.antigravity_profile import (
    AntigravityProfile,
    AntigravityProfileError,
)


def _profile(root: Path, *, allow_hooks: bool = False) -> Path:
    config = root / "gemini-config"
    config.mkdir(parents=True)
    (root / "profile.yaml").write_text(
        "\n".join(
            [
                'schema_version: "1.0"',
                "name: test-agy",
                "gemini_config: gemini-config",
                f"allow_hooks: {str(allow_hooks).lower()}",
                "",
            ]
        )
    )
    (config / "config.json").write_text('{"plugins":{}}\n')
    return root


def test_antigravity_profile_hash_changes_with_content_and_mode(
    tmp_path: Path,
) -> None:
    root = _profile(tmp_path / "profile")
    first = AntigravityProfile.load(root)
    config = root / "gemini-config/config.json"
    config.write_text('{"plugins":{"custom":{"enabled":true}}}\n')
    second = AntigravityProfile.load(root)
    config.chmod(0o755)
    third = AntigravityProfile.load(root)

    assert first.sha256 != second.sha256
    assert second.sha256 != third.sha256


def test_antigravity_profile_requires_explicit_hook_trust(tmp_path: Path) -> None:
    root = _profile(tmp_path / "profile")
    (root / "gemini-config/hooks.json").write_text("{}\n")

    with pytest.raises(AntigravityProfileError, match="allow_hooks is false"):
        AntigravityProfile.load(root)


def test_antigravity_profile_rejects_credentials_and_symlinks(
    tmp_path: Path,
) -> None:
    root = _profile(tmp_path / "profile")
    token = root / "gemini-config/antigravity-oauth-token"
    token.write_text("{}")
    with pytest.raises(AntigravityProfileError, match="must not contain credentials"):
        AntigravityProfile.load(root)

    token.unlink()
    (root / "gemini-config/link").symlink_to("config.json")
    with pytest.raises(AntigravityProfileError, match="symbolic links"):
        AntigravityProfile.load(root)


def test_profiles_cli_initializes_antigravity_overlay(tmp_path: Path) -> None:
    destination = tmp_path / "custom"
    result = CliRunner().invoke(
        app,
        [
            "profiles",
            "init",
            "custom-agy",
            "--agent",
            "antigravity",
            "--output",
            str(destination),
            "--allow-hooks",
        ],
    )

    assert result.exit_code == 0, result.output
    profile = AntigravityProfile.load(destination)
    assert profile.allow_hooks is True
    assert not (destination / "gemini-config/antigravity-oauth-token").exists()
    assert f"sha256={profile.sha256}" in result.output


def test_profiles_cli_infers_antigravity_profile_type(tmp_path: Path) -> None:
    root = _profile(tmp_path / "profile")
    result = CliRunner().invoke(app, ["profiles", "validate", str(root)])

    assert result.exit_code == 0, result.output
    assert "OK antigravity test-agy sha256=" in result.output
