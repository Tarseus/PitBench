from pathlib import Path

import pytest
from typer.testing import CliRunner

from pitbench.cli.main import app
from pitbench.harness.agents.codex_profile import CodexProfile, CodexProfileError


def _profile(root: Path, *, allow_hooks: bool = False) -> Path:
    home = root / "codex-home"
    home.mkdir(parents=True)
    (root / "profile.yaml").write_text(
        "\n".join(
            [
                'schema_version: "1.0"',
                "name: test-profile",
                "codex_home: codex-home",
                f"allow_hooks: {str(allow_hooks).lower()}",
                "",
            ]
        )
    )
    (home / "config.toml").write_text('model_reasoning_effort = "high"\n')
    return root


def test_profile_hash_changes_with_content_and_executable_mode(tmp_path: Path) -> None:
    root = _profile(tmp_path / "profile")
    first = CodexProfile.load(root)
    assert first.file_count == 1

    config = root / "codex-home/config.toml"
    config.write_text('model_reasoning_effort = "xhigh"\n')
    second = CodexProfile.load(root)
    config.chmod(0o755)
    third = CodexProfile.load(root)

    assert first.sha256 != second.sha256
    assert second.sha256 != third.sha256


def test_profile_rejects_credentials_and_symbolic_links(tmp_path: Path) -> None:
    root = _profile(tmp_path / "profile")
    auth = root / "codex-home/auth.json"
    auth.write_text("{}")
    with pytest.raises(CodexProfileError, match="must not contain auth.json"):
        CodexProfile.load(root)

    auth.unlink()
    (root / "codex-home/link").symlink_to("config.toml")
    with pytest.raises(CodexProfileError, match="symbolic links"):
        CodexProfile.load(root)


def test_profiles_cli_initializes_non_secret_overlay(tmp_path: Path) -> None:
    destination = tmp_path / "custom"

    result = CliRunner().invoke(
        app,
        [
            "profiles",
            "init",
            "custom-codex",
            "--output",
            str(destination),
            "--allow-hooks",
        ],
    )

    assert result.exit_code == 0, result.output
    profile = CodexProfile.load(destination)
    assert profile.allow_hooks is True
    assert not (destination / "codex-home/auth.json").exists()
    assert f"sha256={profile.sha256}" in result.output


def test_profiles_cli_preserves_yaml_keyword_as_codex_name(tmp_path: Path) -> None:
    destination = tmp_path / "keyword"

    result = CliRunner().invoke(
        app,
        ["profiles", "init", "true", "--output", str(destination)],
    )

    assert result.exit_code == 0, result.output
    assert CodexProfile.load(destination).name == "true"


def test_profiles_cli_validates_profile(tmp_path: Path) -> None:
    root = _profile(tmp_path / "profile")

    result = CliRunner().invoke(app, ["profiles", "validate", str(root)])

    assert result.exit_code == 0, result.output
    assert "OK codex test-profile sha256=" in result.output
