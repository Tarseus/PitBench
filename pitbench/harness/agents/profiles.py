"""Validated non-secret configuration overlays for coding-agent runtimes."""

from __future__ import annotations

import hashlib
import json
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import yaml


class ProfileError(ValueError):
    pass


@dataclass(frozen=True)
class ProfileOverlay:
    root: Path
    name: str
    content_root: Path
    allow_hooks: bool
    sha256: str
    file_count: int
    size_bytes: int

    provider_name: ClassVar[str]
    content_key: ClassVar[str]
    default_content_directory: ClassVar[str]
    reserved_names: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def validate_file(cls, entry: Path, relative: str, allow_hooks: bool) -> None:
        if entry.name in cls.reserved_names:
            raise ProfileError(
                f"{cls.provider_name} profiles must not contain credentials: {relative}"
            )
        if entry.name == "hooks.json" and not allow_hooks:
            raise ProfileError(
                f"{cls.provider_name} profile contains hooks.json but allow_hooks is false"
            )

    @classmethod
    def load(cls, path: Path):
        root = path.expanduser().resolve()
        manifest_path = root / "profile.yaml"
        if not manifest_path.is_file():
            raise ProfileError(
                f"missing {cls.provider_name} profile manifest: {manifest_path}"
            )
        try:
            payload = yaml.safe_load(manifest_path.read_text()) or {}
        except (OSError, yaml.YAMLError) as error:
            raise ProfileError(
                f"invalid {cls.provider_name} profile manifest: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise ProfileError(
                f"{cls.provider_name} profile manifest must be a mapping"
            )
        supported = {"schema_version", "name", cls.content_key, "allow_hooks"}
        unknown = sorted(set(payload) - supported)
        if unknown:
            raise ProfileError(
                f"unknown {cls.provider_name} profile fields: {', '.join(unknown)}"
            )
        if str(payload.get("schema_version")) != "1.0":
            raise ProfileError(
                f"{cls.provider_name} profile schema_version must be '1.0'"
            )
        name = payload.get("name")
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", name
        ):
            raise ProfileError(
                f"{cls.provider_name} profile name must contain only letters, digits, '.', '_', or '-'"
            )
        relative_content = Path(
            str(payload.get(cls.content_key, cls.default_content_directory))
        )
        if relative_content.is_absolute() or ".." in relative_content.parts:
            raise ProfileError(
                f"{cls.content_key} must stay inside the profile directory"
            )
        raw_content = root / relative_content
        if raw_content.is_symlink():
            raise ProfileError(f"{cls.content_key} must not be a symbolic link")
        content_root = raw_content.resolve()
        if root not in content_root.parents or not content_root.is_dir():
            raise ProfileError(
                f"{cls.provider_name} profile content is not a directory: {raw_content}"
            )
        allow_hooks = payload.get("allow_hooks", False)
        if not isinstance(allow_hooks, bool):
            raise ProfileError("allow_hooks must be true or false")

        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "name": name,
                    cls.content_key: relative_content.as_posix(),
                    "allow_hooks": allow_hooks,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        file_count = 0
        size_bytes = 0
        for entry in sorted(content_root.rglob("*")):
            relative = entry.relative_to(content_root).as_posix()
            if entry.is_symlink():
                raise ProfileError(
                    f"{cls.provider_name} profiles must not contain symbolic links: {relative}"
                )
            mode = entry.stat().st_mode
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode):
                raise ProfileError(
                    f"{cls.provider_name} profiles may contain only regular files: {relative}"
                )
            cls.validate_file(entry, relative, allow_hooks)
            content = entry.read_bytes()
            executable = bool(mode & stat.S_IXUSR)
            digest.update(b"\0path\0" + relative.encode())
            digest.update(b"\0executable\0" + str(executable).encode())
            digest.update(b"\0content\0" + content)
            file_count += 1
            size_bytes += len(content)
        return cls(
            root=root,
            name=name,
            content_root=content_root,
            allow_hooks=allow_hooks,
            sha256=digest.hexdigest(),
            file_count=file_count,
            size_bytes=size_bytes,
        )

    def metadata(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "name": self.name,
            "path": str(self.root),
            "sha256": self.sha256,
            "allow_hooks": self.allow_hooks,
            "file_count": self.file_count,
            "size_bytes": self.size_bytes,
        }

    def trace_files(self) -> dict[str, Path]:
        return {
            "profile.yaml": self.root / "profile.yaml",
            **{
                str(path.relative_to(self.root)): path
                for path in self.content_root.rglob("*")
                if path.is_file()
            },
        }


class CodexProfile(ProfileOverlay):
    provider_name = "Codex"
    content_key = "codex_home"
    default_content_directory = "codex-home"
    reserved_names = frozenset({"auth.json"})

    @property
    def codex_home(self) -> Path:
        return self.content_root


class AntigravityProfile(ProfileOverlay):
    provider_name = "Antigravity"
    content_key = "gemini_config"
    default_content_directory = "gemini-config"
    reserved_names = frozenset(
        {"antigravity-oauth-token", "google_accounts.json", "oauth_creds.json"}
    )

    @property
    def gemini_config(self) -> Path:
        return self.content_root
