from __future__ import annotations

import hashlib
import json
import re
import stat
from dataclasses import dataclass
from pathlib import Path

import yaml


class AntigravityProfileError(ValueError):
    pass


@dataclass(frozen=True)
class AntigravityProfile:
    """A versioned, non-secret ~/.gemini/config overlay."""

    root: Path
    name: str
    gemini_config: Path
    allow_hooks: bool
    sha256: str
    file_count: int
    size_bytes: int

    _RESERVED_NAMES = {
        "antigravity-oauth-token",
        "google_accounts.json",
        "oauth_creds.json",
    }

    @classmethod
    def load(cls, path: Path) -> AntigravityProfile:
        root = path.expanduser().resolve()
        manifest_path = root / "profile.yaml"
        if not manifest_path.is_file():
            raise AntigravityProfileError(
                f"missing Antigravity profile manifest: {manifest_path}"
            )
        try:
            payload = yaml.safe_load(manifest_path.read_text()) or {}
        except (OSError, yaml.YAMLError) as error:
            raise AntigravityProfileError(
                f"invalid Antigravity profile manifest: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise AntigravityProfileError(
                "Antigravity profile manifest must be a mapping"
            )
        supported = {"schema_version", "name", "gemini_config", "allow_hooks"}
        unknown = sorted(set(payload) - supported)
        if unknown:
            raise AntigravityProfileError(
                f"unknown Antigravity profile fields: {', '.join(unknown)}"
            )
        if str(payload.get("schema_version")) != "1.0":
            raise AntigravityProfileError(
                "Antigravity profile schema_version must be '1.0'"
            )
        name = payload.get("name")
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", name
        ):
            raise AntigravityProfileError(
                "Antigravity profile name must contain only letters, digits, '.', '_', or '-'"
            )
        relative_config = Path(str(payload.get("gemini_config", "gemini-config")))
        if relative_config.is_absolute() or ".." in relative_config.parts:
            raise AntigravityProfileError(
                "gemini_config must stay inside the profile directory"
            )
        raw_config = root / relative_config
        if raw_config.is_symlink():
            raise AntigravityProfileError("gemini_config must not be a symbolic link")
        gemini_config = raw_config.resolve()
        if root not in gemini_config.parents or not gemini_config.is_dir():
            raise AntigravityProfileError(
                f"Antigravity profile config is not a directory: {raw_config}"
            )
        allow_hooks = payload.get("allow_hooks", False)
        if not isinstance(allow_hooks, bool):
            raise AntigravityProfileError("allow_hooks must be true or false")

        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "name": name,
                    "gemini_config": relative_config.as_posix(),
                    "allow_hooks": allow_hooks,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        file_count = 0
        size_bytes = 0
        for entry in sorted(gemini_config.rglob("*")):
            relative = entry.relative_to(gemini_config).as_posix()
            if entry.is_symlink():
                raise AntigravityProfileError(
                    f"Antigravity profiles must not contain symbolic links: {relative}"
                )
            mode = entry.stat().st_mode
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode):
                raise AntigravityProfileError(
                    f"Antigravity profiles may contain only regular files: {relative}"
                )
            if entry.name in cls._RESERVED_NAMES:
                raise AntigravityProfileError(
                    f"Antigravity profiles must not contain credentials: {relative}"
                )
            if entry.name == "hooks.json" and not allow_hooks:
                raise AntigravityProfileError(
                    "Antigravity profile contains hooks.json but allow_hooks is false"
                )
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
            gemini_config=gemini_config,
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
