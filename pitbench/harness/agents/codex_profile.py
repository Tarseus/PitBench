from __future__ import annotations

import hashlib
import json
import re
import stat
from dataclasses import dataclass
from pathlib import Path

import yaml


class CodexProfileError(ValueError):
    pass


@dataclass(frozen=True)
class CodexProfile:
    """A versioned, non-secret CODEX_HOME overlay for one agent scaffold."""

    root: Path
    name: str
    codex_home: Path
    allow_hooks: bool
    sha256: str
    file_count: int
    size_bytes: int

    @classmethod
    def load(cls, path: Path) -> CodexProfile:
        root = path.expanduser().resolve()
        manifest_path = root / "profile.yaml"
        if not manifest_path.is_file():
            raise CodexProfileError(f"missing Codex profile manifest: {manifest_path}")
        try:
            payload = yaml.safe_load(manifest_path.read_text()) or {}
        except (OSError, yaml.YAMLError) as error:
            raise CodexProfileError(
                f"invalid Codex profile manifest: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise CodexProfileError("Codex profile manifest must be a mapping")
        supported = {"schema_version", "name", "codex_home", "allow_hooks"}
        unknown = sorted(set(payload) - supported)
        if unknown:
            raise CodexProfileError(
                f"unknown Codex profile fields: {', '.join(unknown)}"
            )
        if str(payload.get("schema_version")) != "1.0":
            raise CodexProfileError("Codex profile schema_version must be '1.0'")
        name = payload.get("name")
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", name
        ):
            raise CodexProfileError(
                "Codex profile name must contain only letters, digits, '.', '_', or '-'"
            )
        relative_home = Path(str(payload.get("codex_home", "codex-home")))
        if relative_home.is_absolute() or ".." in relative_home.parts:
            raise CodexProfileError("codex_home must stay inside the profile directory")
        raw_home = root / relative_home
        if raw_home.is_symlink():
            raise CodexProfileError("codex_home must not be a symbolic link")
        codex_home = raw_home.resolve()
        if root not in codex_home.parents or not codex_home.is_dir():
            raise CodexProfileError(
                f"Codex profile home is not a directory: {raw_home}"
            )
        if (codex_home / "auth.json").exists():
            raise CodexProfileError(
                "Codex profiles must not contain auth.json; OAuth is injected at runtime"
            )
        allow_hooks = payload.get("allow_hooks", False)
        if not isinstance(allow_hooks, bool):
            raise CodexProfileError("allow_hooks must be true or false")

        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "name": name,
                    "codex_home": relative_home.as_posix(),
                    "allow_hooks": allow_hooks,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        file_count = 0
        size_bytes = 0
        for entry in sorted(codex_home.rglob("*")):
            relative = entry.relative_to(codex_home).as_posix()
            if entry.is_symlink():
                raise CodexProfileError(
                    f"Codex profiles must not contain symbolic links: {relative}"
                )
            mode = entry.stat().st_mode
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode):
                raise CodexProfileError(
                    f"Codex profiles may contain only regular files: {relative}"
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
            codex_home=codex_home,
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
