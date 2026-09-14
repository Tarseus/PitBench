"""Shared isolated control-plane containers for native coding-agent CLIs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from pitbench.harness.agents.profiles import (
    AntigravityProfile,
    CodexProfile,
    ProfileOverlay,
)


class IsolatedContainerRunner(ABC):
    container_profile = Path("/opt/pitbench/profile")
    runner_filename = "container_entrypoint.py"
    container_runner = Path("/opt/pitbench/container_entrypoint.py")
    binary_config_key: str
    binary_default: str
    credential_hint: str
    profile_class: type[ProfileOverlay]

    def __init__(
        self,
        image: str,
        profile: ProfileOverlay | None = None,
        recording_dir: Path | None = None,
    ) -> None:
        self.image = image
        self.profile = profile
        self.recording_dir = recording_dir
        self.runner_script = Path(__file__).with_name(self.runner_filename).resolve()
        self.image_id: str | None = None

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @property
    @abstractmethod
    def provider_id(self) -> str: ...

    @property
    @abstractmethod
    def binary_mounts(self) -> list[tuple[Path, Path]]: ...

    @property
    @abstractmethod
    def binary_metadata(self) -> tuple[str, str]: ...

    @classmethod
    def from_binary(
        cls,
        *,
        image: str,
        binary: Path,
        profile: ProfileOverlay | None,
        recording_dir: Path | None = None,
    ) -> IsolatedContainerRunner:
        return cls(image, binary, profile, recording_dir)

    @staticmethod
    def mount(source: Path, destination: Path, *, readonly: bool = True) -> str:
        value = str(source)
        if "," in value:
            raise ValueError(f"Docker bind source cannot contain a comma: {source}")
        option = ",readonly" if readonly else ""
        return f"type=bind,src={value},dst={destination}{option}"

    def validate_inputs(self) -> None:
        if shutil.which("docker") is None:
            raise RuntimeError("Docker CLI is required for the container runner")
        for binary, _ in self.binary_mounts:
            if not binary.is_file() or not os.access(binary, os.X_OK):
                raise RuntimeError(
                    f"{self.provider_name} executable is unavailable: {binary}"
                )
        if not self.runner_script.is_file():
            raise RuntimeError(
                f"container runner script is missing: {self.runner_script}"
            )

    def command_prefix(self) -> list[str]:
        self.validate_inputs()
        command = [
            "docker",
            "run",
            "--rm",
            "-i",
            "--init",
            "--network",
            "host",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "512",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=1g,mode=1777",
        ]
        for source, destination in self.binary_mounts:
            command.extend(["--mount", self.mount(source, destination)])
        command.extend(
            ["--mount", self.mount(self.runner_script, self.container_runner)]
        )
        if self.profile is not None:
            command.extend(
                [
                    "--mount",
                    self.mount(self.profile.content_root, self.container_profile),
                ]
            )
        if self.recording_dir is not None:
            command.extend(
                [
                    "--mount",
                    self.mount(
                        Path(__file__).parents[1] / "utils/recording.py",
                        Path("/opt/pitbench/recording.py"),
                    ),
                    "--mount",
                    self.mount(
                        self.recording_dir,
                        Path("/opt/pitbench/recording"),
                        readonly=False,
                    ),
                ]
            )
        command.extend(
            [self.image, "python3", str(self.container_runner), self.provider_id]
        )
        return command

    def inspect_image(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", self.image],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def prepare(self, *, pull: bool = True) -> dict[str, object]:
        self.validate_inputs()
        inspected = self.inspect_image()
        if inspected.returncode != 0:
            if not pull:
                raise RuntimeError(
                    f"{self.provider_name} runner image is not present locally: {self.image}"
                )
            pulled = subprocess.run(
                ["docker", "pull", self.image],
                capture_output=True,
                text=True,
                timeout=600,
                check=False,
            )
            if pulled.returncode != 0:
                detail = pulled.stderr.strip() or pulled.stdout.strip()
                raise RuntimeError(
                    f"could not pull {self.provider_name} runner image {self.image}: {detail}"
                )
            inspected = self.inspect_image()
        if inspected.returncode != 0:
            detail = inspected.stderr.strip() or inspected.stdout.strip()
            raise RuntimeError(
                f"could not inspect {self.provider_name} runner image {self.image}: {detail}"
            )
        self.image_id = inspected.stdout.strip()
        check = subprocess.run(
            [*self.command_prefix(), "--self-test"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        try:
            payload = json.loads(check.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"{self.provider_name} container runner returned invalid self-test: {check.stderr}"
            ) from error
        if check.returncode != 0:
            detail = check.stderr.strip() or check.stdout.strip()
            raise RuntimeError(
                f"{self.provider_name} container runner self-test failed: {detail}"
            )
        if payload.get("docker_socket_present") or payload.get("docker_socket_access"):
            raise RuntimeError(
                f"{self.provider_name} container runner can see the Docker socket"
            )
        if bool(payload.get("profile_mounted")) != (self.profile is not None):
            raise RuntimeError(
                f"{self.provider_name} container runner profile mount is inconsistent"
            )
        return self.metadata()

    def metadata(self) -> dict[str, object]:
        binary_key, binary_value = self.binary_metadata
        return {
            "schema_version": "1.0",
            "backend": "container",
            "runner_image": self.image,
            "runner_image_id": self.image_id,
            binary_key: binary_value,
            "profile": self.profile.metadata() if self.profile is not None else None,
        }


class CodexContainerRunner(IsolatedContainerRunner):
    provider_name = "Codex"
    provider_id = "codex"
    binary_config_key = "codex_binary"
    binary_default = "codex"
    credential_hint = "Install Codex and run `codex login`."
    profile_class = CodexProfile
    container_codex = Path("/opt/pitbench/bin/codex")
    container_code_mode_host = Path("/opt/pitbench/bin/codex-code-mode-host")

    def __init__(
        self,
        image: str,
        codex_binary: Path,
        profile: CodexProfile | None = None,
        recording_dir: Path | None = None,
    ) -> None:
        self.codex_binary = codex_binary.expanduser().resolve()
        self.code_mode_host = self.codex_binary.parent / "codex-code-mode-host"
        super().__init__(image, profile, recording_dir)

    @property
    def binary_mounts(self) -> list[tuple[Path, Path]]:
        return [
            (self.codex_binary, self.container_codex),
            (self.code_mode_host, self.container_code_mode_host),
        ]

    @property
    def binary_metadata(self) -> tuple[str, str]:
        return "codex_binary", str(self.codex_binary)


class AntigravityContainerRunner(IsolatedContainerRunner):
    provider_name = "Antigravity"
    provider_id = "antigravity"
    binary_config_key = "agy_binary"
    binary_default = "agy"
    credential_hint = "Install agy and complete Google sign-in."
    profile_class = AntigravityProfile
    container_agy = Path("/opt/pitbench/bin/agy")

    def __init__(
        self,
        image: str,
        agy_binary: Path,
        profile: AntigravityProfile | None = None,
        recording_dir: Path | None = None,
    ) -> None:
        self.agy_binary = agy_binary.expanduser().resolve()
        super().__init__(image, profile, recording_dir)

    @property
    def binary_mounts(self) -> list[tuple[Path, Path]]:
        return [(self.agy_binary, self.container_agy)]

    @property
    def binary_metadata(self) -> tuple[str, str]:
        return "agy_binary", str(self.agy_binary)
