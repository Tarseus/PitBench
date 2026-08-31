from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from pitbench.harness.agents.codex_profile import CodexProfile


@dataclass
class CodexContainerRunner:
    image: str
    codex_binary: Path
    profile: CodexProfile | None = None

    CONTAINER_RUNNER = Path("/opt/pitbench/codex_container_runner.py")
    CONTAINER_CODEX = Path("/opt/pitbench/bin/codex")
    CONTAINER_CODE_MODE_HOST = Path("/opt/pitbench/bin/codex-code-mode-host")
    CONTAINER_PROFILE = Path("/opt/pitbench/profile")

    def __post_init__(self) -> None:
        self.codex_binary = self.codex_binary.expanduser().resolve()
        self.code_mode_host = self.codex_binary.parent / "codex-code-mode-host"
        self.runner_script = (
            Path(__file__).with_name("codex_container_runner.py").resolve()
        )
        self.image_id: str | None = None

    @staticmethod
    def _mount(source: Path, destination: Path) -> str:
        value = str(source)
        if "," in value:
            raise ValueError(f"Docker bind source cannot contain a comma: {source}")
        return f"type=bind,src={value},dst={destination},readonly"

    def validate_inputs(self) -> None:
        if shutil.which("docker") is None:
            raise RuntimeError("Docker CLI is required for the container runner")
        for binary in (self.codex_binary, self.code_mode_host):
            if not binary.is_file() or not os.access(binary, os.X_OK):
                raise RuntimeError(f"Codex executable is unavailable: {binary}")
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
            "--mount",
            self._mount(self.codex_binary, self.CONTAINER_CODEX),
            "--mount",
            self._mount(self.code_mode_host, self.CONTAINER_CODE_MODE_HOST),
            "--mount",
            self._mount(self.runner_script, self.CONTAINER_RUNNER),
        ]
        if self.profile is not None:
            command.extend(
                [
                    "--mount",
                    self._mount(self.profile.codex_home, self.CONTAINER_PROFILE),
                ]
            )
        command.extend([self.image, "python3", str(self.CONTAINER_RUNNER)])
        return command

    def _inspect_image(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", self.image],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def prepare(self, *, pull: bool = True) -> dict[str, object]:
        self.validate_inputs()
        inspected = self._inspect_image()
        if inspected.returncode != 0:
            if not pull:
                raise RuntimeError(
                    f"Codex runner image is not present locally: {self.image}"
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
                    f"could not pull Codex runner image {self.image}: {detail}"
                )
            inspected = self._inspect_image()
        if inspected.returncode != 0:
            detail = inspected.stderr.strip() or inspected.stdout.strip()
            raise RuntimeError(
                f"could not inspect Codex runner image {self.image}: {detail}"
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
                f"Codex container runner returned invalid self-test: {check.stderr}"
            ) from error
        if check.returncode != 0:
            detail = check.stderr.strip() or check.stdout.strip()
            raise RuntimeError(f"Codex container runner self-test failed: {detail}")
        if payload.get("docker_socket_present") or payload.get("docker_socket_access"):
            raise RuntimeError("Codex container runner can see the Docker socket")
        if bool(payload.get("profile_mounted")) != (self.profile is not None):
            raise RuntimeError("Codex container runner profile mount is inconsistent")
        return self.metadata()

    def metadata(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "backend": "container",
            "runner_image": self.image,
            "runner_image_id": self.image_id,
            "codex_binary": str(self.codex_binary),
            "profile": self.profile.metadata() if self.profile is not None else None,
        }
