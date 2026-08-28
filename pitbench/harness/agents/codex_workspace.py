from __future__ import annotations

import os
import subprocess
import tarfile
import uuid
from contextlib import AbstractContextManager
from io import BytesIO
from pathlib import Path

from docker.models.containers import Container
from docker.models.networks import Network

import docker
from pitbench.harness.agents.codex_profile import CodexProfile


class CodexWorkspaceRuntime(AbstractContextManager["CodexWorkspaceRuntime"]):
    """Stage Codex in a task container and attach a single-task Docker network."""

    def __init__(
        self,
        *,
        container: Container,
        codex_binary: Path,
        profile: CodexProfile | None,
    ) -> None:
        self.container = container
        self.codex_binary = codex_binary.expanduser().resolve()
        self.code_mode_host = self.codex_binary.parent / "codex-code-mode-host"
        self.bubblewrap = self.codex_binary.parent.parent / "codex-resources/bwrap"
        self.profile = profile
        suffix = uuid.uuid4().hex[:12]
        self.root = f"/opt/pitbench/codex-workspace-{suffix}"
        self.container_codex = f"{self.root}/bin/codex"
        self.codex_home = f"{self.root}/home"
        self.config_profile_name = f"pitbench_workspace_{suffix}"
        self.permission_profile_name = f"pitbench_workspace_relay_{suffix}"
        self.network_name = f"pitbench-codex-{suffix}"
        self.network: Network | None = None
        self.none_network: Network | None = None
        self._restore_none_network = False
        self.gateway_ip: str | None = None
        self.container_ip: str | None = None
        self.relay_ip: str | None = None

    def _docker_cp(self, source: Path | str, destination: str) -> None:
        result = subprocess.run(
            ["docker", "cp", str(source), f"{self.container.id}:{destination}"],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"could not stage Codex in task container: {detail}")

    def _stage_enforcement_profile(self, relay_ip: str) -> None:
        content = (
            f'default_permissions = "{self.permission_profile_name}"\n\n'
            "[features]\n"
            "network_proxy = true\n\n"
            f"[permissions.{self.permission_profile_name}]\n"
            'extends = ":workspace"\n\n'
            f"[permissions.{self.permission_profile_name}.network]\n"
            "enabled = true\n\n"
            f"[permissions.{self.permission_profile_name}.network.domains]\n"
            f'"{relay_ip}" = "allow"\n'
            '"127.0.0.1" = "allow"\n'
            '"localhost" = "allow"\n'
        ).encode()
        archive = BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as bundle:
            info = tarfile.TarInfo(f"{self.config_profile_name}.config.toml")
            info.mode = 0o600
            info.size = len(content)
            bundle.addfile(info, BytesIO(content))
        archive.seek(0)
        if not self.container.put_archive(self.codex_home, archive.read()):
            raise RuntimeError("could not stage the PitBench Codex permission profile")

    def configure_relay(self, relay_ip: str) -> None:
        if not relay_ip:
            raise ValueError("relay sidecar address is required")
        self._stage_enforcement_profile(relay_ip)
        self.relay_ip = relay_ip

    def _stage(self) -> None:
        for binary in (self.codex_binary, self.code_mode_host, self.bubblewrap):
            if not binary.is_file() or not os.access(binary, os.X_OK):
                raise RuntimeError(f"Codex executable is unavailable: {binary}")
        created = self.container.exec_run(
            [
                "mkdir",
                "-p",
                f"{self.root}/bin",
                f"{self.root}/codex-resources",
                self.codex_home,
            ],
            demux=True,
        )
        if created.exit_code != 0:
            raise RuntimeError("could not create Codex runtime inside task container")
        self._docker_cp(self.codex_binary, self.container_codex)
        self._docker_cp(self.code_mode_host, f"{self.root}/bin/codex-code-mode-host")
        self._docker_cp(self.bubblewrap, f"{self.root}/codex-resources/bwrap")
        if self.profile is not None:
            self._docker_cp(f"{self.profile.codex_home}/.", self.codex_home)
        auth_check = self.container.exec_run(
            ["test", "!", "-e", f"{self.codex_home}/auth.json"], demux=True
        )
        if auth_check.exit_code != 0:
            raise RuntimeError("Codex workspace profile attempted to stage auth.json")

    def _connect_network(self) -> None:
        client = docker.from_env()
        self.network = client.networks.create(
            self.network_name,
            driver="bridge",
            internal=True,
            check_duplicate=True,
            labels={"org.pitbench.purpose": "codex-model-relay"},
        )
        self.container.reload()
        attached = self.container.attrs["NetworkSettings"]["Networks"]
        if "none" in attached:
            self.none_network = client.networks.get("none")
            self.none_network.disconnect(self.container, force=True)
            self._restore_none_network = True
        self.network.connect(self.container)
        self.network.reload()
        self.container.reload()
        ipam = self.network.attrs.get("IPAM", {}).get("Config", [])
        endpoint = self.container.attrs["NetworkSettings"]["Networks"].get(
            self.network.name, {}
        )
        self.gateway_ip = ipam[0].get("Gateway") if ipam else None
        self.container_ip = endpoint.get("IPAddress")
        if not self.gateway_ip or not self.container_ip:
            raise RuntimeError("could not resolve the Codex relay-only network")

    def metadata(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "backend": "workspace",
            "container_id": self.container.id,
            "container_image": self.container.image.id,
            "codex_binary": str(self.codex_binary),
            "network_internal": True,
            "shell_network": "relay-only",
            "relay_ip": self.relay_ip,
            "permission_profile": self.permission_profile_name,
            "unified_exec": True,
            "profile": self.profile.metadata() if self.profile else None,
        }

    def command_prefix(self) -> list[str]:
        return [
            "docker",
            "exec",
            "-e",
            f"CODEX_HOME={self.codex_home}",
            "-e",
            f"HOME={self.root}",
            self.container.id,
            self.container_codex,
        ]

    def __enter__(self) -> CodexWorkspaceRuntime:
        try:
            self._stage()
            self._connect_network()
        except Exception as error:
            self.__exit__(type(error), error, error.__traceback__)
            raise
        return self

    def _remove_staged_runtime(self) -> None:
        result = self.container.exec_run(
            ["rm", "-rf", "--", self.root],
            demux=True,
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"could not remove staged Codex runtime from {self.root}: "
                f"{result.output}"
            )

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: object | None = None,
    ) -> None:
        del traceback
        if self.network is not None:
            try:
                self.network.disconnect(self.container, force=True)
            except Exception:
                pass
            try:
                self.network.remove()
            except Exception:
                pass
        if self._restore_none_network and self.none_network is not None:
            try:
                self.none_network.connect(self.container)
            except Exception:
                pass
        try:
            self._remove_staged_runtime()
        except Exception as cleanup_error:
            if exc_type is None:
                raise
            if exc_value is not None:
                exc_value.add_note(
                    f"Codex runtime cleanup also failed: {cleanup_error}"
                )
