from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from pitbench.harness.terminal.docker_compose_manager import DockerComposeManager
from pitbench.harness.terminal.session_manager import TerminalSessionManager


class Terminal(TerminalSessionManager):
    def __init__(
        self,
        client_container_name: str,
        client_image_name: str,
        docker_compose_path: Path,
        docker_image_name_prefix: str | None = None,
        sessions_logs_path: Path | None = None,
        agent_logs_path: Path | None = None,
        agent_model_name: str | None = None,
        commands_path: Path | None = None,
        no_rebuild: bool = False,
        cleanup: bool = False,
        livestream: bool = False,
        disable_recording: bool = False,
        history_limit: int | None = None,
        cpuset_cpus: str | None = None,
        nested_sandbox: bool = False,
    ):
        self._client_container_name = client_container_name
        self._docker_image_name_prefix = docker_image_name_prefix
        self._docker_compose_path = docker_compose_path
        self._agent_logs_path = agent_logs_path
        super().__init__(
            sessions_logs_path=sessions_logs_path,
            commands_path=commands_path,
            livestream=livestream,
            disable_recording=disable_recording,
            history_limit=history_limit,
        )

        self._compose_manager = DockerComposeManager(
            client_container_name=client_container_name,
            client_image_name=client_image_name,
            docker_image_name_prefix=docker_image_name_prefix,
            docker_compose_path=docker_compose_path,
            no_rebuild=no_rebuild,
            cleanup=cleanup,
            sessions_logs_path=sessions_logs_path,
            agent_logs_path=agent_logs_path,
            agent_model_name=agent_model_name,
            cpuset_cpus=cpuset_cpus,
            nested_sandbox=nested_sandbox,
        )

    @property
    def container_session_logs_path(self) -> Path:
        return Path(DockerComposeManager.CONTAINER_SESSION_LOGS_PATH)

    def start(self) -> None:
        self.container = self._compose_manager.start()

    def stop(self) -> None:
        self.close_sessions()
        self._compose_manager.stop()

    def save_container_image(
        self,
        output_path: Path | None = None,
        snapshot_s3_key: str | None = None,
    ) -> None:
        """Save the current container as a gzipped docker image tarball."""
        if output_path is None:
            raise ValueError("local snapshots require output_path")
        self._compose_manager.save_container_image(output_path)

    def copy_to_container(
        self,
        paths: list[Path] | Path,
        container_dir: str | None = None,
        container_filename: str | None = None,
    ):
        self._compose_manager.copy_to_container(
            container=self.container,
            paths=paths,
            container_dir=container_dir,
            container_filename=container_filename,
        )


@contextmanager
def spin_up_terminal(
    client_container_name: str,
    client_image_name: str,
    docker_compose_path: Path,
    docker_image_name_prefix: str | None = None,
    sessions_logs_path: Path | None = None,
    agent_logs_path: Path | None = None,
    agent_model_name: str | None = None,
    commands_path: Path | None = None,
    no_rebuild: bool = False,
    cleanup: bool = False,
    livestream: bool = False,
    disable_recording: bool = False,
    cpuset_cpus: str | None = None,
    nested_sandbox: bool = False,
) -> Generator[Terminal, None, None]:
    terminal = Terminal(
        client_container_name=client_container_name,
        client_image_name=client_image_name,
        docker_image_name_prefix=docker_image_name_prefix,
        docker_compose_path=docker_compose_path,
        sessions_logs_path=sessions_logs_path,
        agent_logs_path=agent_logs_path,
        agent_model_name=agent_model_name,
        commands_path=commands_path,
        no_rebuild=no_rebuild,
        cleanup=cleanup,
        livestream=livestream,
        disable_recording=disable_recording,
        cpuset_cpus=cpuset_cpus,
        nested_sandbox=nested_sandbox,
    )

    try:
        terminal.start()
        yield terminal
    finally:
        terminal.stop()
