"""Shared tmux-session and livestream lifecycle for terminal backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pitbench.harness.terminal.tmux_session import TmuxSession
from pitbench.harness.utils.livestreamer import Livestreamer
from pitbench.harness.utils.logger import logger


class TerminalSessionManager(ABC):
    def __init__(
        self,
        *,
        sessions_logs_path: Path | None,
        commands_path: Path | None,
        livestream: bool,
        disable_recording: bool,
        history_limit: int | None,
    ) -> None:
        self._sessions_logs_path = sessions_logs_path
        self._commands_path = commands_path
        self._livestream = livestream
        self._disable_recording = disable_recording
        self._history_limit = history_limit
        self._sessions: dict[str, TmuxSession] = {}
        self._logger = logger.getChild(type(self).__module__)
        self._livestreamer = None
        self.container = None
        if livestream:
            if sessions_logs_path is None:
                raise ValueError("sessions_logs_path is required to livestream.")
            self._livestreamer = Livestreamer()

    @property
    @abstractmethod
    def container_session_logs_path(self) -> Path: ...

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def save_container_image(
        self,
        output_path: Path | None = None,
        snapshot_s3_key: str | None = None,
    ) -> None: ...

    @abstractmethod
    def copy_to_container(
        self,
        paths: list[Path] | Path,
        container_dir: str | None = None,
        container_filename: str | None = None,
    ) -> None: ...

    @property
    def sessions(self) -> dict[str, TmuxSession]:
        return self._sessions

    @property
    def sessions_logs_path(self) -> Path | None:
        return self._sessions_logs_path

    @property
    def commands_path(self) -> Path | None:
        return self._commands_path

    @property
    def disable_recording(self) -> bool:
        return self._disable_recording

    @property
    def history_limit(self) -> int | None:
        return self._history_limit

    @property
    def livestreamer(self) -> Livestreamer | None:
        return self._livestreamer

    def create_session(
        self,
        session_name: str,
        is_active_stream: bool = False,
        as_configured_user: bool = True,
        recording_filename: str | None = None,
    ) -> TmuxSession:
        if self.container is None:
            raise ValueError("Container not started. Run start() first.")
        if session_name in self._sessions:
            raise ValueError(f"Session {session_name} already exists")
        user = (
            self.container.attrs["Config"].get("User", "")
            if as_configured_user
            else "root"
        )
        session = TmuxSession(
            session_name=session_name,
            container=self.container,
            commands_path=self._commands_path,
            disable_recording=self._disable_recording,
            user=user,
            recording_filename=recording_filename,
            history_limit=self._history_limit,
        )
        self._sessions[session_name] = session
        if is_active_stream:
            self.set_active_stream(session_name)
        session.start()
        return session

    def get_session(self, session_name: str) -> TmuxSession:
        if session_name not in self._sessions:
            raise ValueError(f"Session {session_name} does not exist")
        return self._sessions[session_name]

    def close_session(self, session_name: str) -> None:
        session = self._sessions.pop(session_name, None)
        if session is None:
            return
        try:
            session.stop()
        except Exception as error:
            self._logger.warning(
                "Failed to stop tmux session %s: %s", session_name, error
            )

    def close_sessions(self) -> None:
        for session_name in list(self._sessions):
            self.close_session(session_name)
        if self._livestreamer is not None:
            self._livestreamer.stop()
        self._sessions.clear()

    def set_active_stream(self, session_name: str) -> None:
        if self._livestreamer is None or self._sessions_logs_path is None:
            return
        if session_name not in self._sessions:
            raise ValueError(f"Session '{session_name}' does not exist")
        session = self._sessions[session_name]
        self._livestreamer.change_livestream_path(
            self._sessions_logs_path
            / session.logging_path.relative_to(self.container_session_logs_path)
        )
