"""Shared subprocess lifecycle for isolated command-line coding agents."""

from __future__ import annotations

import os
import signal
import subprocess
import threading

from pitbench.harness.agents.base_agent import BaseAgent


class ManagedProcessAgent(BaseAgent):
    credential_environment_keys: tuple[str, ...] = ()
    _PROXY_KEYS = {
        "ALL_PROXY",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "all_proxy",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._process: subprocess.Popen[str] | None = None
        self._process_lock = threading.Lock()
        self._cancelled = threading.Event()

    @classmethod
    def _subscription_env(cls) -> dict[str, str]:
        env = os.environ.copy()
        for key in (*cls.credential_environment_keys, *cls._PROXY_KEYS):
            env.pop(key, None)
        env["NO_PROXY"] = "127.0.0.1,localhost"
        env["no_proxy"] = env["NO_PROXY"]
        return env

    def _runtime_env(self) -> dict[str, str]:
        env = self._subscription_env()
        proxy_url = getattr(self, "_proxy_url", None)
        if proxy_url:
            for key in (
                "ALL_PROXY",
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "all_proxy",
                "http_proxy",
                "https_proxy",
            ):
                env[key] = proxy_url
        return env

    def _runner_command_prefix(self) -> list[str]:
        if self._runner_backend == "container":
            if self._container_runner is None:
                raise RuntimeError(
                    f"{self.name()} container runner has not passed preflight"
                )
            return self._container_runner.command_prefix()
        return [
            "sudo",
            "-n",
            "-u",
            self._runner_user,
            "--",
            str(self._runner_path),
        ]

    def _set_process(self, process: subprocess.Popen[str] | None) -> None:
        with self._process_lock:
            self._process = process

    @staticmethod
    def _terminate_process(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def cancel(self) -> None:
        super().cancel()
        cancelled = getattr(self, "_cancelled", None)
        if cancelled is not None:
            cancelled.set()
        process_lock = getattr(self, "_process_lock", None)
        if process_lock is None:
            return
        with process_lock:
            process = getattr(self, "_process", None)
        if process is not None:
            self._terminate_process(process)
