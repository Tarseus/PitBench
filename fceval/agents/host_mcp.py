from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path
from typing import Any

import uvicorn
from docker.models.containers import Container
from mcp.server.fastmcp import FastMCP


class TaskTerminal:
    """A bounded command surface for exactly one task container."""

    MAX_TIMEOUT_SEC = 900.0
    MAX_OUTPUT_CHARS = 50_000

    def __init__(
        self,
        container: Container,
        *,
        workdir: str,
        log_path: Path | None = None,
    ) -> None:
        self._container = container
        self._workdir = workdir
        self._log_path = log_path
        self._log_lock = threading.Lock()

    @staticmethod
    def _decode(output: bytes | str | None) -> str:
        if output is None:
            return ""
        if isinstance(output, bytes):
            return output.decode(errors="replace")
        return output

    @classmethod
    def _truncate(cls, output: str) -> tuple[str, bool]:
        if len(output) <= cls.MAX_OUTPUT_CHARS:
            return output, False
        omitted = len(output) - cls.MAX_OUTPUT_CHARS
        return (
            f"[{omitted} earlier characters truncated]\n"
            + output[-cls.MAX_OUTPUT_CHARS :],
            True,
        )

    def _log(self, payload: dict[str, Any]) -> None:
        if self._log_path is None:
            return
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_lock, self._log_path.open("a") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")

    def run_command(
        self,
        command: str,
        timeout_sec: float = 120.0,
    ) -> dict[str, Any]:
        """Run one shell command in the isolated task repository."""
        if not command.strip():
            raise ValueError("command must not be empty")
        timeout = min(max(float(timeout_sec), 0.1), self.MAX_TIMEOUT_SEC)
        started = time.monotonic()
        result = self._container.exec_run(
            [
                "timeout",
                "--signal=TERM",
                "--kill-after=5s",
                f"{timeout}s",
                "bash",
                "-lc",
                command,
            ],
            workdir=self._workdir,
            demux=True,
        )
        elapsed = time.monotonic() - started
        if isinstance(result.output, tuple):
            raw_stdout, raw_stderr = result.output
        else:
            raw_stdout, raw_stderr = result.output, None
        stdout, stdout_truncated = self._truncate(self._decode(raw_stdout))
        stderr, stderr_truncated = self._truncate(self._decode(raw_stderr))
        payload = {
            "command": command,
            "exit_code": int(result.exit_code),
            "timed_out": result.exit_code == 124,
            "elapsed_sec": round(elapsed, 3),
            "stdout": stdout,
            "stderr": stderr,
            "output_truncated": stdout_truncated or stderr_truncated,
        }
        self._log(payload)
        return payload


class LoopbackMCPServer:
    """Expose a TaskTerminal over a per-trial loopback-only MCP endpoint."""

    def __init__(self, terminal: TaskTerminal) -> None:
        self._terminal = terminal
        self._socket: socket.socket | None = None
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self.url: str | None = None

    def _build_app(self):
        mcp = FastMCP(
            "pitbench-terminal",
            instructions=(
                "Use run_command for all repository inspection, editing, building, "
                "profiling, and testing. Commands run inside the isolated task "
                "container."
            ),
            stateless_http=True,
            json_response=True,
            log_level="WARNING",
        )

        @mcp.tool(
            name="run_command",
            description=(
                "Run a shell command in /workspace/repo inside the isolated PitBench "
                "task container. Returns exit code, stdout, stderr, timing, and "
                "timeout status. Combine dependent shell steps into one command "
                "when needed."
            ),
            structured_output=True,
        )
        def run_command(command: str, timeout_sec: float = 120.0) -> dict[str, Any]:
            return self._terminal.run_command(command, timeout_sec)

        return mcp.streamable_http_app()

    def __enter__(self) -> LoopbackMCPServer:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        port = listener.getsockname()[1]
        config = uvicorn.Config(
            self._build_app(),
            log_level="warning",
            access_log=False,
            lifespan="on",
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(
            target=server.run,
            kwargs={"sockets": [listener]},
            name=f"pitbench-host-mcp-{port}",
            daemon=True,
        )
        self._socket = listener
        self._server = server
        self._thread = thread
        self.url = f"http://127.0.0.1:{port}/mcp"
        thread.start()
        deadline = time.monotonic() + 10.0
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not server.started:
            self.__exit__(None, None, None)
            raise RuntimeError("PitBench MCP server failed to start")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive() and self._server is not None:
                self._server.force_exit = True
                self._thread.join(timeout=2.0)
        if self._socket is not None:
            self._socket.close()
