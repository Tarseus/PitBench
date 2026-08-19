from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import uvicorn
from docker.models.containers import Container
from mcp.server.fastmcp import FastMCP

from fceval.agents.agent_name import AgentName
from fceval.agents.base_agent import AgentResult, BaseAgent
from fceval.agents.failure_mode import FailureMode
from fceval.terminal.tmux_session import TmuxSession


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
        raw_stdout: bytes | str | None
        raw_stderr: bytes | str | None
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
                "timeout "
                "status. Combine dependent shell steps into one command when needed."
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
            name=f"pitbench-codex-mcp-{port}",
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


class CodexMCPAgent(BaseAgent):
    """Run host Codex as a no-Docker user against an offline task container."""

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
    _PROMPT = """You are improving a solver in an isolated PitBench task.

You MUST use only the pitbench MCP run_command tool for repository inspection,
editing, building, profiling, and testing. The host working directory is empty and is
not the task repository. Do not use the host shell or web access. Start by running
`pwd`, `git status --short`, and `pitbench inspect` through the MCP tool. Keep all
changes inside the task repository and verify them with the visible PitBench commands.

Task:
{instruction}
"""

    @staticmethod
    def name() -> str:
        return AgentName.CODEX.value

    def __init__(
        self,
        model_name: str,
        *,
        codex_binary: str = "codex",
        codex_auth_path: str = "~/.codex/auth.json",
        runner_path: str = "/usr/local/libexec/pitbench-codex-runner",
        runner_user: str = "pitbench-codex",
        container_workdir: str = "/workspace/repo",
        timeout_sec: float = 3500.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._model_name = model_name.split("/")[-1]
        self._codex_binary = codex_binary
        self._codex_auth_path = Path(codex_auth_path).expanduser()
        self._runner_path = Path(runner_path)
        self._runner_user = runner_user
        self._container_workdir = container_workdir
        self._timeout_sec = timeout_sec
        self._process: subprocess.Popen[str] | None = None
        self._process_lock = threading.Lock()
        self._cancelled = threading.Event()

    def _resolved_codex_binary(self) -> str:
        resolved = shutil.which(self._codex_binary)
        if resolved is None:
            raise RuntimeError(
                f"Codex CLI was not found: {self._codex_binary}. "
                "Install it on the host."
            )
        return resolved

    @staticmethod
    def _subscription_env() -> dict[str, str]:
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)
        no_proxy = [
            item.strip()
            for item in env.get("NO_PROXY", env.get("no_proxy", "")).split(",")
            if item.strip()
        ]
        for host in ("127.0.0.1", "localhost"):
            if host not in no_proxy:
                no_proxy.append(host)
        env["NO_PROXY"] = ",".join(no_proxy)
        env["no_proxy"] = env["NO_PROXY"]
        return env

    def _preflight(self, codex_binary: str, env: dict[str, str]) -> None:
        result = subprocess.run(
            [codex_binary, "login", "status"],
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        status = f"{result.stdout}\n{result.stderr}"
        if result.returncode != 0 or "Logged in using ChatGPT" not in status:
            raise RuntimeError(
                "Host Codex is not logged in with ChatGPT. Run `codex login` first."
            )
        if not self._runner_path.is_file():
            raise RuntimeError(
                "The isolated Codex runner is not installed. Run "
                "`sudo scripts/install-codex-runner.sh`."
            )
        runner_check = subprocess.run(
            [
                "sudo",
                "-n",
                "-u",
                self._runner_user,
                "--",
                str(self._runner_path),
                "--self-test",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if runner_check.returncode != 0:
            detail = runner_check.stderr.strip() or runner_check.stdout.strip()
            raise RuntimeError(f"Isolated Codex runner preflight failed: {detail}")
        try:
            check = json.loads(runner_check.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "Isolated Codex runner returned invalid status"
            ) from error
        if check.get("docker_socket_access") or "docker" in check.get("groups", []):
            raise RuntimeError("Isolated Codex runner still has Docker access")

    def _build_command(
        self,
        *,
        mcp_url: str,
        instruction: str,
    ) -> list[str]:
        return [
            "sudo",
            "-n",
            "-u",
            self._runner_user,
            "--",
            str(self._runner_path),
            "exec",
            "--ignore-user-config",
            "--ephemeral",
            "--json",
            "--color",
            "never",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--model",
            self._model_name,
            "-c",
            f"mcp_servers.pitbench.url={json.dumps(mcp_url)}",
            "-c",
            'mcp_servers.pitbench.default_tools_approval_mode="approve"',
            "--",
            self._PROMPT.format(instruction=self._render_instruction(instruction)),
        ]

    def _runner_payload(self, env: dict[str, str]) -> str:
        try:
            auth_json = self._codex_auth_path.read_text()
            json.loads(auth_json)
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"Invalid host Codex login file: {self._codex_auth_path}"
            ) from error
        proxy_env = {key: env[key] for key in self._PROXY_KEYS if env.get(key)}
        return json.dumps({"auth_json": auth_json, "proxy_env": proxy_env})

    @staticmethod
    def _parse_usage(output: str) -> tuple[int, int]:
        input_tokens = 0
        output_tokens = 0
        for line in output.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "turn.completed":
                continue
            usage = event.get("usage") or {}
            input_tokens += int(usage.get("input_tokens", 0))
            output_tokens += int(usage.get("output_tokens", 0))
        return input_tokens, output_tokens

    @staticmethod
    def _has_successful_mcp_call(output: str) -> bool:
        for line in output.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "item.completed":
                continue
            item = event.get("item") or {}
            if (
                item.get("type") == "mcp_tool_call"
                and item.get("status") == "completed"
                and not item.get("error")
            ):
                return True
        return False

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
        self._cancelled.set()
        with self._process_lock:
            process = self._process
        if process is not None:
            self._terminate_process(process)

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
        portkey_metadata: dict[str, str] | None = None,
        portkey_trace_id: str | None = None,
    ) -> AgentResult:
        del portkey_metadata, portkey_trace_id
        codex_binary = self._resolved_codex_binary()
        env = self._subscription_env()
        self._preflight(codex_binary, env)
        runner_payload = self._runner_payload(env)
        self._cancelled.clear()
        if logging_dir is not None:
            logging_dir.mkdir(parents=True, exist_ok=True)
        terminal_log = (
            logging_dir / "mcp-terminal.jsonl" if logging_dir is not None else None
        )
        terminal = TaskTerminal(
            session.container,
            workdir=self._container_workdir,
            log_path=terminal_log,
        )
        stdout = ""
        stderr = ""
        return_code = 1
        with LoopbackMCPServer(terminal) as mcp_server:
            if mcp_server.url is None:
                raise RuntimeError("PitBench MCP server URL is unavailable")
            command = self._build_command(
                mcp_url=mcp_server.url,
                instruction=instruction,
            )
            process = subprocess.Popen(
                command,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            self._set_process(process)
            try:
                stdout, stderr = process.communicate(
                    input=runner_payload,
                    timeout=self._timeout_sec,
                )
                return_code = process.returncode
            except subprocess.TimeoutExpired:
                self._cancelled.set()
                self._terminate_process(process)
                stdout, stderr = process.communicate()
                return_code = process.returncode
            finally:
                self._set_process(None)
        if logging_dir is not None:
            (logging_dir / "codex.jsonl").write_text(stdout)
            (logging_dir / "codex.stderr.log").write_text(stderr)
        input_tokens, output_tokens = self._parse_usage(stdout)
        if self._cancelled.is_set():
            failure_mode = FailureMode.AGENT_TIMEOUT
        elif return_code != 0 or not self._has_successful_mcp_call(stdout):
            failure_mode = FailureMode.UNKNOWN_AGENT_ERROR
        else:
            failure_mode = FailureMode.NONE
        return AgentResult(
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            total_cost=0.0,
            failure_mode=failure_mode,
        )
