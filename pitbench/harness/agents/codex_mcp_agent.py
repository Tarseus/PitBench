from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import threading
from pathlib import Path

from pitbench.harness.agents.agent_name import AgentName
from pitbench.harness.agents.base_agent import AgentResult, BaseAgent
from pitbench.harness.agents.codex_container import CodexContainerRunner
from pitbench.harness.agents.codex_profile import CodexProfile
from pitbench.harness.agents.failure_mode import FailureMode
from pitbench.harness.agents.host_mcp import LoopbackMCPServer, TaskTerminal
from pitbench.harness.terminal.tmux_session import TmuxSession


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

Treat existing tests as immutable specifications. Never change a test to accommodate
an implementation change. The protected paths reported by `pitbench inspect` must
not be modified. Preserve behavior across every supported problem feature, not only
the visible development instances. Before finishing, run `pitbench validate` and
`pitbench check`, inspect `git status --short`, and remove generated output and
temporary helper files. The final candidate must contain only intentional source
changes.

Task:
{instruction}
"""
    _CONTROL_PLANE_PROMPT = (
        "Reply with exactly PITBENCH_CODEX_CONTROL_PLANE_OK. Do not use tools."
    )

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
        runner_backend: str = "host",
        container_runner_image: str = "python:3.13-slim-bookworm",
        profile_path: str | None = None,
        container_workdir: str = "/workspace/repo",
        timeout_sec: float = 3500.0,
        control_plane_preflight: bool = True,
        control_plane_timeout_sec: float = 120.0,
        proxy_url: str | None = None,
        reasoning_effort: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._model_name = model_name.split("/")[-1]
        self._codex_binary = codex_binary
        self._codex_auth_path = Path(codex_auth_path).expanduser()
        self._runner_path = Path(runner_path)
        self._runner_user = runner_user
        if runner_backend not in {"host", "container"}:
            raise ValueError("runner_backend must be 'host' or 'container'")
        if runner_backend == "host" and profile_path is not None:
            raise ValueError("Codex profiles require runner_backend=container")
        self._runner_backend = runner_backend
        self._container_runner_image = container_runner_image
        self._profile = (
            CodexProfile.load(Path(profile_path)) if profile_path is not None else None
        )
        self._container_runner: CodexContainerRunner | None = None
        self._runner_metadata: dict[str, object] = {
            "schema_version": "1.0",
            "backend": "host",
            "profile": None,
        }
        self._container_workdir = container_workdir
        self._timeout_sec = timeout_sec
        self._control_plane_preflight = control_plane_preflight
        self._control_plane_timeout_sec = control_plane_timeout_sec
        self._proxy_url = proxy_url
        self._reasoning_effort = reasoning_effort
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
        for key in CodexMCPAgent._PROXY_KEYS:
            env.pop(key, None)
        no_proxy = [
            "127.0.0.1",
            "localhost",
        ]
        env["NO_PROXY"] = ",".join(no_proxy)
        env["no_proxy"] = env["NO_PROXY"]
        return env

    def _runtime_env(self) -> dict[str, str]:
        env = self._subscription_env()
        if self._proxy_url:
            for key in (
                "ALL_PROXY",
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "all_proxy",
                "http_proxy",
                "https_proxy",
            ):
                env[key] = self._proxy_url
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
        if self._runner_backend == "container":
            self._container_runner = CodexContainerRunner(
                image=self._container_runner_image,
                codex_binary=Path(codex_binary),
                profile=self._profile,
            )
            self._runner_metadata = self._container_runner.prepare()
            return
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

    def _runner_command_prefix(self) -> list[str]:
        if self._runner_backend == "container":
            if self._container_runner is None:
                raise RuntimeError("Codex container runner has not passed preflight")
            return self._container_runner.command_prefix()
        return [
            "sudo",
            "-n",
            "-u",
            self._runner_user,
            "--",
            str(self._runner_path),
        ]

    def _codex_exec_prefix(self) -> list[str]:
        command = ["exec"]
        if self._runner_backend == "host":
            command.append("--ignore-user-config")
        command.extend(
            [
                "--ephemeral",
                "--json",
                "--color",
                "never",
                "--sandbox",
                "read-only",
                "--skip-git-repo-check",
                "--model",
                self._model_name,
            ]
        )
        return command

    def _build_command(
        self,
        *,
        mcp_url: str,
        instruction: str,
    ) -> list[str]:
        command = [*self._runner_command_prefix(), *self._codex_exec_prefix()]
        command.extend(self._reasoning_effort_args())
        command.extend(
            [
                "-c",
                f"mcp_servers.pitbench.url={json.dumps(mcp_url)}",
                "-c",
                'mcp_servers.pitbench.default_tools_approval_mode="approve"',
                "--",
                self._PROMPT.format(instruction=self._render_instruction(instruction)),
            ]
        )
        return command

    def _reasoning_effort_args(self) -> list[str]:
        if self._reasoning_effort is None:
            return []
        return [
            "-c",
            f"model_reasoning_effort={json.dumps(self._reasoning_effort)}",
        ]

    def _build_control_plane_command(self) -> list[str]:
        command = [*self._runner_command_prefix(), *self._codex_exec_prefix()]
        command.extend(self._reasoning_effort_args())
        command.extend(["--", self._CONTROL_PLANE_PROMPT])
        return command

    def _runner_payload(self, env: dict[str, str]) -> str:
        try:
            auth_json = self._codex_auth_path.read_text()
            json.loads(auth_json)
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"Invalid host Codex login file: {self._codex_auth_path}"
            ) from error
        proxy_env = {key: env[key] for key in self._PROXY_KEYS if env.get(key)}
        return json.dumps(
            {
                "auth_json": auth_json,
                "proxy_env": proxy_env,
                "allow_hooks": bool(self._profile and self._profile.allow_hooks),
                "profile_sha256": self._profile.sha256 if self._profile else None,
            }
        )

    @staticmethod
    def _has_completed_turn(output: str) -> bool:
        for line in output.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "turn.completed":
                return True
        return False

    @staticmethod
    def _failure_detail(output: str, stderr: str) -> str:
        messages: list[str] = []
        for line in output.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "error" and event.get("message"):
                messages.append(str(event["message"]))
            if event.get("type") == "turn.failed":
                error = event.get("error") or {}
                if error.get("message"):
                    messages.append(str(error["message"]))
        if messages:
            return messages[-1]
        stderr_lines = [line.strip() for line in stderr.splitlines() if line.strip()]
        return stderr_lines[-1] if stderr_lines else "no diagnostic was emitted"

    @staticmethod
    def _timeout_output(output: str | bytes | None) -> str:
        if isinstance(output, bytes):
            return output.decode(errors="replace")
        return output or ""

    def _check_control_plane(
        self,
        runner_payload: str,
        logging_dir: Path | None = None,
    ) -> None:
        process = subprocess.Popen(
            self._build_control_plane_command(),
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
                timeout=self._control_plane_timeout_sec,
            )
        except subprocess.TimeoutExpired as error:
            self._terminate_process(process)
            stdout, stderr = process.communicate()
            if logging_dir is not None:
                (logging_dir / "codex-preflight.jsonl").write_text(
                    stdout or self._timeout_output(error.stdout)
                )
                (logging_dir / "codex-preflight.stderr.log").write_text(
                    stderr or self._timeout_output(error.stderr)
                )
            raise RuntimeError(
                "Codex control-plane preflight timed out after "
                f"{self._control_plane_timeout_sec:g}s. The host runner cannot "
                "reach the Codex backend; check outbound access or proxy settings "
                "before starting benchmark episodes."
            ) from error
        finally:
            self._set_process(None)

        if logging_dir is not None:
            (logging_dir / "codex-preflight.jsonl").write_text(stdout)
            (logging_dir / "codex-preflight.stderr.log").write_text(stderr)
        if process.returncode != 0 or not self._has_completed_turn(stdout):
            detail = self._failure_detail(stdout, stderr)
            raise RuntimeError(
                "Codex control-plane preflight failed. The solver container remains "
                f"offline; host Codex backend access failed: {detail}"
            )

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

    def _update_live_progress(self, line: str, counts: dict[str, int]) -> None:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return
        if event.get("type") != "item.completed":
            return
        item = event.get("item") or {}
        if item.get("type") == "mcp_tool_call":
            counts["mcp_calls"] += 1
        elif item.get("type") == "agent_message":
            counts["messages"] += 1
        else:
            return
        self._report_progress(
            f"Agent: MCP calls {counts['mcp_calls']} · "
            f"messages {counts['messages']}"
        )

    def _stream_process(
        self,
        process: subprocess.Popen[str],
        runner_payload: str,
    ) -> tuple[str, str, int]:
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        counts = {"mcp_calls": 0, "messages": 0}

        def read_stdout() -> None:
            if process.stdout is None:
                return
            for line in process.stdout:
                stdout_lines.append(line)
                self._update_live_progress(line, counts)

        def read_stderr() -> None:
            if process.stderr is None:
                return
            stderr_lines.extend(process.stderr)

        stdout_reader = threading.Thread(target=read_stdout, daemon=True)
        stderr_reader = threading.Thread(target=read_stderr, daemon=True)
        stdout_reader.start()
        stderr_reader.start()
        try:
            if process.stdin is not None:
                try:
                    process.stdin.write(runner_payload)
                except BrokenPipeError:
                    pass
                finally:
                    process.stdin.close()
            process.wait(timeout=self._timeout_sec)
        except subprocess.TimeoutExpired:
            self._cancelled.set()
            self._terminate_process(process)
        finally:
            stdout_reader.join()
            stderr_reader.join()
        return "".join(stdout_lines), "".join(stderr_lines), process.returncode or 0

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
        env = self._runtime_env()
        self._preflight(codex_binary, env)
        runner_payload = self._runner_payload(env)
        self._cancelled.clear()
        if logging_dir is not None:
            logging_dir.mkdir(parents=True, exist_ok=True)
            (logging_dir / "codex-runner.json").write_text(
                json.dumps(self._runner_metadata, indent=2, sort_keys=True) + "\n"
            )
        if self._control_plane_preflight:
            self._check_control_plane(runner_payload, logging_dir)
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
                stdout, stderr, return_code = self._stream_process(
                    process, runner_payload
                )
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
