from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import threading
from pathlib import Path
from typing import Any

from pitbench.harness.agents.agent_name import AgentName
from pitbench.harness.agents.base_agent import AgentResult, BaseAgent
from pitbench.harness.agents.failure_mode import FailureMode
from pitbench.harness.agents.host_mcp import LoopbackMCPServer, TaskTerminal
from pitbench.harness.terminal.tmux_session import TmuxSession


class AntigravityMCPAgent(BaseAgent):
    """Run host Antigravity as a no-Docker user against an offline task."""

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

You MUST use only the PitBench MCP server's run_command tool for repository
inspection, editing, building, profiling, and testing. The host working directory is
empty and is not the task repository. Do not use native host file, shell, browser,
web, plugin, skill, or subagent tools. Start by running `pwd`, `git status --short`,
and `pitbench inspect` through the PitBench MCP tool. Keep all changes inside the
task repository and verify them with the visible PitBench commands.

Task:
{instruction}
"""

    @staticmethod
    def name() -> str:
        return AgentName.ANTIGRAVITY.value

    def __init__(
        self,
        model_name: str,
        *,
        agy_binary: str = "agy",
        auth_token_path: str = ("~/.gemini/antigravity-cli/antigravity-oauth-token"),
        settings_path: str = "~/.gemini/settings.json",
        runner_path: str = "/usr/local/libexec/pitbench-antigravity-runner",
        runner_user: str = "pitbench-agy",
        container_workdir: str = "/workspace/repo",
        timeout_sec: float = 3500.0,
        print_timeout: str = "55m",
        effort: str | None = None,
        proxy_url: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._model_name = model_name.split("/")[-1]
        self._agy_binary = agy_binary
        self._auth_token_path = Path(auth_token_path).expanduser()
        self._settings_path = Path(settings_path).expanduser()
        self._runner_path = Path(runner_path)
        self._runner_user = runner_user
        self._container_workdir = container_workdir
        self._timeout_sec = timeout_sec
        self._print_timeout = print_timeout
        self._effort = effort
        self._proxy_url = proxy_url
        self._process: subprocess.Popen[str] | None = None
        self._process_lock = threading.Lock()
        self._cancelled = threading.Event()

    def _resolved_agy_binary(self) -> str:
        resolved = shutil.which(self._agy_binary)
        if resolved is None:
            raise RuntimeError(
                f"Antigravity CLI was not found: {self._agy_binary}. "
                "Install `agy` on the host."
            )
        return resolved

    @staticmethod
    def _subscription_env() -> dict[str, str]:
        env = os.environ.copy()
        for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            env.pop(key, None)
        for key in AntigravityMCPAgent._PROXY_KEYS:
            env.pop(key, None)
        no_proxy = ["127.0.0.1", "localhost"]
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

    def _preflight(self, agy_binary: str, env: dict[str, str]) -> None:
        result = subprocess.run(
            [agy_binary, "models"],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Host Antigravity is not ready. Run `agy` and complete Google "
                "sign-in first."
            )
        available_models = {
            line.split("\t", 1)[0].strip()
            for line in result.stdout.splitlines()
            if line.strip()
        }
        if self._model_name not in available_models:
            raise RuntimeError(f"Antigravity model is unavailable: {self._model_name}")
        if not self._runner_path.is_file():
            raise RuntimeError(
                "The isolated Antigravity runner is not installed. Run "
                "`sudo scripts/install-antigravity-runner.sh`."
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
            raise RuntimeError(
                f"Isolated Antigravity runner preflight failed: {detail}"
            )
        try:
            check = json.loads(runner_check.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "Isolated Antigravity runner returned invalid status"
            ) from error
        if check.get("docker_socket_access") or "docker" in check.get("groups", []):
            raise RuntimeError("Isolated Antigravity runner still has Docker access")

    def _build_command(self, *, mcp_url: str, instruction: str) -> list[str]:
        command = [
            "sudo",
            "-n",
            "-u",
            self._runner_user,
            "--",
            str(self._runner_path),
            "run",
            "--mcp-url",
            mcp_url,
            "--",
            "--print",
            self._PROMPT.format(instruction=self._render_instruction(instruction)),
            "--output-format",
            "stream-json",
            "--model",
            self._model_name,
            "--mode",
            "accept-edits",
            "--sandbox",
            "--disable-slash-commands",
            "--print-timeout",
            self._print_timeout,
        ]
        if self._effort is not None:
            command.extend(["--effort", self._effort])
        return command

    @staticmethod
    def _read_json(path: Path, *, label: str) -> str:
        try:
            content = path.read_text()
            json.loads(content)
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Invalid host Antigravity {label}: {path}") from error
        return content

    def _runner_payload(self, env: dict[str, str]) -> str:
        proxy_env = {key: env[key] for key in self._PROXY_KEYS if env.get(key)}
        return json.dumps(
            {
                "auth_token_json": self._read_json(
                    self._auth_token_path, label="OAuth token"
                ),
                "settings_json": self._read_json(
                    self._settings_path, label="authentication settings"
                ),
                "proxy_env": proxy_env,
            }
        )

    @staticmethod
    def _events(output: str):
        for line in output.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                yield event

    @classmethod
    def _parse_usage(cls, output: str) -> tuple[int, int]:
        for event in cls._events(output):
            if event.get("event") != "result":
                continue
            result = event.get("result") or {}
            usage = result.get("usage") or {}
            return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
        return 0, 0

    @classmethod
    def _has_successful_result(cls, output: str) -> bool:
        return any(
            event.get("event") == "result"
            and (event.get("result") or {}).get("status") == "SUCCESS"
            for event in cls._events(output)
        )

    @classmethod
    def _has_successful_mcp_call(cls, output: str) -> bool:
        for event in cls._events(output):
            if event.get("event") != "step_update":
                continue
            update = event.get("step_update") or {}
            if update.get("state") != "DONE":
                continue
            tool_info = update.get("tool_info") or {}
            if not isinstance(tool_info, dict):
                continue
            serialized = json.dumps(tool_info, sort_keys=True).lower()
            if "pitbench" not in serialized or "run_command" not in serialized:
                continue
            if tool_info.get("error"):
                continue
            output_value: Any = tool_info.get("output")
            if isinstance(output_value, dict) and output_value.get("isError") is True:
                continue
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
        agy_binary = self._resolved_agy_binary()
        env = self._runtime_env()
        self._preflight(agy_binary, env)
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
            process = subprocess.Popen(
                self._build_command(
                    mcp_url=mcp_server.url,
                    instruction=instruction,
                ),
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
            (logging_dir / "antigravity.jsonl").write_text(stdout)
            (logging_dir / "antigravity.stderr.log").write_text(stderr)
        input_tokens, output_tokens = self._parse_usage(stdout)
        if self._cancelled.is_set():
            failure_mode = FailureMode.AGENT_TIMEOUT
        elif (
            return_code != 0
            or not self._has_successful_result(stdout)
            or not self._has_successful_mcp_call(stdout)
        ):
            failure_mode = FailureMode.UNKNOWN_AGENT_ERROR
        else:
            failure_mode = FailureMode.NONE
        return AgentResult(
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            total_cost=0.0,
            failure_mode=failure_mode,
        )
