from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from pitbench.harness.agents.agent_name import AgentName
from pitbench.harness.agents.antigravity_container import AntigravityContainerRunner
from pitbench.harness.agents.antigravity_profile import AntigravityProfile
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

You MUST use only the PitBench MCP server's run_command tool for task repository
inspection, editing, building, profiling, and testing. The runner working directory
is empty and is not the task repository. You may use customizations supplied by the
evaluator's trusted Antigravity profile, but all task repository access must go
through PitBench MCP. Start by running `pwd`, `git status --short`, and `pitbench
inspect` through the PitBench MCP tool. Keep all changes inside the task repository
and verify them with the visible PitBench commands.

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
    _RETRYABLE_NETWORK_MARKERS = (
        "network issue",
        "connection reset",
        "connecting to the server",
        "stream was interrupted",
        "temporarily unavailable",
    )

    @staticmethod
    def name() -> str:
        return AgentName.ANTIGRAVITY.value

    def __init__(
        self,
        model_name: str,
        *,
        agy_binary: str = "agy",
        auth_token_path: str = ("~/.gemini/antigravity-cli/antigravity-oauth-token"),
        settings_path: str = "~/.gemini/antigravity-cli/settings.json",
        runner_path: str = "/usr/local/libexec/pitbench-antigravity-runner",
        runner_user: str = "pitbench-agy",
        runner_backend: str = "host",
        container_runner_image: str = "python:3.13-slim-bookworm",
        profile_path: str | None = None,
        container_workdir: str = "/workspace/repo",
        timeout_sec: float = 3500.0,
        print_timeout: str = "55m",
        effort: str | None = None,
        proxy_url: str | None = None,
        network_retries: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._model_name = model_name.split("/")[-1]
        self._agy_binary = agy_binary
        self._auth_token_path = Path(auth_token_path).expanduser()
        self._settings_path = Path(settings_path).expanduser()
        self._runner_path = Path(runner_path)
        self._runner_user = runner_user
        if runner_backend not in {"host", "container"}:
            raise ValueError("runner_backend must be 'host' or 'container'")
        if runner_backend == "host" and profile_path is not None:
            raise ValueError("Antigravity profiles require runner_backend=container")
        self._runner_backend = runner_backend
        self._container_runner_image = container_runner_image
        self._profile = (
            AntigravityProfile.load(Path(profile_path))
            if profile_path is not None
            else None
        )
        self._container_runner: AntigravityContainerRunner | None = None
        self._runner_metadata: dict[str, object] = {
            "schema_version": "1.0",
            "backend": "host",
            "profile": None,
        }
        self._container_workdir = container_workdir
        self._timeout_sec = timeout_sec
        self._print_timeout = print_timeout
        self._effort = effort
        self._proxy_url = proxy_url
        if network_retries not in {0, 1}:
            raise ValueError("network_retries must be 0 or 1")
        self._network_retries = network_retries
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
        if self._runner_backend == "container":
            self._container_runner = AntigravityContainerRunner(
                image=self._container_runner_image,
                agy_binary=Path(agy_binary),
                profile=self._profile,
            )
            self._runner_metadata = self._container_runner.prepare()
            return
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

    def _runner_command_prefix(self) -> list[str]:
        if self._runner_backend == "container":
            if self._container_runner is None:
                raise RuntimeError(
                    "Antigravity container runner has not passed preflight"
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

    def _build_command(self, *, mcp_url: str, instruction: str) -> list[str]:
        command = [
            *self._runner_command_prefix(),
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
            "--print-timeout",
            self._print_timeout,
        ]
        if self._profile is None:
            command.append("--disable-slash-commands")
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
        auth_token_json = self._read_json(self._auth_token_path, label="OAuth token")
        settings_json = self._read_json(
            self._settings_path, label="authentication settings"
        )
        auth_token = json.loads(auth_token_json)
        settings = json.loads(settings_json)
        auth_method = (
            auth_token.get("auth_method") if isinstance(auth_token, dict) else None
        )
        if not isinstance(auth_method, str) or not auth_method:
            raise RuntimeError(
                f"Invalid host Antigravity OAuth token: {self._auth_token_path}"
            )
        if not isinstance(settings, dict):
            raise RuntimeError(
                f"Invalid host Antigravity authentication settings: "
                f"{self._settings_path}"
            )

        # Older installed runners require this selection, while agy 1.1.20 no
        # longer persists it in settings.json. Derive it without changing the
        # user's settings file so both runner revisions accept the payload.
        security = settings.setdefault("security", {})
        if not isinstance(security, dict):
            raise RuntimeError("Invalid Antigravity security settings")
        auth = security.setdefault("auth", {})
        if not isinstance(auth, dict):
            raise RuntimeError("Invalid Antigravity authentication settings")
        selected_type = auth.get("selectedType")
        if not isinstance(selected_type, str) or not selected_type:
            auth["selectedType"] = auth_method

        return json.dumps(
            {
                "auth_token_json": auth_token_json,
                "settings_json": json.dumps(settings),
                "proxy_env": proxy_env,
                "allow_hooks": bool(self._profile and self._profile.allow_hooks),
                "profile_sha256": self._profile.sha256 if self._profile else None,
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
        input_tokens = 0
        output_tokens = 0
        for event in cls._events(output):
            if event.get("event") != "result":
                continue
            result = event.get("result") or {}
            usage = result.get("usage") or {}
            input_tokens += int(usage.get("input_tokens", 0))
            output_tokens += int(usage.get("output_tokens", 0))
        return input_tokens, output_tokens

    @classmethod
    def _has_successful_result(cls, output: str) -> bool:
        return any(
            event.get("event") == "result"
            and (event.get("result") or {}).get("status") == "SUCCESS"
            for event in cls._events(output)
        )

    @classmethod
    def _has_retryable_network_error(cls, output: str, stderr: str) -> bool:
        messages = [stderr]
        for event in cls._events(output):
            if event.get("event") != "result":
                continue
            result = event.get("result") or {}
            if result.get("status") == "SUCCESS":
                return False
            messages.extend(
                [str(result.get("error") or ""), str(result.get("response") or "")]
            )
        combined = "\n".join(messages).lower()
        return any(marker in combined for marker in cls._RETRYABLE_NETWORK_MARKERS)

    @staticmethod
    def _continuation_instruction(instruction: str) -> str:
        return (
            "A transient connection interrupted the previous attempt. Continue from "
            "the existing task repository state instead of restarting. Inspect the "
            "current diff, finish or repair the implementation, remove forbidden or "
            "temporary changes, then run pitbench validate and pitbench check before "
            f"finishing. The original task was:\n{instruction}"
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
            (logging_dir / "antigravity-runner.json").write_text(
                json.dumps(self._runner_metadata, indent=2, sort_keys=True) + "\n"
            )
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
        deadline = time.monotonic() + self._timeout_sec
        with LoopbackMCPServer(terminal) as mcp_server:
            if mcp_server.url is None:
                raise RuntimeError("PitBench MCP server URL is unavailable")
            attempt_instruction = instruction
            for attempt in range(self._network_retries + 1):
                process = subprocess.Popen(
                    self._build_command(
                        mcp_url=mcp_server.url,
                        instruction=attempt_instruction,
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
                    remaining = max(0.1, deadline - time.monotonic())
                    attempt_stdout, attempt_stderr = process.communicate(
                        input=runner_payload,
                        timeout=remaining,
                    )
                    return_code = process.returncode
                except subprocess.TimeoutExpired:
                    self._cancelled.set()
                    self._terminate_process(process)
                    attempt_stdout, attempt_stderr = process.communicate()
                    return_code = process.returncode
                finally:
                    self._set_process(None)

                stdout += attempt_stdout
                if attempt_stdout and not attempt_stdout.endswith("\n"):
                    stdout += "\n"
                stderr += attempt_stderr
                if attempt_stderr and not attempt_stderr.endswith("\n"):
                    stderr += "\n"

                should_retry = (
                    attempt < self._network_retries
                    and not self._cancelled.is_set()
                    and time.monotonic() < deadline
                    and not self._has_successful_result(attempt_stdout)
                    and self._has_retryable_network_error(
                        attempt_stdout, attempt_stderr
                    )
                )
                if not should_retry:
                    break
                attempt_instruction = self._continuation_instruction(instruction)
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
