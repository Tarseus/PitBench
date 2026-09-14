from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from pitbench.harness.agents.agent_name import AgentName
from pitbench.harness.agents.base_agent import AgentResult
from pitbench.harness.agents.codex_relay import CodexModelRelay
from pitbench.harness.agents.codex_workspace import CodexWorkspaceRuntime
from pitbench.harness.agents.containers import CodexContainerRunner
from pitbench.harness.agents.failure_mode import FailureMode
from pitbench.harness.agents.host_mcp import LoopbackMCPServer, TaskTerminal
from pitbench.harness.agents.process_agent import ManagedProcessAgent
from pitbench.harness.agents.profiles import CodexProfile
from pitbench.harness.terminal.tmux_session import TmuxSession


class CodexMCPAgent(ManagedProcessAgent):
    """Run Codex through MCP isolation or directly inside the task container."""

    credential_environment_keys = ("OPENAI_API_KEY",)
    _PROMPT = """You are improving a solver in an isolated PitBench task.

You MUST use only the pitbench MCP tools for repository inspection, editing,
building, profiling, and testing. Prefer read_file, list_files, search_files,
apply_patch, git_status, and git_diff for structured repository work. Use run_command
for short commands and start_command, poll_command, and cancel_command for long-running
commands or incremental output. The host working directory is empty and is not the
task repository. Do not use the host shell or web access. Start with git_status and run
`pwd` through run_command. Keep all changes inside the task repository and use the
repository's own build and test commands when helpful.

Treat existing tests as immutable specifications. Never change a test to accommodate
an implementation change. The repository is read-only outside the editable paths
stated in the task; keep implementation changes inside those paths. Preserve behavior
across every supported problem feature. Before finishing, run relevant repository
tests, inspect git_status, and remove generated output and temporary helper files. The
final candidate must contain only intentional source changes.

Task:
{instruction}
"""
    _CONTROL_PLANE_PROMPT = (
        "Reply with exactly PITBENCH_CODEX_CONTROL_PLANE_OK. Do not use tools."
    )
    _WORKSPACE_PROMPT = """You are improving a solver in an isolated PitBench task.

You are running inside the solver container with the repository at {repository}.
Use the normal Codex shell, file, search, and patch capabilities. The command sandbox
allows repository writes. The container network can reach only the PitBench model
relay, not the public internet. Do not use web search. Start with `git status --short`,
and `pwd`.

Treat existing tests as immutable specifications. Never change a test to accommodate
an implementation change. The repository is read-only outside the editable paths
stated in the task; keep implementation changes inside those paths. Preserve behavior
across every supported problem feature. Before finishing, run relevant repository
tests, inspect `git status --short`, and remove generated output and temporary helper
files. The final candidate must contain only intentional source changes.

Task:
{instruction}
"""
    _WORKSPACE_RELAY_MARKER = "PITBENCH_RELAY_REACHABLE"
    _WORKSPACE_PUBLIC_BLOCKED_MARKER = "PITBENCH_PUBLIC_BLOCKED"

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
        relay_max_requests: int = 256,
        relay_max_total_tokens: int = 10_000_000,
        relay_max_concurrent_requests: int = 1,
        relay_max_duration_sec: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._model_name = model_name.split("/")[-1]
        self._codex_binary = codex_binary
        self._codex_auth_path = Path(codex_auth_path).expanduser()
        self._runner_path = Path(runner_path)
        self._runner_user = runner_user
        if runner_backend not in {"host", "container", "workspace"}:
            raise ValueError(
                "runner_backend must be 'host', 'container', or 'workspace'"
            )
        if runner_backend == "host" and profile_path is not None:
            raise ValueError(
                "Codex profiles require runner_backend=container or workspace"
            )
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
        self._relay_max_requests = relay_max_requests
        self._relay_max_total_tokens = relay_max_total_tokens
        self._relay_max_concurrent_requests = relay_max_concurrent_requests
        self._relay_max_duration_sec = (
            relay_max_duration_sec
            if relay_max_duration_sec is not None
            else timeout_sec + control_plane_timeout_sec + 60
        )
        budgets = {
            "relay_max_requests": self._relay_max_requests,
            "relay_max_total_tokens": self._relay_max_total_tokens,
            "relay_max_concurrent_requests": self._relay_max_concurrent_requests,
            "relay_max_duration_sec": self._relay_max_duration_sec,
        }
        invalid = [name for name, value in budgets.items() if value <= 0]
        if invalid:
            raise ValueError(f"relay budgets must be positive: {', '.join(invalid)}")

    def _resolved_codex_binary(self) -> str:
        resolved = shutil.which(self._codex_binary)
        if resolved is None:
            raise RuntimeError(
                f"Codex CLI was not found: {self._codex_binary}. "
                "Install it on the host."
            )
        return resolved

    @staticmethod
    def check_login(
        binary: str, auth_path: Path, env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        """Check the configured file, not an unrelated login in the host keyring."""
        raw = auth_path.read_text()
        json.loads(raw)
        with tempfile.TemporaryDirectory(prefix="pitbench-auth-check-") as directory:
            auth_directory = Path(directory)
            copied_auth = auth_directory / "auth.json"
            copied_auth.write_text(raw)
            copied_auth.chmod(0o600)
            (auth_directory / "config.toml").write_text(
                'cli_auth_credentials_store = "file"\n'
            )
            environment = (env if env is not None else os.environ).copy()
            environment.pop("OPENAI_API_KEY", None)
            environment["CODEX_HOME"] = str(auth_directory)
            return subprocess.run(
                [binary, "login", "status"],
                env=environment,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )

    def _preflight(self, codex_binary: str, env: dict[str, str]) -> None:
        result = self.check_login(codex_binary, self._codex_auth_path, env)
        status = f"{result.stdout}\n{result.stderr}"
        if result.returncode != 0 or "Logged in using ChatGPT" not in status:
            raise RuntimeError(
                "The configured Codex auth file is not a ChatGPT login. "
                "Run `pitbench auth login codex` or set codex_auth_path."
            )
        if self._runner_backend == "container":
            self._container_runner = CodexContainerRunner(
                image=self._container_runner_image,
                codex_binary=Path(codex_binary),
                profile=self._profile,
                recording_dir=self._agent_trace.native_inbox
                if self._agent_trace
                else None,
            )
            self._runner_metadata = self._container_runner.prepare()
            return
        if self._runner_backend == "workspace":
            self._runner_metadata = {
                "schema_version": "1.0",
                "backend": "workspace",
                "profile": self._profile.metadata() if self._profile else None,
            }
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
        if self._runner_backend == "workspace":
            return ""
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

    def _workspace_exec_prefix(
        self,
        *,
        runtime: CodexWorkspaceRuntime,
        relay: CodexModelRelay,
    ) -> list[str]:
        command = [
            *runtime.command_prefix(),
            "--profile",
            runtime.config_profile_name,
            "--ask-for-approval",
            "never",
        ]
        if self._agent_trace is not None or (
            self._profile is not None and self._profile.allow_hooks
        ):
            command.append("--dangerously-bypass-hook-trust")
        command.extend(
            [
                "exec",
                "--strict-config",
                "--ephemeral",
                "--json",
                "--color",
                "never",
                "--skip-git-repo-check",
                "--model",
                self._model_name,
                "-C",
                self._container_workdir,
                "-c",
                'model_provider="pitbench_relay"',
                "-c",
                'model_providers.pitbench_relay.name="PitBench Relay"',
                "-c",
                "model_providers.pitbench_relay.base_url=" + json.dumps(relay.url),
                "-c",
                'model_providers.pitbench_relay.wire_api="responses"',
                "-c",
                "model_providers.pitbench_relay.supports_websockets=false",
                "-c",
                'web_search="disabled"',
                "-c",
                "features.unified_exec=true",
                "-c",
                "shell_environment_policy.ignore_default_excludes=false",
                "-c",
                "check_for_update_on_startup=false",
                "-c",
                "feedback.enabled=false",
                "-c",
                'history.persistence="none"',
            ]
        )
        command.extend(self._reasoning_effort_args())
        return command

    def _build_workspace_command(
        self,
        *,
        runtime: CodexWorkspaceRuntime,
        relay: CodexModelRelay,
        instruction: str,
    ) -> list[str]:
        return [
            *self._workspace_exec_prefix(runtime=runtime, relay=relay),
            "--",
            self._WORKSPACE_PROMPT.format(
                instruction=self._render_instruction(instruction),
                repository=self._container_workdir,
            ),
        ]

    def _build_workspace_sandbox_probe(
        self,
        *,
        runtime: CodexWorkspaceRuntime,
        script: str,
    ) -> list[str]:
        return [
            *runtime.command_prefix(),
            "sandbox",
            "-p",
            runtime.config_profile_name,
            "-P",
            runtime.permission_profile_name,
            "-C",
            self._container_workdir,
            "python3",
            "-c",
            script,
        ]

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

    @staticmethod
    def _has_successful_workspace_call(output: str) -> bool:
        for line in output.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "item.completed":
                continue
            item = event.get("item") or {}
            if (
                item.get("type") == "command_execution"
                and item.get("status") == "completed"
                and item.get("exit_code") == 0
            ):
                return True
        return False

    def _check_workspace_isolation(
        self,
        *,
        runtime: CodexWorkspaceRuntime,
        relay: CodexModelRelay,
        env: dict[str, str],
        logging_dir: Path | None,
    ) -> None:
        relay_script = (
            "import os,urllib.error,urllib.request; "
            "assert 'PITBENCH_RELAY_TOKEN' not in os.environ; "
            f"url='http://{relay.container_ip}:{relay.port}/responses'; "
            "\ntry: urllib.request.urlopen(url,timeout=10)\n"
            "except urllib.error.HTTPError: pass\n"
            f"print('{self._WORKSPACE_RELAY_MARKER}')"
        )
        public_script = (
            "import urllib.error,urllib.request; "
            "\ntry: urllib.request.urlopen('http://1.1.1.1',timeout=10)\n"
            "except urllib.error.HTTPError as error:\n"
            " assert error.code == 403\n"
            f" print('{self._WORKSPACE_PUBLIC_BLOCKED_MARKER}')\n"
            "else: raise SystemExit('public network unexpectedly reachable')"
        )
        results: list[tuple[str, str, int]] = []
        for script in (relay_script, public_script):
            process = subprocess.Popen(
                self._build_workspace_sandbox_probe(runtime=runtime, script=script),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            self._set_process(process)
            try:
                stdout, stderr = process.communicate(
                    timeout=self._control_plane_timeout_sec
                )
            except subprocess.TimeoutExpired as error:
                self._terminate_process(process)
                process.communicate()
                raise RuntimeError(
                    "Codex workspace isolation preflight timed out"
                ) from error
            finally:
                self._set_process(None)
            results.append((stdout, stderr, process.returncode or 0))
        relay_stdout, relay_stderr, relay_status = results[0]
        public_stdout, public_stderr, public_status = results[1]
        transcript = (
            "[relay stdout]\n"
            + relay_stdout
            + "[relay stderr]\n"
            + relay_stderr
            + "[public stdout]\n"
            + public_stdout
            + "[public stderr]\n"
            + public_stderr
        )
        if logging_dir is not None:
            (logging_dir / "codex-preflight.log").write_text(transcript)
        if (
            relay_status != 0
            or self._WORKSPACE_RELAY_MARKER not in relay_stdout
            or public_status != 0
            or self._WORKSPACE_PUBLIC_BLOCKED_MARKER not in public_stdout
        ):
            raise RuntimeError(
                "Codex workspace did not prove that shell networking is limited "
                "to the model relay; workspace backend refuses to continue: "
                f"{transcript.strip()}"
            )

    def _perform_workspace_task(
        self,
        *,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None,
        codex_binary: str,
        env: dict[str, str],
    ) -> AgentResult:
        runtime = CodexWorkspaceRuntime(
            container=session.container,
            codex_binary=Path(codex_binary),
            profile=self._profile,
        )
        relay_log = (
            logging_dir / "codex-relay.jsonl" if logging_dir is not None else None
        )
        stdout = ""
        stderr = ""
        return_code = 1
        with runtime:
            if self._agent_trace is not None:
                self._agent_trace.install_native_hooks(
                    session.container,
                    config_path=f"{runtime.codex_home}/hooks.json",
                    provider="codex",
                    root=self._container_workdir,
                )
            if runtime.network is None or runtime.container_ip is None:
                raise RuntimeError("Codex workspace network was not initialized")
            with CodexModelRelay(
                auth_path=self._codex_auth_path,
                model=self._model_name,
                allowed_client_ip=runtime.container_ip,
                network=runtime.network,
                image=session.container.image.id,
                log_path=relay_log,
                proxy_url=self._proxy_url,
                max_requests=self._relay_max_requests,
                max_total_tokens=self._relay_max_total_tokens,
                max_concurrent_requests=self._relay_max_concurrent_requests,
                max_duration_sec=self._relay_max_duration_sec,
                trace=self._agent_trace,
            ) as relay:
                if relay.container_ip is None:
                    raise RuntimeError("Codex relay sidecar address is unavailable")
                runtime.configure_relay(relay.container_ip)
                self._runner_metadata = runtime.metadata()
                self._runner_metadata["relay"] = relay.metadata()
                if logging_dir is not None:
                    (logging_dir / "codex-runner.json").write_text(
                        json.dumps(self._runner_metadata, indent=2, sort_keys=True)
                        + "\n"
                    )
                if self._control_plane_preflight:
                    self._report_progress("Agent: Checking workspace isolation")
                    self._check_workspace_isolation(
                        runtime=runtime,
                        relay=relay,
                        env=env,
                        logging_dir=logging_dir,
                    )
                self._report_progress("Agent: Waiting for model")
                process = subprocess.Popen(
                    self._build_workspace_command(
                        runtime=runtime,
                        relay=relay,
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
                    stdout, stderr, return_code = self._stream_process(
                        process, "", timeout_sec=self._timeout_sec
                    )
                finally:
                    self._set_process(None)
        if logging_dir is not None:
            (logging_dir / "codex.jsonl").write_text(stdout)
            (logging_dir / "codex.stderr.log").write_text(stderr)
        input_tokens, output_tokens = self._parse_usage(stdout)
        if self._cancelled.is_set():
            failure_mode = FailureMode.AGENT_TIMEOUT
        elif return_code != 0 or not self._has_successful_workspace_call(stdout):
            failure_mode = FailureMode.UNKNOWN_AGENT_ERROR
        else:
            failure_mode = FailureMode.NONE
        return AgentResult(
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            total_cost=0.0,
            failure_mode=failure_mode,
        )

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
        self._report_progress("Agent: Checking credentials and runner")
        self._preflight(codex_binary, env)
        self._cancelled.clear()
        if logging_dir is not None:
            logging_dir.mkdir(parents=True, exist_ok=True)
        if self._runner_backend == "workspace":
            self._report_progress("Agent: Preparing workspace and relay")
            return self._perform_workspace_task(
                instruction=instruction,
                session=session,
                logging_dir=logging_dir,
                codex_binary=codex_binary,
                env=env,
            )
        runner_payload = self._runner_payload(env)
        if logging_dir is not None:
            (logging_dir / "codex-runner.json").write_text(
                json.dumps(self._runner_metadata, indent=2, sort_keys=True) + "\n"
            )
        if self._control_plane_preflight:
            self._report_progress("Agent: Checking model connection")
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
        self._report_progress("Agent: Connecting task tools")
        with LoopbackMCPServer(terminal) as mcp_server:
            if mcp_server.url is None:
                raise RuntimeError("PitBench MCP server URL is unavailable")
            command = self._build_command(
                mcp_url=mcp_server.url,
                instruction=instruction,
            )
            self._report_progress("Agent: Waiting for model")
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
                    process, runner_payload, timeout_sec=self._timeout_sec
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
