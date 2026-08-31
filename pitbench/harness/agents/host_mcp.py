from __future__ import annotations

import base64
import json
import shlex
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import uvicorn
from docker.models.containers import Container
from mcp.server.fastmcp import FastMCP


@dataclass
class _CommandJob:
    handle: str
    command: str
    timeout_sec: float
    pid_path: str
    started_at: float = field(default_factory=time.monotonic)
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    done: bool = False
    cancelled: bool = False
    output_truncated: bool = False
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    thread: threading.Thread | None = None


class TaskTerminal:
    """A bounded tool surface for exactly one task container."""

    MAX_TIMEOUT_SEC = 900.0
    MAX_OUTPUT_CHARS = 50_000
    MAX_ASYNC_OUTPUT_CHARS = 5_000_000
    MAX_POLL_CHARS = 100_000
    MAX_ACTIVE_JOBS = 8
    MAX_RETAINED_JOBS = 32
    MAX_PATCH_CHARS = 500_000
    MAX_FILE_READ_CHARS = 500_000
    MAX_SEARCH_RESULTS = 1_000

    def __init__(
        self,
        container: Container,
        *,
        workdir: str,
        log_path: Path | None = None,
    ) -> None:
        self._container = container
        self._workdir = workdir.rstrip("/") or "/"
        self._log_path = log_path
        self._log_lock = threading.Lock()
        self._jobs: dict[str, _CommandJob] = {}
        self._jobs_lock = threading.Lock()

    @staticmethod
    def _decode(output: bytes | str | None) -> str:
        if output is None:
            return ""
        if isinstance(output, bytes):
            return output.decode(errors="replace")
        return output

    @classmethod
    def _truncate(
        cls, output: str, limit: int | None = None
    ) -> tuple[str, bool]:
        limit = cls.MAX_OUTPUT_CHARS if limit is None else limit
        if len(output) <= limit:
            return output, False
        omitted = len(output) - limit
        return f"[{omitted} earlier characters truncated]\n" + output[-limit:], True

    @staticmethod
    def _bounded_timeout(timeout_sec: float) -> float:
        return min(max(float(timeout_sec), 0.1), TaskTerminal.MAX_TIMEOUT_SEC)

    @staticmethod
    def _relative_path(path: str) -> str:
        if not path or path == ".":
            return "."
        candidate = PurePosixPath(path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("path must stay within the task repository")
        normalized = str(candidate)
        return normalized.removeprefix("./") or "."

    @staticmethod
    def _path_guard(path: str, body: str) -> str:
        quoted_path = shlex.quote(path)
        return (
            "root=$(realpath -e .) || exit 70\n"
            f"target=$(realpath -e -- {quoted_path}) || {{ "
            "echo 'path does not exist' >&2; exit 66; }\n"
            'case "$target" in "$root"|"$root"/*) ;; '
            "*) echo 'path escapes task repository' >&2; exit 64 ;; esac\n"
            + body
        )

    def _log(self, payload: dict[str, Any]) -> None:
        if self._log_path is None:
            return
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_lock, self._log_path.open("a") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")

    def _exec(
        self,
        command: str | list[str],
        *,
        timeout_sec: float = 120.0,
        output_limit: int | None = None,
    ) -> dict[str, Any]:
        timeout = self._bounded_timeout(timeout_sec)
        shell_command = command if isinstance(command, str) else shlex.join(command)
        started = time.monotonic()
        result = self._container.exec_run(
            [
                "timeout",
                "--signal=TERM",
                "--kill-after=5s",
                f"{timeout}s",
                "bash",
                "-lc",
                shell_command,
            ],
            workdir=self._workdir,
            demux=True,
        )
        elapsed = time.monotonic() - started
        if isinstance(result.output, tuple):
            raw_stdout, raw_stderr = result.output
        else:
            raw_stdout, raw_stderr = result.output, None
        stdout, stdout_truncated = self._truncate(
            self._decode(raw_stdout), output_limit
        )
        stderr, stderr_truncated = self._truncate(
            self._decode(raw_stderr), output_limit
        )
        return {
            "command": shell_command,
            "exit_code": int(result.exit_code),
            "timed_out": result.exit_code == 124,
            "elapsed_sec": round(elapsed, 3),
            "stdout": stdout,
            "stderr": stderr,
            "output_truncated": stdout_truncated or stderr_truncated,
        }

    def run_command(
        self,
        command: str,
        timeout_sec: float = 120.0,
    ) -> dict[str, Any]:
        """Run one shell command in the isolated task repository."""
        if not command.strip():
            raise ValueError("command must not be empty")
        payload = self._exec(command, timeout_sec=timeout_sec)
        self._log({"tool": "run_command", **payload})
        return payload

    def read_file(
        self,
        path: str,
        offset: int = 0,
        limit: int = 50_000,
    ) -> dict[str, Any]:
        """Read a bounded byte range from a repository text file."""
        path = self._relative_path(path)
        offset = max(int(offset), 0)
        limit = min(max(int(limit), 1), self.MAX_FILE_READ_CHARS)
        script = self._path_guard(
            path,
            'test -f "$target" || { echo \'path is not a file\' >&2; exit 65; }\n'
            'size=$(wc -c < "$target") || exit $?\n'
            'printf \'%s\\n\' "$size"\n'
            f'dd if="$target" bs=1 skip={offset} count={limit} status=none | '
            "base64 | tr -d '\\n'",
        )
        encoded_limit = ((limit + 2) // 3) * 4
        result = self._exec(script, output_limit=encoded_limit + 128)
        if result["exit_code"] != 0:
            raise ValueError(result["stderr"].strip() or "unable to read file")
        size_line, separator, encoded = result["stdout"].partition("\n")
        if not separator:
            raise RuntimeError("file reader returned an invalid response")
        size = int(size_line)
        chunk = base64.b64decode(encoded, validate=True)
        content = chunk.decode(errors="replace")
        payload = {
            "path": path,
            "offset": offset,
            "content": content,
            "next_offset": min(offset + len(chunk), size),
            "size": size,
            "eof": offset + len(chunk) >= size,
        }
        self._log({"tool": "read_file", **payload})
        return payload

    def list_files(
        self,
        path: str = ".",
        max_depth: int = 8,
        max_results: int = 500,
    ) -> dict[str, Any]:
        """List regular files below a repository path."""
        path = self._relative_path(path)
        max_depth = min(max(int(max_depth), 0), 32)
        max_results = min(max(int(max_results), 1), self.MAX_SEARCH_RESULTS)
        script = self._path_guard(
            path,
            'if test -f "$target"; then realpath --relative-to="$root" "$target"; '
            "else "
            f'find -P "$target" -maxdepth {max_depth} -type f -print | '
            f'head -n {max_results} | while IFS= read -r item; do '
            'realpath --relative-to="$root" "$item"; done; fi',
        )
        result = self._exec(script, output_limit=500_000)
        if result["exit_code"] != 0:
            raise ValueError(result["stderr"].strip() or "unable to list files")
        files = [line for line in result["stdout"].splitlines() if line]
        payload = {
            "path": path,
            "files": files[:max_results],
            "result_limit_reached": len(files) >= max_results,
        }
        self._log({"tool": "list_files", **payload})
        return payload

    def search_files(
        self,
        query: str,
        path: str = ".",
        *,
        regex: bool = False,
        glob: str | None = None,
        max_results: int = 200,
    ) -> dict[str, Any]:
        """Search repository files and return bounded line-oriented matches."""
        if not query:
            raise ValueError("query must not be empty")
        path = self._relative_path(path)
        max_results = min(max(int(max_results), 1), self.MAX_SEARCH_RESULTS)
        rg_args = ["rg", "--line-number", "--no-heading", "--color=never"]
        if not regex:
            rg_args.append("--fixed-strings")
        if glob:
            rg_args.extend(["--glob", glob])
        rg_args.extend(["--", query, "$target"])
        grep_args = ["grep", "-RIn"]
        if not regex:
            grep_args.append("-F")
        if glob:
            grep_args.append(f"--include={glob}")
        grep_args.extend(["--", query, "$target"])
        search = (
            f"if command -v rg >/dev/null 2>&1; then {shlex.join(rg_args)}; "
            f"else {shlex.join(grep_args)}; fi"
        ).replace("'$target'", '"$target"')
        script = self._path_guard(
            path,
            f"set +o pipefail\n({search}) | head -n {max_results}\n"
            "status=${PIPESTATUS[0]}\n"
            "test \"$status\" -eq 0 -o \"$status\" -eq 1 -o "
            '"$status" -eq 141',
        )
        result = self._exec(script, output_limit=1_000_000)
        if result["exit_code"] != 0:
            raise ValueError(result["stderr"].strip() or "search failed")
        matches = [line for line in result["stdout"].splitlines() if line]
        payload = {
            "query": query,
            "path": path,
            "matches": matches[:max_results],
            "result_limit_reached": len(matches) >= max_results,
        }
        self._log({"tool": "search_files", **payload})
        return payload

    @staticmethod
    def _validate_patch_paths(patch: str) -> None:
        for line in patch.splitlines():
            if not line.startswith(("--- ", "+++ ")):
                continue
            raw_path = line[4:].split("\t", 1)[0]
            if raw_path == "/dev/null":
                continue
            if raw_path.startswith(("a/", "b/")):
                raw_path = raw_path[2:]
            candidate = PurePosixPath(raw_path)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError("patch paths must stay within the task repository")

    def apply_patch(self, patch: str) -> dict[str, Any]:
        """Apply a unified diff inside the task repository using git apply."""
        if not patch.strip():
            raise ValueError("patch must not be empty")
        if "\x00" in patch or len(patch) > self.MAX_PATCH_CHARS:
            raise ValueError("patch is not valid text or exceeds the size limit")
        self._validate_patch_paths(patch)
        encoded = base64.b64encode(patch.encode()).decode()
        patch_path = f"/tmp/pitbench-mcp-patch-{uuid.uuid4().hex}.diff"
        script = (
            "set -e\n"
            f"patch_file={shlex.quote(patch_path)}\n"
            "trap 'rm -f -- \"$patch_file\"' EXIT\n"
            f"printf %s {shlex.quote(encoded)} | base64 -d > \"$patch_file\"\n"
            "git apply --check \"$patch_file\"\n"
            "git apply \"$patch_file\""
        )
        result = self._exec(script, timeout_sec=120.0, output_limit=200_000)
        payload = {
            "applied": result["exit_code"] == 0,
            "stderr": result["stderr"],
            "elapsed_sec": result["elapsed_sec"],
        }
        self._log({"tool": "apply_patch", **payload})
        if not payload["applied"]:
            raise ValueError(result["stderr"].strip() or "git apply failed")
        return payload

    def git_status(self) -> dict[str, Any]:
        """Return parsed Git index and working-tree status entries."""
        result = self._exec(
            "git status --porcelain=v1 --untracked-files=all",
            output_limit=500_000,
        )
        if result["exit_code"] != 0:
            raise ValueError(result["stderr"].strip() or "git status failed")
        entries = []
        for line in result["stdout"].splitlines():
            if len(line) >= 3:
                entries.append(
                    {"index": line[0], "worktree": line[1], "path": line[3:]}
                )
        payload = {"clean": not entries, "entries": entries}
        self._log({"tool": "git_status", **payload})
        return payload

    def git_diff(
        self,
        path: str | None = None,
        *,
        staged: bool = False,
        max_chars: int = 500_000,
    ) -> dict[str, Any]:
        """Return a bounded Git diff for the repository or one repository path."""
        max_chars = min(max(int(max_chars), 1), 1_000_000)
        args = ["git", "diff"]
        if staged:
            args.append("--cached")
        if path is not None:
            args.extend(["--", self._relative_path(path)])
        result = self._exec(args, output_limit=max_chars)
        if result["exit_code"] != 0:
            raise ValueError(result["stderr"].strip() or "git diff failed")
        payload = {
            "path": path,
            "staged": staged,
            "diff": result["stdout"],
            "output_truncated": result["output_truncated"],
        }
        self._log({"tool": "git_diff", **payload})
        return payload

    def _can_stream(self) -> bool:
        client = getattr(self._container, "client", None)
        api = getattr(client, "api", None)
        return bool(
            api is not None
            and hasattr(api, "exec_create")
            and hasattr(api, "exec_start")
            and hasattr(api, "exec_inspect")
            and getattr(self._container, "id", None)
        )

    @staticmethod
    def _async_wrapper(job: _CommandJob) -> list[str]:
        script = (
            "set +e\n"
            f"pid_file={shlex.quote(job.pid_path)}\n"
            "trap 'rm -f -- \"$pid_file\"' EXIT\n"
            f"setsid timeout --signal=TERM --kill-after=5s {job.timeout_sec}s "
            'bash -lc "$1" &\n'
            "child=$!\n"
            "printf '%s' \"$child\" > \"$pid_file\"\n"
            "wait \"$child\"\n"
            "exit $?"
        )
        return ["bash", "-lc", script, "pitbench-command", job.command]

    def _append_job_output(
        self, job: _CommandJob, stdout: bytes | str | None, stderr: bytes | str | None
    ) -> None:
        with job.lock:
            for attribute, chunk in (("stdout", stdout), ("stderr", stderr)):
                decoded = self._decode(chunk)
                if not decoded:
                    continue
                current = getattr(job, attribute)
                remaining = self.MAX_ASYNC_OUTPUT_CHARS - len(current)
                if remaining <= 0:
                    job.output_truncated = True
                    continue
                setattr(job, attribute, current + decoded[:remaining])
                if len(decoded) > remaining:
                    job.output_truncated = True

    def _run_job(self, job: _CommandJob) -> None:
        try:
            if job.cancelled:
                return
            command = self._async_wrapper(job)
            if self._can_stream():
                api = self._container.client.api
                created = api.exec_create(
                    self._container.id,
                    command,
                    stdout=True,
                    stderr=True,
                    workdir=self._workdir,
                )
                exec_id = created["Id"]
                output = api.exec_start(exec_id, stream=True, demux=True)
                for chunk in output:
                    if isinstance(chunk, tuple):
                        self._append_job_output(job, chunk[0], chunk[1])
                    else:
                        self._append_job_output(job, chunk, None)
                job.exit_code = int(api.exec_inspect(exec_id)["ExitCode"])
            else:
                result = self._container.exec_run(
                    command,
                    workdir=self._workdir,
                    demux=True,
                )
                if isinstance(result.output, tuple):
                    self._append_job_output(job, result.output[0], result.output[1])
                else:
                    self._append_job_output(job, result.output, None)
                job.exit_code = int(result.exit_code)
        except Exception as exc:
            job.error = f"{type(exc).__name__}: {exc}"
        finally:
            with job.lock:
                job.done = True
            self._log(
                {
                    "tool": "command_completed",
                    "handle": job.handle,
                    "command": job.command,
                    "exit_code": job.exit_code,
                    "cancelled": job.cancelled,
                    "error": job.error,
                    "elapsed_sec": round(time.monotonic() - job.started_at, 3),
                    "stdout_chars": len(job.stdout),
                    "stderr_chars": len(job.stderr),
                    "output_truncated": job.output_truncated,
                }
            )

    def _prune_jobs(self) -> None:
        completed = sorted(
            (job for job in self._jobs.values() if job.done),
            key=lambda job: job.started_at,
        )
        while len(self._jobs) >= self.MAX_RETAINED_JOBS and completed:
            old = completed.pop(0)
            self._jobs.pop(old.handle, None)

    def start_command(
        self, command: str, timeout_sec: float = 120.0
    ) -> dict[str, Any]:
        """Start a command and return a handle that can be polled or cancelled."""
        if not command.strip():
            raise ValueError("command must not be empty")
        timeout = self._bounded_timeout(timeout_sec)
        with self._jobs_lock:
            self._prune_jobs()
            active = sum(not job.done for job in self._jobs.values())
            if active >= self.MAX_ACTIVE_JOBS:
                raise RuntimeError("too many active commands")
            handle = uuid.uuid4().hex
            job = _CommandJob(
                handle=handle,
                command=command,
                timeout_sec=timeout,
                pid_path=f"/tmp/pitbench-mcp-command-{handle}.pid",
            )
            self._jobs[handle] = job
        thread = threading.Thread(
            target=self._run_job,
            args=(job,),
            name=f"pitbench-command-{handle[:8]}",
            daemon=True,
        )
        job.thread = thread
        thread.start()
        payload = {"handle": handle, "streaming": self._can_stream()}
        self._log({"tool": "start_command", "command": command, **payload})
        return payload

    def _job(self, handle: str) -> _CommandJob:
        with self._jobs_lock:
            job = self._jobs.get(handle)
        if job is None:
            raise ValueError("unknown or expired command handle")
        return job

    def poll_command(
        self,
        handle: str,
        stdout_offset: int = 0,
        stderr_offset: int = 0,
        max_chars: int = 20_000,
    ) -> dict[str, Any]:
        """Read the next output chunks and current state of an async command."""
        job = self._job(handle)
        stdout_offset = max(int(stdout_offset), 0)
        stderr_offset = max(int(stderr_offset), 0)
        max_chars = min(max(int(max_chars), 1), self.MAX_POLL_CHARS)
        with job.lock:
            stdout = job.stdout[stdout_offset : stdout_offset + max_chars]
            stderr = job.stderr[stderr_offset : stderr_offset + max_chars]
            payload = {
                "handle": handle,
                "done": job.done,
                "exit_code": job.exit_code,
                "timed_out": job.exit_code == 124,
                "cancelled": job.cancelled,
                "elapsed_sec": round(time.monotonic() - job.started_at, 3),
                "stdout": stdout,
                "stderr": stderr,
                "next_stdout_offset": stdout_offset + len(stdout),
                "next_stderr_offset": stderr_offset + len(stderr),
                "stdout_chars": len(job.stdout),
                "stderr_chars": len(job.stderr),
                "output_truncated": job.output_truncated,
                "error": job.error,
            }
        return payload

    def cancel_command(self, handle: str) -> dict[str, Any]:
        """Terminate the isolated process group for an async command."""
        job = self._job(handle)
        with job.lock:
            if job.done:
                return {"handle": handle, "cancelled": job.cancelled, "done": True}
            job.cancelled = True
        script = (
            f"pid_file={shlex.quote(job.pid_path)}\n"
            "for attempt in $(seq 1 20); do test -s \"$pid_file\" && break; "
            "sleep 0.05; done\n"
            "if test -s \"$pid_file\"; then pid=$(cat \"$pid_file\"); "
            "kill -TERM -- \"-$pid\" 2>/dev/null || true; sleep 0.2; "
            "kill -KILL -- \"-$pid\" 2>/dev/null || true; fi"
        )
        self._container.exec_run(
            ["bash", "-lc", script], workdir=self._workdir, demux=True
        )
        self._log({"tool": "cancel_command", "handle": handle})
        return {"handle": handle, "cancelled": True, "done": job.done}

    def close(self) -> None:
        with self._jobs_lock:
            active = [job for job in self._jobs.values() if not job.done]
        for job in active:
            try:
                self.cancel_command(job.handle)
            except Exception:
                pass
        for job in active:
            if job.thread is not None:
                job.thread.join(timeout=2.0)


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
                "Work only inside the isolated task repository. Prefer read_file, "
                "list_files, search_files, apply_patch, git_status, and git_diff for "
                "structured repository operations. Use run_command for short shell "
                "operations and start/poll/cancel_command for long-running work."
            ),
            stateless_http=True,
            json_response=True,
            log_level="WARNING",
        )

        @mcp.tool(
            name="run_command",
            description=(
                "Run a short shell command in /workspace/repo inside the isolated "
                "task container and return its completed output."
            ),
            structured_output=True,
        )
        def run_command(command: str, timeout_sec: float = 120.0) -> dict[str, Any]:
            return self._terminal.run_command(command, timeout_sec)

        @mcp.tool(structured_output=True)
        def read_file(
            path: str, offset: int = 0, limit: int = 50_000
        ) -> dict[str, Any]:
            """Read a byte range from a text file contained in the task repository."""
            return self._terminal.read_file(path, offset, limit)

        @mcp.tool(structured_output=True)
        def list_files(
            path: str = ".", max_depth: int = 8, max_results: int = 500
        ) -> dict[str, Any]:
            """List files below a path contained in the task repository."""
            return self._terminal.list_files(path, max_depth, max_results)

        @mcp.tool(structured_output=True)
        def search_files(
            query: str,
            path: str = ".",
            regex: bool = False,
            glob: str | None = None,
            max_results: int = 200,
        ) -> dict[str, Any]:
            """Search file contents inside the task repository with ripgrep semantics."""
            return self._terminal.search_files(
                query, path, regex=regex, glob=glob, max_results=max_results
            )

        @mcp.tool(structured_output=True)
        def apply_patch(patch: str) -> dict[str, Any]:
            """Apply a unified diff whose paths are contained in the task repository."""
            return self._terminal.apply_patch(patch)

        @mcp.tool(structured_output=True)
        def git_status() -> dict[str, Any]:
            """Return structured Git index and working-tree status entries."""
            return self._terminal.git_status()

        @mcp.tool(structured_output=True)
        def git_diff(
            path: str | None = None,
            staged: bool = False,
            max_chars: int = 500_000,
        ) -> dict[str, Any]:
            """Return the Git diff for the repository or one contained path."""
            return self._terminal.git_diff(path, staged=staged, max_chars=max_chars)

        @mcp.tool(structured_output=True)
        def start_command(
            command: str, timeout_sec: float = 120.0
        ) -> dict[str, Any]:
            """Start a long command and return a handle for polling and cancellation."""
            return self._terminal.start_command(command, timeout_sec)

        @mcp.tool(structured_output=True)
        def poll_command(
            handle: str,
            stdout_offset: int = 0,
            stderr_offset: int = 0,
            max_chars: int = 20_000,
        ) -> dict[str, Any]:
            """Read incremental output and status for an asynchronous command handle."""
            return self._terminal.poll_command(
                handle, stdout_offset, stderr_offset, max_chars
            )

        @mcp.tool(structured_output=True)
        def cancel_command(handle: str) -> dict[str, Any]:
            """Terminate an asynchronous command without affecting other task processes."""
            return self._terminal.cancel_command(handle)

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
        self._terminal.close()
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive() and self._server is not None:
                self._server.force_exit = True
                self._thread.join(timeout=2.0)
        if self._socket is not None:
            self._socket.close()
