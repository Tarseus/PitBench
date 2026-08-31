from __future__ import annotations

import argparse
import base64
import json
import os
import select
import socket
import socketserver
import tempfile
import threading
import time
import urllib.error
import urllib.request
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

if TYPE_CHECKING:
    from docker.models.containers import Container
    from docker.models.networks import Network


class CodexRelayError(RuntimeError):
    pass


@dataclass
class _RelayBudget:
    max_requests: int
    max_total_tokens: int
    max_concurrent_requests: int
    max_duration_sec: float
    started_at: float = field(default_factory=time.monotonic)
    requests: int = 0
    total_tokens: int = 0
    active_requests: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def reserve(self) -> str | None:
        with self.lock:
            if time.monotonic() - self.started_at >= self.max_duration_sec:
                return "relay duration budget exhausted"
            if self.requests >= self.max_requests:
                return "relay request budget exhausted"
            if self.total_tokens >= self.max_total_tokens:
                return "relay token budget exhausted"
            if self.active_requests >= self.max_concurrent_requests:
                return "relay concurrency budget exhausted"
            self.requests += 1
            self.active_requests += 1
            return None

    def finish(self, total_tokens: int) -> None:
        with self.lock:
            self.active_requests = max(0, self.active_requests - 1)
            self.total_tokens += max(0, total_tokens)

    def snapshot(self) -> dict[str, int | float]:
        with self.lock:
            return {
                "requests": self.requests,
                "total_tokens": self.total_tokens,
                "active_requests": self.active_requests,
                "elapsed_sec": round(time.monotonic() - self.started_at, 3),
            }


class _RelayServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int] | str,
        *,
        auth_path: Path,
        model: str,
        upstream_url: str,
        allowed_client_ip: str | None,
        log_path: Path | None,
        proxy_url: str | None,
        budget: _RelayBudget,
    ) -> None:
        super().__init__(server_address, _RelayHandler)
        self.auth_path = auth_path
        self.model = model
        self.upstream_url = upstream_url.rstrip("/") + "/responses"
        self.allowed_client_ip = allowed_client_ip
        self.log_path = log_path
        self.proxy_url = proxy_url
        self.budget = budget
        self.log_lock = threading.Lock()

    def record(self, payload: dict[str, object]) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_lock, self.log_path.open("a") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")


class _UnixRelayServer(_RelayServer):
    address_family = socket.AF_UNIX

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        self.server_name = "localhost"
        self.server_port = 0


class _RelayHandler(BaseHTTPRequestHandler):
    server: _RelayServer
    protocol_version = "HTTP/1.1"
    MAX_REQUEST_BYTES = 32 * 1024 * 1024

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def _client_id(self) -> str:
        if isinstance(self.client_address, tuple) and self.client_address:
            return str(self.client_address[0])
        return "unix-socket"

    def _error(self, status: int, message: str) -> None:
        body = json.dumps({"error": {"message": message}}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        if (
            self.server.allowed_client_ip is not None
            and self._client_id() != self.server.allowed_client_ip
        ):
            self._error(403, "relay client is not the assigned task container")
            return False
        return True

    @staticmethod
    def _has_disallowed_remote_tool(payload: dict[str, object]) -> bool:
        tools = payload.get("tools") or []
        if not isinstance(tools, list):
            return True
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            tool_type = str(tool.get("type", "")).lower()
            if any(
                capability in tool_type
                for capability in (
                    "browser",
                    "code_interpreter",
                    "computer",
                    "file_search",
                    "image_generation",
                    "mcp",
                    "web_search",
                )
            ):
                return True
        return False

    @staticmethod
    def _has_remote_resource_reference(value: object) -> bool:
        if isinstance(value, list):
            return any(
                _RelayHandler._has_remote_resource_reference(item) for item in value
            )
        if not isinstance(value, dict):
            return False
        for key, item in value.items():
            if (
                key in {"file_url", "image_url"}
                and isinstance(item, str)
                and item.lower().startswith(("http://", "https://"))
            ):
                return True
            if _RelayHandler._has_remote_resource_reference(item):
                return True
        return False

    @staticmethod
    def _remove_shell_escalation(value: object) -> int:
        """Restrict model-visible shell schemas to the existing sandbox."""
        if isinstance(value, list):
            return sum(_RelayHandler._remove_shell_escalation(item) for item in value)
        if not isinstance(value, dict):
            return 0
        restricted = 0
        properties = value.get("properties")
        if isinstance(properties, dict) and "sandbox_permissions" in properties:
            properties["sandbox_permissions"] = {
                "type": "string",
                "enum": ["use_default"],
                "description": "PitBench requires the existing workspace sandbox.",
            }
            restricted += 1
        return restricted + sum(
            _RelayHandler._remove_shell_escalation(item) for item in value.values()
        )

    def _load_auth(self) -> tuple[str, str]:
        try:
            payload = json.loads(self.server.auth_path.read_text())
            tokens = payload["tokens"]
            access_token = tokens["access_token"]
            account_id = tokens["account_id"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise CodexRelayError("host Codex OAuth file is invalid") from error
        if not isinstance(access_token, str) or not access_token:
            raise CodexRelayError("host Codex access token is missing")
        if not isinstance(account_id, str) or not account_id:
            raise CodexRelayError("host Codex account id is missing")
        return access_token, account_id

    @staticmethod
    def _event_requests_shell_escalation(line: bytes) -> bool:
        if b"require_escalated" not in line:
            return False
        data = line.strip()
        if data.startswith(b"data:"):
            data = data[5:].strip()
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            return False
        if not isinstance(event, dict):
            return False
        event_type = str(event.get("type", ""))
        if any(
            kind in event_type
            for kind in ("function_call_arguments", "custom_tool_call_input")
        ):
            return True

        def custom_call_contains(value: object) -> bool:
            if isinstance(value, list):
                return any(custom_call_contains(item) for item in value)
            if not isinstance(value, dict):
                return False
            if value.get("type") in {"function_call", "custom_tool_call"}:
                return b"require_escalated" in json.dumps(value).encode()
            return any(custom_call_contains(item) for item in value.values())

        return custom_call_contains(event)

    @staticmethod
    def _event_total_tokens(line: bytes) -> int:
        data = line.strip()
        if data.startswith(b"data:"):
            data = data[5:].strip()
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            return 0
        if not isinstance(event, dict):
            return 0
        candidates: list[object] = [event.get("usage")]
        response = event.get("response")
        if isinstance(response, dict):
            candidates.append(response.get("usage"))
        for usage in candidates:
            if not isinstance(usage, dict):
                continue
            value = usage.get("total_tokens")
            if isinstance(value, int) and value >= 0:
                return value
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
            if isinstance(input_tokens, int) and isinstance(output_tokens, int):
                return max(0, input_tokens) + max(0, output_tokens)
        return 0

    def _copy_response(
        self, source: BinaryIO, headers: object, status: int
    ) -> tuple[bool, int]:
        self.send_response(status)
        content_type = getattr(headers, "get", lambda _name: None)("Content-Type")
        if content_type:
            self.send_header("Content-Type", content_type)
        request_id = getattr(headers, "get", lambda _name: None)("OpenAI-Request-ID")
        if request_id:
            self.send_header("OpenAI-Request-ID", request_id)
        self.send_header("Connection", "close")
        self.end_headers()
        pending = b""
        total_tokens = 0
        while chunk := source.read(64 * 1024):
            pending += chunk
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                if self._event_requests_shell_escalation(line):
                    self.close_connection = True
                    return False, total_tokens
                total_tokens = max(total_tokens, self._event_total_tokens(line))
                self.wfile.write(line + b"\n")
                self.wfile.flush()
        if pending:
            if self._event_requests_shell_escalation(pending):
                self.close_connection = True
                return False, total_tokens
            total_tokens = max(total_tokens, self._event_total_tokens(pending))
            self.wfile.write(pending)
            self.wfile.flush()
        self.close_connection = True
        return True, total_tokens

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/responses":
            self._error(404, "relay exposes only POST /responses")
            return
        if not self._authorized():
            return
        budget_error = self.server.budget.reserve()
        if budget_error is not None:
            self.server.record(
                {
                    "client_ip": self._client_id(),
                    "error": budget_error,
                    "path": self.path,
                    "status": 429,
                    **self.server.budget.snapshot(),
                }
            )
            self._error(429, budget_error)
            return
        response_tokens = 0
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._error(400, "invalid Content-Length")
            self.server.budget.finish(response_tokens)
            return
        if length <= 0 or length > self.MAX_REQUEST_BYTES:
            self._error(413, "invalid relay request size")
            self.server.budget.finish(response_tokens)
            return
        body = self.rfile.read(length)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self._error(400, "request body must be JSON")
            self.server.budget.finish(response_tokens)
            return
        if not isinstance(payload, dict):
            self._error(400, "request body must be an object")
            self.server.budget.finish(response_tokens)
            return
        if payload.get("model") != self.server.model:
            self._error(403, "relay request used an unassigned model")
            self.server.budget.finish(response_tokens)
            return
        if self._has_disallowed_remote_tool(payload):
            self._error(403, "server-side network tools are disabled for PitBench")
            self.server.budget.finish(response_tokens)
            return
        if self._has_remote_resource_reference(payload):
            self._error(403, "remote model input resources are disabled for PitBench")
            self.server.budget.finish(response_tokens)
            return
        restricted_shell_fields = self._remove_shell_escalation(payload.get("tools"))
        if restricted_shell_fields:
            body = json.dumps(payload, separators=(",", ":")).encode()
        forwarded_length = len(body)

        try:
            access_token, account_id = self._load_auth()
            headers = {
                "Authorization": f"Bearer {access_token}",
                "ChatGPT-Account-ID": account_id,
                "Content-Type": "application/json",
                "Accept": self.headers.get("Accept", "text/event-stream"),
                "User-Agent": self.headers.get("User-Agent", "pitbench-codex-relay"),
            }
            for name in ("OpenAI-Beta", "Originator"):
                value = self.headers.get(name)
                if value:
                    headers[name] = value
            request = urllib.request.Request(
                self.server.upstream_url,
                data=body,
                headers=headers,
                method="POST",
            )
            proxy_handler = (
                urllib.request.ProxyHandler(
                    {"http": self.server.proxy_url, "https": self.server.proxy_url}
                )
                if self.server.proxy_url
                else urllib.request.ProxyHandler()
            )
            opener = urllib.request.build_opener(proxy_handler)
            with opener.open(request, timeout=900) as response:
                self.server.record(
                    {
                        "client_ip": self._client_id(),
                        "model": payload.get("model"),
                        "path": self.path,
                        "request_bytes": forwarded_length,
                        "shell_permission_fields_restricted": restricted_shell_fields,
                        "status": response.status,
                    }
                )
                copied, response_tokens = self._copy_response(
                    response, response.headers, response.status
                )
                if not copied:
                    self.server.record(
                        {
                            "client_ip": self._client_id(),
                            "error": "model attempted shell permission escalation",
                            "model": payload.get("model"),
                            "path": self.path,
                            "status": "blocked_response",
                        }
                    )
        except urllib.error.HTTPError as error:
            self.server.record(
                {
                    "client_ip": self._client_id(),
                    "model": payload.get("model"),
                    "path": self.path,
                    "request_bytes": forwarded_length,
                    "shell_permission_fields_restricted": restricted_shell_fields,
                    "status": error.code,
                }
            )
            _, response_tokens = self._copy_response(error, error.headers, error.code)
        except (CodexRelayError, OSError) as error:
            self.server.record(
                {
                    "client_ip": self._client_id(),
                    "error": str(error),
                    "model": payload.get("model"),
                    "path": self.path,
                    "request_bytes": forwarded_length,
                    "shell_permission_fields_restricted": restricted_shell_fields,
                    "status": 502,
                }
            )
            self._error(502, f"model relay failed: {error}")
        finally:
            self.server.budget.finish(response_tokens)
            self.server.record(
                {
                    "client_ip": self._client_id(),
                    "model": payload.get("model")
                    if isinstance(payload, dict)
                    else None,
                    "path": self.path,
                    "response_tokens": response_tokens,
                    "status": "budget_accounting",
                    **self.server.budget.snapshot(),
                }
            )


class CodexModelRelay(AbstractContextManager["CodexModelRelay"]):
    """Single-task Responses relay running outside the solver container."""

    PORT = 8765

    def __init__(
        self,
        *,
        auth_path: Path,
        model: str,
        allowed_client_ip: str,
        network: Network,
        image: str,
        log_path: Path | None = None,
        upstream_url: str = "https://chatgpt.com/backend-api/codex",
        proxy_url: str | None = None,
        max_requests: int = 256,
        max_total_tokens: int = 10_000_000,
        max_concurrent_requests: int = 1,
        max_duration_sec: float = 3720.0,
    ) -> None:
        self.auth_path = auth_path.expanduser().resolve()
        self.model = model
        self.allowed_client_ip = allowed_client_ip
        self.network = network
        self.image = image
        self.log_path = log_path.resolve() if log_path is not None else None
        self.upstream_url = upstream_url
        self.proxy_url = proxy_url
        self.max_requests = max_requests
        self.max_total_tokens = max_total_tokens
        self.max_concurrent_requests = max_concurrent_requests
        self.max_duration_sec = max_duration_sec
        self.container: Container | None = None
        self.container_ip: str | None = None
        self._temporary_log: Path | None = None
        self._socket_directory: tempfile.TemporaryDirectory[str] | None = None
        self._server: _UnixRelayServer | None = None
        self._thread: threading.Thread | None = None

        positive = {
            "max_requests": max_requests,
            "max_total_tokens": max_total_tokens,
            "max_concurrent_requests": max_concurrent_requests,
            "max_duration_sec": max_duration_sec,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"relay budgets must be positive: {', '.join(invalid)}")

    @property
    def url(self) -> str:
        if self.container_ip is None:
            raise RuntimeError("Codex relay sidecar is not running")
        return f"http://{self.container_ip}:{self.PORT}"

    @property
    def port(self) -> int:
        return self.PORT

    def metadata(self) -> dict[str, object]:
        return {
            "topology": "internal-network -> credential-free-frontend -> unix-socket -> host-relay",
            "frontend_ip": self.container_ip,
            "frontend_has_credentials": False,
            "max_requests": self.max_requests,
            "max_total_tokens": self.max_total_tokens,
            "max_concurrent_requests": self.max_concurrent_requests,
            "max_duration_sec": self.max_duration_sec,
        }

    def _stage_files(self) -> None:
        if self.container is None:
            raise RuntimeError("relay sidecar is not running")
        files = (
            (
                "/opt/pitbench",
                "codex_relay.py",
                Path(__file__).read_bytes(),
                0o444,
            ),
        )
        writer = (
            "import base64,os,sys; "
            "path=sys.argv[1]; "
            "open(path,'wb').write(base64.b64decode(sys.argv[2])); "
            "os.chmod(path,int(sys.argv[3],8))"
        )
        for destination, name, content, mode in files:
            result = self.container.exec_run(
                [
                    "python3",
                    "-c",
                    writer,
                    f"{destination}/{name}",
                    base64.b64encode(content).decode(),
                    oct(mode),
                ],
                demux=True,
            )
            if result.exit_code != 0:
                detail = result.output
                raise CodexRelayError(f"could not stage relay sidecar files: {detail}")

    def __enter__(self) -> CodexModelRelay:
        import docker

        if not self.auth_path.is_file():
            raise CodexRelayError(f"Codex OAuth file is unavailable: {self.auth_path}")
        if self.log_path is None:
            self._temporary_log = Path(
                f"/tmp/pitbench-codex-relay-{os.getpid()}-{time.monotonic_ns()}.jsonl"
            )
            self.log_path = self._temporary_log
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.touch(exist_ok=True)
        self._socket_directory = tempfile.TemporaryDirectory(
            prefix="pitbench-codex-relay-"
        )
        socket_root = Path(self._socket_directory.name)
        socket_root.chmod(0o711)
        socket_path = socket_root / "relay.sock"
        budget = _RelayBudget(
            max_requests=self.max_requests,
            max_total_tokens=self.max_total_tokens,
            max_concurrent_requests=self.max_concurrent_requests,
            max_duration_sec=self.max_duration_sec,
        )
        self._server = _UnixRelayServer(
            str(socket_path),
            auth_path=self.auth_path,
            model=self.model,
            upstream_url=self.upstream_url,
            allowed_client_ip=None,
            log_path=self.log_path,
            proxy_url=self.proxy_url,
            budget=budget,
        )
        socket_path.chmod(0o666)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="pitbench-codex-relay",
            daemon=True,
        )
        self._thread.start()
        relay_args = [
            "/opt/pitbench/codex_relay.py",
            "bridge",
            "--unix-socket",
            "/relay/relay.sock",
            "--allowed-client-ip",
            self.allowed_client_ip,
            "--port",
            str(self.PORT),
        ]
        bootstrap = (
            "import os,sys,time\n"
            "paths=('/opt/pitbench/codex_relay.py',)\n"
            "deadline=time.monotonic()+15\n"
            "while not all(os.path.exists(p) for p in paths):\n"
            "    if time.monotonic() >= deadline: raise SystemExit(70)\n"
            "    time.sleep(.05)\n"
            "os.execvp('python3',['python3',*sys.argv[1:]])\n"
        )
        command = ["python3", "-c", bootstrap, *relay_args]
        client = docker.from_env()
        try:
            self.container = client.containers.run(
                self.image,
                command,
                name=f"pitbench-codex-relay-{os.getpid()}-{time.monotonic_ns()}",
                detach=True,
                network=self.network.name,
                read_only=True,
                cap_drop=["ALL"],
                security_opt=["no-new-privileges:true"],
                pids_limit=128,
                mem_limit="256m",
                tmpfs={
                    "/opt/pitbench": "rw,noexec,nosuid,size=4m",
                    "/tmp": "rw,noexec,nosuid,size=16m",
                },
                volumes={str(socket_root): {"bind": "/relay", "mode": "ro"}},
            )
            self._stage_files()
            self.network.reload()
            self.container.reload()
            endpoint = self.container.attrs["NetworkSettings"]["Networks"].get(
                self.network.name, {}
            )
            self.container_ip = endpoint.get("IPAddress") or None
            if self.container_ip is None:
                member = self.network.attrs.get("Containers", {}).get(
                    self.container.id, {}
                )
                address = str(member.get("IPv4Address", ""))
                self.container_ip = address.partition("/")[0] or None
            if not self.container_ip:
                raise CodexRelayError("could not resolve relay sidecar address")
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                self.container.reload()
                if self.container.status != "running":
                    detail = self.container.logs(tail=50).decode(errors="replace")
                    raise CodexRelayError(f"relay sidecar exited: {detail}")
                ready = self.container.exec_run(
                    [
                        "python3",
                        "-c",
                        (
                            "import socket; "
                            f"socket.create_connection(('127.0.0.1',{self.PORT}),1).close()"
                        ),
                    ]
                )
                if ready.exit_code == 0:
                    return self
                time.sleep(0.1)
            raise CodexRelayError("relay sidecar did not become ready")
        except Exception:
            self.__exit__()
            raise

    def __exit__(self, *args: object) -> None:
        if self.container is not None:
            try:
                self.network.disconnect(self.container, force=True)
            except Exception:
                pass
            try:
                self.container.remove(force=True)
            except Exception:
                pass
            self.container = None
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        if self._socket_directory is not None:
            self._socket_directory.cleanup()
            self._socket_directory = None
        if self._temporary_log is not None:
            try:
                self._temporary_log.unlink()
            except OSError:
                pass
            self._temporary_log = None


def _serve(args: argparse.Namespace) -> None:
    budget = _RelayBudget(
        max_requests=args.max_requests,
        max_total_tokens=args.max_total_tokens,
        max_concurrent_requests=args.max_concurrent_requests,
        max_duration_sec=args.max_duration_sec,
    )
    server = _RelayServer(
        (args.bind, args.port),
        auth_path=args.auth_path,
        model=args.model,
        upstream_url=args.upstream_url,
        allowed_client_ip=args.allowed_client_ip,
        log_path=args.log_path,
        proxy_url=args.proxy_url,
        budget=budget,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _bridge(args: argparse.Namespace) -> None:
    unix_socket = str(args.unix_socket)
    allowed_client_ip = args.allowed_client_ip

    class Handler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            if self.client_address[0] != allowed_client_ip:
                return
            upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                upstream.connect(unix_socket)
                peers = (self.request, upstream)
                while True:
                    readable, _, _ = select.select(peers, (), ())
                    for source in readable:
                        data = source.recv(64 * 1024)
                        if not data:
                            return
                        target = upstream if source is self.request else self.request
                        target.sendall(data)
            finally:
                upstream.close()

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    with Server((args.bind, args.port), Handler) as server:
        server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser("serve")
    serve.add_argument("--auth-path", type=Path, required=True)
    serve.add_argument("--model", required=True)
    serve.add_argument("--allowed-client-ip", required=True)
    serve.add_argument("--log-path", type=Path, required=True)
    serve.add_argument("--upstream-url", required=True)
    serve.add_argument("--proxy-url")
    serve.add_argument("--bind", default="0.0.0.0")
    serve.add_argument("--port", type=int, required=True)
    serve.add_argument("--max-requests", type=int, required=True)
    serve.add_argument("--max-total-tokens", type=int, required=True)
    serve.add_argument("--max-concurrent-requests", type=int, required=True)
    serve.add_argument("--max-duration-sec", type=float, required=True)
    bridge = subparsers.add_parser("bridge")
    bridge.add_argument("--unix-socket", type=Path, required=True)
    bridge.add_argument("--allowed-client-ip", required=True)
    bridge.add_argument("--bind", default="0.0.0.0")
    bridge.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if args.command == "serve":
        _serve(args)
    elif args.command == "bridge":
        _bridge(args)


if __name__ == "__main__":
    main()
