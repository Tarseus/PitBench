from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

import docker
from docker import errors as docker_errors
from pitbench.harness.agents.codex_mcp_agent import CodexMCPAgent
from pitbench.harness.agents.codex_relay import CodexModelRelay
from pitbench.harness.agents.codex_workspace import CodexWorkspaceRuntime

pytestmark = pytest.mark.docker


@pytest.mark.timeout(240)
def test_live_codex_workspace_relay_only_shell_network(tmp_path: Path) -> None:
    if os.environ.get("PITBENCH_CODEX_LIVE_SMOKE") != "1":
        pytest.skip("set PITBENCH_CODEX_LIVE_SMOKE=1 to use the live Codex account")
    codex = shutil.which("codex")
    auth_path = Path(
        os.environ.get("PITBENCH_CODEX_AUTH_PATH", "~/.codex/auth.json")
    ).expanduser()
    if codex is None or not auth_path.is_file():
        pytest.skip("host Codex CLI and OAuth login are required")
    client = docker.from_env()
    image = "tb__pyvrp_v0_14_0__client:latest"
    try:
        client.images.get(image)
    except docker_errors.ImageNotFound:
        pytest.skip(f"local smoke-test image is unavailable: {image}")
    container = client.containers.run(
        image,
        ["bash", "-lc", "while true; do sleep 30; done"],
        name=f"pitbench-codex-workspace-test-{uuid.uuid4().hex[:12]}",
        detach=True,
        network_mode="none",
        cap_drop=["ALL"],
        cap_add=["SETGID", "SETUID", "SETFCAP"],
        security_opt=[
            "no-new-privileges:true",
            "seccomp=unconfined",
            "apparmor=unconfined",
        ],
    )
    agent = CodexMCPAgent(
        model_name=os.environ.get("PITBENCH_CODEX_SMOKE_MODEL", "gpt-5.6-luna"),
        codex_auth_path=str(auth_path),
        runner_backend="workspace",
        reasoning_effort="low",
        control_plane_timeout_sec=180,
        proxy_url=os.environ.get("PITBENCH_CODEX_PROXY_URL"),
    )
    runtime = CodexWorkspaceRuntime(
        container=container,
        codex_binary=Path(codex),
        profile=None,
    )
    try:
        with runtime:
            assert runtime.container_ip is not None
            assert runtime.gateway_ip is not None
            assert (
                container.exec_run(
                    ["test", "!", "-e", f"{runtime.codex_home}/auth.json"]
                ).exit_code
                == 0
            )
            sandbox_write = container.exec_run(
                [
                    runtime.container_codex,
                    "sandbox",
                    "-P",
                    ":workspace",
                    "-C",
                    "/workspace/repo",
                    "bash",
                    "-lc",
                    "touch .pitbench-codex-write-probe && "
                    "rm .pitbench-codex-write-probe",
                ],
                environment={"CODEX_HOME": runtime.codex_home, "HOME": runtime.root},
                demux=True,
            )
            assert sandbox_write.exit_code == 0, sandbox_write.output
            with CodexModelRelay(
                auth_path=auth_path,
                model=agent._model_name,
                allowed_client_ip=runtime.container_ip,
                network=runtime.network,
                image=container.image.id,
                log_path=tmp_path / "relay.jsonl",
                proxy_url=agent._proxy_url,
            ) as relay:
                assert relay.container_ip is not None
                runtime.configure_relay(relay.container_ip)
                agent._check_workspace_isolation(
                    runtime=runtime,
                    relay=relay,
                    env=agent._runtime_env(),
                    logging_dir=tmp_path,
                )
                model_probe = subprocess.run(
                    [
                        *agent._workspace_exec_prefix(runtime=runtime, relay=relay),
                        "--",
                        "Do not use tools. Reply with exactly PITBENCH_MODEL_OK.",
                    ],
                    env=agent._runtime_env(),
                    capture_output=True,
                    text=True,
                    timeout=180,
                    check=False,
                )
                assert model_probe.returncode == 0, model_probe.stderr
                assert agent._has_completed_turn(model_probe.stdout)
        assert container.exec_run(["test", "!", "-e", runtime.root]).exit_code == 0
        relay_log = (tmp_path / "relay.jsonl").read_text()
        assert '"status": 200' in relay_log
        assert '"status": "budget_accounting"' in relay_log
        assert '"total_tokens": ' in relay_log
        preflight_log = (tmp_path / "codex-preflight.log").read_text()
        assert (
            "PITBENCH_RELAY_REACHABLE" in preflight_log
            and "PITBENCH_PUBLIC_BLOCKED" in preflight_log
        )
    finally:
        container.remove(force=True)
