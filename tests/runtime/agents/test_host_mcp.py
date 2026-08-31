from __future__ import annotations

import time
import uuid

import pytest

import docker
from docker import errors as docker_errors
from pitbench.harness.agents.host_mcp import TaskTerminal

pytestmark = pytest.mark.docker


@pytest.fixture
def task_container():
    client = docker.from_env()
    image = "tb__pyvrp_v0_14_0__client:latest"
    try:
        client.images.get(image)
    except docker_errors.ImageNotFound:
        pytest.skip(f"local smoke-test image is unavailable: {image}")
    container = client.containers.run(
        image,
        ["bash", "-lc", "while true; do sleep 30; done"],
        name=f"pitbench-host-mcp-test-{uuid.uuid4().hex[:12]}",
        detach=True,
        network_mode="none",
    )
    workdir = "/tmp/pitbench-host-mcp-test/repo"
    setup = container.exec_run(
        [
            "bash",
            "-lc",
            (
                f"mkdir -p {workdir}/src && cd {workdir} && git init -q && "
                "git config user.email pitbench@example.test && "
                "git config user.name PitBench && "
                "printf 'alpha\\nbeta\\n' > src/example.txt && "
                "git add . && git commit -qm initial"
            ),
        ]
    )
    if setup.exit_code != 0:
        container.remove(force=True)
        pytest.fail(setup.output.decode(errors="replace"))
    try:
        yield container, workdir
    finally:
        container.remove(force=True)


def test_task_terminal_tools_against_real_offline_container(task_container):
    container, workdir = task_container
    terminal = TaskTerminal(container, workdir=workdir)

    read = terminal.read_file("src/example.txt", offset=0, limit=6)
    assert read["content"] == "alpha\n"
    assert read["eof"] is False
    assert terminal.list_files("src")["files"] == ["src/example.txt"]
    assert terminal.search_files("beta", glob="*.txt")["matches"]

    applied = terminal.apply_patch(
        """--- a/src/example.txt
+++ b/src/example.txt
@@ -1,2 +1,2 @@
 alpha
-beta
+gamma
"""
    )
    assert applied["applied"] is True
    assert terminal.git_status()["clean"] is False
    assert "+gamma" in terminal.git_diff()["diff"]

    started = terminal.start_command(
        "printf first; sleep 0.2; printf second", timeout_sec=5
    )
    assert started["streaming"] is True
    deadline = time.monotonic() + 5
    observed = ""
    offset = 0
    while time.monotonic() < deadline:
        polled = terminal.poll_command(started["handle"], stdout_offset=offset)
        observed += polled["stdout"]
        offset = polled["next_stdout_offset"]
        if polled["done"]:
            break
        time.sleep(0.02)
    assert polled["done"] is True
    assert polled["exit_code"] == 0
    assert observed == "firstsecond"

    sleeping = terminal.start_command("sleep 30", timeout_sec=60)
    terminal.cancel_command(sleeping["handle"])
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        cancelled = terminal.poll_command(sleeping["handle"])
        if cancelled["done"]:
            break
        time.sleep(0.02)
    assert cancelled["done"] is True
    assert cancelled["cancelled"] is True
    terminal.close()
