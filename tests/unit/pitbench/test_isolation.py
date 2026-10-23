from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pitbench.evaluator.docker_judge import DockerJudge
from pitbench.evaluator.isolation import (
    CANDIDATE_GID,
    IsolationError,
    _restrict,
    execute,
    protect_output,
)
from pitbench.repositories.base import CommandSpec


def _python(workspace, script, *, readable=(), writable=(), timeout=5, **kwargs):
    return execute(
        CommandSpec(argv=[sys.executable, "-c", script], timeout_sec=timeout),
        workspace,
        readable=readable,
        writable=writable,
        **kwargs,
    )


@pytest.mark.parametrize(
    "via", ["direct", "traversal", "symlink", "proc_root", "subprocess"]
)
def test_private_file_reads_are_denied_to_candidate_and_descendants(tmp_path, via):
    work = tmp_path / "work"
    work.mkdir()
    private = tmp_path / "private"
    private.mkdir()
    secret = private / "oracle.json"
    secret.write_text("hidden-answer-marker")
    (work / "alias").symlink_to(secret)
    path = {
        "direct": str(secret),
        "traversal": str(work / ".." / "private" / "oracle.json"),
        "symlink": str(work / "alias"),
        "proc_root": f"/proc/self/root{secret}",
        "subprocess": str(secret),
    }[via]
    script = f"from pathlib import Path; print(Path({path!r}).read_text())"
    if via == "subprocess":
        script = f"import subprocess,sys; sys.exit(subprocess.call([sys.executable,'-c',{script!r}]))"
    result = _python(work, script)
    assert result.returncode != 0
    assert "hidden-answer-marker" not in result.stdout + result.stderr
    assert "Permission denied" in result.stderr
    assert secret.read_text() == "hidden-answer-marker"


def test_instance_is_read_only_and_other_runs_are_invisible(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    instance = tmp_path / "input.json"
    instance.write_text('{"jobs": 4}')
    old = tmp_path / "old-output.json"
    old.write_text("private old result")
    output = tmp_path / "output"
    output.mkdir()
    script = f"""
from pathlib import Path
instance=Path({str(instance)!r})
assert instance.read_text() == '{{"jobs": 4}}'
for path, operation in [(instance, 'write'), (Path({str(old)!r}), 'read')]:
    try:
        path.write_text('changed') if operation == 'write' else path.read_text()
    except PermissionError:
        pass
    else:
        raise AssertionError('access unexpectedly allowed')
Path({str(output / "result.json")!r}).write_text('{{"valid":true}}')
"""
    result = _python(work, script, readable=(instance,), writable=(output,))
    assert result.returncode == 0, result.stderr
    assert (output / "result.json").is_file()
    assert instance.read_text() == '{"jobs": 4}'


def test_parent_environment_and_preopened_descriptors_do_not_cross_domain(
    tmp_path, monkeypatch
):
    work = tmp_path / "work"
    work.mkdir()
    secret = tmp_path / "secret"
    secret.write_text("credential-marker")
    monkeypatch.setenv("GITHUB_TOKEN", "credential-marker")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "credential-marker")
    with secret.open() as handle:
        os.set_inheritable(handle.fileno(), True)
        script = f"""
import os
from pathlib import Path
assert not os.environ.get('GITHUB_TOKEN')
assert not os.environ.get('ANTHROPIC_API_KEY')
for path in [Path('/proc/{os.getpid()}/environ'), Path('/proc/{os.getpid()}/fd/{handle.fileno()}')]:
    try:
        value=path.read_bytes()
    except (PermissionError, FileNotFoundError):
        continue
    assert b'credential-marker' not in value
print('clean')
"""
        result = _python(work, script)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "clean\n"


def test_network_control_sockets_and_signals_cannot_reach_verifier(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        script = f"""
import os,signal,socket
for family in [socket.AF_INET, socket.AF_INET6, socket.AF_UNIX]:
    try:
        socket.socket(family, socket.SOCK_STREAM)
    except PermissionError:
        pass
    else:
        raise AssertionError('socket unexpectedly allowed')
try:
    os.kill({victim.pid}, signal.SIGTERM)
except PermissionError:
    pass
else:
    raise AssertionError('cross-domain signal unexpectedly allowed')
print('blocked')
"""
        result = _python(work, script)
        assert result.returncode == 0, result.stderr
        assert victim.poll() is None
    finally:
        victim.kill()
        victim.wait()


@pytest.mark.parametrize("timeout", [False, True])
def test_supervisor_reaps_daemons_that_escape_the_original_session(tmp_path, timeout):
    work = tmp_path / "work"
    work.mkdir()
    daemon = (
        "import os,time; os.setsid(); print(os.getpid(),flush=True); time.sleep(30)"
    )
    script = f"""
import subprocess,sys,time
child=subprocess.Popen([sys.executable,'-c',{daemon!r}],stdout=subprocess.PIPE,text=True)
print(child.stdout.readline(),end='',flush=True)
{"time.sleep(30)" if timeout else ""}
"""
    if timeout:
        with pytest.raises(subprocess.TimeoutExpired) as caught:
            _python(work, script, timeout=0.5)
        result = caught.value
    else:
        result = _python(work, script)
    pid = int(result.stdout.strip())
    assert not Path(f"/proc/{pid}").exists()
    assert result.isolation["descendants_remaining"] == 0
    assert result.isolation["descendants_killed"] >= 1


def test_logs_are_drained_and_truncation_is_explicit(tmp_path):
    result = _python(
        tmp_path,
        "import sys; print('x'*100000); print('y'*100000,file=sys.stderr)",
        log_bytes=1024,
    )
    assert result.returncode == 0
    assert len(result.stdout) == len(result.stderr) == 1024
    assert (
        result.isolation["stdout_bytes"] == result.isolation["stderr_bytes"] == 100001
    )
    assert result.isolation["stdout_truncated"] and result.isolation["stderr_truncated"]


def test_candidate_startup_hooks_run_only_after_restrictions(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    escaped = tmp_path / "escaped"
    (work / "sitecustomize.py").write_text(
        f"from pathlib import Path; Path({str(escaped)!r}).write_text('escaped')"
    )
    command = CommandSpec(
        argv=[sys.executable, "-c", "print('ran')"], env={"PYTHONPATH": str(work)}
    )
    result = execute(command, work)
    assert result.returncode == 0
    assert not escaped.exists()


def test_old_kernels_fail_closed_before_exec(monkeypatch):
    class OldKernel:
        def syscall(self, *args):
            return 5

    monkeypatch.setattr(
        "pitbench.evaluator.isolation.ctypes.CDLL", lambda *args, **kwargs: OldKernel()
    )
    with pytest.raises(IsolationError, match="ABI >= 6"):
        _restrict([], [])


def test_secret_keys_cannot_be_added_to_the_child_environment(tmp_path):
    command = CommandSpec(
        argv=[sys.executable, "-c", "print('must not run')"],
        env={"GITHUB_TOKEN": "must-not-pass"},
    )
    with pytest.raises(IsolationError, match="unsupported child environment keys"):
        execute(command, tmp_path)


def test_controller_mounts_exclude_reference_data_and_secrets(tmp_path):
    root = Path(__file__).resolve().parents[3]
    task = tmp_path / "task.yaml"
    task.write_text("task_id: test\n")
    patch = tmp_path / "candidate.patch"
    patch.write_text("")
    judge = DockerJudge(
        image="sha256:" + "a" * 64,
        task_config_path=task,
        base_repository=tmp_path,
        private_root=tmp_path,
        candidate_patch=patch,
        output_dir=tmp_path,
    )
    command = judge.command()
    mounts = [
        command[i + 1] for i, value in enumerate(command[:-1]) if value == "--volume"
    ]
    assert not any(value.startswith(str(root) + ":") for value in mounts)
    assert not any(
        "docker.sock" in value
        or "/.pitbench/" in value
        or "configs/references" in value
        for value in mounts
    )
    assert any("/run/pitbench-private/assets:ro" in value for value in mounts)
    assert "no-new-privileges:true" in command
    assert "--privileged" not in command
    assert command[command.index("--user") + 1] == "0:0"
    assert "env -i" in command[command.index("-c") + 1]
    assert judge.checkpoint_identity()["isolation_version"]


def test_seed_validation_reuses_the_same_private_mount_boundary(tmp_path, monkeypatch):
    from scripts.validate_seed_robustness_real_solver import _run_in_judge_image

    output = tmp_path / "output"
    output.mkdir()
    private = tmp_path / "private-answer"
    private.write_text("unchanged")
    (output / "no_change.patch").symlink_to(private)
    task = tmp_path / "task.yaml"
    task.write_text("task_id: test\n")
    captured = []
    monkeypatch.setattr(
        "scripts.validate_seed_robustness_real_solver.subprocess.run",
        lambda command, **kwargs: captured.append(command),
    )
    args = SimpleNamespace(
        repository=tmp_path,
        judge_cpus=2,
        judge_memory="2g",
        judge_image="sha256:" + "a" * 64,
        task_config=task,
        private_root=tmp_path,
        output_dir=output,
        parallel_runs=1,
        instance_limit=2,
        reference_seed_count=3,
        test_seed_count=4,
        test_list_count=5,
        generation_seed=6,
    )
    _run_in_judge_image(args)
    (command,) = captured
    assert (
        command[command.index("--private-root") + 1] == "/run/pitbench-private/assets"
    )
    assert "--artifact-gid" in command
    assert command[command.index("--reference-seed-count") + 1] == "3"
    assert command[command.index("--generation-seed") + 1] == "6"
    assert private.read_text() == "unchanged"
    assert not (output / "no_change.patch").is_symlink()


def test_reserved_candidate_group_cannot_own_published_outputs(tmp_path):
    with pytest.raises(IsolationError, match="reserved candidate group"):
        protect_output(tmp_path, CANDIDATE_GID)


def test_cross_process_memory_descriptors_and_resource_controls_are_denied(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    victim = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import ctypes,time; value=ctypes.create_string_buffer(b'private-memory-marker'); print(ctypes.addressof(value),flush=True); time.sleep(30)",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        address = int(victim.stdout.readline())
        script = f"""
import ctypes,errno,os,resource
class Vector(ctypes.Structure):
    _fields_=[('base',ctypes.c_void_p),('length',ctypes.c_size_t)]
libc=ctypes.CDLL(None,use_errno=True)
libc.process_vm_readv.restype=ctypes.c_ssize_t
data=ctypes.create_string_buffer(20)
local=Vector(ctypes.addressof(data),20)
remote=Vector({address},20)
result=libc.process_vm_readv({victim.pid},ctypes.byref(local),1,ctypes.byref(remote),1,0)
assert result == -1 and ctypes.get_errno() == errno.EPERM
descriptor=os.pidfd_open({victim.pid})
try:
    assert libc.syscall(438,descriptor,1,0) == -1 and ctypes.get_errno() == errno.EPERM
finally:
    os.close(descriptor)
try:
    resource.prlimit({victim.pid},resource.RLIMIT_NOFILE,(128,128))
except PermissionError:
    pass
else:
    raise AssertionError('cross-domain resource control allowed')
"""
        result = _python(work, script)
        assert result.returncode == 0, result.stderr
        assert victim.poll() is None
    finally:
        victim.kill()
        victim.wait()
