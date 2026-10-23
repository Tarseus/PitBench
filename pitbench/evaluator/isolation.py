"""Linux permission domains for untrusted builds and solver processes.

This file is also the trusted launcher, entered with Python -I -S. Keep its
imports in the standard library: candidate Python startup hooks must not run
before the kernel restrictions are installed.
"""

from __future__ import annotations

import ctypes
import errno
import json
import os
import platform
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ISOLATION_VERSION = "landlock6-seccomp-v1"
PRIVATE_ASSET_MOUNT = "/run/pitbench-private/assets"
ARTIFACT_OWNER_UID = 65532
CANDIDATE_GID = 65531
DEFAULT_LOG_BYTES = 4 * 1024 * 1024
DEFAULT_ARTIFACT_BYTES = 128 * 1024 * 1024
IMAGE_ENV_KEYS = (
    "PITBENCH_CHOCO_RUNNER",
    "PITBENCH_ORTOOLS_CP_SAT_RUNNER",
    "PITBENCH_GECODE_RUNNER",
    "PITBENCH_EXACT_SOLVER",
    "PITBENCH_VROOM_SOLVER",
    "MAVEN_CONFIG",
    "JAVA_HOME",
    "PIP_NO_BUILD_ISOLATION",
)
COMMAND_ENV_KEYS = set(IMAGE_ENV_KEYS) | {
    "PATH",
    "LANG",
    "LC_ALL",
    "HOME",
    "TMPDIR",
    "PYTHONPATH",
    "PYTHONNOUSERSITE",
    "JAVA_TOOL_OPTIONS",
    "MAVEN_OPTS",
    "ASAN_OPTIONS",
    "UBSAN_OPTIONS",
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
    "CMAKE_BUILD_PARALLEL_LEVEL",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
}


class IsolationError(OSError):
    """A required isolation mechanism failed; execution must not fall back."""


def _check(result: int, operation: str) -> int:
    if result < 0:
        error = ctypes.get_errno()
        raise IsolationError(error, f"{operation}: {os.strerror(error)}")
    return result


def _restrict(readable: list[str], writable: list[str]) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    if platform.machine() not in {"x86_64", "aarch64"}:
        raise IsolationError("unsupported isolation architecture")
    abi = _check(libc.syscall(444, 0, 0, 1), "Landlock ABI query")
    if abi < 6:
        raise IsolationError(
            "Landlock ABI >= 6 is required (including signal/IPC scopes)"
        )

    class Ruleset(ctypes.Structure):
        _fields_ = [
            ("fs", ctypes.c_uint64),
            ("net", ctypes.c_uint64),
            ("scopes", ctypes.c_uint64),
        ]

    class Rule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("access", ctypes.c_uint64), ("fd", ctypes.c_int32)]

    # Handle every filesystem right through ABI 5, TCP bind/connect, and ABI 6
    # abstract-UNIX/signal scopes. Unknown later rights do not weaken these.
    ruleset = Ruleset((1 << 16) - 1, 3, 3)
    descriptor = _check(
        libc.syscall(444, ctypes.byref(ruleset), ctypes.sizeof(ruleset), 0),
        "Landlock create",
    )
    read = (1 << 0) | (1 << 2) | (1 << 3)
    write = (
        read
        | (1 << 1)
        | (1 << 4)
        | (1 << 5)
        | (1 << 7)
        | (1 << 8)
        | (1 << 12)
        | (1 << 13)
        | (1 << 14)
    )
    try:
        for paths, access in ((readable, read), (writable, write)):
            for raw in paths:
                path = Path(raw)
                if not path.exists():
                    continue
                fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
                try:
                    allowed = access
                    if not stat.S_ISDIR(os.fstat(fd).st_mode):
                        allowed &= (1 << 0) | (1 << 1) | (1 << 2) | (1 << 14)
                    _check(
                        libc.syscall(
                            445, descriptor, 1, ctypes.byref(Rule(allowed, fd)), 0
                        ),
                        f"Landlock allow {path}",
                    )
                finally:
                    os.close(fd)
        _check(libc.prctl(38, 1, 0, 0, 0), "no_new_privs")
        _check(libc.syscall(446, descriptor, 0), "Landlock restrict")
    finally:
        os.close(descriptor)

    class CapHeader(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]

    class CapData(ctypes.Structure):
        _fields_ = [
            ("effective", ctypes.c_uint32),
            ("permitted", ctypes.c_uint32),
            ("inheritable", ctypes.c_uint32),
        ]

    if os.geteuid() == 0:
        os.setgroups([])
        os.setresgid(CANDIDATE_GID, CANDIDATE_GID, CANDIDATE_GID)
    _check(
        libc.capset(ctypes.byref(CapHeader(0x20080522, 0)), (CapData * 2)()),
        "drop capabilities",
    )
    _seccomp(libc)
    return abi


def _seccomp(libc) -> None:
    """Deny networking and cross-process/namespace escape primitives."""

    class Filter(ctypes.Structure):
        _fields_ = [
            ("code", ctypes.c_ushort),
            ("jt", ctypes.c_ubyte),
            ("jf", ctypes.c_ubyte),
            ("k", ctypes.c_uint32),
        ]

    class Program(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ushort), ("filters", ctypes.POINTER(Filter))]

    x86 = platform.machine() == "x86_64"
    arch = 0xC000003E if x86 else 0xC00000B7
    forbidden = (
        [
            41,
            101,
            141,
            155,
            161,
            165,
            166,
            250,
            272,
            298,
            303,
            304,
            308,
            310,
            311,
            312,
            321,
            323,
        ]
        if x86
        else [
            40,
            41,
            51,
            97,
            117,
            140,
            198,
            219,
            241,
            264,
            265,
            268,
            270,
            271,
            272,
            280,
            282,
        ]
    )
    forbidden += [424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 438, 440, 442]
    forbidden += [251, 256, 279] if x86 else [30, 39, 238, 239]
    forbidden += (
        [29, 30, 31, 64, 65, 66, 67, 68, 69, 70, 71, 220]
        if x86
        else list(range(186, 198))
    )
    instructions = [
        Filter(0x20, 0, 0, 4),
        Filter(0x15, 1, 0, arch),
        Filter(0x06, 0, 0, 0x80000000),
        Filter(0x20, 0, 0, 0),
    ]
    if x86:
        instructions += [Filter(0x35, 0, 1, 0x40000000), Filter(0x06, 0, 0, 0x80000000)]
    for number in forbidden:
        instructions += [
            Filter(0x15, 0, 1, number),
            Filter(0x06, 0, 0, 0x50000 | errno.EPERM),
        ]
    # glibc falls back to clone when clone3 is unavailable.
    instructions += [
        Filter(0x15, 0, 1, 435),
        Filter(0x06, 0, 0, 0x50000 | errno.ENOSYS),
    ]
    # Resource/affinity operations may address self (pid 0), never the verifier
    # or another job. Returning EPERM preserves ordinary runtime fallbacks.
    for number in [302, 203, 142, 144, 314] if x86 else [261, 122, 118, 119, 274]:
        instructions += [
            Filter(0x15, 0, 4, number),
            Filter(0x20, 0, 0, 16),
            Filter(0x15, 1, 0, 0),
            Filter(0x06, 0, 0, 0x50000 | errno.EPERM),
            Filter(0x20, 0, 0, 0),
        ]
    instructions += [Filter(0x06, 0, 0, 0x7FFF0000)]
    array = (Filter * len(instructions))(*instructions)
    program = Program(len(array), array)
    _check(libc.prctl(22, 2, ctypes.byref(program), 0, 0), "seccomp filter")


def _children() -> list[int]:
    path = Path(f"/proc/self/task/{os.getpid()}/children")
    return [int(value) for value in path.read_text().split()]


def _reap_all() -> int:
    """Subreaping also catches descendants that escaped their process group."""
    killed = 0
    deadline = time.monotonic() + 10
    while True:
        for pid in _children():
            try:
                os.kill(pid, signal.SIGKILL)
                killed += 1
            except ProcessLookupError:
                pass
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return killed
        if time.monotonic() > deadline:
            raise IsolationError("descendant cleanup did not complete")
        if not pid:
            time.sleep(0.005)


def _launch(spec: dict) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    _check(libc.prctl(36, 1, 0, 0, 0), "child subreaper")
    _check(libc.prctl(4, 0, 0, 0, 0), "supervisor non-dumpable")
    abi = _check(libc.syscall(444, 0, 0, 1), "Landlock ABI query")
    pipes = [os.pipe() for _ in range(2)]
    setup_read, setup_write = os.pipe2(os.O_CLOEXEC)
    pid = os.fork()
    if pid == 0:
        try:
            os.close(setup_read)
            os.setsid()
            null = os.open("/dev/null", os.O_RDONLY)
            os.dup2(null, 0)
            for index, (read_fd, write_fd) in enumerate(pipes, 1):
                os.dup2(write_fd, index)
                os.close(read_fd)
                os.close(write_fd)
            os.close(null)
            _restrict(spec["readable"], spec["writable"])
            os.chdir(spec["cwd"])
            os.execvpe(spec["argv"][0], spec["argv"], spec["env"])
        except BaseException as error:
            os.write(setup_write, str(error).encode()[:8192])
            os._exit(125)
    os.close(setup_write)
    selector = selectors.DefaultSelector()
    buffers = [bytearray(), bytearray()]
    totals = [0, 0]
    for index, (read_fd, write_fd) in enumerate(pipes):
        os.close(write_fd)
        os.set_blocking(read_fd, False)
        selector.register(read_fd, selectors.EVENT_READ, index)
    deadline = time.monotonic() + spec["timeout_sec"]
    status = None
    timed_out = False
    killed = 0
    try:
        while status is None:
            for key, _ in selector.select(timeout=0.02):
                data = os.read(key.fd, 65536)
                if not data:
                    selector.unregister(key.fd)
                    os.close(key.fd)
                    continue
                index = key.data
                totals[index] += len(data)
                buffers[index].extend(
                    data[: max(0, spec["log_bytes"] - len(buffers[index]))]
                )
            found, child_status = os.waitpid(pid, os.WNOHANG)
            if found:
                status = child_status
            elif time.monotonic() > deadline:
                timed_out = True
                os.kill(pid, signal.SIGKILL)
                _, status = os.waitpid(pid, 0)
        killed = _reap_all()
        for key in list(selector.get_map().values()):
            while data := os.read(key.fd, 65536):
                index = key.data
                totals[index] += len(data)
                buffers[index].extend(
                    data[: max(0, spec["log_bytes"] - len(buffers[index]))]
                )
            selector.unregister(key.fd)
            os.close(key.fd)
    finally:
        selector.close()
        _reap_all()
    setup_error = os.read(setup_read, 8192).decode(errors="replace")
    os.close(setup_read)
    result = {
        "version": ISOLATION_VERSION,
        "landlock_abi": abi,
        "returncode": os.waitstatus_to_exitcode(status),
        "timed_out": timed_out,
        "setup_error": setup_error or None,
        "descendants_killed": killed,
        "descendants_remaining": 0,
        "stdout_bytes": totals[0],
        "stderr_bytes": totals[1],
        "stdout_truncated": totals[0] > len(buffers[0]),
        "stderr_truncated": totals[1] > len(buffers[1]),
        "log_limit_bytes": spec["log_bytes"],
        "readable_paths": spec["readable"],
        "writable_paths": spec["writable"],
    }
    Path(spec["result_path"]).write_text(json.dumps(result))
    os.write(1, buffers[0])
    os.write(2, buffers[1])


def runtime_paths(package_root: Path) -> list[Path]:
    """Public runtime/toolchain roots, excluding repository configs/references."""
    paths = [
        Path(value)
        for value in (
            "/usr",
            "/bin",
            "/sbin",
            "/lib",
            "/lib64",
            "/etc/ld.so.cache",
            "/etc/ld.so.conf",
            "/etc/ld.so.conf.d",
            "/etc/os-release",
            "/etc/debian_version",
            "/etc/lsb-release",
            "/etc/mime.types",
            "/etc/localtime",
            "/etc/nsswitch.conf",
            "/etc/passwd",
            "/etc/group",
            "/etc/java-17-openjdk",
            "/etc/java-21-openjdk",
            "/etc/alternatives",
            "/proc/cpuinfo",
            "/proc/meminfo",
            "/proc/self",
            "/dev/random",
            "/dev/urandom",
            "/opt/pitbench-jvm",
            "/opt/pitbench-adapters",
            "/opt/pitbench-maven",
            "/opt/ortools-deps",
            "/opt/java",
            "/root/.m2/repository",
        )
    ]
    paths.extend([Path(sys.prefix), Path(sys.base_prefix)])
    paths.append(package_root / "adapters")
    accessible = []
    for path in paths:
        try:
            if path.exists():
                accessible.append(path)
        except PermissionError:
            # Optional image caches may be absent/inaccessible on the host.
            continue
    return accessible


def protect_output(root: Path, group: int) -> None:
    """Allow the trusted controller/host group, excluding capability-free uid 0."""
    if group == CANDIDATE_GID:
        raise IsolationError(
            "host output group conflicts with reserved candidate group"
        )
    root.mkdir(parents=True, exist_ok=True)
    os.chown(root, ARTIFACT_OWNER_UID, group)
    os.chmod(root, 0o770)


def execute(
    command,
    workspace: Path,
    *,
    readable=(),
    writable=(),
    package_root: Path | None = None,
    log_bytes: int = DEFAULT_LOG_BYTES,
):
    """Run one command in a fresh domain, retaining bounded logs and cleanup."""
    unknown_env = set(command.env) - COMMAND_ENV_KEYS
    if unknown_env:
        raise IsolationError(
            f"unsupported child environment keys: {sorted(unknown_env)}"
        )
    package_root = package_root or Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="pitbench-isolation-") as temporary:
        control = Path(temporary)
        scratch = control / "scratch"
        scratch.mkdir()
        # Python lists PYTHONPATH entries. Stage only public drivers so its
        # importer needs no directory access to repository references/evidence.
        tooling = control / "tooling"
        package = tooling / "pitbench"
        package.mkdir(parents=True)
        shutil.copyfile(package_root / "pitbench/__init__.py", package / "__init__.py")
        shutil.copytree(
            package_root / "pitbench/solver_drivers",
            package / "solver_drivers",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        environment = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "HOME": str(scratch),
            "TMPDIR": str(scratch),
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(tooling),
            "JAVA_TOOL_OPTIONS": f"-Djava.io.tmpdir={scratch}",
        }
        # Existing immutable Java adapters resolve jars via Path.home(). HOME
        # grants no access: only the public Maven repository is allowlisted.
        try:
            if Path("/root/.m2/repository").is_dir():
                environment["HOME"] = "/root"
        except PermissionError:
            pass
        # These image settings name public drivers/toolchains, never secrets.
        for key in IMAGE_ENV_KEYS:
            if key in os.environ:
                environment[key] = os.environ[key]
        environment.update(command.env)
        result_path = control / "result.json"
        spec = {
            "argv": command.argv,
            "cwd": str(workspace / command.cwd),
            "env": environment,
            "timeout_sec": command.timeout_sec,
            "readable": [
                str(p)
                for p in [*runtime_paths(package_root), tooling, workspace, *readable]
            ],
            "writable": [
                str(p)
                for p in [scratch, Path("/dev/null"), Path("/dev/zero"), *writable]
            ],
            "result_path": str(result_path),
            "log_bytes": log_bytes,
        }
        spec_path = control / "spec.json"
        spec_path.write_text(json.dumps(spec))
        completed = subprocess.run(
            [sys.executable, "-I", "-S", str(Path(__file__).resolve()), str(spec_path)],
            env={"PATH": environment["PATH"], "LANG": "C.UTF-8"},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=command.timeout_sec + 15,
        )
        if completed.returncode or not result_path.is_file():
            raise IsolationError(
                f"isolation supervisor failed: {completed.stderr.strip()}"
            )
        metadata = json.loads(result_path.read_text())
        if metadata["setup_error"]:
            raise IsolationError(metadata["setup_error"])
        if metadata["timed_out"]:
            error = subprocess.TimeoutExpired(
                command.argv, command.timeout_sec, completed.stdout, completed.stderr
            )
            error.isolation = metadata
            raise error
        completed.args = command.argv
        completed.returncode = metadata["returncode"]
        completed.isolation = metadata
        return completed


if __name__ == "__main__":
    _launch(json.loads(Path(sys.argv[1]).read_text()))
