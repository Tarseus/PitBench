from __future__ import annotations

import json
import os
import platform
import pwd
import re
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

import docker.errors

import docker
from pitbench.cli.evaluate_config import (
    EvaluationConfig,
    resolve_config_path,
    resolve_repository_path,
)
from pitbench.evaluator.judge import JudgePlan
from pitbench.evaluator.private_assets import PrivateAssetResolver
from pitbench.harness.agents.antigravity_container import (
    AntigravityContainerRunner,
)
from pitbench.harness.agents.antigravity_profile import AntigravityProfile
from pitbench.harness.agents.codex_container import CodexContainerRunner
from pitbench.harness.agents.codex_profile import CodexProfile
from pitbench.tasks import TaskCatalog, TaskRecord

_PYVRP_TASK_ID = "pyvrp_v0_14_0"
_GIB = 1024**3
_PROXY_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


class CheckStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class DoctorCheck:
    status: CheckStatus
    name: str
    detail: str
    recovery: str | None = None


@dataclass(frozen=True)
class _DoctorContext:
    repository_root: Path
    config_path: Path | None
    config: EvaluationConfig
    task: TaskRecord | None


def _check(
    status: CheckStatus,
    name: str,
    detail: str,
    recovery: str | None = None,
) -> DoctorCheck:
    return DoctorCheck(status, name, detail, recovery)


def _format_gib(value: int) -> str:
    return f"{value / _GIB:.1f} GiB"


def _available_memory() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _platform_check() -> DoctorCheck:
    system = platform.system()
    machine = platform.machine().lower()
    supported_arch = machine in {"x86_64", "amd64"}
    if system == "Linux" and supported_arch:
        return _check(CheckStatus.PASS, "platform", f"Linux {machine}")
    return _check(
        CheckStatus.FAIL,
        "platform",
        f"published PyVRP images require Linux x86_64; found {system} {machine}",
        "Run PitBench on a Linux x86_64 host.",
    )


def _resource_checks(repository_root: Path) -> list[DoctorCheck]:
    disk = shutil.disk_usage(repository_root).free
    if disk < 10 * _GIB:
        disk_check = _check(
            CheckStatus.FAIL,
            "disk",
            f"only {_format_gib(disk)} free (10 GiB minimum)",
            "Free at least 10 GiB on the filesystem containing the repository.",
        )
    elif disk < 20 * _GIB:
        disk_check = _check(
            CheckStatus.WARN,
            "disk",
            f"{_format_gib(disk)} free; 20 GiB is recommended for images and builds",
            "Free additional space before running several evaluations.",
        )
    else:
        disk_check = _check(CheckStatus.PASS, "disk", f"{_format_gib(disk)} free")

    memory = _available_memory()
    if memory is None:
        memory_check = _check(
            CheckStatus.WARN,
            "memory",
            "could not read MemAvailable from /proc/meminfo",
            "Verify that at least 8 GiB of memory is available.",
        )
    elif memory < 8 * _GIB:
        memory_check = _check(
            CheckStatus.FAIL,
            "memory",
            f"only {_format_gib(memory)} available (8 GiB minimum)",
            "Use a host with at least 8 GiB available memory.",
        )
    elif memory < 16 * _GIB:
        memory_check = _check(
            CheckStatus.WARN,
            "memory",
            f"{_format_gib(memory)} available; 16 GiB is recommended",
        )
    else:
        memory_check = _check(
            CheckStatus.PASS, "memory", f"{_format_gib(memory)} available"
        )
    return [disk_check, memory_check]


def _load_context(
    repository_root: Path, config_path: Path | None
) -> tuple[_DoctorContext, list[DoctorCheck]]:
    checks: list[DoctorCheck] = []
    resolved_config = resolve_config_path(repository_root, config_path)
    config = EvaluationConfig()
    if resolved_config is None:
        checks.append(
            _check(
                CheckStatus.FAIL,
                "configuration",
                "config/evaluate.local.yaml does not exist",
                "Run `cp config/evaluate.example.yaml config/evaluate.local.yaml` "
                "and edit it for this machine.",
            )
        )
    elif not resolved_config.is_file():
        checks.append(
            _check(
                CheckStatus.FAIL,
                "configuration",
                f"configuration file does not exist: {resolved_config}",
                "Create the file or pass the correct path with `--config`.",
            )
        )
    else:
        try:
            config = EvaluationConfig.from_yaml(resolved_config)
        except Exception as error:
            checks.append(
                _check(
                    CheckStatus.FAIL,
                    "configuration",
                    f"invalid {resolved_config}: {error}",
                    "Fix the YAML using config/evaluate.example.yaml as the template.",
                )
            )
        else:
            checks.append(
                _check(
                    CheckStatus.PASS,
                    "configuration",
                    f"loaded {resolved_config}",
                )
            )

    task: TaskRecord | None = None
    try:
        task = TaskCatalog(repository_root).validate_one(_PYVRP_TASK_ID)
    except Exception as error:
        checks.append(
            _check(
                CheckStatus.FAIL,
                "task config",
                f"cannot validate {_PYVRP_TASK_ID}: {error}",
                "Restore the tracked task and instance-set configs, then run "
                "`uv run pitbench tasks validate`.",
            )
        )
    else:
        checks.append(
            _check(
                CheckStatus.PASS,
                "task config",
                f"{_PYVRP_TASK_ID} {task.task_config_sha256[:12]}",
            )
        )
    return _DoctorContext(repository_root, resolved_config, config, task), checks


def _private_assets_check(context: _DoctorContext) -> DoctorCheck:
    if context.task is None:
        return _check(
            CheckStatus.WARN,
            "private assets",
            "not checked because the task config is invalid",
        )
    private_root = resolve_repository_path(
        context.repository_root, context.config.paths.private_root
    )
    try:
        with tempfile.TemporaryDirectory(prefix="pitbench-doctor-") as generated:
            plan = JudgePlan.from_instance_set_configs(
                context.task.task,
                PrivateAssetResolver(private_root),
                public_root=context.repository_root,
                generated_root=Path(generated),
            )
    except Exception as error:
        return _check(
            CheckStatus.FAIL,
            "private assets",
            f"validation failed under {private_root}: {error}",
            "Provision the PyVRP private bundle at the configured private_root "
            "and verify that it was copied without modification.",
        )
    return _check(
        CheckStatus.PASS,
        "private assets",
        f"validated {len(plan.cases)} hidden cases and their checksums",
    )


def _docker_check(
    client_factory: Callable[[], docker.DockerClient] = docker.from_env,
) -> tuple[object | None, dict, DoctorCheck]:
    try:
        client = client_factory()
        client.ping()
        info = client.info()
    except Exception as error:
        detail = str(error)
        permission = "permission" in detail.lower() or isinstance(
            getattr(error, "__cause__", None), PermissionError
        )
        if permission:
            recovery = (
                "The current process has stale supplementary groups. Leave old tmux "
                "sessions, start a fresh SSH login (or run `newgrp docker`), verify "
                "`docker version`, and only then start tmux again."
            )
        else:
            recovery = "Start Docker and verify `docker version` in this exact shell."
        return (
            None,
            {},
            _check(
                CheckStatus.FAIL,
                "Docker API",
                f"the current process cannot call the Docker server: {detail}",
                recovery,
            ),
        )
    version = info.get("ServerVersion") or "unknown version"
    return (
        client,
        info,
        _check(CheckStatus.PASS, "Docker API", f"server {version} answered ping"),
    )


def _image_reference_check(kind: str, reference: str | None) -> DoctorCheck | None:
    if reference is None:
        return _check(
            CheckStatus.FAIL,
            f"{kind} image reference",
            "no image is configured",
            f"Set tasks.{_PYVRP_TASK_ID}.{kind}_image in config/evaluate.local.yaml.",
        )
    local_id = re.fullmatch(r"sha256:[0-9a-f]{64}", reference)
    digest = re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", reference)
    if kind == "agent" and local_id:
        return _check(
            CheckStatus.FAIL,
            "agent image reference",
            "a digest-only image ID cannot be used in the generated Dockerfile FROM",
            "Tag the image (for example `docker tag IMAGE_ID pitbench/pyvrp-agent:local`) "
            "and configure that tag.",
        )
    if kind == "judge" and not (local_id or digest):
        return _check(
            CheckStatus.FAIL,
            "judge image reference",
            "the judge must use a local sha256 image ID or a digest-pinned reference",
            f"Set tasks.{_PYVRP_TASK_ID}.judge_image to `sha256:...` or "
            "`IMAGE@sha256:...`.",
        )
    return None


def _image_checks(
    context: _DoctorContext, client: object | None
) -> tuple[list[DoctorCheck], list[str], bool]:
    if context.task is None:
        return (
            [
                _check(
                    CheckStatus.WARN,
                    "images",
                    "not checked because the task config is invalid",
                )
            ],
            [],
            False,
        )
    resources = context.config.resources_for(_PYVRP_TASK_ID)
    references = {
        "agent": resources.agent_image or context.task.task.repository.agent_image,
        "judge": resources.judge_image or context.task.task.repository.judge_image,
    }
    checks: list[DoctorCheck] = []
    references_valid = True
    for kind, reference in references.items():
        invalid = _image_reference_check(kind, reference)
        if invalid is not None:
            checks.append(invalid)
            references_valid = False
            continue
        assert reference is not None
        checks.append(_check(CheckStatus.PASS, f"{kind} image reference", reference))
        if client is None:
            checks.append(
                _check(
                    CheckStatus.WARN,
                    f"{kind} image",
                    "not inspected because Docker is unavailable",
                )
            )
            continue
        try:
            client.images.get(reference)  # type: ignore[attr-defined]
        except docker.errors.ImageNotFound:
            checks.append(
                _check(
                    CheckStatus.FAIL,
                    f"{kind} image",
                    f"not present locally: {reference}",
                    f"Run `docker pull {reference}` or configure a valid local "
                    f"{kind} image reference.",
                )
            )
        except Exception as error:
            checks.append(
                _check(
                    CheckStatus.FAIL,
                    f"{kind} image",
                    f"could not inspect {reference}: {error}",
                    "Fix Docker access, then rerun this command.",
                )
            )
        else:
            checks.append(_check(CheckStatus.PASS, f"{kind} image", "present locally"))
    ghcr = [ref for ref in references.values() if ref and ref.startswith("ghcr.io/")]
    return checks, ghcr, references_valid


def _registry_and_proxy_checks(
    client: object | None, docker_info: dict, ghcr_references: list[str]
) -> list[DoctorCheck]:
    if client is None:
        return [
            _check(
                CheckStatus.WARN,
                "GHCR access",
                "not checked because Docker is unavailable",
            )
        ]
    checks: list[DoctorCheck] = []
    if ghcr_references:
        reference = ghcr_references[0]
        try:
            client.images.get_registry_data(reference)  # type: ignore[attr-defined]
        except Exception as error:
            checks.append(
                _check(
                    CheckStatus.FAIL,
                    "GHCR access",
                    f"Docker cannot query the configured public image: {error}",
                    "Configure the Docker daemon's HTTPS proxy or load the images "
                    "offline and replace GHCR references with accepted local references.",
                )
            )
        else:
            checks.append(
                _check(
                    CheckStatus.PASS,
                    "GHCR access",
                    "Docker can query the configured public image anonymously",
                )
            )
    else:
        checks.append(
            _check(
                CheckStatus.PASS,
                "GHCR access",
                "not required by the configured local image references",
            )
        )

    daemon_proxy = (
        docker_info.get("HTTPSProxy")
        or docker_info.get("HTTPProxy")
        or docker_info.get("HttpsProxy")
        or docker_info.get("HttpProxy")
    )
    shell_proxy = next(
        (os.environ[key] for key in _PROXY_KEYS if os.environ.get(key)), None
    )
    if daemon_proxy:
        checks.append(
            _check(CheckStatus.PASS, "Docker proxy", f"daemon proxy: {daemon_proxy}")
        )
    elif shell_proxy:
        checks.append(
            _check(
                CheckStatus.WARN,
                "Docker proxy",
                "the shell has a proxy but the Docker daemon reports none",
                "Shell proxy variables do not configure Docker pulls. Configure the "
                "daemon proxy if direct GHCR access fails.",
            )
        )
    else:
        checks.append(
            _check(CheckStatus.PASS, "Docker proxy", "direct daemon access configured")
        )
    return checks


def _json_file(path: Path, label: str) -> tuple[object | None, DoctorCheck | None]:
    try:
        value = json.loads(path.read_text())
    except Exception as error:
        return None, _check(
            CheckStatus.WARN,
            label,
            f"cannot read valid JSON from {path}: {error}",
        )
    return value, None


def _credential_check(
    agent: str, values: dict[str, object]
) -> tuple[DoctorCheck, bool]:
    if agent == "codex":
        binary = str(values.get("codex_binary") or "codex")
        auth_path = Path(
            str(values.get("codex_auth_path") or "~/.codex/auth.json")
        ).expanduser()
        _, invalid = _json_file(auth_path, "Codex credentials")
        command = [binary, "login", "status"]
        success_text = "Logged in using ChatGPT"
        recovery = "Run `codex login` and complete ChatGPT sign-in."
    else:
        binary = str(values.get("agy_binary") or "agy")
        auth_path = Path(
            str(
                values.get("auth_token_path")
                or "~/.gemini/antigravity-cli/antigravity-oauth-token"
            )
        ).expanduser()
        token, invalid = _json_file(auth_path, "Antigravity credentials")
        if invalid is None and (
            not isinstance(token, dict)
            or not isinstance(token.get("auth_method"), str)
            or not isinstance(token.get("token"), dict)
        ):
            invalid = _check(
                CheckStatus.WARN,
                "Antigravity credentials",
                f"invalid OAuth token schema in {auth_path}",
            )
        settings_path = Path(
            str(
                values.get("settings_path") or "~/.gemini/antigravity-cli/settings.json"
            )
        ).expanduser()
        settings, settings_invalid = _json_file(
            settings_path, "Antigravity credentials"
        )
        if invalid is None and settings_invalid is not None:
            invalid = settings_invalid
        if invalid is None and not isinstance(settings, dict):
            invalid = _check(
                CheckStatus.WARN,
                "Antigravity credentials",
                f"settings must contain a JSON object: {settings_path}",
            )
        command = [binary, "models"]
        success_text = ""
        recovery = "Run `agy` and complete Google sign-in."
    if invalid is not None:
        return DoctorCheck(
            invalid.status, invalid.name, invalid.detail, recovery
        ), False
    if shutil.which(binary) is None and not Path(binary).is_file():
        return _check(
            CheckStatus.WARN,
            f"{agent} credentials",
            f"CLI is not executable: {binary}",
            recovery,
        ), False
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=15, check=False
        )
    except Exception as error:
        return _check(
            CheckStatus.WARN,
            f"{agent} credentials",
            f"could not verify login: {error}",
            recovery,
        ), False
    output = f"{result.stdout}\n{result.stderr}"
    if result.returncode != 0 or (success_text and success_text not in output):
        return _check(
            CheckStatus.WARN,
            f"{agent} credentials",
            "host CLI is not logged in",
            recovery,
        ), False
    return _check(
        CheckStatus.PASS,
        f"{agent} credentials",
        f"host login is ready ({auth_path})",
    ), True


def _runner_check(agent: str, values: dict[str, object]) -> tuple[DoctorCheck, bool]:
    if values.get("runner_backend") == "workspace":
        if agent != "codex":
            return _check(
                CheckStatus.WARN,
                f"{agent} runner",
                "workspace backend is currently supported only for Codex",
            ), False
        return _codex_workspace_runner_check(values)
    if values.get("runner_backend") == "container":
        if agent == "codex":
            return _codex_container_runner_check(values)
        return _antigravity_container_runner_check(values)
    defaults = {
        "codex": ("/usr/local/libexec/pitbench-codex-runner", "pitbench-codex"),
        "antigravity": (
            "/usr/local/libexec/pitbench-antigravity-runner",
            "pitbench-agy",
        ),
    }
    default_path, default_user = defaults[agent]
    runner_path = Path(str(values.get("runner_path") or default_path))
    runner_user = str(values.get("runner_user") or default_user)
    installer = f"scripts/install-{agent}-runner.sh"
    try:
        metadata = runner_path.stat()
    except OSError as error:
        return _check(
            CheckStatus.WARN,
            f"{agent} runner",
            f"runner is unavailable at {runner_path}: {error}",
            f"Ask an administrator to run `sudo {installer}` for your account.",
        ), False
    unsafe_mode = metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    if metadata.st_uid != 0 or unsafe_mode or not os.access(runner_path, os.X_OK):
        return _check(
            CheckStatus.WARN,
            f"{agent} runner",
            f"{runner_path} must be root-owned, executable, and not group/world-writable",
            f"Ask an administrator to reinstall it with `sudo {installer}`.",
        ), False
    try:
        account = pwd.getpwnam(runner_user)
    except KeyError:
        return _check(
            CheckStatus.WARN,
            f"{agent} runner",
            f"dedicated account does not exist: {runner_user}",
            f"Ask an administrator to run `sudo {installer}`.",
        ), False
    try:
        result = subprocess.run(
            [
                "sudo",
                "-n",
                "-u",
                runner_user,
                "--",
                str(runner_path),
                "--self-test",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        payload = json.loads(result.stdout)
    except Exception as error:
        return _check(
            CheckStatus.WARN,
            f"{agent} runner",
            f"self-test could not run: {error}",
            f"Ask an administrator to run `sudo {installer}`.",
        ), False
    groups = payload.get("groups", [])
    has_docker = payload.get("docker_socket_access") or "docker" in groups
    wrong_user = payload.get("uid") != account.pw_uid or account.pw_uid == 0
    if result.returncode != 0 or has_docker or wrong_user:
        detail = result.stderr.strip() or result.stdout.strip()
        if has_docker:
            detail = f"dedicated user {runner_user} still has Docker access"
        elif wrong_user:
            detail = f"self-test did not run as dedicated user {runner_user}"
        return _check(
            CheckStatus.WARN,
            f"{agent} runner",
            detail or "self-test failed",
            f"Remove Docker access from {runner_user}, then reinstall with "
            f"`sudo {installer}`.",
        ), False
    return _check(
        CheckStatus.PASS,
        f"{agent} runner",
        f"root-owned launcher runs as {runner_user} without Docker access",
    ), True


def _codex_workspace_runner_check(
    values: dict[str, object],
) -> tuple[DoctorCheck, bool]:
    binary_value = str(values.get("codex_binary") or "codex")
    resolved = shutil.which(binary_value)
    if resolved is None:
        return _check(
            CheckStatus.WARN,
            "codex workspace runner",
            f"Codex CLI is not executable: {binary_value}",
            "Install Codex and run `codex login`.",
        ), False
    codex_binary = Path(resolved).resolve()
    code_mode_host = codex_binary.parent / "codex-code-mode-host"
    bubblewrap = codex_binary.parent.parent / "codex-resources/bwrap"
    missing_runtime = next(
        (
            path
            for path in (code_mode_host, bubblewrap)
            if not path.is_file() or not os.access(path, os.X_OK)
        ),
        None,
    )
    if missing_runtime is not None:
        return _check(
            CheckStatus.WARN,
            "codex workspace runner",
            f"Codex runtime helper is unavailable: {missing_runtime}",
            "Reinstall the standalone Codex CLI bundle.",
        ), False
    profile: CodexProfile | None = None
    profile_path = values.get("profile_path")
    if profile_path is not None:
        try:
            profile = CodexProfile.load(Path(str(profile_path)))
        except Exception as error:
            return _check(
                CheckStatus.WARN,
                "codex workspace runner",
                f"invalid Codex profile: {error}",
                "Run `uv run pitbench profiles validate PROFILE_PATH`.",
            ), False
    profile_detail = (
        f", profile {profile.name}@{profile.sha256[:12]}" if profile else ""
    )
    return _check(
        CheckStatus.PASS,
        "codex workspace runner",
        "Codex can be staged in the solver container with a relay-only network"
        f"{profile_detail}",
    ), True


def _codex_container_runner_check(
    values: dict[str, object],
) -> tuple[DoctorCheck, bool]:
    binary_value = str(values.get("codex_binary") or "codex")
    resolved_binary = shutil.which(binary_value)
    if resolved_binary is None:
        return _check(
            CheckStatus.WARN,
            "codex runner",
            f"Codex CLI is not executable: {binary_value}",
            "Install Codex and run `codex login`.",
        ), False
    profile: CodexProfile | None = None
    profile_path = values.get("profile_path")
    if profile_path is not None:
        try:
            profile = CodexProfile.load(Path(str(profile_path)))
        except Exception as error:
            return _check(
                CheckStatus.WARN,
                "codex runner",
                f"invalid Codex profile: {error}",
                "Run `uv run pitbench profiles validate PROFILE_PATH`.",
            ), False
    image = str(values.get("container_runner_image") or "python:3.13-slim-bookworm")
    runner = CodexContainerRunner(
        image=image,
        codex_binary=Path(resolved_binary),
        profile=profile,
    )
    try:
        metadata = runner.prepare(pull=False)
    except Exception as error:
        return _check(
            CheckStatus.WARN,
            "codex runner",
            f"container self-test failed: {error}",
            f"Run `docker pull {image}`, then rerun doctor.",
        ), False
    profile_detail = (
        f", profile {profile.name}@{profile.sha256[:12]}" if profile else ""
    )
    return _check(
        CheckStatus.PASS,
        "codex runner",
        "container has no Docker socket "
        f"({metadata['runner_image_id']}{profile_detail})",
    ), True


def _antigravity_container_runner_check(
    values: dict[str, object],
) -> tuple[DoctorCheck, bool]:
    binary_value = str(values.get("agy_binary") or "agy")
    resolved_binary = shutil.which(binary_value)
    if resolved_binary is None:
        return _check(
            CheckStatus.WARN,
            "antigravity runner",
            f"Antigravity CLI is not executable: {binary_value}",
            "Install agy and complete Google sign-in.",
        ), False
    profile: AntigravityProfile | None = None
    profile_path = values.get("profile_path")
    if profile_path is not None:
        try:
            profile = AntigravityProfile.load(Path(str(profile_path)))
        except Exception as error:
            return _check(
                CheckStatus.WARN,
                "antigravity runner",
                f"invalid Antigravity profile: {error}",
                "Run `uv run pitbench profiles validate PROFILE_PATH`.",
            ), False
    image = str(values.get("container_runner_image") or "python:3.13-slim-bookworm")
    runner = AntigravityContainerRunner(
        image=image,
        agy_binary=Path(resolved_binary),
        profile=profile,
    )
    try:
        metadata = runner.prepare(pull=False)
    except Exception as error:
        return _check(
            CheckStatus.WARN,
            "antigravity runner",
            f"container self-test failed: {error}",
            f"Run `docker pull {image}`, then rerun doctor.",
        ), False
    profile_detail = (
        f", profile {profile.name}@{profile.sha256[:12]}" if profile else ""
    )
    return _check(
        CheckStatus.PASS,
        "antigravity runner",
        "container has no Docker socket "
        f"({metadata['runner_image_id']}{profile_detail})",
    ), True


def _agent_checks(config: EvaluationConfig, repository_root: Path) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    ready_agents: list[str] = []
    for agent in ("codex", "antigravity"):
        values: dict[str, object] = dict(config.kwargs_for(agent))
        if values.get("profile_path"):
            values["profile_path"] = str(
                resolve_repository_path(
                    repository_root,
                    Path(str(values["profile_path"])),
                )
            )
        credential, credentials_ready = _credential_check(agent, values)
        runner, runner_ready = _runner_check(agent, values)
        checks.extend([credential, runner])
        if credentials_ready and runner_ready:
            ready_agents.append(agent)
    if ready_agents:
        checks.append(
            _check(
                CheckStatus.PASS,
                "agent availability",
                f"ready: {', '.join(ready_agents)}",
            )
        )
    else:
        checks.append(
            _check(
                CheckStatus.FAIL,
                "agent availability",
                "neither Codex nor Antigravity has both credentials and an isolated runner",
                "Complete login and use a container runner, or ask an administrator "
                "to install a host runner.",
            )
        )
    return checks


def run_doctor(
    profile: str,
    repository_root: Path,
    config_path: Path | None = None,
    *,
    client_factory: Callable[[], docker.DockerClient] = docker.from_env,
) -> list[DoctorCheck]:
    if profile != "pyvrp":
        raise ValueError(f"unsupported doctor profile: {profile}")
    root = repository_root.resolve()
    checks = [_platform_check(), *_resource_checks(root)]
    context, context_checks = _load_context(root, config_path)
    checks.extend(context_checks)
    checks.append(_private_assets_check(context))
    client, docker_info, docker_result = _docker_check(client_factory)
    checks.append(docker_result)
    image_checks, ghcr_references, _ = _image_checks(context, client)
    checks.extend(image_checks)
    checks.extend(_registry_and_proxy_checks(client, docker_info, ghcr_references))
    checks.extend(_agent_checks(context.config, root))
    close = getattr(client, "close", None)
    if callable(close):
        close()
    return checks
