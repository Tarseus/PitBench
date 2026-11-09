from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import deque
from pathlib import Path
from typing import Callable

from pitbench.evaluator.private_assets import PrivateAssetResolver
from pitbench.evaluator.storage import ObservationStore
from pitbench.schema.observation import CodeState, RunObservation


def judge_harness_sha256(package_root: Path) -> str:
    """Fingerprint the Python code mounted into the isolated judge."""
    digest = hashlib.sha256()
    for source_root in (package_root / "pitbench", package_root / "adapters"):
        for path in sorted(
            p for p in source_root.rglob("*") if p.suffix in {".py", ".cpp", ".java"}
        ):
            relative = path.relative_to(package_root).as_posix().encode()
            content = path.read_bytes()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
    return digest.hexdigest()


def docker_cli_image(image: str) -> str:
    """Return a Docker CLI image argument while preserving the pinned identity."""
    if re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        return image.removeprefix("sha256:")
    return image


class DockerJudge:
    """Launch a fresh, network-disabled judge container for one candidate."""

    def __init__(
        self,
        *,
        image: str,
        task_config_path: Path,
        base_repository: Path,
        private_root: Path,
        candidate_patch: Path,
        output_dir: Path,
        cpus: float = 8.0,
        memory: str = "8g",
        tmpfs_size: str = "4g",
        cpuset_cpus: str | None = None,
        code_states: tuple[CodeState, ...] = tuple(CodeState),
        parallel_runs: int = 1,
        progress_callback: Callable[[str], None] | None = None,
        reliability_only: bool = False,
        instances_per_instance_set: int | None = None,
        solver_seed_limit: int | None = None,
        base_observations_path: Path | None = None,
        build_cache_path: Path | None = None,
        configuration_panel: Path | None = None,
    ) -> None:
        digest_pinned = re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", image)
        local_image_id = re.fullmatch(r"sha256:[0-9a-f]{64}", image)
        if digest_pinned is None and local_image_id is None:
            raise ValueError("judge image must be pinned by sha256 digest")
        self.image = image
        # Docker accepts a local image ID as the bare hexadecimal ID in `run`;
        # the `sha256:` prefix is valid for identity checks but is parsed as an
        # invalid repository reference when passed to the CLI.
        self.docker_image = docker_cli_image(image)
        self.task_config_path = task_config_path.resolve()
        self.base_repository = base_repository.resolve()
        self.private_root = PrivateAssetResolver(private_root).root
        self.candidate_patch = candidate_patch.resolve()
        self.output_dir = output_dir.resolve()
        self.cpus = float(cpus)
        self.memory = memory
        self.tmpfs_size = tmpfs_size
        self.cpuset_cpus = cpuset_cpus
        self.code_states = code_states
        self.parallel_runs = parallel_runs
        self.progress_callback = progress_callback
        self.reliability_only = reliability_only
        if instances_per_instance_set is not None and instances_per_instance_set < 1:
            raise ValueError("instances_per_instance_set must be positive")
        self.instances_per_instance_set = instances_per_instance_set
        if solver_seed_limit is not None and solver_seed_limit < 1:
            raise ValueError("solver_seed_limit must be positive")
        self.solver_seed_limit = solver_seed_limit
        self.base_observations_path = base_observations_path
        self.build_cache_path = build_cache_path.resolve() if build_cache_path else None
        self.configuration_panel = configuration_panel
        self.configuration_panel_sha256 = (
            hashlib.sha256(configuration_panel.read_bytes()).hexdigest()
            if configuration_panel
            else None
        )
        self.package_root = Path(__file__).resolve().parents[2]
        self.harness_sha256 = judge_harness_sha256(self.package_root)
        self.candidate_patch_sha256 = hashlib.sha256(
            self.candidate_patch.read_bytes()
        ).hexdigest()
        self.task_config_sha256 = hashlib.sha256(
            self.task_config_path.read_bytes()
        ).hexdigest()

    def checkpoint_identity(self) -> dict[str, object]:
        identity = {
            "candidate_patch_sha256": self.candidate_patch_sha256,
            "task_config_sha256": self.task_config_sha256,
            "harness_sha256": self.harness_sha256,
            "judge_image": self.image,
            "cpus": self.cpus,
            "memory": self.memory,
            "tmpfs_size": self.tmpfs_size,
            "cpuset_cpus": self.cpuset_cpus,
            "code_states": [state.value for state in self.code_states],
            "parallel_runs": self.parallel_runs,
            "reliability_only": self.reliability_only,
            "instances_per_instance_set": self.instances_per_instance_set,
            "solver_seed_limit": self.solver_seed_limit,
        }
        if self.configuration_panel is not None:
            identity["configuration_panel_sha256"] = self.configuration_panel_sha256
        if self.base_observations_path is not None:
            identity["shared_base_sha256"] = hashlib.sha256(
                self.base_observations_path.read_bytes()
            ).hexdigest()
        return identity

    def run(self) -> list[RunObservation]:
        package_root = self.package_root
        self._check_inputs_unchanged("before launch")
        judge_tmp = self.output_dir / "judge-tmp"
        judge_tmp.mkdir(parents=True, exist_ok=True)
        observations = self.output_dir / "judge-observations.jsonl"
        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--label",
            f"pitbench.judge_output={self.output_dir}",
            "--read-only",
            "--cpus",
            str(self.cpus),
            "--memory",
            self.memory,
            "--pids-limit",
            "2048",
            "--tmpfs",
            f"/tmp:rw,exec,nosuid,size={self.tmpfs_size}",
            "--env",
            "PYTHONPATH=/opt/pitbench",
            "--volume",
            f"{package_root}:/opt/pitbench:ro",
            "--volume",
            f"{self.task_config_path}:/input/task.yaml:ro",
            "--volume",
            f"{self.base_repository}:/input/base:ro",
            "--volume",
            f"{self.private_root}:/private:ro",
            "--volume",
            f"{self.candidate_patch}:/input/candidate.patch:ro",
            "--volume",
            f"{self.output_dir}:/output:rw",
            "--volume",
            f"{judge_tmp}:/work-tmp:rw",
            "--env",
            "TMPDIR=/work-tmp",
            self.docker_image,
            "python3",
            "-m",
            "pitbench.evaluator.runner",
            "--task-config",
            "/input/task.yaml",
            "--base-repository",
            "/input/base",
            "--public-root",
            "/opt/pitbench",
            "--private-root",
            "/private",
            "--candidate-patch",
            "/input/candidate.patch",
            "--output-dir",
            "/output",
            "--observations",
            "/output/judge-observations.jsonl",
        ]
        if self.configuration_panel is not None:
            position = command.index(self.docker_image)
            command[position:position] = [
                "--volume",
                f"{self.configuration_panel.resolve()}:/input/configuration-panel.json:ro",
            ]
            command.extend(["--configuration-panel", "/input/configuration-panel.json"])
        if self.cpuset_cpus is not None:
            command[command.index("--memory") : command.index("--memory")] = [
                "--cpuset-cpus",
                self.cpuset_cpus,
            ]
        command.extend(["--parallel-runs", str(self.parallel_runs)])
        command.extend(
            ["--checkpoint-identity", json.dumps(self.checkpoint_identity())]
        )
        if self.base_observations_path is not None:
            command[
                command.index(self.docker_image) : command.index(self.docker_image)
            ] = [
                "--volume",
                f"{self.base_observations_path.resolve()}:/input/shared-base.jsonl:ro",
            ]
            command.extend(["--base-observations", "/input/shared-base.jsonl"])
        if self.build_cache_path is not None:
            self.build_cache_path.mkdir(parents=True, exist_ok=True)
            position = command.index(self.docker_image)
            command[position:position] = [
                "--volume",
                f"{self.build_cache_path}:{self.build_cache_path}:rw",
                "--env",
                f"PITBENCH_BUILD_CACHE={self.build_cache_path}",
                "--env",
                "PITBENCH_BUILD_CONTEXT="
                + json.dumps(
                    {
                        "judge_image": self.image,
                        "harness_sha256": self.harness_sha256,
                        "cpus": self.cpus,
                        "memory": self.memory,
                    }
                ),
            ]
        launch_path = self.output_dir / "judge-launch.json"
        identity = self.checkpoint_identity()
        if observations.is_file() and observations.stat().st_size:
            receipt = (
                launch_path
                if launch_path.is_file()
                else self.output_dir / "judge-checkpoint.json"
            )
            if (
                not receipt.is_file()
                or json.loads(receipt.read_text()).get("identity") != identity
            ):
                raise RuntimeError(
                    "retained observations have a different or missing Judge identity; preserve this output and use a new directory"
                )
            command.append("--resume-existing-observations")
        launch_tmp = launch_path.with_suffix(".json.tmp")
        launch_tmp.write_text(json.dumps({"identity": identity}, sort_keys=True) + "\n")
        launch_tmp.replace(launch_path)
        if self.reliability_only:
            command.append("--reliability-only")
        if self.instances_per_instance_set is not None:
            command.extend(
                [
                    "--instances-per-instance-set",
                    str(self.instances_per_instance_set),
                ]
            )
        if self.solver_seed_limit is not None:
            command.extend(["--solver-seed-limit", str(self.solver_seed_limit)])
        for state in self.code_states:
            command.extend(["--code-state", state.value])
        (self.output_dir / "docker-command.json").write_text(
            json.dumps({"argv": command}, indent=2) + "\n"
        )
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        recent_output: deque[str] = deque(maxlen=20)
        last_progress: str | None = None
        if process.stdout is not None:
            for raw_line in process.stdout:
                line = raw_line.rstrip()
                if line:
                    prefix = "PITBENCH_PROGRESS "
                    if line.startswith(prefix):
                        last_progress = line.removeprefix(prefix)
                        if self.progress_callback is not None:
                            self.progress_callback(last_progress)
                    else:
                        recent_output.append(line)
        returncode = process.wait()
        self._check_inputs_unchanged("during the run", observations)
        if returncode:
            details = ([last_progress] if last_progress is not None else []) + list(
                recent_output
            )
            if observations.is_file() and observations.stat().st_size:
                details.append(f"completed observations retained at {observations}")
            detail = "\n".join(details) or "no diagnostic was emitted"
            raise RuntimeError(f"isolated judge failed: {detail}")
        if not observations.is_file():
            raise RuntimeError("isolated judge produced no observations")
        return ObservationStore.read_jsonl(observations)

    def _check_inputs_unchanged(
        self, phase: str, observations: Path | None = None
    ) -> None:
        checks = {
            "judge harness code": (
                judge_harness_sha256(self.package_root),
                self.harness_sha256,
            ),
            "task configuration": (
                hashlib.sha256(self.task_config_path.read_bytes()).hexdigest(),
                self.task_config_sha256,
            ),
            "candidate patch": (
                hashlib.sha256(self.candidate_patch.read_bytes()).hexdigest(),
                self.candidate_patch_sha256,
            ),
        }
        if self.configuration_panel is not None:
            checks["configuration panel"] = (
                hashlib.sha256(self.configuration_panel.read_bytes()).hexdigest(),
                self.configuration_panel_sha256,
            )
        for name, (actual, expected) in checks.items():
            if actual != expected:
                retained = (
                    f"; observations remain at {observations}"
                    if observations is not None
                    else ""
                )
                raise RuntimeError(f"{name} changed {phase}{retained}")
