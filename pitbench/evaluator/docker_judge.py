from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import deque
from pathlib import Path
from typing import Callable

from pitbench.evaluator.storage import ObservationStore
from pitbench.schema.observation import CodeState, RunObservation


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
        cpuset_cpus: str | None = None,
        code_states: tuple[CodeState, ...] = tuple(CodeState),
        parallel_runs: int = 1,
        progress_callback: Callable[[str], None] | None = None,
        reliability_only: bool = False,
        instances_per_instance_set: int | None = None,
        solver_seed_limit: int | None = None,
    ) -> None:
        digest_pinned = re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", image)
        local_image_id = re.fullmatch(r"sha256:[0-9a-f]{64}", image)
        if digest_pinned is None and local_image_id is None:
            raise ValueError("judge image must be pinned by sha256 digest")
        self.image = image
        self.task_config_path = task_config_path.resolve()
        self.base_repository = base_repository.resolve()
        self.private_root = private_root.resolve()
        self.candidate_patch = candidate_patch.resolve()
        self.output_dir = output_dir.resolve()
        self.cpus = cpus
        self.memory = memory
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

    def checkpoint_identity(self) -> dict[str, object]:
        return {
            "candidate_patch_sha256": hashlib.sha256(
                self.candidate_patch.read_bytes()
            ).hexdigest(),
            "task_config_sha256": hashlib.sha256(
                self.task_config_path.read_bytes()
            ).hexdigest(),
            "judge_image": self.image,
            "cpus": self.cpus,
            "memory": self.memory,
            "cpuset_cpus": self.cpuset_cpus,
            "code_states": [state.value for state in self.code_states],
            "parallel_runs": self.parallel_runs,
            "reliability_only": self.reliability_only,
            "instances_per_instance_set": self.instances_per_instance_set,
            "solver_seed_limit": self.solver_seed_limit,
        }

    def run(self) -> list[RunObservation]:
        package_root = Path(__file__).resolve().parents[2]
        observations = self.output_dir / "judge-observations.jsonl"
        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--cpus",
            str(self.cpus),
            "--memory",
            self.memory,
            "--pids-limit",
            "2048",
            "--tmpfs",
            "/tmp:rw,exec,nosuid,size=4g",
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
            self.image,
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
        if self.cpuset_cpus is not None:
            command[command.index("--memory") : command.index("--memory")] = [
                "--cpuset-cpus",
                self.cpuset_cpus,
            ]
        command.extend(["--parallel-runs", str(self.parallel_runs)])
        command.extend(
            ["--checkpoint-identity", json.dumps(self.checkpoint_identity())]
        )
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
