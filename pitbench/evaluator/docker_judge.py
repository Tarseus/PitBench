from __future__ import annotations

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
            "python",
            "-m",
            "pitbench.evaluator.runner",
            "--task-config",
            "/input/task.yaml",
            "--base-repository",
            "/input/base",
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
            command[command.index("--memory"):command.index("--memory")] = [
                "--cpuset-cpus",
                self.cpuset_cpus,
            ]
        command.extend(["--parallel-runs", str(self.parallel_runs)])
        for state in self.code_states:
            command.extend(["--code-state", state.value])
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        recent_output: deque[str] = deque(maxlen=100)
        if process.stdout is not None:
            for raw_line in process.stdout:
                line = raw_line.rstrip()
                if line:
                    recent_output.append(line)
                    prefix = "PITBENCH_PROGRESS "
                    if line.startswith(prefix) and self.progress_callback is not None:
                        self.progress_callback(line.removeprefix(prefix))
        returncode = process.wait()
        if returncode:
            detail = "\n".join(recent_output) or "no diagnostic was emitted"
            raise RuntimeError(f"isolated judge failed: {detail}")
        if not observations.is_file():
            raise RuntimeError("isolated judge produced no observations")
        return ObservationStore.read_jsonl(observations)
