from __future__ import annotations

import subprocess
from pathlib import Path

from pitbench.evaluator.storage import ObservationStore
from pitbench.schema.observation import RunObservation


class DockerJudge:
    """Launch a fresh, network-disabled judge container for one candidate."""

    def __init__(
        self,
        *,
        image: str,
        manifest_path: Path,
        base_repository: Path,
        private_root: Path,
        candidate_patch: Path,
        output_dir: Path,
        cpus: float = 1.0,
        memory: str = "8g",
    ) -> None:
        if "@sha256:" not in image:
            raise ValueError("judge image must be pinned by sha256 digest")
        self.image = image
        self.manifest_path = manifest_path.resolve()
        self.base_repository = base_repository.resolve()
        self.private_root = private_root.resolve()
        self.candidate_patch = candidate_patch.resolve()
        self.output_dir = output_dir.resolve()
        self.cpus = cpus
        self.memory = memory

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
            f"{self.manifest_path}:/input/task.yaml:ro",
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
            "--manifest",
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
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(
                "isolated judge failed: "
                + (completed.stderr.strip() or completed.stdout.strip())
            )
        if not observations.is_file():
            raise RuntimeError("isolated judge produced no observations")
        return ObservationStore.read_jsonl(observations)
