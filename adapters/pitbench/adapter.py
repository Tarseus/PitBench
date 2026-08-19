from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from adapters.pitbench.agent_tooling import write_agent_tooling
from adapters.pitbench.git_snapshot import GitSnapshot
from pitbench.instances import materialize_population
from pitbench.schema.task import PitBenchTask, PopulationKind
from pitbench.tasks import TaskCatalog

_AGENT_IMAGES = {
    "pitbench.repositories.pyvrp:PyVRPRepositoryPlugin": "python:3.13-bookworm",
    "pitbench.repositories.vroom:VroomRepositoryPlugin": "ubuntu:22.04",
    "pitbench.repositories.highs:HighsRepositoryPlugin": "ubuntu:24.04",
    "pitbench.repositories.choco:ChocoRepositoryPlugin": (
        "maven:3.9-eclipse-temurin-11"
    ),
    "pitbench.repositories.ortools:OrToolsRepositoryPlugin": (
        "maven:3.9-eclipse-temurin-11"
    ),
}

_BUILD_PACKAGES = {
    "pitbench.repositories.pyvrp:PyVRPRepositoryPlugin": (
        "build-essential cmake ninja-build python3-dev"
    ),
    "pitbench.repositories.vroom:VroomRepositoryPlugin": (
        "build-essential libssl-dev libasio-dev libglpk-dev pkg-config"
    ),
    "pitbench.repositories.highs:HighsRepositoryPlugin": "build-essential cmake",
    "pitbench.repositories.choco:ChocoRepositoryPlugin": "",
    "pitbench.repositories.ortools:OrToolsRepositoryPlugin": (
        "build-essential cmake openjdk-11-jdk maven swig"
    ),
}


class PitBenchAdapter:
    """Materialize an agent-safe fceval task from a PitBench manifest."""

    def __init__(self, repository_root: Path, private_root: Path) -> None:
        self.repository_root = repository_root.resolve()
        self.private_root = private_root.resolve()
        self.catalog = TaskCatalog(self.repository_root)

    def materialize(
        self,
        task_id: str,
        destination: Path,
        *,
        repository_source: Path | None = None,
    ) -> Path:
        record = next(
            record
            for record in self.catalog.validate_all()
            if record.task.task_id == task_id
        )
        task = record.task
        task_dir = destination.resolve()
        if task_dir.exists():
            raise FileExistsError(task_dir)
        task_dir.mkdir(parents=True)
        repository = task_dir / "repo"
        if repository_source is None:
            GitSnapshot(
                clone_url=task.repository.clone_url,
                base_commit=task.event.base_commit,
                forbidden_commits=(task.event.human_commit,),
            ).create(repository, task_dir / "agent_repo.bundle")
        else:
            shutil.copytree(repository_source, repository)
            self._validate_repository(task, repository)

        development = next(
            population
            for population in task.populations
            if population.kind == PopulationKind.AGENT_DEV
        )
        materialize_population(
            self.repository_root / development.manifest,
            task_dir / "dev_instances",
        )
        self._write_task_yaml(task, record.manifest_path, repository, task_dir)
        write_agent_tooling(
            repository_root=self.repository_root, task=task, task_dir=task_dir
        )
        (task_dir / "Dockerfile").write_text(self._dockerfile(task))
        (task_dir / "docker-compose.yaml").write_text(self._compose())
        return task_dir

    @staticmethod
    def _validate_repository(task: PitBenchTask, repository: Path) -> None:
        import subprocess

        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip()
        if head != task.event.base_commit:
            raise ValueError(f"repository HEAD {head} != {task.event.base_commit}")
        refs = subprocess.check_output(
            ["git", "for-each-ref", "--format=%(refname)"],
            cwd=repository,
            text=True,
        ).splitlines()
        remotes = subprocess.check_output(
            ["git", "remote"], cwd=repository, text=True
        ).splitlines()
        if refs or remotes:
            raise ValueError("repository source is not time-censored")
        future = subprocess.run(
            ["git", "cat-file", "-e", f"{task.event.human_commit}^{{commit}}"],
            cwd=repository,
            check=False,
            capture_output=True,
        )
        if future.returncode == 0:
            raise ValueError("repository exposes the human commit")

    def _write_task_yaml(
        self,
        task: PitBenchTask,
        manifest: Path,
        repository: Path,
        task_dir: Path,
    ) -> None:
        payload = {
            "instruction": task.instruction,
            "author_name": "PitBench",
            "author_email": "benchmark@pitbench.invalid",
            "difficulty": "hard",
            "category": "solver_optimization",
            "tags": [task.task_type.value, task.problem_family.value],
            "parser_name": None,
            "evaluator_import_path": ("pitbench.evaluator.evaluator:PitBenchEvaluator"),
            "evaluator_config": {
                "manifest_path": str(manifest.resolve()),
                "base_repository": str(repository.resolve()),
                "private_root": str(self.private_root),
            },
            "max_agent_timeout_sec": 3600,
            "max_test_timeout_sec": 1,
            "max_setup_timeout_sec": 1800,
            "run_tests_in_same_shell": False,
        }
        (task_dir / "task.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))

    @staticmethod
    def _dockerfile(task: PitBenchTask) -> str:
        plugin = task.repository.plugin
        image = task.repository.agent_image or _AGENT_IMAGES[plugin]
        packages = (
            "git tmux asciinema python3 python3-pip time " + _BUILD_PACKAGES[plugin]
        )
        return f"""FROM {image}
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y {packages} \\
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /logs /agent-logs /pitbench/dev_instances
COPY agent_tooling /opt/pitbench-tooling
COPY agent_bin/pitbench /usr/local/bin/pitbench
COPY agent_config.json /pitbench/config.json
COPY repo /workspace/repo
COPY dev_instances /pitbench/dev_instances
RUN chmod 0755 /usr/local/bin/pitbench
ENV PYTHONPATH=/opt/pitbench-tooling
WORKDIR /workspace/repo
CMD ["sh", "-c", "sleep infinity"]
"""

    @staticmethod
    def _compose() -> str:
        return """services:
  client:
    build:
      context: .
      dockerfile: Dockerfile
    image: ${T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME}
    container_name: ${T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME}
    network_mode: none
    command: ["sh", "-c", "sleep infinity"]
    volumes:
      - ${T_BENCH_TASK_LOGS_PATH}:${T_BENCH_CONTAINER_LOGS_PATH}
      - ${T_BENCH_TASK_AGENT_LOGS_PATH}:${T_BENCH_CONTAINER_AGENT_LOGS_PATH}
    environment:
      OMP_NUM_THREADS: "1"
      OPENBLAS_NUM_THREADS: "1"
      MKL_NUM_THREADS: "1"
      PYTHONHASHSEED: "0"
      TZ: "UTC"
"""
