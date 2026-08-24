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
    "pitbench.repositories.pyvrp:PyVRPRepositoryPlugin": "python:3.13-trixie",
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

_PYTHON_PACKAGES = {
    "pitbench.repositories.pyvrp:PyVRPRepositoryPlugin": (
        "docblock matplotlib meson ninja numpy pandas poetry-core pyarrow "
        "pybind11 pydantic "
        "pytest pytest-cov pytest-timeout pytest-xdist pyyaml setuptools tqdm "
        "vrplib wheel"
    ),
}

_PREBUILD_COMMANDS = {
    "pitbench.repositories.pyvrp:PyVRPRepositoryPlugin": (
        "RUN python3 -m pip install --break-system-packages "
        "--no-build-isolation --no-deps -e /workspace/repo\n"
    ),
    "pitbench.repositories.vroom:VroomRepositoryPlugin": (
        "RUN cd /workspace/repo && make -j1 CXXFLAGS='-O3 -DNDEBUG'\n"
    ),
    "pitbench.repositories.highs:HighsRepositoryPlugin": (
        "RUN cd /workspace/repo && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && "
        "cmake --build build -j1\n"
    ),
}

# Increment when the generated task image or bundled public tooling becomes
# incompatible with an image produced by an earlier PitBench checkout.
IMAGE_REVISION = "1"
IMAGE_REVISION_LABEL = "org.pitbench.image-revision"
IMAGE_SOURCE_LABEL = "org.pitbench.image-source"


class PitBenchAdapter:
    """Materialize an agent-safe PitBench task from a PitBench manifest."""

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
        agent_image: str | None = None,
        judge_image: str | None = None,
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
                base_commit=task.release.base_commit,
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
        self._write_task_yaml(
            task,
            record.manifest_path,
            repository,
            task_dir,
            judge_image=judge_image,
        )
        write_agent_tooling(
            repository_root=self.repository_root, task=task, task_dir=task_dir
        )
        prepared_image = agent_image or task.repository.agent_image
        (task_dir / "Dockerfile").write_text(
            self._dockerfile(task, image_override=agent_image)
        )
        if prepared_image is not None:
            (task_dir / ".dockerignore").write_text(self._dockerignore())
        (task_dir / "docker-compose.yaml").write_text(self._compose())
        return task_dir

    @staticmethod
    def _validate_repository(task: PitBenchTask, repository: Path) -> None:
        import subprocess

        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip()
        if head != task.release.base_commit:
            tree = subprocess.check_output(
                ["git", "rev-parse", "HEAD^{tree}"], cwd=repository, text=True
            ).strip()
            if task.release.tree_sha is None or tree != task.release.tree_sha:
                raise ValueError(
                    "repository release identity mismatch: "
                    f"HEAD {head} != {task.release.base_commit} and "
                    f"tree {tree} != {task.release.tree_sha}"
                )
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

    def _write_task_yaml(
        self,
        task: PitBenchTask,
        manifest: Path,
        repository: Path,
        task_dir: Path,
        *,
        judge_image: str | None = None,
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
        if judge_image is not None:
            payload["evaluator_config"]["judge_image"] = judge_image
        (task_dir / "task.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))

    @staticmethod
    def _dockerfile(task: PitBenchTask, *, image_override: str | None = None) -> str:
        plugin = task.repository.plugin
        prepared_image = image_override or task.repository.agent_image
        if prepared_image is not None:
            return PitBenchAdapter._prepared_dockerfile(prepared_image)

        image = _AGENT_IMAGES[plugin]
        packages = (
            "git tmux asciinema python3 python3-pip time " + _BUILD_PACKAGES[plugin]
        )
        python_packages = _PYTHON_PACKAGES.get(plugin)
        install_python = (
            "RUN python3 -m pip install --break-system-packages "
            f"--no-cache-dir {python_packages}\n"
            if python_packages
            else ""
        )
        prebuild = _PREBUILD_COMMANDS.get(plugin, "")
        return f"""FROM {image}
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y {packages} \\
    && rm -rf /var/lib/apt/lists/*
{install_python}ENV PIP_NO_BUILD_ISOLATION=1
RUN mkdir -p /logs /agent-logs /pitbench/dev_instances
COPY agent_tooling /opt/pitbench-tooling
COPY agent_bin/pitbench /usr/local/bin/pitbench
COPY agent_config.json /pitbench/config.json
RUN rm -rf /workspace/repo
COPY repo /workspace/repo
{prebuild}COPY dev_instances /pitbench/dev_instances
RUN chmod 0755 /usr/local/bin/pitbench
LABEL {IMAGE_REVISION_LABEL}="{IMAGE_REVISION}" \
      {IMAGE_SOURCE_LABEL}="{image}"
ENV PYTHONPATH=/opt/pitbench-tooling
WORKDIR /workspace/repo
CMD ["sh", "-c", "sleep infinity"]
"""

    @staticmethod
    def _prepared_dockerfile(image: str) -> str:
        return f"""FROM {image}
RUN mkdir -p /logs /agent-logs /pitbench/dev_instances
COPY agent_tooling /opt/pitbench-tooling
COPY agent_bin/pitbench /usr/local/bin/pitbench
COPY agent_config.json /pitbench/config.json
COPY dev_instances /pitbench/dev_instances
RUN chmod 0755 /usr/local/bin/pitbench
LABEL {IMAGE_REVISION_LABEL}="{IMAGE_REVISION}" \
      {IMAGE_SOURCE_LABEL}="{image}"
ENV PYTHONPATH=/opt/pitbench-tooling
WORKDIR /workspace/repo
CMD ["sh", "-c", "sleep infinity"]
"""

    @staticmethod
    def _dockerignore() -> str:
        return """*
!Dockerfile
!agent_tooling/
!agent_tooling/**
!agent_bin/
!agent_bin/**
!agent_config.json
!dev_instances/
!dev_instances/**
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
