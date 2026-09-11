from __future__ import annotations

import json
import shlex
import shutil
from collections.abc import Iterable
from pathlib import Path

import yaml

from adapters.pitbench.agent_tooling import write_agent_tooling
from adapters.pitbench.git_snapshot import GitSnapshot
from pitbench.agent_tools import (
    AGENT_TOOLS_METADATA,
    AgentTool,
    agent_tool_names,
    agent_tools_label,
    needs_development_instances,
    needs_run_directory,
    normalize_agent_tools,
)
from pitbench.instances import materialize_instance_set
from pitbench.repositories.base import RepositoryPluginRegistry
from pitbench.schema.task import InstanceSetKind, PitBenchTask
from pitbench.tasks import TaskCatalog

# Increment when the generated task image or bundled public tooling becomes
# incompatible with an image produced by an earlier PitBench checkout.
IMAGE_REVISION = "6"
IMAGE_REVISION_LABEL = "org.pitbench.image-revision"
IMAGE_SOURCE_LABEL = "org.pitbench.image-source"
IMAGE_TOOLS_LABEL = "org.pitbench.agent-tools"


class PitBenchAdapter:
    """Materialize an agent-safe PitBench task from a task config."""

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
        judge_cpus: float | None = None,
        judge_memory: str | None = None,
        judge_parallel_runs: int | None = None,
        base_observations_path: Path | None = None,
        base_cache_path: Path | None = None,
        use_base_cache: bool = True,
        agent_tools: Iterable[AgentTool | str] = (),
    ) -> Path:
        record = self.catalog.validate_one(task_id)
        task = record.task
        tools = normalize_agent_tools(agent_tools)
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
            self.validate_repository(task, repository)

        if needs_development_instances(tools):
            development = next(
                instance_set
                for instance_set in task.instance_sets
                if instance_set.kind == InstanceSetKind.AGENT_DEV
            )
            materialize_instance_set(
                self.repository_root / development.instance_set_config,
                task_dir / "dev_instances",
            )
        (task_dir / AGENT_TOOLS_METADATA).write_text(
            json.dumps({"agent_tools": agent_tool_names(tools)}, indent=2) + "\n"
        )
        self._write_task_yaml(
            task,
            record.task_config_path,
            repository,
            task_dir,
            judge_image=judge_image,
            judge_cpus=judge_cpus,
            judge_memory=judge_memory,
            judge_parallel_runs=judge_parallel_runs,
            base_observations_path=base_observations_path,
            base_cache_path=base_cache_path,
            use_base_cache=use_base_cache,
            agent_tools=tools,
        )
        if tools:
            write_agent_tooling(
                repository_root=self.repository_root,
                task=task,
                task_dir=task_dir,
                agent_tools=tools,
            )
        prepared_image = agent_image or task.repository.agent_image
        (task_dir / "Dockerfile").write_text(
            self._dockerfile(task, image_override=agent_image, agent_tools=tools)
        )
        if prepared_image is not None:
            (task_dir / ".dockerignore").write_text(self._dockerignore(tools))
        (task_dir / "docker-compose.yaml").write_text(self._compose(tools))
        return task_dir

    @staticmethod
    def validate_repository(task: PitBenchTask, repository: Path) -> None:
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
        status = subprocess.check_output(
            [
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--ignored=matching",
            ],
            cwd=repository,
            text=True,
        ).splitlines()
        if status:
            raise ValueError("repository source has local modifications")

    @staticmethod
    def _format_instruction(
        task: PitBenchTask,
        agent_tools: frozenset[AgentTool] = frozenset(),
    ) -> str:
        editable_paths = ", ".join(task.repository.editable_paths)
        scope = getattr(task, "optimization_scope", None) or "solver performance"
        primary_budget = getattr(task.evaluation, "primary_budget_sec", None)
        budget_info = f"{primary_budget:.1f}s" if primary_budget else "the evaluation budget"

        sections = [
            f"{task.instruction}\n\n"
            f"The repository is read-only outside these editable paths: {editable_paths}. "
            "Keep all implementation changes inside them.",
            "### Task Context\n"
            f"- Target Solver: {task.release.repository} v{task.release.version}\n"
            f"- Problem Family: {task.problem_family.value.upper()}\n"
            f"- Optimization Scope: {scope}\n"
            f"- Primary Evaluation Budget: {budget_info} per instance",
            "### Evaluation & Correctness Requirements\n"
            "- Feasibility & Validity: 100% solution validity, constraint satisfaction, and format compatibility are mandatory. Any crash, timeout, or invalid solution will completely disqualify the candidate (Score = 0).\n"
            "- Evaluation Metric: Solution quality (normalized gap to optimal/BKS) within the fixed time budget.\n"
            "- Robustness: The candidate will be evaluated across unseen test instances, multiple random seeds, and perturbed representations. Do not overfit hyperparameters or hardcode behaviors for specific instances.",
            "### Recommended Performance Engineering SOP\n"
            "1. Baseline: Run visible development instances (located in `agent_dev`) on the clean repository to establish reference runtime, objective values, and feasibility.\n"
            f"2. Profile: Run profiling tools on development instances to pinpoint computational bottlenecks in `{scope}`. Formulate concrete hypotheses before making edits.\n"
            "3. Targeted Optimization: Implement minimal, high-impact, semantics-preserving improvements focused on the identified hot paths (e.g., caching redundant calculations, pruning invalid search neighborhoods early, reducing inner-loop allocations). Avoid blind parameter guessing or disruptive rewrites.\n"
            "4. Differential Verification: Re-run the development instances under the same budget. Confirm that solution validity is 100% maintained and that performance/objective is genuinely improved.\n"
            "5. Regression & Cleanup: Run existing repository unit/integration tests and `pitbench validate` (if available) to ensure no regressions. Remove any scratch scripts, profiling traces, or generated output files so that `git status` contains only intentional source changes.",
        ]

        if agent_tools:
            commands = ", ".join(
                f"`pitbench {tool}`" for tool in agent_tool_names(agent_tools)
            )
            sections.append(f"Optional PitBench helper commands: {commands}.")

        return "\n\n".join(sections)

    def _write_task_yaml(
        self,
        task: PitBenchTask,
        task_config: Path,
        repository: Path,
        task_dir: Path,
        *,
        judge_image: str | None = None,
        judge_cpus: float | None = None,
        judge_memory: str | None = None,
        judge_parallel_runs: int | None = None,
        base_observations_path: Path | None = None,
        base_cache_path: Path | None = None,
        use_base_cache: bool = True,
        agent_tools: frozenset[AgentTool] = frozenset(),
    ) -> None:
        instruction = self._format_instruction(task, agent_tools=agent_tools)
        payload = {
            "instruction": instruction,
            "author_name": "PitBench",
            "author_email": "benchmark@pitbench.invalid",
            "difficulty": "hard",
            "category": "solver_optimization",
            "tags": [task.task_type.value, task.problem_family.value],
            "evaluator_import_path": ("pitbench.evaluator.evaluator:PitBenchEvaluator"),
            "evaluator_config": {
                "task_config_path": str(task_config.resolve()),
                "base_repository": str(repository.resolve()),
                "private_root": str(self.private_root),
                "use_base_cache": use_base_cache,
            },
            "max_agent_timeout_sec": 3600,
            "max_setup_timeout_sec": 1800,
        }
        if judge_image is not None:
            payload["evaluator_config"]["judge_image"] = judge_image
        if judge_cpus is not None:
            payload["evaluator_config"]["judge_cpus"] = judge_cpus
        if judge_memory is not None:
            payload["evaluator_config"]["judge_memory"] = judge_memory
        if judge_parallel_runs is not None:
            payload["evaluator_config"]["judge_parallel_runs"] = judge_parallel_runs
        if base_observations_path is not None:
            payload["evaluator_config"]["base_observations_path"] = str(base_observations_path)
        if base_cache_path is not None:
            payload["evaluator_config"]["base_cache_path"] = str(base_cache_path)
        (task_dir / "task.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))

    @staticmethod
    def _dockerfile(
        task: PitBenchTask,
        *,
        image_override: str | None = None,
        agent_tools: Iterable[AgentTool | str] = (),
    ) -> str:
        repository = RepositoryPluginRegistry.load(task.repository.plugin)
        tools = normalize_agent_tools(agent_tools)
        prepared_image = image_override or task.repository.agent_image
        if prepared_image is not None:
            return PitBenchAdapter._prepared_dockerfile(
                prepared_image, task, agent_tools=tools
            )

        environment = repository.agent_environment
        if environment is None:
            raise ValueError(
                "repository plugin requires an agent environment or configured image"
            )
        image = environment.image
        packages = (
            "git tmux asciinema python3 python3-pip time " + environment.system_packages
        )
        python_packages = environment.python_packages
        install_python = (
            "RUN python3 -m pip install --break-system-packages "
            f"--no-cache-dir {python_packages}\n"
            if python_packages
            else ""
        )
        prebuild = environment.prebuild
        tooling_layers = PitBenchAdapter._tooling_layers(tools)
        development_copy = PitBenchAdapter._development_copy(tools)
        workspace_permissions = PitBenchAdapter._workspace_permissions(
            task, include_runs=needs_run_directory(tools)
        )
        pythonpath = "ENV PYTHONPATH=/opt/pitbench-tooling\n" if tools else ""
        return f"""FROM {image}
USER root
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y {packages} \\
    && rm -rf /var/lib/apt/lists/*
{install_python}ENV PIP_NO_BUILD_ISOLATION=1
RUN mkdir -p /logs /agent-logs
{tooling_layers}
RUN rm -rf /workspace/repo
COPY repo /workspace/repo
{prebuild}{workspace_permissions}{development_copy}LABEL {IMAGE_REVISION_LABEL}="{IMAGE_REVISION}" \
      {IMAGE_SOURCE_LABEL}="{image}" \
      {IMAGE_TOOLS_LABEL}="{agent_tools_label(tools)}"
{pythonpath}WORKDIR /workspace/repo
CMD ["sh", "-c", "sleep infinity"]
"""

    @staticmethod
    def _prepared_dockerfile(
        image: str, task: PitBenchTask, *, agent_tools: frozenset[AgentTool]
    ) -> str:
        tooling_layers = PitBenchAdapter._tooling_layers(agent_tools)
        development_copy = PitBenchAdapter._development_copy(agent_tools)
        workspace_permissions = PitBenchAdapter._workspace_permissions(
            task, include_runs=needs_run_directory(agent_tools)
        )
        pythonpath = "ENV PYTHONPATH=/opt/pitbench-tooling\n" if agent_tools else ""
        return f"""FROM {image}
USER root
RUN mkdir -p /logs /agent-logs
{tooling_layers}{development_copy}{workspace_permissions}LABEL {IMAGE_REVISION_LABEL}="{IMAGE_REVISION}" \
      {IMAGE_SOURCE_LABEL}="{image}" \
      {IMAGE_TOOLS_LABEL}="{agent_tools_label(agent_tools)}"
{pythonpath}WORKDIR /workspace/repo
CMD ["sh", "-c", "sleep infinity"]
"""

    @staticmethod
    def _tooling_layers(agent_tools: frozenset[AgentTool]) -> str:
        if not agent_tools:
            return ""
        return """RUN mkdir -p /pitbench
COPY agent_tooling /opt/pitbench-tooling
COPY agent_bin/pitbench /usr/local/bin/pitbench
COPY agent_config.json /pitbench/config.json
RUN chmod 0755 /usr/local/bin/pitbench
"""

    @staticmethod
    def _development_copy(agent_tools: frozenset[AgentTool]) -> str:
        if not needs_development_instances(agent_tools):
            return ""
        return "COPY dev_instances /pitbench/dev_instances\n"

    @staticmethod
    def _workspace_permissions(task: PitBenchTask, *, include_runs: bool) -> str:
        editable_paths = [
            shlex.quote(f"/workspace/repo/{path}")
            for path in task.repository.editable_paths
        ]
        writable_paths = [*editable_paths, "/logs", "/agent-logs"]
        if include_runs:
            writable_paths.append("/pitbench/runs")
        writable = " ".join(writable_paths)
        return f"""ARG PITBENCH_AGENT_UID=1000
ARG PITBENCH_AGENT_GID=1000
USER root
RUN groupadd --non-unique --gid ${{PITBENCH_AGENT_GID}} pitbench-agent \
    && useradd --non-unique --uid ${{PITBENCH_AGENT_UID}} \
       --gid ${{PITBENCH_AGENT_GID}} --create-home --shell /bin/bash pitbench-agent \
    && chown -R root:root /workspace/repo \
    && chmod -R a=rX /workspace/repo \
    && mkdir -p {writable} \
    && chown -R ${{PITBENCH_AGENT_UID}}:${{PITBENCH_AGENT_GID}} \
       {writable} \
    && chmod -R u+rwX {writable} \
    && git config --system --add safe.directory /workspace/repo
ENV HOME=/home/pitbench-agent
USER pitbench-agent
"""

    @staticmethod
    def _dockerignore(agent_tools: Iterable[AgentTool | str] = ()) -> str:
        tools = normalize_agent_tools(agent_tools)
        paths = ["*", "!Dockerfile"]
        if tools:
            paths.extend(
                [
                    "!agent_tooling/",
                    "!agent_tooling/**",
                    "!agent_bin/",
                    "!agent_bin/**",
                    "!agent_config.json",
                ]
            )
        if needs_development_instances(tools):
            paths.extend(["!dev_instances/", "!dev_instances/**"])
        return "\n".join(paths) + "\n"

    @staticmethod
    def _compose(agent_tools: Iterable[AgentTool | str] = ()) -> str:
        runs_environment = (
            '      PITBENCH_RUNS: "/pitbench/runs"\n'
            if needs_run_directory(agent_tools)
            else ""
        )
        return f"""services:
  client:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        PITBENCH_AGENT_UID: ${{T_BENCH_AGENT_UID:-1000}}
        PITBENCH_AGENT_GID: ${{T_BENCH_AGENT_GID:-1000}}
    image: ${{T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME}}
    container_name: ${{T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME}}
    network_mode: none
    security_opt:
      - no-new-privileges:true
    command: ["sh", "-c", "sleep infinity"]
    volumes:
      - ${{T_BENCH_TASK_LOGS_PATH}}:${{T_BENCH_CONTAINER_LOGS_PATH}}
      - ${{T_BENCH_TASK_AGENT_LOGS_PATH}}:${{T_BENCH_CONTAINER_AGENT_LOGS_PATH}}
    environment:
      OMP_NUM_THREADS: "1"
      OPENBLAS_NUM_THREADS: "1"
      MKL_NUM_THREADS: "1"
{runs_environment}      PYTHONHASHSEED: "0"
      TZ: "UTC"
"""
