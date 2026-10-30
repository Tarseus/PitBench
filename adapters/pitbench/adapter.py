from __future__ import annotations

import fcntl
import hashlib
import json
import shlex
import shutil
import subprocess
import tempfile
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
from pitbench.evaluator.private_assets import PrivateAssetResolver
from pitbench.instances import materialize_instance_set
from pitbench.repositories.base import RepositoryPluginRegistry
from pitbench.research import resolve_research_budget
from pitbench.schema.research import ResearchBudget
from pitbench.schema.task import InstanceSetKind, PerformanceProtocol, PitBenchTask
from pitbench.tasks import TaskCatalog

# Increment when the generated task image or bundled public tooling becomes
# incompatible with an image produced by an earlier PitBench checkout.
IMAGE_REVISION = "17"
IMAGE_REVISION_LABEL = "org.pitbench.image-revision"
IMAGE_SOURCE_LABEL = "org.pitbench.image-source"
IMAGE_TOOLS_LABEL = "org.pitbench.agent-tools"
IMAGE_CONTEXT_LABEL = "org.pitbench.image-context-sha256"


def image_context_digest(task_dir: Path) -> str:
    """Hash files copied into a prepared task image, excluding the hash label."""
    files = [task_dir / "Dockerfile", task_dir / ".dockerignore"]
    for name in ("agent_config.json",):
        path = task_dir / name
        if path.is_file():
            files.append(path)
    for name in ("agent_bin", "agent_tooling", "dev_instances"):
        directory = task_dir / name
        if directory.is_dir():
            files.extend(path for path in directory.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in sorted(files):
        content = path.read_bytes()
        if path.name == "Dockerfile":
            content = (
                b"\n".join(
                    line
                    for line in content.splitlines()
                    if not line.startswith(f"LABEL {IMAGE_CONTEXT_LABEL}=".encode())
                )
                + b"\n"
            )
        digest.update(str(path.relative_to(task_dir)).encode() + b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


class PitBenchAdapter:
    """Materialize an agent-safe PitBench task from a task config."""

    def __init__(self, repository_root: Path, private_root: Path) -> None:
        self.repository_root = repository_root.resolve()
        self.private_root = PrivateAssetResolver(private_root).root
        self.catalog = TaskCatalog(self.repository_root)

    def materialize(
        self,
        task_id: str,
        destination: Path,
        *,
        agent_image: str | None = None,
        judge_image: str | None = None,
        judge_cpus: float | None = None,
        judge_memory: str | None = None,
        judge_parallel_runs: int | None = None,
        judge_cpuset_cpus: str | None = None,
        base_observations_path: Path | None = None,
        base_cache_path: Path | None = None,
        metric_scenario_config: Path | None = None,
        use_base_cache: bool = False,
        smoke_test: bool = False,
        agent_tools: Iterable[AgentTool | str] = (),
        agent_feedback_mode: str = "solver_behavior",
        research_track: str | None = None,
        research_budget: ResearchBudget | None = None,
        candidate_hypotheses_required: bool = False,
    ) -> Path:
        if agent_feedback_mode not in {"outcome_only", "solver_behavior"}:
            raise ValueError("unknown agent feedback mode")
        record = self.catalog.validate_one(task_id)
        task = record.task
        budget = resolve_research_budget(
            self.repository_root,
            track=research_track,
            configured=research_budget or task.research_budget,
            smoke_test=smoke_test,
        )
        tools = normalize_agent_tools(agent_tools)
        task_dir = destination.resolve()
        if task_dir.exists():
            raise FileExistsError(task_dir)
        task_dir.mkdir(parents=True)
        repository = task_dir / "repo"
        self._materialize_cached_repository(
            task, repository, task_dir / "agent_repo.bundle"
        )

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
            json.dumps(
                {
                    "agent_tools": agent_tool_names(tools),
                    "agent_feedback_mode": agent_feedback_mode,
                },
                indent=2,
            )
            + "\n"
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
            judge_cpuset_cpus=judge_cpuset_cpus,
            base_observations_path=base_observations_path,
            base_cache_path=base_cache_path,
            metric_scenario_config=metric_scenario_config,
            use_base_cache=use_base_cache,
            smoke_test=smoke_test,
            agent_tools=tools,
            research_budget=budget,
        )
        if tools:
            write_agent_tooling(
                repository_root=self.repository_root,
                task=task,
                task_dir=task_dir,
                agent_tools=tools,
                agent_feedback_mode=agent_feedback_mode,
                research_budget=budget,
                candidate_hypotheses_required=candidate_hypotheses_required,
            )
        prepared_image = agent_image or task.repository.agent_image
        submodule_paths = self._submodule_paths(repository) if prepared_image else ()
        module_metadata = (repository / ".git/modules").is_dir()
        (task_dir / "Dockerfile").write_text(
            self._dockerfile(
                task,
                image_override=agent_image,
                agent_tools=tools,
                submodule_paths=submodule_paths,
                module_metadata=module_metadata,
            )
        )
        if prepared_image is not None:
            (task_dir / ".dockerignore").write_text(
                self._dockerignore(tools, submodule_paths, module_metadata)
            )
        (task_dir / "docker-compose.yaml").write_text(
            self._compose(tools, research_budget=budget)
        )
        if prepared_image is not None:
            digest = image_context_digest(task_dir)
            with (task_dir / "Dockerfile").open("a") as dockerfile:
                dockerfile.write(f'LABEL {IMAGE_CONTEXT_LABEL}="{digest}"\n')
        return task_dir

    def _materialize_cached_repository(
        self, task: PitBenchTask, destination: Path, bundle_path: Path
    ) -> None:
        cached_repository = self.ensure_source_cache(task)
        shutil.copytree(cached_repository, destination, symlinks=True)
        self.validate_repository(task, destination)
        GitSnapshot._run(
            ["git", "bundle", "create", str(bundle_path.resolve()), "HEAD"],
            destination,
        )

    def ensure_source_cache(self, task: PitBenchTask) -> Path:
        cached_repository = self.source_cache_path(self.repository_root, task)
        cache_root = cached_repository.parent
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_name = cached_repository.name
        lock_path = cache_root / f"{cache_name}.lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if not cached_repository.exists():
                with tempfile.TemporaryDirectory(
                    prefix=f"{cache_name}.tmp-", dir=cache_root
                ) as temporary:
                    prepared = Path(temporary) / "repo"
                    GitSnapshot(
                        clone_url=task.repository.clone_url,
                        base_commit=task.release.base_commit,
                    ).create(prepared)
                    self.validate_repository(task, prepared)
                    prepared.rename(cached_repository)
            self.validate_repository(task, cached_repository)
            head = GitSnapshot._run(
                ["git", "rev-parse", "HEAD"], cached_repository
            ).stdout.strip()
            if head != task.release.base_commit:
                raise ValueError(
                    f"cached source HEAD {head} != {task.release.base_commit}"
                )
        return cached_repository

    @staticmethod
    def source_cache_path(repository_root: Path, task: PitBenchTask) -> Path:
        url_hash = hashlib.sha256(task.repository.clone_url.encode()).hexdigest()[:12]
        name = f"{task.task_id}-{task.release.base_commit[:12]}-{url_hash}"
        return repository_root / ".pitbench/cache/source-snapshots" / name

    @staticmethod
    def validate_repository(task: PitBenchTask, repository: Path) -> None:
        PitBenchAdapter._validate_snapshot(
            repository,
            expected_commit=task.release.base_commit,
            expected_tree=task.release.tree_sha,
        )

    @staticmethod
    def _validate_snapshot(
        repository: Path,
        *,
        expected_commit: str,
        expected_tree: str | None = None,
    ) -> None:
        import subprocess

        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip()
        if head != expected_commit:
            tree = subprocess.check_output(
                ["git", "rev-parse", "HEAD^{tree}"], cwd=repository, text=True
            ).strip()
            if expected_tree is None or tree != expected_tree:
                raise ValueError(
                    "repository release identity mismatch: "
                    f"HEAD {head} != {expected_commit} and "
                    f"tree {tree} != {expected_tree}"
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
        gitlinks = subprocess.check_output(
            ["git", "ls-files", "--stage"],
            cwd=repository,
            text=True,
        ).splitlines()
        for line in gitlinks:
            fields = line.split(maxsplit=3)
            if len(fields) != 4 or fields[0] != "160000":
                continue
            submodule_commit = fields[1]
            submodule_path = fields[3]
            PitBenchAdapter._validate_snapshot(
                repository / submodule_path,
                expected_commit=submodule_commit,
            )

    @staticmethod
    def _format_instruction(
        task: PitBenchTask,
        agent_tools: frozenset[AgentTool] = frozenset(),
    ) -> str:
        editable_paths = ", ".join(task.repository.editable_paths)
        repository_plugin = RepositoryPluginRegistry.load(task.repository.plugin)
        build_paths = ", ".join(repository_plugin.agent_build_paths)
        source_scope = (
            f"Only change implementation source in these editable paths: {editable_paths}. "
            + (
                f"Generated build outputs may be written to: {build_paths}."
                if build_paths
                else "Keep all implementation changes inside them."
            )
        )
        primary_budget = getattr(task.evaluation, "primary_budget_sec", None)
        budget_info = (
            f"{primary_budget:.1f}s" if primary_budget else "the evaluation budget"
        )

        protocol = task.evaluation.performance_protocol
        if protocol is PerformanceProtocol.HEURISTIC_FIXED_BUDGET:
            metric = (
                "- Primary Performance Metric: paired reduction in normalized gap "
                "to the fixed optimum/BKS anchor at the primary fixed time budget; "
                "lower gap is better."
            )
            comparison = (
                "Compare independently checked objectives at the same budget, "
                "instance, and seed; compare normalized gaps only where a public "
                "anchor exists."
            )
        elif protocol is PerformanceProtocol.EXACT_VERIFIED_SOLVE:
            metric = (
                "- Primary Performance Metric: independently verified solved "
                "coverage at the primary budget. When coverage ties, compare "
                f"paired PAR-2 runtime using {task.evaluation.exact_time_basis.value}."
            )
            comparison = (
                "Track termination, objective, and public target where available. "
                "The formal evaluator compares verified solved coverage first, "
                "then paired PAR-2 runtime."
            )
        else:
            raise ValueError(f"unsupported performance protocol: {protocol}")

        has_instances = needs_development_instances(agent_tools)
        development = (
            "1. Baseline: Before editing, run `pitbench inspect` and "
            "`pitbench bench --label base` on visible instances at "
            "`/pitbench/dev_instances`. `pitbench bench` accepts `--instance` "
            "(visible ID/name or a readable custom file), `--seed`, `--budget`, "
            "and `--threads`; `--all-seeds` and `--all-budgets` run the declared "
            "development grid. Use matching values for Base and Agent.\n"
            if AgentTool.INSPECT in agent_tools and AgentTool.BENCH in agent_tools
            else "1. Baseline: Record the clean repository's behavior on public "
            "development cases before editing.\n"
        )
        if not has_instances:
            development = (
                "1. Baseline: Use repository tests or examples to record the "
                "clean repository's behavior before editing. No PitBench "
                "development instances are mounted in this task.\n"
            )
        rerun = (
            "4. Differential Verification: Run "
            "`pitbench bench --label agent --build` "
            "on the same development instances, seeds, and budget. "
            f"{comparison} Public bench output is development feedback; "
            "the evaluator independently verifies submitted solutions. "
            "Without `--build`, bench executes the current artifact. Check its "
            "fingerprint and build warning before trusting a comparison. Run "
            "`pitbench diagnose` after paired runs to see available per-instance "
            "effects, missing grid cells, and the next development experiment. "
            "Inspect regressions before revising the code.\n"
            if AgentTool.BENCH in agent_tools
            else "4. Differential Verification: Rebuild the changed solver and "
            "repeat the baseline cases under the same seed and budget. "
            f"{comparison}\n"
        )
        validation = (
            " and `pitbench validate`" if AgentTool.VALIDATE in agent_tools else ""
        )
        sections = [
            f"{task.instruction}\n\n" + source_scope,
            "### Task Context\n"
            f"- Target Solver: {task.release.repository} v{task.release.version}\n"
            f"- Problem Family: {task.problem_family.value.upper()}\n"
            f"- Performance Protocol: {protocol.value}\n"
            f"- Primary Evaluation Budget: {budget_info} per instance",
            "### Evaluation & Correctness Requirements\n"
            "- Feasibility & Validity: 100% solution validity, constraint satisfaction, and format compatibility are mandatory. Confirmed semantic invalidity disqualifies the candidate.\n"
            "- Operational Failures: Crashes, timeouts, out-of-memory failures, numerical failures, and missing targets are retained and reported under Operational Reliability; they do not by themselves fail the correctness qualification.\n"
            f"{metric}\n"
            "- Robustness: The candidate will be evaluated across unseen test instances, multiple random seeds, and perturbed representations. Do not overfit hyperparameters or hardcode behaviors for specific instances.",
            "### Recommended Performance Engineering SOP\n"
            + development
            + "2. Diagnose: Investigate the solver's computational cost and behavior: incumbent or bound progress, search work, time to useful solutions, and the instance structures that stall or regress. Choose the optimization direction yourself.\n"
            "3. Targeted Optimization: Make a bounded, semantics-preserving change that addresses the diagnosed mechanism. Distinguish implementation throughput from search-policy changes, and do not treat more iterations or lower runtime alone as a solver improvement. Avoid blind parameter guessing or disruptive rewrites.\n"
            + rerun
            + "5. Regression & Cleanup: Run existing repository unit/integration tests"
            f"{validation} to check for regressions. Remove scratch scripts, "
            "profiling traces, and generated files from the submitted source changes.",
            "### Final Submission\n"
            "Your submission is the source diff left in the task repository when "
            "execution ends. Leave the selected candidate implementation in place "
            "when cleaning temporary files. Do not create commits or change "
            "repository HEAD. Before finishing, inspect `git diff HEAD` and "
            "`git status --short`, and confirm that the retained changes match "
            "your final report. If no source changes are retained, state that "
            "explicitly and explain the attempted changes, why they were "
            "reverted or abandoned, and any unresolved blockers. Distinguish "
            "completed validation and measured results from untested claims.",
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
        judge_cpuset_cpus: str | None = None,
        base_observations_path: Path | None = None,
        base_cache_path: Path | None = None,
        metric_scenario_config: Path | None = None,
        use_base_cache: bool = False,
        smoke_test: bool = False,
        agent_tools: frozenset[AgentTool] = frozenset(),
        research_budget: ResearchBudget | None = None,
    ) -> None:
        budget = research_budget or resolve_research_budget(
            self.repository_root,
            configured=task.research_budget,
            smoke_test=smoke_test,
        )
        instruction = self._format_instruction(task, agent_tools=agent_tools)
        instruction += (
            "\n\n### Agent Research Budget\n"
            f"- Track: {budget.track}; total wall time: {budget.wall_time_sec:g} seconds.\n"
            "- Model waits, compilation, local validation and experiments count. "
            "Fixed environment preparation and final hidden Judge are separate.\n"
            "- Intermediate source checkpoints receive no hidden feedback.\n"
            f"- Resource declaration: {budget.model_dump_json()}"
        )
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
            "max_agent_timeout_sec": budget.wall_time_sec,
            "research_budget": budget.model_dump(mode="json"),
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
        if judge_cpuset_cpus is not None:
            payload["evaluator_config"]["judge_cpuset_cpus"] = judge_cpuset_cpus
        if base_observations_path is not None:
            payload["evaluator_config"]["base_observations_path"] = str(
                base_observations_path
            )
        if base_cache_path is not None:
            payload["evaluator_config"]["base_cache_path"] = str(base_cache_path)
        if metric_scenario_config is not None:
            payload["evaluator_config"]["metric_scenario_config"] = str(
                metric_scenario_config.resolve()
            )
        if smoke_test:
            payload["evaluator_config"].update(
                smoke_test=True,
                instances_per_instance_set=1,
                solver_seed_limit=2,
            )
        (task_dir / "task.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))

    @staticmethod
    def _dockerfile(
        task: PitBenchTask,
        *,
        image_override: str | None = None,
        agent_tools: Iterable[AgentTool | str] = (),
        submodule_paths: tuple[str, ...] = (),
        module_metadata: bool = False,
    ) -> str:
        repository = RepositoryPluginRegistry.load(task.repository.plugin)
        tools = normalize_agent_tools(agent_tools)
        prepared_image = image_override or task.repository.agent_image
        if prepared_image is not None:
            return PitBenchAdapter._prepared_dockerfile(
                prepared_image,
                task,
                agent_tools=tools,
                submodule_paths=submodule_paths,
                module_metadata=module_metadata,
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
        image: str,
        task: PitBenchTask,
        *,
        agent_tools: frozenset[AgentTool],
        submodule_paths: tuple[str, ...] = (),
        module_metadata: bool = False,
    ) -> str:
        tooling_layers = PitBenchAdapter._tooling_layers(agent_tools)
        development_copy = PitBenchAdapter._development_copy(agent_tools)
        workspace_permissions = PitBenchAdapter._workspace_permissions(
            task, include_runs=needs_run_directory(agent_tools)
        )
        pythonpath = "ENV PYTHONPATH=/opt/pitbench-tooling\n" if agent_tools else ""
        submodule_copy = ""
        if submodule_paths:
            quoted_paths = " ".join(
                shlex.quote(f"/workspace/repo/{path}") for path in submodule_paths
            )
            submodule_copy = f"RUN rm -rf {quoted_paths}\n"
            if module_metadata:
                submodule_copy += (
                    "COPY repo/.git/modules /workspace/repo/.git/modules\n"
                )
            for path in submodule_paths:
                submodule_copy += (
                    "COPY "
                    + json.dumps([f"repo/{path}", f"/workspace/repo/{path}"])
                    + "\n"
                )
            submodule_copy += (
                "RUN git -C /workspace/repo submodule update "
                "--init --recursive --no-fetch\n"
            )
        submodule_copy += (
            'RUN test "$(git -C /workspace/repo rev-parse HEAD)" = '
            + shlex.quote(task.release.base_commit)
            + " \\\n"
            + "    && git -C /workspace/repo submodule status --recursive > /tmp/pitbench-submodules \\\n"
            + "    && ! grep -Eq '^[+U-]' /tmp/pitbench-submodules \\\n"
            + "    && rm /tmp/pitbench-submodules\n"
        )
        clean_checkout = (
            "RUN git -C /workspace/repo reset --hard HEAD \\\n"
            "    && git -C /workspace/repo submodule foreach --recursive "
            "'git reset --hard HEAD' \\\n"
            "    && git -C /workspace/repo clean -f -- .dockerignore \\\n"
        )
        build_paths = RepositoryPluginRegistry.load(
            task.repository.plugin
        ).agent_build_paths
        if build_paths:
            patterns = " ".join(shlex.quote(f"/{path}/") for path in build_paths)
            clean_checkout += (
                f"    && printf '%s\\n' {patterns} "
                ">> /workspace/repo/.git/info/exclude \\\n"
            )
        clean_checkout += (
            '    && test -z "$(git -C /workspace/repo status --porcelain)"\n'
        )
        return f"""FROM {image}
USER root
RUN mkdir -p /logs /agent-logs
{submodule_copy}{clean_checkout}{tooling_layers}{development_copy}{workspace_permissions}LABEL {IMAGE_REVISION_LABEL}="{IMAGE_REVISION}" \
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
RUN chmod 0755 /usr/local/bin/pitbench \
    && chmod -R a+rX /opt/pitbench-tooling /pitbench/config.json
"""

    @staticmethod
    def _development_copy(agent_tools: frozenset[AgentTool]) -> str:
        if not needs_development_instances(agent_tools):
            return ""
        return (
            "COPY dev_instances /pitbench/dev_instances\n"
            "RUN chmod -R a+rX /pitbench/dev_instances\n"
        )

    @staticmethod
    def _workspace_permissions(task: PitBenchTask, *, include_runs: bool) -> str:
        repository = RepositoryPluginRegistry.load(task.repository.plugin)
        editable_paths = [
            shlex.quote(f"/workspace/repo/{path}")
            for path in task.repository.editable_paths
        ]
        build_paths = [
            shlex.quote(f"/workspace/repo/{path}")
            for path in repository.agent_build_paths
        ]
        writable_paths = [
            *editable_paths,
            *build_paths,
            "/workspace/repo/.codex",
            "/workspace/repo/.agents",
            "/logs",
            "/agent-logs",
        ]
        if include_runs:
            writable_paths.append("/pitbench/runs")
        writable = " ".join(writable_paths)
        support_permissions = "".join(
            f"    && chmod -R a+rX {shlex.quote(path)} \\\n"
            for path in repository.agent_support_paths
        )
        return f"""ARG PITBENCH_AGENT_UID=1000
ARG PITBENCH_AGENT_GID=1000
USER root
RUN (getent group pitbench-agent >/dev/null \
       || groupadd --non-unique --gid ${{PITBENCH_AGENT_GID}} pitbench-agent) \
    && (id --user pitbench-agent >/dev/null 2>&1 \
        || useradd --non-unique --uid ${{PITBENCH_AGENT_UID}} \
           --gid ${{PITBENCH_AGENT_GID}} --create-home --shell /bin/bash pitbench-agent) \
{support_permissions}    && (if [ -d /root/.m2/repository ]; then \
          chmod o+x /root /root/.m2 && chmod -R a+rX /root/.m2/repository \
          && mkdir -p /home/pitbench-agent/.m2 \
          && ln -s /root/.m2/repository /home/pitbench-agent/.m2/repository \
          && chown ${{PITBENCH_AGENT_UID}}:${{PITBENCH_AGENT_GID}} \
             /home/pitbench-agent/.m2; \
        fi) \
    && chown -R root:root /workspace/repo \
    && chmod -R a=rX /workspace/repo \
    && mkdir -p {writable} /workspace/repo/.aws \
    && chown -R ${{PITBENCH_AGENT_UID}}:${{PITBENCH_AGENT_GID}} \
       {writable} \
    && chmod -R u+rwX {writable} \
    && git config --system --add safe.directory /workspace/repo
ENV HOME=/home/pitbench-agent
USER pitbench-agent
"""

    @staticmethod
    def _submodule_paths(repository: Path) -> tuple[str, ...]:
        entries = subprocess.check_output(
            ["git", "ls-files", "--stage", "-z"], cwd=repository
        ).split(b"\0")
        return tuple(
            entry.split(b"\t", 1)[1].decode()
            for entry in entries
            if entry.startswith(b"160000 ")
        )

    @staticmethod
    def _dockerignore(
        agent_tools: Iterable[AgentTool | str] = (),
        submodule_paths: tuple[str, ...] = (),
        module_metadata: bool = False,
    ) -> str:
        tools = normalize_agent_tools(agent_tools)
        paths = ["*", "!Dockerfile"]
        if submodule_paths:
            paths.append("!repo/")
            if module_metadata:
                paths.extend(
                    ["!repo/.git/", "!repo/.git/modules/", "!repo/.git/modules/**"]
                )
            for submodule in submodule_paths:
                parts = Path(submodule).parts
                paths.extend(
                    "!repo/" + "/".join(parts[:index]) + "/"
                    for index in range(1, len(parts) + 1)
                )
                paths.append(f"!repo/{submodule}/**")
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
    def _compose(
        agent_tools: Iterable[AgentTool | str] = (),
        *,
        research_budget: ResearchBudget | None = None,
    ) -> str:
        resources = ""
        if research_budget is not None:
            if research_budget.cpus is not None:
                resources += f"    cpus: {research_budget.cpus}\n"
            if research_budget.memory_gib is not None:
                resources += (
                    f"    mem_limit: {int(research_budget.memory_gib * 1024**3)}\n"
                )
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
{resources}    network_mode: none
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
