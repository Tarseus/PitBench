from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer

from adapters.pitbench.adapter import PitBenchAdapter
from pitbench.agent_tools import AgentTool
from pitbench.cli.auth import auth_app
from pitbench.cli.doctor import CheckStatus, evaluation_checks, run_doctor
from pitbench.cli.evaluate_config import (
    EvaluationConfig,
    encode_agent_kwargs,
    resolve_config_path,
    resolve_repository_path,
)
from pitbench.cli.profiles import profiles_app
from pitbench.cli.task_image import prepare_task_image
from pitbench.evaluator.evaluator import PitBenchEvaluator
from pitbench.evaluator.storage import ObservationStore
from pitbench.harness.agents import AgentName
from pitbench.harness.cli.harness_cli.runs import LogLevel, _process_agent_kwargs
from pitbench.harness.cli.harness_cli.runs import create as run_harness
from pitbench.harness.evaluation import EvaluationRequest
from pitbench.harness.handlers.trial_handler import TrialHandler
from pitbench.harness.terminal.docker_compose_manager import DockerComposeManager
from pitbench.instances import materialize_instance_set
from pitbench.metrics.performance_report import (
    compute_performance_report,
    format_performance_report,
)
from pitbench.metrics.reliability_report import (
    compute_reliability_reports,
    format_reliability_report,
)
from pitbench.metrics.resource_report import (
    compute_resource_report,
    format_resource_report,
)
from pitbench.schema.task import InstanceSetKind, PitBenchTask
from pitbench.tasks import TaskCatalog, TaskNotFoundError

app = typer.Typer(help="PitBench task, execution harness, and evaluation tooling.")
tasks_app = typer.Typer(help="Validate and smoke-test benchmark tasks.")
app.add_typer(tasks_app, name="tasks")
app.add_typer(profiles_app, name="profiles")
app.add_typer(auth_app, name="auth")

# Direct unified run command: `pitbench run --dataset-path ... --agent ...`
app.command(
    name="run",
    help="Run coding agents on materialized benchmark tasks (e.g. Codex on PyVRP).",
)(run_harness)


def _root(value: Path | None) -> Path:
    return (value or Path.cwd()).resolve()


def _warning(message: str) -> None:
    typer.echo(f"WARNING: {message}", err=True)


def _normalize_task_ids(raw_ids: list[str]) -> list[str]:
    normalized = []
    for item in raw_ids:
        cleaned = item.strip("[](),'\" \t\r\n")
        for sub in cleaned.split(","):
            sub = sub.strip("[](),'\" \t\r\n")
            if sub:
                normalized.append(sub)
    return normalized


def _prepare_task_image(task_path: Path, *, rebuild: bool) -> None:
    prepare_task_image(
        task_path,
        rebuild=rebuild,
        progress=typer.echo,
        trial_handler_type=TrialHandler,
        manager_type=DockerComposeManager,
    )


@app.command("doctor")
def doctor_command(
    task_id: Annotated[
        str, typer.Argument(help="Task ID whose environment should be checked")
    ],
    config_path: Annotated[
        Path | None,
        typer.Option(
            "--config",
            help=(
                "Machine-local evaluation YAML; defaults to "
                "config/evaluate.local.yaml when present"
            ),
        ),
    ] = None,
    root: Annotated[Path | None, typer.Option(help="PitBench repository root")] = None,
) -> None:
    """Check whether this machine is ready for a real evaluation."""
    try:
        checks = run_doctor(task_id, _root(root), config_path)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="TASK_ID") from error

    for check in checks:
        typer.echo(f"{check.status.value} {check.name}: {check.detail}")
        if check.recovery is not None:
            typer.echo(f"     Recovery: {check.recovery}")

    failures = sum(check.status == CheckStatus.FAIL for check in checks)
    warnings = sum(check.status == CheckStatus.WARN for check in checks)
    if failures:
        typer.echo(f"NOT READY: {failures} failure(s), {warnings} warning(s)", err=True)
        raise typer.Exit(code=1)
    typer.echo(
        f"READY: {task_id} evaluation prerequisites passed ({warnings} warning(s))"
    )


@app.command("evaluate")
def evaluate_task(
    task_ids: Annotated[
        list[str], typer.Argument(help="One or more PitBench task IDs")
    ],
    agent: Annotated[
        AgentName,
        typer.Option("--agent", "-a", help="Coding agent to evaluate"),
    ],
    model_name: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model used by the coding agent"),
    ] = None,
    output_path: Annotated[
        Path | None, typer.Option(help="Directory for evaluation results")
    ] = None,
    run_id: Annotated[str | None, typer.Option(help="Unique run identifier")] = None,
    workspace_path: Annotated[
        Path | None,
        typer.Option(help="Directory for persistent materialized tasks"),
    ] = None,
    private_root: Annotated[
        Path | None, typer.Option(help="Evaluator-only private storage")
    ] = None,
    repository_source: Annotated[
        Path | None,
        typer.Option(help="Pre-censored local repository snapshot"),
    ] = None,
    agent_image: Annotated[
        str | None,
        typer.Option(help="Prepared solver image override"),
    ] = None,
    judge_image: Annotated[
        str | None,
        typer.Option(help="Immutable judge image digest or local image ID"),
    ] = None,
    judge_parallel_runs: Annotated[
        int | None,
        typer.Option(
            "--judge-parallel-runs",
            min=1,
            help="Parallel solver runs for the isolated judge",
        ),
    ] = None,
    judge_cpus: Annotated[
        float | None,
        typer.Option(
            "--judge-cpus", min=0.1, help="CPU limit for the isolated judge container"
        ),
    ] = None,
    base_cache: Annotated[
        bool,
        typer.Option(
            "--base-cache/--no-base-cache",
            help="Enable automatic caching and reuse of BASE observations",
        ),
    ] = True,
    base_observations_path: Annotated[
        Path | None,
        typer.Option(
            "--base-observations-path",
            help="Path to precomputed BASE observations parquet file",
        ),
    ] = None,
    n_concurrent: Annotated[
        int, typer.Option("--n-concurrent", min=1, help="Concurrent trials")
    ] = 1,
    n_attempts: Annotated[
        int, typer.Option(min=1, help="Attempts for the selected task")
    ] = 1,
    agent_kwargs: Annotated[
        list[str],
        typer.Option(
            "--agent-kwarg",
            "-k",
            help="Agent option in key=value form; may be repeated",
        ),
    ] = [],
    rebuild: Annotated[
        bool, typer.Option("--rebuild", help="Force rebuilding the cached task image")
    ] = False,
    livestream: Annotated[bool, typer.Option(help="Stream the harness log")] = False,
    config_path: Annotated[
        Path | None,
        typer.Option(
            "--config",
            help=(
                "Machine-local evaluation YAML; defaults to "
                "config/evaluate.local.yaml when present"
            ),
        ),
    ] = None,
    check_only: Annotated[
        bool,
        typer.Option(
            "--check-only", help="Check all prerequisites without running agents"
        ),
    ] = False,
    root: Annotated[Path | None, typer.Option(help="PitBench repository root")] = None,
) -> None:
    """Check, materialize, and evaluate tasks together in one harness run."""
    task_ids = _normalize_task_ids(task_ids)
    if not task_ids:
        raise typer.BadParameter(
            "At least one task ID must be provided.", param_hint="TASK_IDS"
        )
    if len(task_ids) != len(set(task_ids)):
        raise typer.BadParameter("task IDs must be unique", param_hint="TASK_IDS")
    if len(task_ids) > 1 and any(
        value is not None for value in (repository_source, agent_image, judge_image)
    ):
        raise typer.BadParameter(
            "For multiple tasks, configure per-task sources and images in --config."
        )
    repository_root = _root(root)
    resolved_config_path = resolve_config_path(repository_root, config_path)
    evaluation_config = (
        EvaluationConfig.from_yaml(resolved_config_path)
        if resolved_config_path is not None
        else EvaluationConfig()
    )
    if resolved_config_path is not None:
        typer.echo(f"Loaded evaluation config {resolved_config_path}")
    else:
        _warning("evaluation config is absent; using built-in configuration defaults")

    configured_paths = evaluation_config.paths
    if output_path is None and "output_path" not in configured_paths.model_fields_set:
        _warning("output_path is not configured; using default runs")
    if (
        workspace_path is None
        and "workspace_path" not in configured_paths.model_fields_set
    ):
        _warning("workspace_path is not configured; using default .pitbench/tasks")
    if private_root is None and "private_root" not in configured_paths.model_fields_set:
        _warning("private_root is not configured; using default private")
    if "agent_tools" not in evaluation_config.model_fields_set:
        _warning("agent_tools is not configured; using default []")
    resolved_output_path = resolve_repository_path(
        repository_root, output_path or configured_paths.output_path
    )
    resolved_workspace_path = resolve_repository_path(
        repository_root, workspace_path or configured_paths.workspace_path
    )
    resolved_private_root = resolve_repository_path(
        repository_root, private_root or configured_paths.private_root
    )
    resolved_run_id = run_id or datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    if (
        not resolved_run_id
        or Path(resolved_run_id).name != resolved_run_id
        or resolved_run_id in {".", ".."}
    ):
        raise typer.BadParameter(
            "run ID must be a directory name", param_hint="--run-id"
        )
    dataset_path = (resolved_workspace_path / resolved_run_id).resolve()
    configured_agent_values = evaluation_config.kwargs_for(agent.value)
    if agent.value not in evaluation_config.agents:
        _warning(
            f"agent parameters for {agent.value} are not configured; unspecified "
            "settings use agent defaults"
        )
    if configured_agent_values.get("profile_path"):
        configured_agent_values["profile_path"] = str(
            resolve_repository_path(
                repository_root,
                Path(str(configured_agent_values["profile_path"])),
            )
        )
    configured_agent_kwargs = encode_agent_kwargs(configured_agent_values)
    harness_agent_kwargs = [*configured_agent_kwargs, *agent_kwargs]
    try:
        effective_agent_kwargs = _process_agent_kwargs(model_name, harness_agent_kwargs)
    except ValueError as error:
        raise typer.BadParameter(
            "agent options must be key=value", param_hint="--agent-kwarg"
        ) from error

    effective_config = evaluation_config.model_copy(deep=True)
    effective_config.paths.private_root = resolved_private_root
    for task_id in task_ids:
        resources = effective_config.resources_for(task_id).model_copy()
        for key, override in (
            ("repository_source", repository_source),
            ("agent_image", agent_image),
            ("judge_image", judge_image),
            ("judge_cpus", judge_cpus),
            ("judge_parallel_runs", judge_parallel_runs),
            ("base_observations_path", base_observations_path),
        ):
            if override is not None:
                setattr(resources, key, override)
            elif getattr(resources, key) is None and key in (
                "repository_source",
                "agent_image",
                "judge_image",
            ):
                _warning(
                    f"{key} for {task_id} is not configured; using the task config default"
                )
        if resources.repository_source is not None:
            resources.repository_source = resolve_repository_path(
                repository_root, resources.repository_source
            )
        effective_config.tasks[task_id] = resources

    checks = evaluation_checks(
        task_ids, repository_root, effective_config, agent.value, effective_agent_kwargs
    )
    for check in checks:
        typer.echo(f"{check.status.value} {check.name}: {check.detail}")
        if check.recovery:
            typer.echo(f"     Recovery: {check.recovery}")
    if any(check.status == CheckStatus.FAIL for check in checks):
        typer.echo("NOT READY: no agent was started.", err=True)
        raise typer.Exit(1)
    if check_only:
        typer.echo("READY: prerequisite checks passed; no agent was started.")
        return
    if dataset_path.exists() or (resolved_output_path / resolved_run_id).exists():
        raise typer.BadParameter("run ID already exists; choose a new --run-id")
    adapter = PitBenchAdapter(repository_root, resolved_private_root)
    for task_id in task_ids:
        resources = effective_config.resources_for(task_id)
        task_path = dataset_path / task_id
        materialize_kwargs: dict[str, Any] = {
            "repository_source": resources.repository_source,
            "agent_image": resources.agent_image,
            "judge_image": resources.judge_image,
            "agent_tools": evaluation_config.agent_tools,
        }
        judge_cpus_val = resources.judge_cpus or effective_config.judge_cpus
        if judge_cpus_val is not None:
            materialize_kwargs["judge_cpus"] = judge_cpus_val
        judge_mem_val = resources.judge_memory or effective_config.judge_memory
        if judge_mem_val is not None:
            materialize_kwargs["judge_memory"] = judge_mem_val
        judge_par_val = (
            resources.judge_parallel_runs or effective_config.judge_parallel_runs
        )
        if judge_par_val is not None:
            materialize_kwargs["judge_parallel_runs"] = judge_par_val
        if resources.base_observations_path is not None:
            materialize_kwargs["base_observations_path"] = (
                resources.base_observations_path
            )
        if effective_config.paths.base_cache_path != Path(".pitbench/cache/base"):
            materialize_kwargs["base_cache_path"] = (
                effective_config.paths.base_cache_path
            )
        active_base_cache = (
            base_cache if base_cache is not None else effective_config.base_cache
        )
        if not active_base_cache:
            materialize_kwargs["use_base_cache"] = False

        adapter.materialize(
            task_id,
            task_path,
            **materialize_kwargs,
        )
        typer.echo(f"Materialized {task_id} at {task_path}")
        _prepare_task_image(
            task_path, rebuild=rebuild or resources.agent_image is not None
        )
    typer.echo(f"Starting evaluation run {resolved_run_id}")
    run_harness(
        dataset_path=dataset_path,
        dataset_config=None,
        output_path=resolved_output_path,
        run_id=resolved_run_id,
        no_rebuild=True,
        cleanup=False,
        model_name=model_name,
        agent=agent,
        agent_import_path=None,
        agents=None,
        config=None,
        log_level=LogLevel.INFO,
        livestream=livestream,
        n_concurrent_trials=n_concurrent,
        _n_concurrent_trials_compat=None,
        n_attempts=n_attempts,
        agent_kwargs=harness_agent_kwargs,
        global_timeout_multiplier=1.0,
        global_agent_timeout_sec=None,
        global_setup_timeout_sec=None,
        remote_build=False,
        history_limit=None,
    )


@app.command("judge")
def judge_candidate(
    task_id: Annotated[str, typer.Argument(help="PitBench task ID")],
    candidate_patch: Annotated[
        Path, typer.Argument(help="Previously captured candidate.patch")
    ],
    expected_patch_sha256: Annotated[
        str,
        typer.Option(
            "--expected-patch-sha256",
            help="SHA256 recorded when the agent candidate was captured",
        ),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help="Judge artifacts; defaults to the patch's containing directory",
        ),
    ] = None,
    private_root: Annotated[
        Path | None, typer.Option(help="Evaluator-only private storage")
    ] = None,
    repository_source: Annotated[
        Path | None,
        typer.Option(help="Pre-censored base repository snapshot"),
    ] = None,
    judge_image: Annotated[
        str | None,
        typer.Option(help="Immutable judge image digest or local image ID"),
    ] = None,
    judge_cpus: Annotated[
        float, typer.Option(min=0.1, help="CPU limit for the isolated judge")
    ] = 8.0,
    judge_memory: Annotated[
        str, typer.Option(help="Memory limit for the isolated judge")
    ] = "8g",
    judge_parallel_runs: Annotated[
        int | None,
        typer.Option(
            "--judge-parallel-runs",
            min=1,
            help="Parallel solver runs for the isolated judge",
        ),
    ] = None,
    reliability_only: Annotated[
        bool, typer.Option(help="Run only the configured boundary reliability suite")
    ] = False,
    base_cache: Annotated[
        bool | None,
        typer.Option(
            "--base-cache/--no-base-cache",
            help="Enable or disable automatic caching of BASE observations",
        ),
    ] = None,
    base_observations_path: Annotated[
        Path | None,
        typer.Option(
            "--base-observations-path",
            help="Path to precomputed BASE observations parquet file",
        ),
    ] = None,
    config_path: Annotated[
        Path | None,
        typer.Option(
            "--config",
            help=(
                "Machine-local evaluation YAML; defaults to "
                "config/evaluate.local.yaml when present"
            ),
        ),
    ] = None,
    root: Annotated[Path | None, typer.Option(help="PitBench repository root")] = None,
) -> None:
    """Run only the evaluator-owned judge on a previously captured patch."""
    repository_root = _root(root)
    resolved_config_path = resolve_config_path(repository_root, config_path)
    evaluation_config = (
        EvaluationConfig.from_yaml(resolved_config_path)
        if resolved_config_path is not None
        else EvaluationConfig()
    )
    if resolved_config_path is not None:
        typer.echo(f"Loaded evaluation config {resolved_config_path}")
    else:
        _warning("evaluation config is absent; using built-in configuration defaults")

    try:
        record = TaskCatalog(repository_root).validate_one(task_id)
    except TaskNotFoundError as exc:
        raise typer.BadParameter(
            f"Unknown task '{task_id}'", param_hint="TASK_ID"
        ) from exc

    patch_path = candidate_patch.expanduser().resolve()
    if not patch_path.is_file():
        raise typer.BadParameter(
            f"Candidate patch does not exist: {patch_path}",
            param_hint="CANDIDATE_PATCH",
        )
    normalized_patch_sha256 = expected_patch_sha256.lower()
    if len(normalized_patch_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in normalized_patch_sha256
    ):
        raise typer.BadParameter(
            "Expected patch SHA256 must contain exactly 64 hexadecimal characters",
            param_hint="--expected-patch-sha256",
        )

    configured_paths = evaluation_config.paths
    configured_resources = evaluation_config.resources_for(task_id)
    if private_root is None and "private_root" not in configured_paths.model_fields_set:
        _warning("private_root is not configured; using default private")
    resolved_private_root = resolve_repository_path(
        repository_root, private_root or configured_paths.private_root
    )
    configured_repository_source = configured_resources.repository_source
    selected_repository_source = repository_source or configured_repository_source
    if selected_repository_source is None:
        raise typer.BadParameter(
            "Judge requires --repository-source or tasks.<task_id>.repository_source "
            "in the evaluation config"
        )
    resolved_repository_source = resolve_repository_path(
        repository_root, selected_repository_source
    ).resolve()
    if not resolved_repository_source.is_dir():
        raise typer.BadParameter(
            f"Repository snapshot does not exist: {resolved_repository_source}"
        )

    resolved_judge_image = (
        judge_image
        or configured_resources.judge_image
        or record.task.repository.judge_image
    )
    if judge_image is None and configured_resources.judge_image is None:
        _warning(
            f"judge_image for {task_id} is not configured; using the task config "
            "image default"
        )
    if not resolved_judge_image:
        raise typer.BadParameter(
            "Judge requires --judge-image or tasks.<task_id>.judge_image in the "
            "evaluation config"
        )

    resolved_output_dir = (
        resolve_repository_path(repository_root, output_dir).resolve()
        if output_dir is not None
        else patch_path.parent
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    active_base_cache = (
        base_cache if base_cache is not None else evaluation_config.base_cache
    )
    active_base_observations_path = (
        base_observations_path
        if base_observations_path is not None
        else configured_resources.base_observations_path
    )
    judge_evaluator_config: dict[str, Any] = {
        "task_config_path": str(record.task_config_path),
        "base_repository": str(resolved_repository_source),
        "private_root": str(resolved_private_root),
        "judge_image": resolved_judge_image,
        "judge_cpus": judge_cpus,
        "judge_memory": judge_memory,
        "judge_parallel_runs": judge_parallel_runs
        or configured_resources.judge_parallel_runs
        or evaluation_config.judge_parallel_runs,
        "reliability_only": reliability_only,
        "use_base_cache": active_base_cache,
        "base_cache_path": str(evaluation_config.paths.base_cache_path),
        "_progress_callback": typer.echo,
    }
    if active_base_observations_path is not None:
        judge_evaluator_config["base_observations_path"] = str(
            active_base_observations_path
        )

    envelope = PitBenchEvaluator().envelope(
        EvaluationRequest(
            task_id=task_id,
            task_path=repository_root,
            candidate_patch_path=patch_path,
            candidate_patch_sha256=normalized_patch_sha256,
            output_dir=resolved_output_dir,
            agent_name="replay",
            model_name=None,
            evaluator_config=judge_evaluator_config,
        )
    )
    result_path = resolved_output_dir / "evaluation.json"
    result_path.write_text(
        json.dumps(envelope.model_dump(mode="json"), indent=2) + "\n"
    )
    if not envelope.completed:
        typer.echo(f"Judge failed: {envelope.error}", err=True)
        raise typer.Exit(code=1)

    summary = envelope.payload["summary"]
    typer.echo(
        "Judge complete: "
        f"{summary['valid_observation_count']}/{summary['observation_count']} "
        "valid observations"
    )
    typer.echo(f"Artifacts: {resolved_output_dir}")
    typer.echo(
        f"Report: pitbench report {resolved_output_dir} --task-config {record.task_config_path}"
    )


@tasks_app.command("validate")
def validate_tasks(
    root: Annotated[Path | None, typer.Option(help="PitBench repository root")] = None,
) -> None:
    records = TaskCatalog(_root(root)).validate_all()
    for record in records:
        typer.echo(f"OK {record.task.task_id} {record.task_config_sha256[:12]}")
    typer.echo(f"Validated {len(records)} task configs")


@tasks_app.command("fixture")
def fixture_tasks(
    root: Annotated[Path | None, typer.Option(help="PitBench repository root")] = None,
    instances_per_instance_set: Annotated[
        int, typer.Option(min=1, help="Synthetic cases per judge instance set")
    ] = 1,
) -> None:
    """Run the deterministic evaluator contract fixture for every task.

    This command does not build repositories, run solvers, or load private judge data.
    """
    repository_root = _root(root)
    records = TaskCatalog(repository_root).validate_all()
    summaries: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="pitbench-fixture-") as temporary:
        run_root = Path(temporary)
        for record in records:
            output = run_root / record.task.task_id
            output.mkdir(parents=True)
            patch = output / "candidate.patch"
            patch.write_text("")
            envelope = PitBenchEvaluator().envelope(
                EvaluationRequest(
                    task_id=record.task.task_id,
                    task_path=repository_root,
                    candidate_patch_path=patch,
                    output_dir=output,
                    agent_name="fixture",
                    model_name=None,
                    evaluator_config={
                        "task_config_path": str(record.task_config_path),
                        "fixture_mode": True,
                        "fixture_instances_per_instance_set": instances_per_instance_set,
                    },
                )
            )
            if not envelope.completed:
                raise RuntimeError(f"{record.task.task_id}: {envelope.error}")
            summary = envelope.payload["summary"]
            summaries.append({"task_id": record.task.task_id, **summary})
            typer.echo(
                f"OK {record.task.task_id} observations={summary['observation_count']}"
            )
    typer.echo(json.dumps(summaries, indent=2))


@tasks_app.command("materialize-dev")
def materialize_dev(
    output: Annotated[Path, typer.Option(help="Destination directory")],
    root: Annotated[Path | None, typer.Option(help="PitBench repository root")] = None,
) -> None:
    repository_root = _root(root)
    records = TaskCatalog(repository_root).validate_all()
    for record in records:
        instance_set = next(
            item
            for item in record.task.instance_sets
            if item.kind == InstanceSetKind.AGENT_DEV
        )
        paths = materialize_instance_set(
            repository_root / instance_set.instance_set_config,
            output / record.task.task_id,
        )
        typer.echo(f"OK {record.task.task_id} instances={len(paths)}")


@tasks_app.command("materialize")
def materialize_task(
    task_id: Annotated[str, typer.Argument(help="PitBench task ID")],
    output: Annotated[Path, typer.Option(help="Exact task destination")],
    private_root: Annotated[
        Path, typer.Option(help="Evaluator-only private storage")
    ] = Path("private"),
    repository_source: Annotated[
        Path | None,
        typer.Option(help="Pre-censored local repository snapshot"),
    ] = None,
    agent_image: Annotated[
        str | None,
        typer.Option(help="Prepared solver image override"),
    ] = None,
    judge_image: Annotated[
        str | None,
        typer.Option(help="Immutable judge image digest or local image ID"),
    ] = None,
    agent_tools: Annotated[
        list[AgentTool],
        typer.Option(
            "--agent-tool",
            help="Optional PitBench command exposed inside the task; may be repeated",
        ),
    ] = [],
    root: Annotated[Path | None, typer.Option(help="PitBench repository root")] = None,
) -> None:
    repository_root = _root(root)
    task_dir = PitBenchAdapter(
        repository_root,
        private_root if private_root.is_absolute() else repository_root / private_root,
    ).materialize(
        task_id,
        output,
        repository_source=repository_source,
        agent_image=agent_image,
        judge_image=judge_image,
        agent_tools=agent_tools,
    )
    typer.echo(f"Materialized {task_id} at {task_dir}")


@app.command("report")
def report_command(
    path: Annotated[
        Path,
        typer.Argument(
            help="Path to trials.parquet, observations file, or evaluation directory"
        ),
    ],
    task_config: Annotated[
        Path,
        typer.Option(
            "--task-config",
            help="Task config declaring the report's primary budget",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output report as structured JSON"),
    ] = False,
) -> None:
    """Generate the performance-first PitBench evaluation report."""
    target = path.resolve()
    if target.is_dir():
        parquet_file = target / "trials.parquet"
        jsonl_file = target / "observations.jsonl"
        if parquet_file.is_file():
            target = parquet_file
        elif jsonl_file.is_file():
            target = jsonl_file
        else:
            raise typer.BadParameter(
                f"No trials.parquet or observations.jsonl found in directory: {path}"
            )

    if not target.is_file():
        raise typer.BadParameter(f"File does not exist: {target}")

    if target.suffix == ".parquet":
        observations = ObservationStore.read(target)
    elif target.suffix in {".jsonl", ".json"}:
        observations = ObservationStore.read_jsonl(target)
    else:
        observations = ObservationStore.read(target)

    task = PitBenchTask.from_yaml(task_config)
    observation_task_ids = {item.task_id for item in observations}
    if observation_task_ids != {task.task_id}:
        observed = ", ".join(sorted(observation_task_ids)) or "none"
        raise typer.BadParameter(
            f"task ID mismatch: observations={observed}; task-config={task.task_id}",
            param_hint="--task-config",
        )

    standard = [item for item in observations if item.test_suite is None]
    performance_report = (
        compute_performance_report(
            standard,
            primary_budget_sec=task.evaluation.primary_budget_sec,
        )
        if standard
        else None
    )
    resource_report = (
        compute_resource_report(
            standard,
            primary_budget_sec=task.evaluation.primary_budget_sec,
            budgets_sec=task.evaluation.budgets_sec,
            expected_instance_counts={
                item.name: item.size for item in task.instance_sets
            },
            expected_seed_count=(
                task.evaluation.seed_robustness.seed_selection.seed_count
                if task.evaluation.seed_robustness is not None
                else len(task.evaluation.solver_seeds or [])
            ),
        )
        if standard
        else None
    )
    reliability_report = None
    if task.evaluation.operational_reliability:
        reliability_report, _ = compute_reliability_reports(observations, task=task)

    if json_output:
        combined = {
            "performance": performance_report.model_dump()
            if performance_report
            else None,
            "resource_usage": resource_report.model_dump() if resource_report else None,
            "operational_reliability": reliability_report.model_dump()
            if reliability_report
            else None,
        }
        typer.echo(json.dumps(combined, indent=2))
        return

    typer.echo("\n========================================================")
    typer.echo(f"  PitBench Evaluation Report: {target.name}")
    typer.echo(f"  Total observations: {len(observations)}")
    typer.echo("========================================================\n")

    if performance_report is not None:
        typer.echo(format_performance_report(performance_report))
    if resource_report is not None:
        typer.echo("\n" + format_resource_report(resource_report))
    if reliability_report is not None:
        typer.echo("\n" + format_reliability_report(reliability_report))


if __name__ == "__main__":
    app()
