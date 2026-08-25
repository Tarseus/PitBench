from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from adapters.pitbench.adapter import (
    IMAGE_REVISION,
    IMAGE_REVISION_LABEL,
    IMAGE_SOURCE_LABEL,
    PitBenchAdapter,
)
from pitbench.cli.evaluate_config import (
    EvaluationConfig,
    encode_agent_kwargs,
    resolve_config_path,
    resolve_repository_path,
)
from pitbench.evaluator.evaluator import PitBenchEvaluator
from pitbench.evaluator.storage import ObservationStore
from pitbench.harness.agents import AgentName
from pitbench.harness.cli.harness_cli.runs import LogLevel
from pitbench.harness.cli.harness_cli.runs import create as run_harness
from pitbench.harness.evaluation import EvaluationRequest
from pitbench.harness.handlers.trial_handler import TrialHandler
from pitbench.harness.terminal.docker_compose_manager import DockerComposeManager
from pitbench.instances import materialize_population
from pitbench.metrics.performance_report import (
    compute_performance_report,
    format_performance_report,
)
from pitbench.schema.task import PopulationKind
from pitbench.tasks import TaskCatalog

app = typer.Typer(help="PitBench task, execution harness, and evaluation tooling.")
tasks_app = typer.Typer(help="Validate and smoke-test benchmark tasks.")
app.add_typer(tasks_app, name="tasks")

# Direct unified run command: `pitbench run --dataset-path ... --agent ...`
app.command(
    name="run",
    help="Run coding agents on materialized benchmark tasks (e.g. Codex on PyVRP).",
)(run_harness)


def _root(value: Path | None) -> Path:
    return (value or Path.cwd()).resolve()


def _prepare_task_image(task_path: Path, *, rebuild: bool) -> None:
    trial = TrialHandler(trial_name="pitbench-image-prepare", input_path=task_path)
    manager = DockerComposeManager(
        client_container_name=trial.client_container_name,
        client_image_name=trial.client_image_name,
        docker_image_name_prefix=trial.docker_image_name_prefix,
        docker_compose_path=trial.task_paths.docker_compose_path,
        sessions_logs_path=task_path / ".image-prepare/sessions",
        agent_logs_path=task_path / ".image-prepare/agent-logs",
    )
    expected_source = (
        (task_path / "Dockerfile").read_text().splitlines()[0].removeprefix("FROM ")
    )
    cached_revision = manager.image_label(IMAGE_REVISION_LABEL)
    cached_source = manager.image_label(IMAGE_SOURCE_LABEL)
    if rebuild or cached_revision != IMAGE_REVISION or cached_source != expected_source:
        typer.echo(f"Building task image {trial.client_image_name}")
        manager.build()
    else:
        typer.echo(f"Reusing task image {trial.client_image_name}")


@app.command("evaluate")
def evaluate_task(
    task_id: Annotated[str, typer.Argument(help="PitBench task ID")],
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
    root: Annotated[Path | None, typer.Option(help="PitBench repository root")] = None,
) -> None:
    """Materialize one PitBench task and immediately evaluate an agent on it."""
    repository_root = _root(root)
    resolved_config_path = resolve_config_path(repository_root, config_path)
    evaluation_config = (
        EvaluationConfig.from_yaml(resolved_config_path)
        if resolved_config_path is not None
        else EvaluationConfig()
    )
    if resolved_config_path is not None:
        typer.echo(f"Loaded evaluation config {resolved_config_path}")

    configured_paths = evaluation_config.paths
    configured_resources = evaluation_config.resources_for(task_id)
    resolved_output_path = resolve_repository_path(
        repository_root, output_path or configured_paths.output_path
    )
    resolved_workspace_path = resolve_repository_path(
        repository_root, workspace_path or configured_paths.workspace_path
    )
    resolved_private_root = resolve_repository_path(
        repository_root, private_root or configured_paths.private_root
    )
    configured_repository_source = (
        resolve_repository_path(repository_root, configured_resources.repository_source)
        if configured_resources.repository_source is not None
        else None
    )
    resolved_repository_source = repository_source or configured_repository_source
    if resolved_repository_source is not None:
        resolved_repository_source = resolve_repository_path(
            repository_root, resolved_repository_source
        )
    resolved_agent_image = agent_image or configured_resources.agent_image
    resolved_judge_image = judge_image or configured_resources.judge_image

    resolved_run_id = run_id or datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    dataset_path = (resolved_workspace_path / resolved_run_id).resolve()
    task_path = dataset_path / task_id

    try:
        PitBenchAdapter(
            repository_root,
            resolved_private_root,
        ).materialize(
            task_id,
            task_path,
            repository_source=resolved_repository_source,
            agent_image=resolved_agent_image,
            judge_image=resolved_judge_image,
        )
    except StopIteration as exc:
        available = ", ".join(
            record.task.task_id
            for record in TaskCatalog(repository_root).validate_all()
        )
        raise typer.BadParameter(
            f"Unknown task '{task_id}'. Available tasks: {available}",
            param_hint="TASK_ID",
        ) from exc

    typer.echo(f"Materialized {task_id} at {task_path}")
    _prepare_task_image(task_path, rebuild=rebuild or resolved_agent_image is not None)
    typer.echo(f"Starting evaluation run {resolved_run_id}")
    configured_agent_kwargs = encode_agent_kwargs(
        evaluation_config.kwargs_for(agent.value)
    )
    harness_agent_kwargs = [
        "no_rebuild=False",
        *configured_agent_kwargs,
        *agent_kwargs,
    ]
    run_harness(
        dataset=None,
        _dataset_name_compat=None,
        _dataset_version_compat=None,
        dataset_path=dataset_path,
        dataset_config=None,
        registry_url=None,
        local_registry_path=None,
        output_path=resolved_output_path,
        run_id=resolved_run_id,
        upload_results=False,
        task_ids=None,
        n_tasks=None,
        exclude_task_ids=None,
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
        global_test_timeout_sec=None,
        global_setup_timeout_sec=None,
        remote_build=False,
        history_limit=None,
    )


@tasks_app.command("validate")
def validate_tasks(
    root: Annotated[Path | None, typer.Option(help="PitBench repository root")] = None,
) -> None:
    records = TaskCatalog(_root(root)).validate_all()
    for record in records:
        typer.echo(f"OK {record.task.task_id} {record.manifest_sha256[:12]}")
    typer.echo(f"Validated {len(records)} task manifests")


@tasks_app.command("smoke")
def smoke_tasks(
    root: Annotated[Path | None, typer.Option(help="PitBench repository root")] = None,
    instances_per_population: Annotated[
        int, typer.Option(min=1, help="Synthetic cases per hidden population")
    ] = 1,
) -> None:
    """Run the explicit deterministic contract fixture for every task.

    This command never claims to execute a solver or hidden judge. Production smoke
    tests use ``--real`` once private assets and judge images are provisioned.
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
                        "manifest_path": str(record.manifest_path),
                        "fixture_mode": True,
                        "fixture_instances_per_population": instances_per_population,
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
        population = next(
            item
            for item in record.task.populations
            if item.kind == PopulationKind.AGENT_DEV
        )
        paths = materialize_population(
            repository_root / population.manifest,
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

    performance_report = compute_performance_report(observations)

    if json_output:
        combined = {"performance": performance_report.model_dump()}
        typer.echo(json.dumps(combined, indent=2))
        return

    typer.echo("\n========================================================")
    typer.echo(f"  PitBench Evaluation Report: {target.name}")
    typer.echo(f"  Total observations: {len(observations)}")
    typer.echo("========================================================\n")

    typer.echo(format_performance_report(performance_report))


if __name__ == "__main__":
    app()
