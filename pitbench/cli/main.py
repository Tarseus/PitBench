from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from adapters.pitbench.adapter import PitBenchAdapter
from pitbench.agent_tools import AgentTool
from pitbench.cli.doctor import CheckStatus, run_doctor
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
from pitbench.tasks import TaskCatalog, TaskNotFoundError

app = typer.Typer(help="PitBench task, execution harness, and evaluation tooling.")
tasks_app = typer.Typer(help="Validate and smoke-test benchmark tasks.")
app.add_typer(tasks_app, name="tasks")
app.add_typer(profiles_app, name="profiles")

# Direct unified run command: `pitbench run --dataset-path ... --agent ...`
app.command(
    name="run",
    help="Run coding agents on materialized benchmark tasks (e.g. Codex on PyVRP).",
)(run_harness)


def _root(value: Path | None) -> Path:
    return (value or Path.cwd()).resolve()


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
    profile: Annotated[
        str, typer.Argument(help="Environment profile to diagnose (currently: pyvrp)")
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
        checks = run_doctor(profile, _root(root), config_path)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="PROFILE") from error

    for check in checks:
        typer.echo(f"{check.status.value} {check.name}: {check.detail}")
        if check.recovery is not None:
            typer.echo(f"     Recovery: {check.recovery}")

    failures = sum(check.status == CheckStatus.FAIL for check in checks)
    warnings = sum(check.status == CheckStatus.WARN for check in checks)
    if failures:
        typer.echo(f"NOT READY: {failures} failure(s), {warnings} warning(s)", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"READY: PyVRP evaluation prerequisites passed ({warnings} warning(s))")


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
            agent_tools=evaluation_config.agent_tools,
        )
    except TaskNotFoundError as exc:
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
    configured_agent_values = evaluation_config.kwargs_for(agent.value)
    if configured_agent_values.get("profile_path"):
        configured_agent_values["profile_path"] = str(
            resolve_repository_path(
                repository_root,
                Path(str(configured_agent_values["profile_path"])),
            )
        )
    configured_agent_kwargs = encode_agent_kwargs(configured_agent_values)
    harness_agent_kwargs = [*configured_agent_kwargs, *agent_kwargs]
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
        global_test_timeout_sec=None,
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
    typer.echo(f"Starting isolated judge for {task_id}")
    envelope = PitBenchEvaluator().envelope(
        EvaluationRequest(
            task_id=task_id,
            task_path=repository_root,
            candidate_patch_path=patch_path,
            candidate_patch_sha256=normalized_patch_sha256,
            output_dir=resolved_output_dir,
            agent_name="replay",
            model_name=None,
            evaluator_config={
                "manifest_path": str(record.manifest_path),
                "base_repository": str(resolved_repository_source),
                "private_root": str(resolved_private_root),
                "judge_image": resolved_judge_image,
                "judge_cpus": judge_cpus,
                "judge_memory": judge_memory,
                "_progress_callback": typer.echo,
            },
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
    typer.echo(f"Report: pitbench report {resolved_output_dir}")


@tasks_app.command("validate")
def validate_tasks(
    root: Annotated[Path | None, typer.Option(help="PitBench repository root")] = None,
) -> None:
    records = TaskCatalog(_root(root)).validate_all()
    for record in records:
        typer.echo(f"OK {record.task.task_id} {record.manifest_sha256[:12]}")
    typer.echo(f"Validated {len(records)} task manifests")


@tasks_app.command("fixture")
def fixture_tasks(
    root: Annotated[Path | None, typer.Option(help="PitBench repository root")] = None,
    instances_per_population: Annotated[
        int, typer.Option(min=1, help="Synthetic cases per judge population")
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
