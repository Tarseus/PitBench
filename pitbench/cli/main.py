from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer

from adapters.pitbench.adapter import PitBenchAdapter
from adapters.pitbench.git_snapshot import GitSnapshotError
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
from pitbench.cli.solver_images import build_solver_image
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
from pitbench.metrics.controlled_transfer_report import (
    compute_controlled_transfer_report,
    format_controlled_transfer_report,
)
from pitbench.metrics.objective_report import (
    compute_objective_report,
    format_objective_report,
)
from pitbench.metrics.performance_report import (
    PerformanceClassification,
    PerformanceConfirmation,
    adjudicate_performance_confirmation,
    compute_task_performance_report,
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
from pitbench.schema.observation import (
    ExpectedRun,
    ExpectedRunGrid,
    RunObservation,
    RunStatus,
)
from pitbench.schema.task import InstanceSetKind, PerformanceProtocol, PitBenchTask
from pitbench.tasks import TaskCatalog, TaskNotFoundError

app = typer.Typer(help="PitBench task, execution harness, and evaluation tooling.")
tasks_app = typer.Typer(help="Validate and smoke-test benchmark tasks.")
images_app = typer.Typer(help="Build pinned source images for solver tasks.")
app.add_typer(tasks_app, name="tasks")
app.add_typer(images_app, name="images")
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


def _run_configuration_robustness(
    repository_root: Path,
    evaluation_config: EvaluationConfig,
    output_root: Path,
    run_id: str,
    *,
    instances_per_panel: int | None = None,
    seeds_per_panel: int | None = None,
) -> Path | None:
    configuration = evaluation_config.configuration_robustness
    if not configuration.enabled:
        return None
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError(
            "Configuration Robustness is enabled by default but uv is unavailable"
        )
    experiment_config = resolve_repository_path(
        repository_root, configuration.experiment_config
    )
    output = resolve_repository_path(
        repository_root, output_root / run_id / configuration.output_path
    )
    cpus = configuration.cpus or sorted(os.sched_getaffinity(0))
    if not cpus:
        raise RuntimeError("Configuration Robustness requires at least one CPU")
    prepare = [
        uv,
        "run",
        "--extra",
        "configuration",
        "python",
        "-m",
        "scripts.collect_configuration_results",
        "prepare",
        "--config",
        str(experiment_config),
        "--output",
        str(output),
        "--cpus",
        *[str(cpu) for cpu in cpus],
        "--search-seed",
        str(configuration.search_seed),
    ]
    if instances_per_panel is not None:
        prepare.extend(["--instances-per-panel", str(instances_per_panel)])
    if seeds_per_panel is not None:
        prepare.extend(["--seeds-per-panel", str(seeds_per_panel)])
    subprocess.run(prepare, cwd=repository_root, check=True)
    run = [
        uv,
        "run",
        "--extra",
        "configuration",
        "python",
        "-m",
        "scripts.collect_configuration_results",
        "run",
        "--output",
        str(output),
    ]
    subprocess.run(run, cwd=repository_root, check=True)
    return output


def _run_identity(item: ExpectedRun | RunObservation) -> tuple:
    return (
        item.task_id,
        item.code_state.value,
        item.instance_set,
        item.instance_set_kind,
        item.instance_id,
        item.solver_seed,
        item.budget_sec,
        item.equivalence_parent_id,
        item.test_suite,
    )


def _limited_evaluation_failures(
    task: PitBenchTask,
    output_dir: Path,
    payload: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    expected_grid_path = output_dir / "expected-run-grid.json"
    observations_path = output_dir / "judge-observations.jsonl"
    if not expected_grid_path.is_file():
        return ["expected run grid is missing"]
    if not observations_path.is_file():
        return ["judge observations are missing"]

    expected_grid = ExpectedRunGrid.model_validate_json(expected_grid_path.read_text())
    observations = ObservationStore.read_jsonl(observations_path)
    expected_runs = {_run_identity(item) for item in expected_grid.runs}
    observed_runs: set[tuple] = set()
    duplicate_runs = 0
    for observation in observations:
        identity = _run_identity(observation)
        if identity in observed_runs:
            duplicate_runs += 1
        observed_runs.add(identity)
    missing_runs = expected_runs - observed_runs
    unexpected_runs = observed_runs - expected_runs
    if missing_runs:
        failures.append(f"{len(missing_runs)} expected solver runs are missing")
    if unexpected_runs:
        failures.append(f"{len(unexpected_runs)} unexpected solver runs were recorded")
    if duplicate_runs:
        failures.append(f"{duplicate_runs} duplicate solver runs were recorded")

    infrastructure_errors = [
        observation
        for observation in observations
        if observation.status is RunStatus.INFRASTRUCTURE_ERROR
    ]
    if infrastructure_errors:
        failures.append(
            f"{len(infrastructure_errors)} infrastructure-error observations: "
            f"{infrastructure_errors[0].error or 'no detail'}"
        )

    summary = payload.get("summary", {})
    performance = summary.get("performance")
    if performance is None:
        failures.append("performance report is missing")
    elif performance.get("classification") == "incomplete":
        failures.append("performance report is incomplete")
    if summary.get("resource_usage") is None:
        failures.append("resource report is missing")
    if (
        task.evaluation.seed_robustness is not None
        and summary.get("nuisance_robustness") is None
    ):
        failures.append("seed robustness report is missing")
    if task.evaluation.operational_reliability:
        reliability = summary.get("operational_reliability")
        if reliability is None:
            failures.append("operational reliability report is missing")
        elif not reliability.get("complete"):
            failures.append("operational reliability report is incomplete")
    if task.evaluation.representation_robustness is not None:
        representation = payload.get("artifacts", {}).get(
            "representation_robustness_details"
        )
        if representation is None:
            failures.append("representation robustness report is missing")
        else:
            details_path = output_dir / representation["path"]
            if not details_path.is_file():
                failures.append("representation robustness details are missing")
            else:
                details = json.loads(details_path.read_text())
                if details.get("completed_run_count") != details.get(
                    "expected_run_count"
                ):
                    failures.append("representation robustness report is incomplete")
    return failures


def _configuration_dry_run_failures(output_dir: Path) -> list[str]:
    manifest_path = output_dir / "experiment.json"
    report_path = output_dir / "configuration_report.json"
    if not manifest_path.is_file():
        return ["configuration experiment manifest is missing"]
    if not report_path.is_file():
        return ["configuration report is missing"]
    manifest = json.loads(manifest_path.read_text())
    report = json.loads(report_path.read_text())
    report_statuses = {
        item["id"]: item.get("status") for item in report.get("searches", [])
    }
    failures = []
    for search in manifest.get("searches", []):
        status = report_statuses.get(search["id"])
        if status != "completed":
            failures.append(
                f"configuration search {search['id']} is {status or 'missing'}"
            )
    return failures


def _configuration_smoke_failures(output_dir: Path) -> list[str]:
    """Check that every small search reached a terminal state."""
    manifest_path = output_dir / "experiment.json"
    report_path = output_dir / "configuration_report.json"
    if not manifest_path.is_file() or not report_path.is_file():
        return ["configuration smoke artifacts are missing"]
    manifest = json.loads(manifest_path.read_text())
    report = json.loads(report_path.read_text())
    statuses = {item["id"]: item.get("status") for item in report.get("searches", [])}
    terminal = {
        "completed",
        "baseline_unavailable",
        "stopped_after_unavailable_feedback",
        "no_candidate",
    }
    return [
        f"configuration search {item['id']} is {statuses.get(item['id']) or 'missing'}"
        for item in manifest.get("searches", [])
        if statuses.get(item["id"]) not in terminal
    ]


def _evaluation_metric_failures(
    task: PitBenchTask,
    output_dir: Path,
    payload: dict[str, Any],
    *,
    smoke_test: bool = False,
) -> list[str]:
    """Validate normal-evaluation metric coverage from final stored observations."""
    artifacts = payload.get("artifacts", {})
    observations_ref = artifacts.get("observations")
    expected_grid_ref = artifacts.get("expected_run_grid")
    if not isinstance(observations_ref, dict) or not observations_ref.get("path"):
        return ["final observations artifact is missing"]
    if not isinstance(expected_grid_ref, dict) or not expected_grid_ref.get("path"):
        return ["expected run grid artifact is missing"]
    observations_path = output_dir / observations_ref["path"]
    expected_grid_path = output_dir / expected_grid_ref["path"]
    if not observations_path.is_file():
        return ["final observations artifact is missing"]
    if not expected_grid_path.is_file():
        return ["expected run grid artifact is missing"]

    failures: list[str] = []
    expected_grid = ExpectedRunGrid.model_validate_json(expected_grid_path.read_text())
    observations = ObservationStore.read(observations_path)
    expected_runs = {_run_identity(item) for item in expected_grid.runs}
    observed_runs = {_run_identity(item) for item in observations}
    if missing_runs := expected_runs - observed_runs:
        failures.append(f"{len(missing_runs)} expected solver runs are missing")
    if unexpected_runs := observed_runs - expected_runs:
        failures.append(f"{len(unexpected_runs)} unexpected solver runs were recorded")
    if len(observed_runs) != len(observations):
        failures.append("duplicate solver runs were recorded")

    if smoke_test:
        infrastructure_errors = sum(
            item.status is RunStatus.INFRASTRUCTURE_ERROR for item in observations
        )
        if infrastructure_errors:
            failures.append(
                f"{infrastructure_errors} infrastructure-error observations"
            )
        if payload.get("summary", {}).get("observation_count") != len(observations):
            failures.append("observation summary count differs from stored results")
        return failures

    summary = payload.get("summary", {})
    performance = summary.get("performance")
    if performance is None:
        failures.append("performance report is missing")
    elif performance.get("classification") == "incomplete":
        failures.append("performance report is incomplete")
    if summary.get("resource_usage") is None:
        failures.append("resource report is missing")
    if (
        task.evaluation.seed_robustness is not None
        and summary.get("nuisance_robustness") is None
    ):
        failures.append("seed robustness report is missing")
    if task.evaluation.operational_reliability:
        reliability = summary.get("operational_reliability")
        if reliability is None:
            failures.append("operational reliability report is missing")
        elif not reliability.get("complete"):
            failures.append("operational reliability report is incomplete")
    if task.evaluation.representation_robustness is not None:
        representation = artifacts.get("representation_robustness_details")
        if not isinstance(representation, dict) or not representation.get("path"):
            failures.append("representation robustness report is missing")
        else:
            details_path = output_dir / representation["path"]
            if not details_path.is_file():
                failures.append("representation robustness details are missing")
            else:
                details = json.loads(details_path.read_text())
                if details.get("completed_run_count") != details.get(
                    "expected_run_count"
                ):
                    failures.append("representation robustness report is incomplete")
    return failures


def _normal_metric_coverage(
    repository_root: Path,
    output_root: Path,
    run_id: str,
    task_ids: list[str],
    configuration_output: Path | None,
    configuration_error: str | None = None,
    *,
    smoke_test: bool = False,
) -> dict[str, Any]:
    """Return one auditable metric coverage record for a normal evaluate run."""
    run_output = output_root / run_id
    results_path = run_output / "results.json"
    coverage: dict[str, Any] = {
        "run_id": run_id,
        "tasks": [],
        "configuration_robustness": None,
        "complete": False,
        "smoke_test": smoke_test,
    }
    all_failures: list[str] = []
    if not results_path.is_file():
        all_failures.append("harness results are missing")
    else:
        result_records = json.loads(results_path.read_text()).get("results", [])
        catalog = TaskCatalog(repository_root)
        for task_id in task_ids:
            task = catalog.validate_one(task_id).task
            trials = [item for item in result_records if item.get("task_id") == task_id]
            if not trials:
                failures = ["task has no completed trial result"]
                coverage["tasks"].append({"task_id": task_id, "failures": failures})
                all_failures.extend(f"{task_id}: {failure}" for failure in failures)
                continue
            for trial in trials:
                evaluation = trial.get("evaluation")
                failures = []
                if not isinstance(evaluation, dict) or not evaluation.get("completed"):
                    failures.append(
                        "evaluator did not complete: "
                        + str((evaluation or {}).get("error", "missing evaluation"))
                    )
                else:
                    trial_output = (
                        run_output / task_id / trial["trial_name"] / "evaluation"
                    )
                    failures.extend(
                        _evaluation_metric_failures(
                            task,
                            trial_output,
                            evaluation.get("payload", {}),
                            smoke_test=smoke_test,
                        )
                    )
                coverage["tasks"].append(
                    {
                        "task_id": task_id,
                        "trial_name": trial.get("trial_name"),
                        "failures": failures,
                    }
                )
                all_failures.extend(f"{task_id}: {failure}" for failure in failures)
    if configuration_output is not None or configuration_error is not None:
        configuration_failures = []
        if configuration_output is not None:
            try:
                configuration_failures = (
                    _configuration_smoke_failures(configuration_output)
                    if smoke_test
                    else _configuration_dry_run_failures(configuration_output)
                )
            except (OSError, ValueError, KeyError, TypeError) as error:
                configuration_failures = [
                    f"configuration artifacts could not be read: {type(error).__name__}: {error}"
                ]
        if configuration_error is not None:
            configuration_failures.append(configuration_error)
        coverage["configuration_robustness"] = {
            "output": str(configuration_output) if configuration_output else None,
            "failures": configuration_failures,
        }
        all_failures.extend(
            f"Configuration Robustness: {failure}" for failure in configuration_failures
        )
    coverage["complete"] = not all_failures
    coverage["failures"] = all_failures
    return coverage


@images_app.command("build")
def build_images(
    task_ids: Annotated[
        list[str],
        typer.Argument(help="One or more task IDs whose source images to build"),
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
    image_tag: Annotated[
        str | None,
        typer.Option(
            "--tag",
            help="Local image tag; allowed only when building one task",
        ),
    ] = None,
    root: Annotated[Path | None, typer.Option(help="PitBench repository root")] = None,
) -> None:
    """Build fixed source images, optionally using local proxy settings."""
    task_ids = _normalize_task_ids(task_ids)
    if not task_ids:
        raise typer.BadParameter(
            "At least one task ID is required", param_hint="TASK_IDS"
        )
    if len(task_ids) != len(set(task_ids)):
        raise typer.BadParameter("task IDs must be unique", param_hint="TASK_IDS")
    if image_tag is not None and len(task_ids) != 1:
        raise typer.BadParameter("--tag requires exactly one task ID")

    repository_root = _root(root)
    resolved_config_path = resolve_config_path(repository_root, config_path)
    config = (
        EvaluationConfig.from_yaml(resolved_config_path)
        if resolved_config_path is not None
        else EvaluationConfig()
    )
    for task_id in task_ids:
        built = build_solver_image(
            task_id,
            repository_root,
            config=config.solver_image_build,
            image_tag=image_tag,
            progress=typer.echo,
        )
        typer.echo(f"Built {built.task_id}: {built.image_tag} ({built.image_id})")


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
        bool | None,
        typer.Option(
            "--base-cache/--no-base-cache",
            help="Enable automatic caching and reuse of BASE observations",
        ),
    ] = None,
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
    research_track: Annotated[
        str | None,
        typer.Option(
            "--research-track", help="Agent wall-budget track: lite, standard, research"
        ),
    ] = None,
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
    configuration_robustness: Annotated[
        bool,
        typer.Option(
            "--configuration-robustness/--no-configuration-robustness",
            help="Run the default release-only configuration search",
        ),
    ] = True,
    smoke_test: Annotated[
        bool,
        typer.Option(
            "--smoke-test",
            help=(
                "Run one instance per set and at most two seeds per case, with a "
                "five-minute Agent limit; results are not formal metric evidence"
            ),
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
        value is not None for value in (agent_image, judge_image)
    ):
        raise typer.BadParameter(
            "For multiple tasks, configure per-task images in --config."
        )
    repository_root = _root(root)
    resolved_config_path = resolve_config_path(repository_root, config_path)
    evaluation_config = (
        EvaluationConfig.from_yaml(resolved_config_path)
        if resolved_config_path is not None
        else EvaluationConfig()
    )
    if research_track is not None:
        if research_track not in {"lite", "standard", "research"}:
            raise typer.BadParameter(
                "expected lite, standard or research", param_hint="--research-track"
            )
        evaluation_config.research_track = research_track
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
        _warning(
            "agent_tools is not configured; using default [inspect, workspace, bench]"
        )
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
    if not configuration_robustness:
        effective_config.configuration_robustness.enabled = False
    for task_id in task_ids:
        resources = effective_config.resources_for(task_id).model_copy()
        for key, override in (
            ("agent_image", agent_image),
            ("judge_image", judge_image),
            ("judge_cpus", judge_cpus),
            ("judge_parallel_runs", judge_parallel_runs),
            ("base_observations_path", base_observations_path),
        ):
            if override is not None:
                setattr(resources, key, override)
            elif getattr(resources, key) is None and key in (
                "agent_image",
                "judge_image",
            ):
                _warning(
                    f"{key} for {task_id} is not configured; using the task config default"
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
            "agent_image": resources.agent_image,
            "judge_image": resources.judge_image,
            "agent_tools": evaluation_config.agent_tools,
            "agent_feedback_mode": evaluation_config.agent_feedback_mode,
        }
        if effective_config.research_track is not None:
            materialize_kwargs["research_track"] = effective_config.research_track
        elif (
            resources.research_budget is not None
            or effective_config.research_budget is not None
        ):
            materialize_kwargs["research_budget"] = (
                resources.research_budget or effective_config.research_budget
            )
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
        if not active_base_cache or smoke_test:
            materialize_kwargs["use_base_cache"] = False
        if smoke_test:
            materialize_kwargs["smoke_test"] = True

        adapter.materialize(
            task_id,
            task_path,
            **materialize_kwargs,
        )
        typer.echo(f"Materialized {task_id} at {task_path}")
        _prepare_task_image(task_path, rebuild=rebuild)
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
    configuration_output = None
    configuration_error = None
    if configuration_robustness and effective_config.configuration_robustness.enabled:
        configuration_output = resolve_repository_path(
            repository_root,
            resolved_output_path
            / resolved_run_id
            / effective_config.configuration_robustness.output_path,
        )
        try:
            _run_configuration_robustness(
                repository_root,
                effective_config,
                resolved_output_path,
                resolved_run_id,
                instances_per_panel=1 if smoke_test else None,
                seeds_per_panel=2 if smoke_test else None,
            )
        except Exception as error:
            configuration_error = (
                f"configuration collection failed: {type(error).__name__}: {error}"
            )
    coverage = _normal_metric_coverage(
        repository_root,
        resolved_output_path,
        resolved_run_id,
        task_ids,
        configuration_output,
        configuration_error,
        smoke_test=smoke_test,
    )
    coverage_path = resolved_output_path / resolved_run_id / "metric-coverage.json"
    coverage_path.write_text(json.dumps(coverage, indent=2) + "\n")
    typer.echo(f"Metric coverage: {coverage_path}")
    if not coverage["complete"]:
        for failure in coverage["failures"]:
            typer.echo(f"FAIL metric coverage: {failure}", err=True)
        raise typer.Exit(1)


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
    judge_image: Annotated[
        str | None,
        typer.Option(help="Immutable judge image digest or local image ID"),
    ] = None,
    judge_cpus: Annotated[
        float | None,
        typer.Option(min=0.1, help="CPU limit for the isolated judge"),
    ] = None,
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
    try:
        cached_repository = PitBenchAdapter(
            repository_root, resolved_private_root
        ).ensure_source_cache(record.task)
    except (GitSnapshotError, OSError, ValueError) as error:
        raise typer.BadParameter(
            f"Could not prepare the pinned repository source: {error}",
            param_hint="TASK_ID",
        ) from error

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
        "base_repository": str(cached_repository),
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
    confirmation = (summary.get("performance") or {}).get("confirmation")
    if confirmation is not None:
        typer.echo(
            "Performance confirmation: "
            f"{confirmation['repeat_valid_observation_count']}/"
            f"{confirmation['repeat_observation_count']} valid observations; "
            f"final={summary['performance']['classification']}"
        )
    typer.echo(f"Artifacts: {resolved_output_dir}")
    typer.echo(
        f"Report: pitbench report {resolved_output_dir} --task-config {record.task_config_path}"
    )


@app.command("dry-run")
def dry_run_tasks(
    task_ids: Annotated[
        list[str], typer.Argument(help="One or more PitBench task IDs")
    ],
    run_id: Annotated[str, typer.Option(help="Unique dry-run identifier")],
    instances_per_instance_set: Annotated[
        int,
        typer.Option(
            "--instances-per-instance-set",
            min=1,
            help="Real instances to run from each declared instance set",
        ),
    ] = 1,
    private_root: Annotated[
        Path | None, typer.Option(help="Evaluator-only private storage")
    ] = None,
    judge_parallel_runs: Annotated[
        int, typer.Option("--judge-parallel-runs", min=1)
    ] = 1,
    judge_cpus: Annotated[float, typer.Option("--judge-cpus", min=0.1)] = 8.0,
    judge_memory: Annotated[str, typer.Option("--judge-memory")] = "8g",
    configuration_robustness: Annotated[
        bool,
        typer.Option(
            "--configuration-robustness/--no-configuration-robustness",
            help="Run the release-only configuration search on one instance per panel",
        ),
    ] = True,
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
    """Run the full evaluator pipeline on a limited empty-patch panel without an agent."""
    task_ids = _normalize_task_ids(task_ids)
    if not task_ids:
        raise typer.BadParameter(
            "at least one task ID is required", param_hint="TASK_IDS"
        )
    if len(task_ids) != len(set(task_ids)):
        raise typer.BadParameter("task IDs must be unique", param_hint="TASK_IDS")
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise typer.BadParameter(
            "run ID must be a directory name", param_hint="--run-id"
        )

    repository_root = _root(root)
    resolved_config_path = resolve_config_path(repository_root, config_path)
    evaluation_config = (
        EvaluationConfig.from_yaml(resolved_config_path)
        if resolved_config_path is not None
        else EvaluationConfig()
    )
    resolved_private_root = resolve_repository_path(
        repository_root,
        private_root or evaluation_config.paths.private_root,
    )
    workspace_root = resolve_repository_path(
        repository_root,
        evaluation_config.paths.workspace_path / f"dry-run-{run_id}",
    )
    output_root = resolve_repository_path(
        repository_root,
        evaluation_config.paths.output_path / run_id / "dry-run",
    )
    if workspace_root.exists() or output_root.exists():
        raise typer.BadParameter("dry-run identifier already exists; choose another")

    adapter = PitBenchAdapter(repository_root, resolved_private_root)
    failures = 0
    empty_patch_sha256 = hashlib.sha256(b"").hexdigest()
    for task_id in task_ids:
        resources = evaluation_config.resources_for(task_id)
        task_workspace = workspace_root / task_id
        task_output = output_root / task_id
        task_output.mkdir(parents=True, exist_ok=True)
        patch_path = task_output / "candidate.patch"
        patch_path.write_text("")
        try:
            adapter.materialize(
                task_id,
                task_workspace,
                agent_image=resources.agent_image,
                judge_image=resources.judge_image,
                use_base_cache=False,
            )
            _prepare_task_image(task_workspace, rebuild=True)
            record = TaskCatalog(repository_root).validate_one(task_id)
            judge_image = resources.judge_image or record.task.repository.judge_image
            if not judge_image:
                raise ValueError("dry-run requires a pinned judge image")
            envelope = PitBenchEvaluator().envelope(
                EvaluationRequest(
                    task_id=task_id,
                    task_path=repository_root,
                    candidate_patch_path=patch_path,
                    candidate_patch_sha256=empty_patch_sha256,
                    output_dir=task_output,
                    agent_name="dry-run",
                    model_name=None,
                    evaluator_config={
                        "task_config_path": str(record.task_config_path),
                        "base_repository": str(task_workspace / "repo"),
                        "private_root": str(resolved_private_root),
                        "judge_image": judge_image,
                        "judge_cpus": judge_cpus,
                        "judge_memory": judge_memory,
                        "judge_parallel_runs": judge_parallel_runs,
                        "use_base_cache": False,
                        "instances_per_instance_set": instances_per_instance_set,
                    },
                )
            )
            (task_output / "evaluation.json").write_text(
                json.dumps(envelope.model_dump(mode="json"), indent=2) + "\n"
            )
            if envelope.completed:
                preflight_failures = _limited_evaluation_failures(
                    record.task,
                    task_output,
                    envelope.payload,
                )
                if preflight_failures:
                    failures += 1
                    typer.echo(
                        f"FAIL {task_id}: " + "; ".join(preflight_failures),
                        err=True,
                    )
                else:
                    typer.echo(
                        f"PASS {task_id}: empty-patch limited-panel evaluation completed"
                    )
            else:
                failures += 1
                typer.echo(
                    f"FAIL {task_id}: {envelope.error or 'judge failed'}",
                    err=True,
                )
        except Exception as error:
            failures += 1
            typer.echo(f"FAIL {task_id}: {type(error).__name__}: {error}", err=True)
    if configuration_robustness:
        try:
            configuration_output = _run_configuration_robustness(
                repository_root,
                evaluation_config,
                evaluation_config.paths.output_path / run_id,
                "dry-run",
                instances_per_panel=instances_per_instance_set,
            )
            if configuration_output is not None:
                configuration_failures = _configuration_dry_run_failures(
                    configuration_output
                )
                if configuration_failures:
                    failures += 1
                    typer.echo(
                        "FAIL Configuration Robustness: "
                        + "; ".join(configuration_failures),
                        err=True,
                    )
        except Exception as error:
            failures += 1
            typer.echo(
                f"FAIL Configuration Robustness: {type(error).__name__}: {error}",
                err=True,
            )
    typer.echo(f"Dry-run artifacts: {output_root}")
    if failures:
        raise typer.Exit(code=1)


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
    expected_run_grid_path = target.parent / "expected-run-grid.json"
    expected_run_grid = (
        ExpectedRunGrid.model_validate_json(expected_run_grid_path.read_text())
        if expected_run_grid_path.is_file()
        else None
    )
    if (
        task.evaluation.performance_protocol == PerformanceProtocol.EXACT_VERIFIED_SOLVE
        and standard
        and expected_run_grid is None
    ):
        raise typer.BadParameter(
            "exact performance requires sibling expected-run-grid.json",
            param_hint=str(target),
        )
    if task.evaluation.controlled_transfer is not None and expected_run_grid is None:
        raise typer.BadParameter(
            "controlled transfer requires sibling expected-run-grid.json",
            param_hint=str(target),
        )
    performance_report = (
        compute_task_performance_report(
            standard,
            task=task,
            expected_run_grid=expected_run_grid,
        )
        if standard
        else None
    )
    initial_checkpoint_path = target.parent / "judge-checkpoint.json"
    if performance_report is not None and initial_checkpoint_path.is_file():
        initial_identity = json.loads(initial_checkpoint_path.read_text())["identity"]
        if set(initial_identity["code_states"]) != {
            "base",
            "agent",
        } and performance_report.classification in {
            PerformanceClassification.IMPROVED,
            PerformanceClassification.REGRESSED,
        }:
            performance_report = performance_report.model_copy(
                update={
                    "classification": PerformanceClassification.INCONCLUSIVE,
                    "confirmation": PerformanceConfirmation(
                        initial_classification=performance_report.classification,
                        confirmed=False,
                        repeat_observation_count=0,
                        repeat_valid_observation_count=0,
                        repeat_order="not_run_cached_base",
                    ),
                }
            )
    repeat_observations = None
    repeat_grid = None
    repeat_identity_verified = False
    confirmation_dir = target.parent / "performance-confirmation"
    if confirmation_dir.is_dir():
        confirmation_parquet = confirmation_dir / "trials.parquet"
        confirmation_grid_path = confirmation_dir / "expected-run-grid.json"
        if not confirmation_parquet.is_file() or not confirmation_grid_path.is_file():
            raise typer.BadParameter(
                "Performance confirmation artifacts are incomplete",
                param_hint=str(confirmation_dir),
            )
        initial_checkpoint = target.parent / "judge-checkpoint.json"
        repeat_checkpoint = confirmation_dir / "judge-checkpoint.json"
        if not initial_checkpoint.is_file() or not repeat_checkpoint.is_file():
            raise typer.BadParameter(
                "Performance confirmation is missing its judge identity",
                param_hint=str(confirmation_dir),
            )
        initial_identity = json.loads(initial_checkpoint.read_text())["identity"]
        repeat_identity = json.loads(repeat_checkpoint.read_text())["identity"]
        for field in (
            "candidate_patch_sha256",
            "task_config_sha256",
            "judge_image",
            "cpus",
            "memory",
            "cpuset_cpus",
            "parallel_runs",
            "harness_sha256",
        ):
            if initial_identity.get(field) != repeat_identity.get(field):
                raise typer.BadParameter(
                    f"Performance confirmation differs in {field}",
                    param_hint=str(confirmation_dir),
                )
        if initial_identity["code_states"] != list(
            reversed(repeat_identity["code_states"])
        ):
            raise typer.BadParameter(
                "Performance confirmation did not reverse Base/Agent order",
                param_hint=str(confirmation_dir),
            )
        repeat_grid = ExpectedRunGrid.model_validate_json(
            confirmation_grid_path.read_text()
        )
        if expected_run_grid != repeat_grid:
            raise typer.BadParameter(
                "Performance confirmation uses a different expected run grid",
                param_hint=str(confirmation_dir),
            )
        repeat_observations = ObservationStore.read(confirmation_parquet)
        repeat_identity_verified = True
        repeat_report = compute_task_performance_report(
            [item for item in repeat_observations if item.test_suite is None],
            task=task,
            expected_run_grid=repeat_grid,
        )
        if (
            performance_report is not None
            and repeat_report is not None
            and performance_report.classification
            in {
                PerformanceClassification.IMPROVED,
                PerformanceClassification.REGRESSED,
            }
        ):
            performance_report = adjudicate_performance_confirmation(
                performance_report,
                repeat_report,
                repeat_observation_count=len(repeat_observations),
                repeat_valid_observation_count=sum(
                    item.valid for item in repeat_observations
                ),
            )
    controlled_transfer_report = (
        compute_controlled_transfer_report(
            standard,
            task=task,
            expected_run_grid=expected_run_grid,
        )
        if task.evaluation.controlled_transfer is not None
        and expected_run_grid is not None
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
    if task.evaluation.operational_reliability or any(
        item.test_suite == "operational_reliability" for item in observations
    ):
        reliability_report, _ = compute_reliability_reports(observations, task=task)

    objective_report = compute_objective_report(
        task=task,
        performance=performance_report,
        resources=resource_report,
        transfer=controlled_transfer_report,
        reliability=reliability_report,
        observations=standard,
        expected_run_grid=expected_run_grid,
        repeat_observations=repeat_observations,
        repeat_expected_run_grid=repeat_grid,
        repeat_identity_verified=repeat_identity_verified,
    )
    if json_output:
        combined = {
            "performance": performance_report.model_dump()
            if performance_report
            else None,
            "controlled_transfer": controlled_transfer_report.model_dump()
            if controlled_transfer_report
            else None,
            "resource_usage": resource_report.model_dump() if resource_report else None,
            "operational_reliability": reliability_report.model_dump()
            if reliability_report
            else None,
            "objective_measurements": objective_report.model_dump()
            if objective_report
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
    if controlled_transfer_report is not None:
        typer.echo("\n" + format_controlled_transfer_report(controlled_transfer_report))
    if resource_report is not None:
        typer.echo("\n" + format_resource_report(resource_report))
    if reliability_report is not None:
        typer.echo("\n" + format_reliability_report(reliability_report))
    if objective_report is not None:
        typer.echo("\n" + format_objective_report(objective_report))


if __name__ == "__main__":
    app()
