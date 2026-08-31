from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Annotated

import typer

from adapters.pitbench.adapter import PitBenchAdapter
from pitbench.evaluator.evaluator import PitBenchEvaluator
from pitbench.evaluator.storage import ObservationStore
from pitbench.harness.cli.harness_cli.runs import create as run_harness
from pitbench.harness.evaluation import EvaluationRequest
from pitbench.instances import materialize_population
from pitbench.metrics.behavior_metrics import compute_behavior_metric_report
from pitbench.metrics.outcome_metrics import (
    compute_outcome_metrics,
    format_outcome_report_table,
)
from pitbench.metrics.sensitivity_metrics import (
    compute_sensitivity_report,
    format_sensitivity_report_table,
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
        typer.Option(help="Agent container base image override"),
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
    """Generate complete PitBench evaluation report (Outcome 3D + Sensitivity)."""
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

    outcome_report = compute_outcome_metrics(observations)
    sensitivity_report = compute_sensitivity_report(observations)
    behavior_report = compute_behavior_metric_report(observations)

    if json_output:
        combined = {
            "outcomes": outcome_report.model_dump(),
            "sensitivity": sensitivity_report.model_dump(),
            "behavior": behavior_report.model_dump(),
        }
        typer.echo(json.dumps(combined, indent=2))
        return

    typer.echo("\n========================================================")
    typer.echo(f"  PitBench Evaluation Report: {target.name}")
    typer.echo(f"  Total observations: {len(observations)}")
    typer.echo("========================================================\n")

    typer.echo("=== I. Outcome 3D (Performance, Reliability, Resource) ===")
    typer.echo(format_outcome_report_table(outcome_report))

    typer.echo("\n=== II. Sensitivity Dimensions & Summary Matrix ===")
    typer.echo(format_sensitivity_report_table(sensitivity_report))

    typer.echo("\n=== III. Population-Conditional Solver Distances ===")
    for item in behavior_report.slices:
        coordinates = ", ".join(
            f"{name}={coordinate.distance:.6g}"
            if coordinate.distance is not None
            else f"{name}=undefined"
            for name, coordinate in (
                ("performance", item.performance),
                ("reliability", item.reliability),
                ("resource", item.resource),
            )
        )
        typer.echo(
            f"{item.population} @ {item.budget_sec:g}s, p={item.p:g}: {coordinates}"
        )

    if outcome_report.by_population:
        typer.echo("\n--- By Population Outcome Breakdown ---")
        for pop_name, pop_report in outcome_report.by_population.items():
            typer.echo(f"\n[Population: {pop_name}]")
            typer.echo(format_outcome_report_table(pop_report))


if __name__ == "__main__":
    app()
