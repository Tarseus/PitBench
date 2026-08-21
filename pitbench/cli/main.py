from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Annotated

import typer

from adapters.pitbench.adapter import PitBenchAdapter
from fceval.evaluation import EvaluationRequest
from pitbench.distribution.qoi_axiom_validation import (
    run_cvrp_qoi_axiom_validation,
)
from pitbench.evaluator.evaluator import PitBenchEvaluator
from pitbench.instances import materialize_population
from pitbench.schema.task import PopulationKind
from pitbench.tasks import TaskCatalog

app = typer.Typer(help="PitBench task and evaluator tooling.")
tasks_app = typer.Typer(help="Validate and smoke-test benchmark tasks.")
qoi_app = typer.Typer(help="Versioned quantities-of-interest tooling.")
app.add_typer(tasks_app, name="tasks")
app.add_typer(qoi_app, name="qoi")


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
    root: Annotated[Path | None, typer.Option(help="PitBench repository root")] = None,
) -> None:
    repository_root = _root(root)
    task_dir = PitBenchAdapter(
        repository_root,
        private_root if private_root.is_absolute() else repository_root / private_root,
    ).materialize(task_id, output)
    typer.echo(f"Materialized {task_id} at {task_dir}")


@qoi_app.command("validate-cvrp-axioms")
def validate_cvrp_qoi_axioms(
    output: Annotated[
        Path,
        typer.Option(help="JSON destination for the axis-level QoI axiom report"),
    ],
    calibration_size: Annotated[
        int,
        typer.Option(min=8, help="Solver-free designs used for robust axis scales"),
    ] = 128,
) -> None:
    """Validate methodology axioms 1--8 against CVRP instance QoIs."""
    report = run_cvrp_qoi_axiom_validation(calibration_size=calibration_size)
    payload = report.as_dict()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    typer.echo(
        json.dumps(
            {
                "conclusion": report.conclusion,
                "axioms": {
                    name: evidence["passed"]
                    for name, evidence in report.axioms.items()
                },
                "artifact": str(output),
                "solver_runs_used": report.solver_runs_used,
                "solver_runs_created": report.solver_runs_created,
                "production_geometry_changed": report.production_geometry_changed,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    app()
