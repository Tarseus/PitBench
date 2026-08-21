from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Annotated

import typer
import yaml

from adapters.pitbench.adapter import PitBenchAdapter
from fceval.evaluation import EvaluationRequest
from pitbench.distribution.cvrp_v2_matching import (
    run_cvrp_v2_matching_validation,
)
from pitbench.distribution.qoi_axiom_validation import (
    run_cvrp_qoi_axiom_validation,
)
from pitbench.distribution.single_qoi_response import run_pyvrp_single_qoi_pilot
from pitbench.distribution.uchoa_construct_validation import (
    UchoaConstructCase,
    run_uchoa_construct_validation,
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
    qoi_version: Annotated[
        str,
        typer.Option(help="Frozen CVRP instance QoI version (1.0 or 1.1)"),
    ] = "1.1",
) -> None:
    """Validate methodology axioms 1--8 against CVRP instance QoIs."""
    if output.exists():
        raise typer.BadParameter(
            "refusing to overwrite an existing QoI result artifact", param_hint="output"
        )
    namespace = f"cvrp-qoi-axiom-calibration-v{qoi_version}"
    report = run_cvrp_qoi_axiom_validation(
        calibration_size=calibration_size,
        calibration_namespace=namespace,
        qoi_version=qoi_version,
    )
    payload = report.as_dict()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    typer.echo(
        json.dumps(
            {
                "conclusion": report.conclusion,
                "claim_scope": report.claim_scope,
                "axioms": {
                    name: evidence["passed"] for name, evidence in report.axioms.items()
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


@qoi_app.command("validate-uchoa-construct")
def validate_uchoa_construct(
    manifest: Annotated[
        Path,
        typer.Option(help="YAML manifest for locally provisioned Uchoa-X instances"),
    ],
    output: Annotated[Path, typer.Option(help="New JSON construct report path")],
    permutations: Annotated[
        int, typer.Option(min=1, help="Label permutations per primary contrast")
    ] = 4_999,
) -> None:
    """Run solver-free construct validation against actual Uchoa-X metadata."""

    if output.exists():
        raise typer.BadParameter(
            "refusing to overwrite an existing QoI result artifact", param_hint="output"
        )
    payload = yaml.safe_load(manifest.read_text())
    cases = []
    for item in payload["instances"]:
        path = Path(item["path"])
        if not path.is_absolute():
            path = manifest.parent / path
        cases.append(
            UchoaConstructCase(
                instance_id=item["id"],
                instance=json.loads(path.read_text()),
                depot_positioning=item["depot_positioning"],
                customer_positioning=item["customer_positioning"],
                demand_family=item["demand_family"],
                route_size=float(item["route_size"]),
            )
        )
    report = run_uchoa_construct_validation(cases, permutations=permutations)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n")
    typer.echo(
        json.dumps(
            {
                "conclusion": report.conclusion,
                "artifact": str(output),
                "solver_runs_used": 0,
                "solver_runs_created": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


@qoi_app.command("validate-cvrp-v2-matching")
def validate_cvrp_v2_matching(
    output: Annotated[
        Path,
        typer.Option(help="New JSON CVRP v2 matching report path"),
    ],
    pair_count: Annotated[
        int,
        typer.Option(min=4, help="Hidden CRN pairs per treatment"),
    ] = 48,
    generator_seed: Annotated[
        int,
        typer.Option(help="Deterministic Uchoa-style panel seed"),
    ] = 20_260_821,
) -> None:
    """Validate treatment-conditioned matching with CVRP QoI v2-candidate.0."""

    if output.exists():
        raise typer.BadParameter(
            "refusing to overwrite an existing QoI result artifact", param_hint="output"
        )
    report = run_cvrp_v2_matching_validation(
        pair_count=pair_count,
        generator_seed=generator_seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n")
    typer.echo(
        json.dumps(
            {
                "artifact": str(output),
                "conclusion": report.conclusion,
                "summary": report.summary,
                "solver_runs_used": report.solver_runs_used,
                "solver_runs_created": report.solver_runs_created,
            },
            indent=2,
            sort_keys=True,
        )
    )


@qoi_app.command("run-cvrp-single-qoi-pilot")
def run_cvrp_single_qoi_pilot(
    output: Annotated[Path, typer.Option(help="New JSON pilot artifact path")],
    budget_sec: Annotated[
        float, typer.Option(min=0.01, help="Fixed PyVRP budget per run")
    ] = 0.5,
    reference_budget_sec: Annotated[
        float, typer.Option(min=0.02, help="Longer reference budget per pair level")
    ] = 2.0,
    generator_seeds: Annotated[
        str, typer.Option(help="Comma-separated unique generator seeds")
    ] = "101,211,307,401,503,601",
    solver_seeds: Annotated[
        str, typer.Option(help="Comma-separated matched PyVRP seeds")
    ] = "0,1,2",
    bootstrap_repetitions: Annotated[
        int, typer.Option(min=100, help="Generator-cluster bootstrap repetitions")
    ] = 2_000,
) -> None:
    """Run exact single-v1.0-QoI interventions with matched solver seeds."""

    if output.exists():
        raise typer.BadParameter(
            "refusing to overwrite an existing QoI result artifact", param_hint="output"
        )

    def parse_seeds(value: str) -> tuple[int, ...]:
        try:
            seeds = tuple(
                int(item.strip()) for item in value.split(",") if item.strip()
            )
        except ValueError as exc:
            raise typer.BadParameter("seeds must be comma-separated integers") from exc
        if not seeds or len(set(seeds)) != len(seeds):
            raise typer.BadParameter("seeds must be non-empty and unique")
        return seeds

    report = run_pyvrp_single_qoi_pilot(
        generator_seeds=parse_seeds(generator_seeds),
        solver_seeds=parse_seeds(solver_seeds),
        budget_sec=budget_sec,
        reference_budget_sec=reference_budget_sec,
        bootstrap_repetitions=bootstrap_repetitions,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n")
    typer.echo(
        json.dumps(
            {
                "artifact": str(output),
                "axes": list(report.axis_effects),
                "reference_runs": len(report.reference_runs),
                "runs": len(report.runs),
                "solver": report.solver,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    app()
