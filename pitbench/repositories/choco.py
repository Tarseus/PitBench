from pitbench.repositories.base import (
    BuildKind,
    CommandSpec,
    RepositoryPlugin,
    SolverRunSpec,
)


class ChocoRepositoryPlugin(RepositoryPlugin):
    name = "choco"

    def build_commands(self, kind: BuildKind) -> list[CommandSpec]:
        goals = ["test"] if kind == BuildKind.VALIDATION else ["package", "-DskipTests"]
        return [CommandSpec(argv=["./mvnw", "-q", *goals])]

    def run_command(self, run: SolverRunSpec) -> CommandSpec:
        return CommandSpec(
            argv=[
                "python",
                "-m",
                "pitbench.solver_drivers.choco",
                "--instance",
                str(run.instance_path),
                "--output",
                str(run.output_path),
                "--trajectory",
                str(run.trajectory_path),
                "--seed",
                str(run.solver_seed),
                "--budget",
                str(run.budget_sec),
                "--threads",
                str(run.threads),
            ],
            timeout_sec=run.budget_sec + 60,
        )
