from pitbench.repositories.base import (
    BuildKind,
    CommandSpec,
    RepositoryPlugin,
    SolverRunSpec,
)


class PyVRPRepositoryPlugin(RepositoryPlugin):
    name = "pyvrp"

    def build_commands(self, kind: BuildKind) -> list[CommandSpec]:
        if kind == BuildKind.VALIDATION:
            return [
                CommandSpec(
                    argv=[
                        "python",
                        "-m",
                        "pip",
                        "install",
                        "--no-deps",
                        "-e",
                        ".",
                    ]
                ),
                CommandSpec(argv=["python", "-m", "pytest", "-q"]),
            ]
        return [CommandSpec(argv=["python", "-m", "pip", "install", "--no-deps", "."])]

    def run_command(self, run: SolverRunSpec) -> CommandSpec:
        return CommandSpec(
            argv=[
                "python",
                "-m",
                "pitbench.drivers.pyvrp",
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
