from pitbench.repositories.base import (
    BuildKind,
    CommandSpec,
    RepositoryPlugin,
    SolverRunSpec,
)


class OrToolsRepositoryPlugin(RepositoryPlugin):
    name = "ortools"
    deterministic = True

    def build_commands(self, kind: BuildKind) -> list[CommandSpec]:
        config = "Debug" if kind == BuildKind.VALIDATION else "Release"
        return [
            CommandSpec(
                argv=[
                    "cmake",
                    "-S",
                    ".",
                    "-B",
                    "build",
                    f"-DCMAKE_BUILD_TYPE={config}",
                    "-DBUILD_JAVA=ON",
                ]
            ),
            CommandSpec(argv=["cmake", "--build", "build", "-j1"]),
        ]

    def run_command(self, run: SolverRunSpec) -> CommandSpec:
        return CommandSpec(
            argv=[
                "python",
                "-m",
                "pitbench.drivers.ortools_model_build",
                "--instance",
                str(run.instance_path),
                "--output",
                str(run.output_path),
                "--seed",
                str(run.solver_seed),
                "--budget",
                str(run.budget_sec),
                "--threads",
                str(run.threads),
            ],
            timeout_sec=run.budget_sec + 60,
        )
