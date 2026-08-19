from pitbench.repositories.base import (
    BuildKind,
    CommandSpec,
    RepositoryPlugin,
    SolverRunSpec,
)


class HighsRepositoryPlugin(RepositoryPlugin):
    name = "highs"

    def build_commands(self, kind: BuildKind) -> list[CommandSpec]:
        build_type = "Debug" if kind == BuildKind.VALIDATION else "Release"
        flags = "-fsanitize=address,undefined" if kind == BuildKind.VALIDATION else ""
        return [
            CommandSpec(
                argv=[
                    "cmake",
                    "-S",
                    ".",
                    "-B",
                    "build",
                    f"-DCMAKE_BUILD_TYPE={build_type}",
                    f"-DCMAKE_CXX_FLAGS={flags}",
                ]
            ),
            CommandSpec(argv=["cmake", "--build", "build", "-j1"]),
        ]

    def run_command(self, run: SolverRunSpec) -> CommandSpec:
        return CommandSpec(
            argv=[
                "python",
                "-m",
                "pitbench.drivers.highs",
                "--solver",
                "./build/bin/highs",
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
