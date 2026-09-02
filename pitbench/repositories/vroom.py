from pitbench.repositories.base import (
    BuildKind,
    CommandSpec,
    RepositoryPlugin,
    SolverRunSpec,
)


class VroomRepositoryPlugin(RepositoryPlugin):
    name = "vroom"
    deterministic = True

    def build_commands(self, kind: BuildKind) -> list[CommandSpec]:
        flags = "-O1 -g -fsanitize=address,undefined"
        if kind == BuildKind.PERFORMANCE:
            flags = "-O3 -DNDEBUG"
        return [CommandSpec(argv=["make", "-j1", f"CXXFLAGS={flags}"])]

    def run_command(self, run: SolverRunSpec) -> CommandSpec:
        return CommandSpec(
            argv=[
                "python",
                "-m",
                "pitbench.solver_drivers.vroom",
                "--solver",
                "./bin/vroom",
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
