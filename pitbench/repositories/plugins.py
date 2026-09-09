"""Build and execution plugins for supported solver repositories."""

from pitbench.repositories.base import (
    BuildKind,
    CommandSpec,
    RepositoryPlugin,
    SolverRunSpec,
)


class PyVRPRepositoryPlugin(RepositoryPlugin):
    name = "pyvrp"
    _PYTHON = ".pitbench-venv/bin/python"
    # This test pins the exact local-search trace and move count. Those are
    # performance-policy outputs, not correctness contracts for this benchmark.
    _TRAJECTORY_REGRESSION_TESTS = (
        "tests/search/test_LocalSearch.py::test_debug_operator_logs",
    )

    def build_commands(self, kind: BuildKind) -> list[CommandSpec]:
        create_environment = CommandSpec(
            argv=["python", "-m", "venv", "--system-site-packages", ".pitbench-venv"]
        )
        if kind == BuildKind.VALIDATION:
            return [
                create_environment,
                CommandSpec(
                    argv=[
                        self._PYTHON,
                        "buildtools/build_extensions.py",
                        "--clean",
                        "--build_type",
                        "debug",
                        "--additional",
                        "-Doptimization=1",
                        "-Ddebug=true",
                        "-Db_sanitize=address,undefined",
                        "-Db_coverage=false",
                        "-Db_lto=false",
                    ],
                ),
                CommandSpec(
                    argv=[
                        self._PYTHON,
                        "-c",
                        (
                            "from poetry.core.masonry.api import "
                            "prepare_metadata_for_build_wheel; "
                            "prepare_metadata_for_build_wheel("
                            "'.pitbench-metadata')"
                        ),
                    ],
                ),
                CommandSpec(
                    argv=[
                        self._PYTHON,
                        "-m",
                        "pytest",
                        "-q",
                        "-o",
                        "addopts=",
                        "tests",
                        "--ignore=tests/plotting",
                        *[
                            f"--deselect={test}"
                            for test in self._TRAJECTORY_REGRESSION_TESTS
                        ],
                    ],
                    env={
                        "ASAN_OPTIONS": (
                            "detect_leaks=0:halt_on_error=1:"
                            "malloc_context_size=5:quarantine_size_mb=64"
                        ),
                        "LD_PRELOAD": "libasan.so.8:libstdc++.so.6",
                        "PYTHONPATH": ".pitbench-metadata",
                        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                        "UBSAN_OPTIONS": "halt_on_error=1:print_stacktrace=1",
                    },
                ),
            ]
        return [
            create_environment,
            CommandSpec(
                argv=[
                    self._PYTHON,
                    "-m",
                    "pip",
                    "install",
                    "--no-build-isolation",
                    "--no-deps",
                    ".",
                ]
            ),
        ]

    def run_command(self, run: SolverRunSpec) -> CommandSpec:
        return CommandSpec(
            argv=[
                self._PYTHON,
                "-m",
                "pitbench.solver_drivers.run",
                "pyvrp",
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
                "pitbench.solver_drivers.run",
                "vroom",
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
                "pitbench.solver_drivers.run",
                "highs",
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
                "pitbench.solver_drivers.run",
                "choco",
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
                "pitbench.solver_drivers.run",
                "ortools_model_build",
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
