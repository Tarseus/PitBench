from pitbench.repositories.base import (
    BuildKind,
    CommandSpec,
    RepositoryPlugin,
    SolverRunSpec,
)


class PyVRPRepositoryPlugin(RepositoryPlugin):
    name = "pyvrp"
    _PYTHON = ".pitbench-venv/bin/python"

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
