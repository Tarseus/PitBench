"""Build and execution plugins for supported solver repositories."""

from pitbench.repositories.base import (
    AgentEnvironment,
    BuildKind,
    CommandSpec,
    RepositoryPlugin,
)


class PyVRPRepositoryPlugin(RepositoryPlugin):
    agent_environment = AgentEnvironment(
        image="python:3.13-trixie",
        system_packages="build-essential cmake ninja-build python3-dev",
        python_packages="docblock matplotlib meson ninja numpy pandas poetry-core pyarrow pybind11 pydantic pytest pytest-cov pytest-timeout pytest-xdist pyyaml setuptools tqdm vrplib wheel",
        prebuild="RUN python3 -m pip install --break-system-packages --no-build-isolation --no-deps -e /workspace/repo\n",
    )
    name = "pyvrp"
    agent_requirement = "import:pyvrp"
    agent_python = "python3"
    collection_backend = "pitbench.evaluator.collection:PyVRPCollectionBackend"
    representations = {"customer_relabeling": ("judge", "isolated")}
    _PYTHON = ".pitbench-venv/bin/python"
    driver_name = "pyvrp"
    driver_python = _PYTHON
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


class VroomRepositoryPlugin(RepositoryPlugin):
    agent_environment = AgentEnvironment(
        image="ubuntu:22.04",
        system_packages="build-essential libssl-dev libasio-dev libglpk-dev pkg-config",
        python_packages="",
        prebuild="RUN cd /workspace/repo && make -j1 CXXFLAGS='-O3 -DNDEBUG'\n",
    )
    agent_python = "python3"
    name = "vroom"
    agent_requirement = "file:bin/vroom"
    representations = {"customer_relabeling": ("judge",)}
    deterministic = True
    driver_name = "vroom"
    driver_solver = "./bin/vroom"

    def build_commands(self, kind: BuildKind) -> list[CommandSpec]:
        flags = "-O1 -g -fsanitize=address,undefined"
        if kind == BuildKind.PERFORMANCE:
            flags = "-O3 -DNDEBUG"
        return [CommandSpec(argv=["make", "-j1", f"CXXFLAGS={flags}"])]


class HighsRepositoryPlugin(RepositoryPlugin):
    agent_environment = AgentEnvironment(
        image="ubuntu:24.04",
        system_packages="build-essential cmake",
        python_packages="",
        prebuild="RUN cd /workspace/repo && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j1\n",
    )
    agent_python = "python3"
    name = "highs"
    agent_requirement = "file:build/bin/highs"
    collection_backend = "pitbench.evaluator.collection:HighsCollectionBackend"
    representations = {"row_column_permutation": ("isolated",)}
    driver_name = "highs"
    driver_solver = "./build/bin/highs"

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


class ChocoRepositoryPlugin(RepositoryPlugin):
    agent_environment = AgentEnvironment(
        image="maven:3.9-eclipse-temurin-11",
        system_packages="",
        python_packages="",
        prebuild="",
    )
    agent_python = "python3"
    name = "choco"
    agent_requirement = "env:PITBENCH_CHOCO_RUNNER"
    driver_name = "choco"

    def build_commands(self, kind: BuildKind) -> list[CommandSpec]:
        goals = ["test"] if kind == BuildKind.VALIDATION else ["package", "-DskipTests"]
        return [CommandSpec(argv=["./mvnw", "-q", *goals])]


class OrToolsRepositoryPlugin(RepositoryPlugin):
    agent_environment = AgentEnvironment(
        image="maven:3.9-eclipse-temurin-11",
        system_packages="build-essential cmake openjdk-11-jdk maven swig",
        python_packages="",
        prebuild="",
    )
    agent_python = "python3"
    name = "ortools"
    agent_requirement = "env:PITBENCH_ORTOOLS_JAVA_RUNNER"
    deterministic = True
    driver_name = "ortools_model_build"
    driver_records_trajectory = False

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
