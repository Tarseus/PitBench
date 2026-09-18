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
        image="ubuntu:24.04",
        system_packages="build-essential libssl-dev libasio-dev libglpk-dev pkg-config",
        python_packages="",
        prebuild="RUN make -C /workspace/repo/src -j1\n",
    )
    agent_python = "python3"
    name = "vroom"
    agent_requirement = "file:bin/vroom"
    representations = {"customer_relabeling": ("judge",)}
    collection_backend = "pitbench.evaluator.collection:VroomCollectionBackend"
    deterministic = True
    driver_name = "vroom"
    driver_solver = "./bin/vroom"

    def build_commands(self, kind: BuildKind) -> list[CommandSpec]:
        extra_flags = ""
        if kind == BuildKind.VALIDATION:
            extra_flags = "CXXFLAGS+=-O1 -g -fsanitize=address,undefined"
        return [
            CommandSpec(
                argv=["make", "-j1", *([extra_flags] if extra_flags else [])],
                cwd="src",
            )
        ]


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
    driver_python = "python3"
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
        image="maven:3.9-eclipse-temurin-17",
        system_packages="",
        python_packages="",
        prebuild="",
    )
    agent_python = "python3"
    name = "choco"
    driver_python = "python3"
    agent_requirement = "env:PITBENCH_CHOCO_RUNNER"
    collection_backend = "pitbench.evaluator.collection:ChocoCollectionBackend"
    driver_name = "choco"

    def build_commands(self, kind: BuildKind) -> list[CommandSpec]:
        del kind
        goals = ["package", "-DskipTests"]
        return [
            CommandSpec(argv=["mkdir", "-p", "/tmp/choco-maven"]),
            CommandSpec(
                argv=["cp", "-a", "/root/.m2/repository/.", "/tmp/choco-maven"]
            ),
            CommandSpec(
                argv=["mvn", "-q", *goals],
                env={"MAVEN_OPTS": "-Dmaven.repo.local=/tmp/choco-maven"},
            ),
            CommandSpec(
                argv=[
                    "python3",
                    "/opt/pitbench-jvm/runner.py",
                    "compile",
                    "--solver",
                    "choco",
                ]
            ),
        ]


class OrToolsRepositoryPlugin(RepositoryPlugin):
    agent_environment = AgentEnvironment(
        image="maven:3.9-eclipse-temurin-17",
        system_packages="build-essential cmake ninja-build swig",
        python_packages="",
        prebuild="",
    )
    agent_python = "python3"
    name = "ortools"
    driver_python = "python3"
    agent_requirement = "env:PITBENCH_ORTOOLS_CP_SAT_RUNNER"
    deterministic = False
    driver_name = "ortools_cp_sat_exact"
    driver_records_trajectory = False

    def build_commands(self, kind: BuildKind) -> list[CommandSpec]:
        del kind
        return [
            CommandSpec(argv=["mkdir", "-p", "/tmp/ortools-deps"]),
            CommandSpec(argv=["mkdir", "-p", "/tmp/ortools-maven"]),
            CommandSpec(
                argv=["cp", "-a", "/root/.m2/repository/.", "/tmp/ortools-maven"]
            ),
            CommandSpec(
                argv=[
                    "cmake",
                    "-S",
                    ".",
                    "-B",
                    "build",
                    "-G",
                    "Ninja",
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DFETCHCONTENT_BASE_DIR=/tmp/ortools-deps",
                    "-DFETCHCONTENT_SOURCE_DIR_ZLIB=/opt/ortools-deps/zlib-src",
                    "-DFETCHCONTENT_SOURCE_DIR_BZIP2=/opt/ortools-deps/bzip2-src",
                    "-DFETCHCONTENT_SOURCE_DIR_ABSL=/opt/ortools-deps/absl-src",
                    "-DFETCHCONTENT_SOURCE_DIR_PROTOBUF=/opt/ortools-deps/protobuf-src",
                    "-DFETCHCONTENT_SOURCE_DIR_RE2=/opt/ortools-deps/re2-src",
                    "-DFETCHCONTENT_SOURCE_DIR_EIGEN3=/opt/ortools-deps/eigen3-src",
                    "-DBUILD_JAVA=ON",
                    "-DBUILD_DEPS=ON",
                    "-DBUILD_MATH_OPT=ON",
                    "-DBUILD_TESTING=OFF",
                    "-DBUILD_SAMPLES=OFF",
                    "-DBUILD_EXAMPLES=OFF",
                    "-DBUILD_FLATZINC=OFF",
                    "-DUSE_BOP=ON",
                    "-DUSE_COINOR=OFF",
                    "-DUSE_GUROBI=ON",
                    "-DUSE_HIGHS=OFF",
                    "-DUSE_PDLP=OFF",
                    "-DUSE_SCIP=OFF",
                    "-DUSE_XPRESS=ON",
                ]
            ),
            CommandSpec(
                argv=["cmake", "--build", "build", "--target", "java_package", "-j6"],
                env={"MAVEN_OPTS": "-Dmaven.repo.local=/tmp/ortools-maven"},
            ),
        ]


class OrToolsCpSatExactRepositoryPlugin(OrToolsRepositoryPlugin):
    """Build the OR-Tools Java runtime and fixed-proto CP-SAT solve adapter."""

    driver_name = "ortools_cp_sat_exact"
    collection_backend = (
        "pitbench.evaluator.collection:OrToolsCpSatCollectionBackend"
    )

    def build_commands(self, kind: BuildKind) -> list[CommandSpec]:
        return [
            *super().build_commands(kind),
            CommandSpec(
                argv=[
                    "python3",
                    "/opt/pitbench-jvm/runner.py",
                    "compile",
                    "--solver",
                    "ortools_cp_sat",
                ]
            ),
        ]
