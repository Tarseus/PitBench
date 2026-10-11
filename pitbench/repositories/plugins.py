"""Build and execution plugins for supported solver repositories."""

import os
import subprocess
from pathlib import Path

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
    agent_artifact_globs = ("pyvrp/**/*.so",)
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

    def agent_build_commands(self) -> list[CommandSpec]:
        return [
            CommandSpec(
                argv=[
                    "python3",
                    "buildtools/build_extensions.py",
                    "--build_dir",
                    "/tmp/pitbench-pyvrp-build",
                ]
            )
        ]

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
    driver_python = "python3"
    name = "vroom"
    agent_requirement = "file:bin/vroom"
    agent_artifact_globs = ("bin/vroom",)
    agent_build_paths = ("lib",)
    collector_environment = {"PITBENCH_VROOM_SOLVER": "/workspace/repo/bin/vroom"}
    representations = {"customer_relabeling": ("judge",)}
    collection_backend = "pitbench.evaluator.collection:VroomCollectionBackend"
    deterministic = True
    driver_name = "vroom"
    driver_solver = "./bin/vroom"

    def build_commands(self, kind: BuildKind) -> list[CommandSpec]:
        make_flags = (
            "-MMD -MP -I. -Isrc -std=c++20 -Wextra -Wpedantic -Wall "
            "-O3 -DASIO_STANDALONE -DUSE_ROUTING=true -DNDEBUG "
            "-D USE_LIBGLPK=true"
        )
        if kind == BuildKind.VALIDATION:
            make_flags += " -O1 -g -fsanitize=address,undefined"
        return [
            CommandSpec(
                argv=[
                    "sh",
                    "-lc",
                    "cd src && test -f structures/typedefs.h && exec make -j1 "
                    + f'"CXXFLAGS={make_flags}" 2>&1',
                ]
            ),
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
    agent_artifact_globs = ("solver/target/*-jar-with-dependencies.jar",)
    agent_build_paths = ("target", "parsers/target", "examples/target")
    judge_environment_requirements = {"MAVEN_CONFIG": "/tmp/choco-maven"}
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
    agent_artifact_globs = (
        "build/java/ortools-java/target/ortools-java-*.jar",
        "build/java/ortools-linux-x86-64/target/ortools-linux-x86-64-*.jar",
    )
    deterministic = False
    driver_name = "ortools_cp_sat_exact"

    def agent_build_commands(self) -> list[CommandSpec]:
        # The prepared task image has a configured Ninja build. Reconfiguring
        # it would write to root-owned dependency directories under /opt.
        return [
            CommandSpec(argv=["mkdir", "-p", "/tmp/ortools-maven"]),
            CommandSpec(
                argv=["cp", "-a", "/root/.m2/repository/.", "/tmp/ortools-maven"]
            ),
            CommandSpec(
                argv=["cmake", "--build", "build", "--target", "java_package", "-j6"],
                env={"MAVEN_OPTS": "-Dmaven.repo.local=/tmp/ortools-maven"},
            ),
        ]

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
    collection_backend = "pitbench.evaluator.collection:OrToolsCpSatCollectionBackend"

    def agent_build_commands(self) -> list[CommandSpec]:
        return [
            *super().agent_build_commands(),
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

    def prepare_independent_verifier(self, output_dir: Path):
        """Compile a checker against the immutable OR-Tools judge runtime."""
        from adapters.jvm.runner import _ORTOOLS_CP_SAT_ASSIGNMENT_VERIFIER_SOURCE
        from pitbench.problem_families.verification import ProcessVerifierFamily

        runtime_root = Path("/workspace/repo/build/java")
        java_jar = runtime_root / "ortools-java/target/ortools-java-9.15.9999.jar"
        native_jar = (
            runtime_root
            / "ortools-linux-x86-64/target/ortools-linux-x86-64-9.15.9999.jar"
        )
        if not java_jar.is_file() or not native_jar.is_file():
            raise RuntimeError("immutable OR-Tools judge runtime is unavailable")

        def maven_jar(*path_parts: str) -> Path:
            artifacts = sorted(
                Path("/root/.m2/repository").joinpath(*path_parts).glob("*/*.jar")
            )
            if len(artifacts) != 1:
                raise RuntimeError(
                    "immutable OR-Tools judge dependency is unavailable: "
                    + "/".join(path_parts)
                )
            return artifacts[0]

        output_dir.mkdir(parents=True, exist_ok=True)
        source_path = output_dir / "CpSatAssignmentVerifier.java"
        source_path.write_text(_ORTOOLS_CP_SAT_ASSIGNMENT_VERIFIER_SOURCE)
        classpath = os.pathsep.join(
            (
                str(java_jar),
                str(native_jar),
                str(maven_jar("com", "google", "protobuf", "protobuf-java")),
                str(maven_jar("net", "java", "dev", "jna", "jna")),
                str(maven_jar("net", "java", "dev", "jna", "jna-platform")),
            )
        )
        completed = subprocess.run(
            ["javac", "-cp", classpath, "-d", str(output_dir), str(source_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                f"CP-SAT assignment verifier compilation failed: {detail}"
            )
        return ProcessVerifierFamily(
            [
                "java",
                f"-Djava.library.path={runtime_root.parent / 'lib'}",
                "-cp",
                os.pathsep.join((str(output_dir), classpath)),
                "org.pitbench.adapters.CpSatAssignmentVerifier",
            ]
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
