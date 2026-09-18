from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pitbench.repositories.base import (
    BuildKind,
    CommandSpec,
    RepositoryPlugin,
    RepositoryPluginRegistry,
    SolverRunSpec,
)
from pitbench.repositories.plugins import (
    ChocoRepositoryPlugin,
    HighsRepositoryPlugin,
    PyVRPRepositoryPlugin,
)
from pitbench.solver_drivers.run import PyVRPDriver, SolverDriver

ROOT = Path(__file__).resolve().parents[3]

# Tests consolidated from tests/unit/pitbench/test_solver_drivers.py


@pytest.mark.parametrize(
    ("plugin_name", "driver_name", "class_name"),
    [
        ("pyvrp", "pyvrp", "PyVRPRepositoryPlugin"),
        ("vroom", "vroom", "VroomRepositoryPlugin"),
        ("highs", "highs", "HighsRepositoryPlugin"),
        ("choco", "choco", "ChocoRepositoryPlugin"),
        (
            "ortools",
            "ortools_cp_sat_exact",
            "OrToolsCpSatExactRepositoryPlugin",
        ),
    ],
)
def test_repository_plugins_resolve_to_runnable_drivers(
    plugin_name, driver_name, class_name, tmp_path
):
    plugin = RepositoryPluginRegistry.load(
        f"pitbench.repositories.plugins:{class_name}"
    )
    assert plugin.name == plugin_name
    assert plugin.driver_name == driver_name
    run = SolverRunSpec(
        instance_path=tmp_path / "instance.json",
        output_path=tmp_path / "result.json",
        trajectory_path=tmp_path / "trajectory.jsonl",
        solver_seed=17,
        budget_sec=10,
        threads=1,
    )
    command = plugin.run_command(run)
    completed = subprocess.run(
        [sys.executable, *command.argv[1:], "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--instance" in completed.stdout
    assert "--budget" in completed.stdout


def test_solver_driver_registry_rejects_duplicate_names():
    name = "test-duplicate-driver"

    class FirstDriver(SolverDriver):
        @staticmethod
        def main(argv=None):
            return None

    FirstDriver.name = name
    SolverDriver._drivers[name] = FirstDriver
    try:
        with pytest.raises(ValueError, match="duplicate solver driver"):

            class DuplicateDriver(SolverDriver):
                name = "test-duplicate-driver"

                @staticmethod
                def main(argv=None):
                    return None
    finally:
        SolverDriver._drivers.pop(name, None)


def test_collection_backends_share_validated_interface():
    from pitbench.evaluator.collection import CollectionBackend

    for plugin in (PyVRPRepositoryPlugin(), HighsRepositoryPlugin()):
        assert issubclass(plugin.load_collection_backend(), CollectionBackend)

    class InvalidRepository(RepositoryPlugin):
        name = "invalid"
        driver_name = "invalid"
        collection_backend = "pitbench.solver_drivers.run:PyVRPDriver"

        def build_commands(self, kind):
            return []

    with pytest.raises(TypeError, match="is not a CollectionBackend"):
        InvalidRepository().load_collection_backend()


def test_choco_builds_official_source_and_generic_jvm_adapter() -> None:
    plugin = ChocoRepositoryPlugin()

    validation = plugin.build_commands(BuildKind.VALIDATION)
    performance = plugin.build_commands(BuildKind.PERFORMANCE)

    assert validation[0] == performance[0] == CommandSpec(
        argv=["mkdir", "-p", "/tmp/choco-maven"]
    )
    assert validation[1] == performance[1] == CommandSpec(
        argv=["cp", "-a", "/root/.m2/repository/.", "/tmp/choco-maven"]
    )
    assert validation[2] == CommandSpec(
        argv=["mvn", "-q", "package", "-DskipTests"],
        env={"MAVEN_OPTS": "-Dmaven.repo.local=/tmp/choco-maven"},
    )
    assert performance[2] == CommandSpec(
        argv=["mvn", "-q", "package", "-DskipTests"],
        env={"MAVEN_OPTS": "-Dmaven.repo.local=/tmp/choco-maven"},
    )
    assert (
        validation[3].argv
        == performance[3].argv
        == [
            "python3",
            "/opt/pitbench-jvm/runner.py",
            "compile",
            "--solver",
            "choco",
        ]
    )
    assert plugin.agent_environment is not None
    assert plugin.agent_environment.image.endswith("eclipse-temurin-17")


def test_highs_driver_uses_python3() -> None:
    assert HighsRepositoryPlugin().driver_python == "python3"


def test_choco_driver_uses_python3() -> None:
    assert ChocoRepositoryPlugin().driver_python == "python3"


def test_vroom_build_uses_its_source_makefile_without_replacing_flags() -> None:
    from pitbench.repositories.plugins import VroomRepositoryPlugin

    validation = VroomRepositoryPlugin().build_commands(BuildKind.VALIDATION)
    performance = VroomRepositoryPlugin().build_commands(BuildKind.PERFORMANCE)

    assert validation == [
        CommandSpec(
            argv=["make", "-j1", "CXXFLAGS+=-O1 -g -fsanitize=address,undefined"],
            cwd="src",
        )
    ]
    assert performance == [CommandSpec(argv=["make", "-j1"], cwd="src")]


def test_ortools_build_reuses_evaluator_owned_dependency_cache() -> None:
    from pitbench.repositories.plugins import (
        OrToolsCpSatExactRepositoryPlugin,
        OrToolsRepositoryPlugin,
    )

    prepare_cache, prepare_maven, copy_maven, configure, build = (
        OrToolsRepositoryPlugin().build_commands(BuildKind.PERFORMANCE)
    )

    assert prepare_cache.argv == ["mkdir", "-p", "/tmp/ortools-deps"]
    assert prepare_maven.argv == ["mkdir", "-p", "/tmp/ortools-maven"]
    assert copy_maven.argv == [
        "cp",
        "-a",
        "/root/.m2/repository/.",
        "/tmp/ortools-maven",
    ]
    assert configure.argv[:6] == ["cmake", "-S", ".", "-B", "build", "-G"]
    assert "-DFETCHCONTENT_BASE_DIR=/tmp/ortools-deps" in configure.argv
    assert "-DCMAKE_BUILD_TYPE=Release" in configure.argv
    assert "-DFETCHCONTENT_SOURCE_DIR_ZLIB=/opt/ortools-deps/zlib-src" in configure.argv
    assert (
        "-DFETCHCONTENT_SOURCE_DIR_BZIP2=/opt/ortools-deps/bzip2-src" in configure.argv
    )
    assert "-DFETCHCONTENT_SOURCE_DIR_ABSL=/opt/ortools-deps/absl-src" in configure.argv
    assert (
        "-DFETCHCONTENT_SOURCE_DIR_PROTOBUF=/opt/ortools-deps/protobuf-src"
        in configure.argv
    )
    assert "-DFETCHCONTENT_SOURCE_DIR_RE2=/opt/ortools-deps/re2-src" in configure.argv
    assert (
        "-DFETCHCONTENT_SOURCE_DIR_EIGEN3=/opt/ortools-deps/eigen3-src"
        in configure.argv
    )
    assert "-DBUILD_JAVA=ON" in configure.argv
    assert "-DBUILD_DEPS=ON" in configure.argv
    assert "-DBUILD_TESTING=OFF" in configure.argv
    assert "-DUSE_GUROBI=ON" in configure.argv
    assert "-DBUILD_SAMPLES=OFF" in configure.argv
    assert "-DBUILD_EXAMPLES=OFF" in configure.argv
    assert build.argv == [
        "cmake",
        "--build",
        "build",
        "--target",
        "java_package",
        "-j6",
    ]
    assert build.env == {"MAVEN_OPTS": "-Dmaven.repo.local=/tmp/ortools-maven"}

    exact_commands = OrToolsCpSatExactRepositoryPlugin().build_commands(
        BuildKind.PERFORMANCE
    )
    assert OrToolsCpSatExactRepositoryPlugin().driver_python == "python3"
    assert exact_commands[-1].argv == [
        "python3",
        "/opt/pitbench-jvm/runner.py",
        "compile",
        "--solver",
        "ortools_cp_sat",
    ]


def test_each_target_plugin_has_an_image_definition() -> None:
    for name in ("pyvrp", "vroom", "highs", "choco", "ortools"):
        assert (ROOT / "docker/solver-images" / name / "Dockerfile").is_file()


class _LegacyRoute:
    def visits(self) -> list[int]:
        return [1, 4]


class _ModernRoute:
    def __iter__(self):
        return iter(
            [
                SimpleNamespace(idx=0, is_client=lambda: False),
                SimpleNamespace(idx=0, is_client=lambda: True),
                SimpleNamespace(idx=3, is_client=lambda: True),
                SimpleNamespace(idx=0, is_client=lambda: False),
            ]
        )


def test_route_extraction_supports_pre_and_post_v0_14_apis() -> None:
    assert PyVRPDriver._route_visits(_LegacyRoute()) == [1, 4]
    assert PyVRPDriver._route_visits(_ModernRoute()) == [1, 4]


def test_statistics_supports_v0_12_instance_set_layout() -> None:
    stats = SimpleNamespace(
        runtimes=[0.1, 0.2],
        feas_stats=[
            SimpleNamespace(size=0, best_cost=float("nan")),
            SimpleNamespace(size=2, best_cost=123),
        ],
    )

    rows = list(PyVRPDriver._statistics_rows(stats))

    assert rows[0][0:2] == (0.1, False)
    assert rows[1] == (0.2, True, 123)


def test_statistics_supports_v0_13_and_v0_14_iterable_layout() -> None:
    class Stats:
        runtimes = [0.1, 0.2]

        def __iter__(self):
            return iter(
                [
                    SimpleNamespace(best_feas=False, best_cost=999),
                    SimpleNamespace(best_feas=True, best_cost=123),
                ]
            )

    assert list(PyVRPDriver._statistics_rows(Stats())) == [
        (0.1, False, 999),
        (0.2, True, 123),
    ]


def test_vrplib_uses_declared_euc_2d_semantics(tmp_path) -> None:
    instance = {
        "name": "rounded",
        "coordinates": [[0, 0], [1, 1]],
        "demands": [0, 1],
        "capacity": 1,
        "distance_metric": "EUC_2D",
    }
    path = tmp_path / "instance.vrp"

    PyVRPDriver._vrplib(instance, path)

    assert "EDGE_WEIGHT_TYPE : EUC_2D" in path.read_text()


def test_vrplib_rejects_incompatible_declared_distance_semantics(tmp_path) -> None:
    instance = {
        "coordinates": [[0, 0], [1, 1]],
        "demands": [0, 1],
        "capacity": 1,
        "distance_metric": "EXACT_2D",
    }

    with pytest.raises(ValueError, match="only supports.*EUC_2D"):
        PyVRPDriver._vrplib(instance, tmp_path / "instance.vrp")


def test_pyvrp_builds_and_runs_in_per_workspace_virtual_environments(
    tmp_path,
) -> None:
    plugin = PyVRPRepositoryPlugin()

    validation = plugin.build_commands(BuildKind.VALIDATION)
    performance = plugin.build_commands(BuildKind.PERFORMANCE)
    run = plugin.run_command(
        SolverRunSpec(
            instance_path=tmp_path / "instance.json",
            output_path=tmp_path / "result.json",
            trajectory_path=tmp_path / "trajectory.jsonl",
            solver_seed=0,
            budget_sec=1,
            threads=1,
        )
    )

    assert validation[0].argv[-2:] == ["--system-site-packages", ".pitbench-venv"]
    assert performance[0].argv[-2:] == ["--system-site-packages", ".pitbench-venv"]
    assert all(
        command.argv[0] == ".pitbench-venv/bin/python"
        for command in (*validation[1:], *performance[1:])
    )
    assert validation[1].argv[3:5] == ["--build_type", "debug"]
    assert validation[1].argv[-5:] == [
        "-Doptimization=1",
        "-Ddebug=true",
        "-Db_sanitize=address,undefined",
        "-Db_coverage=false",
        "-Db_lto=false",
    ]
    assert "prepare_metadata_for_build_wheel" in validation[2].argv[-1]
    assert validation[3].env["LD_PRELOAD"] == "libasan.so.8:libstdc++.so.6"
    assert validation[3].env["PYTHONPATH"] == ".pitbench-metadata"
    assert validation[3].env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert "--no-build-isolation" in performance[1].argv
    assert validation[3].argv[-5:] == [
        "-o",
        "addopts=",
        "tests",
        "--ignore=tests/plotting",
        "--deselect=tests/search/test_LocalSearch.py::test_debug_operator_logs",
    ]
    assert run.argv[0] == ".pitbench-venv/bin/python"
