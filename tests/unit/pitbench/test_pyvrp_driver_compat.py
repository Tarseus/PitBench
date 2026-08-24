from types import SimpleNamespace

import pytest

from pitbench.drivers.pyvrp import _route_visits, _statistics_rows, _vrplib
from pitbench.repositories.base import BuildKind, SolverRunSpec
from pitbench.repositories.pyvrp import PyVRPRepositoryPlugin


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
    assert _route_visits(_LegacyRoute()) == [1, 4]
    assert _route_visits(_ModernRoute()) == [1, 4]


def test_statistics_supports_v0_12_population_layout() -> None:
    stats = SimpleNamespace(
        runtimes=[0.1, 0.2],
        feas_stats=[
            SimpleNamespace(size=0, best_cost=float("nan")),
            SimpleNamespace(size=2, best_cost=123),
        ],
    )

    rows = list(_statistics_rows(stats))

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

    assert list(_statistics_rows(Stats())) == [
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

    _vrplib(instance, path)

    assert "EDGE_WEIGHT_TYPE : EUC_2D" in path.read_text()


def test_vrplib_rejects_incompatible_declared_distance_semantics(tmp_path) -> None:
    instance = {
        "coordinates": [[0, 0], [1, 1]],
        "demands": [0, 1],
        "capacity": 1,
        "distance_metric": "EXACT_2D",
    }

    with pytest.raises(ValueError, match="only supports.*EUC_2D"):
        _vrplib(instance, tmp_path / "instance.vrp")


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
    assert validation[3].argv[-4:] == [
        "-o",
        "addopts=",
        "tests",
        "--ignore=tests/plotting",
    ]
    assert run.argv[0] == ".pitbench-venv/bin/python"
