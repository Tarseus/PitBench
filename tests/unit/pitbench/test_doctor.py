from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from pitbench.cli.doctor import (
    CheckStatus,
    _agent_runtime_check,
    _configuration_collector_checks,
    _configuration_robustness_check,
    _judge_environment_check,
    _reliability_plan_check,
    _repository_source_check,
)
from pitbench.cli.evaluate_config import EvaluationConfig, TaskResources
from pitbench.cli.main import (
    _configuration_dry_run_failures,
    _run_configuration_robustness,
)


def doctor_context(
    repository_root: Path,
    *,
    repository_source: Path | None = None,
    clone_url: str = "https://github.com/PyVRP/PyVRP.git",
) -> SimpleNamespace:
    config = EvaluationConfig()
    config.tasks["pyvrp_v0_14_0"] = TaskResources(
        repository_source=repository_source,
    )
    task = SimpleNamespace(
        task_id="pyvrp_v0_14_0",
        repository=SimpleNamespace(clone_url=clone_url),
    )
    return SimpleNamespace(
        repository_root=repository_root,
        config_path=None,
        config=config,
        task=SimpleNamespace(
            task=SimpleNamespace(
                task_id=task.task_id,
                repository=task.repository,
                evaluation=SimpleNamespace(operational_reliability=True),
            )
        ),
    )


def test_repository_source_rejects_local_snapshot_override(tmp_path: Path) -> None:
    context = doctor_context(tmp_path, repository_source=Path("snapshot"))

    check = _repository_source_check(context, {})

    assert check.status is CheckStatus.FAIL
    assert "local repository snapshot override" in check.detail


def test_repository_source_checks_github_with_global_transport(
    tmp_path: Path,
) -> None:
    context = doctor_context(tmp_path)

    def fake_run(command, **kwargs):
        if command[:3] == ["git", "config", "--global"]:
            return subprocess.CompletedProcess(command, 1, "", "")
        return subprocess.CompletedProcess(command, 0, "refs/heads/main\n", "")

    with patch("pitbench.cli.doctor.subprocess.run", side_effect=fake_run):
        check = _repository_source_check(context, {})

    assert check.status is CheckStatus.PASS
    assert "reachable (direct)" in check.detail


def test_agent_runtime_requires_tmux_and_asciinema() -> None:
    client = SimpleNamespace(
        containers=SimpleNamespace(
            run=Mock(return_value=b"/usr/bin/tmux\n/usr/bin/asciinema\n")
        )
    )

    check = _agent_runtime_check("pitbench-agent:test", client)

    assert check.status is CheckStatus.PASS
    client.containers.run.assert_called_once()


def test_agent_runtime_reports_missing_asciinema() -> None:
    client = SimpleNamespace(
        containers=SimpleNamespace(run=Mock(return_value=b"/usr/bin/tmux\n"))
    )

    check = _agent_runtime_check("pitbench-agent:test", client)

    assert check.status is CheckStatus.FAIL
    assert "asciinema" in check.detail


@pytest.mark.parametrize(
    ("environment", "expected_status"),
    [
        (["MAVEN_CONFIG=/tmp/choco-maven"], CheckStatus.PASS),
        ([], CheckStatus.FAIL),
    ],
)
def test_judge_environment_uses_repository_plugin_contract(
    environment, expected_status
) -> None:
    context = SimpleNamespace(
        task=SimpleNamespace(
            task=SimpleNamespace(
                repository=SimpleNamespace(
                    plugin="pitbench.repositories.plugins:ChocoRepositoryPlugin"
                )
            )
        )
    )
    client = SimpleNamespace(
        images=SimpleNamespace(
            get=Mock(return_value=SimpleNamespace(attrs={"Config": {"Env": environment}}))
        )
    )

    check = _judge_environment_check(context, "pitbench-judge:test", client)

    assert check.status is expected_status
    assert "MAVEN_CONFIG=/tmp/choco-maven" in check.detail


def test_configuration_robustness_requires_experiment_config(
    tmp_path: Path,
) -> None:
    config = EvaluationConfig()

    check = _configuration_robustness_check(tmp_path, config)

    assert check.status is CheckStatus.FAIL
    assert "experiment config does not exist" in check.detail


def test_configuration_robustness_passes_when_enabled_config_exists(
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "configuration.yaml"
    experiment.write_text("name: test\n")
    config = EvaluationConfig()
    config.configuration_robustness.experiment_config = experiment

    check = _configuration_robustness_check(tmp_path, config)

    assert check.status is CheckStatus.PASS


def test_configuration_preflight_rejects_stale_collector_before_agent_work(
    tmp_path: Path, monkeypatch
) -> None:
    experiment = tmp_path / "configuration.yaml"
    experiment.write_text(
        "collector_images:\n"
        "  ortools_cp_sat: pitbench-ortools:stale\n"
        "panels:\n"
        "  - task_config: configs/tasks/ortools_cp_sat_solve_v9_15.yaml\n"
        "    collector: ortools_cp_sat\n"
    )
    config = EvaluationConfig()
    config.configuration_robustness.experiment_config = experiment
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            1,
            "",
            "compiled ortools_cp_sat adapter differs from source; rebuild the image",
        )

    monkeypatch.setattr("pitbench.cli.doctor.subprocess.run", run)
    root = Path(__file__).resolve().parents[3]

    checks = _configuration_collector_checks(root, config)

    assert len(checks) == 1
    assert checks[0].status is CheckStatus.FAIL
    assert "differs from source" in checks[0].detail
    assert "--pull=never" in commands[0]


def test_configuration_collector_forwards_the_limited_panel_size(
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "configuration.yaml"
    experiment.write_text("name: test\n")
    config = EvaluationConfig()
    config.configuration_robustness.experiment_config = experiment
    config.configuration_robustness.cpus = [min(os.sched_getaffinity(0))]
    commands = []

    with (
        patch("pitbench.cli.main.shutil.which", return_value="uv"),
        patch(
            "pitbench.cli.main.subprocess.run",
            side_effect=lambda command, **kwargs: commands.append(command),
        ),
    ):
        _run_configuration_robustness(
            tmp_path,
            config,
            Path("runs/run"),
            "dry-run",
            instances_per_panel=1,
        )

    assert len(commands) == 2
    assert commands[0][-2:] == ["--instances-per-panel", "1"]
    assert commands[1][-2:] == [
        "--output",
        str(tmp_path / "runs/run/dry-run/configuration-robustness"),
    ]


def test_configuration_dry_run_rejects_incomplete_searches(tmp_path: Path) -> None:
    (tmp_path / "experiment.json").write_text(
        '{"searches": [{"id": "solver/budget-10"}]}'
    )
    (tmp_path / "configuration_report.json").write_text(
        '{"searches": [{"id": "solver/budget-10", "status": "baseline_unavailable"}]}'
    )

    failures = _configuration_dry_run_failures(tmp_path)

    assert failures == [
        "configuration search solver/budget-10 is baseline_unavailable"
    ]


def test_reliability_plan_check_reports_materialized_cases(
    tmp_path: Path,
) -> None:
    context = doctor_context(tmp_path)

    with patch(
        "pitbench.cli.doctor.prepare_boundary_cases",
        return_value=[object(), object()],
    ):
        check = _reliability_plan_check(context)

    assert check.status is CheckStatus.PASS
    assert "2 boundary cases" in check.detail
