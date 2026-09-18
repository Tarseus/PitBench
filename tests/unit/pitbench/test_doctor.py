from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pitbench.cli.doctor import (
    CheckStatus,
    _agent_runtime_check,
    _configuration_robustness_check,
    _repository_source_check,
)
from pitbench.cli.evaluate_config import EvaluationConfig, TaskResources


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
        task=SimpleNamespace(task=task),
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
