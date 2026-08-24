from pathlib import Path
from unittest.mock import Mock, patch

from typer.testing import CliRunner

from adapters.pitbench.adapter import (
    IMAGE_REVISION,
    IMAGE_REVISION_LABEL,
    IMAGE_SOURCE_LABEL,
)
from pitbench.cli.main import _prepare_task_image, app
from pitbench.harness.agents import AgentName


def test_prepare_task_image_reuses_matching_revision(tmp_path: Path) -> None:
    (tmp_path / "Dockerfile").write_text("FROM python:3.13-bookworm\n")
    trial = Mock()
    trial.client_container_name = "prepare"
    trial.client_image_name = "tb__pyvrp__client"
    trial.docker_image_name_prefix = "tb__pyvrp"
    trial.task_paths.docker_compose_path = tmp_path / "docker-compose.yaml"

    with (
        patch("pitbench.cli.main.TrialHandler", return_value=trial),
        patch("pitbench.cli.main.DockerComposeManager") as manager_class,
    ):
        manager = manager_class.return_value
        manager.image_label.side_effect = {
            IMAGE_REVISION_LABEL: IMAGE_REVISION,
            IMAGE_SOURCE_LABEL: "python:3.13-bookworm",
        }.get

        _prepare_task_image(tmp_path, rebuild=False)

    manager.build.assert_not_called()


def test_prepare_task_image_builds_missing_or_forced_image(tmp_path: Path) -> None:
    (tmp_path / "Dockerfile").write_text("FROM python:3.13-bookworm\n")
    trial = Mock()
    trial.client_container_name = "prepare"
    trial.client_image_name = "tb__pyvrp__client"
    trial.docker_image_name_prefix = "tb__pyvrp"
    trial.task_paths.docker_compose_path = tmp_path / "docker-compose.yaml"

    with (
        patch("pitbench.cli.main.TrialHandler", return_value=trial),
        patch("pitbench.cli.main.DockerComposeManager") as manager_class,
    ):
        manager = manager_class.return_value
        manager.image_label.return_value = None

        _prepare_task_image(tmp_path, rebuild=False)
        _prepare_task_image(tmp_path, rebuild=True)

    assert manager.build.call_count == 2


def test_evaluate_materializes_task_and_starts_harness(tmp_path: Path) -> None:
    runner = CliRunner()
    repository_root = tmp_path / "pitbench"
    workspace = tmp_path / "workspaces"
    output = tmp_path / "runs"
    repository_root.mkdir()

    with (
        patch("pitbench.cli.main.PitBenchAdapter") as adapter_class,
        patch("pitbench.cli.main._prepare_task_image") as prepare_task_image,
        patch("pitbench.cli.main.run_harness") as run_harness,
    ):
        adapter = Mock()
        adapter_class.return_value = adapter

        result = runner.invoke(
            app,
            [
                "evaluate",
                "pyvrp_v0_14_0",
                "--agent",
                "codex",
                "--model",
                "gpt-5.6-terra",
                "--run-id",
                "paper-run",
                "--workspace-path",
                str(workspace),
                "--output-path",
                str(output),
                "--root",
                str(repository_root),
                "--n-concurrent",
                "2",
                "--n-attempts",
                "3",
                "--agent-kwarg",
                "proxy_url=http://127.0.0.1:7897",
            ],
        )

    assert result.exit_code == 0, result.output
    dataset_path = (workspace / "paper-run").resolve()
    adapter_class.assert_called_once_with(
        repository_root.resolve(), repository_root.resolve() / "private"
    )
    adapter.materialize.assert_called_once_with(
        "pyvrp_v0_14_0",
        dataset_path / "pyvrp_v0_14_0",
        repository_source=None,
        agent_image=None,
        judge_image=None,
    )
    prepare_task_image.assert_called_once_with(
        dataset_path / "pyvrp_v0_14_0", rebuild=False
    )
    harness_kwargs = run_harness.call_args.kwargs
    assert harness_kwargs["dataset_path"] == dataset_path
    assert harness_kwargs["agent"] == AgentName.CODEX
    assert harness_kwargs["model_name"] == "gpt-5.6-terra"
    assert harness_kwargs["run_id"] == "paper-run"
    assert harness_kwargs["output_path"] == output
    assert harness_kwargs["n_concurrent_trials"] == 2
    assert harness_kwargs["n_attempts"] == 3
    assert harness_kwargs["no_rebuild"] is True
    assert harness_kwargs["cleanup"] is False
    assert harness_kwargs["agent_kwargs"] == [
        "no_rebuild=False",
        "proxy_url=http://127.0.0.1:7897"
    ]


def test_evaluate_stops_when_materialization_fails(tmp_path: Path) -> None:
    runner = CliRunner()

    with (
        patch("pitbench.cli.main.PitBenchAdapter") as adapter_class,
        patch("pitbench.cli.main._prepare_task_image") as prepare_task_image,
        patch("pitbench.cli.main.run_harness") as run_harness,
    ):
        adapter_class.return_value.materialize.side_effect = RuntimeError(
            "materialization failed"
        )
        result = runner.invoke(
            app,
            [
                "evaluate",
                "pyvrp_v0_14_0",
                "--agent",
                "codex",
                "--root",
                str(tmp_path),
            ],
        )

    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)
    prepare_task_image.assert_not_called()
    run_harness.assert_not_called()
