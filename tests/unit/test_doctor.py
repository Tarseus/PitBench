from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from adapters.pitbench.adapter import PitBenchAdapter
from pitbench.cli.doctor import (
    CheckStatus,
    _agent_runtime_check,
    _configuration_collector_checks,
    _configuration_robustness_check,
    _docker_storage_check,
    _image_reference_check,
    _judge_environment_check,
    _pinned_source_check,
    _reliability_plan_check,
)
from pitbench.cli.evaluate_config import DryRunConfig, EvaluationConfig
from pitbench.cli.main import (
    _configuration_dry_run_failures,
    _run_configuration_robustness,
)
from pitbench.schema.task import PitBenchTask
from pitbench.tasks import TaskCatalog
from tests.support import ROOT


@pytest.mark.parametrize("kind", ["agent", "judge"])
def test_doctor_accepts_local_image_ids_supported_by_task_preparation(kind):
    assert _image_reference_check(kind, "sha256:" + "a" * 64) is None


@pytest.mark.parametrize("budget", [0.1, 0.25])
def test_dry_run_task_views_shorten_all_solver_budgets_without_editing_formal_tasks(
    budget,
):
    root = ROOT
    config = DryRunConfig(solver_budget_sec=budget)
    for record in TaskCatalog(root).records():
        original = record.task.model_dump(mode="json")
        view = config.task_view(record.task)
        assert view.evaluation.budgets_sec == [budget]
        assert view.evaluation.primary_budget_sec == budget
        for panel in view.instance_sets:
            if panel.budgets_sec is not None:
                assert panel.budgets_sec == [budget]
                assert panel.primary_budget_sec == budget
        assert all(s.budget_sec == budget for s in view.metric_scenarios)
        if view.evaluation.controlled_transfer:
            assert all(
                c.budget_sec in (None, budget)
                for c in view.evaluation.controlled_transfer.conditions
            )
        assert record.task.model_dump(mode="json") == original
        assert (
            PitBenchTask.from_yaml(record.task_config_path).model_dump(mode="json")
            == original
        )
        assert view.evaluation.solver_seeds == record.task.evaluation.solver_seeds
        assert view.evaluation.seed_robustness == record.task.evaluation.seed_robustness


def test_dry_run_budget_is_configurable_and_must_be_positive_finite(tmp_path):
    from pydantic import ValidationError

    path = tmp_path / "evaluate.yaml"
    path.write_text("dry_run:\n  solver_budget_sec: 0.25\n")
    assert EvaluationConfig.from_yaml(path).dry_run.solver_budget_sec == 0.25
    assert EvaluationConfig().dry_run.solver_budget_sec == 0.1
    for invalid in [0, -1, float("inf"), float("nan")]:
        with pytest.raises(ValidationError):
            DryRunConfig(solver_budget_sec=invalid)


@pytest.mark.parametrize("external_task", [False, True])
def test_dry_run_cli_passes_empty_patch_and_saved_short_task_to_evaluator(
    tmp_path, external_task
):
    from typer.testing import CliRunner

    from pitbench.cli.main import app

    root = ROOT
    config = tmp_path / "evaluate.yaml"
    config.write_text(
        f"paths:\n  output_path: {tmp_path}/runs\n"
        f"  workspace_path: {tmp_path}/workspaces\n"
        "dry_run:\n  solver_budget_sec: 0.25\n"
    )
    original_task = root / "configs/tasks/vroom_v1_15_0.yaml"
    if external_task:
        external = tmp_path / "task-bundle" / "vroom_v1_15_0.yaml"
        external.parent.mkdir()
        external.write_bytes(original_task.read_bytes())
        original_task = external
        with config.open("a") as target:
            target.write(
                f"tasks:\n  vroom_v1_15_0:\n    task_config_path: {external}\n"
            )
    requests = []

    def evaluate(request):
        requests.append(request)
        assert request.candidate_patch_path.read_bytes() == b""
        task = PitBenchTask.from_yaml(
            Path(request.evaluator_config["task_config_path"])
        )
        assert task.evaluation.budgets_sec == [0.25]
        return SimpleNamespace(
            completed=True,
            payload={"summary": {}},
            model_dump=lambda **kwargs: {},
        )

    with (
        patch("pitbench.cli.main.PitBenchAdapter.materialize") as materialize,
        patch("pitbench.cli.main._prepare_task_image"),
        patch("pitbench.cli.main.PitBenchEvaluator.envelope", side_effect=evaluate),
        patch("pitbench.cli.main._limited_evaluation_failures", return_value=[]),
    ):
        result = CliRunner().invoke(
            app,
            [
                "dry-run",
                "vroom_v1_15_0",
                "--run-id",
                "short",
                "--config",
                str(config),
                "--root",
                str(root),
                "--no-configuration-robustness",
            ],
        )
    assert result.exit_code == 0, result.output
    assert len(requests) == 1
    assert materialize.call_args.kwargs["task_config_path"] == Path(
        requests[0].evaluator_config["task_config_path"]
    )
    assert "0.25s per solve" in result.output
    assert requests[0].evaluator_config["use_base_cache"] is False
    import json

    receipt = json.loads((requests[0].output_dir / "dry-run.json").read_text())
    assert receipt["original_task_config"] == str(original_task)


def doctor_context(
    repository_root: Path,
    *,
    clone_url: str = "https://github.com/PyVRP/PyVRP.git",
) -> SimpleNamespace:
    config = EvaluationConfig()
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
                release=SimpleNamespace(base_commit="pinned"),
                evaluation=SimpleNamespace(operational_reliability=True),
            )
        ),
    )


def test_pinned_source_rejects_broken_generated_cache(tmp_path: Path) -> None:
    context = doctor_context(tmp_path)
    source = PitBenchAdapter.source_cache_path(tmp_path, context.task.task)
    source.parent.mkdir(parents=True)
    source.symlink_to(tmp_path / "missing", target_is_directory=True)

    check = _pinned_source_check(context, {})

    assert check.status is CheckStatus.FAIL
    assert "does not exist" in check.detail


def test_pinned_source_rejects_modified_local_snapshot(tmp_path: Path) -> None:
    context = doctor_context(tmp_path)
    source = PitBenchAdapter.source_cache_path(tmp_path, context.task.task)
    source.mkdir(parents=True)
    with (
        patch("pitbench.cli.doctor.subprocess.check_output", return_value="pinned\n"),
        patch(
            "pitbench.cli.doctor.PitBenchAdapter.validate_repository",
            side_effect=ValueError("local modifications"),
        ),
    ):
        check = _pinned_source_check(context, {})

    assert check.status is CheckStatus.FAIL
    assert "local modifications" in check.detail


def test_pinned_source_reuses_generated_cache_without_remote_check(
    tmp_path: Path,
) -> None:
    context = doctor_context(tmp_path)
    source = PitBenchAdapter.source_cache_path(tmp_path, context.task.task)
    source.mkdir(parents=True)
    with (
        patch("pitbench.cli.doctor.subprocess.check_output", return_value="pinned\n"),
        patch("pitbench.cli.doctor.PitBenchAdapter.validate_repository") as validate,
        patch("pitbench.cli.doctor._github_source_access_check") as remote,
    ):
        check = _pinned_source_check(context, {})

    assert check.status is CheckStatus.PASS
    validate.assert_called_once_with(context.task.task, source)
    remote.assert_not_called()


def test_pinned_source_checks_github_with_global_transport(
    tmp_path: Path,
) -> None:
    context = doctor_context(tmp_path)

    def fake_run(command, **kwargs):
        if command[:3] == ["git", "config", "--global"]:
            return subprocess.CompletedProcess(command, 1, "", "")
        return subprocess.CompletedProcess(command, 0, "refs/heads/main\n", "")

    with patch("pitbench.cli.doctor.subprocess.run", side_effect=fake_run):
        check = _pinned_source_check(context, {})

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


def test_agent_runtime_reports_recorder_startup_failure():
    client = SimpleNamespace(
        containers=SimpleNamespace(
            run=Mock(side_effect=RuntimeError("OSError: No space left on device"))
        )
    )
    check = _agent_runtime_check("pitbench-agent:test", client)
    assert check.status is CheckStatus.FAIL
    assert "No space left on device" in check.detail


@pytest.mark.parametrize(
    "free_gib,status",
    [(2, CheckStatus.FAIL), (15, CheckStatus.WARN), (40, CheckStatus.PASS)],
)
def test_docker_storage_checks_daemon_filesystem(free_gib, status):
    with patch(
        "pitbench.cli.doctor.shutil.disk_usage",
        return_value=SimpleNamespace(free=free_gib * 1024**3),
    ) as usage:
        check = _docker_storage_check({"DockerRootDir": "/docker-on-system-disk"})
    usage.assert_called_once_with("/docker-on-system-disk")
    assert check.status is status


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
            get=Mock(
                return_value=SimpleNamespace(attrs={"Config": {"Env": environment}})
            )
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
    config.configuration_robustness.scope = "release"

    check = _configuration_robustness_check(tmp_path, config)

    assert check.status is CheckStatus.PASS


@pytest.mark.parametrize(
    "task_id,expected",
    [
        ("release_highs_v1_15_1", CheckStatus.PASS),
        ("release_cbc_v2_10_13", CheckStatus.WARN),
        ("release_scip_v10_1_0", CheckStatus.WARN),
        ("release_gecode_v6_4_0", CheckStatus.WARN),
    ],
)
def test_native_parameter_preflight_reports_real_coverage(task_id, expected):
    from pitbench.tasks import TaskCatalog

    root = ROOT
    task = TaskCatalog(root).validate_one(task_id).task
    config = EvaluationConfig()
    config.configuration_robustness.experiment_config = Path(
        "configs/experiments/configuration_robustness.yaml"
    )
    check = _configuration_robustness_check(root, config, task)
    assert check.status is expected
    if expected is CheckStatus.WARN:
        assert "unavailable" in check.detail


def test_default_native_parameter_panels_cover_all_eight_latest_tasks():
    import json

    from pitbench.tasks import TaskCatalog

    root = ROOT
    plan = json.loads(
        (root / "configs/research/latest_high_algorithm_space_v4.json").read_text()
    )
    config = EvaluationConfig()
    for task_path in plan["task_configs"]:
        task = TaskCatalog(root).validate_file(root / task_path).task
        assert (
            _configuration_robustness_check(root, config, task).status
            is CheckStatus.PASS
        )


def test_native_parameter_preflight_rejects_bad_domains_without_legacy_probes(
    tmp_path, monkeypatch
):
    path = tmp_path / "parameters.yaml"
    path.write_text(
        "panels:\n  - parameters:\n      effort: {type: real, bounds: [1, 0], default: 0.5}\n"
    )
    config = EvaluationConfig()
    config.configuration_robustness.experiment_config = path
    monkeypatch.setattr(
        "pitbench.evaluator.configuration.probe_collector",
        lambda *a, **kw: pytest.fail("historical collector was probed"),
    )
    checks = _configuration_collector_checks(tmp_path, config)
    assert len(checks) == 1 and checks[0].status is CheckStatus.FAIL
    assert "invalid parameter domain" in checks[0].detail


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
    config.configuration_robustness.scope = "release"
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
    root = ROOT

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

    assert failures == ["configuration search solver/budget-10 is baseline_unavailable"]


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


def test_preflight_validates_configured_evaluator_and_agent_views(
    tmp_path, monkeypatch
):
    import yaml

    from pitbench.cli.doctor import _load_context

    config_path = tmp_path / "evaluate.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "tasks": {
                    "task": {
                        "task_config_path": "evaluator.yaml",
                        "agent_task_config_path": "agent.yaml",
                    }
                }
            }
        )
    )
    calls = []

    class Catalog:
        def __init__(self, root):
            pass

        def validate_file(self, path):
            calls.append(path.name)
            if path.name == "agent.yaml":
                raise ValueError("public development inputs are missing")
            return SimpleNamespace(task=SimpleNamespace(task_id="task"))

    monkeypatch.setattr("pitbench.cli.doctor.TaskCatalog", Catalog)
    context, checks = _load_context(tmp_path, config_path, "task")
    assert calls == ["evaluator.yaml", "agent.yaml"]
    assert any(
        c.status is CheckStatus.FAIL and "public development inputs" in c.detail
        for c in checks
    )


def test_doctor_rejects_judge_without_declared_scenario_dependencies(tmp_path):
    context = doctor_context(tmp_path)
    context.task.task.repository.plugin = (
        "pitbench.repositories.plugins:PyVRPRepositoryPlugin"
    )
    context.task.task.metric_scenarios = [
        SimpleNamespace(representation="row_column_permutation")
    ]
    run = Mock(side_effect=RuntimeError("No module named 'numpy'"))
    client = SimpleNamespace(containers=SimpleNamespace(run=run))
    result = _judge_environment_check(context, "immutable-image", client)
    assert result.status is CheckStatus.FAIL
    assert "numpy" in result.detail
    assert run.call_args.kwargs["network_disabled"] is True
