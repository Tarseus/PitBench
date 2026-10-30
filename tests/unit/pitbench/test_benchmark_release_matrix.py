"""Check pinned release tasks preserve the approved experimental design."""

import json
from pathlib import Path

import pytest
import yaml

from adapters.jvm.runner import _ortools_java_artifact
from pitbench.tasks import TaskCatalog

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = json.loads((ROOT / "configs/benchmarks/0.0.1.json").read_text())


@pytest.mark.parametrize("row", MANIFEST["task_matrix"], ids=lambda row: row["task_id"])
def test_release_task_matches_pin_and_preserves_template_design(row):
    catalog = TaskCatalog(ROOT)
    task = catalog.validate_one(row["task_id"]).task
    template = catalog.validate_one(row["template_task_id"]).task
    selected = MANIFEST["selected_releases"][row["repository"]][row["role"]]
    assert task.release.version == selected["version"]
    assert task.release.base_commit == selected["commit_sha"]
    assert task.release.tree_sha == selected["tree_sha"]
    for field in ("instance_sets", "metric_scenarios"):
        assert getattr(task, field) == getattr(template, field)
    evaluation = task.evaluation.model_dump()
    original_evaluation = template.evaluation.model_dump()
    for payload in (evaluation, original_evaluation):
        if payload.get("seed_robustness"):
            payload["seed_robustness"].pop("evaluation_seeds_file")
            payload["seed_robustness"].pop("evaluation_seeds_file_sha256")
    assert evaluation == original_evaluation
    oracle = task.oracle.model_dump(exclude={"source", "source_sha256"})
    assert oracle == template.oracle.model_dump(exclude={"source", "source_sha256"})
    assert task.repository.plugin == row.get(
        "repository_plugin", template.repository.plugin
    )


@pytest.mark.external
@pytest.mark.parametrize("row", MANIFEST["task_matrix"], ids=lambda row: row["task_id"])
def test_release_private_assets_preserve_seeds_and_reference_records(row, tmp_path):
    from pitbench.evaluator.judge import JudgePlan
    from pitbench.evaluator.private_assets import (
        PrivateAssetResolver,
        load_private_seed_robustness_config,
    )

    catalog = TaskCatalog(ROOT)
    task = catalog.validate_one(row["task_id"]).task
    template = catalog.validate_one(row["template_task_id"]).task
    resolver = PrivateAssetResolver(ROOT / "private")
    if task.evaluation.seed_robustness:
        selected = load_private_seed_robustness_config(
            resolver,
            task_id=task.task_id,
            public_config=task.evaluation.seed_robustness,
        )
        original = load_private_seed_robustness_config(
            resolver,
            task_id=template.task_id,
            public_config=template.evaluation.seed_robustness,
        )
        assert selected.evaluation_seeds == original.evaluation_seeds
    if task.task_type.value == "exact_solver":
        selected = yaml.safe_load(
            resolver.resolve(task.oracle.source, task.oracle.source_sha256).read_text()
        )
        original = yaml.safe_load(
            resolver.resolve(
                template.oracle.source, template.oracle.source_sha256
            ).read_text()
        )
        assert selected.pop("task_id") == task.task_id
        original.pop("task_id")
        assert selected == original
    plan = JudgePlan.from_instance_set_configs(
        task, resolver, public_root=ROOT, generated_root=tmp_path
    )
    assert plan.cases


@pytest.mark.parametrize("version", ["9.14.9999", "9.15.9999"])
def test_java_runtime_artifact_selection_accepts_both_official_lines(tmp_path, version):
    target = tmp_path / "build/java/ortools-java/target"
    target.mkdir(parents=True)
    runtime = target / f"ortools-java-{version}.jar"
    runtime.touch()
    (target / f"ortools-java-{version}-sources.jar").touch()
    (target / f"ortools-java-{version}-javadoc.jar").touch()
    assert _ortools_java_artifact(tmp_path, "ortools-java") == runtime
    (target / "ortools-java-unexpected.jar").touch()
    with pytest.raises(RuntimeError, match="expected one"):
        _ortools_java_artifact(tmp_path, "ortools-java")


def test_release_model_matrix_keeps_the_two_requested_efforts():
    matrix = yaml.safe_load(
        (ROOT / "configs/experiments/gpt_6_1_matrix.yaml").read_text()
    )
    assert [(row["model"], row["setting"]) for row in matrix["runs"]] == [
        ("gpt-6.1-sol", "reasoning_effort=xhigh"),
        ("gpt-6.1-sol", "reasoning_effort=high"),
    ]


def test_model_runs_require_the_complete_real_baseline(tmp_path):
    from types import SimpleNamespace

    from scripts.run_benchmark_release import models

    args = SimpleNamespace(output=tmp_path)
    task_ids = [row["task_id"] for row in MANIFEST["task_matrix"]]
    with pytest.raises(ValueError, match="missing"):
        models(task_ids, args)
    (tmp_path / "baseline.json").write_text(
        json.dumps({task_ids[0]: {"status": "passed"}})
    )
    with pytest.raises(ValueError, match="every manifest task"):
        models(task_ids, args)
    evidence = {
        task_id: {"status": "passed", "identity": {"task_config_sha256": "stale"}}
        for task_id in task_ids
    }
    (tmp_path / "baseline.json").write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="changed after"):
        models(task_ids, args)


def test_materialized_judge_receives_the_selected_cpu_pool(tmp_path):
    from adapters.pitbench.adapter import PitBenchAdapter
    from pitbench.cli.evaluate_config import EvaluationConfig

    task = TaskCatalog(ROOT).validate_one(MANIFEST["task_matrix"][0]["task_id"]).task
    config = EvaluationConfig.model_validate(
        {
            "agent_cpuset_cpus": "16-31",
            "tasks": {task.task_id: {"judge_cpuset_cpus": "16-31"}},
        }
    )
    assert (
        config.agent_cpuset_cpus == config.resources_for(task.task_id).judge_cpuset_cpus
    )
    PitBenchAdapter(ROOT, tmp_path / "private")._write_task_yaml(
        task,
        ROOT / "configs/tasks" / f"{task.task_id}.yaml",
        tmp_path / "source",
        tmp_path,
        judge_cpuset_cpus=config.resources_for(task.task_id).judge_cpuset_cpus,
    )
    materialized = yaml.safe_load((tmp_path / "task.yaml").read_text())
    assert materialized["evaluator_config"]["judge_cpuset_cpus"] == "16-31"


def test_source_build_keeps_a_named_agent_image_and_immutable_judge(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from pitbench.cli.evaluate_config import EvaluationConfig
    from scripts import run_benchmark_release as campaign

    args = SimpleNamespace(
        output=tmp_path,
        config=tmp_path / "evaluate.local.yaml",
        cpu_pools=[],
        workers=1,
        retry_errors=False,
    )
    monkeypatch.setattr(
        campaign,
        "build",
        lambda *_: {
            "image_tag": "pitbench-source:fixed-commit",
            "image_id": "sha256:" + "a" * 64,
        },
    )
    commands = []
    monkeypatch.setattr(
        campaign.subprocess, "run", lambda command, **_: commands.append(command)
    )
    task_id = MANIFEST["task_matrix"][0]["task_id"]
    config = EvaluationConfig.model_validate({"tasks": {task_id: {"judge_cpus": 7.0}}})
    assert campaign.run_phase("build", [task_id], args, config)
    saved = EvaluationConfig.from_yaml(args.config).resources_for(task_id)
    assert saved.agent_image == "pitbench-agent:sha256-" + "a" * 64
    assert saved.judge_image == "sha256:" + "a" * 64
    assert saved.judge_cpus == 7.0
    assert commands == [
        ["docker", "image", "tag", saved.judge_image, saved.agent_image]
    ]


def test_failed_build_retains_full_stdout_even_when_stderr_is_present(tmp_path):
    import sys
    from types import SimpleNamespace

    from pitbench.evaluator.judge import LocalProcessJudge
    from pitbench.repositories.base import BuildKind, CommandSpec
    from pitbench.schema.observation import CodeState

    source = tmp_path / "source"
    source.mkdir()
    judge = object.__new__(LocalProcessJudge)
    judge.base_repository = source
    judge.validation_test_patch = None
    judge.output_dir = tmp_path / "output"
    command = CommandSpec(
        argv=[
            sys.executable,
            "-c",
            'import sys; print("complete compiler diagnostic"); print("brief wrapper error", file=sys.stderr); sys.exit(1)',
        ]
    )
    judge.repository = SimpleNamespace(build_commands=lambda _: [command])
    with pytest.raises(RuntimeError, match="base validation build failed"):
        judge._workspace(CodeState.BASE, tmp_path / "work", BuildKind.VALIDATION)
    logs = list(judge.output_dir.glob("build-logs/validation/base/*"))
    assert len(logs) == 1
    assert (logs[0] / "stdout.txt").read_text() == "complete compiler diagnostic\n"
    assert (logs[0] / "stderr.txt").read_text() == "brief wrapper error\n"
    assert json.loads((logs[0] / "status.json").read_text()) == {"returncode": 1}


def test_maven_settings_records_the_build_proxy(tmp_path):
    import xml.etree.ElementTree as ET

    from adapters.jvm.runner import write_maven_settings

    output = tmp_path / "settings.xml"
    write_maven_settings(output, "http://127.0.0.1:17893")
    proxy = ET.parse(output).find("proxies/proxy")
    assert proxy.findtext("active") == "true"
    assert proxy.findtext("protocol") == "http"
    assert proxy.findtext("host") == "127.0.0.1"
    assert proxy.findtext("port") == "17893"
    write_maven_settings(output, None)
    assert ET.parse(output).find("proxies") is None


def test_model_settings_share_one_queue_and_keep_separate_trials(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace

    from pitbench.cli.evaluate_config import EvaluationConfig
    from scripts import run_benchmark_release as campaign

    task_id = MANIFEST["task_matrix"][0]["task_id"]
    record = TaskCatalog(ROOT).validate_one(task_id)
    campaign.write_json(
        tmp_path / "baseline.json",
        {
            task_id: {
                "status": "passed",
                "identity": {"task_config_sha256": record.task_config_sha256},
            }
        },
    )
    config = tmp_path / "evaluate.local.yaml"
    config.write_text(
        yaml.safe_dump(
            EvaluationConfig.model_validate(
                {
                    "tasks": {task_id: {}},
                    "paths": {"output_path": str(tmp_path / "runs")},
                }
            ).model_dump(mode="json")
        )
    )
    args = SimpleNamespace(
        output=tmp_path,
        config=config,
        models=ROOT / "configs/experiments/gpt_6_1_matrix.yaml",
        workers=2,
        cpu_pools=["0-7", "8-15"],
    )
    high_started = threading.Event()

    def run(command, log):
        setting = command[command.index("--agent-kwarg") + 1]
        if setting == "reasoning_effort=high":
            high_started.set()
        else:
            assert high_started.wait(3), (
                "second setting was blocked behind a model-wide barrier"
            )
        run_id = command[command.index("--run-id") + 1]
        campaign.write_json(
            tmp_path / "runs" / run_id / "results.json",
            {
                "results": [
                    {
                        "task_id": task_id,
                        "trial_name": "trial",
                        "failure_mode": "unset",
                        "total_input_tokens": 10,
                        "total_output_tokens": 2,
                        "evaluation": {
                            "completed": True,
                            "payload": {"validity": {"accepted": True}},
                        },
                    }
                ]
            },
        )

    monkeypatch.setattr(campaign, "checked_run", run)
    campaign.models([task_id], args)
    results = json.loads((tmp_path / "model-runs.json").read_text())
    assert len(results) == 2
    assert {row["setting"] for row in results} == {
        "reasoning_effort=high",
        "reasoning_effort=xhigh",
    }
    assert len({row["run_id"] for row in results}) == 2
    assert {row["cpu_pool"] for row in results} == {"0-7", "8-15"}
    assert all(row["status"] == "command_completed" for row in results)


def test_host_pressure_delays_dispatch_until_the_reserve_recovers(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from scripts import run_benchmark_release as campaign

    campaign.write_json(
        tmp_path / "resource-policy.json",
        {
            "min_available_memory_gib": 64,
            "max_cpu_temperature_c": 86,
            "max_load_1min": 112,
            "check_every_sec": 10,
        },
    )
    states = iter(
        [
            {"available_memory_gib": 32, "cpu_temperature_c": 88, "load_1min": 120},
            {"available_memory_gib": 128, "cpu_temperature_c": 70, "load_1min": 50},
        ]
    )
    monkeypatch.setattr(campaign, "host_snapshot", lambda: next(states))
    sleeps = []
    monkeypatch.setattr(campaign.time, "sleep", sleeps.append)
    campaign.wait_for_capacity(SimpleNamespace(output=tmp_path))
    assert sleeps == [10]
    record = json.loads((tmp_path / "admission-waits.jsonl").read_text())
    assert record["reasons"] == ["memory reserve", "CPU temperature", "host CPU load"]


def test_model_audit_rejects_rate_limited_zero_token_cli_success(tmp_path):
    from scripts import run_benchmark_release as campaign

    run = tmp_path / "run"
    task_id = "example"
    trial = "trial"
    campaign.write_json(
        run / "results.json",
        {
            "results": [
                {
                    "task_id": task_id,
                    "trial_name": trial,
                    "failure_mode": "unknown_agent_error",
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "evaluation": {"completed": True},
                }
            ]
        },
    )
    log = run / task_id / trial / "agent-logs/codex.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text(
        json.dumps(
            {"type": "turn.failed", "error": {"message": "429 Too Many Requests"}}
        )
        + "\n"
    )
    audit = campaign.audit_model_run(run, task_id)
    assert not audit["complete"] and audit["retryable"]
    assert "rate limited" in audit["reason"]


def test_model_resume_preserves_completed_work_and_archives_infra_attempt(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from pitbench.cli.evaluate_config import EvaluationConfig
    from scripts import run_benchmark_release as campaign

    task_id = MANIFEST["task_matrix"][0]["task_id"]
    record = TaskCatalog(ROOT).validate_one(task_id)
    campaign.write_json(
        tmp_path / "baseline.json",
        {
            task_id: {
                "status": "passed",
                "identity": {"task_config_sha256": record.task_config_sha256},
            }
        },
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            EvaluationConfig.model_validate(
                {
                    "tasks": {task_id: {}},
                    "paths": {
                        "output_path": str(tmp_path / "runs"),
                        "workspace_path": str(tmp_path / "workspaces"),
                    },
                }
            ).model_dump(mode="json")
        )
    )
    matrix = ROOT / "configs/experiments/gpt_6_1_matrix.yaml"
    rows = yaml.safe_load(matrix.read_text())["runs"]
    calls = []
    for index, row in enumerate(rows):
        rid = f"{tmp_path.name}-{row['label']}-{task_id}"
        campaign.write_json(
            tmp_path / "model-status" / rid / "status.json",
            {
                "run_id": rid,
                "task_id": task_id,
                "status": "command_completed" if index == 0 else "command_failed",
            },
        )
        if index == 0:
            campaign.write_json(
                tmp_path / "runs" / rid / "results.json",
                {
                    "results": [
                        {
                            "task_id": task_id,
                            "trial_name": "trial",
                            "failure_mode": "unset",
                            "total_input_tokens": 5,
                            "evaluation": {"completed": True},
                        }
                    ]
                },
            )
        else:
            (tmp_path / "workspaces" / rid).mkdir(parents=True)

    def run(command, log):
        calls.append(command)
        rid = command[command.index("--run-id") + 1]
        campaign.write_json(
            tmp_path / "runs" / rid / "results.json",
            {
                "results": [
                    {
                        "task_id": task_id,
                        "trial_name": "trial",
                        "failure_mode": "unset",
                        "total_input_tokens": 5,
                        "evaluation": {"completed": True},
                    }
                ]
            },
        )

    monkeypatch.setattr(campaign, "checked_run", run)
    args = SimpleNamespace(
        output=tmp_path,
        config=config,
        models=matrix,
        workers=1,
        cpu_pools=["0-7"],
        resume_models=True,
        dispatch_interval_sec=0,
    )
    campaign.models([task_id], args)
    assert len(calls) == 1 and calls[0][calls[0].index("--run-id") + 1].endswith(
        "-infra2"
    )
    result = json.loads((tmp_path / "model-runs.json").read_text())
    assert len(result) == 2 and all(r["status"] == "command_completed" for r in result)
    assert len(list((tmp_path / "model-status").glob("*/attempts/*.json"))) == 1
