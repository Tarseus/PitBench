"""Check pinned release tasks preserve the approved experimental design."""

import json
from pathlib import Path

import pytest
import yaml

from adapters.jvm.runner import _ortools_java_artifact
from pitbench.tasks import TaskCatalog

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = json.loads((ROOT / "configs/benchmarks/0.0.1.json").read_text())


def test_default_run_profile_needs_no_environment_variables(tmp_path, monkeypatch):
    from pitbench.cli.evaluate_config import EvaluationConfig
    from scripts import run_benchmark_release as campaign

    config = tmp_path / "evaluate.yaml"
    config.write_text(EvaluationConfig().model_dump_json())
    profile = tmp_path / "config/study.local.yaml"
    profile.parent.mkdir()
    profile.write_text(
        yaml.safe_dump(
            {
                "phase": "study-models",
                "manifest": str(ROOT / "configs/benchmarks/0.0.1.json"),
                "config": str(config),
                "output": str(tmp_path / "cohort"),
                "study_plan": str(
                    ROOT / "configs/research/generalization_latest_high_v2.json"
                ),
                "study_views": str(tmp_path / "views.json"),
                "workers": 1,
            }
        )
    )
    called = []
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    monkeypatch.setattr(campaign, "study_models", called.append)
    monkeypatch.setattr(campaign.sys, "argv", ["run_benchmark_release"])
    campaign.main()
    assert len(called) == 1
    assert called[0].config == config
    assert called[0].study_plan.name == "generalization_latest_high_v2.json"


def test_parameter_supplementation_keeps_original_cohort_and_never_runs_an_agent(
    tmp_path, monkeypatch
):
    import hashlib
    from types import SimpleNamespace

    from pitbench.cli.evaluate_config import EvaluationConfig
    from scripts import run_benchmark_release as campaign

    source = tmp_path / "original"
    source.mkdir()
    job = {
        "run_id": "run-1",
        "task_id": "task",
        "model": {"agent": "codex", "model": "frozen-model"},
        "arm": "standard",
    }
    campaign.write_json(source / "cohort.json", {"jobs": [job]})
    patch = source / "candidate.patch"
    patch.write_text("frozen source patch")
    campaign.write_json(
        source / "agent-status/run-1/status.json",
        {
            "status": "evaluation_completed",
            "audit": {
                "candidate_patch": str(patch),
                "candidate_patch_sha256": hashlib.sha256(
                    patch.read_bytes()
                ).hexdigest(),
            },
        },
    )
    task_path = source / "workspaces/run-1/task"
    task_path.mkdir(parents=True)
    (task_path / "task.yaml").write_text(
        yaml.safe_dump({"evaluator_config": {"judge_image": "frozen-image"}})
    )
    config = tmp_path / "evaluation.local.yaml"
    config.write_text(EvaluationConfig().model_dump_json())
    original = {
        p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()
    }
    requests = []

    def parameters(request, settings):
        requests.append(request)
        directory = request.output_dir / "parameters"
        campaign.write_json(
            directory / "experiment.json", {"searches": [{"id": "paired"}]}
        )
        campaign.write_json(
            directory / "configuration_report.json",
            {"searches": [{"id": "paired", "status": "completed"}]},
        )
        return directory

    monkeypatch.setattr(campaign, "run_candidate_configuration", parameters)
    monkeypatch.setattr(
        campaign, "PitBenchEvaluator", lambda: pytest.fail("native grid was restarted")
    )
    monkeypatch.setattr(
        campaign.subprocess, "Popen", lambda *a, **kw: pytest.fail("Agent was started")
    )
    output = tmp_path / "supplement"
    campaign.study_parameters(
        SimpleNamespace(
            source_cohort=source,
            output=output,
            config=config,
            evaluation_cpu_pools=["48-55"],
            evaluation_workers=1,
            task_id=None,
        )
    )
    assert len(requests) == 1 and requests[0].candidate_patch_path == patch
    assert (
        json.loads((output / "parameter-results.json").read_text())[0]["status"]
        == "completed"
    )
    assert original == {
        p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize(
    "completed,gaps",
    [(True, []), (False, []), (True, ["missing scenario observations"])],
)
@pytest.mark.parametrize("parameter_enabled", [False, True])
def test_captured_patch_queue_uses_native_evaluator_without_new_agent(
    tmp_path, monkeypatch, completed, gaps, parameter_enabled
):
    import hashlib
    from types import SimpleNamespace

    from pitbench.harness.evaluation import EvaluationEnvelope
    from scripts import run_benchmark_release as campaign

    run_id, task_id = "run-1", "task"
    patch = tmp_path / "runs" / run_id / task_id / "trial/evaluation/candidate.patch"
    patch.parent.mkdir(parents=True)
    patch.write_text("captured source patch")
    task_path = tmp_path / "workspaces" / run_id / task_id
    task_path.mkdir(parents=True)
    (task_path / "task.yaml").write_text(
        yaml.safe_dump(
            {
                "evaluator_config": {
                    "task_config_path": str(
                        ROOT / "configs/tasks/release_cbc_v2_10_13.yaml"
                    ),
                    "judge_cpus": 8,
                    "judge_memory": "32g",
                    "judge_parallel_runs": 4,
                }
            }
        )
    )
    campaign.write_json(
        tmp_path / "runs" / run_id / "results.json",
        {"results": [{"task_id": task_id, "trial_name": "trial"}]},
    )
    calls = []

    def envelope(request):
        calls.append(request)
        return EvaluationEnvelope(
            evaluator="pitbench",
            evaluator_version="test",
            completed=completed,
            payload={},
            error=None if completed else "retained failure",
        )

    monkeypatch.setattr(
        campaign, "PitBenchEvaluator", lambda: SimpleNamespace(envelope=envelope)
    )
    monkeypatch.setattr(campaign, "_evaluation_metric_failures", lambda *a, **k: gaps)
    monkeypatch.setattr(
        campaign.subprocess, "Popen", lambda *a, **k: pytest.fail("new Agent")
    )
    state = {
        "run_id": run_id,
        "task_id": task_id,
        "model": {"agent": "codex", "model": "requested-model"},
        "audit": {
            "candidate_patch": str(patch),
            "candidate_patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
        },
    }
    from pitbench.cli.evaluate_config import EvaluationConfig

    configuration = EvaluationConfig()
    configuration.configuration_robustness.enabled = parameter_enabled
    config_path = tmp_path / "evaluation.local.yaml"
    config_path.write_text(configuration.model_dump_json())
    parameter_calls = []

    def parameters(request, settings):
        if not settings.enabled:
            return None
        parameter_calls.append(request)
        directory = patch.parent / "parameters"
        campaign.write_json(
            directory / "experiment.json", {"searches": [{"id": "paired"}]}
        )
        campaign.write_json(
            directory / "configuration_report.json",
            {"searches": [{"id": "paired", "status": "completed"}]},
        )
        return directory

    monkeypatch.setattr(campaign, "run_candidate_configuration", parameters)
    result = campaign.evaluate_study_candidate(
        state, SimpleNamespace(output=tmp_path, config=config_path), "48-55"
    )
    assert result["status"] == (
        ("evaluation_incomplete" if gaps else "evaluation_completed")
        if completed
        else "evaluation_failed"
    )
    assert result["metric_gaps"] == gaps
    assert len(calls) == 1 and calls[0].candidate_patch_path == patch
    assert calls[0].evaluator_config["judge_cpuset_cpus"] == "48-55"
    assert calls[0].evaluator_config["judge_parallel_runs"] == 4
    saved = json.loads((tmp_path / "runs" / run_id / "results.json").read_text())
    assert saved["results"][0]["evaluation"]["completed"] is completed
    assert patch.read_text() == "captured source patch"
    assert len(parameter_calls) == int(completed and parameter_enabled)
    if completed and parameter_enabled:
        assert (
            saved["results"][0]["evaluation"]["payload"]["summary"][
                "configuration_robustness"
            ]["searches"][0]["status"]
            == "completed"
        )


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
    judge.repository = SimpleNamespace(
        build_commands=lambda _: [command], agent_build_paths=()
    )
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


@pytest.mark.parametrize("has_patch", [False, True])
def test_deferred_model_audit_requires_captured_patch_and_model_execution(
    tmp_path, has_patch
):
    from scripts import run_benchmark_release as campaign

    task_id = "example"
    campaign.write_json(
        tmp_path / "results.json",
        {
            "results": [
                {
                    "task_id": task_id,
                    "trial_name": "trial",
                    "failure_mode": "agent_timeout",
                    "total_input_tokens": 17,
                }
            ]
        },
    )
    candidate = tmp_path / task_id / "trial/evaluation/candidate.patch"
    if has_patch:
        candidate.parent.mkdir(parents=True)
        candidate.write_bytes(b"")
    audit = campaign.audit_model_run(tmp_path, task_id, evaluation_required=False)
    assert audit["complete"] is has_patch
    assert not audit["retryable"]
    if has_patch:
        # An empty final patch is retained; it is not a reason for a new Agent.
        assert audit["evaluation_pending"]
        assert audit["candidate_patch"] == str(candidate)
    assert not campaign.audit_model_run(tmp_path, task_id)["complete"]


@pytest.mark.parametrize(
    "plan_name",
    [
        "generalization_latest_high_v2.json",
        "latest_high_replicates_v3.json",
        "latest_high_algorithm_space_v4.json",
    ],
)
def test_study_prepare_selects_only_planned_tasks_without_starting_agents(
    tmp_path, monkeypatch, plan_name
):
    from types import SimpleNamespace

    from pitbench.cli.evaluate_config import EvaluationConfig
    from scripts import run_benchmark_release as campaign

    plan_path = ROOT / "configs/research" / plan_name
    plan = json.loads(plan_path.read_text())
    plan["evaluation_seed_policy"] = None
    plan["task_evaluation_seed_policies"] = {}
    plan["evaluation_run_target"] = None
    plan["supplemental_seed_count"] = None
    plan["supplemental_primary_budget_only"] = False
    plan_path = tmp_path / "selection-plan.json"
    plan_path.write_text(json.dumps(plan))
    selected = [Path(path).stem for path in plan["task_configs"]]
    latest = {
        row["task_id"] for row in MANIFEST["task_matrix"] if row["role"] == "latest"
    }
    assert set(selected) <= latest
    assert len(selected) == (
        8 if plan.get("comparison_mode") == "agent_augmentation" else 3
    )
    records = {task: TaskCatalog(ROOT).validate_one(task) for task in selected}
    validated = []

    class Catalog:
        def __init__(self, root):
            pass

        def validate_file(self, path):
            validated.append(path.stem)
            return records[path.stem]

        def validate_one(self, task_id):
            return records[task_id]

    def no_model_process(*args, **kwargs):
        raise AssertionError("prepare-only must not start an Agent")

    monkeypatch.setattr(campaign, "TaskCatalog", Catalog)
    monkeypatch.setattr(campaign.subprocess, "Popen", no_model_process)
    views = {
        "status": "development_frozen_evaluation_gated",
        "tasks": {"unselected": {"evaluator_task_config": "missing.yaml"}},
        "arms": {arm["id"]: {} for arm in plan["arms"]},
    }
    import hashlib

    from tests.unit.pitbench.test_scenario_metrics import protocol

    private = tmp_path / "private"
    private.mkdir()
    for task_id, record in records.items():
        if (
            plan.get("comparison_mode") == "agent_augmentation"
            and task_id == selected[-1]
        ):
            views["tasks"][task_id] = {
                "evaluator_task_config": str(record.task_config_path)
            }
            public = tmp_path / "public" / (task_id + ".yaml")
            public.parent.mkdir(exist_ok=True)
            public.write_bytes(record.task_config_path.read_bytes())
            for arm in plan["arms"]:
                views["arms"][arm["id"]][task_id] = {
                    "agent_task_config_path": str(public)
                }
            continue
        scenario = tmp_path / (task_id + ".scenario.json")
        model = private / (task_id + ".json")
        model.write_text("frozen model input")
        digest = hashlib.sha256(model.read_bytes()).hexdigest()
        p = protocol("setup")
        p.model_sequences_sha256 = {"judge_id": {"one": [digest]}}
        record.task.metric_scenarios = [p]
        next(
            panel for panel in record.task.instance_sets if panel.name == "judge_id"
        ).size = 1
        scenario.write_text(
            json.dumps(
                {
                    "source_build": {"source_commit": record.task.release.base_commit},
                    "instances": {
                        "judge_id": [
                            {
                                "id": "one",
                                "models_by_scenario": {
                                    p.id: ["private://" + model.name]
                                },
                            }
                        ]
                    },
                }
            )
        )
        views["tasks"][task_id] = {
            "evaluator_task_config": str(record.task_config_path),
            "metric_scenario_config": str(scenario),
        }
        for arm in plan["arms"]:
            views["arms"][arm["id"]][task_id] = {
                "agent_task_config_path": str(record.task_config_path)
            }
    views_path = tmp_path / "views.json"
    views_path.write_text(json.dumps(views))
    config = tmp_path / "config.yaml"
    config.write_text(
        EvaluationConfig(paths={"private_root": private}).model_dump_json()
    )
    args = SimpleNamespace(
        study_plan=plan_path,
        study_views=views_path,
        config=config,
        output=tmp_path / "cohort",
        schedule_seed=20261101,
        prepare_only=True,
    )
    campaign.study_models(args)
    frozen = json.loads((args.output / "cohort.json").read_text())
    assert (
        len(frozen["jobs"])
        == len(selected) * len(plan["arms"]) * 2 * plan["agent_attempts_per_condition"]
    )
    assert set(validated) == set(selected)
    assert {row["model"]["label"] for row in frozen["jobs"]} == {
        "sol-high",
        "terra-high",
    }
    if plan.get("comparison_mode") == "model_repeats":
        assert {row["arm"] for row in frozen["jobs"]} == {"standard"}
        assert {row["attempt"] for row in frozen["jobs"]} == {1, 2, 3}
        assert frozen["agent_attempts_per_condition"] == 3
        for task_id in selected:
            for model in ("gpt-6.1-sol", "gpt-6.1-terra"):
                assert (
                    len(
                        [
                            row
                            for row in frozen["jobs"]
                            if row["task_id"] == task_id
                            and row["model"]["model"] == model
                        ]
                    )
                    == 3
                )
    elif plan.get("comparison_mode") == "agent_augmentation":
        assert len(frozen["jobs"]) == 96
        assert (
            str(tmp_path / "public" / (selected[-1] + ".yaml"))
            in frozen["input_files_sha256"]
        )
        for task_id in selected:
            for model in ("gpt-6.1-sol", "gpt-6.1-terra"):
                for arm in plan["arms"]:
                    rows = [
                        j
                        for j in frozen["jobs"]
                        if j["task_id"] == task_id
                        and j["model"]["model"] == model
                        and j["arm"] == arm["id"]
                    ]
                    assert {j["attempt"] for j in rows} == {1, 2, 3}
                    assert all(
                        j.get("agent_import_path") == arm.get("agent_import_path")
                        for j in rows
                    )
    else:
        assert all(row["attempt"] == 1 for row in frozen["jobs"])
    assert all(
        row["hypotheses_required"] == (row["arm"] == "mechanism_aware")
        for row in frozen["jobs"]
    )
    assert not frozen["scientific_evaluation_ready"]
    del views["tasks"][selected[0]]
    views_path.write_text(json.dumps(views))
    with pytest.raises(ValueError, match="absent from the frozen views"):
        campaign.study_models(args)


def test_study_custom_agents_share_options_and_use_existing_import_interface():
    from pitbench.cli.evaluate_config import EvaluationConfig
    from pitbench.schema.research import ResearchBudget
    from scripts import run_benchmark_release as campaign

    job = {"model": {"agent": "codex"}, "agent_import_path": "example:EnhancedAgent"}
    config = EvaluationConfig(
        agents={
            "codex": {"runner_backend": "workspace", "proxy_url": "http://localhost:1"}
        }
    )
    campaign.configure_study_agent(config, job, ResearchBudget(wall_time_sec=14400))
    assert config.agents["example:EnhancedAgent"] == {
        "runner_backend": "workspace",
        "proxy_url": "http://localhost:1",
        "timeout_sec": 14400,
    }
    assert campaign.study_agent_arguments(job) == [
        "--agent-import-path",
        "example:EnhancedAgent",
    ]
    assert campaign.study_agent_arguments({"model": {"agent": "codex"}}) == [
        "--agent",
        "codex",
    ]


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


def test_restarted_evaluation_archives_prior_attempt_and_clears_live_completion(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from scripts import run_benchmark_release as campaign

    status_path = tmp_path / "agent-status/run/status.json"
    status_path.parent.mkdir(parents=True)
    previous = {
        "run_id": "run",
        "task_id": "task",
        "model": {"agent": "codex", "model": "model"},
        "audit": {"candidate_patch": str(tmp_path / "candidate.patch")},
        "evaluation_started_at": "old start",
        "evaluation_finished_at": "old finish",
        "evaluation_error": "old error",
        "metrics_complete": True,
        "metric_gaps": ["old gap"],
    }
    status_path.write_text(json.dumps(previous))
    writes = campaign.write_json

    def inspect(path, value):
        if path == status_path and value["status"] == "evaluating":
            assert value["evaluation_error"] is None
            assert value["evaluation_finished_at"] is None
            assert value["metrics_complete"] is False
            assert value["metric_gaps"] == []
        writes(path, value)

    monkeypatch.setattr(campaign, "write_json", inspect)
    result = campaign.evaluate_study_candidate(
        previous, SimpleNamespace(output=tmp_path), "0-7"
    )
    # The intentionally absent captured task is a new failure, never a leftover success.
    assert result["status"] == "evaluation_failed"
    assert result["metrics_complete"] is False
    assert result["evaluation_error"] != "old error"
    (archived,) = status_path.parent.glob("evaluation-attempts/*.json")
    assert json.loads(archived.read_text()) == previous
