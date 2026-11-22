"""Check pinned release tasks preserve the approved experimental design."""

import json
from pathlib import Path

import pytest
import yaml

from adapters.jvm.runner import _ortools_java_artifact
from pitbench.tasks import TaskCatalog
from tests.support import ROOT

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
            evaluation_cpu_pools=[",".join(map(str, sorted(campaign.os.sched_getaffinity(0))[:8]))],
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
    "completed,gaps,extension_gaps",
    [
        (True, [], []),
        (False, [], []),
        (True, ["missing primary observations"], []),
        (True, [], ["missing scenario observations"]),
    ],
)
@pytest.mark.parametrize("parameter_enabled", [False, True])
def test_captured_patch_queue_uses_native_evaluator_without_new_agent(
    tmp_path, monkeypatch, completed, gaps, extension_gaps, parameter_enabled
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
    def metric_failures(*args, extension_failures):
        extension_failures.extend(extension_gaps)
        return gaps

    monkeypatch.setattr(campaign, "_evaluation_metric_failures", metric_failures)
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
    result = campaign.evaluate_study_candidate(
        state, SimpleNamespace(output=tmp_path, config=config_path), "48-55"
    )
    assert result["status"] == (
        ("evaluation_incomplete" if gaps else "evaluation_completed")
        if completed
        else "evaluation_failed"
    )
    assert result["primary_metric_gaps"] == gaps
    assert result["extension_metric_gaps"] == extension_gaps
    assert result["metric_gaps"] == [*gaps, *extension_gaps]
    if extension_gaps:
        assert result["primary_metrics_status"] == "complete"
        assert result["extension_metrics_status"] == "incomplete"
    assert len(calls) == 1
    assert calls[0].candidate_patch_path != patch
    assert calls[0].candidate_patch_path.read_bytes() == patch.read_bytes()
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


def test_model_audit_marks_codex_relay_network_failure_retryable(tmp_path):
    from scripts import run_benchmark_release as campaign

    run = tmp_path / "run"
    task_id, trial = "example", "trial"
    campaign.write_json(run / "results.json", {"results": [{
        "task_id": task_id, "trial_name": trial,
        "failure_mode": "unknown_agent_error", "total_input_tokens": 0,
        "total_output_tokens": 0,
    }]})
    trace_dir = run / task_id / trial / "agent-trace" / "execution"
    detail = "objects/rejection.json"
    campaign.write_json(trace_dir / detail, {
        "status": 502,
        "message": "model relay failed: <urlopen error [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol>",
    })
    campaign.write_json(trace_dir / "events.jsonl", {})
    (trace_dir / "events.jsonl").write_text(json.dumps({
        "event": "model.rejected", "data": {"path": detail}
    }) + "\n")

    audit = campaign.audit_model_run(run, task_id, evaluation_required=False)
    assert audit["retryable"] and audit["tokens"] == 0
    assert campaign.incomplete_condition_retry_reason("development_failed", audit) == "codex_relay_network_unavailable"


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


def test_timeout_cleanup_trace_race_is_audited_for_fresh_recovery(tmp_path):
    from scripts import run_benchmark_release as campaign

    run = tmp_path / "run"
    campaign.write_json(
        run / "results.json",
        {
            "results": [
                {
                    "task_id": "example",
                    "trial_name": "trial",
                    "failure_mode": "unknown_agent_error",
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                }
            ]
        },
    )
    (run / "pipeline_trace.jsonl").write_text(
        json.dumps(
            {
                "stage": "terminal.lifecycle",
                "status": "failed",
                "error": {
                    "type": "RuntimeError",
                    "message": "agent trace is already closed",
                },
            }
        )
        + "\n"
    )

    audit = campaign.audit_model_run(run, "example", evaluation_required=False)
    assert audit["retryable"]
    assert audit["trace_closed_during_cleanup"]
    assert (
        campaign.incomplete_condition_retry_reason("development_failed", audit)
        == "trace_close_race"
    )


@pytest.mark.parametrize(
    ("status", "audit", "expected"),
    [
        (
            "development_failed",
            {
                "failure_mode": "agent_timeout",
                "complete": False,
                "evaluation_pending": False,
                "completed_turn": False,
            },
            "incomplete_agent_timeout",
        ),
        (
            "development_failed",
            {
                "reason": "model stream disconnected; retain partial infrastructure-failed attempt",
                "tokens": 0,
                "completed_turn": False,
            },
            "stream_disconnect",
        ),
        (
            "development_failed",
            {
                "failure_mode": "agent_timeout",
                "complete": True,
                "evaluation_pending": True,
                "completed_turn": True,
            },
            None,
        ),
        (
            "evaluation_pending",
            {
                "failure_mode": "agent_timeout",
                "complete": False,
                "evaluation_pending": False,
                "completed_turn": False,
            },
            None,
        ),
    ],
)
def test_failed_incomplete_condition_recovery_is_limited_to_missing_turns(
    status, audit, expected
):
    from scripts import run_benchmark_release as campaign

    assert campaign.incomplete_condition_retry_reason(status, audit) == expected


def test_preflight_recovery_requires_explicit_no_agent_evidence(tmp_path):
    from scripts import run_benchmark_release as campaign

    log = tmp_path / "controller.log"
    audit = {"reason": "no terminal harness result", "retryable": True}
    assert campaign.preflight_retry_reason("development_failed", audit, log) is None
    log.write_text("NOT READY: no agent was started.\n")
    assert campaign.preflight_retry_reason("development_failed", audit, log) == "preflight_agent_not_started"
    assert campaign.preflight_retry_reason("development_failed", {**audit, "retryable": False}, log) is None


@pytest.mark.parametrize("tokens, retryable", [(0, True), (5, False)])
def test_antigravity_quota_audit_preserves_partial_work(tmp_path, tokens, retryable):
    from scripts import run_benchmark_release as campaign

    run = tmp_path / "run"
    campaign.write_json(run / "results.json", {"results": [{
        "task_id": "task", "trial_name": "trial", "failure_mode": "unknown_agent_error",
        "total_input_tokens": tokens, "total_output_tokens": 0,
    }]})
    logs = run / "task/trial/agent-logs"
    logs.mkdir(parents=True)
    (logs / "antigravity.stderr.log").write_text("Individual quota reached. Resets in 82h32m9s.")
    audit = campaign.audit_model_run(run, "task", evaluation_required=False)
    assert audit["reason"] == "antigravity quota reached"
    assert audit["retryable"] is retryable
    assert audit["continuation_required"] is (tokens > 0)
    assert audit["quota_retry_after"]


def test_antigravity_quota_audit_recovers_usage_missing_from_timed_out_harness(tmp_path):
    from scripts import run_benchmark_release as campaign

    run = tmp_path / "run"
    campaign.write_json(run / "results.json", {"results": [{"task_id": "task", "trial_name": "trial",
        "failure_mode": "agent_timeout", "total_input_tokens": 0, "total_output_tokens": 0}]})
    logs = run / "task/trial/agent-logs"
    logs.mkdir(parents=True)
    candidate = run / "task/trial/evaluation/candidate.patch"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("partial patch")
    (logs / "antigravity.jsonl").write_text(json.dumps({"event": "result", "result": {
        "status": "ERROR", "error": "Individual quota reached. Resets in 1h.",
        "usage": {"input_tokens": 100, "output_tokens": 20}}}) + "\n")
    audit = campaign.audit_model_run(run, "task", evaluation_required=False)
    assert audit["tokens"] == 120
    assert audit["continuation_required"] is True and audit["retryable"] is False


def test_timeout_after_quota_retry_counts_later_work_and_not_old_cooldown(tmp_path):
    from scripts import run_benchmark_release as campaign

    run = tmp_path / "run"
    campaign.write_json(run / "results.json", {"results": [{"task_id": "task", "trial_name": "trial",
        "failure_mode": "agent_timeout", "total_input_tokens": 0, "total_output_tokens": 0,
        "agent_started_at": "2026-01-01T00:00:00+00:00", "agent_ended_at": "2026-01-01T00:01:40+00:00"}]})
    logs = run / "task/trial/agent-logs"
    logs.mkdir(parents=True)
    events = [
        {"event": "result", "result": {"status": "ERROR", "duration_seconds": 20,
            "error": "Individual quota reached. Resets in 10s.", "usage": {"input_tokens": 100, "output_tokens": 20}}},
        {"event": "init"},
        {"event": "step_update", "step_update": {"step_index": 1, "step_type": "agent_response",
            "state": "DONE", "usage": {"input_tokens": 30, "output_tokens": 5}}},
    ]
    (logs / "antigravity.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    storage = run / "task/trial/development-storage"
    storage.mkdir()
    audit = campaign.audit_model_run(run, "task", evaluation_required=False)
    assert audit["reason"] == "antigravity interrupted after quota wait"
    assert audit["tokens"] == 155
    assert audit["agent_duration_sec"] == 85
    assert audit["continuation_required"] is True


def test_quota_continuation_waits_for_reset_and_requires_retained_workspace():
    from scripts import run_benchmark_release as campaign

    audit = {"reason": "antigravity quota reached", "retryable": False,
             "continuation_required": True, "development_storage": "/retained",
             "quota_retry_after": "2000-01-01T00:00:00+00:00"}
    assert campaign.incomplete_condition_retry_reason("development_failed", audit) == "quota_workspace_continuation"
    assert campaign.incomplete_condition_retry_reason("development_failed", {**audit, "development_storage": None}) is None
    assert campaign.incomplete_condition_retry_reason("development_failed", {**audit, "quota_retry_after": "2999-01-01T00:00:00+00:00"}) is None


def test_continuation_context_preserves_recorded_tool_calls(tmp_path):
    from scripts import run_benchmark_release as campaign

    trace = tmp_path / "run/task/trial/agent-trace/session"
    campaign.write_json(trace / "objects/call", {"name": "run_command", "inputs": {"command": "make test"}})
    campaign.write_json(trace / "objects/result", {"result": {"stdout": "passed"}})
    (trace / "events.jsonl").write_text("\n".join(json.dumps(event) for event in [
        {"event": "mcp.started", "data": {"path": "objects/call"}},
        {"event": "mcp.result", "data": {"path": "objects/result"}},
        {"event": "native.hook", "data": {}},
    ]) + "\n")
    output = tmp_path / "context.jsonl"
    assert campaign.write_continuation_context(tmp_path / "run", output) == 2
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows[0]["data"]["inputs"]["command"] == "make test"
    assert rows[1]["data"]["result"]["stdout"] == "passed"


@pytest.mark.parametrize("zero_token_retry", [False, True])
def test_quota_continuation_restores_workspace_and_remaining_budget(tmp_path, monkeypatch, zero_token_retry):
    import os
    import time
    from types import SimpleNamespace

    from pitbench.schema.research import GeneralizationStudyPlan, ResearchBudget
    from scripts import run_benchmark_release as campaign

    job = {"run_id": "run", "task_id": "task", "arm": "standard", "hypotheses_required": False,
           "model": {"agent": "antigravity", "model": "model", "setting": "network_retries=1"}}
    campaign.write_json(tmp_path / "cohort.json", {"jobs": [job]})
    campaign.write_json(tmp_path / "agent-status/run/status.json", {**job, "status": "development_failed"})
    old = tmp_path / "runs/run"
    campaign.write_json(old / "results.json", {"results": [{"task_id": "task", "trial_name": "trial",
        "failure_mode": "unknown_agent_error", "total_input_tokens": 5, "total_output_tokens": 0}]})
    storage = old / "task/trial/development-storage"
    (storage / "workspace").mkdir(parents=True)
    (storage / "workspace/work.txt").write_text("previous work")
    logs = old / "task/trial/agent-logs"
    logs.mkdir()
    (logs / "antigravity.jsonl").write_text(json.dumps({"event": "result", "result": {
        "status": "ERROR", "error": "Individual quota reached. Resets in 1s.",
        "duration_seconds": 20, "conversation_id": "old"}}) + "\n")
    stderr = logs / "antigravity.stderr.log"
    stderr.write_text("Individual quota reached. Resets in 1s.")
    os.utime(stderr, (time.time() - 30, time.time() - 30))
    trace = old / "task/trial/agent-trace/session"
    campaign.write_json(trace / "objects/call", {"name": "run_command", "inputs": {"command": "ls"}})
    (trace / "events.jsonl").write_text(json.dumps({"event": "mcp.started", "data": {"path": "objects/call"}}) + "\n")
    if zero_token_retry:
        previous_run = tmp_path / "runs/run-infra1"
        campaign.write_json(previous_run / "results.json", {"results": [{"task_id": "task", "trial_name": "trial",
            "failure_mode": "unknown_agent_error", "total_input_tokens": 0, "total_output_tokens": 0}]})
        previous_logs = previous_run / "task/trial/agent-logs"
        previous_logs.mkdir(parents=True)
        previous_stderr = previous_logs / "antigravity.stderr.log"
        previous_stderr.write_text("Individual quota reached. Resets in 1s.")
        os.utime(previous_stderr, (time.time() - 30, time.time() - 30))
        campaign.write_json(tmp_path / "infrastructure-retries/run/infra1/retry.json",
                            {"status": "retry_failed", "retry_run_id": "run-infra1"})
    views = tmp_path / "views.json"
    views.write_text(json.dumps({"tasks": {"task": {}}, "arms": {"standard": {"task": {
        "task_config_path": "/task.yaml", "agent_task_config_path": "/agent.yaml"}}}}))
    plan_file = tmp_path / "plan.json"
    plan_file.write_text("{}")
    budget = ResearchBudget(wall_time_sec=100, cpus=1, memory_gib=1)
    plan = SimpleNamespace(evaluation_run_target=None, research_budget=budget, shared_agent_tools=["inspect"])
    monkeypatch.setattr(campaign, "read_study_plan", lambda *_: plan)
    config = tmp_path / "config.yaml"
    config.write_text("agents: {antigravity: {}}\n")

    def run(command, **kwargs):
        assert kwargs["env"]["PITBENCH_CONTINUATION_STORAGE"] == str(storage)
        assert Path(kwargs["env"]["PITBENCH_CONTINUATION_CONTEXT"]).is_file()
        settings = yaml.safe_load(Path(command[command.index("--config") + 1]).read_text())
        assert settings["research_budget"]["wall_time_sec"] == 80
        assert settings["agents"]["antigravity"]["continuation_context_available"] is True
        rid = command[command.index("--run-id") + 1]
        campaign.write_json(tmp_path / "runs" / rid / "results.json", {"results": [{"task_id": "task", "trial_name": "trial",
            "failure_mode": "unset", "total_input_tokens": 3, "total_output_tokens": 0}]})
        candidate = tmp_path / "runs" / rid / "task/trial/evaluation/candidate.patch"
        candidate.parent.mkdir(parents=True)
        candidate.write_text("continued patch")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(campaign.subprocess, "run", run)
    campaign.retry_unexecuted_condition(SimpleNamespace(
        study_plan=plan_file, study_views=views, output=tmp_path, retry_run_id="run",
        config=config, cpu_pool="0",
    ))
    status = json.loads((tmp_path / "agent-status/run/status.json").read_text())
    assert status["status"] == "evaluation_pending"
    assert status["audit"]["prior_tokens"] == 5
    assert status["audit"]["continuation_kind"] == "restored_workspace_and_recorded_tool_history"


def test_condition_recovery_lock_rejects_concurrent_duplicate(tmp_path):
    from scripts import run_benchmark_release as campaign

    root = tmp_path / "recovery"
    first = campaign.acquire_condition_recovery_lock(root)
    try:
        with pytest.raises(ValueError, match="already running"):
            campaign.acquire_condition_recovery_lock(root)
    finally:
        first.close()

    second = campaign.acquire_condition_recovery_lock(root)
    second.close()


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
    model_matrix = yaml.safe_load((ROOT / plan["model_matrix"]).read_text())["runs"]
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

    from tests.support import protocol

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
        == len(selected)
        * len(plan["arms"])
        * len(model_matrix)
        * plan["agent_attempts_per_condition"]
    )
    assert set(validated) == set(selected)
    assert {row["model"]["label"] for row in frozen["jobs"]} == {
        row["label"] for row in model_matrix
    }
    if plan.get("comparison_mode") == "model_repeats":
        assert {row["arm"] for row in frozen["jobs"]} == {"standard"}
        assert {row["attempt"] for row in frozen["jobs"]} == {1, 2, 3}
        assert frozen["agent_attempts_per_condition"] == 3
        for task_id in selected:
            for model in model_matrix:
                assert (
                    len(
                        [
                            row
                            for row in frozen["jobs"]
                            if row["task_id"] == task_id
                            and row["model"]["model"] == model["model"]
                        ]
                    )
                    == plan["agent_attempts_per_condition"]
                )
    elif plan.get("comparison_mode") == "agent_augmentation":
        assert (
            str(tmp_path / "public" / (selected[-1] + ".yaml"))
            in frozen["input_files_sha256"]
        )
        for task_id in selected:
            for model in model_matrix:
                for arm in plan["arms"]:
                    rows = [
                        j
                        for j in frozen["jobs"]
                        if j["task_id"] == task_id
                        and j["model"]["model"] == model["model"]
                        and j["arm"] == arm["id"]
                    ]
                    assert {j["attempt"] for j in rows} == set(
                        range(1, plan["agent_attempts_per_condition"] + 1)
                    )
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


@pytest.fixture
def captured_study(tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace

    from pitbench.harness.evaluation import EvaluationEnvelope
    from scripts import run_benchmark_release as campaign

    patch = tmp_path / "runs/run/task/trial/evaluation/candidate.patch"
    patch.parent.mkdir(parents=True)
    patch.write_text("retained candidate")
    task = tmp_path / "workspaces/run/task/task.yaml"
    task.parent.mkdir(parents=True)
    task.write_text(yaml.safe_dump({"evaluator_config": {}}))
    state = {
        "run_id": "run",
        "task_id": "task",
        "arm": "standard",
        "status": "evaluation_pending",
        "execution_run_id": None,
        "model": {
            "agent": "codex",
            "model": "frozen",
            "setting": "reasoning_effort=high",
        },
        "audit": {
            "candidate_patch": str(patch),
            "candidate_patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
        },
    }
    campaign.write_json(tmp_path / "cohort.json", {"jobs": [state]})
    campaign.write_json(tmp_path / "agent-status/run/status.json", state)
    campaign.write_json(
        tmp_path / "runs/run/results.json", {"results": [{"task_id": "task"}]}
    )
    requests = []

    def envelope(request):
        requests.append(request)
        return EvaluationEnvelope(
            evaluator="pitbench",
            evaluator_version="test",
            completed=False,
            payload={},
            error="retained failure",
        )

    monkeypatch.setattr(
        campaign, "PitBenchEvaluator", lambda: SimpleNamespace(envelope=envelope)
    )
    return campaign, SimpleNamespace(output=tmp_path), state, requests


def test_retry_retains_prior_judge_and_refreshes_live_index(captured_study):
    campaign, args, state, requests = captured_study
    patch = Path(state["audit"]["candidate_patch"])
    old_observations = patch.parent / "judge-observations.jsonl"
    old_observations.write_text("original measurements\n")
    first = campaign.evaluate_study_candidate(state, args, "0-7")
    first_output = Path(first["evaluation_output"])
    first_envelope = (first_output / "evaluation-envelope.json").read_bytes()
    second = campaign.evaluate_study_candidate(first, args, "8-15")
    assert first_output != Path(second["evaluation_output"])
    assert [row.evaluator_config["judge_cpuset_cpus"] for row in requests] == [
        "0-7",
        "8-15",
    ]
    assert all(
        row.candidate_patch_path.read_bytes() == patch.read_bytes() for row in requests
    )
    assert old_observations.read_text() == "original measurements\n"
    assert (first_output / "evaluation-envelope.json").read_bytes() == first_envelope
    assert second["evaluation_attempt"] == 2
    index = json.loads((args.output / "model-runs.json").read_text())
    assert index[0]["evaluation_attempt"] == 2
    assert index[0]["status"] == "evaluation_failed"
    assert json.loads((args.output / "study-status.json").read_text())["statuses"] == {
        "evaluation_failed": 1
    }


@pytest.mark.parametrize("interrupted_by_reboot", [False, True])
def test_interrupted_evaluation_resume_reuses_checkpoint_and_cpu_pool(
    captured_study, interrupted_by_reboot
):
    campaign, args, state, requests = captured_study
    source_patch = Path(state["audit"]["candidate_patch"])
    output = args.output / "evaluations/run/attempt-original"
    output.mkdir(parents=True)
    (output / "candidate.patch").write_bytes(source_patch.read_bytes())
    (output / "judge-observations.jsonl").write_text("retained observations\n")
    previous = {**state, "status": "evaluating" if interrupted_by_reboot else "evaluation_failed",
                "evaluation_output": str(output),
                "evaluation_cpu_pool": "0-7", "evaluation_attempt": 1}
    campaign.write_json(args.output / "agent-status/run/status.json", previous)
    args.resume_evaluation = not interrupted_by_reboot
    result = campaign.evaluate_study_candidate(
        {**previous, "interrupted_by_reboot": interrupted_by_reboot}, args, "0-7"
    )
    assert Path(result["evaluation_output"]) == output
    assert requests[0].output_dir == output
    assert (output / "judge-observations.jsonl").read_text() == "retained observations\n"
    assert list(output.glob("resume-*.json"))


def test_competing_manager_cannot_overwrite_active_condition(
    captured_study, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from types import SimpleNamespace

    campaign, args, state, requests = captured_study
    entered, release = Event(), Event()
    original = campaign.PitBenchEvaluator().envelope

    def envelope(request):
        entered.set()
        assert release.wait(5)
        return original(request)

    monkeypatch.setattr(
        campaign, "PitBenchEvaluator", lambda: SimpleNamespace(envelope=envelope)
    )
    path = args.output / "agent-status/run/status.json"
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(campaign.evaluate_study_candidate, state, args, "0-7")
        try:
            assert entered.wait(5)
            before = path.read_bytes()
            duplicate = campaign.evaluate_study_candidate(state, args, "8-15")
            assert duplicate["status"] == "evaluating"
            assert path.read_bytes() == before
        finally:
            release.set()
        result = future.result()
    assert len(requests) == 1
    # A snapshot queued before the other manager completed must also be skipped.
    assert campaign.evaluate_study_candidate(state, args, "8-15") == result
    assert len(requests) == 1


def test_stop_leaves_queued_condition_unchanged(captured_study):
    campaign, args, state, requests = captured_study
    (args.output / "STOP").touch()
    path = args.output / "agent-status/run/status.json"
    before = path.read_bytes()
    assert campaign.evaluate_study_candidate(state, args, "0-7") == state
    assert path.read_bytes() == before
    assert not requests


@pytest.mark.parametrize("already_stopped", [False, True])
def test_stop_at_dispatch_wait_never_starts_agent(
    tmp_path, monkeypatch, already_stopped
):
    import io
    import time
    from types import SimpleNamespace

    from scripts import run_benchmark_release as campaign

    path = tmp_path / "agent-status/run/status.json"
    previous = {"run_id": "run", "status": "queued"}
    campaign.write_json(
        path, {**previous, "status": "preparing", "started_at": "not an agent start"}
    )
    if already_stopped:
        (tmp_path / "STOP").touch()
    monkeypatch.setattr(
        campaign.time, "sleep", lambda delay: (tmp_path / "STOP").touch()
    )
    monkeypatch.setattr(
        campaign.subprocess,
        "Popen",
        lambda *a, **k: pytest.fail("Agent started after STOP"),
    )
    process = campaign.launch_study_process(
        ["agent"],
        SimpleNamespace(output=tmp_path),
        path,
        io.StringIO(),
        time.monotonic() + 10,
        previous,
    )
    assert process is None
    assert json.loads(path.read_text()) == previous


def test_evaluation_without_worker_pool_fails_before_dispatch(captured_study):
    campaign, args, state, requests = captured_study
    args.evaluation_workers = 1
    args.evaluation_cpu_pools = []
    with pytest.raises(ValueError, match="one CPU pool"):
        campaign.study_evaluations(args)
    assert not requests
    assert (
        json.loads((args.output / "agent-status/run/status.json").read_text()) == state
    )


def test_cpu_reservations_include_overlapping_worker_pools(tmp_path):
    import os

    from scripts import run_benchmark_release as campaign

    first, second = sorted(os.sched_getaffinity(0))[:2]
    with campaign.study_cpu_lease([str(first)], lock_root=tmp_path):
        with pytest.raises(RuntimeError, match="reserved"):
            with campaign.study_cpu_lease([f"{second},{first}"], lock_root=tmp_path):
                pytest.fail("overlapping CPU reservation was accepted")
        with campaign.study_cpu_lease([str(second)], lock_root=tmp_path):
            pass
    with campaign.study_cpu_lease([str(first)], lock_root=tmp_path):
        pass


def test_status_command_reads_live_conditions(captured_study, monkeypatch):
    campaign, args, state, _ = captured_study
    state = {**state, "status": "evaluation_incomplete"}
    campaign.write_json(args.output / "agent-status/run/status.json", state)
    campaign.write_json(args.output / "model-runs.json", [{"status": "evaluating"}])
    monkeypatch.setattr(
        campaign.sys,
        "argv",
        [
            "run_benchmark_release",
            "--phase",
            "study-status",
            "--output",
            str(args.output),
        ],
    )
    campaign.main()
    index = json.loads((args.output / "model-runs.json").read_text())
    assert index[0]["status"] == "evaluation_incomplete"


def test_live_study_snapshot_includes_recovery_agent_activity(tmp_path):
    from scripts import run_benchmark_release as campaign

    job = {"run_id": "run", "task_id": "task", "model": {"model": "model", "agent": "antigravity"}, "arm": "standard"}
    campaign.write_json(tmp_path / "cohort.json", {"jobs": [job]})
    campaign.write_json(tmp_path / "agent-status/run/status.json", {"status": "development_failed"})
    campaign.write_json(tmp_path / "infrastructure-retries/run/infra1/retry.json", {"status": "retry_running", "retry_run_id": "run-infra1"})
    trace = tmp_path / "runs/run-infra1/task/trial/agent-trace/session"
    campaign.write_json(trace / "objects/detail", {"detail": "Agent: Running tool"})
    (trace / "events.jsonl").write_text(json.dumps({"event": "agent.progress", "data": {"path": "objects/detail"}}) + "\n")
    snapshot = campaign.study_live_snapshot(tmp_path)
    assert snapshot["statuses"] == {"recovery_running": 1}
    assert snapshot["active_agents"][0]["execution_run_id"] == "run-infra1"
    assert snapshot["active_agents"][0]["activity"] == "Agent: Running tool"
    cursors = {}
    assert campaign.study_live_events(tmp_path, snapshot, cursors) == []
    campaign.write_json(trace / "objects/command", {"name": "run_command", "inputs": {"command": "make test"}})
    with (trace / "events.jsonl").open("a") as stream:
        stream.write(json.dumps({"event": "mcp.started", "data": {"path": "objects/command"}}) + "\n")
    events = campaign.study_live_events(tmp_path, snapshot, cursors)
    assert events[0]["data"]["inputs"]["command"] == "make test"
    assert campaign.study_live_events(tmp_path, snapshot, cursors) == []


def test_adopt_running_agent_preserves_process_and_records_completion(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys
    from functools import partial
    from types import SimpleNamespace

    from scripts import run_benchmark_release as campaign

    cpu = str(min(os.sched_getaffinity(0)))
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(.2)", "adopted-run"])
    job = {"run_id": "adopted-run", "task_id": "task", "model": {"agent": "antigravity", "model": "m"}, "arm": "standard"}
    campaign.write_json(tmp_path / "cohort.json", {"jobs": [job]})
    campaign.write_json(tmp_path / "agent-status/adopted-run/status.json", {
        **job, "status": "development_running", "pid": process.pid, "cpu_pool": cpu,
    })
    real_lease = campaign.study_cpu_lease
    monkeypatch.setattr(campaign, "study_cpu_lease", partial(real_lease, lock_root=tmp_path / "locks"))
    monkeypatch.setattr(campaign, "audit_model_run", lambda *a, **k: {"complete": True, "candidate_patch": "/retained"})
    try:
        campaign.adopt_running_study_agents(SimpleNamespace(output=tmp_path))
    finally:
        process.wait()
    state = json.loads((tmp_path / "agent-status/adopted-run/status.json").read_text())
    assert state["status"] == "evaluation_pending" and state["adopted_dispatcher"] is True
    assert json.loads((tmp_path / "adoption-ready.json").read_text())["conditions"]["adopted-run"]["pid"] == process.pid


def test_automatic_pools_use_memory_limit_and_spread_each_worker_over_sockets():
    from scripts import run_benchmark_release as campaign

    sockets = [list(range(start, start + 8)) for start in (0, 8, 16, 24)]
    pools = campaign.automatic_study_pools(sockets, 8, 32, 160, 64)
    assert len(pools) == 3
    occupied = set()
    for pool in pools:
        cpus = campaign.study_cpu_ids(pool)
        assert len(cpus) == 8 and not occupied & cpus
        assert [sum(cpu in group for cpu in cpus) for group in sockets] == [2, 2, 2, 2]
        occupied |= cpus


def test_automatic_cpu_selection_does_not_use_a_reserved_core_smt_sibling(
    tmp_path, monkeypatch
):
    import fcntl

    from scripts import run_benchmark_release as campaign

    monkeypatch.setattr(campaign.os, "sched_getaffinity", lambda _: set(range(8)))
    topology, locks = tmp_path / "cpu", tmp_path / "locks"
    locks.mkdir()
    for cpu in range(8):
        folder = topology / f"cpu{cpu}" / "topology"
        folder.mkdir(parents=True)
        (folder / "physical_package_id").write_text(str((cpu % 4) // 2))
        (folder / "core_id").write_text(str(cpu % 2))
    with (locks / "cpu-0.lock").open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        sockets = campaign.available_study_sockets(topology=topology, lock_root=locks)
    assert sockets == [[1], [2, 3]]


def test_agent_phase_completion_recycles_pools_into_parallel_evaluation(
    tmp_path, monkeypatch
):
    import threading
    from types import SimpleNamespace

    from pitbench.cli.evaluate_config import EvaluationConfig
    from scripts import run_benchmark_release as campaign

    args = SimpleNamespace(
        output=tmp_path,
        workers=2,
        evaluate_candidates=True,
        evaluation_cpu_pools=["16-23"],
        reuse_agent_pools=True,
    )
    config = EvaluationConfig(judge_cpus=8, judge_memory="32g")
    monkeypatch.setattr(
        campaign, "host_snapshot", lambda: {"available_memory_gib": 128}
    )
    barrier = threading.Barrier(3)
    pools_used = []

    def evaluate(state, args, pool):
        pools_used.append(pool)
        barrier.wait(timeout=5)
        return {**state, "status": "evaluation_completed"}

    monkeypatch.setattr(campaign, "evaluate_study_candidate", evaluate)
    campaign.run_study_jobs(
        [1, 2, 3],
        lambda job: {"run_id": str(job), "status": "evaluation_pending"},
        args,
        ["0-7", "8-15"],
        config,
        ["task"],
    )
    assert set(pools_used) == {"0-7", "8-15", "16-23"}
    receipt = json.loads((tmp_path / "evaluation-parallelism.json").read_text())
    assert receipt["evaluation_workers"] == 3


def test_evaluation_admission_wait_does_not_start_or_overwrite_a_condition(
    captured_study, monkeypatch
):
    campaign, args, state, requests = captured_study
    args.evaluation_workers = 1
    args.evaluation_cpu_pools = ["0-7"]
    monkeypatch.setattr(campaign, "wait_for_study_dispatch", lambda *a: False)
    monkeypatch.setattr(
        campaign,
        "evaluate_study_candidate",
        lambda *a: pytest.fail("dispatched while stopped"),
    )
    campaign._study_evaluations(args)
    assert not requests
    assert (
        json.loads((args.output / "agent-status/run/status.json").read_text()) == state
    )


def test_last_evaluator_finalizes_report_only_after_all_conditions_are_terminal(
    captured_study, monkeypatch
):
    campaign, args, state, _ = captured_study
    reports = []
    monkeypatch.setattr(
        campaign, "write_campaign_report", lambda *a, **kw: reports.append(kw)
    )
    assert campaign.finalize_study_report(args) is False
    assert not (args.output / "execution-complete.json").exists()
    campaign.write_json(
        args.output / "agent-status/run/status.json",
        {**state, "status": "evaluation_completed"},
    )
    assert campaign.finalize_study_report(
        args, {"tasks": {"task": {"evaluator_task_config": "frozen.yaml"}}}
    )
    assert len(reports) == 1
    assert json.loads((args.output / "execution-complete.json").read_text())[
        "statuses"
    ] == {"evaluation_completed": 1}


def test_evaluation_background_launcher_is_independent_of_live_agent_manager(
    tmp_path, monkeypatch
):
    import os
    from types import SimpleNamespace

    from scripts import run_benchmark_release as campaign

    campaign.write_json(tmp_path / "launcher-process.json", {"pid": os.getpid()})
    calls = []
    monkeypatch.setattr(
        campaign.subprocess,
        "Popen",
        lambda *a, **kw: calls.append(a) or SimpleNamespace(pid=12345),
    )
    campaign.launch_background(
        SimpleNamespace(output=tmp_path, phase="study-evaluations"),
        ["--phase", "study-evaluations"],
    )
    assert len(calls) == 1
    assert (
        json.loads((tmp_path / "launcher-study-evaluations-process.json").read_text())[
            "pid"
        ]
        == 12345
    )
    assert (
        json.loads((tmp_path / "launcher-process.json").read_text())["pid"]
        == os.getpid()
    )


@pytest.mark.parametrize("requested_workers,expected_agents", [(None, 3), (2, 2)])
def test_omitted_worker_counts_select_host_capacity_instead_of_one(
    tmp_path, monkeypatch, requested_workers, expected_agents
):
    from contextlib import nullcontext

    from pitbench.cli.evaluate_config import EvaluationConfig
    from scripts import run_benchmark_release as campaign

    config = tmp_path / "evaluation.yaml"
    config.write_text(
        EvaluationConfig(judge_cpus=8, judge_memory="32g").model_dump_json()
    )
    profile = tmp_path / "study.yaml"
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
                "evaluate_candidates": True,
                "resource_policy": {"min_available_memory_gib": 64},
            }
        )
    )
    if requested_workers is not None:
        configured = yaml.safe_load(profile.read_text())
        configured["workers"] = requested_workers
        profile.write_text(yaml.safe_dump(configured))
    monkeypatch.setattr(
        campaign,
        "available_study_sockets",
        lambda **kw: [list(range(i, i + 16)) for i in (0, 16, 32, 48)],
    )
    monkeypatch.setattr(
        campaign, "host_snapshot", lambda: {"available_memory_gib": 192}
    )
    monkeypatch.setattr(campaign, "study_cpu_lease", lambda _: nullcontext())
    called = []
    monkeypatch.setattr(campaign, "study_models", called.append)
    monkeypatch.setattr(
        campaign.sys, "argv", ["run_benchmark_release", "--profile", str(profile)]
    )
    campaign.main()
    assert called[0].workers == expected_agents
    assert called[0].evaluation_workers == 1
    assert (
        len(called[0].cpu_pools) + len(called[0].evaluation_cpu_pools)
        == expected_agents + 1
    )


def test_study_reads_explicit_external_task_configs_instead_of_default_catalog(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from pitbench.cli.evaluate_config import EvaluationConfig
    from scripts import run_benchmark_release as campaign

    task_id = "release_vroom_v1_15_0"
    external = tmp_path / "task-package" / f"{task_id}.yaml"
    external.parent.mkdir()
    external.write_bytes((ROOT / f"configs/tasks/{task_id}.yaml").read_bytes())
    plan = json.loads(
        (ROOT / "configs/research/latest_high_algorithm_space_v4.json").read_text()
    )
    plan.update(
        task_configs=[str(external)],
        task_targets={},
        task_evaluation_seed_policies={},
        evaluation_run_target=None,
        supplemental_seed_count=None,
        supplemental_primary_budget_only=False,
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    views = {
        "status": "development_frozen_evaluation_gated",
        "tasks": {task_id: {"evaluator_task_config": str(external)}},
        "arms": {
            arm["id"]: {task_id: {"agent_task_config_path": str(external)}}
            for arm in plan["arms"]
        },
    }
    views_path = tmp_path / "views.json"
    views_path.write_text(json.dumps(views))
    config = tmp_path / "evaluation.yaml"
    config.write_text(EvaluationConfig().model_dump_json())
    monkeypatch.setattr(
        campaign.TaskCatalog,
        "validate_one",
        lambda *a: pytest.fail("ignored the configured task path"),
    )
    output = tmp_path / "cohort"
    campaign.study_models(
        SimpleNamespace(
            study_plan=plan_path,
            study_views=views_path,
            config=config,
            output=output,
            schedule_seed=20261101,
            prepare_only=True,
        )
    )
    jobs = json.loads((output / "cohort.json").read_text())["jobs"]
    assert jobs and {job["task_id"] for job in jobs} == {task_id}


def test_restarting_an_active_background_profile_does_not_reallocate_its_cpus(
    tmp_path, monkeypatch
):
    import os

    from scripts import run_benchmark_release as campaign

    cohort = tmp_path / "cohort"
    campaign.write_json(cohort / "launcher-process.json", {"pid": os.getpid()})
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        yaml.safe_dump(
            {
                "phase": "study-models",
                "manifest": str(ROOT / "configs/benchmarks/0.0.1.json"),
                "config": "unchanged-evaluation.yaml",
                "output": str(cohort),
                "study_plan": "frozen-plan.json",
                "study_views": "frozen-views.json",
                "background": True,
            }
        )
    )
    monkeypatch.setattr(
        campaign,
        "configure_study_parallelism",
        lambda _: pytest.fail("reallocated a live manager"),
    )
    monkeypatch.setattr(
        campaign.sys, "argv", ["run_benchmark_release", "--profile", str(profile)]
    )
    campaign.main()
    assert not (cohort / "parallelism.json").exists()


def test_saved_cohort_replay_preserves_source_and_creates_only_evaluation_queue(tmp_path):
    from types import SimpleNamespace

    from pitbench.evaluator.artifacts import write_json
    from scripts import run_benchmark_release as campaign

    source = tmp_path / "original"
    output = tmp_path / "replay"
    job = {"run_id": "run", "task_id": "task", "model": {"agent": "codex", "model": "model"}}
    write_json(source / "cohort.json", {"jobs": [job]})
    patch = source / "runs/run/task/trial/evaluation/candidate.patch"
    patch.parent.mkdir(parents=True)
    patch.write_text("captured patch")
    write_json(source / "agent-status/run/status.json", {
        "status": "evaluation_completed", "audit": {"candidate_patch": str(patch)}
    })
    write_json(source / "runs/run/results.json", {
        "results": [{"task_id": "task", "trial_name": "trial", "evaluation": {"completed": True}}]
    })
    task = source / "workspaces/run/task/task.yaml"
    task.parent.mkdir(parents=True)
    task.write_text(yaml.safe_dump({"evaluator_config": {"base_repository": "original-base", "task_config_path": "original-task"}}))
    original = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    args = SimpleNamespace(source_cohort=source, output=output, config=config)
    campaign.prepare_study_replay(args)
    campaign.prepare_study_replay(args)
    status = json.loads((output / "agent-status/run/status.json").read_text())
    assert status["status"] == "evaluation_pending"
    assert status["new_agent_starts"] == 0
    assert status["audit"]["candidate_patch"] == str(patch)
    assert json.loads((output / "runs/run/results.json").read_text())["results"][0]["evaluation"] is None
    assert original == {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}


def test_saved_patch_replay_expands_one_candidate_to_multiple_task_configs(tmp_path):
    from types import SimpleNamespace

    from pitbench.evaluator.artifacts import write_json
    from pitbench.harness.harness import campaign

    source, output = tmp_path / "original", tmp_path / "replay"
    base_path = ROOT / "configs/tasks/release_choco_v6_0_2.yaml"
    base = yaml.safe_load(base_path.read_text())
    target_paths = {}
    for task_id in ("target_one", "target_two"):
        target = {**base, "task_id": task_id}
        target_paths[task_id] = tmp_path / (task_id + ".yaml")
        target_paths[task_id].write_text(yaml.safe_dump(target))
    job = {"run_id": "run", "task_id": base["task_id"], "model": {"agent": "codex", "model": "model"}}
    write_json(source / "cohort.json", {"jobs": [job]})
    patch = source / "candidate.patch"
    patch.write_text("captured patch")
    write_json(source / "agent-status/run/status.json", {"audit": {"candidate_patch": str(patch)}})
    write_json(source / "runs/run/results.json", {"results": [{"task_id": base["task_id"], "trial_name": "trial", "evaluation": {"completed": True}}]})
    task_dir = source / "workspaces/run" / base["task_id"]
    task_dir.mkdir(parents=True)
    (task_dir / "task.yaml").write_text(yaml.safe_dump({"evaluator_config": {"task_config_path": str(base_path)}}))
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"tasks": {name: {"task_config_path": str(path)} for name, path in target_paths.items()}}))
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(yaml.safe_dump({base["task_id"]: list(target_paths)}))
    args = SimpleNamespace(source_cohort=source, output=output, config=config, target_task_matrix=matrix)
    campaign.prepare_study_replay(args)
    cohort = json.loads((output / "cohort.json").read_text())
    assert len(cohort["jobs"]) == 2
    for job in cohort["jobs"]:
        run_id, task_id = job["run_id"], job["task_id"]
        status = json.loads((output / "agent-status" / run_id / "status.json").read_text())
        assert status["audit"]["candidate_patch"] == str(patch)
        assert status["source_task_id"] == base["task_id"]
        assert status["new_agent_starts"] == 0
        result = json.loads((output / "runs" / run_id / "results.json").read_text())
        assert result["results"][0]["task_id"] == task_id
        assert result["results"][0]["evaluation"] is None


def test_replay_adds_candidates_captured_after_first_snapshot(tmp_path):
    from types import SimpleNamespace

    from pitbench.evaluator.artifacts import write_json
    from scripts import run_benchmark_release as campaign

    source, output = tmp_path / "source", tmp_path / "replay"
    job = {"run_id": "run", "task_id": "task", "model": {"agent": "codex", "model": "model"}}
    write_json(source / "cohort.json", {"jobs": [job]})
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    args = SimpleNamespace(source_cohort=source, output=output, config=config)
    campaign.prepare_study_replay(args)
    status = output / "agent-status/run/status.json"
    assert json.loads(status.read_text())["status"] == "awaiting_candidate"
    patch = source / "candidate.patch"
    patch.write_text("captured patch")
    write_json(source / "agent-status/run/status.json", {"audit": {"candidate_patch": str(patch)}})
    write_json(source / "runs/run/results.json", {"results": []})
    task = source / "workspaces/run/task/task.yaml"
    task.parent.mkdir(parents=True)
    task.write_text(yaml.safe_dump({"evaluator_config": {"task_config_path": "original-task"}}))
    campaign.prepare_study_replay(args)
    assert json.loads(status.read_text())["status"] == "evaluation_pending"
    write_json(status, {"status": "evaluation_completed"})
    campaign.prepare_study_replay(args)
    assert json.loads(status.read_text())["status"] == "evaluation_completed"


def test_replay_does_not_evaluate_patch_from_interrupted_agent(tmp_path):
    from types import SimpleNamespace

    from scripts import run_benchmark_release as campaign

    source, output = tmp_path / "source", tmp_path / "replay"
    job = {"run_id": "run", "task_id": "task", "model": {"agent": "antigravity", "model": "model"}}
    campaign.write_json(source / "cohort.json", {"jobs": [job]})
    patch = source / "candidate.patch"
    patch.write_text("partial work")
    status = source / "agent-status/run/status.json"
    campaign.write_json(status, {"status": "development_failed", "audit": {"complete": False, "candidate_patch": str(patch)}})
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    args = SimpleNamespace(source_cohort=source, output=output, config=config)
    campaign.prepare_study_replay(args)
    target = output / "agent-status/run/status.json"
    assert json.loads(target.read_text())["status"] == "awaiting_candidate"
    campaign.write_json(target, {"status": "evaluation_pending", "audit": {"candidate_patch": str(patch)}})
    campaign.prepare_study_replay(args)
    assert json.loads(target.read_text())["status"] == "awaiting_candidate"
    campaign.write_json(status, {"status": "evaluation_pending", "audit": {"complete": True, "candidate_patch": str(patch)}})
    campaign.write_json(source / "runs/run/results.json", {"results": []})
    task = source / "workspaces/run/task/task.yaml"
    task.parent.mkdir(parents=True)
    task.write_text(yaml.safe_dump({"evaluator_config": {"task_config_path": "task.yaml"}}))
    campaign.prepare_study_replay(args)
    assert json.loads(target.read_text())["status"] == "evaluation_pending"


def test_repair_replaces_only_complete_matching_boundary_grid():
    from types import SimpleNamespace

    from scripts import run_benchmark_release as campaign

    def row(instance, suite, status):
        return SimpleNamespace(task_id="task", code_state="base", instance_set="panel",
            instance_id=instance, solver_seed=0, budget_sec=1, test_suite=suite,
            equivalence_parent_id=None, status=status)

    performance = row("normal", None, "completed")
    broken = row("boundary", "operational_reliability", "crashed")
    repaired = row("boundary", "operational_reliability", "completed")
    assert campaign.replace_study_observation_suite([performance, broken], [repaired], "operational_reliability") == [performance, repaired]
    with pytest.raises(ValueError, match="grid differs"):
        campaign.replace_study_observation_suite([performance, broken], [], "operational_reliability")


def test_provisional_performance_waits_for_complete_grid(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from pitbench.evaluator.storage import ObservationStore
    from pitbench.schema.observation import ExpectedRun, ExpectedRunGrid, RunObservation
    from scripts import run_benchmark_release as campaign

    source = tmp_path / "measurement"
    source.mkdir()
    row = RunObservation(task_id="task", code_state="base", instance_set="judge_id",
        instance_id="case", solver_seed=0, budget_sec=1, status="completed", valid=True)
    expected = ExpectedRun(task_id="task", code_state="base", instance_set="judge_id",
        instance_set_kind="judge_id", instance_id="case", solver_seed=0, budget_sec=1)
    (source / "expected-run-grid.json").write_text(ExpectedRunGrid(task_id="task", runs=[expected]).model_dump_json())
    ObservationStore.write_jsonl(source / "judge-observations.jsonl", [])
    campaign.write_json(tmp_path / "agent-status/run/status.json", {"run_id": "run", "task_id": "task",
        "status": "evaluating", "evaluation_output": str(source)})
    task = tmp_path / "workspaces/run/task/task.yaml"
    task.parent.mkdir(parents=True)
    task.write_text(yaml.safe_dump({"evaluator_config": {"task_config_path": "/task.yaml"}}))
    monkeypatch.setattr(campaign.PitBenchTask, "from_yaml", lambda _: object())
    monkeypatch.setattr("pitbench.metrics.performance_report.compute_task_performance_report",
                        lambda *a, **k: SimpleNamespace(model_dump=lambda **kw: {"classification": "inconclusive"}))
    args = SimpleNamespace(output=tmp_path, follow_source=False)
    campaign.watch_study_primary_results(args)
    assert not (tmp_path / "early-performance/run.json").exists()
    ObservationStore.write_jsonl(source / "judge-observations.jsonl", [row])
    campaign.watch_study_primary_results(args)
    result = json.loads((tmp_path / "early-performance/run.json").read_text())
    assert result["provisional"] and result["performance"]["classification"] == "inconclusive"


def test_native_patch_evaluation_status_uses_packaged_controller(tmp_path):
    from typer.testing import CliRunner

    from pitbench.cli.main import app
    from pitbench.evaluator.artifacts import write_json

    output = tmp_path / "cohort"
    write_json(output / "cohort.json", {"jobs": []})
    config = tmp_path / "patch-evaluation.yaml"
    config.write_text(yaml.safe_dump({"phase": "study-evaluations", "output": str(output)}))
    result = CliRunner().invoke(app, ["evaluate-patches", "--config", str(config), "--status"])
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    assert '"total": 0' in result.output


def test_native_patch_evaluation_rejects_agent_launch_profile(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from pitbench.cli.main import app
    from pitbench.harness.harness import campaign

    config = tmp_path / "agents.yaml"
    config.write_text("phase: study-models\n")
    monkeypatch.setattr(campaign, "main", lambda *a, **kw: pytest.fail("must not launch Agent study"))
    result = CliRunner().invoke(app, ["evaluate-patches", "--config", str(config)])
    assert result.exit_code == 2
    assert "study-evaluations" in result.output
    assert "study-parameters" in result.output


def test_patch_cohort_python_api_forwards_without_changing_process_arguments(tmp_path, monkeypatch):
    import sys

    from pitbench.harness.harness import campaign

    config = tmp_path / "evaluation.yaml"
    config.write_text("phase: study-evaluations\n")
    calls = []
    monkeypatch.setattr(campaign, "main", lambda argv, **kw: calls.append((argv, kw)))
    before = list(sys.argv)
    campaign.evaluate_saved_patch_cohort(config, source_cohort=tmp_path / "source", output=tmp_path / "output", workers=0, prepare_only=True, root=tmp_path)
    assert sys.argv == before
    argv, options = calls[0]
    assert argv[argv.index("--source-cohort") + 1] == str(tmp_path / "source")
    assert "--prepare-only" in argv
    assert options["root"] == tmp_path


def test_dynamic_evaluation_grows_when_peer_cores_become_free(tmp_path, monkeypatch):
    import threading
    from contextlib import contextmanager
    from types import SimpleNamespace

    from pitbench.cli.evaluate_config import EvaluationConfig
    from pitbench.harness.harness import campaign

    config = tmp_path / "config.json"
    config.write_text(EvaluationConfig().model_dump_json())
    campaign.write_json(tmp_path / "resource-policy.json", {"check_every_sec": 0.01})
    args = SimpleNamespace(output=tmp_path, config=config, dynamic_evaluation_workers=True, dynamic_evaluation_limit=0, dispatch_interval_sec=0)
    leased = set()
    inspections = []
    second_started = threading.Event()

    @contextmanager
    def lease(pools):
        pool = pools[0]
        assert pool not in leased
        leased.add(pool)
        try:
            yield
        finally:
            leased.remove(pool)

    def sockets():
        inspections.append(True)
        available = [0] if len(inspections) == 1 else [0, 1]
        return [[cpu for cpu in available if str(cpu) not in leased]]

    def worker(job, settings, pool):
        if job["id"] == 1:
            assert second_started.wait(2), "newly free core was never admitted"
        else:
            assert "0" in leased, "first job must remain running during expansion"
            second_started.set()
        return job

    monkeypatch.setattr(campaign, "study_cpu_lease", lease)
    monkeypatch.setattr(campaign, "available_study_sockets", sockets)
    monkeypatch.setattr(campaign, "study_judge_resources", lambda *a: (1, 1))
    monkeypatch.setattr(campaign, "host_snapshot", lambda: {"available_memory_gib": 100, "load_1min": 0, "cpu_temperature_c": None})
    records = campaign.run_evaluation_jobs(args, [{"task_id": "task", "id": 1}, {"task_id": "task", "id": 2}], worker)
    assert len(records) == 2 and not leased
    allocation = json.loads((tmp_path / "parallelism.json").read_text())
    assert allocation["peak_evaluation_workers"] == 2
    assert allocation["evaluation_workers"] == 0


def test_dynamic_evaluation_releases_finished_slot_while_peer_still_runs(tmp_path, monkeypatch):
    import threading
    from contextlib import contextmanager
    from types import SimpleNamespace

    from pitbench.cli.evaluate_config import EvaluationConfig
    from pitbench.harness.harness import campaign

    config = tmp_path / "config.json"
    config.write_text(EvaluationConfig().model_dump_json())
    campaign.write_json(tmp_path / "resource-policy.json", {"check_every_sec": 0.01})
    args = SimpleNamespace(output=tmp_path, config=config, dynamic_evaluation_workers=True, dynamic_evaluation_limit=2, dispatch_interval_sec=0)
    held, releases = set(), []
    last_started = threading.Event()

    @contextmanager
    def lease(pools):
        pool = pools[0]
        assert pool not in held
        held.add(pool)
        try:
            yield
        finally:
            held.remove(pool)
            releases.append(pool)

    def worker(job, settings, pool):
        if job["id"] == 2:
            assert last_started.wait(2)
        if job["id"] == 3:
            assert releases, "completed job did not return its CPU lease"
            last_started.set()
        return job

    monkeypatch.setattr(campaign, "study_cpu_lease", lease)
    monkeypatch.setattr(campaign, "available_study_sockets", lambda: [[cpu for cpu in [0, 1] if str(cpu) not in held]])
    monkeypatch.setattr(campaign, "study_judge_resources", lambda *a: (1, 1))
    monkeypatch.setattr(campaign, "host_snapshot", lambda: {"available_memory_gib": 100, "load_1min": 0, "cpu_temperature_c": None})
    jobs = [{"task_id": "task", "id": i} for i in [1, 2, 3]]
    assert len(campaign.run_evaluation_jobs(args, jobs, worker)) == 3
    assert not held


def test_dynamic_evaluation_honors_stop_without_dispatching_pending_jobs(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from pitbench.cli.evaluate_config import EvaluationConfig
    from pitbench.harness.harness import campaign

    config = tmp_path / "config.json"
    config.write_text(EvaluationConfig().model_dump_json())
    args = SimpleNamespace(output=tmp_path, config=config, dynamic_evaluation_workers=True, dynamic_evaluation_limit=1, dispatch_interval_sec=0)
    calls = []
    def worker(job, settings, pool):
        calls.append(job)
        (tmp_path / "STOP").touch()
        return job
    monkeypatch.setattr(campaign, "study_cpu_lease", lambda *a: nullcontext())
    monkeypatch.setattr(campaign, "available_study_sockets", lambda: [[0]])
    monkeypatch.setattr(campaign, "study_judge_resources", lambda *a: (1, 1))
    monkeypatch.setattr(campaign, "host_snapshot", lambda: {"available_memory_gib": 100, "load_1min": 0, "cpu_temperature_c": None})
    campaign.run_evaluation_jobs(args, [{"task_id": "task"}] * 3, worker)
    assert len(calls) == 1


def test_automatic_evaluation_can_wait_for_capacity_at_startup(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from pitbench.cli.evaluate_config import EvaluationConfig
    from pitbench.harness.harness import campaign

    config = tmp_path / "config.json"
    config.write_text(EvaluationConfig().model_dump_json())
    campaign.write_json(tmp_path / "cohort.json", {"jobs": [{"task_id": "task"}]})
    args = SimpleNamespace(phase="study-evaluations", workers=0, evaluation_workers=0,
        cpu_pools=[], evaluation_cpu_pools=[], config=config, study_plan=None,
        source_cohort=None, output=tmp_path, reuse_agent_pools=True)
    monkeypatch.setattr(campaign, "available_study_sockets", lambda **kw: [])
    monkeypatch.setattr(campaign, "host_snapshot", lambda: {"available_memory_gib": 100})
    campaign.configure_study_parallelism(args)
    assert args.dynamic_evaluation_workers
    assert args.evaluation_cpu_pools == []
    assert args.evaluation_workers == 0


def test_following_source_does_not_rebuild_unchanged_provisional_report(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts import run_benchmark_release as campaign

    rows = [{"condition_id": "one", "status": "evaluation_completed"},
            {"condition_id": "two", "status": "development_failed", "evaluation_error": "source has no captured candidate"}]
    monkeypatch.setattr(campaign, "_study_evaluations", lambda args: None)
    monkeypatch.setattr(campaign, "refresh_study_index", lambda output: rows)
    calls = []
    monkeypatch.setattr(campaign, "finalize_study_report", lambda args: calls.append(True))
    sleep_count = iter([None, None, KeyboardInterrupt()])
    def sleep(_):
        value = next(sleep_count)
        if value:
            raise value
    monkeypatch.setattr(campaign.time, "sleep", sleep)
    args = SimpleNamespace(output=tmp_path, source_cohort=None, dynamic_evaluation_workers=True,
                           follow_source=True, follow_interval_sec=1)
    with pytest.raises(KeyboardInterrupt):
        campaign.study_evaluations(args)
    assert len(calls) == 1


def test_agent_priority_pause_resumes_evaluation_after_recovery_is_scheduled(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts import run_benchmark_release as campaign

    output, source = tmp_path / "evaluation", tmp_path / "source"
    campaign.write_json(output / "agent-priority-pause.json", {"owner": "automatic resource balancer"})
    (output / "STOP").touch()
    campaign.write_json(source / "recovery-status.json", {"eligible_waiting": 0, "recovery_errors": {}})
    calls = []
    monkeypatch.setattr(campaign.subprocess, "run", lambda command, **kwargs:
                        calls.append(command) or SimpleNamespace(returncode=0, stdout="started", stderr=""))
    campaign.rebalance_study_agents(SimpleNamespace(output=output, source_cohort=source, profile=tmp_path / "profile.yaml"))
    assert not (output / "STOP").exists()
    assert not (output / "agent-priority-pause.json").exists()
    assert "--follow-source" in calls[0]


def test_prepare_only_never_executes_an_existing_evaluation_queue(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from pitbench.harness.harness import campaign

    monkeypatch.setattr(campaign, "_study_evaluations", lambda *a: pytest.fail("prepare-only dispatched evaluation"))
    campaign.study_evaluations(SimpleNamespace(source_cohort=None, prepare_only=True))


@pytest.mark.parametrize("legacy", [True, False])
def test_auto_launcher_joins_only_a_legacy_static_manager(tmp_path, monkeypatch, legacy):
    from pitbench.harness.harness import campaign

    profile = tmp_path / "profile.yaml"
    profile.write_text(yaml.safe_dump({
        "phase": "study-evaluations", "background": True,
        "manifest": str(tmp_path / "manifest.json"), "config": str(tmp_path / "evaluation.yaml"),
        "output": str(tmp_path / "output"), "source_cohort": str(tmp_path / "source"),
        "study_plan": str(tmp_path / "plan.json"), "study_views": str(tmp_path / "views.json"),
    }))
    calls = []
    def existing(args):
        return {"pid": 1, "dynamic_evaluation_workers": not legacy} if not args.worker_group else None
    def configure(args):
        args.workers = args.evaluation_workers = 1
        args.dynamic_evaluation_workers = True
        args.evaluation_cpu_pools = ["0"]
    monkeypatch.setattr(campaign, "active_study_launcher", existing)
    monkeypatch.setattr(campaign, "configure_study_parallelism", configure)
    monkeypatch.setattr(campaign, "launch_background", lambda args, argv: calls.append((args.worker_group, list(argv))))
    campaign.main(["--profile", str(profile)])
    if legacy:
        assert calls[0][0] == "elastic"
        assert "--worker-group" in calls[0][1]
    else:
        assert not calls


def test_native_reports_only_bypasses_all_resource_and_solver_dispatch(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from pitbench.cli.main import app
    from pitbench.harness.harness import campaign

    output = tmp_path / "source"
    campaign.write_json(output / "cohort.json", {"jobs": []})
    profile = tmp_path / "evaluation.yaml"
    profile.write_text(yaml.safe_dump({"phase": "study-evaluations", "output": str(output),
        "manifest": str(tmp_path / "manifest.json"), "config": str(tmp_path / "config.yaml")}))
    calls = []
    monkeypatch.setattr(campaign, "configure_study_parallelism", lambda *a: pytest.fail("report-only started evaluation scheduling"))
    monkeypatch.setattr(campaign, "rebuild_study_reports", lambda args: calls.append(args))
    result = CliRunner().invoke(app, ["evaluate-patches", "--config", str(profile), "--reports-only", "--foreground", "--report-output", str(tmp_path / "reports")])
    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    assert calls[0].phase == "study-reports"
    assert calls[0].report_output == tmp_path / "reports"


def test_report_worker_is_process_picklable_and_uses_only_retained_data(tmp_path, monkeypatch):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from pitbench.evaluator.evaluator import PitBenchEvaluator
    from pitbench.evaluator.storage import ObservationStore
    from pitbench.harness.evaluation import EvaluationRequest
    from pitbench.harness.harness.campaign import rebuild_study_condition_reports

    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    source, target = tmp_path / "source", tmp_path / "reports"
    measured = source / "measurements"
    measured.mkdir(parents=True)
    patch = measured / "candidate.patch"
    patch.write_bytes(b"")
    task_id = "vroom_v1_15_0"
    context = {"task_config_path": str(ROOT / "configs/tasks/vroom_v1_15_0.yaml"), "fixture_mode": True}
    request = EvaluationRequest(task_id=task_id, task_path=ROOT, candidate_patch_path=patch,
        output_dir=measured, agent_name="fixture", evaluator_config=context)
    PitBenchEvaluator().evaluate(request)
    ObservationStore.write_jsonl(measured / "judge-observations.jsonl", ObservationStore.read(measured / "trials.parquet"))
    from pitbench.evaluator.artifacts import write_json
    write_json(source / "agent-status/run/status.json", {"status": "evaluation_failed", "evaluation_output": str(measured)})
    write_json(source / "runs/run/results.json", {"results": [{"task_id": task_id, "trial_name": "trial"}]})
    manifest = source / "workspaces/run" / task_id / "task.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(yaml.safe_dump({"evaluator_config": context}))
    job = {"run_id": "run", "task_id": task_id, "model": {"agent": "fixture", "model": "fixture"}}
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as executor:
        result = executor.submit(rebuild_study_condition_reports, job, source, target, None).result(timeout=30)
    assert result["status"] in {"evaluation_completed", "evaluation_incomplete"}
    receipt = json.loads((target / "evaluations/run/report-rebuild.json").read_text())
    assert receipt["new_solver_runs"] == receipt["new_agent_starts"] == 0
