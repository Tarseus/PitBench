import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from pitbench.evaluator.evaluator import PitBenchEvaluator
from pitbench.evaluator.storage import ObservationStore
from pitbench.harness.evaluation import EvaluationRequest
from pitbench.schema.observation import CodeState, RunObservation, RunStatus
from pitbench.tasks import TaskCatalog

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("record", TaskCatalog(ROOT).validate_all())
def test_every_task_runs_explicit_fixture_grid(record, tmp_path: Path) -> None:
    output = tmp_path / record.task.task_id
    output.mkdir()
    patch = output / "candidate.patch"
    patch.write_text("")
    envelope = PitBenchEvaluator().envelope(
        EvaluationRequest(
            task_id=record.task.task_id,
            task_path=ROOT,
            candidate_patch_path=patch,
            candidate_patch_sha256=hashlib.sha256(patch.read_bytes()).hexdigest(),
            output_dir=output,
            agent_name="fixture",
            evaluator_config={
                "manifest_path": str(record.manifest_path),
                "fixture_mode": True,
                "fixture_instances_per_population": 1,
            },
        )
    )
    assert envelope.completed, envelope.error
    observations = ObservationStore.read(output / "trials.parquet")
    assert observations
    assert {item.code_state for item in observations} == set(CodeState)
    assert all(item.task_id == record.task.task_id for item in observations)
    summary = envelope.payload["summary"]
    assert summary["performance"] is not None
    assert "outcomes" not in summary
    assert "sensitivity" not in summary
    assert "behavior" not in summary


def test_real_evaluation_without_private_assets_fails_closed(tmp_path: Path) -> None:
    record = TaskCatalog(ROOT).records()[0]
    output = tmp_path / "output"
    output.mkdir()
    patch = output / "candidate.patch"
    patch.write_text("")
    envelope = PitBenchEvaluator().envelope(
        EvaluationRequest(
            task_id=record.task.task_id,
            task_path=ROOT,
            candidate_patch_path=patch,
            candidate_patch_sha256=hashlib.sha256(patch.read_bytes()).hexdigest(),
            output_dir=output,
            agent_name="test",
            evaluator_config={"manifest_path": str(record.manifest_path)},
        )
    )
    assert envelope.completed is False
    assert "real judge missing configuration" in (envelope.error or "")


def test_real_evaluation_requires_captured_patch_identity(tmp_path: Path) -> None:
    record = TaskCatalog(ROOT).records()[0]
    output = tmp_path / "output"
    output.mkdir()
    candidate = output / "candidate.patch"
    candidate.write_text("")

    envelope = PitBenchEvaluator().envelope(
        EvaluationRequest(
            task_id=record.task.task_id,
            task_path=ROOT,
            candidate_patch_path=candidate,
            output_dir=output,
            agent_name="test",
            evaluator_config={"manifest_path": str(record.manifest_path)},
        )
    )

    assert envelope.completed is False
    assert "real judge requires candidate_patch_sha256" in (envelope.error or "")


def test_evaluator_rejects_candidate_patch_identity_mismatch(
    tmp_path: Path,
) -> None:
    record = TaskCatalog(ROOT).records()[0]
    output = tmp_path / "output"
    output.mkdir()
    candidate = output / "candidate.patch"
    candidate.write_text("")

    envelope = PitBenchEvaluator().envelope(
        EvaluationRequest(
            task_id=record.task.task_id,
            task_path=ROOT,
            candidate_patch_path=candidate,
            candidate_patch_sha256="f" * 64,
            output_dir=output,
            agent_name="test",
            evaluator_config={
                "manifest_path": str(record.manifest_path),
                "fixture_mode": True,
            },
        )
    )

    assert envelope.completed is False
    assert "candidate patch identity mismatch" in (envelope.error or "")


def test_evaluator_rejects_repository_snapshot_mismatch_before_judge(
    tmp_path: Path,
) -> None:
    record = TaskCatalog(ROOT).records()[0]
    output = tmp_path / "output"
    output.mkdir()
    candidate = output / "candidate.patch"
    candidate.write_text("")
    base_repository = tmp_path / "base"
    private_root = tmp_path / "private"
    base_repository.mkdir()
    private_root.mkdir()
    request = EvaluationRequest(
        task_id=record.task.task_id,
        task_path=ROOT,
        candidate_patch_path=candidate,
        candidate_patch_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
        output_dir=output,
        agent_name="test",
        evaluator_config={
            "manifest_path": str(record.manifest_path),
            "base_repository": str(base_repository),
            "private_root": str(private_root),
            "judge_image": "sha256:" + "a" * 64,
        },
    )

    with (
        patch(
            "pitbench.evaluator.evaluator.PitBenchAdapter.validate_repository",
            side_effect=ValueError("repository release identity mismatch"),
        ) as validate_repository,
        patch("pitbench.evaluator.evaluator.DockerJudge") as docker_judge,
    ):
        envelope = PitBenchEvaluator().envelope(request)

    assert envelope.completed is False
    assert "repository release identity mismatch" in (envelope.error or "")
    validate_repository.assert_called_once_with(record.task, base_repository)
    docker_judge.assert_not_called()


def test_real_evaluation_ignores_legacy_local_judge_flag(tmp_path: Path) -> None:
    record = TaskCatalog(ROOT).records()[0]
    output = tmp_path / "output"
    output.mkdir()
    candidate = output / "candidate.patch"
    candidate.write_text("")
    base_repository = tmp_path / "base"
    private_root = tmp_path / "private"
    base_repository.mkdir()
    private_root.mkdir()
    request = EvaluationRequest(
        task_id=record.task.task_id,
        task_path=ROOT,
        candidate_patch_path=candidate,
        candidate_patch_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
        output_dir=output,
        agent_name="test",
        evaluator_config={
            "manifest_path": str(record.manifest_path),
            "base_repository": str(base_repository),
            "private_root": str(private_root),
            "judge_image": "sha256:" + "a" * 64,
            "unsafe_local_judge": True,
        },
    )

    with (
        patch("pitbench.evaluator.evaluator.PitBenchAdapter.validate_repository"),
        patch("pitbench.evaluator.evaluator.DockerJudge") as docker_judge,
    ):
        docker_judge.return_value.run.return_value = []
        envelope = PitBenchEvaluator().envelope(request)

    assert envelope.completed, envelope.error
    docker_judge.assert_called_once()


def test_cached_base_observations_merge_with_agent_fixture(tmp_path: Path) -> None:
    record = next(
        item
        for item in TaskCatalog(ROOT).validate_all()
        if item.task.task_id == "pyvrp_v0_14_0"
    )
    patch = tmp_path / "candidate.patch"
    patch.write_text("")
    base_dir = tmp_path / "base"
    agent_dir = tmp_path / "agent"
    base_dir.mkdir()
    agent_dir.mkdir()
    common = {
        "manifest_path": str(record.manifest_path),
        "fixture_mode": True,
        "fixture_instances_per_population": 1,
    }
    base = PitBenchEvaluator().envelope(
        EvaluationRequest(
            task_id=record.task.task_id,
            task_path=ROOT,
            candidate_patch_path=patch,
            output_dir=base_dir,
            agent_name="base",
            evaluator_config={**common, "code_states": ["base"]},
        )
    )
    assert base.completed, base.error

    agent = PitBenchEvaluator().envelope(
        EvaluationRequest(
            task_id=record.task.task_id,
            task_path=ROOT,
            candidate_patch_path=patch,
            output_dir=agent_dir,
            agent_name="agent",
            evaluator_config={
                **common,
                "code_states": ["agent"],
                "base_observations_path": str(base_dir / "trials.parquet"),
            },
        )
    )

    assert agent.completed, agent.error
    observations = ObservationStore.read(agent_dir / "trials.parquet")
    assert {item.code_state for item in observations} == set(CodeState)


def test_model_equivalence_fixture_uses_performance_decision_path(
    tmp_path: Path,
) -> None:
    record = next(
        record
        for record in TaskCatalog(ROOT).validate_all()
        if record.task.task_id == "ortools_v9_15"
    )
    output = tmp_path / record.task.task_id
    output.mkdir()
    patch = output / "candidate.patch"
    patch.write_text("")

    envelope = PitBenchEvaluator().envelope(
        EvaluationRequest(
            task_id=record.task.task_id,
            task_path=ROOT,
            candidate_patch_path=patch,
            output_dir=output,
            agent_name="fixture",
            evaluator_config={
                "manifest_path": str(record.manifest_path),
                "fixture_mode": True,
                "fixture_instances_per_population": 1,
            },
        )
    )

    assert envelope.completed, envelope.error
    decision = envelope.payload["summary"]["decision"]
    assert decision["policy_name"] == "pitbench-performance-first"
    assert decision["performance_complete"] is False
    assert decision["classification"] == "incomplete"
    assert decision["is_resolved"] is False
    assert envelope.is_resolved is False
    assert "outcome_complete" not in decision
    assert "resource_telemetry_complete" not in decision
    assert "sensitivity_complete" not in decision


def test_evaluator_does_not_gate_candidate_paths(tmp_path: Path) -> None:
    record = TaskCatalog(ROOT).records()[0]
    output = tmp_path / "output"
    output.mkdir()
    candidate = output / "candidate.patch"
    candidate.write_text(
        "diff --git a/pitbench/evaluator/judge.py b/pitbench/evaluator/judge.py\n"
        "--- a/pitbench/evaluator/judge.py\n"
        "+++ b/pitbench/evaluator/judge.py\n"
    )
    envelope = PitBenchEvaluator().envelope(
        EvaluationRequest(
            task_id=record.task.task_id,
            task_path=ROOT,
            candidate_patch_path=candidate,
            candidate_patch_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
            output_dir=output,
            agent_name="fixture",
            evaluator_config={
                "manifest_path": str(record.manifest_path),
                "fixture_mode": True,
                "fixture_instances_per_population": 1,
            },
        )
    )

    assert envelope.completed is True
    assert envelope.payload["validity"]["accepted"] is True
    assert envelope.payload["summary"]["observation_count"] > 0


def test_semantically_invalid_agent_run_disqualifies_candidate(
    tmp_path: Path,
) -> None:
    record = TaskCatalog(ROOT).records()[0]
    output = tmp_path / "output"
    output.mkdir()
    candidate = output / "candidate.patch"
    candidate.write_text("")
    invalid = RunObservation(
        task_id=record.task.task_id,
        code_state=CodeState.AGENT,
        population="judge_id",
        population_kind="judge_id",
        instance_id="invalid",
        instance_seed=0,
        solver_seed=0,
        budget_sec=1.0,
        status=RunStatus.INVALID,
        valid=False,
    )

    with patch(
        "pitbench.evaluator.evaluator.FixtureJudge.run",
        return_value=[invalid],
    ):
        envelope = PitBenchEvaluator().envelope(
            EvaluationRequest(
                task_id=record.task.task_id,
                task_path=ROOT,
                candidate_patch_path=candidate,
                output_dir=output,
                agent_name="fixture",
                evaluator_config={
                    "manifest_path": str(record.manifest_path),
                    "fixture_mode": True,
                },
            )
        )

    assert envelope.completed is True
    assert envelope.payload["validity"]["accepted"] is False
    assert envelope.payload["validity"]["checks"][-1]["code"] == "solution"
    assert envelope.payload["observations"][0]["status"] == "invalid"
    assert envelope.payload["summary"]["performance"] is not None
    assert envelope.payload["summary"]["decision"]["is_resolved"] is False
