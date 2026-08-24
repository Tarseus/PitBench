from pathlib import Path

import pytest

from pitbench.evaluator.evaluator import PitBenchEvaluator
from pitbench.evaluator.patch_policy import PatchPolicy
from pitbench.evaluator.storage import ObservationStore
from pitbench.harness.evaluation import EvaluationRequest
from pitbench.schema.observation import CodeState
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
            output_dir=output,
            agent_name="test",
            evaluator_config={"manifest_path": str(record.manifest_path)},
        )
    )
    assert envelope.completed is False
    assert "real judge missing configuration" in (envelope.error or "")


def test_model_build_fixture_uses_model_size_decision_path(tmp_path: Path) -> None:
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
    assert decision["policy_name"] == ("pitbench-model-build-pareto-gated-improvement")
    assert decision["outcome_complete"] is True
    assert decision["resource_telemetry_complete"] is True
    assert decision["paired_model_runs"] == 1
    assert decision["model_variable_ratio"] < 1
    assert decision["model_constraint_ratio"] < 1
    assert decision["classification"] == "improved"
    assert decision["is_resolved"] is True
    assert envelope.is_resolved is True


def test_patch_policy_protects_judge_surfaces(tmp_path: Path) -> None:
    patch = tmp_path / "candidate.patch"
    patch.write_text(
        "diff --git a/pitbench/evaluator/judge.py b/pitbench/evaluator/judge.py\n"
        "--- a/pitbench/evaluator/judge.py\n"
        "+++ b/pitbench/evaluator/judge.py\n"
    )
    result = PatchPolicy().inspect(patch)
    assert result.accepted is False
    assert result.violations == ["protected path: pitbench/evaluator/judge.py"]
