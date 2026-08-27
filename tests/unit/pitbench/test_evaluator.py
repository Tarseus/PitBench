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
            output_dir=output,
            agent_name="test",
            evaluator_config={"manifest_path": str(record.manifest_path)},
        )
    )
    assert envelope.completed is False
    assert "real judge missing configuration" in (envelope.error or "")


def test_model_build_fixture_uses_performance_decision_path(tmp_path: Path) -> None:
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


@pytest.mark.parametrize(
    "patch_text",
    [
        (
            "diff --git a/tests/test_solver.py b/tests/test_solver.py\n"
            "deleted file mode 100644\n"
            "--- a/tests/test_solver.py\n"
            "+++ /dev/null\n"
        ),
        (
            "diff --git a/tests/data.bin b/tests/data.bin\n"
            "index 1234567..7654321 100644\n"
            "GIT binary patch\n"
        ),
    ],
)
def test_patch_policy_protects_deleted_and_binary_paths(
    tmp_path: Path, patch_text: str
) -> None:
    patch = tmp_path / "candidate.patch"
    patch.write_text(patch_text)

    result = PatchPolicy().inspect(patch)

    assert result.accepted is False
    assert result.violations == [
        "protected path: tests/data.bin"
        if "data.bin" in patch_text
        else "protected path: tests/test_solver.py"
    ]
