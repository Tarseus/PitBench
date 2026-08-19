from pathlib import Path

from fceval.evaluation import EvaluationRequest
from pitbench.evaluator.evaluator import PitBenchEvaluator
from pitbench.tasks import TaskCatalog

ROOT = Path(__file__).resolve().parents[3]


def test_missing_candidate_is_a_typed_invalid_result(tmp_path: Path) -> None:
    record = TaskCatalog(ROOT).records()[0]
    envelope = PitBenchEvaluator().envelope(
        EvaluationRequest(
            task_id=record.task.task_id,
            task_path=ROOT,
            candidate_patch_path=tmp_path / "missing.patch",
            output_dir=tmp_path,
            agent_name="test",
            evaluator_config={
                "manifest_path": str(record.manifest_path),
                "fixture_mode": True,
            },
        )
    )
    assert envelope.completed is True
    assert envelope.payload["validity"]["accepted"] is False
    assert envelope.payload["observations"] == []
    assert envelope.payload["artifacts"]["candidate_patch"] is None
