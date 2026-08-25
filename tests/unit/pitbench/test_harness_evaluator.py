from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from pydantic import BaseModel

from pitbench.harness.evaluation import EvaluationRequest, Evaluator
from pitbench.harness.handlers.trial_handler import TrialPaths
from pitbench.harness.harness.harness import Harness
from pitbench.harness.harness.models import TrialResults
from pitbench.harness.utils.pipeline_trace import PipelineTrace


class EchoResult(BaseModel):
    task_id: str
    patch_size: int


class EchoEvaluator(Evaluator):
    name = "echo"

    def evaluate(self, request: EvaluationRequest) -> EchoResult:
        return EchoResult(
            task_id=request.task_id,
            patch_size=request.candidate_patch_path.stat().st_size,
        )


class VerdictResult(BaseModel):
    is_resolved: bool


class VerdictEvaluator(Evaluator):
    name = "verdict"

    def evaluate(self, request: EvaluationRequest) -> VerdictResult:
        return VerdictResult(is_resolved=True)


class FakeContainer:
    attrs = {"Config": {"WorkingDir": "/workspace/repo"}}

    def exec_run(self, command, workdir=None):
        assert command[-1].endswith("git diff --binary --no-ext-diff HEAD")
        assert workdir == "/workspace/repo"
        return SimpleNamespace(exit_code=0, output=b"diff --git a/a.py b/a.py\n")


def test_harness_treats_evaluator_payload_as_opaque(tmp_path: Path) -> None:
    paths = TrialPaths(tmp_path, "task", "trial")
    paths.mkdir()
    handler = SimpleNamespace(
        task_id="task",
        trial_name="trial",
        task_paths=SimpleNamespace(input_path=tmp_path),
        trial_paths=paths,
        task=SimpleNamespace(
            evaluator_import_path=(
                "tests.unit.pitbench.test_harness_evaluator:EchoEvaluator"
            ),
            evaluator_config={},
        ),
    )
    results = TrialResults(trial_name="trial", task_id="task", instruction="test")
    harness = Harness.__new__(Harness)
    harness._pipeline_trace = PipelineTrace(tmp_path / "trace.jsonl", "run")

    harness._evaluate_candidate(
        terminal=SimpleNamespace(container=FakeContainer()),
        trial_handler=handler,
        results=results,
        agent_label="nop",
        model_name=None,
    )

    assert results.evaluation is not None
    assert results.evaluation.completed is True
    assert results.evaluation.payload == {"task_id": "task", "patch_size": 25}
    assert results.is_resolved is None
    assert (
        (paths.task_output_path / "evaluation/candidate.patch")
        .read_bytes()
        .startswith(b"diff --git")
    )


def test_harness_consumes_standard_evaluator_verdict_without_parsing_payload(
    tmp_path: Path,
) -> None:
    paths = TrialPaths(tmp_path, "task", "trial")
    paths.mkdir()
    handler = SimpleNamespace(
        task_id="task",
        trial_name="trial",
        task_paths=SimpleNamespace(input_path=tmp_path),
        trial_paths=paths,
        task=SimpleNamespace(
            evaluator_import_path=(
                "tests.unit.pitbench.test_harness_evaluator:VerdictEvaluator"
            ),
            evaluator_config={},
        ),
    )
    results = TrialResults(trial_name="trial", task_id="task", instruction="test")
    harness = Harness.__new__(Harness)
    harness._pipeline_trace = PipelineTrace(tmp_path / "trace.jsonl", "run")

    harness._evaluate_candidate(
        terminal=SimpleNamespace(container=FakeContainer()),
        trial_handler=handler,
        results=results,
        agent_label="nop",
        model_name=None,
    )

    assert results.evaluation is not None
    assert results.evaluation.payload == {"is_resolved": True}
    assert results.is_resolved is True


def test_pipeline_stage_updates_non_livestream_progress_description() -> None:
    progress = Mock()
    harness = Harness.__new__(Harness)
    harness._progress_display = {
        "progress": progress,
        "task": 7,
        "completed": 0,
        "total": 1,
        "accuracy": 0.0,
    }

    harness._update_progress_from_stage(
        stage="agent.execute",
        status="started",
        task_id="pyvrp_v0_14_0",
        trial_name="trial-1",
    )

    progress.update.assert_called_once_with(
        7,
        description=(
            "Running tasks (0/1, Accuracy: 0.00%) — pyvrp_v0_14_0: Coding agent"
        ),
    )
