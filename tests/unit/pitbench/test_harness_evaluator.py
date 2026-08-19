from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel

from fceval.evaluation import EvaluationRequest, Evaluator
from fceval.handlers.trial_handler import TrialPaths
from fceval.harness.harness import Harness
from fceval.harness.models import TrialResults
from fceval.utils.pipeline_trace import PipelineTrace


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
    assert (
        (paths.task_output_path / "evaluation/candidate.patch")
        .read_bytes()
        .startswith(b"diff --git")
    )
