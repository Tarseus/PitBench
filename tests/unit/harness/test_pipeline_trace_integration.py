import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from pitbench.harness.agents.agent_name import AgentName
from pitbench.harness.agents.base_agent import AgentResult
from pitbench.harness.agents.failure_mode import FailureMode
from pitbench.harness.handlers.trial_handler import TrialPaths
from pitbench.harness.harness.harness import Harness
from pitbench.harness.harness.models import TrialResults
from pitbench.harness.utils.pipeline_trace import PipelineTrace
from tests.unit.pitbench.test_harness_evaluator import FakeContainer


class _Session:
    def __init__(self, name: str) -> None:
        self.name = name

    def capture_pane(self, capture_entire: bool = False) -> str:
        return f"captured output from {self.name}"

    def send_keys(self, *args, **kwargs) -> None:
        return None


class _Terminal:
    def __init__(self) -> None:
        self.closed_sessions: list[str] = []
        self.container = FakeContainer()

    def create_session(self, session_name: str, **kwargs) -> _Session:
        return _Session(session_name)

    def close_session(self, session_name: str) -> None:
        self.closed_sessions.append(session_name)


def test_trial_pipeline_records_stage_inputs_and_outputs(tmp_path: Path) -> None:
    setup_script = tmp_path / "run-setup.sh"
    setup_script.write_text("#!/bin/bash\necho setup\n")
    trial_paths = TrialPaths(tmp_path / "runs", "task-1", "trial-1")
    trial_paths.mkdir()
    trial_handler = SimpleNamespace(
        task_id="task-1",
        trial_name="trial-1",
        instruction="Optimize the task",
        task_paths=SimpleNamespace(
            run_setup_path=setup_script,
            input_path=tmp_path,
        ),
        trial_paths=trial_paths,
        task=SimpleNamespace(
            max_setup_timeout_sec=10,
            max_agent_timeout_sec=20,
            disable_asciinema=True,
            evaluator_import_path=(
                "tests.unit.pitbench.test_harness_evaluator:EchoEvaluator"
            ),
            evaluator_config={},
        ),
    )

    harness = Harness.__new__(Harness)
    harness._pipeline_trace = PipelineTrace(tmp_path / "pipeline.jsonl", "run-1")
    harness._livestream = False
    harness._global_setup_timeout_sec = None
    harness._global_agent_timeout_sec = None
    harness._global_timeout_multiplier = 1.0
    harness._agent_kwargs = {"api_key": "must-not-leak"}
    harness._agent_name = AgentName.NOP
    harness._agent_import_path = None
    harness._agent_class = Mock
    harness._model_name = "nop"
    harness._run_id = "run-1"
    harness._run_uuid = "uuid-1"
    harness._remote_build = False
    harness._evaluation_snapshots_bucket = None
    harness._logger = Mock()
    harness._run_setup = Mock(return_value=FailureMode.NONE)
    harness._create_agent_for_task = Mock(return_value=Mock())
    harness._run_agent = Mock(
        return_value=(
            AgentResult(total_input_tokens=3, total_output_tokens=2),
            FailureMode.NONE,
        )
    )
    harness._maybe_save_evaluation_snapshot = Mock()

    results = TrialResults(
        trial_name=trial_handler.trial_name,
        task_id=trial_handler.task_id,
        instruction=trial_handler.instruction,
    )
    terminal = _Terminal()

    harness._run_single_agent_trial_terminal(
        trial_handler=trial_handler,
        terminal=terminal,
        results=results,
        agent_name=AgentName.NOP,
        model_name="nop",
    )

    events = [
        json.loads(line)
        for line in harness._pipeline_trace.path.read_text().splitlines()
        if line
    ]
    assert [(event["stage"], event["status"]) for event in events] == [
        ("setup.execute", "started"),
        ("setup.execute", "completed"),
        ("agent.execute", "started"),
        ("agent.execute", "completed"),
        ("evaluator.execute", "started"),
        ("evaluator.execute", "completed"),
    ]
    assert events[0]["inputs"]["script"]["content"].endswith("echo setup\n")
    assert events[3]["outputs"]["captured_pane"] == "captured output from agent"
    evaluation = events[-1]["outputs"]["evaluation"]
    assert evaluation["completed"] is True
    assert evaluation["payload"]["patch_size"] == 25
    assert results.evaluation is not None
    assert results.evaluation.payload == evaluation["payload"]
    assert "must-not-leak" not in harness._pipeline_trace.path.read_text()
    assert terminal.closed_sessions == ["agent", "setup"]
