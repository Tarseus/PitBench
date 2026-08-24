import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pitbench.harness.agents.failure_mode import FailureMode
from pitbench.harness.utils.pipeline_trace import PipelineTrace


def _read_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_pipeline_trace_records_pipeline_contract(tmp_path: Path) -> None:
    trace_path = tmp_path / "pipeline_trace.jsonl"
    trace = PipelineTrace(trace_path, "run-1")

    trace.record(
        stage="tests.execute",
        status="completed",
        inputs={"script": tmp_path / "run-tests.sh"},
        outputs={"failure_mode": FailureMode.NONE, "captured_pane": "ok"},
        execution={"component": "Harness._run_tests", "command": "bash /tests"},
        task_id="task-1",
        trial_name="task-1.1",
    )

    events = _read_events(trace_path)
    assert events == [
        {
            "sequence": 1,
            "timestamp": events[0]["timestamp"],
            "run_id": "run-1",
            "task_id": "task-1",
            "trial_name": "task-1.1",
            "stage": "tests.execute",
            "status": "completed",
            "inputs": {"script": str(tmp_path / "run-tests.sh")},
            "outputs": {"failure_mode": "none", "captured_pane": "ok"},
            "execution": {
                "component": "Harness._run_tests",
                "command": "bash /tests",
            },
            "error": None,
        }
    ]


def test_pipeline_trace_redacts_credentials_but_keeps_token_metrics(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "pipeline_trace.jsonl"
    trace = PipelineTrace(trace_path, "run-1")

    trace.record(
        stage="agent.execute",
        status="completed",
        inputs={
            "api_key": "top-secret",
            "headers": {"Authorization": "Bearer abc.def"},
            "command": "API_KEY=top-secret tool --password=hunter2",
        },
        outputs={"total_input_tokens": 123, "total_output_tokens": 45},
    )

    event = _read_events(trace_path)[0]
    assert event["inputs"]["api_key"] == "[REDACTED]"
    assert event["inputs"]["headers"]["Authorization"] == "[REDACTED]"
    assert "top-secret" not in event["inputs"]["command"]
    assert "hunter2" not in event["inputs"]["command"]
    assert event["outputs"]["total_input_tokens"] == 123
    assert event["outputs"]["total_output_tokens"] == 45


def test_pipeline_trace_appends_with_thread_safe_sequence(tmp_path: Path) -> None:
    trace_path = tmp_path / "pipeline_trace.jsonl"
    trace = PipelineTrace(trace_path, "run-1")
    trace.record(stage="run.execute", status="started")

    resumed_trace = PipelineTrace(trace_path, "run-1", append=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(
            executor.map(
                lambda index: resumed_trace.record(
                    stage="trial.execute",
                    status="completed",
                    outputs={"index": index},
                ),
                range(12),
            )
        )

    events = _read_events(trace_path)
    assert [event["sequence"] for event in events] == list(range(1, 14))
    assert events[0]["stage"] == "run.execute"
    assert {event["outputs"]["index"] for event in events[1:]} == set(range(12))
