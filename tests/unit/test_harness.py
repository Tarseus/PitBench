from __future__ import annotations

import io
import json
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from pitbench.harness.agents.antigravity_mcp_agent import AntigravityMCPAgent
from pitbench.harness.agents.codex_mcp_agent import CodexMCPAgent
from pitbench.harness.harness.candidate_evaluation import CandidateEvaluator
from pitbench.harness.harness.harness import Harness
from pitbench.harness.harness.models import BenchmarkResults, TrialResults
from pitbench.harness.harness.run_artifacts import RunArtifactStore
from pitbench.harness.utils.progress import AgentActivity, TaskProgressDisplay

# Tests consolidated from tests/unit/harness/test_progress.py

"""Tests for TaskProgressDisplay and summary progress functionality."""


def test_task_progress_display_summary_row():
    current_time = [100.0]

    def get_time():
        current_time[0] += 1.0
        return current_time[0]

    console = MagicMock()
    console.get_time = get_time
    console.is_jupyter = False
    console.is_terminal = False
    console.is_dumb_terminal = True

    display = TaskProgressDisplay(console=console, get_time=get_time)
    display.add("task-1", "task-1.trial-1")
    display.add("task-2", "task-2.trial-1")

    assert display.summary_row is not None
    assert display.total_trials == 2

    summary_task = display.progress.tasks[display.summary_row]
    assert summary_task.fields["is_summary"] is True
    assert summary_task.total == 2
    assert summary_task.fields["queued"] == 2

    display.start()

    # Stage update
    display.stage("task-1", "task-1.trial-1", "agent.execute", "started")
    assert summary_task.fields["running"] == 1
    assert summary_task.fields["queued"] == 1

    # Finish task-1 successfully
    result1 = SimpleNamespace(
        task_id="task-1",
        trial_name="task-1.trial-1",
        failure_mode="none",
        evaluation=SimpleNamespace(completed=True),
    )
    display.finish(result1)

    assert summary_task.completed == 1
    assert summary_task.fields["passed"] == 1
    assert summary_task.fields["running"] == 0

    # Finish task-2 with failure
    result2 = SimpleNamespace(
        task_id="task-2",
        trial_name="task-2.trial-1",
        failure_mode="timeout",
        evaluation=None,
    )
    display.finish(result2)

    assert summary_task.completed == 2
    assert summary_task.fields["passed"] == 1
    assert summary_task.fields["failed"] == 1

    display.stop()


# Tests consolidated from tests/unit/utils/test_progress.py


def display(clock=None, width=150):
    stream = io.StringIO()
    progress = TaskProgressDisplay(
        console=Console(file=stream, width=width),
        get_time=(lambda: clock[0]) if clock is not None else None,
    )
    return progress, stream


def rendered(progress, stream):
    stream.seek(0)
    stream.truncate()
    progress.progress.console.print(progress.progress.get_renderable())
    return stream.getvalue()


def test_task_rows_and_stage_eta_are_independent():
    clock = [0.0]
    progress, stream = display(clock)
    progress.add("routing", "routing.1")
    progress.add("linear", "linear.1")
    progress.stage(
        "routing",
        "routing.1",
        "agent.execute",
        "started",
        execution={"timeout_seconds": 3600},
    )
    progress.detail("routing", "routing.1", "Agent: Reading · rounds 2 · tools 3")
    progress.detail("linear", "linear.1", "Judge plan: 10 instances, 100 solver runs")
    clock[0] = 200  # Compilation must not inflate the solver-stage ETA.
    progress.detail(
        "linear", "linear.1", "Judge progress: solver runs 0/100, workers 2"
    )
    clock[0] = 220
    progress.detail(
        "linear", "linear.1", "Judge progress: solver runs 20/100, valid 20"
    )
    text = rendered(progress, stream)
    assert "routing" in text and "linear" in text
    assert "Reading" in text and "rounds 2" in text
    assert "20/100" in text
    assert "ETA ~01:20" in text
    assert "limit 1:00:00" in text
    assert progress.rows["routing.1"].total is None
    assert progress.rows["linear.1"].grid_started == 200


def test_phase_changes_clear_eta_and_restore_indeterminate_bar():
    clock = [0.0]
    progress, stream = display(clock)
    progress.add("task", "trial")
    progress.detail("task", "trial", "Judge progress: solver runs 0/10")
    clock[0] = 10
    progress.detail("task", "trial", "Judge progress: solver runs 5/10")
    assert "ETA ~00:10" in rendered(progress, stream)
    progress.stage("task", "trial", "agent.execute", "started")
    state = progress.rows["trial"]
    assert state.total is None and state.grid_started is None
    assert "ETA ~" not in rendered(progress, stream)
    bar = next(
        column
        for column in progress.progress.columns
        if type(column).__name__ == "_StageBarColumn"
    )
    assert bar.render(progress.progress.tasks[0]).pulse is True


def test_multiple_attempts_and_agent_subtrials_are_routed_without_cross_talk():
    progress, _ = display()
    progress.add("task", "attempt.1", attempt=1, attempts=2)
    progress.add("task", "attempt.2", attempt=2, attempts=2)
    progress.detail("task", "attempt.1.agent-1-example", "Agent: Reading")
    progress.detail("task", "unknown-attempt", "Agent: Editing")
    assert progress.rows["attempt.1"].phase == "Coding agent"
    assert progress.rows["attempt.2"].phase == "Queued"
    row1 = progress.rows["attempt.1"].row
    row2 = progress.rows["attempt.2"].row
    assert progress.progress.tasks[row1].fields["detail"] == "Reading"
    assert progress.progress.tasks[row2].description == "task [2/2]"


def test_failure_retains_partial_counts_and_late_events_do_not_restart_it():
    clock = [0.0]
    progress, stream = display(clock)
    progress.add("task", "trial")
    progress.detail("task", "trial", "Judge progress: solver runs 0/10")
    clock[0] = 10
    progress.detail("task", "trial", "Judge progress: solver runs 3/10")
    progress.stage("task", "trial", "evaluator.execute", "failed")
    progress.finish(
        TrialResults(
            task_id="task",
            trial_name="trial",
            instruction="",
            failure_mode="unknown_agent_error",
        )
    )
    progress.detail("task", "trial", "Judge progress: solver runs 10/10")
    assert progress.rows["trial"].completed == 3
    assert progress.progress.tasks[progress.rows["trial"].row].completed == 0
    assert "Failed" in rendered(progress, stream)
    assert "ETA ~" not in rendered(progress, stream)


def test_stopping_run_marks_queued_and_active_rows_without_completion():
    progress, stream = display()
    progress.add("active", "active.1")
    progress.add("queued", "queued.1")
    progress.stage("active", "active.1", "agent.execute", "started")
    progress.stop()
    assert all(state.outcome == "Stopped" for state in progress.rows.values())
    assert "Done" not in rendered(progress, stream)


def test_small_terminal_keeps_task_stage_and_eta_readable():
    progress, stream = display(width=90)
    progress.add("pyvrp_v0_14_0", "trial")
    progress.stage("pyvrp_v0_14_0", "trial", "agent.execute", "started")
    progress.detail("pyvrp_v0_14_0", "trial", "Agent: Reading · rounds 2 · tools 3")
    text = rendered(progress, stream)
    assert (
        "pyvrp_v0_14_0" in text
        and "Coding agent" in text
        and "Reading" in text
        and "ETA" in text
    )


def test_agent_event_activity_deduplicates_updates_and_hides_arguments():
    activity = AgentActivity()

    def feed(state):
        return activity.feed(
            json.dumps(
                {
                    "event": "step_update",
                    "step_update": {
                        "conversation_id": "session",
                        "step_index": 2,
                        "step_type": "tool",
                        "state": state,
                        "tool_name": "call_mcp_tool",
                        "tool_info": {
                            "parameters": {
                                "ToolName": "run_command",
                                "Arguments": json.dumps(
                                    {
                                        "command": "python -m pytest",
                                        "secret": "never-display",
                                    }
                                ),
                            }
                        },
                    },
                }
            )
        )

    assert "Testing" in feed("ACTIVE")
    assert feed("ACTIVE") is None
    assert "tools 1" in feed("DONE")
    assert feed("DONE") is None
    assert activity.tools == 1
    text = activity.feed(
        json.dumps(
            {
                "type": "item.started",
                "item": {
                    "type": "reasoning",
                    "text": "private reasoning must never be displayed",
                },
            }
        )
    )
    assert text == "Agent: Planning · rounds 0 · tools 1"
    assert activity.feed("not json") is None


@pytest.mark.parametrize("agent_type", [AntigravityMCPAgent, CodexMCPAgent])
def test_output_is_reported_while_process_is_still_waiting(tmp_path, agent_type):
    ready = tmp_path / "callback-received"
    script = """
import sys,time
from pathlib import Path
print('{"type":"turn.started"}',flush=True)
deadline=time.monotonic()+3
while not Path(sys.argv[1]).exists() and time.monotonic()<deadline: time.sleep(0.01)
if not Path(sys.argv[1]).exists(): raise SystemExit(3)
print('diagnostic',file=sys.stderr,flush=True)
print('{"type":"item.completed","item":{"type":"file_change","id":"edit-1"}}',flush=True)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(ready)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    agent = agent_type(model_name="test")
    messages = []

    def received(detail):
        messages.append(detail)
        ready.touch()

    agent.set_progress_callback(received)
    stdout, stderr, code = agent._stream_process(process, "{}", timeout_sec=5)
    assert code == 0 and ready.exists()
    assert "file_change" in stdout and stderr == "diagnostic\n"
    assert messages[0].startswith("Agent: Waiting for model")
    assert messages[-1] == "Agent: Editing · rounds 0 · tools 1"


@pytest.mark.parametrize("agent_type", [AntigravityMCPAgent, CodexMCPAgent])
def test_stream_timeout_preserves_output_and_stops_process(agent_type):
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; print('partial',flush=True); time.sleep(30)",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    agent = agent_type(model_name="test")
    agent._set_process(process)
    stdout, _, code = agent._stream_process(process, "{}", timeout_sec=0.2)
    assert agent._cancelled.is_set()
    assert code != 0 and process.poll() is not None
    assert stdout == "partial\n"


def test_concurrent_harness_dispatch_has_one_row_per_task(monkeypatch):
    class Dataset(list):
        @property
        def task_ids(self):
            return [path.name for path in self]

    progress, _ = display()
    monkeypatch.setattr(
        "pitbench.harness.harness.harness.TaskProgressDisplay", lambda: progress
    )
    harness = Harness.__new__(Harness)
    harness._dataset = Dataset([Path("one"), Path("two")])
    harness._is_resuming = harness._livestream = False
    harness._progress_bar = True
    harness._n_concurrent_trials = 2
    harness._n_attempts = 1
    harness._run_id = "run"
    harness._trace_pipeline = lambda **kwargs: None
    harness._run_artifacts = SimpleNamespace(write_results=lambda result: None)
    barrier = threading.Barrier(2)

    def execute(trial_name, task_path):
        barrier.wait(timeout=3)
        harness._update_progress_from_stage(
            stage="agent.execute",
            status="started",
            task_id=task_path.name,
            trial_name=trial_name,
        )
        harness._update_progress_detail(task_path.name, trial_name, "Agent: Reading")
        return TrialResults(
            task_id=task_path.name, trial_name=trial_name, instruction=""
        )

    harness._execute_single_trial = execute
    result = harness._execute_tasks()
    assert len(result.results) == len(progress.rows) == 2
    assert all(state.outcome == "Done" for state in progress.rows.values())
    assert harness._progress_display is None


def test_run_artifact_store_owns_result_writes_and_resume_loading(tmp_path):
    run_path = tmp_path / "run"
    aggregate_dir = run_path / "task" / "task.1-of-1.run"
    agent_dir = run_path / "task" / "task.1-of-1.run.agent-1-codex"
    aggregate_dir.mkdir(parents=True)
    agent_dir.mkdir()
    aggregate = TrialResults(
        task_id="task", trial_name=aggregate_dir.name, instruction="improve"
    )
    subtrial = TrialResults(
        task_id="task", trial_name=agent_dir.name, instruction="improve"
    )
    (aggregate_dir / "results.json").write_text(aggregate.model_dump_json())
    (agent_dir / "results.json").write_text(subtrial.model_dump_json())
    trace = MagicMock()
    logger = MagicMock()
    store = RunArtifactStore(
        run_path=run_path,
        run_id="run",
        results_path=run_path / "results.json",
        metadata_path=run_path / "run_metadata.json",
        s3_bucket=None,
        logger=logger,
        trace=trace,
    )

    loaded = store.load_previous_results()
    assert loaded is not None
    assert [item.trial_name for item in loaded.results] == [aggregate.trial_name]
    store.write_results(BenchmarkResults(results=[aggregate]))
    assert BenchmarkResults.model_validate_json(
        store.results_path.read_text()
    ).results == [aggregate]
    trace.assert_called_once()
    store.handle_upload(True)
    logger.warning.assert_called()

    partial = run_path / "unfinished" / "unfinished.1-of-1.run"
    partial.mkdir(parents=True)
    dataset = SimpleNamespace(_tasks=[Path("task"), Path("unfinished")])
    store.filter_completed_tasks(
        dataset=dataset,
        attempts=1,
        trial_name=lambda path, attempt: f"{path.name}.{attempt}-of-1.run",
    )
    assert dataset._tasks == [Path("unfinished")]
    assert not (run_path / "unfinished").exists()
    assert list((run_path / "interrupted").glob("*/unfinished"))


def test_candidate_evaluator_owns_remote_snapshot_metadata():
    logger = MagicMock()
    evaluator = CandidateEvaluator(
        defer_evaluation=False,
        remote_build=True,
        snapshot_bucket="snapshots",
        logger=logger,
        trace=lambda **kwargs: None,
        progress_stage=lambda **kwargs: None,
        progress_detail=lambda *args: None,
    )
    terminal = MagicMock()
    results = TrialResults(task_id="task", trial_name="trial", instruction="")
    evaluator.maybe_save_snapshot(
        terminal=terminal,
        trial_handler=SimpleNamespace(task_id="task"),
        results=results,
        snapshot_name="setup",
        snapshot_s3_key="run/task/setup.tar.gz",
        reason="setup failure",
    )
    terminal.save_container_image.assert_called_once_with(
        snapshot_s3_key="run/task/setup.tar.gz"
    )
    assert results.evaluation_snapshot_bucket_name == "snapshots"
    assert results.evaluation_snapshot_s3_keys == {"setup": "run/task/setup.tar.gz"}
