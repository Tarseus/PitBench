"""Per-trial terminal progress and activity derived from agent event streams."""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.progress import (
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TextColumn,
)
from rich.progress_bar import ProgressBar
from rich.table import Column
from rich.text import Text

from pitbench.harness.utils.pipeline_trace import _redact_text


def duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return (
        f"{hours}:{minutes:02}:{seconds:02}" if hours else f"{minutes:02}:{seconds:02}"
    )


@dataclass
class TrialProgress:
    task_id: str
    trial_name: str
    row: int
    started: float | None = None
    ended: float | None = None
    phase: str = "Queued"
    outcome: str | None = None
    grid_started: float | None = None
    completed: int = 0
    total: int | None = None
    agent_budget: float | None = None


class _TimingColumn(ProgressColumn):
    def __init__(self, *, remaining: bool = False):
        super().__init__()
        self.remaining = remaining

    def render(self, task: Task) -> Text:
        if task.fields.get("is_summary"):
            now = task.get_time()
            started = task.fields.get("started")
            completed = task.completed
            total = task.total
            if not self.remaining:
                value = duration(now - started) if started is not None else "—"
                return Text(f"elapsed {value}", style="bold dim")
            if not completed or not total or completed >= total or started is None:
                return Text("ETA —", style="bold dim")
            elapsed = now - started
            remaining = (total - completed) * (elapsed / completed)
            return Text(f"ETA ~{duration(remaining)}", style="bold cyan")

        state: TrialProgress = task.fields["state"]
        now = task.get_time()
        started, ended = state.started, state.ended
        grid_started, completed, total = (
            state.grid_started,
            state.completed,
            state.total,
        )
        if not self.remaining:
            value = (
                duration((ended if ended is not None else now) - started)
                if started is not None
                else "—"
            )
            return Text(f"elapsed {value}", style="dim")
        if state.outcome or grid_started is None or not completed or total is None:
            return Text("ETA —", style="dim")
        remaining = (total - completed) * (now - grid_started) / completed
        return Text(f"ETA ~{duration(remaining)}", style="cyan")


class _CountColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        if task.fields.get("is_summary"):
            total = int(task.total) if task.total else 0
            completed = int(task.completed)
            return Text(f"{completed}/{total}", style="bold white")
        state: TrialProgress = task.fields["state"]
        if state.outcome:
            return Text(
                state.outcome, style="green" if state.outcome == "Done" else "red"
            )
        if state.total is not None:
            return Text(f"{state.completed}/{state.total}")
        if state.phase == "Coding agent" and state.agent_budget is not None:
            return Text(f"limit {duration(state.agent_budget)}", style="dim")
        return Text("—", style="dim")


class _StatusColumn(SpinnerColumn):
    def render(self, task: Task):
        if task.fields.get("is_summary"):
            if task.finished or (task.total and task.completed >= task.total):
                failed = task.fields.get("failed", 0)
                passed = task.fields.get("passed", 0)
                if failed == 0 and passed > 0:
                    return Text("✓", style="bold green")
                elif passed == 0 and failed > 0:
                    return Text("×", style="bold red")
                else:
                    return Text("•", style="bold yellow")
            started = task.fields.get("started")
            if started is None:
                return Text("·", style="dim")
            return super().render(task)
        state: TrialProgress = task.fields["state"]
        if state.outcome:
            return Text(
                "✓" if state.outcome == "Done" else "×",
                style="green" if state.outcome == "Done" else "red",
            )
        if state.started is None:
            return Text("·", style="dim")
        return super().render(task)


class _StageBarColumn(ProgressColumn):
    def render(self, task: Task) -> ProgressBar:
        if task.fields.get("is_summary"):
            total = task.total or 1
            completed = task.completed
            failed = task.fields.get("failed", 0)
            finished = task.finished or completed >= total
            return ProgressBar(
                total=total,
                completed=completed,
                width=16,
                pulse=not finished and completed == 0,
                animation_time=task.get_time(),
                complete_style="red" if (failed > 0 and finished) else "bar.complete",
                finished_style="green" if failed == 0 else "yellow",
            )
        state: TrialProgress = task.fields["state"]
        if state.outcome == "Done":
            return ProgressBar(
                total=1,
                completed=1,
                width=16,
                complete_style="green",
                finished_style="green",
            )
        if state.started is None:
            return ProgressBar(total=1, completed=0, width=16)
        return ProgressBar(
            total=state.total,
            completed=state.completed,
            width=16,
            pulse=state.total is None and state.outcome is None,
            animation_time=task.get_time(),
            complete_style="red" if state.outcome else "bar.complete",
        )


class _PhaseColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        if task.fields.get("is_summary"):
            passed = task.fields.get("passed", 0)
            failed = task.fields.get("failed", 0)
            running = task.fields.get("running", 0)
            queued = task.fields.get("queued", 0)
            parts = []
            if passed:
                parts.append(f"[green]Done: {passed}[/green]")
            if failed:
                parts.append(f"[red]Failed: {failed}[/red]")
            if running:
                parts.append(f"[cyan]Running: {running}[/cyan]")
            if queued:
                parts.append(f"[dim]Queued: {queued}[/dim]")
            detail_str = " · ".join(parts) if parts else "Initializing"
            res = Text.from_markup(detail_str, overflow="ellipsis")
            res.no_wrap = True
            return res
        phase, detail = task.fields["phase"], task.fields["detail"]
        return Text(
            phase + (f"\n{detail}" if detail else ""),
            style="cyan",
            no_wrap=True,
            overflow="ellipsis",
        )


class TaskProgressDisplay:
    """One independent row per task attempt; the bar measures its current stage."""

    STAGES = {
        "trial.load": "Preparing",
        "terminal.lifecycle": "Container",
        "setup.execute": "Setup",
        "agent.execute": "Coding agent",
        "candidate.capture": "Capture patch",
        "evaluator.execute": "Evaluation",
        "results.merge": "Finalizing",
    }

    def __init__(self, *, console: Console | None = None, get_time=None):
        kwargs = {"get_time": get_time} if get_time is not None else {}
        self.progress = Progress(
            _StatusColumn(finished_text="•"),
            TextColumn(
                "{task.description}", markup=False, table_column=Column(no_wrap=True)
            ),
            _PhaseColumn(
                table_column=Column(ratio=1, overflow="ellipsis", no_wrap=True)
            ),
            _StageBarColumn(),
            _CountColumn(),
            _TimingColumn(),
            _TimingColumn(remaining=True),
            console=console,
            refresh_per_second=4,
            expand=True,
            **kwargs,
        )
        self.rows: dict[str, TrialProgress] = {}
        self.summary_row: int | None = None
        self.total_trials = 0
        self._lock = threading.RLock()

    def _ensure_summary_row(self) -> None:
        if self.summary_row is None:
            self.summary_row = self.progress.add_task(
                "Overall Progress",
                total=0,
                completed=0,
                start=False,
                is_summary=True,
                passed=0,
                failed=0,
                running=0,
                queued=0,
                started=None,
            )

    def _update_summary_counts(self) -> None:
        if self.summary_row is None:
            return
        passed = sum(1 for s in self.rows.values() if s.outcome == "Done")
        failed = sum(
            1
            for s in self.rows.values()
            if s.outcome is not None and s.outcome != "Done"
        )
        running = sum(
            1 for s in self.rows.values() if s.started is not None and s.outcome is None
        )
        queued = sum(1 for s in self.rows.values() if s.started is None)
        completed = passed + failed
        self.progress.update(
            self.summary_row,
            total=self.total_trials,
            completed=completed,
            passed=passed,
            failed=failed,
            running=running,
            queued=queued,
        )
        if completed >= self.total_trials and self.total_trials > 0:
            self.progress.stop_task(self.summary_row)

    def add(
        self, task_id: str, trial_name: str, *, attempt: int = 1, attempts: int = 1
    ) -> None:
        with self._lock:
            self._ensure_summary_row()
            if trial_name in self.rows:
                return
            label = task_id if attempts == 1 else f"{task_id} [{attempt}/{attempts}]"
            row = self.progress.add_task(
                label, total=None, start=False, phase="Queued", detail=""
            )
            state = TrialProgress(task_id, trial_name, row)
            self.rows[trial_name] = state
            self.progress.update(row, state=state)
            self.total_trials += 1
            self._update_summary_counts()

    def start(self) -> None:
        with self._lock:
            if self.summary_row is not None:
                now = self.progress.get_time()
                self.progress.update(self.summary_row, started=now)
                self.progress.start_task(self.summary_row)
        self.progress.console.print(
            "Task progress (top row: overall summary; sub rows: current stage and activity)",
            style="dim",
        )
        self.progress.start()

    def _find(
        self, task_id: str | None, trial_name: str | None
    ) -> TrialProgress | None:
        if trial_name in self.rows:
            return self.rows[trial_name]
        if trial_name:
            parents = [
                state
                for state in self.rows.values()
                if state.task_id == task_id
                and trial_name.startswith(state.trial_name + ".agent-")
            ]
            if len(parents) == 1:
                return parents[0]
            return None
        matches = [
            state
            for state in self.rows.values()
            if state.task_id == task_id and state.outcome is None
        ]
        # Multi-agent subtrials share the parent task's row. Never route an
        # ambiguous callback onto an unrelated attempt.
        return matches[0] if len(matches) == 1 else None

    def _phase(self, state: TrialProgress, phase: str, detail: str = "") -> None:
        now = self.progress.get_time()
        if state.started is None:
            state.started = now
        if state.phase != phase:
            state.phase = phase
            state.grid_started = None
            state.completed, state.total = 0, None
            self.progress.reset(state.row, total=None, completed=0, start=True)
        self.progress.update(state.row, phase=phase, detail=_redact_text(detail))

    def stage(
        self,
        task_id: str | None,
        trial_name: str | None,
        stage: str,
        status: str,
        *,
        execution: dict | None = None,
    ) -> None:
        with self._lock:
            state = self._find(task_id, trial_name)
            if state is None or state.outcome:
                return
            phase = self.STAGES.get(stage)
            if status == "failed":
                self.progress.update(
                    state.row,
                    phase=f"{state.phase} failed",
                    detail="Saving diagnostics",
                )
            elif stage == "terminal.lifecycle" and status == "stopped":
                self._phase(state, "Finalizing", "Collecting logs")
            elif phase and status in {"started", "ready", "completed"}:
                self._phase(
                    state,
                    phase,
                    "Complete"
                    if status == "completed"
                    else "Ready"
                    if status == "ready"
                    else "Starting",
                )
            if stage == "agent.execute" and status == "started":
                state.agent_budget = (execution or {}).get("timeout_seconds")
            self._update_summary_counts()

    def detail(self, task_id: str | None, trial_name: str | None, detail: str) -> None:
        with self._lock:
            state = self._find(task_id, trial_name)
            if state is None or state.outcome:
                return
            runs = re.search(r"solver runs (\d+)/(\d+)", detail)
            if runs:
                completed, total = map(int, runs.groups())
                if (
                    state.phase != "Solver runs"
                    or total != state.total
                    or completed < state.completed
                ):
                    self._phase(state, "Solver runs")
                    state.grid_started = self.progress.get_time()
                    state.total = total
                    self.progress.reset(state.row, total=total, completed=0, start=True)
                state.completed = completed
                self.progress.update(
                    state.row,
                    total=total,
                    completed=completed,
                    phase=state.phase,
                    detail=_redact_text(detail.removeprefix("Judge progress: ")),
                )
            elif detail.startswith("Judge plan:"):
                self._phase(state, "Evaluation", detail.removeprefix("Judge plan: "))
            elif detail.startswith("Judge validation build"):
                self._phase(state, "Validation build", detail.partition(": ")[2])
            elif detail.startswith("Judge performance build"):
                self._phase(state, "Performance build", detail.partition(": ")[2])
            elif detail.startswith("Judge solver grid complete"):
                self._phase(state, "Evaluation", "Computing reports")
            elif detail.startswith("Agent:"):
                self._phase(state, "Coding agent", detail.removeprefix("Agent: "))
            else:
                self.progress.update(state.row, detail=_redact_text(detail))
            self._update_summary_counts()

    def finish(self, result) -> None:
        with self._lock:
            state = self._find(result.task_id, result.trial_name)
            if state is None:
                return
            failure = getattr(result.failure_mode, "value", result.failure_mode)
            evaluation = result.evaluation
            failed = failure not in {None, "none", "unset"} or (
                evaluation is not None and not evaluation.completed
            )
            state.ended = self.progress.get_time()
            state.outcome = "Failed" if failed else "Done"
            state.phase = state.outcome
            self.progress.update(
                state.row,
                total=1,
                completed=0 if failed else 1,
                phase=state.outcome,
                detail=str(failure) if failed else "",
            )
            self.progress.stop_task(state.row)
            self._update_summary_counts()

    def stop(self) -> None:
        with self._lock:
            for state in self.rows.values():
                if state.outcome is None:
                    state.ended = self.progress.get_time()
                    state.outcome = "Stopped"
                    self.progress.update(
                        state.row, phase="Stopped", detail="Run interrupted"
                    )
                    self.progress.stop_task(state.row)
            self._update_summary_counts()
        self.progress.stop()


def _tool_activity(name: str, parameters: Any = None) -> str:
    name = name.lower()
    if name in {"call_mcp_tool", "mcp_tool_call"} and isinstance(parameters, dict):
        tool = (
            parameters.get("ToolName")
            or parameters.get("tool_name")
            or parameters.get("tool")
        )
        arguments = parameters.get("Arguments") or parameters.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = {}
        if tool:
            return _tool_activity(str(tool), arguments)
    if any(
        word in name for word in ("patch", "write", "replace", "file_change", "edit")
    ):
        return "Editing"
    if any(
        word in name
        for word in (
            "read",
            "view",
            "list",
            "search",
            "grep",
            "find",
            "git_status",
            "git_diff",
        )
    ):
        return "Reading"
    if "plan" in name:
        return "Planning"
    if isinstance(parameters, dict):
        command = str(parameters.get("command") or parameters.get("CommandLine") or "")
        if re.search(
            r"\b(pytest|ctest|unittest|tox)\b|\b(cargo|npm|make)\s+test\b|\bpitbench\s+(validate|bench)\b",
            command,
        ):
            return "Testing"
        if re.search(r"\b(cmake|ninja|meson|make)\b|\b(cargo|npm)\s+build\b", command):
            return "Building"
        if re.search(
            r"\b(cat|head|tail|sed|rg|grep|ls|pwd)\b|\bgit\s+(status|diff|show|log)\b",
            command,
        ):
            return "Reading"
    return "Running tool"


class AgentActivity:
    """Expose event types and tool categories, never model reasoning or raw arguments."""

    def __init__(self):
        self.rounds = 0
        self.tools = 0
        self._seen: set[tuple] = set()
        self._last_detail: str | None = None

    def feed(self, line: str) -> str | None:
        try:
            event = json.loads(line)
        except ValueError:
            return None
        if not isinstance(event, dict):
            return None
        activity = None
        kind = event.get("event") or event.get("type")
        if kind in {"init", "thread.started", "turn.started"}:
            activity = "Waiting for model"
        elif kind == "step_update":
            step = event.get("step_update") or {}
            state, step_type = step.get("state"), step.get("step_type")
            tool = step.get("tool_info") or {}
            identity = (step.get("conversation_id"), step.get("step_index"), step_type)
            if tool or step_type == "tool":
                activity = _tool_activity(
                    str(step.get("tool_name") or tool.get("name") or "tool"),
                    tool.get("parameters"),
                )
                if state in {"DONE", "ERROR"} and (
                    step.get("step_index") is None or identity not in self._seen
                ):
                    self.tools += 1
                    self._seen.add(identity)
                if state == "ERROR":
                    activity += " (tool error)"
            elif step_type in {
                "agent_response",
                "think",
                "reasoning",
                "planner_response",
            }:
                activity = (
                    "Model response" if step_type == "agent_response" else "Planning"
                )
                if state == "DONE" and identity not in self._seen:
                    self.rounds += 1
                    self._seen.add(identity)
        elif kind in {"item.started", "item.updated", "item.completed"}:
            item = event.get("item") or {}
            item_type = item.get("type", "")
            if item_type in {
                "mcp_tool_call",
                "command_execution",
                "file_change",
                "web_search",
                "tool_call",
            }:
                activity = _tool_activity(
                    str(item.get("tool") or item_type), item.get("arguments") or item
                )
                identity = ("item", item.get("id"))
                if kind == "item.completed" and (
                    item.get("id") is None or identity not in self._seen
                ):
                    self.tools += 1
                    self._seen.add(identity)
            elif item_type in {"reasoning", "agent_message", "plan"}:
                activity = (
                    "Model response" if item_type == "agent_message" else "Planning"
                )
        elif kind == "turn.completed":
            self.rounds += 1
            activity = "Response complete"
        elif kind == "result":
            activity = (
                "Response complete"
                if (event.get("result") or {}).get("status") == "SUCCESS"
                else "Response failed"
            )
        if activity is None:
            return None
        detail = f"Agent: {activity} · rounds {self.rounds} · tools {self.tools}"
        if detail == self._last_detail:
            return None
        self._last_detail = detail
        return detail
