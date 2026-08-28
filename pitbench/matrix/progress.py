from __future__ import annotations

import re
import threading
from types import TracebackType

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)


class MatrixProgressDisplay:
    """Thread-safe Rich display for concurrent agent and judge matrix jobs."""

    _JUDGE_PROGRESS = re.compile(
        r"Judge progress: instances (?P<instances>\d+/\d+), "
        r"seed groups (?P<seeds>\d+/\d+), "
        r"solver runs (?P<completed>\d+)/(?P<total>\d+), "
        r"valid (?P<valid>\d+)"
    )
    _AGENT_PROGRESS = re.compile(
        r"Agent: (?:MCP|tool) calls (?P<calls>\d+) · messages (?P<messages>\d+)"
    )

    def __init__(self, *, console: Console | None = None) -> None:
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("{task.fields[counts]}"),
            TimeElapsedColumn(),
            console=console,
        )
        self._tasks: dict[str, int] = {}
        self._lock = threading.Lock()

    def __enter__(self) -> MatrixProgressDisplay:
        self._progress.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._progress.stop()

    def _task(self, context: str, *, description: str) -> int:
        task = self._tasks.get(context)
        if task is None:
            task = self._progress.add_task(description, total=None, counts="starting")
            self._tasks[context] = task
        return task

    def _finish(self, context: str, message: str) -> None:
        task = self._tasks.pop(context, None)
        if task is not None:
            self._progress.remove_task(task)
        self._progress.console.print(message)

    def __call__(self, message: str) -> None:
        with self._lock:
            if message.startswith("Candidate ready "):
                context = message.removeprefix("Candidate ready ")
                task = self._task(context, description=f"{context} · Agent")
                self._progress.update(
                    task, description=f"{context} · Agent complete", counts="ready"
                )
                return
            if message.startswith("Evaluation complete "):
                context = message.removeprefix("Evaluation complete ")
                self._finish(context, message)
                return
            if message.startswith("BASE ready "):
                _, _, task_id, _, repeat = message.split()
                self._finish(f"{task_id}/base/repeat-{repeat}", message)
                return
            if " — " not in message:
                self._progress.console.print(message)
                return

            context, detail = message.split(" — ", 1)
            task = self._task(context, description=context)
            judge = self._JUDGE_PROGRESS.fullmatch(detail)
            if judge is not None:
                completed = int(judge["completed"])
                total = int(judge["total"])
                counts = (
                    f"runs {completed}/{total} · instances {judge['instances']} · "
                    f"seeds {judge['seeds']} · valid {judge['valid']}"
                )
                self._progress.update(
                    task,
                    description=f"{context} · Judge",
                    completed=completed,
                    total=total,
                    counts=counts,
                )
                return

            agent = self._AGENT_PROGRESS.fullmatch(detail)
            if agent is not None:
                self._progress.update(
                    task,
                    description=f"{context} · Agent",
                    total=None,
                    counts=(f"tools {agent['calls']} · messages {agent['messages']}"),
                )
                return

            self._progress.update(
                task, description=f"{context} · {detail}", total=None, counts=""
            )
