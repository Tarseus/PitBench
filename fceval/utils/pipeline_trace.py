"""Structured, append-only tracing for the evaluation data pipeline."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

_REDACTED = "[REDACTED]"
_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "access_token",
    "auth_token",
    "authorization",
    "cookie",
    "credentials",
    "password",
    "refresh_token",
    "secret",
}
_SENSITIVE_SUFFIXES = (
    "_api_key",
    "_access_token",
    "_auth_token",
    "_password",
    "_secret",
)
_ASSIGNMENT_SECRET_RE = re.compile(
    r"(?i)(api[_-]?key|access[_-]?token|auth[_-]?token|authorization|"
    r"password|secret)(\s*[:=]\s*)([^\s,;]+)"
)
_BEARER_SECRET_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")


def _is_sensitive_key(key: str) -> bool:
    normalized = key.strip().lower()
    return normalized in _SENSITIVE_KEYS or normalized.endswith(_SENSITIVE_SUFFIXES)


def _redact_text(value: str) -> str:
    value = _ASSIGNMENT_SECRET_RE.sub(rf"\1\2{_REDACTED}", value)
    return _BEARER_SECRET_RE.sub(f"Bearer {_REDACTED}", value)


def _json_safe(value: Any, *, key: str | None = None) -> Any:
    """Convert runtime values to JSON data while redacting common credentials."""
    if key is not None and _is_sensitive_key(key):
        return _REDACTED

    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, BaseModel):
        return _json_safe(value.model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(item_key): _json_safe(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, BaseException):
        return {
            "type": type(value).__name__,
            "message": _redact_text(str(value)),
        }

    return _redact_text(repr(value))


class PipelineTrace:
    """Write ordered JSONL events for one harness run.

    Writes are protected by a process-local lock because trials can complete on
    multiple worker threads. Each line is independently valid JSON so a failed or
    interrupted run still leaves all events flushed up to that point.
    """

    def __init__(self, path: Path, run_id: str, *, append: bool = False) -> None:
        self.path = path
        self.run_id = run_id
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if append and self.path.exists():
            self._sequence = self._last_sequence()
        else:
            self.path.write_text("")
            self._sequence = 0

    def _last_sequence(self) -> int:
        last_sequence = 0
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            sequence = event.get("sequence")
            if isinstance(sequence, int):
                last_sequence = max(last_sequence, sequence)
        return last_sequence

    def record(
        self,
        *,
        stage: str,
        status: str,
        inputs: Any = None,
        outputs: Any = None,
        execution: Any = None,
        task_id: str | None = None,
        trial_name: str | None = None,
        error: BaseException | str | None = None,
    ) -> None:
        """Append one fully flushed pipeline event."""
        with self._lock:
            self._sequence += 1
            event = {
                "sequence": self._sequence,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_id": self.run_id,
                "task_id": task_id,
                "trial_name": trial_name,
                "stage": stage,
                "status": status,
                "inputs": _json_safe(inputs),
                "outputs": _json_safe(outputs),
                "execution": _json_safe(execution),
                "error": _json_safe(error),
            }
            with self.path.open("a") as trace_file:
                trace_file.write(json.dumps(event, ensure_ascii=False) + "\n")
                trace_file.flush()
