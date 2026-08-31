from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pitbench.harness.agents.codex_workspace import CodexWorkspaceRuntime


def _runtime(container: MagicMock) -> CodexWorkspaceRuntime:
    return CodexWorkspaceRuntime(
        container=container,
        codex_binary=Path("/opt/codex/bin/codex"),
        profile=None,
    )


def test_workspace_runtime_removes_staged_bundle_on_exit() -> None:
    container = MagicMock()
    container.exec_run.return_value = SimpleNamespace(exit_code=0, output=(b"", b""))
    runtime = _runtime(container)

    runtime.__exit__()

    container.exec_run.assert_called_once_with(
        ["rm", "-rf", "--", runtime.root],
        demux=True,
    )


def test_workspace_runtime_cleans_partial_staging_failure() -> None:
    container = MagicMock()
    container.exec_run.return_value = SimpleNamespace(exit_code=0, output=(b"", b""))
    runtime = _runtime(container)
    runtime._stage = MagicMock(side_effect=RuntimeError("copy failed"))

    with pytest.raises(RuntimeError, match="copy failed"):
        runtime.__enter__()

    container.exec_run.assert_called_once_with(
        ["rm", "-rf", "--", runtime.root],
        demux=True,
    )
