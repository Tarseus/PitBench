from unittest.mock import patch

import pytest

from pitbench.evaluator.judge import _limit_solver_cpus
from pitbench.repositories.base import CommandSpec


def test_solver_cpu_affinity_does_not_limit_build_container() -> None:
    command = CommandSpec(argv=["solver", "--threads", "2"])

    with patch("pitbench.evaluator.judge.os.sched_getaffinity", return_value={3, 7, 8}):
        limited = _limit_solver_cpus(command, threads=2)

    assert limited.argv == [
        "taskset",
        "--cpu-list",
        "3,7",
        "solver",
        "--threads",
        "2",
    ]
    assert command.argv == ["solver", "--threads", "2"]


def test_solver_cpu_affinity_rejects_unavailable_threads() -> None:
    command = CommandSpec(argv=["solver"])

    with (
        patch("pitbench.evaluator.judge.os.sched_getaffinity", return_value={3}),
        pytest.raises(ValueError, match="requests 2 threads"),
    ):
        _limit_solver_cpus(command, threads=2)
