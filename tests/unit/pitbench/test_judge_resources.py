from pathlib import Path
from threading import Barrier, Lock
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pitbench.evaluator.judge import (
    InstanceCase,
    JudgePlan,
    LocalProcessJudge,
    _limit_solver_cpus,
)
from pitbench.repositories.base import CommandSpec
from pitbench.schema.observation import CodeState, RunObservation, RunStatus


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


def test_solver_cpu_affinity_uses_assigned_worker_slot() -> None:
    command = CommandSpec(argv=["solver"])

    limited = _limit_solver_cpus(command, threads=1, cpu_ids=(19,))

    assert limited.argv[:3] == ["taskset", "--cpu-list", "19"]


def test_local_judge_runs_grid_on_distinct_cpu_slots(tmp_path: Path) -> None:
    instance_set = SimpleNamespace(
        name="judge_id",
        kind=SimpleNamespace(value="judge_id"),
        randomness=SimpleNamespace(
            instance_seed=1, coordinate_seed=None, demand_seed=None
        ),
    )
    case = InstanceCase(
        instance_set=instance_set,
        instance_id="one",
        path=tmp_path / "one.json",
        anchor=1.0,
        solver_seeds=(0, 1),
        budgets_sec=(1.0,),
    )
    case.path.write_text("{}")
    task = SimpleNamespace(
        task_id="task",
        evaluation=SimpleNamespace(threads=1, solver_seeds=[0, 1], budgets_sec=[1.0]),
    )
    judge = LocalProcessJudge.__new__(LocalProcessJudge)
    judge.task = task
    judge.public_root = tmp_path
    judge.resolver = object()
    judge.output_dir = tmp_path
    judge.code_states = (CodeState.BASE,)
    judge.evaluation_seeds = (0, 1)
    judge.parallel_runs = 2
    judge.run_validation_builds = True
    progress_messages = []
    judge.progress_callback = progress_messages.append
    judge._workspace = lambda state, root, kind: tmp_path
    barrier = Barrier(2)
    lock = Lock()
    assigned = []
    saved_observations = []

    def run_case(workspace, current_case, state, seed, budget, *, cpu_ids):
        with lock:
            assigned.append(cpu_ids)
        barrier.wait(timeout=2)
        return RunObservation(
            task_id="task",
            code_state=state,
            instance_set="judge_id",
            instance_set_kind="judge_id",
            instance_id="one",
            instance_seed=1,
            solver_seed=seed,
            budget_sec=budget,
            status=RunStatus.COMPLETED,
            valid=True,
        )

    judge._run_case = run_case
    plan = JudgePlan(task, [case])
    with (
        patch.object(JudgePlan, "from_instance_set_configs", return_value=plan),
        patch("pitbench.evaluator.judge.os.sched_getaffinity", return_value={3, 7}),
    ):
        observations = judge.run(save_observation=saved_observations.append)

    assert assigned == [(3,), (7,)]
    assert [item.solver_seed for item in observations] == [0, 1]
    assert {item.solver_seed for item in saved_observations} == {0, 1}
    assert progress_messages[-2] == (
        "Judge progress: instances 1/1, seed groups 2/2, solver runs 2/2, valid 2"
    )


def test_local_judge_skips_completed_runs(tmp_path: Path) -> None:
    instance_set = SimpleNamespace(
        name="agent_dev",
        kind=SimpleNamespace(value="agent_dev"),
        randomness=None,
    )
    case = InstanceCase(
        instance_set=instance_set,
        instance_id="one",
        path=tmp_path / "one.json",
        anchor=1.0,
        solver_seeds=(10, 20),
        budgets_sec=(1.0,),
    )
    case.path.write_text("{}")
    task = SimpleNamespace(
        task_id="task",
        evaluation=SimpleNamespace(threads=1, solver_seeds=[10, 20], budgets_sec=[1.0]),
    )
    judge = LocalProcessJudge.__new__(LocalProcessJudge)
    judge.task = task
    judge.public_root = tmp_path
    judge.resolver = object()
    judge.output_dir = tmp_path
    judge.code_states = (CodeState.BASE,)
    judge.evaluation_seeds = (10, 20)
    judge.parallel_runs = 1
    judge.run_validation_builds = False
    judge.progress_callback = None
    judge._workspace = lambda state, root, kind: tmp_path
    executed_seeds = []

    def run_case(workspace, current_case, state, seed, budget, *, cpu_ids):
        executed_seeds.append(seed)
        return RunObservation(
            task_id="task",
            code_state=state,
            instance_set="agent_dev",
            instance_set_kind="agent_dev",
            instance_id="one",
            solver_seed=seed,
            budget_sec=budget,
            status=RunStatus.COMPLETED,
            valid=True,
        )

    judge._run_case = run_case
    completed_runs = {("agent_dev", "one", CodeState.BASE, 10, 1.0)}

    with patch("pitbench.evaluator.judge.os.sched_getaffinity", return_value={3}):
        observations = judge.run([case], completed_runs=completed_runs)

    assert executed_seeds == [20]
    assert [item.solver_seed for item in observations] == [20]
