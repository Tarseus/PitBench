from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from pitbench.evaluator.judge import LocalProcessJudge
from pitbench.evaluator.reliability import prepare_boundary_cases
from pitbench.evaluator.validity import evaluator_validity
from pitbench.repositories.base import CommandSpec, RepositoryPlugin
from pitbench.schema.observation import CodeState, RunStatus
from pitbench.schema.task import PitBenchTask

ROOT = Path(__file__).resolve().parents[3]


class ScriptRepository(RepositoryPlugin):
    name = "controlled_process"

    def __init__(self, script, timeout=5):
        self.script, self.timeout = script, timeout

    def build_commands(self, kind):
        return []

    def run_command(self, run):
        return CommandSpec(
            argv=[
                sys.executable,
                "-c",
                "from pathlib import Path\nimport json,sys\np=Path(sys.argv[1])\n"
                + self.script,
                str(run.output_path),
            ],
            timeout_sec=self.timeout,
        )


@pytest.fixture
def judge_case(tmp_path):
    task = PitBenchTask.from_yaml(ROOT / "configs/tasks/pyvrp_v0_14_0.yaml")
    task.evaluation.seed_robustness = None
    task.evaluation.solver_seeds = [0]
    task.evaluation.representation_robustness = None
    cases = prepare_boundary_cases(task, tmp_path / "out")
    judge = LocalProcessJudge(
        task,
        tmp_path,
        ROOT,
        tmp_path / "absent",
        None,
        tmp_path / "out",
        evaluation_seeds=(0,),
        family=cases[0].verifier,
        code_states=(CodeState.BASE,),
        run_validation_builds=False,
        progress_callback=lambda message: None,
    )
    return judge, cases[0]


GOOD = "p.write_text(json.dumps({'valid': True, 'has_solution': True, 'objective': 10, 'solver_status': 'Time limit reached'}))\np.with_suffix('.solution.json').write_text(json.dumps({'routes': [[1]]}))\nprint('native stdout')\nprint('native stderr', file=sys.stderr)\n"


@pytest.mark.parametrize(
    "script,expected",
    [
        (GOOD, RunStatus.COMPLETED),
        ("sys.exit(7)", RunStatus.CRASHED),
        ("pass", RunStatus.OUTPUT_ERROR),
        ("p.write_text('{bad json')", RunStatus.OUTPUT_ERROR),
        ("p.write_text(json.dumps({'valid': True}))", RunStatus.OUTPUT_ERROR),
        (
            "p.write_text(json.dumps({'valid': False, 'has_solution': False, 'solver_status': 'Time limit reached'}))",
            RunStatus.NO_SOLUTION,
        ),
        (GOOD.replace("[[1]]", "[[999]]"), RunStatus.INVALID),
        (
            "p.write_text(json.dumps({'valid': False, 'failure_reason': 'out_of_memory', 'error': 'MemoryError'}))\nsys.exit(1)",
            RunStatus.OUT_OF_MEMORY,
        ),
        (
            "p.write_text(json.dumps({'valid': False, 'failure_reason': 'timed_out', 'error': 'native process exceeded deadline'}))\nsys.exit(1)",
            RunStatus.TIMED_OUT,
        ),
        (
            "p.write_text(json.dumps({'valid': False, 'failure_reason': 'solver_error', 'solver_status': 'Solve error'}))",
            RunStatus.SOLVER_ERROR,
        ),
        ("import os,signal\nos.kill(os.getpid(),signal.SIGKILL)", RunStatus.CRASHED),
    ],
)
def test_real_subprocess_failures_remain_distinct(
    judge_case, tmp_path, script, expected
):
    judge, case = judge_case
    judge.repository = ScriptRepository(script)
    observation = judge._run_case(tmp_path, case, CodeState.AGENT, 0, 5)
    assert observation.status == expected
    assert observation.test_suite == "operational_reliability"
    assert Path(observation.stdout_path).is_file()
    assert Path(observation.stderr_path).is_file()
    validity = evaluator_validity(
        patch_exists=True, fixture_mode=False, observations=[observation]
    )
    assert validity.accepted is (expected != RunStatus.INVALID)
    if expected == RunStatus.COMPLETED:
        assert observation.objective == 10
        assert observation.solver_status == "Time limit reached"
        assert Path(observation.stdout_path).read_text() == "native stdout\n"


def test_an_old_result_cannot_make_a_later_missing_output_pass(judge_case, tmp_path):
    judge, case = judge_case
    judge.repository = ScriptRepository(GOOD)
    assert judge._run_case(tmp_path, case, CodeState.BASE, 0, 5).valid
    judge.repository = ScriptRepository("pass")
    assert (
        judge._run_case(tmp_path, case, CodeState.BASE, 0, 5).status
        == RunStatus.OUTPUT_ERROR
    )


def test_watchdog_terminates_the_solver_process_group(judge_case, tmp_path):
    judge, case = judge_case
    child_pid = tmp_path / "child.pid"
    script = (
        "import subprocess,time\n"
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])\n"
        f"Path({str(child_pid)!r}).write_text(str(child.pid))\n"
        "print('started',flush=True)\ntime.sleep(30)\n"
    )
    judge.repository = ScriptRepository(script, timeout=0.5)
    started = time.monotonic()
    result = judge._run_case(tmp_path, case, CodeState.BASE, 0, 5)
    assert result.status == RunStatus.TIMED_OUT
    assert time.monotonic() - started < 5
    assert "started" in Path(result.stdout_path).read_text()
    pid = int(child_pid.read_text())
    status = Path(f"/proc/{pid}/stat")
    # A killed child may briefly await reaping by the container init process.
    assert not status.exists() or status.read_text().split()[2] == "Z"


def test_failed_case_does_not_abort_remaining_cases(judge_case, tmp_path, monkeypatch):
    judge, first = judge_case
    cases = [
        replace(first, instance_id=name, solver_seeds=(0,), budgets_sec=(5,))
        for name in ("bad", "good")
    ]
    judge.repository = ScriptRepository(
        "\nif p.parent.name == 'bad': sys.exit(7)\n" + GOOD
    )
    monkeypatch.setattr(judge, "_workspace", lambda *args: tmp_path)
    saved = []
    results = judge.run(cases, save_observation=saved.append)
    assert len(saved) == len(results) == 2
    assert {item.instance_id: item.status for item in results} == {
        "bad": RunStatus.CRASHED,
        "good": RunStatus.COMPLETED,
    }


def test_driver_preserves_inner_timeout_reason(tmp_path, monkeypatch):
    from pitbench.solver_drivers.run import HighsDriver

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("native solver", 65)

    monkeypatch.setattr(subprocess, "run", timeout)
    output = tmp_path / "result.json"
    with pytest.raises(subprocess.TimeoutExpired):
        HighsDriver.main(
            [
                "--solver",
                "highs",
                "--instance",
                str(tmp_path / "input.lp"),
                "--output",
                str(output),
                "--trajectory",
                str(tmp_path / "trajectory.jsonl"),
                "--seed",
                "0",
                "--budget",
                "5",
                "--threads",
                "1",
            ]
        )
    assert json.loads(output.read_text())["failure_reason"] == "timed_out"


def test_native_status_and_objective_can_be_read_from_solution_file(
    tmp_path, monkeypatch, capsys
):
    from pitbench.solver_drivers.run import HighsDriver

    raw = "Model status\nOptimal\n\n# Primal solution values\nFeasible\nObjective 5\n# Columns 2\nx0 1\nx1 2\n# Rows 1\nc0 3\n"

    def solve(argv, **kwargs):
        output = next(
            value.split("=", 1)[1]
            for value in argv
            if value.startswith("--solution_file=")
        )
        Path(output).write_text(raw)
        return subprocess.CompletedProcess(
            argv, 0, "Solving report\n  Status Optimal\n", "native diagnostic\n"
        )

    monkeypatch.setattr(subprocess, "run", solve)
    output = tmp_path / "result.json"
    HighsDriver.main(
        [
            "--solver",
            "highs",
            "--instance",
            str(tmp_path / "input.lp"),
            "--output",
            str(output),
            "--trajectory",
            str(tmp_path / "trajectory.jsonl"),
            "--seed",
            "0",
            "--budget",
            "5",
            "--threads",
            "1",
        ]
    )
    result = json.loads(output.read_text())
    assert result["solver_status"] == "Optimal" and result["objective"] == 5
    assert result["has_solution"]
    assert "native diagnostic" in capsys.readouterr().err
