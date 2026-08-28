from concurrent.futures import Future
from pathlib import Path
from threading import Event
from unittest.mock import patch

from rich.console import Console
from typer.testing import CliRunner

from pitbench.cli.evaluate_config import (
    EvaluationConfig,
    EvaluationPaths,
    PipelineResources,
)
from pitbench.cli.main import app
from pitbench.matrix.models import MatrixAgentSpec, MatrixSpec, PipelineStatus
from pitbench.matrix.progress import MatrixProgressDisplay
from pitbench.matrix.runner import MatrixRunner

ROOT = Path(__file__).resolve().parents[3]


def _config(tmp_path: Path) -> EvaluationConfig:
    return EvaluationConfig(
        paths=EvaluationPaths(
            output_path=tmp_path / "runs",
            workspace_path=tmp_path / "tasks",
            private_root=tmp_path / "private",
        ),
        pipeline=PipelineResources(
            agent_workers=1,
            agent_cpus_per_worker=1,
            agent_cpu_pool="0",
            judge_workers=1,
            judge_parallel_runs=1,
            judge_cpu_pool="1",
            infrastructure_retries=0,
        ),
    )


def _spec(agent_count: int = 2) -> MatrixSpec:
    return MatrixSpec(
        experiment_id="test",
        tasks=["pyvrp_v0_14_0"],
        repeats=1,
        agents=[
            MatrixAgentSpec(id=f"agent-{index}", agent="codex", model="model")
            for index in range(agent_count)
        ],
    )


def _runner(tmp_path: Path, spec: MatrixSpec | None = None) -> MatrixRunner:
    with (
        patch("pitbench.matrix.runner._physical_cpus", return_value=(0, 1, 2)),
        patch("pitbench.matrix.runner.os.sched_getaffinity", return_value={0, 1, 2}),
    ):
        return MatrixRunner(
            repository_root=ROOT,
            spec=spec or _spec(),
            config=_config(tmp_path),
            run_id="run",
            progress=lambda _: None,
        )


def test_formal_matrix_expands_to_72_unique_jobs(tmp_path: Path) -> None:
    spec = MatrixSpec.from_yaml(ROOT / "experiments/pyvrp-4x6x3.yaml")
    config = _config(tmp_path)
    config.pipeline = PipelineResources(
        agent_workers=3,
        agent_cpus_per_worker=1,
        agent_cpu_pool="0-2",
        judge_workers=4,
        judge_parallel_runs=1,
        judge_cpu_pool="3-6",
    )
    with (
        patch("pitbench.matrix.runner._physical_cpus", return_value=tuple(range(8))),
        patch(
            "pitbench.matrix.runner.os.sched_getaffinity", return_value=set(range(8))
        ),
    ):
        runner = MatrixRunner(
            repository_root=ROOT,
            spec=spec,
            config=config,
            run_id="run",
            progress=lambda _: None,
        )

    jobs = runner._jobs()

    assert len(jobs) == 72
    assert len({job.id for job in jobs}) == 72


def test_matrix_progress_switches_from_agent_pulse_to_judge_bar() -> None:
    display = MatrixProgressDisplay(
        console=Console(record=True, force_terminal=False, width=160)
    )

    with display:
        display("job — Agent: MCP calls 18 · messages 12")
        task_id = display._tasks["job"]
        agent_task = display._progress.tasks[task_id]
        assert agent_task.total is None
        assert agent_task.fields["counts"] == "tools 18 · messages 12"

        display(
            "job — Judge progress: instances 17/48, seed groups 83/240, "
            "solver runs 249/720, valid 247"
        )
        judge_task = display._progress.tasks[task_id]
        assert judge_task.completed == 249
        assert judge_task.total == 720
        assert judge_task.fields["counts"].startswith("runs 249/720")


def test_matrix_runs_generation_while_base_is_running(tmp_path: Path) -> None:
    runner = _runner(tmp_path, _spec(agent_count=1))
    base_started = Event()
    generation_started = Event()
    base_path = tmp_path / "base.parquet"

    def run_base(task_id: str, repeat: int, slot: int) -> Path:
        base_started.set()
        assert generation_started.wait(timeout=2)
        base_path.write_text("base")
        return base_path

    def generate(job, slot):
        assert base_started.wait(timeout=2)
        generation_started.set()
        state = runner._load_state(job)
        patch_path = runner._trial_dir(job) / "candidate.patch"
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        patch_path.write_text("")
        state.candidate_patch = str(patch_path)
        state.status = PipelineStatus.CANDIDATE_READY
        runner._save_state(job, state)
        return state

    def evaluate(job, base_future: Future[Path], slot: int):
        assert base_future.result() == base_path
        state = runner._load_state(job)
        state.status = PipelineStatus.COMPLETED
        runner._save_state(job, state)
        return state

    with (
        patch.object(runner, "_prepare"),
        patch.object(runner, "_run_base", side_effect=run_base),
        patch.object(runner, "_generate_candidate", side_effect=generate),
        patch.object(runner, "_evaluate_candidate", side_effect=evaluate),
    ):
        runner.run()

    assert runner._load_state(runner._jobs()[0]).status == PipelineStatus.COMPLETED
    assert (runner.run_root / "matrix-trials.parquet").is_file()
    assert (runner.run_root / "matrix-summary.json").is_file()
    assert (runner.run_root / "matrix-report.md").is_file()


def test_completed_trial_is_not_generated_again(tmp_path: Path) -> None:
    runner = _runner(tmp_path, _spec(agent_count=1))
    job = runner._jobs()[0]
    state = runner._load_state(job)
    state.status = PipelineStatus.COMPLETED
    runner._save_state(job, state)
    base = Future()
    base.set_result(tmp_path / "base.parquet")

    with (
        patch.object(runner, "_prepare"),
        patch.object(runner, "_run_base", return_value=tmp_path / "base.parquet"),
        patch.object(runner, "_generate_candidate") as generate,
        patch.object(runner, "_write_summary"),
    ):
        runner.run()

    generate.assert_not_called()


def test_matrix_cli_loads_spec_and_runs_pipeline(tmp_path: Path) -> None:
    spec_path = tmp_path / "matrix.yaml"
    spec_path.write_text(
        """schema_version: "1.0"
experiment_id: test
tasks: [pyvrp_v0_14_0]
repeats: 1
agents:
  - id: codex-test
    agent: codex
    model: test
"""
    )
    config_path = tmp_path / "evaluate.yaml"
    config_path.write_text("paths:\n  output_path: runs\n")

    with patch("pitbench.cli.main.MatrixRunner") as runner_class:
        result = CliRunner().invoke(
            app,
            [
                "matrix",
                str(spec_path),
                "--config",
                str(config_path),
                "--run-id",
                "matrix-test",
                "--root",
                str(ROOT),
            ],
        )

    assert result.exit_code == 0, result.output
    assert runner_class.call_args.kwargs["run_id"] == "matrix-test"
    runner_class.return_value.run.assert_called_once_with()
