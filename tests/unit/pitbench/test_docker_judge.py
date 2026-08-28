import io
from pathlib import Path
from unittest.mock import patch

import pytest

from pitbench.evaluator.docker_judge import DockerJudge
from pitbench.schema.observation import CodeState, RunObservation, RunStatus


def _judge(tmp_path: Path, image: str) -> DockerJudge:
    manifest = tmp_path / "task.yaml"
    repository = tmp_path / "repository"
    private = tmp_path / "private"
    candidate = tmp_path / "candidate.patch"
    output = tmp_path / "output"
    manifest.write_text("task_id: test\n")
    repository.mkdir()
    private.mkdir()
    candidate.write_text("")
    output.mkdir()
    return DockerJudge(
        image=image,
        manifest_path=manifest,
        base_repository=repository,
        private_root=private,
        candidate_patch=candidate,
        output_dir=output,
    )


def test_judge_image_must_be_digest_pinned(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="pinned"):
        _judge(tmp_path, "pitbench/pyvrp:latest")


def test_judge_accepts_immutable_local_image_id(tmp_path: Path) -> None:
    judge = _judge(tmp_path, "sha256:" + "b" * 64)

    assert judge.image == "sha256:" + "b" * 64


def test_docker_judge_disables_network_and_reads_observations(tmp_path: Path) -> None:
    judge = _judge(tmp_path, "pitbench/pyvrp@sha256:" + "a" * 64)
    progress_messages = []
    judge.progress_callback = progress_messages.append
    observation = RunObservation(
        task_id="test",
        code_state=CodeState.AGENT,
        population="judge_id",
        instance_id="one",
        instance_seed=1,
        solver_seed=0,
        budget_sec=1,
        status=RunStatus.COMPLETED,
        valid=True,
    )

    def fake_popen(command, **kwargs):
        assert command[command.index("--network") + 1] == "none"
        assert "--read-only" in command
        assert command[command.index("--cpus") + 1] == "8.0"
        (judge.output_dir / "judge-observations.jsonl").write_text(
            observation.model_dump_json() + "\n"
        )
        return type(
            "Process",
            (),
            {
                "stdout": io.StringIO(
                    "PITBENCH_PROGRESS Judge instances 1/1: judge_id/one\n"
                ),
                "wait": lambda self: 0,
            },
        )()

    with patch("pitbench.evaluator.docker_judge.subprocess.Popen", fake_popen):
        assert judge.run() == [observation]
    assert progress_messages == ["Judge instances 1/1: judge_id/one"]


def test_explicit_build_cpu_limit_is_preserved(tmp_path: Path) -> None:
    judge = _judge(tmp_path, "pitbench/pyvrp@sha256:" + "a" * 64)
    judge.cpus = 4.0

    def fake_popen(command, **kwargs):
        assert command[command.index("--cpus") + 1] == "4.0"
        (judge.output_dir / "judge-observations.jsonl").write_text("")
        return type(
            "Process",
            (),
            {"stdout": io.StringIO(), "wait": lambda self: 0},
        )()

    with patch("pitbench.evaluator.docker_judge.subprocess.Popen", fake_popen):
        assert judge.run() == []


def test_docker_judge_forwards_cpuset_states_and_parallelism(tmp_path: Path) -> None:
    judge = _judge(tmp_path, "pitbench/pyvrp@sha256:" + "a" * 64)
    judge.cpuset_cpus = "28-35"
    judge.code_states = (CodeState.AGENT,)
    judge.parallel_runs = 8

    def fake_popen(command, **kwargs):
        assert command[command.index("--cpuset-cpus") + 1] == "28-35"
        assert command[command.index("--parallel-runs") + 1] == "8"
        assert command[command.index("--code-state") + 1] == "agent"
        (judge.output_dir / "judge-observations.jsonl").write_text("")
        return type(
            "Process",
            (),
            {"stdout": io.StringIO(), "wait": lambda self: 0},
        )()

    with patch("pitbench.evaluator.docker_judge.subprocess.Popen", fake_popen):
        assert judge.run() == []


def test_docker_judge_failure_includes_streamed_output(tmp_path: Path) -> None:
    judge = _judge(tmp_path, "pitbench/pyvrp@sha256:" + "a" * 64)

    process = type(
        "Process",
        (),
        {
            "stdout": io.StringIO("judge: validation build failed\n"),
            "wait": lambda self: 2,
        },
    )()
    with (
        patch("pitbench.evaluator.docker_judge.subprocess.Popen", return_value=process),
        pytest.raises(RuntimeError, match="validation build failed"),
    ):
        judge.run()
