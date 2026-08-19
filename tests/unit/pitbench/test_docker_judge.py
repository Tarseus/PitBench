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


def test_docker_judge_disables_network_and_reads_observations(tmp_path: Path) -> None:
    judge = _judge(tmp_path, "pitbench/pyvrp@sha256:" + "a" * 64)
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

    def fake_run(command, **kwargs):
        assert command[command.index("--network") + 1] == "none"
        assert "--read-only" in command
        (judge.output_dir / "judge-observations.jsonl").write_text(
            observation.model_dump_json() + "\n"
        )
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    with patch("pitbench.evaluator.docker_judge.subprocess.run", fake_run):
        assert judge.run() == [observation]
