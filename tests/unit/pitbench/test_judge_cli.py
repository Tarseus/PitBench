import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from typer.testing import CliRunner

from pitbench.cli.main import app
from pitbench.harness.evaluation import EvaluationEnvelope


def test_judge_replays_saved_patch_with_local_config(tmp_path: Path) -> None:
    repository_root = tmp_path / "pitbench"
    repository_root.mkdir()
    config_dir = repository_root / "config"
    config_dir.mkdir()
    snapshot = repository_root / "snapshot"
    snapshot.mkdir()
    private_root = repository_root / "private"
    private_root.mkdir()
    (config_dir / "evaluate.local.yaml").write_text(
        """
paths:
  private_root: private
tasks:
  pyvrp_v0_14_0:
    repository_source: snapshot
    judge_image: sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
"""
    )
    evaluation_dir = tmp_path / "run" / "evaluation"
    evaluation_dir.mkdir(parents=True)
    candidate_patch = evaluation_dir / "candidate.patch"
    candidate_patch.write_text("diff --git a/a b/a\n")
    candidate_patch_sha256 = hashlib.sha256(candidate_patch.read_bytes()).hexdigest()
    task_config_path = repository_root / "configs/tasks/pyvrp.yaml"
    record = SimpleNamespace(
        task_config_path=task_config_path,
        task=SimpleNamespace(
            task_id="pyvrp_v0_14_0",
            repository=SimpleNamespace(judge_image=None),
        ),
    )
    envelope = EvaluationEnvelope(
        evaluator="pitbench",
        evaluator_version="2",
        completed=True,
        payload={
            "summary": {
                "observation_count": 1440,
                "valid_observation_count": 1440,
            }
        },
        is_resolved=True,
    )

    with (
        patch("pitbench.cli.main.TaskCatalog.validate_one", return_value=record),
        patch("pitbench.cli.main.PitBenchEvaluator") as evaluator_class,
    ):
        evaluator_class.return_value.envelope.return_value = envelope
        result = CliRunner().invoke(
            app,
            [
                "judge",
                "pyvrp_v0_14_0",
                str(candidate_patch),
                "--expected-patch-sha256",
                candidate_patch_sha256,
                "--root",
                str(repository_root),
            ],
        )

    assert result.exit_code == 0, result.output
    request = evaluator_class.return_value.envelope.call_args.args[0]
    assert request.candidate_patch_path == candidate_patch
    assert request.candidate_patch_sha256 == candidate_patch_sha256
    assert request.output_dir == evaluation_dir
    assert request.evaluator_config["base_repository"] == str(snapshot)
    assert request.evaluator_config["private_root"] == str(private_root)
    assert request.evaluator_config["judge_cpus"] == 8.0
    assert request.evaluator_config["judge_memory"] == "8g"
    saved = json.loads((evaluation_dir / "evaluation.json").read_text())
    assert saved["completed"] is True
    assert "1440/1440 valid observations" in result.output
