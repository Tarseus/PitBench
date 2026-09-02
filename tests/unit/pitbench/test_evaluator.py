import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from pitbench.evaluator.evaluator import PitBenchEvaluator
from pitbench.evaluator.storage import ObservationStore
from pitbench.harness.evaluation import EvaluationRequest
from pitbench.schema.observation import CodeState, RunObservation, RunStatus
from pitbench.tasks import TaskCatalog

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("record", TaskCatalog(ROOT).validate_all())
def test_every_task_runs_explicit_fixture_grid(record, tmp_path: Path) -> None:
    output = tmp_path / record.task.task_id
    output.mkdir()
    patch = output / "candidate.patch"
    patch.write_text("")
    envelope = PitBenchEvaluator().envelope(
        EvaluationRequest(
            task_id=record.task.task_id,
            task_path=ROOT,
            candidate_patch_path=patch,
            candidate_patch_sha256=hashlib.sha256(patch.read_bytes()).hexdigest(),
            output_dir=output,
            agent_name="fixture",
            evaluator_config={
                "task_config_path": str(record.task_config_path),
                "fixture_mode": True,
                "fixture_instances_per_instance_set": 1,
            },
        )
    )
    assert envelope.completed, envelope.error
    observations = ObservationStore.read(output / "trials.parquet")
    assert observations
    assert {item.code_state for item in observations} == set(CodeState)
    assert all(item.task_id == record.task.task_id for item in observations)
    summary = envelope.payload["summary"]
    assert summary["performance"] is not None
    assert "outcomes" not in summary
    assert "sensitivity" not in summary
    assert "behavior" not in summary
    assert "observations" not in envelope.payload
    assert envelope.payload["artifacts"]["observations"]["private"] is True


def test_real_evaluation_without_private_assets_fails_closed(tmp_path: Path) -> None:
    record = TaskCatalog(ROOT).records()[0]
    output = tmp_path / "output"
    output.mkdir()
    patch = output / "candidate.patch"
    patch.write_text("")
    envelope = PitBenchEvaluator().envelope(
        EvaluationRequest(
            task_id=record.task.task_id,
            task_path=ROOT,
            candidate_patch_path=patch,
            candidate_patch_sha256=hashlib.sha256(patch.read_bytes()).hexdigest(),
            output_dir=output,
            agent_name="test",
            evaluator_config={"task_config_path": str(record.task_config_path)},
        )
    )
    assert envelope.completed is False
    assert "real judge missing configuration" in (envelope.error or "")


def test_real_evaluation_requires_captured_patch_identity(tmp_path: Path) -> None:
    record = TaskCatalog(ROOT).records()[0]
    output = tmp_path / "output"
    output.mkdir()
    candidate = output / "candidate.patch"
    candidate.write_text("")

    envelope = PitBenchEvaluator().envelope(
        EvaluationRequest(
            task_id=record.task.task_id,
            task_path=ROOT,
            candidate_patch_path=candidate,
            output_dir=output,
            agent_name="test",
            evaluator_config={"task_config_path": str(record.task_config_path)},
        )
    )

    assert envelope.completed is False
    assert "real judge requires candidate_patch_sha256" in (envelope.error or "")


def test_evaluator_rejects_candidate_patch_identity_mismatch(
    tmp_path: Path,
) -> None:
    record = TaskCatalog(ROOT).records()[0]
    output = tmp_path / "output"
    output.mkdir()
    candidate = output / "candidate.patch"
    candidate.write_text("")

    envelope = PitBenchEvaluator().envelope(
        EvaluationRequest(
            task_id=record.task.task_id,
            task_path=ROOT,
            candidate_patch_path=candidate,
            candidate_patch_sha256="f" * 64,
            output_dir=output,
            agent_name="test",
            evaluator_config={
                "task_config_path": str(record.task_config_path),
                "fixture_mode": True,
            },
        )
    )

    assert envelope.completed is False
    assert "candidate patch identity mismatch" in (envelope.error or "")


def test_evaluator_rejects_repository_snapshot_mismatch_before_judge(
    tmp_path: Path,
) -> None:
    record = TaskCatalog(ROOT).records()[0]
    output = tmp_path / "output"
    output.mkdir()
    candidate = output / "candidate.patch"
    candidate.write_text("")
    base_repository = tmp_path / "base"
    private_root = tmp_path / "private"
    base_repository.mkdir()
    private_root.mkdir()
    request = EvaluationRequest(
        task_id=record.task.task_id,
        task_path=ROOT,
        candidate_patch_path=candidate,
        candidate_patch_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
        output_dir=output,
        agent_name="test",
        evaluator_config={
            "task_config_path": str(record.task_config_path),
            "base_repository": str(base_repository),
            "private_root": str(private_root),
            "judge_image": "sha256:" + "a" * 64,
        },
    )

    with (
        patch(
            "pitbench.evaluator.evaluator.PitBenchAdapter.validate_repository",
            side_effect=ValueError("repository release identity mismatch"),
        ) as validate_repository,
        patch("pitbench.evaluator.evaluator.DockerJudge") as docker_judge,
    ):
        envelope = PitBenchEvaluator().envelope(request)

    assert envelope.completed is False
    assert "repository release identity mismatch" in (envelope.error or "")
    validate_repository.assert_called_once_with(record.task, base_repository)
    docker_judge.assert_not_called()


def test_real_evaluation_ignores_legacy_local_judge_flag(tmp_path: Path) -> None:
    record = TaskCatalog(ROOT).records()[0]
    output = tmp_path / "output"
    output.mkdir()
    candidate = output / "candidate.patch"
    candidate.write_text("")
    base_repository = tmp_path / "base"
    private_root = tmp_path / "private"
    base_repository.mkdir()
    private_root.mkdir()
    request = EvaluationRequest(
        task_id=record.task.task_id,
        task_path=ROOT,
        candidate_patch_path=candidate,
        candidate_patch_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
        output_dir=output,
        agent_name="test",
        evaluator_config={
            "task_config_path": str(record.task_config_path),
            "base_repository": str(base_repository),
            "private_root": str(private_root),
            "judge_image": "sha256:" + "a" * 64,
            "unsafe_local_judge": True,
        },
    )

    with (
        patch("pitbench.evaluator.evaluator.PitBenchAdapter.validate_repository"),
        patch("pitbench.evaluator.evaluator.DockerJudge") as docker_judge,
    ):
        docker_judge.return_value.run.return_value = []
        envelope = PitBenchEvaluator().envelope(request)

    assert envelope.completed, envelope.error
    docker_judge.assert_called_once()


def test_real_seed_robustness_result_keeps_run_details_private(
    tmp_path: Path,
) -> None:
    record = next(
        item
        for item in TaskCatalog(ROOT).validate_all()
        if item.task.task_id == "pyvrp_v0_14_0"
    )
    seed_robustness = record.task.evaluation.seed_robustness
    assert seed_robustness is not None
    evaluation_seeds = yaml.safe_load(
        (ROOT / "private/seed_robustness/pyvrp_v0_14_0.yaml").read_text()
    )["evaluation_seeds"]
    observations = []
    for instance_set, instance_set_kind, instance_id, seeds in (
        (
            "agent_dev",
            "agent_dev",
            "dev-1",
            seed_robustness.development_seeds,
        ),
        ("judge_id", "judge_id", "judge-1", evaluation_seeds),
    ):
        for budget_sec in record.task.evaluation.budgets_sec:
            for seed_index, solver_seed in enumerate(seeds):
                for code_state, gap_scale in (
                    (CodeState.BASE, 1.0),
                    (CodeState.AGENT, 0.5),
                ):
                    observations.append(
                        RunObservation(
                            task_id=record.task.task_id,
                            code_state=code_state,
                            instance_set=instance_set,
                            instance_set_kind=instance_set_kind,
                            instance_id=instance_id,
                            solver_seed=solver_seed,
                            budget_sec=budget_sec,
                            status=RunStatus.COMPLETED,
                            valid=True,
                            objective=1000 + seed_index,
                            optimal_or_bks=1000,
                            normalized_gap=gap_scale * seed_index / 100,
                        )
                    )

    output = tmp_path / "output"
    output.mkdir()
    candidate = output / "candidate.patch"
    candidate.write_text("")
    base_repository = tmp_path / "base"
    base_repository.mkdir()
    request = EvaluationRequest(
        task_id=record.task.task_id,
        task_path=ROOT,
        candidate_patch_path=candidate,
        candidate_patch_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
        output_dir=output,
        agent_name="test",
        evaluator_config={
            "task_config_path": str(record.task_config_path),
            "base_repository": str(base_repository),
            "private_root": str(ROOT / "private"),
            "judge_image": "sha256:" + "a" * 64,
        },
    )

    with (
        patch("pitbench.evaluator.evaluator.PitBenchAdapter.validate_repository"),
        patch("pitbench.evaluator.evaluator.DockerJudge") as docker_judge,
    ):
        docker_judge.return_value.run.return_value = observations
        envelope = PitBenchEvaluator().envelope(request)

    assert envelope.completed, envelope.error
    assert "observations" not in envelope.payload
    artifacts = envelope.payload["artifacts"]
    assert artifacts["observations"]["private"] is True
    assert artifacts["seed_robustness_details"]["private"] is True
    public_robustness = envelope.payload["summary"]["nuisance_robustness"]
    assert public_robustness is not None
    assert "development_seeds" not in str(public_robustness)
    assert "evaluation_seeds" not in str(public_robustness)
    private_details = json.loads(
        (output / "seed_robustness_details.json").read_text()
    )
    assert private_details["development_seeds"] == seed_robustness.development_seeds
    assert private_details["evaluation_seeds"] == evaluation_seeds


def test_cached_base_observations_merge_with_agent_fixture(tmp_path: Path) -> None:
    record = next(
        item
        for item in TaskCatalog(ROOT).validate_all()
        if item.task.task_id == "pyvrp_v0_14_0"
    )
    patch = tmp_path / "candidate.patch"
    patch.write_text("")
    base_dir = tmp_path / "base"
    agent_dir = tmp_path / "agent"
    base_dir.mkdir()
    agent_dir.mkdir()
    common = {
        "task_config_path": str(record.task_config_path),
        "fixture_mode": True,
        "fixture_instances_per_instance_set": 1,
    }
    base = PitBenchEvaluator().envelope(
        EvaluationRequest(
            task_id=record.task.task_id,
            task_path=ROOT,
            candidate_patch_path=patch,
            output_dir=base_dir,
            agent_name="base",
            evaluator_config={**common, "code_states": ["base"]},
        )
    )
    assert base.completed, base.error

    agent = PitBenchEvaluator().envelope(
        EvaluationRequest(
            task_id=record.task.task_id,
            task_path=ROOT,
            candidate_patch_path=patch,
            output_dir=agent_dir,
            agent_name="agent",
            evaluator_config={
                **common,
                "code_states": ["agent"],
                "base_observations_path": str(base_dir / "trials.parquet"),
            },
        )
    )

    assert agent.completed, agent.error
    observations = ObservationStore.read(agent_dir / "trials.parquet")
    assert {item.code_state for item in observations} == set(CodeState)


def test_model_equivalence_fixture_uses_performance_result_path(
    tmp_path: Path,
) -> None:
    record = next(
        record
        for record in TaskCatalog(ROOT).validate_all()
        if record.task.task_id == "ortools_v9_15"
    )
    output = tmp_path / record.task.task_id
    output.mkdir()
    patch = output / "candidate.patch"
    patch.write_text("")

    envelope = PitBenchEvaluator().envelope(
        EvaluationRequest(
            task_id=record.task.task_id,
            task_path=ROOT,
            candidate_patch_path=patch,
            output_dir=output,
            agent_name="fixture",
            evaluator_config={
                "task_config_path": str(record.task_config_path),
                "fixture_mode": True,
                "fixture_instances_per_instance_set": 1,
            },
        )
    )

    assert envelope.completed, envelope.error
    performance = envelope.payload["summary"]["performance"]
    assert performance["classification"] == "incomplete"
    assert "decision" not in envelope.payload["summary"]
    assert "is_resolved" not in envelope.model_dump()


def test_evaluator_does_not_gate_candidate_paths(tmp_path: Path) -> None:
    record = TaskCatalog(ROOT).records()[0]
    output = tmp_path / "output"
    output.mkdir()
    candidate = output / "candidate.patch"
    candidate.write_text(
        "diff --git a/pitbench/evaluator/judge.py b/pitbench/evaluator/judge.py\n"
        "--- a/pitbench/evaluator/judge.py\n"
        "+++ b/pitbench/evaluator/judge.py\n"
    )
    envelope = PitBenchEvaluator().envelope(
        EvaluationRequest(
            task_id=record.task.task_id,
            task_path=ROOT,
            candidate_patch_path=candidate,
            candidate_patch_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
            output_dir=output,
            agent_name="fixture",
            evaluator_config={
                "task_config_path": str(record.task_config_path),
                "fixture_mode": True,
                "fixture_instances_per_instance_set": 1,
            },
        )
    )

    assert envelope.completed is True
    assert envelope.payload["validity"]["accepted"] is True
    assert envelope.payload["summary"]["observation_count"] > 0


def test_semantically_invalid_agent_run_disqualifies_candidate(
    tmp_path: Path,
) -> None:
    record = TaskCatalog(ROOT).records()[0]
    output = tmp_path / "output"
    output.mkdir()
    candidate = output / "candidate.patch"
    candidate.write_text("")
    invalid = RunObservation(
        task_id=record.task.task_id,
        code_state=CodeState.AGENT,
        instance_set="judge_id",
        instance_set_kind="judge_id",
        instance_id="invalid",
        instance_seed=0,
        solver_seed=0,
        budget_sec=record.task.evaluation.primary_budget_sec,
        status=RunStatus.INVALID,
        valid=False,
    )

    with patch(
        "pitbench.evaluator.evaluator.FixtureJudge.run",
        return_value=[invalid],
    ):
        envelope = PitBenchEvaluator().envelope(
            EvaluationRequest(
                task_id=record.task.task_id,
                task_path=ROOT,
                candidate_patch_path=candidate,
                output_dir=output,
                agent_name="fixture",
                evaluator_config={
                    "task_config_path": str(record.task_config_path),
                    "fixture_mode": True,
                },
            )
        )

    assert envelope.completed is True
    assert envelope.payload["validity"]["accepted"] is False
    assert envelope.payload["validity"]["checks"][-1]["code"] == "solution"
    private_observations = ObservationStore.read(output / "trials.parquet")
    assert private_observations[0].status is RunStatus.INVALID
    assert envelope.payload["artifacts"]["observations"]["private"] is True
    assert envelope.payload["summary"]["performance"] is not None
    assert envelope.payload["summary"]["performance"]["classification"] == "incomplete"
