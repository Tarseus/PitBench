from __future__ import annotations

from collections import Counter
from pathlib import Path

from pitbench.evaluator.artifacts import artifact_ref
from pitbench.evaluator.docker_judge import DockerJudge
from pitbench.evaluator.judge import FixtureJudge, JudgePlan, LocalProcessJudge
from pitbench.evaluator.patch_policy import PatchPolicy, PatchPolicyResult
from pitbench.evaluator.storage import ObservationStore
from pitbench.evaluator.validity import evaluator_validity
from pitbench.harness.evaluation import EvaluationRequest, Evaluator
from pitbench.metrics.behavior_metrics import compute_behavior_metric_report
from pitbench.metrics.decision_metrics import (
    compute_benchmark_decision,
    compute_model_build_decision,
)
from pitbench.metrics.outcome_metrics import compute_outcome_metrics
from pitbench.metrics.sensitivity_metrics import compute_sensitivity_report
from pitbench.schema.evaluation import (
    ArtifactManifest,
    EvaluationResult,
    EvaluationSummary,
)
from pitbench.schema.task import PitBenchTask, TaskType


class PitBenchEvaluator(Evaluator):
    name = "pitbench"
    version = "2"

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        config = request.evaluator_config
        manifest_path = Path(config["manifest_path"])
        task = PitBenchTask.from_yaml(manifest_path)
        if task.task_id != request.task_id:
            raise ValueError("task ID does not match evaluator manifest")

        patch_exists = request.candidate_patch_path.is_file()
        policy = (
            PatchPolicy().inspect(request.candidate_patch_path)
            if patch_exists
            else PatchPolicyResult(accepted=False, violations=["patch missing"])
        )
        fixture_mode = bool(config.get("fixture_mode", False))
        validity = evaluator_validity(
            patch_exists=patch_exists,
            policy_passed=policy.accepted,
            fixture_mode=fixture_mode,
        )
        if not validity.accepted:
            observations = []
        elif fixture_mode:
            limit = int(config.get("fixture_instances_per_population", 2))
            observations = FixtureJudge().run(JudgePlan.fixture(task, limit))
        else:
            required = ("base_repository", "private_root")
            missing = [key for key in required if key not in config]
            if missing:
                raise ValueError(f"real judge missing configuration: {missing}")
            if config.get("unsafe_local_judge", False):
                observations = LocalProcessJudge(
                    task=task,
                    base_repository=Path(config["base_repository"]),
                    private_root=Path(config["private_root"]),
                    candidate_patch=request.candidate_patch_path,
                    output_dir=request.output_dir,
                ).run()
            else:
                image = config.get("judge_image") or task.repository.judge_image
                if not image:
                    raise ValueError("real judge requires a pinned judge_image")
                observations = DockerJudge(
                    image=image,
                    manifest_path=manifest_path,
                    base_repository=Path(config["base_repository"]),
                    private_root=Path(config["private_root"]),
                    candidate_patch=request.candidate_patch_path,
                    output_dir=request.output_dir,
                    cpus=float(config.get("judge_cpus", 1.0)),
                    memory=str(config.get("judge_memory", "8g")),
                ).run()

        parquet_path = request.output_dir / "trials.parquet"
        ObservationStore.write(parquet_path, observations)
        counts = Counter(item.code_state for item in observations)
        artifacts = ArtifactManifest(
            candidate_patch=(
                artifact_ref(
                    request.candidate_patch_path,
                    root=request.output_dir,
                    media_type="text/x-diff",
                )
                if patch_exists
                else None
            ),
            observations=artifact_ref(
                parquet_path,
                root=request.output_dir,
                media_type="application/vnd.apache.parquet",
            ),
        )
        outcomes = compute_outcome_metrics(observations) if observations else None
        behavior = (
            compute_behavior_metric_report(observations) if observations else None
        )
        sensitivity = compute_sensitivity_report(observations) if observations else None
        decision = None
        if outcomes is not None and sensitivity is not None:
            if task.task_type in {TaskType.MODEL_BUILD, TaskType.PRESOLVE}:
                decision = compute_model_build_decision(
                    outcomes,
                    sensitivity,
                    observations,
                    task.evaluation.decision,
                    validity_accepted=validity.accepted,
                )
            else:
                decision = compute_benchmark_decision(
                    outcomes,
                    sensitivity,
                    task.evaluation.decision,
                    validity_accepted=validity.accepted,
                )
        return EvaluationResult(
            task_id=task.task_id,
            validity=validity,
            observations=observations,
            artifacts=artifacts,
            summary=EvaluationSummary(
                observation_count=len(observations),
                valid_observation_count=sum(item.valid for item in observations),
                counts_by_state=dict(counts),
                outcomes=outcomes,
                sensitivity=sensitivity,
                behavior=behavior,
                decision=decision,
            ),
        )
