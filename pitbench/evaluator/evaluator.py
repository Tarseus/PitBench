from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

from adapters.pitbench.adapter import PitBenchAdapter
from pitbench.evaluator.artifacts import artifact_ref
from pitbench.evaluator.docker_judge import DockerJudge
from pitbench.evaluator.judge import FixtureJudge, JudgePlan
from pitbench.evaluator.storage import ObservationStore
from pitbench.evaluator.validity import evaluator_validity
from pitbench.harness.evaluation import EvaluationRequest, Evaluator
from pitbench.metrics.decision_metrics import compute_performance_decision
from pitbench.metrics.performance_report import compute_performance_report
from pitbench.schema.evaluation import (
    ArtifactManifest,
    EvaluationResult,
    EvaluationSummary,
)
from pitbench.schema.observation import CodeState
from pitbench.schema.task import PitBenchTask


class PitBenchEvaluator(Evaluator):
    name = "pitbench"
    version = "5"

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        config = request.evaluator_config
        progress_callback = config.get("_progress_callback")
        if not callable(progress_callback):
            progress_callback = None
        task_config_path = Path(config["task_config_path"])
        task = PitBenchTask.from_yaml(task_config_path)
        if task.task_id != request.task_id:
            raise ValueError("task ID does not match evaluator task config")

        patch_exists = request.candidate_patch_path.is_file()
        fixture_mode = bool(config.get("fixture_mode", False))
        if patch_exists:
            actual_patch_sha256 = hashlib.sha256(
                request.candidate_patch_path.read_bytes()
            ).hexdigest()
            if request.candidate_patch_sha256 is None:
                if not fixture_mode:
                    raise ValueError("real judge requires candidate_patch_sha256")
            elif actual_patch_sha256 != request.candidate_patch_sha256:
                raise ValueError(
                    "candidate patch identity mismatch: "
                    f"{actual_patch_sha256} != {request.candidate_patch_sha256}"
                )
        code_states = tuple(
            CodeState(value)
            for value in config.get("code_states", [state.value for state in CodeState])
        )
        preflight_validity = evaluator_validity(
            patch_exists=patch_exists,
            fixture_mode=fixture_mode,
        )
        if not preflight_validity.accepted:
            observations = []
        elif fixture_mode:
            limit = int(config.get("fixture_instances_per_instance_set", 2))
            observations = FixtureJudge().run(
                JudgePlan.fixture(task, limit), code_states=code_states
            )
        else:
            required = ("base_repository", "private_root")
            missing = [key for key in required if key not in config]
            if missing:
                raise ValueError(f"real judge missing configuration: {missing}")
            PitBenchAdapter.validate_repository(task, Path(config["base_repository"]))
            image = config.get("judge_image") or task.repository.judge_image
            if not image:
                raise ValueError("real judge requires a pinned judge_image")
            observations = DockerJudge(
                image=image,
                task_config_path=task_config_path,
                base_repository=Path(config["base_repository"]),
                private_root=Path(config["private_root"]),
                candidate_patch=request.candidate_patch_path,
                output_dir=request.output_dir,
                cpus=float(config.get("judge_cpus", 8.0)),
                memory=str(config.get("judge_memory", "8g")),
                cpuset_cpus=config.get("judge_cpuset_cpus"),
                code_states=code_states,
                parallel_runs=int(config.get("judge_parallel_runs", 1)),
                progress_callback=progress_callback,
            ).run()

        base_observations_path = config.get("base_observations_path")
        if base_observations_path is not None:
            cached_base = ObservationStore.read(Path(base_observations_path))
            if any(item.task_id != task.task_id for item in cached_base):
                raise ValueError("cached BASE observations belong to another task")
            if any(item.code_state != CodeState.BASE for item in cached_base):
                raise ValueError("cached BASE artifact contains non-BASE observations")
            observations = [*cached_base, *observations]

        validity = evaluator_validity(
            patch_exists=patch_exists,
            fixture_mode=fixture_mode,
            observations=observations,
        )

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
        performance = (
            compute_performance_report(
                observations,
                primary_budget_sec=task.evaluation.primary_budget_sec,
            )
            if observations
            else None
        )
        if performance is None:
            decision = None
        else:
            decision = compute_performance_decision(
                performance,
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
                performance=performance,
                decision=decision,
            ),
        )
