from __future__ import annotations

import hashlib
import os
from collections import Counter
from pathlib import Path

from adapters.pitbench.adapter import PitBenchAdapter
from pitbench.evaluator.artifacts import artifact_ref
from pitbench.evaluator.docker_judge import DockerJudge
from pitbench.evaluator.judge import FixtureJudge, JudgePlan
from pitbench.evaluator.private_assets import (
    PrivateAssetResolver,
    load_private_seed_robustness_config,
)
from pitbench.evaluator.storage import ObservationStore
from pitbench.evaluator.validity import evaluator_validity
from pitbench.harness.evaluation import EvaluationRequest, Evaluator
from pitbench.metrics.performance_report import compute_performance_report
from pitbench.metrics.reliability_report import compute_reliability_reports
from pitbench.metrics.resource_report import compute_resource_reports
from pitbench.metrics.seed_robustness_report import (
    SeedSelectionMetadata,
    compute_seed_robustness_details,
    compute_seed_robustness_report,
)
from pitbench.schema.evaluation import (
    ArtifactManifest,
    EvaluationResult,
    EvaluationSummary,
)
from pitbench.schema.observation import CodeState
from pitbench.schema.task import PitBenchTask


def _default_judge_parallel_runs(task: PitBenchTask) -> int:
    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available = os.cpu_count() or 1
    threads = getattr(task.evaluation, "threads", 1) or 1
    slots = available // threads
    if slots >= 6:
        return max(1, slots - 2)
    if slots >= 3:
        return max(1, slots - 1)
    return max(1, slots)


def _default_judge_cpus(parallel_runs: int, task: PitBenchTask) -> float:
    threads = getattr(task.evaluation, "threads", 1) or 1
    return max(float(parallel_runs * threads), 8.0)


class PitBenchEvaluator(Evaluator):
    name = "pitbench"
    version = "6"

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

        is_real_full_eval = not fixture_mode and not config.get("reliability_only", False)
        base_observations_path = config.get("base_observations_path")
        use_base_cache = bool(config.get("use_base_cache", False)) and is_real_full_eval
        base_cache_file: Path | None = None

        if base_observations_path is None and use_base_cache:
            cache_root = Path(config.get("base_cache_path", ".pitbench/cache/base"))
            candidate_cache = cache_root / f"{task.task_id}_base.parquet"
            if candidate_cache.is_file():
                base_observations_path = str(candidate_cache)
            else:
                base_cache_file = candidate_cache

        cached_base: list[RunObservation] | None = None
        if base_observations_path is not None:
            cached_base = ObservationStore.read(Path(base_observations_path))
            if any(item.task_id != task.task_id for item in cached_base):
                raise ValueError("cached BASE observations belong to another task")
            if any(item.code_state != CodeState.BASE for item in cached_base):
                raise ValueError("cached BASE artifact contains non-BASE observations")

        if cached_base is not None and "code_states" not in config:
            judge_code_states = (CodeState.AGENT,)
        else:
            judge_code_states = code_states

        preflight_validity = evaluator_validity(
            patch_exists=patch_exists,
            fixture_mode=fixture_mode,
        )
        if not preflight_validity.accepted:
            observations = []
        elif fixture_mode:
            limit = int(config.get("fixture_instances_per_instance_set", 2))
            observations = FixtureJudge().run(
                JudgePlan.fixture(task, limit), code_states=judge_code_states
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
            parallel_runs = int(
                config.get("judge_parallel_runs") or _default_judge_parallel_runs(task)
            )
            cpus = float(
                config.get("judge_cpus") or _default_judge_cpus(parallel_runs, task)
            )
            observations = DockerJudge(
                image=image,
                task_config_path=task_config_path,
                base_repository=Path(config["base_repository"]),
                private_root=Path(config["private_root"]),
                candidate_patch=request.candidate_patch_path,
                output_dir=request.output_dir,
                cpus=cpus,
                memory=str(config.get("judge_memory", "8g")),
                cpuset_cpus=config.get("judge_cpuset_cpus"),
                code_states=judge_code_states,
                parallel_runs=parallel_runs,
                progress_callback=progress_callback,
                reliability_only=bool(config.get("reliability_only", False)),
            ).run()

        if cached_base is not None:
            existing_states = {item.code_state for item in observations}
            if CodeState.BASE not in existing_states:
                observations = [*cached_base, *observations]
        elif use_base_cache and base_cache_file is not None and observations:
            base_obs = [item for item in observations if item.code_state == CodeState.BASE]
            if base_obs:
                try:
                    ObservationStore.write(base_cache_file, base_obs)
                except Exception:
                    pass

        validity = evaluator_validity(
            patch_exists=patch_exists,
            fixture_mode=fixture_mode,
            observations=observations,
        )

        parquet_path = request.output_dir / "trials.parquet"
        ObservationStore.write(parquet_path, observations)
        counts = Counter(item.code_state for item in observations)
        nuisance_robustness = None
        seed_robustness_details_ref = None
        original_observations = [
            item
            for item in observations
            if item.equivalence_parent_id is None and item.test_suite is None
        ]
        seed_robustness = task.evaluation.seed_robustness
        if original_observations and seed_robustness is not None and not fixture_mode:
            private_seed_config = load_private_seed_robustness_config(
                PrivateAssetResolver(Path(config["private_root"])),
                task_id=task.task_id,
                public_config=seed_robustness,
            )
            seed_selection = SeedSelectionMetadata(
                seed_min=seed_robustness.seed_selection.seed_min,
                seed_max=seed_robustness.seed_selection.seed_max,
                seed_count=seed_robustness.seed_selection.seed_count,
            )
            seed_report_inputs = {
                "task_id": task.task_id,
                "budgets_sec": task.evaluation.budgets_sec,
                "primary_budget_sec": task.evaluation.primary_budget_sec,
                "seed_selection": seed_selection,
                "development_seeds": seed_robustness.development_seeds,
                "evaluation_seeds": private_seed_config.evaluation_seeds,
            }
            nuisance_robustness = compute_seed_robustness_report(
                original_observations,
                **seed_report_inputs,
            )
            seed_robustness_details = compute_seed_robustness_details(
                original_observations,
                **seed_report_inputs,
            )
            seed_robustness_details_path = (
                request.output_dir / "seed_robustness_details.json"
            )
            seed_robustness_details_path.write_text(
                seed_robustness_details.model_dump_json(indent=2) + "\n"
            )
            seed_robustness_details_ref = artifact_ref(
                seed_robustness_details_path,
                root=request.output_dir,
                media_type="application/json",
                private=True,
            )
        representation_details_ref = None
        if (
            task.evaluation.representation_robustness is not None
            and not fixture_mode
            and preflight_validity.accepted
            and not config.get("reliability_only", False)
        ):
            representation_details_ref = artifact_ref(
                request.output_dir / "representation" / "details.json",
                root=request.output_dir,
                media_type="application/json",
                private=True,
            )
        resource_usage = None
        resource_details_ref = None
        if original_observations:
            resource_usage, resource_details = compute_resource_reports(
                original_observations,
                primary_budget_sec=task.evaluation.primary_budget_sec,
                budgets_sec=task.evaluation.budgets_sec,
                expected_instance_counts={
                    item.name: item.size for item in task.instance_sets
                },
                expected_seed_count=(
                    seed_robustness.seed_selection.seed_count
                    if seed_robustness is not None
                    else len(task.evaluation.solver_seeds or [])
                ),
            )
            resource_details_path = request.output_dir / "resource_details.json"
            resource_details_path.write_text(
                resource_details.model_dump_json(indent=2) + "\n"
            )
            resource_details_ref = artifact_ref(
                resource_details_path,
                root=request.output_dir,
                media_type="application/json",
                private=True,
            )
        reliability = None
        reliability_details_ref = None
        if (
            task.evaluation.operational_reliability
            and not fixture_mode
            and preflight_validity.accepted
        ):
            reliability, reliability_details = compute_reliability_reports(
                observations,
                task=task,
                code_states=tuple(CodeState)
                if base_observations_path is not None
                else code_states,
            )
            reliability_dir = request.output_dir / "reliability"
            try:
                reliability_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
            details_path = reliability_dir / "details.json"
            try:
                details_path.write_text(
                    reliability_details.model_dump_json(indent=2) + "\n"
                )
                (reliability_dir / "report.json").write_text(
                    reliability.model_dump_json(indent=2) + "\n"
                )
            except PermissionError:
                details_path = request.output_dir / "reliability_details.json"
                details_path.write_text(
                    reliability_details.model_dump_json(indent=2) + "\n"
                )
                (request.output_dir / "reliability_report.json").write_text(
                    reliability.model_dump_json(indent=2) + "\n"
                )
            reliability_details_ref = artifact_ref(
                details_path,
                root=request.output_dir,
                media_type="application/json",
                private=True,
            )
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
                private=True,
            ),
            seed_robustness_details=seed_robustness_details_ref,
            representation_robustness_details=representation_details_ref,
            resource_details=resource_details_ref,
            reliability_details=reliability_details_ref,
        )
        performance = (
            compute_performance_report(
                original_observations,
                primary_budget_sec=task.evaluation.primary_budget_sec,
            )
            if original_observations
            else None
        )
        return EvaluationResult(
            task_id=task.task_id,
            validity=validity,
            artifacts=artifacts,
            summary=EvaluationSummary(
                observation_count=len(observations),
                valid_observation_count=sum(item.valid for item in observations),
                counts_by_state=dict(counts),
                performance=performance,
                nuisance_robustness=nuisance_robustness,
                resource_usage=resource_usage,
                operational_reliability=reliability,
            ),
        )
