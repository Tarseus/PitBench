from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

from adapters.pitbench.adapter import PitBenchAdapter
from pitbench.evaluator.artifacts import artifact_ref, write_json
from pitbench.evaluator.docker_judge import DockerJudge, judge_harness_sha256
from pitbench.evaluator.judge import FixtureJudge, JudgePlan, _evaluation_seeds
from pitbench.evaluator.private_assets import (
    PrivateAssetResolver,
    load_private_seed_robustness_config,
)
from pitbench.evaluator.storage import ObservationStore, cache_lock, file_manifest
from pitbench.evaluator.validity import evaluator_validity
from pitbench.harness.evaluation import EvaluationRequest, Evaluator
from pitbench.metrics.behavior_report import compute_behavior_metric_report
from pitbench.metrics.controlled_transfer_report import (
    compute_controlled_transfer_report,
)
from pitbench.metrics.objective_report import compute_objective_report
from pitbench.metrics.performance_report import (
    ExactPerformanceReport,
    ExactRunOutcomeKind,
    compute_task_performance_report,
)
from pitbench.metrics.reliability_report import compute_reliability_reports
from pitbench.metrics.resource_report import compute_resource_reports
from pitbench.metrics.seed_robustness_report import (
    SeedSelectionMetadata,
    compute_exact_seed_robustness_report,
    compute_seed_robustness_details,
    compute_seed_robustness_report,
    compute_task_seed_robustness_report,
)
from pitbench.schema.evaluation import (
    ArtifactManifest,
    EvaluationResult,
    EvaluationSummary,
)
from pitbench.schema.observation import (
    CodeState,
    ExpectedRun,
    ExpectedRunGrid,
    RunObservation,
)
from pitbench.schema.task import (
    ControlledTransferConfig,
    PerformanceProtocol,
    PitBenchTask,
)


def _default_judge_parallel_runs(task: PitBenchTask) -> int:
    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available = os.cpu_count() or 1
    threads = getattr(task.evaluation, "threads", 1) or 1
    slots = available // threads
    return max(1, slots)


def _default_judge_cpus(parallel_runs: int, task: PitBenchTask) -> float:
    threads = getattr(task.evaluation, "threads", 1) or 1
    return max(float(parallel_runs * threads), 8.0)


def _base_cache_is_complete(
    task: PitBenchTask,
    observations: list[RunObservation],
    private_root: Path,
) -> bool:
    """Reject reduced-panel BASE caches before they contaminate reports."""
    if any(
        item.code_state != CodeState.BASE
        or item.test_suite is not None
        or item.threads != task.evaluation.threads
        for item in observations
    ):
        return False
    seed_robustness = task.evaluation.seed_robustness
    development_seeds = tuple(
        seed_robustness.development_seeds
        if seed_robustness is not None
        else task.evaluation.solver_seeds or ()
    )
    if seed_robustness is not None:
        evaluation_seeds = tuple(
            load_private_seed_robustness_config(
                PrivateAssetResolver(private_root),
                task_id=task.task_id,
                public_config=seed_robustness,
            ).evaluation_seeds
        )
    else:
        evaluation_seeds = development_seeds
    expected_count = 0
    for instance_set in task.instance_sets:
        if instance_set.scenario_only:
            continue
        instance_set_kind = getattr(instance_set, "kind", "judge_id")
        instance_set_kind = getattr(instance_set_kind, "value", instance_set_kind)
        if (
            instance_set_kind == "agent_dev"
            and seed_robustness is None
            and task.evaluation.representation_robustness is None
        ):
            continue
        seeds = (
            development_seeds if instance_set_kind == "agent_dev" else evaluation_seeds
        )
        expected_count += (
            instance_set.size
            * (instance_set.solver_seed_count or len(seeds))
            * len(instance_set.budgets_sec or task.evaluation.budgets_sec)
        )
    keys = {
        (item.instance_set, item.instance_id, item.solver_seed, item.budget_sec)
        for item in observations
        if item.equivalence_parent_id is None
    }
    originals = [row for row in observations if row.equivalence_parent_id is None]
    transformed = [row for row in observations if row.equivalence_parent_id is not None]
    representation = task.evaluation.representation_robustness
    expected_transformed = 0
    if representation is not None:
        source = next(
            p for p in task.instance_sets if p.name == representation.instance_set
        )
        expected_transformed = (
            source.size
            * representation.relabelings_per_instance
            * len(task.evaluation.budgets_sec)
        )
        if any(
            row.solver_seed != representation.solver_seed
            or row.budget_sec not in task.evaluation.budgets_sec
            or row.instance_set != source.name
            for row in transformed
        ):
            return False
    transformed_keys = {
        (r.instance_set, r.instance_id, r.solver_seed, r.budget_sec)
        for r in transformed
    }
    return len(originals) == expected_count == len(keys) and len(
        transformed
    ) == expected_transformed == len(transformed_keys)


def _base_cache_path(
    cache_root: Path,
    task: PitBenchTask,
    task_config_path: Path,
    judge_image: str,
    measurement_identity: dict | None = None,
) -> Path:
    task_config_digest = hashlib.sha256(task_config_path.read_bytes()).hexdigest()[:16]
    judge_image_digest = hashlib.sha256(judge_image.encode()).hexdigest()[:16]
    harness_digest = judge_harness_sha256(Path(__file__).resolve().parents[2])[:16]
    profile = (
        (
            "-profile-"
            + hashlib.sha256(
                json.dumps(measurement_identity, sort_keys=True).encode()
            ).hexdigest()[:16]
        )
        if measurement_identity is not None
        else ""
    )
    return cache_root / (
        f"{task.task_id}-threads-{task.evaluation.threads}-"
        f"task-{task_config_digest}-image-{judge_image_digest}-"
        f"harness-{harness_digest}{profile}_base.parquet"
    )


def _base_input_digest(task: PitBenchTask, private_root: Path) -> str:
    """Bind cached solves to the actual planned inputs and reference values."""
    root = Path(__file__).resolve().parents[2]
    resolver = PrivateAssetResolver(private_root)
    hashes = {}

    def checksum(path):
        if path not in hashes:
            with path.open("rb") as handle:
                hashes[path] = hashlib.file_digest(handle, "sha256").hexdigest()
        return hashes[path]

    with tempfile.TemporaryDirectory(prefix="pitbench-cache-inputs-") as directory:
        plan = JudgePlan.from_instance_set_configs(
            task,
            resolver,
            public_root=root,
            generated_root=Path(directory),
            evaluation_seeds=_evaluation_seeds(task, resolver),
        )
        inputs = []
        for case in plan.cases:
            spec = case.instance_set
            config_path = (
                root / spec.instance_set_config
                if spec.kind.value == "agent_dev"
                else resolver.resolve(
                    spec.instance_set_config, spec.instance_set_config_sha256
                )
            )
            inputs.append(
                {
                    "set": spec.name,
                    "id": case.instance_id,
                    "input_sha256": checksum(case.path),
                    "config_sha256": checksum(config_path),
                    "anchor": case.anchor,
                    "scale": case.problem_scale,
                    "condition": case.condition_id,
                    "template": case.template_id,
                }
            )
        oracle = (
            checksum(resolver.resolve(task.oracle.source, task.oracle.source_sha256))
            if task.evaluation.performance_protocol
            == PerformanceProtocol.EXACT_VERIFIED_SOLVE
            else None
        )
        return hashlib.sha256(
            json.dumps(
                {"inputs": inputs, "oracle_sha256": oracle}, sort_keys=True
            ).encode()
        ).hexdigest()


def _measurement_hardware_identity(available):
    from pitbench.evaluator.scenarios import _hardware_identity

    hardware = _hardware_identity(available)
    hardware.pop("cpu_pool")
    hardware["boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    hardware["smt_widths"] = sorted(
        len(
            set(available)
            & {
                value
                for part in Path(
                    f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
                )
                .read_text()
                .strip()
                .split(",")
                for value in range(
                    int(part.split("-")[0]), int(part.split("-")[-1]) + 1
                )
            }
        )
        for cpu in available
    )
    return hardware


def _base_measurement_identity(config, task, parallel_runs, cpus):
    available = sorted(os.sched_getaffinity(0))
    pool = config.get("judge_cpuset_cpus")
    if pool:
        available = [
            cpu
            for part in pool.split(",")
            for cpu in (range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1))
        ]
    hardware = _measurement_hardware_identity(available)
    return {
        "inputs_sha256": _base_input_digest(task, Path(config["private_root"])),
        "hardware": hardware,
        "cpus": cpus,
        "parallel_runs": parallel_runs,
        "threads": task.evaluation.threads,
        "memory": str(config.get("judge_memory", "8g")),
        "tmpfs_size": str(config.get("judge_tmpfs_size", "4g")),
    }


def _restore_shared_base(path: Path, identity: dict, output: Path):
    receipt_path = path.with_suffix(".json")
    if not path.is_file() or not receipt_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text())
    artifacts = Path(receipt["artifacts"])
    if (
        receipt["measurement_identity"] != identity
        or receipt["observations_sha256"]
        != hashlib.sha256(path.read_bytes()).hexdigest()
        or not artifacts.is_dir()
        or file_manifest(artifacts) != receipt["files"]
    ):
        return None
    for folder in ("base", "representation"):
        if (artifacts / folder).is_dir():
            shutil.copytree(artifacts / folder, output / folder, dirs_exist_ok=True)
    rows = ObservationStore.read(path)
    original = receipt["source_output_dir"]
    fields = ("trajectory_path", "solution_path", "stdout_path", "stderr_path")
    rows = [
        row.model_copy(
            update={
                field: str(output) + getattr(row, field)[len(original) :]
                for field in fields
                if getattr(row, field)
                and getattr(row, field).startswith(original + "/")
            }
        )
        for row in rows
    ]
    write_json(
        output / "shared-base.json", {"cache": str(path), "reused": True, **receipt}
    )
    return rows


def _publish_shared_base(path: Path, identity: dict, output: Path, rows):
    artifacts = path.parent / (path.stem + "-artifacts-" + uuid.uuid4().hex)
    artifacts.mkdir(parents=True)
    if (output / "base").is_dir():
        shutil.copytree(output / "base", artifacts / "base")
    grid_path = output / "expected-run-grid.json"
    if grid_path.is_file():
        grid = ExpectedRunGrid.model_validate_json(grid_path.read_text())

        def key(row):
            return (
                row.code_state,
                row.instance_set,
                row.instance_id,
                row.solver_seed,
                row.budget_sec,
            )

        if {key(row) for row in rows} != {
            key(row) for row in grid.runs if row.code_state == CodeState.BASE
        }:
            raise ValueError(
                "shared Base observations differ from the complete source grid"
            )
        shutil.copy2(grid_path, artifacts / "expected-run-grid.json")
    for name in ("inputs",):
        source = output / "representation" / name
        if source.is_dir():
            shutil.copytree(source, artifacts / "representation" / name)
    if (output / "representation/transformations.json").is_file():
        target = artifacts / "representation/transformations.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output / "representation/transformations.json", target)
    temporary = path.with_suffix(".parquet.tmp")
    ObservationStore.write(temporary, rows)
    temporary.replace(path)
    receipt = {
        "measurement_identity": identity,
        "source_output_dir": str(output),
        "observations_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "artifacts": str(artifacts),
        "files": file_manifest(artifacts),
        "source_checkpoint": json.loads((output / "judge-checkpoint.json").read_text())
        if (output / "judge-checkpoint.json").is_file()
        else None,
    }
    write_json(path.with_suffix(".json"), receipt)
    write_json(
        output / "shared-base.json", {"cache": str(path), "reused": False, **receipt}
    )


def _checkpoint_observations_if_complete(
    output_dir: Path,
    task_id: str,
    identity: dict[str, object],
) -> list[RunObservation] | None:
    """Return a completed Judge checkpoint, without accepting partial data.

    Complete checkpoints go straight to reporting. Matching partial ledgers are
    resumed by DockerJudge; mismatched identities never overwrite old evidence.
    """
    observations_path = output_dir / "judge-observations.jsonl"
    grid_path = output_dir / "expected-run-grid.json"
    marker_path = output_dir / "judge-checkpoint.json"
    if not all(path.is_file() for path in (observations_path, grid_path, marker_path)):
        return None
    try:
        marker = json.loads(marker_path.read_text())
        if marker.get("identity") != identity:
            return None
        if (
            marker.get("observations_sha256")
            != hashlib.sha256(observations_path.read_bytes()).hexdigest()
            or marker.get("expected_grid_sha256")
            != hashlib.sha256(grid_path.read_bytes()).hexdigest()
        ):
            return None
        observations = ObservationStore.read_jsonl(observations_path)
        expected = ExpectedRunGrid.model_validate_json(grid_path.read_text())
    except (OSError, ValueError):
        return None
    if expected.task_id != task_id:
        return None

    def run_identity(item: ExpectedRun | RunObservation) -> tuple[object, ...]:
        return (
            item.code_state,
            item.instance_set,
            item.instance_set_kind,
            item.instance_id,
            item.solver_seed,
            item.budget_sec,
            item.condition_id,
            item.template_id,
            item.equivalence_parent_id,
            item.test_suite,
        )

    planned_states = {
        CodeState(value)
        for value in identity.get("code_states", [state.value for state in CodeState])
    }
    if identity.get("shared_base_sha256"):
        planned_states.add(CodeState.BASE)
    expected_keys = {
        run_identity(item)
        for item in expected.runs
        if item.code_state in planned_states
    }
    observed_keys = {run_identity(item) for item in observations}
    if (
        marker.get("observation_count") != len(observations)
        or len(observations) != len(expected_keys)
        or len(observed_keys) != len(observations)
        or observed_keys != expected_keys
        or any(item.task_id != task_id for item in observations)
    ):
        return None
    return observations


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
        actual_patch_sha256 = None
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

        with ExitStack() as lease:
            is_real_full_eval = not fixture_mode and not config.get(
                "reliability_only", False
            )
            base_observations_path = config.get("base_observations_path")
            use_base_cache = (
                bool(config.get("use_base_cache", False))
                and is_real_full_eval
                and not task.evaluation.operational_reliability
                and set(code_states) == set(CodeState)
                and config.get("instances_per_instance_set") is None
                and config.get("solver_seed_limit") is None
            )
            available = len(os.sched_getaffinity(0))
            if config.get("judge_cpuset_cpus"):
                pool = config["judge_cpuset_cpus"]
                available = sum(
                    int(part.split("-")[-1]) - int(part.split("-")[0]) + 1
                    for part in pool.split(",")
                )
            parallel_runs = min(
                int(
                    config.get("judge_parallel_runs")
                    or _default_judge_parallel_runs(task)
                ),
                max(1, available // task.evaluation.threads),
            )
            cpus = float(
                config.get("judge_cpus")
                or min(available, _default_judge_cpus(parallel_runs, task))
            )
            parallel_runs = min(
                parallel_runs, max(1, int(cpus) // task.evaluation.threads)
            )
            shared_identity = (
                _base_measurement_identity(config, task, parallel_runs, cpus)
                if use_base_cache
                else None
            )
            build_cache_path = (
                Path(config["build_cache_path"])
                if config.get("build_cache_path")
                else (
                    Path(config.get("base_cache_path", ".pitbench/cache/base")).parent
                    / "build"
                    if use_base_cache
                    else None
                )
            )
            shared_input = None
            own_measurements = False
            if (request.output_dir / "judge-observations.jsonl").is_file():
                marker = request.output_dir / "judge-launch.json"
                if not marker.is_file():
                    marker = request.output_dir / "judge-checkpoint.json"
                if marker.is_file():
                    prior = json.loads(marker.read_text())["identity"]
                    own_measurements = set(prior["code_states"]) == {
                        state.value for state in CodeState
                    }
                    if use_base_cache:
                        current = {
                            "task_config_sha256": hashlib.sha256(
                                task_config_path.read_bytes()
                            ).hexdigest(),
                            "candidate_patch_sha256": actual_patch_sha256,
                            "harness_sha256": judge_harness_sha256(
                                Path(__file__).resolve().parents[2]
                            ),
                            "judge_image": config.get("judge_image")
                            or task.repository.judge_image,
                            "cpus": cpus,
                            "memory": str(config.get("judge_memory", "8g")),
                            "cpuset_cpus": config.get("judge_cpuset_cpus"),
                            "parallel_runs": parallel_runs,
                        }
                        if any(
                            prior.get(key) != value for key, value in current.items()
                        ):
                            raise ValueError(
                                "retained Judge identity changed; preserve this output and use a new directory"
                            )
                        if prior.get("shared_base_sha256"):
                            shared_input = (
                                request.output_dir / "shared-base-observations.jsonl"
                            )
                            base_receipt = request.output_dir / "shared-base.json"
                            if (
                                not base_receipt.is_file()
                                or json.loads(base_receipt.read_text())[
                                    "measurement_identity"
                                ]
                                != shared_identity
                            ):
                                raise ValueError(
                                    "retained shared Base hardware or resource profile changed"
                                )
                            if (
                                not shared_input.is_file()
                                or hashlib.sha256(shared_input.read_bytes()).hexdigest()
                                != prior["shared_base_sha256"]
                            ):
                                raise ValueError(
                                    "retained shared Base input is missing or changed"
                                )
                            # Resume with the original local Base even if the
                            # campaign cache has since been removed or repaired.
                            base_observations_path = str(shared_input)

            base_cache_file: Path | None = None

            if base_observations_path is None and use_base_cache:
                cache_root = Path(config.get("base_cache_path", ".pitbench/cache/base"))
                candidate_cache = _base_cache_path(
                    cache_root,
                    task,
                    Path(config["task_config_path"]),
                    str(config.get("judge_image") or task.repository.judge_image),
                    measurement_identity=shared_identity,
                )
                lease.enter_context(cache_lock(candidate_cache.with_suffix(".lock")))
                base_cache_file = candidate_cache
                if not own_measurements:
                    restored = _restore_shared_base(
                        candidate_cache, shared_identity, request.output_dir
                    )
                    if restored is not None:
                        base_observations_path = str(candidate_cache)
                        shared_input = (
                            request.output_dir / "shared-base-observations.jsonl"
                        )
                        ObservationStore.write_jsonl(shared_input, restored)

            cached_base: list[RunObservation] | None = None
            if base_observations_path is not None:
                cached_base = (
                    ObservationStore.read_jsonl(shared_input)
                    if shared_input is not None
                    else ObservationStore.read(Path(base_observations_path))
                )
                if any(item.task_id != task.task_id for item in cached_base):
                    raise ValueError("cached BASE observations belong to another task")
                if any(item.code_state != CodeState.BASE for item in cached_base):
                    raise ValueError(
                        "cached BASE artifact contains non-BASE observations"
                    )
                if not _base_cache_is_complete(
                    task, cached_base, Path(config["private_root"])
                ):
                    cached_base = None
                    base_observations_path = None
                    shared_input = None

            if cached_base is not None and "code_states" not in config:
                judge_code_states = (CodeState.AGENT,)
                # Publication is serialized; candidate solves must fan out after
                # the immutable Base observations and artifacts have been read.
                lease.close()
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
                fixture_plan = JudgePlan.fixture(task, limit)
                observations = FixtureJudge().run(
                    fixture_plan,
                    code_states=judge_code_states,
                )
                expected_run_grid = fixture_plan.expected_run_grid(
                    code_states=tuple(CodeState),
                )
                request.output_dir.mkdir(parents=True, exist_ok=True)
                (request.output_dir / "expected-run-grid.json").write_text(
                    expected_run_grid.model_dump_json(indent=2) + "\n"
                )
            else:
                required = ("base_repository", "private_root")
                missing = [key for key in required if key not in config]
                if missing:
                    raise ValueError(f"real judge missing configuration: {missing}")
                PitBenchAdapter.validate_repository(
                    task, Path(config["base_repository"])
                )
                image = config.get("judge_image") or task.repository.judge_image
                if not image:
                    raise ValueError("real judge requires a pinned judge_image")
                judge_arguments = dict(
                    image=image,
                    task_config_path=task_config_path,
                    base_repository=Path(config["base_repository"]),
                    private_root=Path(config["private_root"]),
                    candidate_patch=request.candidate_patch_path,
                    output_dir=request.output_dir,
                    cpus=cpus,
                    memory=str(config.get("judge_memory", "8g")),
                    tmpfs_size=str(config.get("judge_tmpfs_size", "4g")),
                    cpuset_cpus=config.get("judge_cpuset_cpus"),
                    code_states=judge_code_states,
                    parallel_runs=parallel_runs,
                    progress_callback=progress_callback,
                    reliability_only=bool(config.get("reliability_only", False)),
                    instances_per_instance_set=config.get("instances_per_instance_set"),
                    solver_seed_limit=config.get("solver_seed_limit"),
                    base_observations_path=shared_input,
                    build_cache_path=build_cache_path,
                )
                if (
                    use_base_cache
                    and base_cache_file is not None
                    and cached_base is None
                    and not own_measurements
                ):
                    pool_key = hashlib.sha256(
                        str(config.get("judge_cpuset_cpus")).encode()
                    ).hexdigest()[:12]
                    producer = base_cache_file.parent / (
                        base_cache_file.stem + "-producer-" + pool_key
                    )
                    producer.mkdir(parents=True, exist_ok=True)
                    empty_patch = producer / "candidate.patch"
                    if not empty_patch.exists():
                        empty_patch.write_bytes(b"")
                    baseline_args = {
                        **judge_arguments,
                        "candidate_patch": empty_patch,
                        "output_dir": producer,
                        "code_states": (CodeState.BASE,),
                        "base_observations_path": None,
                    }
                    baseline_judge = DockerJudge(**baseline_args)
                    base_obs = (
                        _checkpoint_observations_if_complete(
                            producer, task.task_id, baseline_judge.checkpoint_identity()
                        )
                        or baseline_judge.run()
                    )
                    if not _base_cache_is_complete(
                        task, base_obs, Path(config["private_root"])
                    ):
                        raise ValueError(
                            "shared Base producer did not return the full declared Base grid"
                        )
                    _publish_shared_base(
                        base_cache_file, shared_identity, producer, base_obs
                    )
                    cached_base = _restore_shared_base(
                        base_cache_file, shared_identity, request.output_dir
                    )
                    shared_input = request.output_dir / "shared-base-observations.jsonl"
                    ObservationStore.write_jsonl(shared_input, cached_base)
                    judge_arguments.update(
                        code_states=(CodeState.AGENT,),
                        base_observations_path=shared_input,
                    )
                    lease.close()
                judge = DockerJudge(**judge_arguments)
                judge_identity = judge.checkpoint_identity()
                observations = None
                if not config.get("reliability_only", False):
                    observations = _checkpoint_observations_if_complete(
                        request.output_dir,
                        task.task_id,
                        judge_identity,
                    )
                    if observations is not None and progress_callback is not None:
                        progress_callback(
                            "Judge checkpoint complete; resuming report aggregation "
                            f"from {len(observations)} observations"
                        )
                if observations is None:
                    observations = judge.run()

            if cached_base is not None:
                existing_states = {item.code_state for item in observations}
                if CodeState.BASE not in existing_states:
                    observations = [*cached_base, *observations]
            elif use_base_cache and base_cache_file is not None and observations:
                base_obs = [
                    item for item in observations if item.code_state == CodeState.BASE
                ]
                if base_obs and _base_cache_is_complete(
                    task, base_obs, Path(config["private_root"])
                ):
                    _publish_shared_base(
                        base_cache_file, shared_identity, request.output_dir, base_obs
                    )

        parquet_path = request.output_dir / "trials.parquet"
        ObservationStore.write(parquet_path, observations)
        expected_run_grid_path = request.output_dir / "expected-run-grid.json"
        expected_run_grid = (
            ExpectedRunGrid.model_validate_json(expected_run_grid_path.read_text())
            if expected_run_grid_path.is_file()
            else None
        )
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
                development_seed_count=seed_robustness.seed_selection.development_seed_count,
            )
            seed_report_inputs = {
                "task_id": task.task_id,
                "budgets_sec": task.evaluation.budgets_sec,
                "primary_budget_sec": task.evaluation.primary_budget_sec,
                "seed_selection": seed_selection,
                "development_seeds": seed_robustness.development_seeds,
                "evaluation_seeds": private_seed_config.evaluation_seeds,
            }
            supplemental_seed_panels = {
                panel.name
                for panel in task.instance_sets
                if panel.performance_protocol is not None
                or panel.budgets_sec is not None
                or panel.solver_seed_count is not None
            }
            native_seed_observations = [
                row
                for row in original_observations
                if row.instance_set not in supplemental_seed_panels
            ]
            if (
                task.evaluation.performance_protocol
                is PerformanceProtocol.EXACT_VERIFIED_SOLVE
            ):
                nuisance_robustness = compute_exact_seed_robustness_report(
                    native_seed_observations,
                    time_basis=task.evaluation.exact_time_basis,
                    **seed_report_inputs,
                )
                seed_robustness_details = nuisance_robustness
            else:
                nuisance_robustness = compute_seed_robustness_report(
                    native_seed_observations,
                    **seed_report_inputs,
                )
                seed_robustness_details = compute_seed_robustness_details(
                    native_seed_observations,
                    **seed_report_inputs,
                )
            if expected_run_grid is not None and any(
                panel.performance_protocol is not None
                or panel.budgets_sec is not None
                or panel.solver_seed_count is not None
                for panel in task.instance_sets
            ):
                nuisance_robustness = compute_task_seed_robustness_report(
                    original_observations,
                    task=task,
                    expected_run_grid=expected_run_grid,
                    evaluation_seeds=private_seed_config.evaluation_seeds,
                )
                if (
                    task.evaluation.performance_protocol
                    is PerformanceProtocol.EXACT_VERIFIED_SOLVE
                ):
                    seed_robustness_details = nuisance_robustness
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
                budgets_by_instance_set={
                    item.name: item.budgets_sec or task.evaluation.budgets_sec
                    for item in task.instance_sets
                },
                expected_instance_counts={
                    item.name: item.size
                    for item in task.instance_sets
                    if not item.scenario_only
                },
                expected_seed_count=(
                    seed_robustness.seed_selection.seed_count
                    if seed_robustness is not None
                    else len(task.evaluation.solver_seeds or [])
                ),
                expected_seed_counts_by_instance_set={
                    item.name: item.solver_seed_count
                    if item.solver_seed_count is not None
                    else len(seed_robustness.development_seeds)
                    for item in task.instance_sets
                    if item.solver_seed_count is not None
                    or (seed_robustness is not None and item.kind.value == "agent_dev")
                },
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
            (
                task.evaluation.operational_reliability
                or config.get("reliability_only", False)
            )
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
            shared_base_receipt=artifact_ref(
                request.output_dir / "shared-base.json",
                root=request.output_dir,
                media_type="application/json",
                private=True,
            )
            if (request.output_dir / "shared-base.json").is_file()
            else None,
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
            expected_run_grid=(
                artifact_ref(
                    expected_run_grid_path,
                    root=request.output_dir,
                    media_type="application/json",
                    private=True,
                )
                if expected_run_grid is not None
                else None
            ),
            seed_robustness_details=seed_robustness_details_ref,
            representation_robustness_details=representation_details_ref,
            resource_details=resource_details_ref,
            reliability_details=reliability_details_ref,
        )
        performance = (
            compute_task_performance_report(
                original_observations,
                task=task,
                expected_run_grid=expected_run_grid,
            )
            if not config.get("reliability_only", False)
            and (
                original_observations
                or task.evaluation.performance_protocol
                == PerformanceProtocol.EXACT_VERIFIED_SOLVE
            )
            else None
        )
        scenario_report = None
        scenario_error = None
        scenario_config = config.get("metric_scenario_config")
        scenario_build_context = {
            "image": config.get("judge_image"),
            "base_repository": config.get("base_repository"),
            "private_root": config.get("private_root"),
            "cpuset_cpus": config.get("judge_cpuset_cpus"),
            "cpus": config.get("judge_cpus") or 8,
            "memory": str(config.get("judge_memory") or "32g"),
            "build_cache_path": str(build_cache_path) if build_cache_path else None,
            "base_cache_path": config.get("base_cache_path")
            if use_base_cache
            else None,
        }
        if task.metric_scenarios and scenario_config and patch_exists:
            from pitbench.evaluator.scenarios import collect_for_evaluation

            try:
                scenario_report, scenario_path = collect_for_evaluation(
                    task,
                    request.candidate_patch_path,
                    request.output_dir,
                    Path(scenario_config),
                    build_context=scenario_build_context,
                )
                artifacts.metric_scenario_campaign = artifact_ref(
                    scenario_path,
                    root=request.output_dir,
                    media_type="application/json",
                    private=True,
                )
            except Exception as error:
                scenario_error = f"{type(error).__name__}: {error}"
                (request.output_dir / "metric-scenario-error.json").write_text(
                    json.dumps({"error": scenario_error}) + "\n"
                )
        controlled_transfer = None
        controlled_transfer_config = getattr(
            task.evaluation,
            "controlled_transfer",
            None,
        )
        if isinstance(controlled_transfer_config, ControlledTransferConfig):
            if expected_run_grid is None:
                raise ValueError("controlled transfer requires an expected run grid")
            controlled_transfer = compute_controlled_transfer_report(
                original_observations,
                task=task,
                expected_run_grid=expected_run_grid,
            )
        exact_qualification_failures = 0
        if isinstance(performance, ExactPerformanceReport):
            exact_qualification_failures = sum(
                budget.agent.outcome_counts[
                    ExactRunOutcomeKind.QUALIFICATION_FAILURE.value
                ]
                for budget in performance.by_budget.values()
            )
        validity = evaluator_validity(
            patch_exists=patch_exists,
            fixture_mode=fixture_mode,
            observations=observations,
            exact_qualification_failures=exact_qualification_failures,
            scenario_verification_failures=sum(
                r.get("code_state") == "agent"
                for report in (scenario_report,)
                if report is not None
                for r in report.invalid_runs
            ),
        )
        return EvaluationResult(
            task_id=task.task_id,
            validity=validity,
            artifacts=artifacts,
            summary=EvaluationSummary(
                base_reuse={
                    key: value
                    for key, value in json.loads(
                        (request.output_dir / "shared-base.json").read_text()
                    ).items()
                    if key
                    in {
                        "cache",
                        "reused",
                        "observations_sha256",
                        "measurement_identity",
                        "source_output_dir",
                    }
                }
                if (request.output_dir / "shared-base.json").is_file()
                else None,
                observation_count=len(observations),
                valid_observation_count=sum(item.valid for item in observations),
                counts_by_state=dict(counts),
                performance=performance,
                controlled_transfer=controlled_transfer,
                nuisance_robustness=nuisance_robustness,
                resource_usage=resource_usage,
                operational_reliability=reliability,
                behavior_geometry=compute_behavior_metric_report(original_observations),
                metric_scenarios=scenario_report,
                metric_scenario_error=scenario_error,
                objective_measurements=compute_objective_report(
                    task=task,
                    performance=performance,
                    resources=resource_usage,
                    seeds=nuisance_robustness,
                    transfer=controlled_transfer,
                    reliability=reliability,
                    observations=observations,
                    validity_accepted=validity.accepted,
                    expected_run_grid=expected_run_grid,
                    scenario_report=scenario_report,
                    candidate_sha256=actual_patch_sha256,
                ),
            ),
        )
