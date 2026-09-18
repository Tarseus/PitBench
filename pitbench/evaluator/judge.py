from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Callable

import yaml

from pitbench.evaluator.private_assets import (
    PrivateAssetResolver,
    load_private_seed_robustness_config,
)
from pitbench.instances import materialize_generated_instance_set, verify_public_file
from pitbench.problem_families.base import ProblemFamilyPlugin, ProblemFamilyRegistry
from pitbench.problem_families.verification import (
    ExternalVerifierFamily,
    TrustedOptimumFamily,
    TrustedOptimumOracle,
)
from pitbench.repositories.base import (
    BuildKind,
    CommandSpec,
    RepositoryPluginRegistry,
    SolverRunSpec,
)
from pitbench.schema.observation import (
    CodeState,
    ExpectedRun,
    ExpectedRunGrid,
    RunObservation,
    RunStatus,
)
from pitbench.schema.task import (
    InstanceSetKind,
    InstanceSetSpec,
    PerformanceProtocol,
    PitBenchTask,
)


@dataclass(frozen=True)
class InstanceCase:
    instance_set: InstanceSetSpec
    instance_id: str
    path: Path | None
    anchor: float | None
    problem_scale: float | None = None
    equivalence_parent_id: str | None = None
    equivalence_transform: str | None = None
    solver_seeds: tuple[int, ...] | None = None
    budgets_sec: tuple[float, ...] | None = None
    test_suite: str | None = None
    verifier: ProblemFamilyPlugin | None = None


def _customer_count(path: Path | None) -> float | None:
    if path is None or path.suffix.lower() != ".json":
        return None
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    coordinates = payload.get("coordinates")
    if not isinstance(coordinates, list) or not coordinates:
        return None
    return float(len(coordinates) - 1)


def _uses_model_equivalence(task: PitBenchTask) -> bool:
    return task.evaluation.verifier.rsplit("/", 1)[-1].endswith("model_equivalence")


def _development_seeds(task: PitBenchTask) -> tuple[int, ...]:
    seed_robustness = task.evaluation.seed_robustness
    if seed_robustness is not None:
        return tuple(seed_robustness.development_seeds)
    if task.evaluation.solver_seeds is None:
        raise ValueError("task does not provide development seeds")
    return tuple(task.evaluation.solver_seeds)


def _exact_target_verifier(
    record,
    resolver: PrivateAssetResolver,
) -> TrustedOptimumFamily:
    reference_solution_path = None
    if record.reference_solution_uri is not None:
        reference_solution_path = resolver.resolve(
            record.reference_solution_uri,
            record.reference_solution_sha256,
        )
    return TrustedOptimumFamily(record, reference_solution_path)


def _instance_generation_seeds(
    instance_set: InstanceSetSpec,
) -> dict[str, int | None]:
    randomness = instance_set.randomness
    return {
        "instance_seed": randomness.instance_seed if randomness is not None else None,
        "coordinate_seed": (
            randomness.coordinate_seed if randomness is not None else None
        ),
        "demand_seed": randomness.demand_seed if randomness is not None else None,
    }


def _evaluation_seeds(
    task: PitBenchTask,
    resolver: PrivateAssetResolver,
) -> tuple[int, ...]:
    seed_robustness = task.evaluation.seed_robustness
    if seed_robustness is not None:
        private_config = load_private_seed_robustness_config(
            resolver,
            task_id=task.task_id,
            public_config=seed_robustness,
        )
        return tuple(private_config.evaluation_seeds)
    if task.evaluation.solver_seeds is None:
        raise ValueError("task does not provide evaluation seeds")
    return tuple(task.evaluation.solver_seeds)


def _limit_solver_cpus(
    command: CommandSpec,
    threads: int,
    cpu_ids: tuple[int, ...] | None = None,
) -> CommandSpec:
    """Bind a solver process to its declared CPU count without throttling builds."""

    available = sorted(cpu_ids or os.sched_getaffinity(0))
    if threads > len(available):
        raise ValueError(
            f"solver requests {threads} threads but only {len(available)} CPUs are "
            "available"
        )
    cpu_list = ",".join(str(cpu) for cpu in available[:threads])
    return command.model_copy(
        update={"argv": ["taskset", "--cpu-list", cpu_list, *command.argv]}
    )


class JudgePlan:
    """Expands task protocol into a common-random-number evaluation grid."""

    def __init__(self, task: PitBenchTask, cases: list[InstanceCase]) -> None:
        self.task = task
        self.cases = cases

    def expected_run_grid(
        self,
        *,
        evaluation_seeds: tuple[int, ...] | None = None,
        code_states: tuple[CodeState, ...] = tuple(CodeState),
    ) -> ExpectedRunGrid:
        runs = []
        for case in self.cases:
            seeds = (
                case.solver_seeds or evaluation_seeds or _development_seeds(self.task)
            )
            budgets = case.budgets_sec or tuple(self.task.evaluation.budgets_sec)
            for seed in seeds:
                for budget in budgets:
                    for state in code_states:
                        runs.append(
                            ExpectedRun(
                                task_id=self.task.task_id,
                                code_state=state,
                                instance_set=case.instance_set.name,
                                instance_set_kind=case.instance_set.kind.value,
                                instance_id=case.instance_id,
                                solver_seed=seed,
                                budget_sec=budget,
                                equivalence_parent_id=case.equivalence_parent_id,
                                test_suite=case.test_suite,
                            )
                        )
        runs.sort(
            key=lambda item: (
                item.instance_set,
                item.instance_id,
                item.solver_seed,
                item.budget_sec,
                item.code_state.value,
            )
        )
        return ExpectedRunGrid(task_id=self.task.task_id, runs=runs)

    @classmethod
    def fixture(
        cls, task: PitBenchTask, instances_per_instance_set: int = 2
    ) -> "JudgePlan":
        cases = []
        for instance_set in task.instance_sets:
            if (
                instance_set.kind == InstanceSetKind.AGENT_DEV
                and task.evaluation.seed_robustness is None
            ):
                continue
            for index in range(min(instances_per_instance_set, instance_set.size)):
                objective_scored = not _uses_model_equivalence(task)
                cases.append(
                    InstanceCase(
                        instance_set=instance_set,
                        instance_id=f"{instance_set.name}_{index:04d}",
                        path=None,
                        anchor=1000.0 if objective_scored else None,
                        problem_scale=float(index + 1),
                    )
                )
        return cls(task, cases)

    @classmethod
    def from_instance_set_configs(
        cls,
        task: PitBenchTask,
        resolver: PrivateAssetResolver,
        *,
        public_root: Path,
        generated_root: Path | None = None,
        evaluation_seeds: tuple[int, ...] | None = None,
    ) -> "JudgePlan":
        cases: list[InstanceCase] = []
        development_seeds = _development_seeds(task)
        trusted_records = {}
        if (
            task.evaluation.performance_protocol
            == PerformanceProtocol.EXACT_VERIFIED_SOLVE
        ):
            if task.oracle.kind not in {"known_optimum", "best_known_solution"}:
                raise ValueError(
                    "exact verified solve requires a known-optimum or best-known-solution oracle"
                )
            trusted_oracle_path = resolver.resolve(
                task.oracle.source,
                task.oracle.source_sha256,
            )
            trusted_oracle = TrustedOptimumOracle.from_yaml(trusted_oracle_path)
            if trusted_oracle.task_id != task.task_id:
                raise ValueError("trusted optimum oracle belongs to a different task")
            if any(
                record.objective_sense != task.oracle.objective_sense
                for record in trusted_oracle.records
            ):
                raise ValueError("trusted optimum objective sense differs from task")
            expected_target_kind = (
                "trusted_optimum"
                if task.oracle.kind == "known_optimum"
                else "published_bks"
            )
            if any(
                record.target_kind != expected_target_kind
                for record in trusted_oracle.records
            ):
                raise ValueError("exact target kind differs from task oracle kind")
            trusted_records = {
                (record.instance_set, record.instance_id): record
                for record in trusted_oracle.records
            }
        for instance_set in task.instance_sets:
            public_instance_set = instance_set.kind == InstanceSetKind.AGENT_DEV
            solver_seeds = (
                development_seeds if public_instance_set else evaluation_seeds
            )
            if (
                public_instance_set
                and task.evaluation.seed_robustness is None
                and task.evaluation.representation_robustness is None
            ):
                continue
            if public_instance_set:
                instance_set_config_path = verify_public_file(
                    public_root.resolve(),
                    instance_set.instance_set_config,
                    instance_set.instance_set_config_sha256,
                )
            else:
                instance_set_config_path = resolver.resolve(
                    instance_set.instance_set_config,
                    instance_set.instance_set_config_sha256,
                )
            payload = yaml.safe_load(instance_set_config_path.read_text())
            if "generator" in payload:
                if generated_root is None:
                    raise ValueError(
                        f"instance set {instance_set.name} requires a generated_root"
                    )
                paths = materialize_generated_instance_set(
                    payload,
                    generated_root / instance_set.name,
                    expected_visibility=("agent" if public_instance_set else "judge"),
                    stem_prefix=instance_set.name,
                )
                if len(paths) != instance_set.size:
                    raise ValueError(
                        f"instance set {instance_set.name} size does not match config"
                    )
                trusted_instance_records = {
                    instance_id: record
                    for (
                        record_instance_set,
                        instance_id,
                    ), record in trusted_records.items()
                    if record_instance_set == instance_set.name
                }
                generated_instance_ids = {path.stem for path in paths}
                if (
                    instance_set.kind == InstanceSetKind.JUDGE_ID
                    and task.evaluation.performance_protocol
                    == PerformanceProtocol.EXACT_VERIFIED_SOLVE
                    and not generated_instance_ids <= set(trusted_instance_records)
                ):
                    raise ValueError(
                        f"instance set {instance_set.name} lacks trusted optimum support "
                        "for generated instances"
                    )
                anchors: dict[str, dict] = {}
                oracle_spec = payload.get("oracle")
                if oracle_spec is not None:
                    oracle_path = resolver.resolve(
                        oracle_spec["uri"], oracle_spec.get("sha256")
                    )
                    oracle_payload = yaml.safe_load(oracle_path.read_text())
                    for item in oracle_payload.get("anchors", []):
                        instance_id = item["id"]
                        if instance_id in anchors:
                            raise ValueError(
                                f"instance set {instance_set.name} has duplicate anchor "
                                f"for {instance_id}"
                            )
                        resolver.resolve(
                            item["bks_solution_uri"],
                            item.get("bks_solution_sha256"),
                        )
                        anchors[instance_id] = item
                    expected_ids = {path.stem for path in paths}
                    if set(anchors) != expected_ids:
                        raise ValueError(
                            f"instance set {instance_set.name} oracle support does not "
                            "match generated instances"
                        )
                for path in paths:
                    anchor = anchors.get(path.stem)
                    trusted_record = trusted_instance_records.get(path.stem)
                    if anchor is not None:
                        instance_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
                        if instance_sha256 != anchor.get("instance_sha256"):
                            raise ValueError(
                                f"instance set {instance_set.name} generated instance "
                                f"hash mismatch for {path.stem}"
                            )
                    trusted_verifier = None
                    if trusted_record is not None:
                        trusted_verifier = _exact_target_verifier(
                            trusted_record, resolver
                        )
                    cases.append(
                        InstanceCase(
                            instance_set=instance_set,
                            instance_id=path.stem,
                            path=path,
                            anchor=(
                                float(trusted_record.optimal_objective)
                                if trusted_record is not None
                                else float(anchor["bks"])
                                if anchor is not None
                                else None
                            ),
                            problem_scale=_customer_count(path),
                            solver_seeds=solver_seeds,
                            verifier=trusted_verifier,
                        )
                    )
                continue
            if len(payload["instances"]) != instance_set.size:
                raise ValueError(
                    f"instance set {instance_set.name} size does not match config"
                )
            exact_instance_records = {
                instance_id: record
                for (
                    record_instance_set,
                    instance_id,
                ), record in trusted_records.items()
                if record_instance_set == instance_set.name
            }
            if (
                instance_set.kind == InstanceSetKind.JUDGE_ID
                and task.evaluation.performance_protocol
                == PerformanceProtocol.EXACT_VERIFIED_SOLVE
                and set(exact_instance_records)
                != {item["id"] for item in payload["instances"]}
            ):
                raise ValueError(
                    f"instance set {instance_set.name} exact target support does not "
                    "match configured instances"
                )
            for item in payload["instances"]:
                exact_record = exact_instance_records.get(item["id"])
                if public_instance_set:
                    path = verify_public_file(
                        instance_set_config_path.parent,
                        item["instance_file"],
                        item["instance_file_sha256"],
                    )
                    verify_public_file(
                        instance_set_config_path.parent,
                        item["bks_solution_file"],
                        item["bks_solution_file_sha256"],
                    )
                    anchor = item.get("bks")
                    if anchor is None:
                        raise ValueError(
                            f"instance set {instance_set.name} has no BKS for "
                            f"{item['id']}"
                        )
                else:
                    uri = item.get("uri", item.get("instance_uri"))
                    anchor = item.get("optimal_or_bks", item.get("bks"))
                    objective_scored = (
                        instance_set.kind == InstanceSetKind.JUDGE_ID
                        and not _uses_model_equivalence(task)
                    )
                    if uri is None or (objective_scored and anchor is None):
                        raise ValueError(
                            f"instance set {instance_set.name} has an incomplete "
                            "instance"
                        )
                    path = resolver.resolve(uri)
                if exact_record is not None:
                    if anchor != exact_record.optimal_objective:
                        raise ValueError(
                            f"instance set {instance_set.name} exact target value "
                            f"does not match {item['id']}"
                        )
                    anchor = exact_record.optimal_objective
                cases.append(
                    InstanceCase(
                        instance_set=instance_set,
                        instance_id=item["id"],
                        path=path,
                        anchor=float(anchor) if anchor is not None else None,
                        problem_scale=_customer_count(path),
                        solver_seeds=solver_seeds,
                        verifier=(
                            _exact_target_verifier(exact_record, resolver)
                            if exact_record is not None
                            else None
                        ),
                    )
                )
        return cls(task, cases)


class FixtureJudge:
    """Deterministic contract smoke test; never selected implicitly."""

    _STATE_FACTOR = {
        CodeState.BASE: 1.0,
        CodeState.AGENT: 0.82,
    }

    def run(
        self,
        plan: JudgePlan,
        code_states: tuple[CodeState, ...] = tuple(CodeState),
    ) -> list[RunObservation]:
        observations: list[RunObservation] = []
        task = plan.task
        for case in plan.cases:
            seeds = case.solver_seeds or _development_seeds(task)
            budgets = case.budgets_sec or tuple(task.evaluation.budgets_sec)
            for seed in seeds:
                for budget in budgets:
                    for state in code_states:
                        observations.append(
                            self._observation(task, case, state, seed, budget)
                        )
        return observations

    def _observation(
        self,
        task: PitBenchTask,
        case: InstanceCase,
        state: CodeState,
        seed: int,
        budget: float,
    ) -> RunObservation:
        digest = hashlib.sha256(
            f"{task.task_id}:{case.instance_id}:{seed}".encode()
        ).digest()
        noise = int.from_bytes(digest[:2], "big") / 65535 * 0.01
        factor = self._STATE_FACTOR[state]
        common = dict(
            task_id=task.task_id,
            code_state=state,
            instance_set=case.instance_set.name,
            instance_set_kind=case.instance_set.kind.value,
            instance_id=case.instance_id,
            **_instance_generation_seeds(case.instance_set),
            solver_seed=seed,
            budget_sec=budget,
            threads=task.evaluation.threads,
            status=RunStatus.COMPLETED,
            valid=True,
            wall_time_sec=budget,
            problem_scale=case.problem_scale,
            equivalence_parent_id=case.equivalence_parent_id,
            equivalence_transform=case.equivalence_transform,
        )
        if _uses_model_equivalence(task):
            variables = {
                CodeState.BASE: 9011,
                CodeState.AGENT: 1200,
            }[state]
            return RunObservation(
                **common,
                model_variables=variables,
                model_constraints=max(1, variables // 2),
                cpu_time_sec=budget * factor,
                peak_rss_bytes=int(128 * 1024 * 1024 * factor),
            )
        objective_reference = case.anchor if case.anchor is not None else 1000.0
        gap = (0.08 + noise) * factor / max(budget, 1) ** 0.25
        return RunObservation(
            **common,
            objective=objective_reference * (1 + gap),
            optimal_or_bks=case.anchor,
            normalized_gap=gap if case.anchor is not None else None,
            iterations=int(budget * 100 / factor),
            nodes=(
                int(budget * 20 / factor)
                if task.problem_family.value == "mip"
                else None
            ),
        )


class LocalProcessJudge:
    """Reference judge engine used inside an isolated judge container."""

    def __init__(
        self,
        task: PitBenchTask,
        base_repository: Path,
        public_root: Path,
        private_root: Path,
        candidate_patch: Path | None,
        output_dir: Path,
        code_states: tuple[CodeState, ...] = tuple(CodeState),
        parallel_runs: int = 1,
        run_validation_builds: bool = True,
        progress_callback: Callable[[str], None] | None = None,
        evaluation_seeds: tuple[int, ...] | None = None,
        family: ProblemFamilyPlugin | None = None,
        additional_cases: list[InstanceCase] | None = None,
    ) -> None:
        self.task = task
        self.base_repository = base_repository
        self.public_root = public_root
        self.resolver = PrivateAssetResolver(private_root)
        self.evaluation_seeds = evaluation_seeds or _evaluation_seeds(
            task, self.resolver
        )
        self.candidate_patch = candidate_patch
        self.output_dir = output_dir
        self.code_states = code_states
        self.parallel_runs = parallel_runs
        self.run_validation_builds = run_validation_builds
        self.progress_callback = progress_callback
        self.additional_cases = additional_cases or []
        self.repository = RepositoryPluginRegistry.load(task.repository.plugin)
        self.family = family or ProblemFamilyRegistry.load(task.problem_family)
        if isinstance(
            self.family,
            ExternalVerifierFamily,
        ) and task.evaluation.verifier.startswith("private://"):
            self.family.verifier = self.resolver.resolve(task.evaluation.verifier)

    @staticmethod
    def _run(command: CommandSpec, workspace: Path) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.update(command.env)
        with subprocess.Popen(
            command.argv,
            cwd=workspace / command.cwd,
            env=None if not environment else environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=True,
        ) as process:
            try:
                stdout, stderr = process.communicate(timeout=command.timeout_sec)
            except subprocess.TimeoutExpired as error:
                # A timed-out wrapper must not leave a native solver running behind it.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = process.communicate()
                raise subprocess.TimeoutExpired(
                    command.argv, command.timeout_sec, output=stdout, stderr=stderr
                ) from error
            return subprocess.CompletedProcess(
                command.argv, process.returncode, stdout, stderr
            )

    def _state_patch(self, state: CodeState) -> Path | None:
        if state == CodeState.BASE:
            return None
        if self.candidate_patch is None:
            raise ValueError("agent judge state requires a candidate patch")
        return self.candidate_patch

    def _workspace(self, state: CodeState, root: Path, build_kind: BuildKind) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        workspace = root / state.value
        shutil.copytree(self.base_repository, workspace)
        patch = self._state_patch(state)
        if patch is not None and patch.stat().st_size:
            applied = subprocess.run(
                ["git", "apply", "--whitespace=nowarn", str(patch)],
                cwd=workspace,
                check=False,
                capture_output=True,
                text=True,
            )
            if applied.returncode:
                raise RuntimeError(f"{state.value} patch failed: {applied.stderr}")
        for command in self.repository.build_commands(build_kind):
            built = self._run(command, workspace)
            if built.returncode:
                details = built.stderr.strip() or built.stdout.strip()
                raise RuntimeError(
                    f"{state.value} {build_kind.value} build failed "
                    f"(exit {built.returncode}): {details}"
                )
        return workspace

    def _progress(self, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(message)
        else:
            print(f"PITBENCH_PROGRESS {message}", flush=True)

    def run(
        self,
        cases: list[InstanceCase] | None = None,
        *,
        completed_runs: set[tuple[str, str, CodeState, int, float]] | None = None,
        save_observation: Callable[[RunObservation], None] | None = None,
    ) -> list[RunObservation]:
        observations: list[RunObservation] = []
        with tempfile.TemporaryDirectory(prefix="pitbench-judge-") as temporary:
            root = Path(temporary)
            completed_runs = completed_runs or set()
            if cases is None:
                plan = JudgePlan.from_instance_set_configs(
                    self.task,
                    self.resolver,
                    public_root=self.public_root,
                    generated_root=root / "generated-instance-sets",
                    evaluation_seeds=self.evaluation_seeds,
                )
                cases = plan.cases
            cases = [*cases, *self.additional_cases]
            expected_run_grid = JudgePlan(self.task, cases).expected_run_grid(
                evaluation_seeds=self.evaluation_seeds,
            )
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "expected-run-grid.json").write_text(
                expected_run_grid.model_dump_json(indent=2) + "\n"
            )
            total_solver_runs = sum(
                (
                    case.instance_set.name,
                    case.instance_id,
                    state,
                    seed,
                    budget,
                )
                not in completed_runs
                for case in cases
                for seed in (case.solver_seeds or self.evaluation_seeds)
                for budget in (
                    case.budgets_sec or tuple(self.task.evaluation.budgets_sec)
                )
                for state in self.code_states
            )
            self._progress(
                f"Judge plan: {len(cases)} instances, {total_solver_runs} solver runs"
            )
            if self.run_validation_builds:
                for state in self.code_states:
                    started = time.monotonic()
                    self._progress(f"Judge validation build: {state.value}")
                    self._workspace(state, root / "validation", BuildKind.VALIDATION)
                    self._progress(
                        f"Judge validation build complete: {state.value} in "
                        f"{time.monotonic() - started:.1f}s"
                    )
            workspaces = {}
            for state in self.code_states:
                started = time.monotonic()
                self._progress(f"Judge performance build: {state.value}")
                workspaces[state] = self._workspace(
                    state, root / "performance", BuildKind.PERFORMANCE
                )
                self._progress(
                    f"Judge performance build complete: {state.value} in "
                    f"{time.monotonic() - started:.1f}s"
                )
            jobs = []
            for case in cases:
                if case.path is None:
                    raise RuntimeError("real judge case has no instance path")
                seeds = case.solver_seeds or self.evaluation_seeds
                budgets = case.budgets_sec or tuple(self.task.evaluation.budgets_sec)
                for seed in seeds:
                    for budget in budgets:
                        for state, workspace in workspaces.items():
                            run_identity = (
                                case.instance_set.name,
                                case.instance_id,
                                state,
                                seed,
                                budget,
                            )
                            if run_identity in completed_runs:
                                continue
                            jobs.append((case, state, workspace, seed, budget))

            available = sorted(os.sched_getaffinity(0))
            threads = self.task.evaluation.threads
            possible_slots = len(available) // threads
            workers = min(self.parallel_runs, possible_slots, len(jobs))
            if workers < 1:
                raise ValueError(
                    f"judge requires {threads} CPUs per run but only "
                    f"{len(available)} are available"
                )
            slots: Queue[tuple[int, ...]] = Queue()
            for index in range(workers):
                start = index * threads
                slots.put(tuple(available[start : start + threads]))

            def execute(job):
                cpu_ids = slots.get()
                try:
                    case, state, workspace, seed, budget = job
                    return self._run_case(
                        workspace, case, state, seed, budget, cpu_ids=cpu_ids
                    )
                finally:
                    slots.put(cpu_ids)

            # Start the solver-stage clock after compilation, so UI ETA measures
            # observed solver throughput rather than including build time.
            self._progress(
                f"Judge progress: solver runs 0/{len(jobs)}, workers {workers}"
            )
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_job = {executor.submit(execute, job): job for job in jobs}
                seed_group_totals: dict[tuple[str, str, CodeState, int], int] = {}
                instance_totals: dict[tuple[str, str], int] = {}
                for case, state, _, seed, _ in jobs:
                    instance_key = (case.instance_set.name, case.instance_id)
                    seed_key = (*instance_key, state, seed)
                    seed_group_totals[seed_key] = seed_group_totals.get(seed_key, 0) + 1
                    instance_totals[instance_key] = (
                        instance_totals.get(instance_key, 0) + 1
                    )

                seed_group_counts: dict[tuple[str, str, CodeState, int], int] = {}
                instance_counts: dict[tuple[str, str], int] = {}
                completed_seed_groups = 0
                completed_instances = 0
                valid_count = 0
                for future in as_completed(future_to_job):
                    observation = future.result()
                    observations.append(observation)
                    if save_observation is not None:
                        save_observation(observation)
                    case, state, _, seed, _ = future_to_job[future]
                    instance_key = (case.instance_set.name, case.instance_id)
                    seed_key = (*instance_key, state, seed)
                    seed_group_counts[seed_key] = seed_group_counts.get(seed_key, 0) + 1
                    instance_counts[instance_key] = (
                        instance_counts.get(instance_key, 0) + 1
                    )
                    if seed_group_counts[seed_key] == seed_group_totals[seed_key]:
                        completed_seed_groups += 1
                    if instance_counts[instance_key] == instance_totals[instance_key]:
                        completed_instances += 1
                    valid_count += int(observation.valid)
                    self._progress(
                        f"Judge progress: instances {completed_instances}/"
                        f"{len(instance_totals)}, seed groups {completed_seed_groups}/"
                        f"{len(seed_group_totals)}, solver runs {len(observations)}/"
                        f"{len(jobs)}, valid {valid_count}"
                    )
            observations.sort(
                key=lambda item: (
                    item.instance_set,
                    item.instance_id,
                    item.solver_seed,
                    item.budget_sec,
                    item.code_state.value,
                )
            )
            self._progress(
                f"Judge solver grid complete: {valid_count}/{len(observations)} "
                "valid observations"
            )
        return observations

    def _run_case(
        self,
        workspace: Path,
        case: InstanceCase,
        state: CodeState,
        seed: int,
        budget: float,
        *,
        cpu_ids: tuple[int, ...] | None = None,
    ) -> RunObservation:
        run_dir = (
            self.output_dir / state.value / case.instance_set.name / case.instance_id
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        stem = f"seed-{seed}-budget-{budget:g}"
        output = run_dir / f"{stem}.json"
        trajectory = run_dir / f"{stem}.trajectory.jsonl"
        for artifact in (output, output.with_suffix(".solution.json"), trajectory):
            artifact.unlink(missing_ok=True)
        threads = self.task.evaluation.threads
        command = _limit_solver_cpus(
            self.repository.run_command(
                SolverRunSpec(
                    instance_path=case.path,
                    output_path=output,
                    trajectory_path=trajectory,
                    solver_seed=seed,
                    budget_sec=budget,
                    threads=threads,
                )
            ),
            threads,
            cpu_ids,
        )
        started = time.monotonic()
        try:
            completed = self._run(command, workspace)
        except subprocess.TimeoutExpired as error:
            completed = None
            stdout, stderr = error.stdout or "", error.stderr or ""
        else:
            stdout, stderr = completed.stdout, completed.stderr
        elapsed = time.monotonic() - started
        stdout_path = output.with_suffix(".stdout.log")
        stderr_path = output.with_suffix(".stderr.log")
        for path, content in ((stdout_path, stdout), (stderr_path, stderr)):
            path.write_text(
                content.decode(errors="replace")
                if isinstance(content, bytes)
                else content
            )
        exit_code = completed.returncode if completed is not None else None
        output.with_suffix(".process.json").write_text(
            json.dumps(
                {
                    "argv": command.argv,
                    "returncode": exit_code,
                    "timed_out": completed is None,
                    "elapsed_sec": elapsed,
                },
                indent=2,
            )
            + "\n"
        )

        def failure(status, detail, parsed=None):
            observation = self._failure(case, state, seed, budget, status, str(detail))
            return observation.model_copy(
                update={
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                    "process_exit_code": exit_code,
                    "wall_time_sec": elapsed,
                    "solver_status": parsed.solver_status if parsed else None,
                    "solver_termination": (
                        parsed.solver_termination if parsed else None
                    ),
                    "reported_objective": parsed.objective if parsed else None,
                }
            )

        if completed is None:
            return failure(RunStatus.TIMED_OUT, "external process deadline exceeded")
        parsed = None
        parse_error = None
        try:
            parsed = self.repository.parse_output(output)
        except (ValueError, OSError, TypeError) as error:
            parse_error = error
        if completed.returncode:
            reasons = {
                "timed_out": RunStatus.TIMED_OUT,
                "out_of_memory": RunStatus.OUT_OF_MEMORY,
                "solver_error": RunStatus.SOLVER_ERROR,
            }
            status = reasons.get(
                parsed.failure_reason if parsed else None, RunStatus.CRASHED
            )
            return failure(
                status,
                (parsed.error if parsed else None)
                or stderr
                or f"process exit {exit_code}",
                parsed,
            )
        if parse_error is not None:
            return failure(RunStatus.OUTPUT_ERROR, f"result output: {parse_error}")
        assert parsed is not None
        if parsed.failure_reason:
            status = {
                "timed_out": RunStatus.TIMED_OUT,
                "out_of_memory": RunStatus.OUT_OF_MEMORY,
            }.get(parsed.failure_reason, RunStatus.SOLVER_ERROR)
            return failure(status, parsed.error or parsed.failure_reason, parsed)
        if case.test_suite is not None and not parsed.solver_status:
            return failure(
                RunStatus.OUTPUT_ERROR, "missing solver termination status", parsed
            )
        if parsed.has_solution is False:
            return failure(
                RunStatus.NO_SOLUTION,
                "solver stopped without a candidate solution",
                parsed,
            )
        solution = output.with_suffix(".solution.json")
        try:
            verified = (case.verifier or self.family).verify(case.path, solution)
        except (
            ValueError,
            KeyError,
            IndexError,
            TypeError,
            FileNotFoundError,
        ) as error:
            return failure(RunStatus.OUTPUT_ERROR, f"solution output: {error}", parsed)
        if verified.infrastructure_error:
            return failure(
                RunStatus.INFRASTRUCTURE_ERROR,
                verified.detail or "independent verifier infrastructure error",
                parsed,
            )
        objective = (
            verified.objective if verified.objective is not None else parsed.objective
        )
        if objective is not None and not math.isfinite(objective):
            return failure(RunStatus.INVALID, "non-finite objective", parsed)
        normalized_gap = None
        if case.anchor is not None:
            normalized_gap = self.family.normalized_gap(
                objective,
                case.anchor,
                objective_sense=self.task.oracle.objective_sense,
            )
        return RunObservation(
            task_id=self.task.task_id,
            code_state=state,
            instance_set=case.instance_set.name,
            instance_set_kind=case.instance_set.kind.value,
            instance_id=case.instance_id,
            **_instance_generation_seeds(case.instance_set),
            solver_seed=seed,
            budget_sec=budget,
            threads=self.task.evaluation.threads,
            status=RunStatus.COMPLETED if verified.feasible else RunStatus.INVALID,
            valid=verified.feasible,
            objective=objective,
            reported_objective=parsed.objective,
            optimal_or_bks=case.anchor,
            normalized_gap=normalized_gap,
            primal_bound=parsed.primal_bound,
            dual_bound=parsed.dual_bound,
            wall_time_sec=parsed.wall_time_sec,
            cpu_time_sec=parsed.cpu_time_sec,
            iterations=parsed.iterations,
            nodes=parsed.nodes,
            model_variables=parsed.model_variables,
            model_constraints=parsed.model_constraints,
            peak_rss_bytes=parsed.peak_rss_bytes,
            resource_scope=parsed.resource_scope,
            solver_status=parsed.solver_status,
            solver_termination=parsed.solver_termination,
            test_suite=case.test_suite,
            process_exit_code=exit_code,
            problem_scale=case.problem_scale,
            equivalence_parent_id=case.equivalence_parent_id,
            equivalence_transform=case.equivalence_transform,
            trajectory_path=str(trajectory) if trajectory.exists() else None,
            solution_path=str(solution) if solution.exists() else None,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            error=None if verified.feasible else verified.detail,
        )

    def _failure(
        self,
        case: InstanceCase,
        state: CodeState,
        seed: int,
        budget: float,
        status: RunStatus,
        error: str | None = None,
    ) -> RunObservation:
        return RunObservation(
            task_id=self.task.task_id,
            code_state=state,
            instance_set=case.instance_set.name,
            instance_set_kind=case.instance_set.kind.value,
            instance_id=case.instance_id,
            **_instance_generation_seeds(case.instance_set),
            solver_seed=seed,
            budget_sec=budget,
            threads=self.task.evaluation.threads,
            status=status,
            valid=False,
            test_suite=case.test_suite,
            problem_scale=case.problem_scale,
            equivalence_parent_id=case.equivalence_parent_id,
            equivalence_transform=case.equivalence_transform,
            error=error,
        )
