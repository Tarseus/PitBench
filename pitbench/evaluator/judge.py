from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Callable

import yaml

from pitbench.evaluator.private_assets import PrivateAssetResolver
from pitbench.families.base import ProblemFamilyRegistry
from pitbench.families.external import ExternalVerifierFamily
from pitbench.instances import materialize_generated_instance_set
from pitbench.repositories.base import (
    BuildKind,
    CommandSpec,
    RepositoryPluginRegistry,
    SolverRunSpec,
)
from pitbench.schema.observation import CodeState, RunObservation, RunStatus
from pitbench.schema.task import InstanceSetKind, InstanceSetSpec, PitBenchTask


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

    @classmethod
    def fixture(
        cls, task: PitBenchTask, instances_per_instance_set: int = 2
    ) -> "JudgePlan":
        cases = []
        for instance_set in task.instance_sets:
            if instance_set.kind == InstanceSetKind.AGENT_DEV:
                continue
            for index in range(min(instances_per_instance_set, instance_set.size)):
                objective_scored = (
                    instance_set.kind == InstanceSetKind.JUDGE_ID
                    and not _uses_model_equivalence(task)
                )
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
    def from_private_instance_set_configs(
        cls,
        task: PitBenchTask,
        resolver: PrivateAssetResolver,
        *,
        generated_root: Path | None = None,
    ) -> "JudgePlan":
        cases: list[InstanceCase] = []
        for instance_set in task.instance_sets:
            if instance_set.kind == InstanceSetKind.AGENT_DEV:
                continue
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
                    expected_visibility="judge",
                    stem_prefix=instance_set.name,
                )
                if len(paths) != instance_set.size:
                    raise ValueError(
                        f"instance set {instance_set.name} size does not match config"
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
                    if anchor is not None:
                        instance_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
                        if instance_sha256 != anchor.get("instance_sha256"):
                            raise ValueError(
                                f"instance set {instance_set.name} generated instance "
                                f"hash mismatch for {path.stem}"
                            )
                    cases.append(
                        InstanceCase(
                            instance_set=instance_set,
                            instance_id=path.stem,
                            path=path,
                            anchor=(
                                float(anchor["bks"]) if anchor is not None else None
                            ),
                            problem_scale=_customer_count(path),
                        )
                    )
                continue
            if len(payload["instances"]) != instance_set.size:
                raise ValueError(
                    f"instance set {instance_set.name} size does not match config"
                )
            for item in payload["instances"]:
                uri = item.get("uri", item.get("instance_uri"))
                anchor = item.get("optimal_or_bks", item.get("bks"))
                objective_scored = (
                    instance_set.kind == InstanceSetKind.JUDGE_ID
                    and not _uses_model_equivalence(task)
                )
                if uri is None or (objective_scored and anchor is None):
                    raise ValueError(
                        f"instance set {instance_set.name} has an incomplete instance"
                    )
                path = resolver.resolve(uri)
                cases.append(
                    InstanceCase(
                        instance_set=instance_set,
                        instance_id=item["id"],
                        path=path,
                        anchor=float(anchor) if anchor is not None else None,
                        problem_scale=_customer_count(path),
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
            seeds = case.solver_seeds or tuple(task.evaluation.solver_seeds)
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
            instance_seed=case.instance_set.randomness.instance_seed,
            coordinate_seed=case.instance_set.randomness.coordinate_seed,
            demand_seed=case.instance_set.randomness.demand_seed,
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
        private_root: Path,
        candidate_patch: Path | None,
        output_dir: Path,
        code_states: tuple[CodeState, ...] = tuple(CodeState),
        parallel_runs: int = 1,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.task = task
        self.base_repository = base_repository
        self.resolver = PrivateAssetResolver(private_root)
        self.candidate_patch = candidate_patch
        self.output_dir = output_dir
        self.code_states = code_states
        self.parallel_runs = parallel_runs
        self.progress_callback = progress_callback
        self.repository = RepositoryPluginRegistry.load(task.repository.plugin)
        self.family = ProblemFamilyRegistry.load(task.problem_family)
        if isinstance(self.family, ExternalVerifierFamily):
            self.family.verifier = self.resolver.resolve(task.evaluation.verifier)

    @staticmethod
    def _run(command: CommandSpec, workspace: Path) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.update(command.env)
        return subprocess.run(
            command.argv,
            cwd=workspace / command.cwd,
            env=None if not environment else environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=command.timeout_sec,
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

    def run(self) -> list[RunObservation]:
        observations: list[RunObservation] = []
        with tempfile.TemporaryDirectory(prefix="pitbench-judge-") as temporary:
            root = Path(temporary)
            plan = JudgePlan.from_private_instance_set_configs(
                self.task,
                self.resolver,
                generated_root=root / "generated-instance-sets",
            )
            total_solver_runs = sum(
                len(case.solver_seeds or tuple(self.task.evaluation.solver_seeds))
                * len(case.budgets_sec or tuple(self.task.evaluation.budgets_sec))
                * len(self.code_states)
                for case in plan.cases
            )
            self._progress(
                f"Judge plan: {len(plan.cases)} instances, "
                f"{total_solver_runs} solver runs"
            )
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
            for case in plan.cases:
                if case.path is None:
                    raise RuntimeError("real judge case has no instance path")
                seeds = case.solver_seeds or tuple(self.task.evaluation.solver_seeds)
                budgets = case.budgets_sec or tuple(self.task.evaluation.budgets_sec)
                for seed in seeds:
                    for budget in budgets:
                        for state, workspace in workspaces.items():
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
        try:
            completed = self._run(command, workspace)
        except subprocess.TimeoutExpired:
            return self._failure(case, state, seed, budget, RunStatus.TIMED_OUT)
        if completed.returncode or not output.is_file():
            return self._failure(
                case,
                state,
                seed,
                budget,
                RunStatus.CRASHED,
                completed.stderr,
            )
        parsed = self.repository.parse_output(output)
        solution = output.with_suffix(".solution.json")
        verified = self.family.verify(case.path, solution)
        objective = (
            verified.objective if verified.objective is not None else parsed.objective
        )
        return RunObservation(
            task_id=self.task.task_id,
            code_state=state,
            instance_set=case.instance_set.name,
            instance_set_kind=case.instance_set.kind.value,
            instance_id=case.instance_id,
            instance_seed=case.instance_set.randomness.instance_seed,
            coordinate_seed=case.instance_set.randomness.coordinate_seed,
            demand_seed=case.instance_set.randomness.demand_seed,
            solver_seed=seed,
            budget_sec=budget,
            threads=self.task.evaluation.threads,
            status=RunStatus.COMPLETED if verified.feasible else RunStatus.INVALID,
            valid=verified.feasible,
            objective=objective,
            optimal_or_bks=case.anchor,
            normalized_gap=self.family.normalized_gap(objective, case.anchor),
            primal_bound=parsed.primal_bound,
            dual_bound=parsed.dual_bound,
            wall_time_sec=parsed.wall_time_sec,
            cpu_time_sec=parsed.cpu_time_sec,
            iterations=parsed.iterations,
            nodes=parsed.nodes,
            model_variables=parsed.model_variables,
            model_constraints=parsed.model_constraints,
            peak_rss_bytes=parsed.peak_rss_bytes,
            problem_scale=case.problem_scale,
            equivalence_parent_id=case.equivalence_parent_id,
            equivalence_transform=case.equivalence_transform,
            trajectory_path=str(trajectory) if trajectory.exists() else None,
            solution_path=str(solution) if solution.exists() else None,
            stdout_path=str(output),
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
            instance_seed=case.instance_set.randomness.instance_seed,
            coordinate_seed=case.instance_set.randomness.coordinate_seed,
            demand_seed=case.instance_set.randomness.demand_seed,
            solver_seed=seed,
            budget_sec=budget,
            threads=self.task.evaluation.threads,
            status=status,
            valid=False,
            problem_scale=case.problem_scale,
            equivalence_parent_id=case.equivalence_parent_id,
            equivalence_transform=case.equivalence_transform,
            error=error,
        )
