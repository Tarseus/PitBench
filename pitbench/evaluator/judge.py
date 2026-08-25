from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml

from pitbench.evaluator.private_assets import PrivateAssetResolver
from pitbench.families.base import ProblemFamilyRegistry
from pitbench.families.external import ExternalVerifierFamily
from pitbench.instances import materialize_generated_population
from pitbench.repositories.base import (
    BuildKind,
    CommandSpec,
    RepositoryPluginRegistry,
    SolverRunSpec,
)
from pitbench.schema.observation import CodeState, RunObservation, RunStatus
from pitbench.schema.task import PitBenchTask, PopulationKind, PopulationSpec, TaskType


@dataclass(frozen=True)
class InstanceCase:
    population: PopulationSpec
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


def _limit_solver_cpus(command: CommandSpec, threads: int) -> CommandSpec:
    """Bind a solver process to its declared CPU count without throttling builds."""

    available = sorted(os.sched_getaffinity(0))
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
        cls, task: PitBenchTask, instances_per_population: int = 2
    ) -> "JudgePlan":
        cases = []
        for population in task.populations:
            if population.kind == PopulationKind.AGENT_DEV:
                continue
            for index in range(min(instances_per_population, population.size)):
                objective_scored = (
                    population.kind == PopulationKind.JUDGE_ID
                    and task.task_type
                    not in {TaskType.MODEL_BUILD, TaskType.PRESOLVE}
                )
                cases.append(
                    InstanceCase(
                        population=population,
                        instance_id=f"{population.name}_{index:04d}",
                        path=None,
                        anchor=1000.0 if objective_scored else None,
                        problem_scale=float(index + 1),
                    )
                )
        return cls(task, cases)

    @classmethod
    def from_private_manifests(
        cls,
        task: PitBenchTask,
        resolver: PrivateAssetResolver,
        *,
        generated_root: Path | None = None,
    ) -> "JudgePlan":
        cases: list[InstanceCase] = []
        for population in task.populations:
            if population.kind == PopulationKind.AGENT_DEV:
                continue
            manifest_path = resolver.resolve(
                population.manifest, population.manifest_sha256
            )
            payload = yaml.safe_load(manifest_path.read_text())
            if "generator" in payload:
                if generated_root is None:
                    raise ValueError(
                        f"population {population.name} requires a generated_root"
                    )
                paths = materialize_generated_population(
                    payload,
                    generated_root / population.name,
                    expected_visibility="judge",
                    stem_prefix=population.name,
                )
                if len(paths) != population.size:
                    raise ValueError(
                        f"population {population.name} size does not match manifest"
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
                                f"population {population.name} has duplicate anchor "
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
                            f"population {population.name} oracle support does not "
                            "match generated instances"
                        )
                for path in paths:
                    anchor = anchors.get(path.stem)
                    if anchor is not None:
                        instance_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
                        if instance_sha256 != anchor.get("instance_sha256"):
                            raise ValueError(
                                f"population {population.name} generated instance "
                                f"hash mismatch for {path.stem}"
                            )
                    cases.append(
                        InstanceCase(
                            population=population,
                            instance_id=path.stem,
                            path=path,
                            anchor=(
                                float(anchor["bks"]) if anchor is not None else None
                            ),
                            problem_scale=_customer_count(path),
                        )
                    )
                continue
            if len(payload["instances"]) != population.size:
                raise ValueError(
                    f"population {population.name} size does not match manifest"
                )
            for item in payload["instances"]:
                uri = item.get("uri", item.get("instance_uri"))
                anchor = item.get("optimal_or_bks", item.get("bks"))
                objective_scored = (
                    population.kind == PopulationKind.JUDGE_ID
                    and task.task_type
                    not in {
                        TaskType.MODEL_BUILD,
                        TaskType.PRESOLVE,
                    }
                )
                if uri is None or (objective_scored and anchor is None):
                    raise ValueError(
                        f"population {population.name} has an incomplete instance"
                    )
                path = resolver.resolve(uri)
                cases.append(
                    InstanceCase(
                        population=population,
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

    def run(self, plan: JudgePlan) -> list[RunObservation]:
        observations: list[RunObservation] = []
        task = plan.task
        for case in plan.cases:
            seeds = case.solver_seeds or tuple(task.evaluation.solver_seeds)
            budgets = case.budgets_sec or tuple(task.evaluation.budgets_sec)
            for seed in seeds:
                for budget in budgets:
                    for state in CodeState:
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
            population=case.population.name,
            population_kind=case.population.kind.value,
            instance_id=case.instance_id,
            instance_seed=case.population.randomness.instance_seed,
            coordinate_seed=case.population.randomness.coordinate_seed,
            demand_seed=case.population.randomness.demand_seed,
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
        if task.task_type in {TaskType.MODEL_BUILD, TaskType.PRESOLVE}:
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
        candidate_patch: Path,
        output_dir: Path,
    ) -> None:
        self.task = task
        self.base_repository = base_repository
        self.resolver = PrivateAssetResolver(private_root)
        self.candidate_patch = candidate_patch
        self.output_dir = output_dir
        self.repository = RepositoryPluginRegistry.load(task.repository.plugin)
        self.family = ProblemFamilyRegistry.load(task.evaluation.family_plugin)
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
        return None if state == CodeState.BASE else self.candidate_patch

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

    def run(self) -> list[RunObservation]:
        observations: list[RunObservation] = []
        with tempfile.TemporaryDirectory(prefix="pitbench-judge-") as temporary:
            root = Path(temporary)
            plan = JudgePlan.from_private_manifests(
                self.task,
                self.resolver,
                generated_root=root / "generated-populations",
            )
            for state in CodeState:
                self._workspace(state, root / "validation", BuildKind.VALIDATION)
            workspaces = {
                state: self._workspace(
                    state, root / "performance", BuildKind.PERFORMANCE
                )
                for state in CodeState
            }
            for case in plan.cases:
                if case.path is None:
                    raise RuntimeError("real judge case has no instance path")
                seeds = case.solver_seeds or tuple(self.task.evaluation.solver_seeds)
                budgets = case.budgets_sec or tuple(self.task.evaluation.budgets_sec)
                for seed in seeds:
                    for budget in budgets:
                        for state, workspace in workspaces.items():
                            observations.append(
                                self._run_case(workspace, case, state, seed, budget)
                            )
        return observations

    def _run_case(
        self,
        workspace: Path,
        case: InstanceCase,
        state: CodeState,
        seed: int,
        budget: float,
    ) -> RunObservation:
        run_dir = (
            self.output_dir / state.value / case.population.name / case.instance_id
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
            population=case.population.name,
            population_kind=case.population.kind.value,
            instance_id=case.instance_id,
            instance_seed=case.population.randomness.instance_seed,
            coordinate_seed=case.population.randomness.coordinate_seed,
            demand_seed=case.population.randomness.demand_seed,
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
            population=case.population.name,
            population_kind=case.population.kind.value,
            instance_id=case.instance_id,
            instance_seed=case.population.randomness.instance_seed,
            coordinate_seed=case.population.randomness.coordinate_seed,
            demand_seed=case.population.randomness.demand_seed,
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
