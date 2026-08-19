from __future__ import annotations

import hashlib
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
                cases.append(
                    InstanceCase(
                        population=population,
                        instance_id=f"{population.name}_{index:04d}",
                        path=None,
                        anchor=1000.0,
                    )
                )
        return cls(task, cases)

    @classmethod
    def from_private_manifests(
        cls, task: PitBenchTask, resolver: PrivateAssetResolver
    ) -> "JudgePlan":
        cases: list[InstanceCase] = []
        for population in task.populations:
            if population.kind == PopulationKind.AGENT_DEV:
                continue
            manifest_path = resolver.resolve(population.manifest)
            payload = yaml.safe_load(manifest_path.read_text())
            if len(payload["instances"]) != population.size:
                raise ValueError(
                    f"population {population.name} size does not match manifest"
                )
            for item in payload["instances"]:
                cases.append(
                    InstanceCase(
                        population=population,
                        instance_id=item["id"],
                        path=resolver.resolve(item["uri"]),
                        anchor=item.get("optimal_or_bks"),
                    )
                )
        return cls(task, cases)


class FixtureJudge:
    """Deterministic contract smoke test; never selected implicitly."""

    _STATE_FACTOR = {
        CodeState.BASE: 1.0,
        CodeState.HUMAN: 0.70,
        CodeState.AGENT: 0.82,
    }

    def run(self, plan: JudgePlan) -> list[RunObservation]:
        observations: list[RunObservation] = []
        task = plan.task
        for case in plan.cases:
            for seed in task.evaluation.solver_seeds:
                for budget in task.evaluation.budgets_sec:
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
        )
        if task.task_type in {TaskType.MODEL_BUILD, TaskType.PRESOLVE}:
            variables = {
                CodeState.BASE: 9011,
                CodeState.HUMAN: 1003,
                CodeState.AGENT: 1200,
            }[state]
            return RunObservation(
                **common,
                model_variables=variables,
                model_constraints=max(1, variables // 2),
                cpu_time_sec=budget * factor,
            )
        anchor = case.anchor or 1000.0
        gap = (0.08 + noise) * factor / max(budget, 1) ** 0.25
        return RunObservation(
            **common,
            objective=anchor * (1 + gap),
            optimal_or_bks=anchor,
            normalized_gap=gap,
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
        if state == CodeState.BASE:
            return None
        if state == CodeState.HUMAN:
            return self.resolver.resolve(
                self.task.references.human_patch.uri,
                self.task.references.human_patch.sha256,
            )
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
                raise RuntimeError(
                    f"{state.value} {build_kind.value} build failed: {built.stderr}"
                )
        return workspace

    def run(self) -> list[RunObservation]:
        plan = JudgePlan.from_private_manifests(self.task, self.resolver)
        observations: list[RunObservation] = []
        with tempfile.TemporaryDirectory(prefix="pitbench-judge-") as temporary:
            root = Path(temporary)
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
                for seed in self.task.evaluation.solver_seeds:
                    for budget in self.task.evaluation.budgets_sec:
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
        command = self.repository.run_command(
            SolverRunSpec(
                instance_path=case.path,
                output_path=output,
                trajectory_path=trajectory,
                solver_seed=seed,
                budget_sec=budget,
                threads=self.task.evaluation.threads,
            )
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
            instance_id=case.instance_id,
            instance_seed=case.population.randomness.instance_seed,
            coordinate_seed=case.population.randomness.coordinate_seed,
            demand_seed=case.population.randomness.demand_seed,
            solver_seed=seed,
            budget_sec=budget,
            threads=self.task.evaluation.threads,
            status=status,
            valid=False,
            error=error,
        )
