from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import yaml

from adapters.pitbench.adapter import PitBenchAdapter
from adapters.pitbench.git_snapshot import GitSnapshot
from pitbench.cli.main import (
    _evaluation_metric_failures,
    _limited_evaluation_failures,
    _normal_metric_coverage,
)
from pitbench.evaluator.collection import AnchorCollection
from pitbench.evaluator.docker_judge import DockerJudge, judge_harness_sha256
from pitbench.evaluator.evaluator import (
    PitBenchEvaluator,
    _base_cache_is_complete,
    _base_cache_path,
    _checkpoint_observations_if_complete,
    _default_judge_cpus,
    _default_judge_parallel_runs,
)
from pitbench.evaluator.judge import JudgePlan, LocalProcessJudge
from pitbench.evaluator.private_assets import PrivateAssetResolver
from pitbench.evaluator.reliability import prepare_boundary_cases
from pitbench.evaluator.storage import ObservationStore
from pitbench.evaluator.validity import evaluator_validity
from pitbench.harness.evaluation import EvaluationRequest
from pitbench.instances.generate import (
    balanced_subset_sum_data,
    first_fit_decreasing_packing,
    materialize_generated_instance_set,
    prepare_cp_sat_job_shop_panel,
    prepare_trusted_optimum_oracle,
)
from pitbench.metrics.performance_report import PerformanceClassification
from pitbench.problem_families.base import (
    ProblemFamilyPlugin,
    ProblemFamilyRegistry,
    VerificationResult,
)
from pitbench.problem_families.verification import (
    CPFamily,
    CVRPFamily,
    ExternalVerifierFamily,
    MIPFamily,
    ProcessVerifierFamily,
    TrustedOptimumFamily,
    TrustedOptimumOracle,
    TrustedOptimumRecord,
    expected_mip_variable_names,
)
from pitbench.repositories.base import BuildKind, CommandSpec, RepositoryPlugin
from pitbench.schema.evaluation import ValidityCode
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
    ValidationTestPatch,
)
from pitbench.tasks import TaskCatalog, TaskRecord, TaskValidationError

# Tests consolidated from tests/unit/pitbench/test_judge_failures.py


ROOT = Path(__file__).resolve().parents[3]


def test_native_mip_assignment_uses_existing_independent_checker(
    tmp_path: Path,
) -> None:
    panel = ROOT / "configs/instance_sets/highs_multidimensional_knapsack_v1_15_1_dev"
    instance = panel / "instances/dev_0000.lp"
    reference = json.loads((panel / "bks_solutions/dev_0000.solution.json").read_text())
    names = expected_mip_variable_names(instance)
    values = {name: int(int(name[1:]) in reference["selected_items"]) for name in names}
    solution = tmp_path / "solution.json"
    solution.write_text(json.dumps({"objective": -4542, "values": values}))

    checked = MIPFamily().verify(instance, solution)

    assert checked.feasible is True
    assert checked.objective == -4542


def test_completed_judge_checkpoint_requires_matching_identity_and_hashes(tmp_path):
    observation = RunObservation(
        task_id="task",
        code_state=CodeState.AGENT,
        instance_set="judge_id",
        instance_set_kind="judge_id",
        instance_id="one",
        solver_seed=7,
        budget_sec=5.0,
        status=RunStatus.COMPLETED,
        valid=True,
    )
    observations_path = tmp_path / "judge-observations.jsonl"
    ObservationStore.write_jsonl(observations_path, [observation])
    grid_path = tmp_path / "expected-run-grid.json"
    grid_path.write_text(
        ExpectedRunGrid(
            task_id="task",
            runs=[
                ExpectedRun(
                    task_id="task",
                    code_state=CodeState.AGENT,
                    instance_set="judge_id",
                    instance_set_kind="judge_id",
                    instance_id="one",
                    solver_seed=7,
                    budget_sec=5.0,
                )
            ],
        ).model_dump_json()
    )
    identity = {"candidate_patch_sha256": "first"}
    marker = {
        "identity": identity,
        "observation_count": 1,
        "observations_sha256": hashlib.sha256(
            observations_path.read_bytes()
        ).hexdigest(),
        "expected_grid_sha256": hashlib.sha256(grid_path.read_bytes()).hexdigest(),
    }
    (tmp_path / "judge-checkpoint.json").write_text(json.dumps(marker))

    assert _checkpoint_observations_if_complete(tmp_path, "task", identity) == [
        observation
    ]
    assert (
        _checkpoint_observations_if_complete(
            tmp_path, "task", {"candidate_patch_sha256": "different"}
        )
        is None
    )
    observations_path.write_text(observations_path.read_text() + "\n")
    assert _checkpoint_observations_if_complete(tmp_path, "task", identity) is None


def test_judge_runner_marks_complete_checkpoint_only_after_final_write(
    tmp_path, monkeypatch
):
    from pitbench.evaluator import runner

    observation = RunObservation(
        task_id="vroom_v1_15_0",
        code_state=CodeState.BASE,
        instance_set="judge_id",
        instance_set_kind="judge_id",
        instance_id="one",
        solver_seed=0,
        budget_sec=1.0,
        status=RunStatus.COMPLETED,
        valid=True,
    )

    class FakeJudge:
        def __init__(self, **kwargs):
            self.output_dir = kwargs["output_dir"]

        def run(self):
            self.output_dir.mkdir(parents=True, exist_ok=True)
            grid = ExpectedRunGrid(
                task_id=observation.task_id,
                runs=[
                    ExpectedRun(
                        task_id=observation.task_id,
                        code_state=observation.code_state,
                        instance_set=observation.instance_set,
                        instance_set_kind="judge_id",
                        instance_id=observation.instance_id,
                        solver_seed=observation.solver_seed,
                        budget_sec=observation.budget_sec,
                    )
                ],
            )
            (self.output_dir / "expected-run-grid.json").write_text(
                grid.model_dump_json()
            )
            return [observation]

    monkeypatch.setattr(runner, "LocalProcessJudge", FakeJudge)
    monkeypatch.setattr(runner, "prepare_boundary_cases", lambda *args, **kwargs: [])
    output = tmp_path / "output"
    identity = {"candidate_patch_sha256": "first"}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pitbench.evaluator.runner",
            "--task-config",
            str(ROOT / "configs/tasks/vroom_v1_15_0.yaml"),
            "--base-repository",
            str(tmp_path),
            "--public-root",
            str(ROOT),
            "--private-root",
            str(tmp_path),
            "--candidate-patch",
            str(tmp_path / "candidate.patch"),
            "--output-dir",
            str(output),
            "--observations",
            str(output / "judge-observations.jsonl"),
            "--checkpoint-identity",
            json.dumps(identity),
        ],
    )
    runner.main()
    assert _checkpoint_observations_if_complete(
        output, observation.task_id, identity
    ) == [observation]


def git(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=repository,
        text=True,
    ).strip()


def initialize_repository(path: Path) -> None:
    path.mkdir()
    git(path, "init", "--quiet")
    git(path, "config", "user.name", "PitBench Test")
    git(path, "config", "user.email", "pitbench@example.invalid")


def test_git_snapshot_materializes_and_censors_pinned_submodules(
    tmp_path: Path,
) -> None:
    submodule = tmp_path / "submodule-source"
    initialize_repository(submodule)
    (submodule / "library.txt").write_text("pinned submodule\n")
    git(submodule, "add", "library.txt")
    git(submodule, "commit", "--quiet", "-m", "submodule")
    submodule_commit = git(submodule, "rev-parse", "HEAD")

    repository = tmp_path / "repository-source"
    initialize_repository(repository)
    (repository / ".gitmodules").write_text(
        f'[submodule "library"]\n\tpath = vendor/library\n\turl = {submodule}\n'
    )
    git(repository, "add", ".gitmodules")
    git(
        repository,
        "update-index",
        "--add",
        "--cacheinfo",
        f"160000,{submodule_commit},vendor/library",
    )
    git(repository, "commit", "--quiet", "-m", "repository")
    repository_commit = git(repository, "rev-parse", "HEAD")

    destination = tmp_path / "snapshot"
    GitSnapshot(str(repository), repository_commit).create(destination)

    assert (destination / "vendor/library/library.txt").read_text() == (
        "pinned submodule\n"
    )
    assert git(destination, "remote") == ""
    assert git(destination / "vendor/library", "remote") == ""
    assert git(destination / "vendor/library", "rev-parse", "HEAD") == (
        submodule_commit
    )
    PitBenchAdapter._validate_snapshot(
        destination,
        expected_commit=repository_commit,
    )
    git(destination, "submodule", "update", "--init", "--recursive", "--no-fetch")
    assert (
        git(destination, "submodule", "status", "--recursive")
        .lstrip()
        .startswith(submodule_commit)
    )


def test_primary_budget_must_belong_to_evaluation_budgets() -> None:
    task = PitBenchTask.from_yaml(ROOT / "configs/tasks/pyvrp_v0_14_0.yaml")
    payload = task.model_dump()
    payload["evaluation"]["primary_budget_sec"] = 60

    with pytest.raises(
        ValueError,
        match="primary budget must belong to evaluation budgets",
    ):
        PitBenchTask.model_validate(payload)


def test_current_task_catalog_rejects_multithreaded_solver_runs() -> None:
    task = PitBenchTask.from_yaml(
        ROOT / "configs/tasks/ortools_cp_sat_solve_v9_15.yaml"
    )
    payload = task.model_dump()
    payload["evaluation"]["threads"] = 8

    historical_task = PitBenchTask.model_validate(payload)
    with pytest.raises(
        TaskValidationError, match="current tasks require one solver thread"
    ):
        TaskCatalog(ROOT).validate(
            TaskRecord(
                ROOT / "configs/tasks/ortools_cp_sat_solve_v9_15.yaml", historical_task
            )
        )


def test_catalog_retains_current_releases_and_four_human_references() -> None:
    records = TaskCatalog(ROOT).validate_all()
    task_ids = {record.task.task_id for record in records}
    current_release_ids = {
        "cbc_v2_10_12",
        "choco_graph_coloring_v6_0_1",
        "choco_v6_0_1",
        "highs_multidimensional_knapsack_v1_15_1",
        "highs_v1_15_1",
        "ortools_cp_sat_solve_v9_15",
        "pyvrp_v0_14_0",
        "scip_v9_2_4",
        "vroom_v1_15_0",
    }
    historical_ids = {
        "pyvrp_search_20250717",
        "pyvrp_search_20251020",
        "pyvrp_search_20260728",
        "pyvrp_search_20260806",
    }
    assert len(task_ids) == len(records)
    assert current_release_ids | historical_ids <= task_ids
    releases_by_repository: dict[str, set[str]] = {}
    for record in records:
        if record.task.task_id not in current_release_ids:
            continue
        releases_by_repository.setdefault(record.task.release.repository, set()).add(
            record.task.release.version
        )
    assert all(len(versions) == 1 for versions in releases_by_repository.values())
    current = next(
        record.task for record in records if record.task.task_id == "pyvrp_v0_14_0"
    )
    historical_commits = {
        record.task.release.base_commit
        for record in records
        if record.task.task_id in historical_ids
    }
    assert len(historical_commits) == 4
    assert current.release.base_commit not in historical_commits


@pytest.mark.parametrize(
    ("task_id", "reference_name", "changed_files"),
    [
        ("pyvrp_search_20250717", "pyvrp_pr865", 3),
        ("pyvrp_search_20251020", "pyvrp_pr919", 3),
        ("pyvrp_search_20260728", "pyvrp_pr1146", 15),
        ("pyvrp_search_20260806", "pyvrp_pr1156", 1),
    ],
)
def test_historical_human_reference_is_bound_to_source_only_patch(
    task_id: str, reference_name: str, changed_files: int
) -> None:
    task = PitBenchTask.from_yaml(ROOT / f"configs/tasks/{task_id}.yaml")
    reference = yaml.safe_load(
        (ROOT / f"configs/references/{reference_name}.yaml").read_text()
    )
    patch_path = ROOT / reference["patch_file"]

    assert reference["task_id"] == task.task_id
    assert reference["base_commit"] == task.release.base_commit
    assert (
        hashlib.sha256(patch_path.read_bytes()).hexdigest() == reference["patch_sha256"]
    )
    assert patch_path.read_text().count("diff --git ") == changed_files
    assert all(
        file.startswith("pyvrp/")
        for file in reference.get(
            "changed_source_files", [reference.get("changed_source_file")]
        )
    )
    assert reference_name.removeprefix("pyvrp_") not in task.task_id
    assert reference_name.removeprefix("pyvrp_") not in task.instruction


def test_removed_performance_decision_is_rejected() -> None:
    task = PitBenchTask.from_yaml(ROOT / "configs/tasks/pyvrp_v0_14_0.yaml")
    payload = task.model_dump()
    payload["evaluation"]["decision"] = {"minimum_success_rate_delta": 0}

    with pytest.raises(ValueError, match="evaluation.decision has been removed"):
        PitBenchTask.model_validate(payload)


def test_removed_information_regime_is_rejected() -> None:
    task = PitBenchTask.from_yaml(ROOT / "configs/tasks/pyvrp_v0_14_0.yaml")
    payload = task.model_dump()
    payload["information_regime"] = "snapshot_only"

    with pytest.raises(ValueError, match="information_regime has been removed"):
        PitBenchTask.model_validate(payload)


def test_removed_repository_language_is_rejected() -> None:
    task = PitBenchTask.from_yaml(ROOT / "configs/tasks/pyvrp_v0_14_0.yaml")
    payload = task.model_dump()
    payload["repository"]["language"] = "cpp_python"

    with pytest.raises(ValueError, match="repository.language has been removed"):
        PitBenchTask.model_validate(payload)


@pytest.mark.parametrize("kind", ["solver_certificate", "independent_measurement"])
def test_oracle_rejects_unsupported_kind(kind: str) -> None:
    task = PitBenchTask.from_yaml(ROOT / "configs/tasks/highs_v1_15_1.yaml")
    payload = task.model_dump()
    payload["oracle"]["kind"] = kind

    with pytest.raises(ValueError, match="known_optimum"):
        PitBenchTask.model_validate(payload)


def test_removed_instance_set_shift_is_rejected() -> None:
    task = PitBenchTask.from_yaml(ROOT / "configs/tasks/pyvrp_v0_14_0.yaml")
    payload = task.model_dump()
    payload["instance_sets"][0]["shift"] = "distributional"

    with pytest.raises(ValueError, match="instance_set.shift has been removed"):
        PitBenchTask.model_validate(payload)


def test_workspace_permissions_reuses_existing_agent_account() -> None:
    task = PitBenchTask.from_yaml(ROOT / "configs/tasks/highs_v1_15_1.yaml")

    dockerfile = PitBenchAdapter._workspace_permissions(task, include_runs=False)

    assert "getent group pitbench-agent >/dev/null" in dockerfile
    assert "id --user pitbench-agent >/dev/null 2>&1" in dockerfile
    assert "|| groupadd --non-unique" in dockerfile
    assert "|| useradd --non-unique" in dockerfile
    assert "/workspace/repo/.codex" in dockerfile
    assert "/workspace/repo/.agents" in dockerfile
    assert "/workspace/repo/.aws" in dockerfile


@pytest.mark.parametrize(
    ("task_id", "expected_protocol"),
    [
        ("pyvrp_v0_14_0", PerformanceProtocol.HEURISTIC_FIXED_BUDGET),
        ("vroom_v1_15_0", PerformanceProtocol.HEURISTIC_FIXED_BUDGET),
        ("highs_v1_15_1", PerformanceProtocol.EXACT_VERIFIED_SOLVE),
        ("choco_v6_0_1", PerformanceProtocol.EXACT_VERIFIED_SOLVE),
    ],
)
def test_task_declares_performance_protocol(
    task_id: str,
    expected_protocol: PerformanceProtocol,
) -> None:
    task = PitBenchTask.from_yaml(ROOT / "configs/tasks" / f"{task_id}.yaml")

    assert task.evaluation.performance_protocol == expected_protocol


def test_task_rejects_performance_protocol_that_conflicts_with_task_type() -> None:
    heuristic = PitBenchTask.from_yaml(ROOT / "configs/tasks/pyvrp_v0_14_0.yaml")
    heuristic_payload = heuristic.model_dump()
    heuristic_payload["evaluation"]["performance_protocol"] = "exact_verified_solve"
    with pytest.raises(ValueError, match="heuristic task requires"):
        PitBenchTask.model_validate(heuristic_payload)

    exact = PitBenchTask.from_yaml(ROOT / "configs/tasks/highs_v1_15_1.yaml")
    exact_payload = exact.model_dump()
    exact_payload["evaluation"]["performance_protocol"] = "heuristic_fixed_budget"
    with pytest.raises(ValueError, match="exact task requires"):
        PitBenchTask.model_validate(exact_payload)


def test_fixture_plan_materializes_the_complete_expected_run_grid() -> None:
    task = PitBenchTask.from_yaml(ROOT / "configs/tasks/pyvrp_v0_14_0.yaml")
    grid = JudgePlan.fixture(task, instances_per_instance_set=1).expected_run_grid()
    judge_id_runs = [
        run
        for run in grid.runs
        if run.instance_set_kind == "judge_id"
        and run.equivalence_parent_id is None
        and run.test_suite is None
    ]
    expected_count = (
        len(task.evaluation.seed_robustness.development_seeds)
        * len(task.evaluation.budgets_sec)
        * len(CodeState)
    )

    assert len(judge_id_runs) == expected_count
    assert len(
        {
            (
                run.instance_set,
                run.instance_id,
                run.code_state,
                run.solver_seed,
                run.budget_sec,
            )
            for run in grid.runs
        }
    ) == len(grid.runs)


def test_real_judge_plan_limits_instances_per_instance_set(tmp_path: Path) -> None:
    task = PitBenchTask.from_yaml(ROOT / "configs/tasks/pyvrp_v0_14_0.yaml")
    plan = JudgePlan.from_instance_set_configs(
        task,
        PrivateAssetResolver(ROOT / "private"),
        public_root=ROOT,
        generated_root=tmp_path / "generated",
        evaluation_seeds=(2147483647,),
        instances_per_instance_set=1,
    )

    counts = {}
    for case in plan.cases:
        counts[case.instance_set.name] = counts.get(case.instance_set.name, 0) + 1
    assert counts
    assert all(count == 1 for count in counts.values())


def test_limited_real_judge_plan_preserves_full_seed_and_budget_grid(
    tmp_path: Path,
) -> None:
    task = PitBenchTask.from_yaml(ROOT / "configs/tasks/pyvrp_v0_14_0.yaml")
    evaluation_seeds = tuple(
        yaml.safe_load(
            (ROOT / "private/seed_robustness/pyvrp_v0_14_0.yaml").read_text()
        )["evaluation_seeds"]
    )
    plan = JudgePlan.from_instance_set_configs(
        task,
        PrivateAssetResolver(ROOT / "private"),
        public_root=ROOT,
        generated_root=tmp_path / "generated",
        evaluation_seeds=evaluation_seeds,
        instances_per_instance_set=1,
    )
    grid = plan.expected_run_grid(evaluation_seeds=evaluation_seeds)
    expected_runs_per_instance = len(task.evaluation.budgets_sec) * len(CodeState) * 30

    assert len({(case.instance_set.name, case.instance_id) for case in plan.cases}) == 3
    for instance_set in ("agent_dev", "judge_id", "judge_shift"):
        runs = [run for run in grid.runs if run.instance_set == instance_set]
        assert len(runs) == expected_runs_per_instance


def test_limited_evaluation_preflight_rejects_grid_and_infrastructure_errors(
    tmp_path: Path,
) -> None:
    expected_run = ExpectedRun(
        task_id="dry-run",
        code_state=CodeState.BASE,
        instance_set="judge_id",
        instance_set_kind="judge_id",
        instance_id="instance",
        solver_seed=0,
        budget_sec=10,
    )
    missing_run = expected_run.model_copy(update={"code_state": CodeState.AGENT})
    (tmp_path / "expected-run-grid.json").write_text(
        ExpectedRunGrid(
            task_id="dry-run", runs=[expected_run, missing_run]
        ).model_dump_json()
    )
    ObservationStore.write_jsonl(
        tmp_path / "judge-observations.jsonl",
        [
            RunObservation(
                task_id="dry-run",
                code_state=CodeState.BASE,
                instance_set="judge_id",
                instance_set_kind="judge_id",
                instance_id="instance",
                solver_seed=0,
                budget_sec=10,
                status=RunStatus.INFRASTRUCTURE_ERROR,
                valid=False,
                error="independent verifier is unavailable",
            )
        ],
    )
    task = SimpleNamespace(
        evaluation=SimpleNamespace(
            seed_robustness=None,
            operational_reliability=False,
            representation_robustness=None,
        )
    )

    failures = _limited_evaluation_failures(
        task,
        tmp_path,
        {
            "summary": {
                "performance": {"classification": "incomplete"},
                "resource_usage": {},
            },
            "artifacts": {},
        },
    )

    assert "1 expected solver runs are missing" in failures
    assert any("infrastructure-error" in failure for failure in failures)
    assert "performance report is incomplete" in failures


def test_normal_metric_coverage_requires_complete_declared_reports(tmp_path: Path):
    task_id = "vroom_v1_15_0"
    trial_name = "vroom-trial"
    evaluation_dir = tmp_path / "run" / task_id / trial_name / "evaluation"
    evaluation_dir.mkdir(parents=True)
    expected = ExpectedRun(
        task_id=task_id,
        code_state=CodeState.BASE,
        instance_set="judge_id",
        instance_set_kind="judge_id",
        instance_id="instance",
        solver_seed=0,
        budget_sec=1,
    )
    (evaluation_dir / "expected-run-grid.json").write_text(
        ExpectedRunGrid(task_id=task_id, runs=[expected]).model_dump_json()
    )
    ObservationStore.write(
        evaluation_dir / "trials.parquet",
        [
            RunObservation(
                task_id=task_id,
                code_state=CodeState.BASE,
                instance_set="judge_id",
                instance_set_kind="judge_id",
                instance_id="instance",
                solver_seed=0,
                budget_sec=1,
                status=RunStatus.COMPLETED,
                valid=True,
            )
        ],
    )
    payload = {
        "artifacts": {
            "observations": {"path": "trials.parquet"},
            "expected_run_grid": {"path": "expected-run-grid.json"},
        },
        "summary": {
            "performance": {"classification": "inconclusive"},
            "resource_usage": {},
            "operational_reliability": {"complete": True},
        },
    }
    (tmp_path / "run" / "results.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "task_id": task_id,
                        "trial_name": trial_name,
                        "evaluation": {"completed": True, "payload": payload},
                    }
                ]
            }
        )
    )

    coverage = _normal_metric_coverage(ROOT, tmp_path, "run", [task_id], None)

    assert coverage["complete"]
    assert _evaluation_metric_failures(
        PitBenchTask.from_yaml(ROOT / "configs/tasks/vroom_v1_15_0.yaml"),
        evaluation_dir,
        {**payload, "summary": {**payload["summary"], "resource_usage": None}},
    ) == ["resource report is missing"]


def test_normal_metric_coverage_retains_configuration_collection_failure(
    tmp_path: Path,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "results.json").write_text('{"results": []}')

    coverage = _normal_metric_coverage(
        ROOT,
        tmp_path,
        "run",
        [],
        run_dir / "configuration-robustness",
        "configuration collection failed: collector image unavailable",
    )

    assert coverage["complete"] is False
    assert (
        "configuration collection failed: collector image unavailable"
        in coverage["failures"][-1]
    )
    assert coverage["configuration_robustness"]["output"] == str(
        run_dir / "configuration-robustness"
    )


def test_problem_family_plugins_register_and_reject_duplicate_names():
    assert isinstance(ProblemFamilyRegistry.load("cvrp"), CVRPFamily)
    with pytest.raises(ValueError, match="duplicate problem family plugin"):

        class DuplicateFamily(CVRPFamily):
            name = "cvrp"


@pytest.mark.parametrize(
    ("objective_sense", "anchor", "objective", "expected_gap"),
    [
        ("minimize", 100.0, 110.0, 0.1),
        ("minimize", 100.0, 90.0, -0.1),
        ("maximize", 100.0, 90.0, 0.1),
        ("maximize", 100.0, 110.0, -0.1),
        ("minimize", -100.0, -90.0, 0.1),
        ("minimize", -100.0, -110.0, -0.1),
        ("maximize", -100.0, -110.0, 0.1),
        ("maximize", -100.0, -90.0, -0.1),
    ],
)
def test_normalized_gap_respects_objective_sense_and_anchor_sign(
    objective_sense: str,
    anchor: float,
    objective: float,
    expected_gap: float,
) -> None:
    assert ProblemFamilyPlugin.normalized_gap(
        objective,
        anchor,
        objective_sense=objective_sense,
    ) == pytest.approx(expected_gap)


@pytest.mark.parametrize("anchor", [math.inf, -math.inf, math.nan])
def test_normalized_gap_rejects_invalid_anchor(anchor: float) -> None:
    with pytest.raises(
        ValueError,
        match="normalized gap requires a finite anchor",
    ):
        ProblemFamilyPlugin.normalized_gap(
            1.0,
            anchor,
            objective_sense="minimize",
        )


def test_normalized_gap_uses_unit_scale_for_zero_optimum() -> None:
    assert ProblemFamilyPlugin.normalized_gap(
        2.0, 0.0, objective_sense="minimize"
    ) == 2.0
    assert ProblemFamilyPlugin.normalized_gap(
        -2.0, 0.0, objective_sense="maximize"
    ) == 2.0
    assert ProblemFamilyPlugin.normalized_gap(
        0.0, 0.0, objective_sense="minimize"
    ) == 0.0


@pytest.mark.parametrize("objective", [None, math.inf, -math.inf, math.nan])
def test_normalized_gap_omits_missing_or_nonfinite_objective(
    objective: float | None,
) -> None:
    assert (
        ProblemFamilyPlugin.normalized_gap(
            objective,
            100.0,
            objective_sense="minimize",
        )
        is None
    )


def test_normalized_gap_requires_explicit_objective_sense() -> None:
    with pytest.raises(ValueError, match="normalized gap requires an objective sense"):
        ProblemFamilyPlugin.normalized_gap(
            110.0,
            100.0,
            objective_sense=None,
        )


def test_missing_external_verifier_is_an_infrastructure_error(tmp_path: Path) -> None:
    result = ExternalVerifierFamily().verify(
        tmp_path / "instance",
        tmp_path / "solution",
    )

    assert result.feasible is False
    assert result.infrastructure_error is True


def prepare_trusted_verifier(
    tmp_path: Path,
    *,
    task_id: str,
    generator: dict,
) -> tuple[Path, TrustedOptimumFamily]:
    private_root = tmp_path / "private"
    private_root.mkdir()
    instance_set_config = tmp_path / "instance-set.yaml"
    instance_set_config.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "visibility": "judge",
                "format": "lp"
                if generator["kind"] == "single_machine_scheduling_mip"
                else "normalized_bin_packing_json",
                "generator": generator,
            },
            sort_keys=False,
        )
    )
    oracle_path = private_root / "oracle.yaml"
    references = private_root / "references"
    payload = prepare_trusted_optimum_oracle(
        task_config_path=ROOT / "configs/tasks" / f"{task_id}.yaml",
        instance_set_name="judge_id",
        instance_set_config_path=instance_set_config,
        private_root=private_root,
        output_path=oracle_path,
        reference_solution_dir=references,
    )
    oracle = TrustedOptimumOracle.model_validate(payload)
    record = oracle.records[0]
    reference_path = private_root / record.reference_solution_uri.removeprefix(
        "private://"
    )
    instance_path = materialize_generated_instance_set(
        yaml.safe_load(instance_set_config.read_text()),
        tmp_path / "instances",
        expected_visibility="judge",
        stem_prefix="judge_id",
    )[0]
    return instance_path, TrustedOptimumFamily(record, reference_path)


def test_scheduling_trusted_optimum_verifies_integer_model_and_objective(
    tmp_path: Path,
) -> None:
    instance_path, verifier = prepare_trusted_verifier(
        tmp_path,
        task_id="highs_v1_15_1",
        generator={
            "kind": "single_machine_scheduling_mip",
            "count": 1,
            "jobs": [4],
            "randomness": {"instance_seed": 17},
        },
    )
    processing_times = verifier.record.optimality_basis["processing_times"]
    reference = json.loads(verifier.reference_solution_path.read_text())
    starts = reference["start_times"]
    values = {f"s{index}": start for index, start in enumerate(starts)}
    for first in range(len(starts)):
        for second in range(first + 1, len(starts)):
            values[f"z{first}_{second}"] = int(
                starts[first] + processing_times[first] <= starts[second]
            )
    objective = sum(starts)
    raw_solution = "\n".join(
        [
            f"Objective {objective}",
            f"# Columns {len(values)}",
            *[f"{name} {value}" for name, value in values.items()],
            "# Rows 0",
        ]
    )
    solution_path = tmp_path / "scheduling.solution.json"
    solution_path.write_text(json.dumps({"raw_solution": raw_solution}))

    result = verifier.verify(instance_path, solution_path)

    assert result.feasible is True
    assert result.objective == objective == verifier.record.optimal_objective
    assert "General" in instance_path.read_text()

    positive_index = next(index for index, start in enumerate(starts) if start > 0)
    positive_start = starts[positive_index]
    near_integer = raw_solution.replace(
        f"Objective {objective}", f"Objective {objective}.0000000000002"
    ).replace(
        f"\ns{positive_index} {positive_start}\n",
        f"\ns{positive_index} {positive_start}.0000000000002\n",
    )
    solution_path.write_text(json.dumps({"raw_solution": near_integer}))
    near_integer_result = verifier.verify(instance_path, solution_path)
    assert near_integer_result.feasible is True
    assert near_integer_result.objective == objective

    outside_tolerance = raw_solution.replace(
        f"\ns{positive_index} {positive_start}\n",
        f"\ns{positive_index} {positive_start}.01\n",
    )
    solution_path.write_text(json.dumps({"raw_solution": outside_tolerance}))
    assert verifier.verify(instance_path, solution_path).feasible is False

    overlapping = raw_solution.replace(
        next(f"s{index} {start}" for index, start in enumerate(starts) if start > 0),
        next(f"s{index} 0" for index, start in enumerate(starts) if start > 0),
    )
    solution_path.write_text(json.dumps({"raw_solution": overlapping}))
    assert verifier.verify(instance_path, solution_path).feasible is False


def test_bin_packing_trusted_optimum_verifies_capacity_certificate(
    tmp_path: Path,
) -> None:
    instance_path, verifier = prepare_trusted_verifier(
        tmp_path,
        task_id="choco_v6_0_1",
        generator={
            "kind": "bin_packing",
            "count": 1,
            "items": [50],
            "capacity": 100,
            "weight_distribution": "uniform_integer",
            "selected_instance_seeds": [60118],
            "randomness": {"instance_seed": 60110},
        },
    )
    result = verifier.verify(instance_path, verifier.reference_solution_path)

    assert result.feasible is True
    assert result.objective == verifier.record.optimal_objective
    assert result.objective == verifier.record.optimality_basis["lower_bound"]

    weights = json.loads(instance_path.read_text())["weights"]
    invalid_solution = tmp_path / "invalid-bin-packing.solution.json"
    invalid_solution.write_text(json.dumps({"bins": [list(range(len(weights)))]}))
    assert verifier.verify(instance_path, invalid_solution).feasible is False

    instance_path.write_text(instance_path.read_text() + "\n")
    hash_mismatch = verifier.verify(instance_path, verifier.reference_solution_path)
    assert hash_mismatch.feasible is False
    assert hash_mismatch.infrastructure_error is True


def test_bin_packing_trusted_optimum_accepts_hash_bound_external_proof(
    tmp_path: Path,
) -> None:
    private_root = tmp_path / "private"
    private_root.mkdir()
    config_path = tmp_path / "instance-set.yaml"
    config = {
        "schema_version": "1.0",
        "visibility": "judge",
        "format": "normalized_bin_packing_json",
        "generator": {
            "kind": "bin_packing",
            "count": 1,
            "items": [50],
            "capacity": 100,
            "weight_distribution": "uniform_integer",
            "selected_instance_seeds": [60118],
            "randomness": {"instance_seed": 60110},
        },
    }
    config_path.write_text(yaml.safe_dump(config))
    instance_path = materialize_generated_instance_set(
        config,
        tmp_path / "instances",
        expected_visibility="judge",
        stem_prefix="judge_id",
    )[0]
    instance = json.loads(instance_path.read_text())
    bins = first_fit_decreasing_packing(instance["weights"], instance["capacity"])
    proof_commit = "a" * 40
    proof = {
        "instance_sha256": hashlib.sha256(instance_path.read_bytes()).hexdigest(),
        "solver_commit": proof_commit,
        "adapter_source_sha256": "b" * 64,
        "solver_threads": 1,
        "solver_seed": 0,
        "budget_sec": 10,
        "status": "OPTIMAL",
        "objective": len(bins),
        "best_bound": float(len(bins)),
        "bins": bins,
    }
    proof_dir = private_root / "proofs"
    proof_dir.mkdir()
    (proof_dir / "judge_id_0000.proof.json").write_text(json.dumps(proof))
    payload = prepare_trusted_optimum_oracle(
        task_config_path=ROOT / "configs/tasks/choco_v6_0_1.yaml",
        instance_set_name="judge_id",
        instance_set_config_path=config_path,
        private_root=private_root,
        output_path=private_root / "oracle.yaml",
        reference_solution_dir=private_root / "references",
        bin_packing_proof_dir=proof_dir,
        proof_solver_commit=proof_commit,
    )
    record = TrustedOptimumOracle.model_validate(payload).records[0]
    reference_path = private_root / record.reference_solution_uri.removeprefix(
        "private://"
    )
    verifier = TrustedOptimumFamily(record, reference_path)
    assert verifier.verify(instance_path, reference_path).feasible is True
    assert record.optimality_basis["kind"] == "independent_solver_optimal"

    altered_proof = {**proof, "instance_sha256": "0" * 64}
    altered_record = record.model_copy(
        update={
            "optimality_basis": {
                **record.optimality_basis,
                "proof": altered_proof,
            }
        }
    )
    invalid = TrustedOptimumFamily(altered_record, reference_path).verify(
        instance_path, reference_path
    )
    assert invalid.feasible is False
    assert invalid.infrastructure_error is True


def test_balanced_subset_sum_oracle_matches_exhaustive_optimum(tmp_path: Path) -> None:
    config = {
        "randomness": {"instance_seed": 71},
        "item_counts": [8],
        "max_weight": 12,
        "cardinality": 4,
        "domain_modes": ["enumerated"],
        "target_offsets": [1],
    }
    instance, selected, basis = balanced_subset_sum_data(config, 0)
    optimum = min(
        abs(sum(instance["weights"][item] for item in subset) - instance["target"])
        for subset in itertools.combinations(range(8), 4)
    )
    instance_path = tmp_path / "instance.json"
    solution_path = tmp_path / "solution.json"
    instance_path.write_text(json.dumps(instance))
    solution_path.write_text(json.dumps({"selected_items": selected}))
    record = TrustedOptimumRecord(
        instance_set="judge_id",
        instance_id="case-1",
        instance_sha256=hashlib.sha256(instance_path.read_bytes()).hexdigest(),
        problem_kind="balanced_subset_sum",
        objective_sense="minimize",
        optimal_objective=optimum,
        reference_solution_uri="private://reference.json",
        reference_solution_sha256=hashlib.sha256(
            solution_path.read_bytes()
        ).hexdigest(),
        optimality_basis=basis,
        generation_provenance={},
        verification_provenance=(
            "pitbench.problem_families.verification:TrustedOptimumFamily"
        ),
    )

    assert TrustedOptimumFamily(record, solution_path).verify(
        instance_path, solution_path
    ).objective == optimum
    assert CPFamily().verify(instance_path, solution_path).objective == optimum
    invalid = record.model_copy(
        update={"optimality_basis": {**basis, "reachable_bitset_sha256": "0" * 64}}
    )
    assert TrustedOptimumFamily(invalid, solution_path).verify(
        instance_path, solution_path
    ).infrastructure_error


def test_cp_family_verifies_public_normalized_bin_packing_without_private_verifier(
    tmp_path: Path,
) -> None:
    instance_path = tmp_path / "instance.json"
    instance_path.write_text(json.dumps({"weights": [6, 4, 8], "capacity": 10}))
    solution_path = tmp_path / "solution.json"
    solution_path.write_text(json.dumps({"bins": [[0, 1], [2]]}))

    verified = CPFamily().verify(instance_path, solution_path)

    assert verified.feasible is True
    assert verified.objective == 2
    assert verified.infrastructure_error is False

    solution_path.write_text(json.dumps({"bins": [[0, 2], [1]]}))
    invalid = CPFamily().verify(instance_path, solution_path)
    assert invalid.feasible is False
    assert invalid.infrastructure_error is False

    model_path = tmp_path / "model.pb"
    model_path.write_bytes(b"fixed-cp-model-proto")
    unsupported = CPFamily().verify(model_path, solution_path)
    assert unsupported.feasible is False
    assert unsupported.infrastructure_error is True


def test_mip_family_verifies_public_scheduling_lp_without_private_verifier(
    tmp_path: Path,
) -> None:
    instance_path = tmp_path / "instance.lp"
    instance_path.write_text(
        "Minimize\n"
        " obj: s0 + s1\n"
        "Subject To\n"
        " c0: s0 - s1 + 7 z0_1 <= 4\n"
        " c1: s1 - s0 - 7 z0_1 <= -4\n"
        "Bounds\n"
        " 0 <= s0 <= 7\n"
        " 0 <= s1 <= 7\n"
        "General\n s0\n s1\nBinary\n z0_1\nEnd\n"
    )
    solution_path = tmp_path / "solution.json"
    solution_path.write_text(
        json.dumps(
            {
                "raw_solution": "\n".join(
                    [
                        "Objective 3",
                        "# Columns",
                        "s0 0",
                        "s1 3",
                        "z0_1 1",
                        "# End",
                    ]
                )
            }
        )
    )

    result = MIPFamily().verify(instance_path, solution_path)

    assert result.feasible is True
    assert result.objective == 3
    assert result.infrastructure_error is False


def test_job_shop_trusted_optimum_verifies_start_variables_and_makespan(
    tmp_path: Path,
) -> None:
    instance_path = tmp_path / "instance.pb"
    instance_path.write_bytes(b"fixed-cp-model-proto")
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps({"start_times": [0, 3, 0, 3]}))
    record = TrustedOptimumRecord(
        instance_set="judge_id",
        instance_id="la01",
        instance_sha256=hashlib.sha256(instance_path.read_bytes()).hexdigest(),
        problem_kind="job_shop_scheduling",
        objective_sense="minimize",
        optimal_objective=5,
        reference_solution_uri="private://references/la01.json",
        reference_solution_sha256=hashlib.sha256(
            reference_path.read_bytes()
        ).hexdigest(),
        optimality_basis={
            "kind": "published_job_shop_optimum",
            "jobs": [
                [[0, 3], [1, 2]],
                [[1, 2], [0, 1]],
            ],
            "start_variable_indices": [0, 3, 6, 8],
        },
        generation_provenance={"source": "test"},
        verification_provenance=(
            "pitbench.problem_families.verification:TrustedOptimumFamily"
        ),
    )
    verifier = TrustedOptimumFamily(record, reference_path)
    solution_path = tmp_path / "solution.json"
    solution_path.write_text(json.dumps({"values": [0, 0, 0, 3, 0, 0, 0, 0, 3]}))

    verified = verifier.verify(instance_path, solution_path)

    assert verified.feasible is True
    assert verified.objective == 5
    solution_path.write_text(json.dumps({"values": [0, 0, 0, 1, 0, 0, 2, 0, 0]}))
    assert verifier.verify(instance_path, solution_path).feasible is False


def test_cp_sat_job_shop_panel_rejects_missing_independent_schedule(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="independent reference schedules"):
        prepare_cp_sat_job_shop_panel(
            task_id="test",
            instance_set_name="agent_dev",
            visibility="agent",
            source_root=tmp_path / "source",
            instance_ids=["la01"],
            model_preparer_image="sha256:model-preparer",
            instance_output_dir=tmp_path / "instances",
            reference_solution_dir=None,
            instance_set_config_path=tmp_path / "instance-set.yaml",
        )


def test_cp_sat_job_shop_panel_prepares_fixed_proto_in_isolated_container(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "jsplib"
    instances = source_root / "instances"
    instances.mkdir(parents=True)
    (instances / "la01").write_text("1 1\n0 2\n")
    (source_root / "instances.json").write_text(
        json.dumps([{"name": "la01", "path": "instances/la01", "optimum": 2}])
    )
    reference_source = tmp_path / "source-schedules"
    reference_source.mkdir()
    (reference_source / "la01.solution.json").write_text(
        json.dumps({"start_times": [0]})
    )
    instance_output = tmp_path / "generated" / "instances"
    reference_output = tmp_path / "generated" / "references"
    instance_set_config = tmp_path / "generated" / "instance-set.yaml"

    def prepare_model(command: list[str], **_: object) -> subprocess.CompletedProcess:
        (instance_output / "la01.pb").write_bytes(b"fixed-proto")
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=json.dumps({"start_variable_indices": [0]}),
            stderr="",
        )

    with patch(
        "pitbench.instances.generate.subprocess.run", side_effect=prepare_model
    ) as run:
        panel = prepare_cp_sat_job_shop_panel(
            task_id="agent-development",
            instance_set_name="agent_dev",
            visibility="agent",
            source_root=source_root,
            instance_ids=["la01"],
            model_preparer_image="sha256:model-preparer",
            instance_output_dir=instance_output,
            reference_solution_dir=reference_output,
            instance_set_config_path=instance_set_config,
            reference_solution_source_dirs=[reference_source],
        )

    command = run.call_args.args[0]
    assert command[:5] == ["docker", "run", "--rm", "--network", "none"]
    assert "sha256:model-preparer" in command
    assert command[command.index("--solver") + 1] == "ortools_cp_sat"
    assert command[command.index("--solver") - 1] == "prepare-jssp"
    assert panel["instances"][0]["bks"] == 2
    assert json.loads((reference_output / "la01.solution.json").read_text()) == {
        "start_times": [0]
    }


def test_cp_sat_judge_panel_uses_hash_bound_published_bks_without_schedule(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "jsplib"
    instances = source_root / "instances"
    instances.mkdir(parents=True)
    (instances / "la21").write_text("1 1\n0 2\n")
    source_index = source_root / "instances.json"
    source_index.write_text(
        json.dumps([{"name": "la21", "path": "instances/la21", "optimum": 2}])
    )
    private_root = tmp_path / "private"
    instance_output = private_root / "judge_id" / "instances"
    instance_set_config = private_root / "judge_id.yaml"
    oracle_path = private_root / "oracle.yaml"

    def prepare_model(command: list[str], **_: object) -> subprocess.CompletedProcess:
        (instance_output / "la21.pb").write_bytes(b"fixed-proto")
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=json.dumps({"start_variable_indices": [0]}),
            stderr="",
        )

    with patch("pitbench.instances.generate.subprocess.run", side_effect=prepare_model):
        payload = prepare_cp_sat_job_shop_panel(
            task_id="cp-sat",
            instance_set_name="judge_id",
            visibility="judge",
            source_root=source_root,
            instance_ids=["la21"],
            model_preparer_image="sha256:model-preparer",
            instance_output_dir=instance_output,
            reference_solution_dir=None,
            instance_set_config_path=instance_set_config,
            private_root=private_root,
            oracle_output_path=oracle_path,
        )

    record = payload["records"][0]
    assert record["target_kind"] == "published_bks"
    assert "reference_solution_uri" not in record
    assert record["generation_provenance"]["published_bks_source"] == {
        "repository": "https://github.com/tamy0612/JSPLIB",
        "artifact_path": "instances.json",
        "artifact_sha256": hashlib.sha256(source_index.read_bytes()).hexdigest(),
        "instance_path": "instances/la21",
    }
    assert (
        "reference_solution_uri"
        not in yaml.safe_load(oracle_path.read_text())["records"][0]
    )


def test_job_shop_published_bks_verifies_candidate_without_reference_schedule(
    tmp_path: Path,
) -> None:
    instance_path = tmp_path / "instance.pb"
    instance_path.write_bytes(b"fixed-cp-model-proto")
    record = TrustedOptimumRecord(
        instance_set="judge_id",
        instance_id="la21",
        instance_sha256=hashlib.sha256(instance_path.read_bytes()).hexdigest(),
        target_kind="published_bks",
        problem_kind="job_shop_scheduling",
        objective_sense="minimize",
        optimal_objective=5,
        optimality_basis={
            "kind": "published_job_shop_bks",
            "jobs": [
                [[0, 3], [1, 2]],
                [[1, 2], [0, 1]],
            ],
            "start_variable_indices": [0, 3, 6, 8],
        },
        generation_provenance={
            "published_bks_source": {
                "repository": "https://example.invalid/jsplib",
                "artifact_path": "instances.json",
                "artifact_sha256": "0" * 64,
                "instance_path": "instances/la21",
            }
        },
        verification_provenance=(
            "pitbench.problem_families.verification:TrustedOptimumFamily"
        ),
    )
    solution_path = tmp_path / "solution.json"
    solution_path.write_text(json.dumps({"values": [0, 0, 0, 3, 0, 0, 0, 0, 3]}))

    verified = TrustedOptimumFamily(record, None).verify(instance_path, solution_path)

    assert verified.feasible is True
    assert verified.objective == 5


def test_static_exact_judge_case_uses_published_bks_verifier(
    tmp_path: Path,
) -> None:
    private_root = tmp_path / "private"
    instance_path = private_root / "instances" / "la21.pb"
    instance_path.parent.mkdir(parents=True)
    instance_path.write_bytes(b"fixed-cp-model-proto")
    record = TrustedOptimumRecord(
        instance_set="judge_id",
        instance_id="la21",
        instance_sha256=hashlib.sha256(instance_path.read_bytes()).hexdigest(),
        target_kind="published_bks",
        problem_kind="job_shop_scheduling",
        objective_sense="minimize",
        optimal_objective=5,
        optimality_basis={
            "kind": "published_job_shop_bks",
            "jobs": [[[0, 5]]],
            "start_variable_indices": [0],
        },
        generation_provenance={
            "published_bks_source": {
                "repository": "https://example.invalid/jsplib",
                "artifact_path": "instances.json",
                "artifact_sha256": "0" * 64,
                "instance_path": "instances/la21",
            }
        },
        verification_provenance=(
            "pitbench.problem_families.verification:TrustedOptimumFamily"
        ),
    )
    oracle_path = private_root / "oracle.yaml"
    oracle_path.write_text(
        yaml.safe_dump(
            TrustedOptimumOracle(task_id="published-bks", records=[record]).model_dump(
                mode="json", exclude_none=True
            ),
            sort_keys=False,
        )
    )
    judge_config = private_root / "judge.yaml"
    judge_config.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "visibility": "judge",
                "format": "cp_sat_model_proto",
                "instances": [
                    {
                        "id": "la21",
                        "uri": "private://instances/la21.pb",
                        "optimal_or_bks": 5,
                    }
                ],
            },
            sort_keys=False,
        )
    )
    task = PitBenchTask.from_yaml(ROOT / "configs/tasks/choco_v6_0_1.yaml")
    task.task_id = "published-bks"
    task.oracle.kind = "best_known_solution"
    task.oracle.source = "private://oracle.yaml"
    task.oracle.source_sha256 = hashlib.sha256(oracle_path.read_bytes()).hexdigest()
    task.evaluation.verifier = "exact_target"
    task.evaluation.seed_robustness = None
    task.evaluation.solver_seeds = [0]
    task.instance_sets = [
        InstanceSetSpec(
            name="agent_dev",
            kind=InstanceSetKind.AGENT_DEV,
            instance_set_config="unused.yaml",
            size=1,
        ),
        InstanceSetSpec(
            name="judge_id",
            kind=InstanceSetKind.JUDGE_ID,
            instance_set_config="private://judge.yaml",
            size=1,
        ),
    ]

    plan = JudgePlan.from_instance_set_configs(
        task,
        PrivateAssetResolver(private_root),
        public_root=tmp_path,
    )

    assert len(plan.cases) == 1
    assert isinstance(plan.cases[0].verifier, TrustedOptimumFamily)
    assert plan.cases[0].anchor == 5


class ScriptRepository(RepositoryPlugin):
    name = "controlled_process"

    def __init__(self, script, timeout=5):
        self.script, self.timeout = script, timeout

    def build_commands(self, kind):
        return []

    def run_command(self, run):
        return CommandSpec(
            argv=[
                sys.executable,
                "-c",
                "from pathlib import Path\nimport json,sys\np=Path(sys.argv[1])\n"
                + self.script,
                str(run.output_path),
            ],
            timeout_sec=self.timeout,
        )


def test_evaluator_only_test_patch_applies_to_both_validation_states(tmp_path):
    task = PitBenchTask.from_yaml(ROOT / "configs/tasks/pyvrp_v0_14_0.yaml")
    task.evaluation.seed_robustness = None
    task.evaluation.solver_seeds = [0]
    base = tmp_path / "source"
    (base / "tests").mkdir(parents=True)
    (base / "tests" / "expectation.txt").write_text("budget=3\n")
    (base / "solver.txt").write_text("base\n")
    private = tmp_path / "private"
    (private / "validation_tests").mkdir(parents=True)
    test_patch = private / "validation_tests" / "task.patch"
    test_patch.write_text(
        "diff --git a/tests/expectation.txt b/tests/expectation.txt\n"
        "--- a/tests/expectation.txt\n"
        "+++ b/tests/expectation.txt\n"
        "@@ -1 +1 @@\n-budget=3\n+budget=5\n"
    )
    task.evaluation.validation_test_patch = ValidationTestPatch(
        source="private://validation_tests/task.patch",
        source_sha256=hashlib.sha256(test_patch.read_bytes()).hexdigest(),
    )
    candidate = tmp_path / "candidate.patch"
    candidate.write_text(
        "diff --git a/solver.txt b/solver.txt\n"
        "--- a/solver.txt\n"
        "+++ b/solver.txt\n"
        "@@ -1 +1 @@\n-base\n+agent\n"
    )
    judge = LocalProcessJudge(
        task,
        base,
        tmp_path,
        private,
        candidate,
        tmp_path / "output",
        evaluation_seeds=(0,),
    )
    judge.repository = ScriptRepository("pass")

    for state in (CodeState.BASE, CodeState.AGENT):
        workspace = judge._workspace(
            state, tmp_path / state.value, BuildKind.VALIDATION
        )
        assert (workspace / "tests" / "expectation.txt").read_text() == "budget=5\n"
    performance = judge._workspace(
        CodeState.AGENT, tmp_path / "performance", BuildKind.PERFORMANCE
    )
    assert (performance / "tests" / "expectation.txt").read_text() == "budget=3\n"
    assert (performance / "solver.txt").read_text() == "agent\n"


@pytest.fixture
def judge_case(tmp_path):
    task = PitBenchTask.from_yaml(ROOT / "configs/tasks/pyvrp_v0_14_0.yaml")
    task.evaluation.seed_robustness = None
    task.evaluation.solver_seeds = [0]
    task.evaluation.representation_robustness = None
    cases = prepare_boundary_cases(task, tmp_path / "out")
    judge = LocalProcessJudge(
        task,
        tmp_path,
        ROOT,
        tmp_path / "absent",
        None,
        tmp_path / "out",
        evaluation_seeds=(0,),
        family=cases[0].verifier,
        code_states=(CodeState.BASE,),
        run_validation_builds=False,
        progress_callback=lambda message: None,
    )
    return judge, cases[0]


GOOD = "p.write_text(json.dumps({'valid': True, 'has_solution': True, 'objective': 10, 'solver_status': 'Time limit reached'}))\np.with_suffix('.solution.json').write_text(json.dumps({'routes': [[1]]}))\nprint('native stdout')\nprint('native stderr', file=sys.stderr)\n"


@pytest.mark.parametrize(
    "script,expected",
    [
        (GOOD, RunStatus.COMPLETED),
        ("sys.exit(7)", RunStatus.CRASHED),
        ("pass", RunStatus.OUTPUT_ERROR),
        ("p.write_text('{bad json')", RunStatus.OUTPUT_ERROR),
        ("p.write_text(json.dumps({'valid': True}))", RunStatus.OUTPUT_ERROR),
        (
            "p.write_text(json.dumps({'valid': False, 'has_solution': False, 'solver_status': 'Time limit reached'}))",
            RunStatus.NO_SOLUTION,
        ),
        (GOOD.replace("[[1]]", "[[999]]"), RunStatus.INVALID),
        (
            "p.write_text(json.dumps({'valid': False, 'failure_reason': 'out_of_memory', 'error': 'MemoryError'}))\nsys.exit(1)",
            RunStatus.OUT_OF_MEMORY,
        ),
        (
            "p.write_text(json.dumps({'valid': False, 'failure_reason': 'timed_out', 'error': 'native process exceeded deadline'}))\nsys.exit(1)",
            RunStatus.TIMED_OUT,
        ),
        (
            "p.write_text(json.dumps({'valid': False, 'failure_reason': 'solver_error', 'solver_status': 'Solve error'}))",
            RunStatus.SOLVER_ERROR,
        ),
        ("import os,signal\nos.kill(os.getpid(),signal.SIGKILL)", RunStatus.CRASHED),
    ],
)
def test_real_subprocess_failures_remain_distinct(
    judge_case, tmp_path, script, expected
):
    judge, case = judge_case
    judge.repository = ScriptRepository(script)
    observation = judge._run_case(tmp_path, case, CodeState.AGENT, 0, 5)
    assert observation.status == expected
    assert observation.test_suite == "operational_reliability"
    assert Path(observation.stdout_path).is_file()
    assert Path(observation.stderr_path).is_file()
    validity = evaluator_validity(
        patch_exists=True, fixture_mode=False, observations=[observation]
    )
    assert validity.accepted is (expected != RunStatus.INVALID)
    if expected == RunStatus.COMPLETED:
        assert observation.objective == 10
        assert observation.solver_status == "Time limit reached"
        assert Path(observation.stdout_path).read_text() == "native stdout\n"


def test_solver_process_launch_failure_is_retained_as_infrastructure_error(
    judge_case, tmp_path, monkeypatch
):
    judge, case = judge_case

    def fail_to_start(command, workspace):
        del command, workspace
        raise FileNotFoundError("solver command is unavailable")

    monkeypatch.setattr(judge, "_run", fail_to_start)
    observation = judge._run_case(tmp_path, case, CodeState.BASE, 0, 5)

    assert observation.status is RunStatus.INFRASTRUCTURE_ERROR
    assert observation.valid is False
    assert "solver process launch failed" in observation.error
    assert Path(observation.stderr_path).is_file()
    process_path = Path(
        str(observation.stderr_path).removesuffix(".stderr.log") + ".process.json"
    )
    process = json.loads(process_path.read_text())
    assert process["timed_out"] is False


def test_an_old_result_cannot_make_a_later_missing_output_pass(judge_case, tmp_path):
    judge, case = judge_case
    judge.repository = ScriptRepository(GOOD)
    assert judge._run_case(tmp_path, case, CodeState.BASE, 0, 5).valid
    judge.repository = ScriptRepository("pass")
    assert (
        judge._run_case(tmp_path, case, CodeState.BASE, 0, 5).status
        == RunStatus.OUTPUT_ERROR
    )


def test_failed_parsed_run_retains_resource_measurements(judge_case, tmp_path):
    judge, case = judge_case
    judge.repository = ScriptRepository(
        "p.write_text(json.dumps({'valid': False, 'has_solution': False, "
        "'solver_status': 'Time limit reached', 'cpu_time_sec': 2.5, "
        "'peak_rss_bytes': 1234, 'resource_scope': 'single_native_child_lifetime'}))"
    )

    observation = judge._run_case(tmp_path, case, CodeState.BASE, 0, 5)

    assert observation.status is RunStatus.NO_SOLUTION
    assert observation.cpu_time_sec == 2.5
    assert observation.peak_rss_bytes == 1234
    assert observation.resource_scope == "single_native_child_lifetime"


def test_false_infeasible_claim_fails_qualification(judge_case, tmp_path):
    judge, case = judge_case
    judge.repository = ScriptRepository(
        "p.write_text(json.dumps({'valid': False, 'has_solution': False, "
        "'solver_status': 'Problem proven infeasible'}))"
    )
    case = replace(case, anchor=10)

    observation = judge._run_case(tmp_path, case, CodeState.AGENT, 0, 5)

    assert observation.status is RunStatus.INVALID
    assert "verified feasible reference" in observation.error
    assert not evaluator_validity(
        patch_exists=True, fixture_mode=False, observations=[observation]
    ).accepted


def test_prepared_repository_verifier_takes_precedence_over_family_verifier(
    judge_case, tmp_path
):
    judge, case = judge_case
    judge.repository = ScriptRepository(GOOD)
    case = replace(case, verifier=None)

    class PreparedVerifier(ProblemFamilyPlugin):
        name = ""

        def verify(self, instance_path, solution_path):
            del instance_path, solution_path
            return VerificationResult(feasible=True, objective=77)

    judge.prepared_verifier = PreparedVerifier()

    observation = judge._run_case(tmp_path, case, CodeState.BASE, 0, 5)

    assert observation.status is RunStatus.COMPLETED
    assert observation.objective == 77


def test_process_verifier_parses_evaluator_owned_json_result(tmp_path) -> None:
    verifier = ProcessVerifierFamily(
        [
            sys.executable,
            "-c",
            "import json; print(json.dumps({'feasible': True, 'objective': 4}))",
        ]
    )

    result = verifier.verify(tmp_path / "instance", tmp_path / "solution")

    assert result.feasible is True
    assert result.objective == 4
    assert result.infrastructure_error is False


def test_watchdog_terminates_the_solver_process_group(judge_case, tmp_path):
    judge, case = judge_case
    child_pid = tmp_path / "child.pid"
    script = (
        "import subprocess,time\n"
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])\n"
        f"Path({str(child_pid)!r}).write_text(str(child.pid))\n"
        "print('started',flush=True)\ntime.sleep(30)\n"
    )
    judge.repository = ScriptRepository(script, timeout=0.5)
    started = time.monotonic()
    result = judge._run_case(tmp_path, case, CodeState.BASE, 0, 5)
    assert result.status == RunStatus.TIMED_OUT
    assert time.monotonic() - started < 5
    assert "started" in Path(result.stdout_path).read_text()
    pid = int(child_pid.read_text())
    status = Path(f"/proc/{pid}/stat")
    # A killed child may briefly await reaping by the container init process.
    assert not status.exists() or status.read_text().split()[2] == "Z"


def test_failed_case_does_not_abort_remaining_cases(judge_case, tmp_path, monkeypatch):
    judge, first = judge_case
    cases = [
        replace(first, instance_id=name, solver_seeds=(0,), budgets_sec=(5,))
        for name in ("bad", "good")
    ]
    judge.repository = ScriptRepository(
        "\nif p.parent.name == 'bad': sys.exit(7)\n" + GOOD
    )
    monkeypatch.setattr(judge, "_workspace", lambda *args: tmp_path)
    saved = []
    results = judge.run(cases, save_observation=saved.append)
    assert len(saved) == len(results) == 2
    assert {item.instance_id: item.status for item in results} == {
        "bad": RunStatus.CRASHED,
        "good": RunStatus.COMPLETED,
    }


def test_judge_keeps_completed_observations_when_a_later_run_aborts(
    judge_case, tmp_path, monkeypatch
):
    judge, first = judge_case
    judge.observations_path = tmp_path / "judge-observations.jsonl"
    cases = [
        replace(first, instance_id=name, solver_seeds=(0,), budgets_sec=(5,))
        for name in ("completed", "aborted")
    ]
    monkeypatch.setattr(judge, "_workspace", lambda *args: tmp_path)

    def run_case(workspace, case, state, seed, budget, *, cpu_ids):
        del workspace, cpu_ids
        if case.instance_id == "aborted":
            time.sleep(0.1)
            raise RuntimeError("later run aborted")
        return judge._failure(case, state, seed, budget, RunStatus.CRASHED)

    monkeypatch.setattr(judge, "_run_case", run_case)
    with pytest.raises(RuntimeError, match="later run aborted"):
        judge.run(cases)

    retained = ObservationStore.read_jsonl(judge.observations_path)
    assert len(retained) == 1
    assert retained[0].instance_id == "completed"
    assert retained[0].status is RunStatus.CRASHED


def test_docker_judge_failure_reports_last_progress_and_retained_observations(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (tmp_path / "candidate.patch").write_bytes(b"")
    (tmp_path / "task.yaml").write_text("task_id: test\n")
    observations = output_dir / "judge-observations.jsonl"
    observations.write_text('{"completed":true}\n')

    class FailedProcess:
        stdout = [
            *(
                f"PITBENCH_PROGRESS Judge progress: {index}/100\n"
                for index in range(50)
            ),
            "docker connection closed\n",
        ]

        def wait(self):
            return 1

    monkeypatch.setattr(
        "pitbench.evaluator.docker_judge.subprocess.Popen",
        lambda *args, **kwargs: FailedProcess(),
    )
    judge = DockerJudge(
        image="sha256:" + "a" * 64,
        task_config_path=tmp_path / "task.yaml",
        base_repository=tmp_path,
        private_root=tmp_path,
        candidate_patch=tmp_path / "candidate.patch",
        output_dir=output_dir,
    )

    with pytest.raises(RuntimeError) as error:
        judge.run()

    message = str(error.value)
    assert "Judge progress: 49/100" in message
    assert "Judge progress: 48/100" not in message
    assert "docker connection closed" in message
    assert str(observations) in message


def test_limited_judge_run_omits_retired_reliability_suite(tmp_path, monkeypatch):
    from pitbench.evaluator import runner

    received = {}

    class FakeJudge:
        def __init__(self, **kwargs):
            received.update(kwargs)

        def run(self):
            return []

    output_dir = tmp_path / "output"
    monkeypatch.setattr(runner, "LocalProcessJudge", FakeJudge)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pitbench.evaluator.runner",
            "--task-config",
            str(ROOT / "configs/tasks/vroom_v1_15_0.yaml"),
            "--base-repository",
            str(tmp_path),
            "--public-root",
            str(ROOT),
            "--private-root",
            str(tmp_path),
            "--candidate-patch",
            str(tmp_path / "candidate.patch"),
            "--output-dir",
            str(output_dir),
            "--observations",
            str(output_dir / "judge-observations.jsonl"),
            "--instances-per-instance-set",
            "1",
        ],
    )

    runner.main()

    assert received["instances_per_instance_set"] == 1
    assert received["additional_cases"] == []


def test_driver_preserves_inner_timeout_reason(tmp_path, monkeypatch):
    from pitbench.solver_drivers.run import HighsDriver

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("native solver", 65)

    monkeypatch.setattr(subprocess, "run", timeout)
    output = tmp_path / "result.json"
    with pytest.raises(subprocess.TimeoutExpired):
        HighsDriver.main(
            [
                "--solver",
                "highs",
                "--instance",
                str(tmp_path / "input.lp"),
                "--output",
                str(output),
                "--trajectory",
                str(tmp_path / "trajectory.jsonl"),
                "--seed",
                "0",
                "--budget",
                "5",
                "--threads",
                "1",
            ]
        )
    assert json.loads(output.read_text())["failure_reason"] == "timed_out"


def test_native_status_and_objective_can_be_read_from_solution_file(
    tmp_path, monkeypatch, capsys
):
    from pitbench.solver_drivers.run import HighsDriver

    raw = "Model status\nOptimal\n\n# Primal solution values\nFeasible\nObjective 5\n# Columns 2\nx0 1\nx1 2\n# Rows 1\nc0 3\n"

    def solve(argv, **kwargs):
        output = next(
            value.split("=", 1)[1]
            for value in argv
            if value.startswith("--solution_file=")
        )
        Path(output).write_text(raw)
        return subprocess.CompletedProcess(
            argv, 0, "Solving report\n  Status Optimal\n", "native diagnostic\n"
        )

    monkeypatch.setattr(subprocess, "run", solve)
    output = tmp_path / "result.json"
    HighsDriver.main(
        [
            "--solver",
            "highs",
            "--instance",
            str(tmp_path / "input.lp"),
            "--output",
            str(output),
            "--trajectory",
            str(tmp_path / "trajectory.jsonl"),
            "--seed",
            "0",
            "--budget",
            "5",
            "--threads",
            "1",
        ]
    )
    result = json.loads(output.read_text())
    assert result["solver_status"] == "Optimal" and result["objective"] == 5
    assert result["solver_termination"] == "optimal"
    assert result["has_solution"]
    assert "native diagnostic" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("exit_code", "report"),
    [
        (0, "Primal bound 5\nModel status: Time limit reached\n"),
        (
            1,
            "Solving report\n  Status            Time limit reached\n  Primal bound      5\n",
        ),
    ],
)
def test_highs_time_limit_does_not_export_fractional_relaxation(
    tmp_path, monkeypatch, exit_code, report
):
    from pitbench.solver_drivers.run import HighsDriver

    raw = "Model status\nTime limit reached\n\n# Columns 1\nx0 0.5\n"

    def solve(argv, **kwargs):
        output = next(
            value.split("=", 1)[1]
            for value in argv
            if value.startswith("--solution_file=")
        )
        Path(output).write_text(raw)
        return subprocess.CompletedProcess(
            argv,
            exit_code,
            report,
            "",
        )

    monkeypatch.setattr(subprocess, "run", solve)
    output = tmp_path / "result.json"
    HighsDriver.main(
        [
            "--solver",
            "highs",
            "--instance",
            str(tmp_path / "input.lp"),
            "--output",
            str(output),
            "--trajectory",
            str(tmp_path / "trajectory.jsonl"),
            "--seed",
            "0",
            "--budget",
            "5",
            "--threads",
            "1",
        ]
    )

    result = json.loads(output.read_text())
    assert result["solver_termination"] == "time_limit"
    assert result["has_solution"] is False
    assert result["valid"] is False
    assert not output.with_suffix(".solution.json").exists()


@pytest.mark.parametrize(
    ("option_exit_code", "option_message"),
    [
        (255, "The following argument was not expected: --threads=1\n"),
        (0, "ERROR:   Error parsing options: Option ‘threads’ does not exist\n"),
    ],
)
def test_highs_driver_retries_only_unsupported_threads_option(
    tmp_path, monkeypatch, option_exit_code, option_message
):
    from pitbench.solver_drivers.run import HighsDriver

    calls = []

    def solve(argv, **kwargs):
        calls.append(argv)
        if any(part.startswith("--threads=") for part in argv):
            return subprocess.CompletedProcess(
                argv,
                option_exit_code,
                option_message,
                "",
            )
        output = next(
            part.split("=", 1)[1]
            for part in argv
            if part.startswith("--solution_file=")
        )
        Path(output).write_text("Model status\nOptimal\nObjective 5\n")
        return subprocess.CompletedProcess(
            argv, 0, "Solving report\n  Status Optimal\n  Primal bound 5\n", ""
        )

    monkeypatch.setattr(subprocess, "run", solve)
    output = tmp_path / "result.json"
    HighsDriver.main(
        [
            "--solver",
            "highs",
            "--instance",
            str(tmp_path / "input.lp"),
            "--output",
            str(output),
            "--trajectory",
            str(tmp_path / "trajectory.jsonl"),
            "--seed",
            "0",
            "--budget",
            "5",
            "--threads",
            "1",
        ]
    )
    result = json.loads(output.read_text())
    assert len(calls) == 2
    assert all(not part.startswith("--threads=") for part in calls[1])
    assert result["cli_threads_option_fallback"] is True
    assert result["solver_termination"] == "optimal"


# Tests consolidated from tests/unit/pitbench/test_judge_parallel_and_cache.py


def test_default_judge_parallel_runs_and_cpus():
    mock_task = Mock()
    mock_task.evaluation.threads = 1

    with patch("os.sched_getaffinity", return_value=set(range(12))):
        assert _default_judge_parallel_runs(mock_task) == 10
        assert _default_judge_cpus(10, mock_task) == 10.0

    with patch("os.sched_getaffinity", return_value=set(range(4))):
        assert _default_judge_parallel_runs(mock_task) == 3
        assert _default_judge_cpus(3, mock_task) == 8.0

    with patch("os.sched_getaffinity", return_value={0}):
        assert _default_judge_parallel_runs(mock_task) == 1
        assert _default_judge_cpus(1, mock_task) == 8.0


def test_cp_sat_uses_one_solver_thread_with_parallel_judge_processes():
    task = PitBenchTask.from_yaml(
        ROOT / "configs/tasks/ortools_cp_sat_solve_v9_15.yaml"
    )
    assert task.evaluation.threads == 1
    with patch("os.sched_getaffinity", return_value=set(range(12))):
        assert _default_judge_parallel_runs(task) == 10


def test_base_cache_rejects_observations_from_other_thread_protocol(tmp_path: Path):
    task = Mock()
    task.task_id = "test_task"
    task.evaluation.threads = 1
    task.evaluation.seed_robustness = None
    task.evaluation.solver_seeds = [7]
    task.evaluation.budgets_sec = [5.0]
    task.evaluation.representation_robustness = None
    task.instance_sets = [SimpleNamespace(name="judge_id", kind="judge_id", size=1)]
    observation = RunObservation(
        task_id=task.task_id,
        code_state=CodeState.BASE,
        instance_set="judge_id",
        instance_id="inst_1",
        budget_sec=5.0,
        solver_seed=7,
        threads=8,
        status=RunStatus.COMPLETED,
        valid=True,
    )
    assert not _base_cache_is_complete(task, [observation], tmp_path)
    assert _base_cache_is_complete(
        task, [observation.model_copy(update={"threads": 1})], tmp_path
    )


def test_base_cache_path_changes_with_task_config_and_judge_image(tmp_path: Path):
    task = SimpleNamespace(task_id="test_task", evaluation=SimpleNamespace(threads=1))
    task_config = tmp_path / "task.yaml"
    task_config.write_text("old panel")
    first = _base_cache_path(tmp_path, task, task_config, "sha256:" + "a" * 64)
    task_config.write_text("new panel")
    changed_panel = _base_cache_path(tmp_path, task, task_config, "sha256:" + "a" * 64)
    changed_image = _base_cache_path(tmp_path, task, task_config, "sha256:" + "b" * 64)
    with patch(
        "pitbench.evaluator.evaluator.judge_harness_sha256",
        return_value="f" * 64,
    ):
        changed_harness = _base_cache_path(
            tmp_path, task, task_config, "sha256:" + "b" * 64
        )
    assert first != changed_panel != changed_image
    assert changed_image != changed_harness
    assert first.name != "test_task-threads-1_base.parquet"


def test_judge_harness_fingerprint_changes_with_mounted_code(tmp_path: Path) -> None:
    source = tmp_path / "pitbench" / "evaluator" / "runner.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = 1\n")
    adapter = tmp_path / "adapters" / "pitbench" / "adapter.py"
    adapter.parent.mkdir(parents=True)
    adapter.write_text("value = 1\n")

    first = judge_harness_sha256(tmp_path)
    adapter.write_text("value = 2\n")
    assert judge_harness_sha256(tmp_path) != first


def test_evaluator_uses_base_cache_when_available(tmp_path: Path):
    task_id = "test_task"
    (tmp_path / "task.yaml").write_text("task")
    base_cache_file = (
        tmp_path / "cache" / f"{task_id}-threads-1-task-{'0' * 16}-image-{'0' * 16}-"
        f"harness-{'0' * 16}_base.parquet"
    )
    base_obs = [
        RunObservation(
            task_id=task_id,
            code_state=CodeState.BASE,
            instance_set="judge_id",
            instance_id="inst_1",
            budget_sec=5.0,
            solver_seed=1,
            threads=1,
            status=RunStatus.COMPLETED,
            valid=True,
        )
    ]
    ObservationStore.write(base_cache_file, base_obs)

    agent_obs = [
        RunObservation(
            task_id=task_id,
            code_state=CodeState.AGENT,
            instance_set="judge_id",
            instance_id="inst_1",
            budget_sec=5.0,
            solver_seed=1,
            threads=1,
            status=RunStatus.COMPLETED,
            valid=True,
        )
    ]

    mock_task = Mock()
    mock_task.task_id = task_id
    mock_task.evaluation.seed_robustness = None
    mock_task.evaluation.representation_robustness = None
    mock_task.evaluation.operational_reliability = False
    mock_task.evaluation.performance_protocol = "heuristic_fixed_budget"
    mock_task.evaluation.primary_budget_sec = 5.0
    mock_task.evaluation.budgets_sec = [5.0]
    mock_task.evaluation.solver_seeds = [1]
    mock_task.evaluation.threads = 1
    mock_task.instance_sets = [SimpleNamespace(name="judge_id", size=1)]

    request = EvaluationRequest(
        task_id=task_id,
        task_path=tmp_path,
        candidate_patch_path=tmp_path / "candidate.patch",
        candidate_patch_sha256="0" * 64,
        output_dir=tmp_path / "output",
        agent_name="test_agent",
        model_name="test_model",
        evaluator_config={
            "task_config_path": str(tmp_path / "task.yaml"),
            "base_repository": str(tmp_path / "repo"),
            "private_root": str(tmp_path / "private"),
            "judge_image": "sha256:" + "a" * 64,
            "base_cache_path": str(tmp_path / "cache"),
            "use_base_cache": True,
        },
    )
    (tmp_path / "candidate.patch").write_bytes(b"")

    with (
        patch(
            "pitbench.evaluator.evaluator.PitBenchTask.from_yaml",
            return_value=mock_task,
        ),
        patch("pitbench.evaluator.evaluator.PitBenchAdapter.validate_repository"),
        patch("pitbench.evaluator.evaluator.DockerJudge") as mock_judge_class,
        patch("pitbench.evaluator.evaluator.hashlib.sha256") as mock_hash,
    ):
        mock_hash.return_value.hexdigest.return_value = "0" * 64
        mock_judge_instance = mock_judge_class.return_value
        mock_judge_instance.run.return_value = agent_obs

        result = PitBenchEvaluator().evaluate(request)

        # Ensure DockerJudge only ran with AGENT state
        _, kwargs = mock_judge_class.call_args
        assert kwargs["code_states"] == (CodeState.AGENT,)
        assert kwargs["parallel_runs"] >= 1

        # Check that result observations merged cached base and new agent
        parquet_obs = ObservationStore.read(tmp_path / "output" / "trials.parquet")
        assert len(parquet_obs) == 2
        states = {o.code_state for o in parquet_obs}
        assert states == {CodeState.BASE, CodeState.AGENT}
        assert result.validity.accepted is True


def test_evaluator_saves_base_cache_on_first_run(tmp_path: Path):
    task_id = "test_task_first_run"
    cache_dir = tmp_path / "cache"
    (tmp_path / "task.yaml").write_text("task")
    base_cache_file = cache_dir / (
        f"{task_id}-threads-1-task-{'0' * 16}-image-{'0' * 16}-"
        f"harness-{'0' * 16}_base.parquet"
    )

    all_obs = [
        RunObservation(
            task_id=task_id,
            code_state=CodeState.BASE,
            instance_set="judge_id",
            instance_id="inst_1",
            budget_sec=5.0,
            solver_seed=1,
            threads=1,
            status=RunStatus.COMPLETED,
            valid=True,
        ),
        RunObservation(
            task_id=task_id,
            code_state=CodeState.AGENT,
            instance_set="judge_id",
            instance_id="inst_1",
            budget_sec=5.0,
            solver_seed=1,
            threads=1,
            status=RunStatus.COMPLETED,
            valid=True,
        ),
    ]

    mock_task = Mock()
    mock_task.task_id = task_id
    mock_task.evaluation.seed_robustness = None
    mock_task.evaluation.representation_robustness = None
    mock_task.evaluation.operational_reliability = False
    mock_task.evaluation.performance_protocol = "heuristic_fixed_budget"
    mock_task.evaluation.primary_budget_sec = 5.0
    mock_task.evaluation.budgets_sec = [5.0]
    mock_task.evaluation.solver_seeds = [1]
    mock_task.evaluation.threads = 1
    mock_task.instance_sets = [SimpleNamespace(name="judge_id", size=1)]

    request = EvaluationRequest(
        task_id=task_id,
        task_path=tmp_path,
        candidate_patch_path=tmp_path / "candidate.patch",
        candidate_patch_sha256="0" * 64,
        output_dir=tmp_path / "output",
        agent_name="test_agent",
        model_name="test_model",
        evaluator_config={
            "task_config_path": str(tmp_path / "task.yaml"),
            "base_repository": str(tmp_path / "repo"),
            "private_root": str(tmp_path / "private"),
            "judge_image": "sha256:" + "a" * 64,
            "base_cache_path": str(cache_dir),
            "use_base_cache": True,
        },
    )
    (tmp_path / "candidate.patch").write_bytes(b"")

    with (
        patch(
            "pitbench.evaluator.evaluator.PitBenchTask.from_yaml",
            return_value=mock_task,
        ),
        patch("pitbench.evaluator.evaluator.PitBenchAdapter.validate_repository"),
        patch("pitbench.evaluator.evaluator.DockerJudge") as mock_judge_class,
        patch("pitbench.evaluator.evaluator.hashlib.sha256") as mock_hash,
    ):
        mock_hash.return_value.hexdigest.return_value = "0" * 64
        mock_judge_instance = mock_judge_class.return_value
        mock_judge_instance.run.return_value = all_obs

        PitBenchEvaluator().evaluate(request)

        # Ensure DockerJudge was asked to run both states
        _, kwargs = mock_judge_class.call_args
        assert set(kwargs["code_states"]) == {CodeState.BASE, CodeState.AGENT}

        # Ensure base cache was saved automatically
        assert base_cache_file.is_file()
        cached = ObservationStore.read(base_cache_file)
        assert len(cached) == 1
        assert cached[0].code_state == CodeState.BASE


def test_directional_performance_requires_fresh_opposite_order_confirmation(
    tmp_path: Path,
) -> None:
    task_id = "test_task"
    task_config = tmp_path / "task.yaml"
    task_config.write_text("test task")
    patch_path = tmp_path / "candidate.patch"
    patch_path.write_bytes(b"")
    task = Mock()
    task.task_id = task_id
    task.evaluation.seed_robustness = None
    task.evaluation.representation_robustness = None
    task.evaluation.operational_reliability = False
    task.evaluation.performance_protocol = PerformanceProtocol.HEURISTIC_FIXED_BUDGET
    task.evaluation.primary_budget_sec = 5.0
    task.evaluation.budgets_sec = [5.0]
    task.evaluation.solver_seeds = [1]
    task.evaluation.threads = 1
    task.instance_sets = [SimpleNamespace(name="judge_id", size=1)]

    def observation(state: CodeState, gap: float) -> RunObservation:
        return RunObservation(
            task_id=task_id,
            code_state=state,
            instance_set="judge_id",
            instance_set_kind="judge_id",
            instance_id="one",
            solver_seed=1,
            budget_sec=5.0,
            status=RunStatus.COMPLETED,
            valid=True,
            normalized_gap=gap,
        )

    states_seen = []

    class FakeJudge:
        def __init__(self, **kwargs):
            self.output_dir = kwargs["output_dir"]
            self.states = kwargs["code_states"]
            states_seen.append(self.states)

        def checkpoint_identity(self):
            return {"code_states": [state.value for state in self.states]}

        def run(self):
            self.output_dir.mkdir(parents=True, exist_ok=True)
            grid = ExpectedRunGrid(
                task_id=task_id,
                runs=[
                    ExpectedRun(
                        task_id=task_id,
                        code_state=state,
                        instance_set="judge_id",
                        instance_set_kind="judge_id",
                        instance_id="one",
                        solver_seed=1,
                        budget_sec=5.0,
                    )
                    for state in CodeState
                ],
            )
            (self.output_dir / "expected-run-grid.json").write_text(
                grid.model_dump_json()
            )
            agent_gap = (
                0.2 if self.output_dir.name == "performance-confirmation" else 0.1
            )
            return [
                observation(CodeState.BASE, 0.2),
                observation(CodeState.AGENT, agent_gap),
            ]

    request = EvaluationRequest(
        task_id=task_id,
        task_path=tmp_path,
        candidate_patch_path=patch_path,
        candidate_patch_sha256=hashlib.sha256(b"").hexdigest(),
        output_dir=tmp_path / "output",
        agent_name="replay",
        model_name=None,
        evaluator_config={
            "task_config_path": str(task_config),
            "base_repository": str(tmp_path / "repo"),
            "private_root": str(tmp_path / "private"),
            "judge_image": "sha256:" + "a" * 64,
            "judge_cpus": 12,
            "judge_parallel_runs": 12,
            "use_base_cache": False,
        },
    )
    with (
        patch("pitbench.evaluator.evaluator.PitBenchTask.from_yaml", return_value=task),
        patch("pitbench.evaluator.evaluator.PitBenchAdapter.validate_repository"),
        patch("pitbench.evaluator.evaluator.DockerJudge", side_effect=FakeJudge),
    ):
        result = PitBenchEvaluator().evaluate(request)

    assert states_seen == [
        (CodeState.BASE, CodeState.AGENT),
        (CodeState.AGENT, CodeState.BASE),
    ]
    assert result.summary.performance is not None
    assert (
        result.summary.performance.classification
        is PerformanceClassification.INCONCLUSIVE
    )
    assert result.summary.performance.confirmation is not None
    assert (
        result.summary.performance.confirmation.initial_classification
        is PerformanceClassification.IMPROVED
    )
    assert (
        result.summary.performance.confirmation.repeat_classification
        is PerformanceClassification.INCONCLUSIVE
    )
    assert result.summary.performance_confirmation_observation_count == 2
    assert result.artifacts.performance_confirmation_observations is not None
    assert (request.output_dir / "performance-confirmation" / "report.json").is_file()


# Tests consolidated from tests/unit/pitbench/test_collection.py


ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "task_id,direction,expected",
    [
        ("pyvrp_v0_14_0", "minimize", 7),
        ("highs_v1_15_1", "maximize", 12),
    ],
)
@pytest.mark.parametrize("failed", [False, True])
def test_anchor_collection_uses_task_judge_and_retains_results(
    tmp_path, monkeypatch, task_id, direction, expected, failed
):
    task = PitBenchTask.from_yaml(ROOT / "configs/tasks" / f"{task_id}.yaml")
    task = task.model_copy(
        update={"oracle": task.oracle.model_copy(update={"objective_sense": direction})}
    )
    monkeypatch.setattr(PitBenchTask, "from_yaml", lambda path: task)
    monkeypatch.setattr(PitBenchAdapter, "validate_repository", lambda *args: None)
    private = tmp_path / "private"
    verifier_path = private / task.evaluation.verifier.removeprefix("private://")
    verifier_path.parent.mkdir(parents=True, exist_ok=True)
    verifier_path.touch()
    config = tmp_path / "instances.yaml"
    generator = (
        {
            "kind": "euclidean_cvrp",
            "count": 1,
            "customers": [3],
            "capacity_ratio": 1,
            "randomness": {"coordinate_seed": 1, "demand_seed": 2},
            "distance_metric": "EUC_2D",
        }
        if task.problem_family == "cvrp"
        else {
            "kind": "single_machine_scheduling_mip",
            "count": 1,
            "jobs": [2],
            "randomness": {"instance_seed": 1},
        }
    )
    config.write_text(yaml.safe_dump({"visibility": "judge", "generator": generator}))
    output = private / "oracles" / "custom_panel"

    def run(judge, cases, *, save_observation):
        assert judge.task is task
        assert judge.repository.name == task_id.split("_v")[0]
        assert judge.evaluation_seeds == (8, 3)
        assert judge.code_states == (CodeState.BASE,)
        assert len(cases) == 1
        case = cases[0]
        assert case.budgets_sec == (35,)
        observations = []
        for seed, objective in [(8, 12), (3, 7)]:
            solution = output / f"solution-{seed}.json"
            solution.write_text(json.dumps({"objective": objective}))
            observation = RunObservation(
                task_id=task_id,
                code_state=CodeState.BASE,
                instance_set=case.instance_set.name,
                instance_id=case.instance_id,
                solver_seed=seed,
                budget_sec=35,
                status=RunStatus.INVALID if failed else RunStatus.COMPLETED,
                valid=not failed,
                objective=objective,
                solution_path=str(solution),
            )
            observations.append(observation)
            save_observation(observation)
        return observations

    monkeypatch.setattr(LocalProcessJudge, "run", run)
    argv = [
        "--task-config",
        "task.yaml",
        "--repository",
        str(tmp_path),
        "--instance-set-config",
        str(config),
        "--private-root",
        str(private),
        "--output-dir",
        str(output),
        "--budget-sec",
        "35",
        "--seeds",
        "8",
        "3",
    ]
    if failed:
        with pytest.raises(ValueError, match="invalid BKS candidate"):
            AnchorCollection.main(argv)
        assert not (output / "oracle.yaml").exists()
    else:
        AnchorCollection.main(argv)
        oracle = yaml.safe_load((output / "oracle.yaml").read_text())
        assert oracle["problem_family"] == task.problem_family
        assert oracle["objective_sense"] == direction
        assert oracle["solver"]["version"] == task.release.version
        assert oracle["anchors"][0]["bks"] == expected
        assert (
            oracle["anchors"][0]["bks_solution_uri"]
            == "private://oracles/custom_panel/judge_shift_0000.bks.solution.json"
        )
        assert oracle["protocol"]["seeds"] == [8, 3]
        assert oracle["protocol"]["budget_sec"] == 35
    assert len((output / "observations.jsonl").read_text().splitlines()) == 2
    assert (output / "observations.parquet").exists()


def test_seed_matrix_uses_explicit_tasks_and_their_images(tmp_path):
    batch = tmp_path / "batch with spaces"
    private = tmp_path / "private"
    private.mkdir()
    harness = batch / "harness"
    script = harness / "scripts/validate_seed_robustness_real_solver.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "output=Path(sys.argv[sys.argv.index('--output-dir')+1])\n"
        "(output/'arguments.json').write_text(json.dumps(sys.argv[1:]))\n"
    )
    configs = harness / "configs/tasks"
    configs.mkdir(parents=True)
    for task, image in [
        ("solver_a", "sha256:" + "a" * 64),
        ("solver_b", "sha256:" + "b" * 64),
    ]:
        (batch / "sources" / task / ".git").mkdir(parents=True)
        (configs / f"{task}.yaml").write_text(
            yaml.safe_dump({"repository": {"judge_image": image}})
        )
    environment = {**os.environ, "PITBENCH_PYTHON": sys.executable}
    environment.pop("PITBENCH_JUDGE_IMAGE", None)
    command = [
        "bash",
        str(ROOT / "scripts/run-seed-validation-matrix.sh"),
        str(batch),
        str(private),
    ]
    result = subprocess.run(command, env=environment, capture_output=True, text=True)
    assert result.returncode != 0 and "at least one task ID" in result.stderr
    subprocess.run(
        [*command, "solver_a", "solver_b"],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    for task, letter in [("solver_a", "a"), ("solver_b", "b")]:
        argv = json.loads((batch / task / "arguments.json").read_text())
        assert argv[argv.index("--judge-image") + 1] == "sha256:" + letter * 64
        assert argv[argv.index("--repository") + 1] == str(batch / "sources" / task)
        assert argv[argv.index("--private-root") + 1] == str(private)


def nuisance_fixture(tmp_path, monkeypatch, execution="isolated"):
    from pitbench.evaluator.collection import COLLECTION_EXECUTORS

    instance = tmp_path / "instance.json"
    instance.write_text(
        json.dumps(
            {
                "depot": 0,
                "distance_metric": "EUC_2D",
                "coordinates": [[0, 0], [1, 1], [3, 0], [3, 4], [5, 1], [6, 3]],
                "demands": [0, 1, 1, 1, 1, 1],
                "capacity": 5,
            }
        )
    )
    index = tmp_path / "index.yaml"
    index.write_text(
        yaml.safe_dump(
            {"instances": [{"id": "tiny", "path": str(instance), "bks": 20}]}
        )
    )
    config = {
        "task_config": "configs/tasks/pyvrp_v0_14_0.yaml",
        "execution": execution,
        "instance_source": str(index),
        "instance_count": 1,
        "axes": ["seed", "representation", "control"],
        "code_states": ["base"],
        "solver_seeds": [3, 7],
        "control_repeats": 1,
        "schedule_seed": 19,
        "watchdog_grace_sec": 60,
        "representation": {
            "kind": "customer_relabeling",
            "count": 2,
            "solver_seed": 0,
            "generation_seed": 13,
        },
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(
        COLLECTION_EXECUTORS[execution],
        "identity",
        lambda task, repository: {"version": task.release.version},
    )
    return path


def test_nuisance_resume_preserves_failed_results_and_runs_only_missing_jobs(
    tmp_path, monkeypatch
):
    from pitbench.evaluator.collection import IsolatedCollection, NuisanceCollection

    config = nuisance_fixture(tmp_path, monkeypatch)
    output = tmp_path / "out"
    first = NuisanceCollection.prepare(config, output)
    assert len(first["jobs"]) == 10
    assert NuisanceCollection.prepare(config, output) == first
    calls = []

    def run(output, manifest, jobs, cpus):
        for job in jobs:
            calls.append(job["run_id"])
            path = output / "runs" / job["run_id"] / "result.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        **job,
                        "execution_status": "watchdog_timeout",
                        "verified_feasible": False,
                    }
                )
            )

    monkeypatch.setattr(IsolatedCollection, "run", run)
    cpus = [min(os.sched_getaffinity(0))]
    partial = NuisanceCollection.run(output, cpus, limit=1)
    assert partial["completed_runs"] == 1 and partial["missing_runs"] == 9
    first_path = output / "runs" / calls[0] / "result.json"
    first_result = first_path.read_bytes()
    complete = NuisanceCollection.run(output, cpus)
    assert complete["complete"] and complete["execution_statuses"] == {
        "watchdog_timeout": 10
    }
    assert len(set(calls)) == len(calls) == 10
    assert first_path.read_bytes() == first_result
    NuisanceCollection.run(output, cpus)
    assert len(calls) == 10
    payload = yaml.safe_load(config.read_text())
    payload["representation"]["solver_seed"] = 9
    config.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="differs"):
        NuisanceCollection.prepare(config, output)


def test_nuisance_judge_uses_exact_saved_grid_and_independent_mapping(
    tmp_path, monkeypatch
):
    from pitbench.evaluator.collection import NuisanceCollection

    config = nuisance_fixture(tmp_path, monkeypatch, "judge")
    output = tmp_path / "out"
    manifest = NuisanceCollection.prepare(config, output, repository=tmp_path)
    monkeypatch.setattr(LocalProcessJudge, "_workspace", lambda *args: tmp_path)
    calls = []

    def run_case(judge, workspace, case, state, seed, budget, **kwargs):
        calls.append((case.instance_id, state, seed, budget))
        solution = (
            judge.output_dir
            / state.value
            / case.instance_set.name
            / case.instance_id
            / f"seed-{seed}-budget-{budget:g}.solution.json"
        )
        solution.parent.mkdir(parents=True, exist_ok=True)
        solution.write_text(json.dumps({"routes": [[1, 2, 3, 4, 5]]}))
        verified = case.verifier.verify(case.path, solution)
        return RunObservation(
            task_id=judge.task.task_id,
            code_state=state,
            instance_set=case.instance_set.name,
            instance_set_kind="agent_dev",
            instance_id=case.instance_id,
            solver_seed=seed,
            budget_sec=budget,
            status=RunStatus.COMPLETED,
            valid=verified.feasible,
            objective=verified.objective,
            solver_status="Budget stop",
            equivalence_parent_id=case.equivalence_parent_id,
            equivalence_transform=case.equivalence_transform,
            solution_path=str(solution),
        )

    monkeypatch.setattr(LocalProcessJudge, "_run_case", run_case)
    cpus = [min(os.sched_getaffinity(0))]
    result = NuisanceCollection.run(output, cpus, limit=1)
    assert result["completed_runs"] == 1
    result = NuisanceCollection.run(output, cpus)
    assert result["complete"] and len(calls) == len(set(calls)) == len(manifest["jobs"])
    for job in manifest["jobs"]:
        record = json.loads(
            (output / "runs" / job["run_id"] / "result.json").read_text()
        )
        assert record["verified_feasible"]
        if job["transformation"]:
            assert record["representation_verification"]["objective_preserved"]


def test_collection_instance_paths_are_data_driven(tmp_path):
    from pitbench.instances.generate import prepare_collection_instances

    source = tmp_path / "source.bin"
    source.write_bytes(b"input")
    index = tmp_path / "index.json"
    index.write_text(
        json.dumps({"instances": [{"name": "custom", "location": "source.bin"}]})
    )
    result = prepare_collection_instances(
        index, tmp_path / "inputs", path_template="{location}"
    )
    assert result[0]["id"] == "custom"
    assert Path(result[0]["path"]).read_bytes() == b"input"
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        prepare_collection_instances(
            index, tmp_path / "inputs", path_template="{location}"
        )


def test_model_matrix_uses_only_yaml_runs_and_explicit_task(tmp_path):
    binary = tmp_path / "bin"
    binary.mkdir()
    docker = binary / "docker"
    docker.write_text("#!/bin/sh\nexit 0\n")
    docker.chmod(0o755)
    capture = tmp_path / "calls.jsonl"
    uv = binary / "uv"
    uv.write_text(
        "#!/usr/bin/python3\nimport os, json, sys\nwith open(os.environ['MATRIX_CAPTURE'], 'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
    )
    uv.chmod(0o755)
    config = tmp_path / "local.yaml"
    config.write_text("{}\n")
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        yaml.safe_dump(
            {
                "runs": [
                    {"agent": "first", "model": "model-a", "label": "a"},
                    {
                        "agent": "second",
                        "model": "model-b",
                        "setting": "effort=small",
                        "label": "b",
                    },
                ]
            }
        )
    )
    subprocess.run(
        ["bash", str(ROOT / "scripts/run-model-matrix.sh"), "custom_task", str(matrix)],
        env={
            **os.environ,
            "PATH": str(binary) + os.pathsep + os.environ["PATH"],
            "PITBENCH_PYTHON": sys.executable,
            "PITBENCH_CONFIG": str(config),
            "PITBENCH_LOG_DIR": str(tmp_path / "logs"),
            "MATRIX_CAPTURE": str(capture),
        },
        check=True,
        capture_output=True,
        text=True,
    )
    calls = [json.loads(line) for line in capture.read_text().splitlines()]
    assert len(calls) == 2
    assert all(
        call[:4] == ["run", "pitbench", "evaluate", "custom_task"] for call in calls
    )
    assert "--agent-kwarg" not in calls[0]
    assert calls[1][-2:] == ["--agent-kwarg", "effort=small"]


# Tests consolidated from tests/unit/pitbench/test_validity.py


def _observation(state: CodeState, status: RunStatus) -> RunObservation:
    return RunObservation(
        task_id="validity",
        code_state=state,
        instance_set="judge_id",
        instance_set_kind="judge_id",
        instance_id="instance",
        instance_seed=0,
        solver_seed=0,
        budget_sec=1.0,
        status=status,
        valid=status == RunStatus.COMPLETED,
    )


def test_confirmed_invalid_agent_solution_fails_qualification() -> None:
    result = evaluator_validity(
        patch_exists=True,
        fixture_mode=False,
        observations=[_observation(CodeState.AGENT, RunStatus.INVALID)],
    )

    assert result.accepted is False
    assert [(check.code, check.passed) for check in result.checks] == [
        (ValidityCode.PATCH_APPLY, True),
        (ValidityCode.SOLUTION, False),
    ]


def test_semantically_invalid_agent_run_disqualifies_evaluation(
    tmp_path: Path,
) -> None:
    task = PitBenchTask.from_yaml(ROOT / "configs/tasks/pyvrp_v0_14_0.yaml")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    candidate_patch = output_dir / "candidate.patch"
    candidate_patch.write_text("")
    invalid_observation = RunObservation(
        task_id=task.task_id,
        code_state=CodeState.AGENT,
        instance_set="judge_id",
        instance_set_kind="judge_id",
        instance_id="invalid",
        solver_seed=0,
        budget_sec=task.evaluation.primary_budget_sec,
        status=RunStatus.INVALID,
        valid=False,
    )

    with patch(
        "pitbench.evaluator.evaluator.FixtureJudge.run",
        return_value=[invalid_observation],
    ):
        envelope = PitBenchEvaluator().envelope(
            EvaluationRequest(
                task_id=task.task_id,
                task_path=ROOT,
                candidate_patch_path=candidate_patch,
                output_dir=output_dir,
                agent_name="fixture",
                evaluator_config={
                    "task_config_path": str(ROOT / "configs/tasks/pyvrp_v0_14_0.yaml"),
                    "fixture_mode": True,
                },
            )
        )

    assert envelope.completed is True
    assert envelope.payload["validity"]["accepted"] is False
    assert envelope.payload["validity"]["checks"][-1]["code"] == "solution"
    private_observations = ObservationStore.read(output_dir / "trials.parquet")
    assert private_observations == [invalid_observation]
    assert envelope.payload["artifacts"]["observations"]["private"] is True
    assert envelope.payload["summary"]["performance"]["classification"] == (
        "incomplete"
    )


def test_exact_evaluator_uses_expected_grid_and_exact_report(tmp_path: Path) -> None:
    task = PitBenchTask.from_yaml(ROOT / "configs/tasks/highs_v1_15_1.yaml")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    candidate_patch = output_dir / "candidate.patch"
    candidate_patch.write_text("")

    envelope = PitBenchEvaluator().envelope(
        EvaluationRequest(
            task_id=task.task_id,
            task_path=ROOT,
            candidate_patch_path=candidate_patch,
            output_dir=output_dir,
            agent_name="fixture",
            evaluator_config={
                "task_config_path": str(ROOT / "configs/tasks/highs_v1_15_1.yaml"),
                "fixture_mode": True,
                "fixture_instances_per_instance_set": 1,
            },
        )
    )

    assert envelope.completed is True
    performance = envelope.payload["summary"]["performance"]
    assert performance["classification"] == "inconclusive"
    assert performance["primary"]["base"]["verified_solved_coverage"] == 0
    assert performance["primary"]["agent"]["verified_solved_coverage"] == 0
    assert performance["primary"]["base"]["complete"] is True
    expected_grid = envelope.payload["artifacts"]["expected_run_grid"]
    assert expected_grid["private"] is True
    assert (output_dir / expected_grid["path"]).is_file()


def test_dry_run_config_does_not_skip_the_performance_report(tmp_path: Path) -> None:
    task = PitBenchTask.from_yaml(ROOT / "configs/tasks/highs_v1_15_1.yaml")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    candidate_patch = output_dir / "candidate.patch"
    candidate_patch.write_text("")

    envelope = PitBenchEvaluator().envelope(
        EvaluationRequest(
            task_id=task.task_id,
            task_path=ROOT,
            candidate_patch_path=candidate_patch,
            output_dir=output_dir,
            agent_name="fixture",
            evaluator_config={
                "task_config_path": str(ROOT / "configs/tasks/highs_v1_15_1.yaml"),
                "fixture_mode": True,
                "fixture_instances_per_instance_set": 1,
                "dry_run": True,
            },
        )
    )

    assert envelope.completed is True
    assert envelope.payload["summary"]["performance"] is not None


def test_false_exact_optimality_claim_fails_evaluator_validity(
    tmp_path: Path,
) -> None:
    task = PitBenchTask.from_yaml(ROOT / "configs/tasks/highs_v1_15_1.yaml")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    candidate_patch = output_dir / "candidate.patch"
    candidate_patch.write_text("")
    false_claim = RunObservation(
        task_id=task.task_id,
        code_state=CodeState.AGENT,
        instance_set="judge_id",
        instance_set_kind="judge_id",
        instance_id="judge_id_0000",
        solver_seed=0,
        budget_sec=task.evaluation.primary_budget_sec,
        status=RunStatus.COMPLETED,
        valid=True,
        objective=42,
        reported_objective=41,
        optimal_or_bks=42,
        cpu_time_sec=1,
        solver_termination="optimal",
    )

    with patch(
        "pitbench.evaluator.evaluator.FixtureJudge.run",
        return_value=[false_claim],
    ):
        envelope = PitBenchEvaluator().envelope(
            EvaluationRequest(
                task_id=task.task_id,
                task_path=ROOT,
                candidate_patch_path=candidate_patch,
                output_dir=output_dir,
                agent_name="fixture",
                evaluator_config={
                    "task_config_path": str(ROOT / "configs/tasks/highs_v1_15_1.yaml"),
                    "fixture_mode": True,
                    "fixture_instances_per_instance_set": 1,
                },
            )
        )

    assert envelope.completed is True
    assert envelope.payload["validity"]["accepted"] is False
    assert envelope.payload["validity"]["checks"][-1]["code"] == "objective"
    assert envelope.payload["summary"]["performance"]["classification"] == (
        "incomplete"
    )


@pytest.mark.parametrize("status", [RunStatus.TIMED_OUT, RunStatus.CRASHED])
def test_operational_agent_failure_does_not_fail_qualification(
    status: RunStatus,
) -> None:
    result = evaluator_validity(
        patch_exists=True,
        fixture_mode=False,
        observations=[_observation(CodeState.AGENT, status)],
    )

    assert result.accepted is True
    assert all(check.code != ValidityCode.SOLUTION for check in result.checks)


def test_invalid_base_solution_does_not_disqualify_candidate() -> None:
    result = evaluator_validity(
        patch_exists=True,
        fixture_mode=False,
        observations=[_observation(CodeState.BASE, RunStatus.INVALID)],
    )

    assert result.accepted is True
    assert all(check.code != ValidityCode.SOLUTION for check in result.checks)
