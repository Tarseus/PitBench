from __future__ import annotations

import hashlib
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
from pitbench.evaluator.collection import AnchorCollection
from pitbench.evaluator.evaluator import (
    PitBenchEvaluator,
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
    materialize_generated_instance_set,
    prepare_cp_sat_job_shop_panel,
    prepare_trusted_optimum_oracle,
)
from pitbench.problem_families.base import ProblemFamilyPlugin, ProblemFamilyRegistry
from pitbench.problem_families.verification import (
    CVRPFamily,
    ExternalVerifierFamily,
    TrustedOptimumFamily,
    TrustedOptimumOracle,
    TrustedOptimumRecord,
)
from pitbench.repositories.base import CommandSpec, RepositoryPlugin
from pitbench.schema.evaluation import ValidityCode
from pitbench.schema.observation import CodeState, RunObservation, RunStatus
from pitbench.schema.task import (
    InstanceSetKind,
    InstanceSetSpec,
    PerformanceProtocol,
    PitBenchTask,
)

# Tests consolidated from tests/unit/pitbench/test_judge_failures.py


ROOT = Path(__file__).resolve().parents[3]


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


def test_primary_budget_must_belong_to_evaluation_budgets() -> None:
    task = PitBenchTask.from_yaml(ROOT / "configs/tasks/pyvrp_v0_14_0.yaml")
    payload = task.model_dump()
    payload["evaluation"]["primary_budget_sec"] = 60

    with pytest.raises(
        ValueError,
        match="primary budget must belong to evaluation budgets",
    ):
        PitBenchTask.model_validate(payload)


def test_removed_performance_decision_is_rejected() -> None:
    task = PitBenchTask.from_yaml(ROOT / "configs/tasks/pyvrp_v0_14_0.yaml")
    payload = task.model_dump()
    payload["evaluation"]["decision"] = {"minimum_success_rate_delta": 0}

    with pytest.raises(ValueError, match="evaluation.decision has been removed"):
        PitBenchTask.model_validate(payload)


def test_workspace_permissions_reuses_existing_agent_account() -> None:
    task = PitBenchTask.from_yaml(ROOT / "configs/tasks/highs_v1_15_1.yaml")

    dockerfile = PitBenchAdapter._workspace_permissions(task, include_runs=False)

    assert "getent group pitbench-agent >/dev/null" in dockerfile
    assert "id --user pitbench-agent >/dev/null 2>&1" in dockerfile
    assert "|| groupadd --non-unique" in dockerfile
    assert "|| useradd --non-unique" in dockerfile


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


@pytest.mark.parametrize("anchor", [0.0, math.inf, -math.inf, math.nan])
def test_normalized_gap_rejects_invalid_anchor(anchor: float) -> None:
    with pytest.raises(
        ValueError,
        match="normalized gap requires a finite nonzero anchor",
    ):
        ProblemFamilyPlugin.normalized_gap(
            1.0,
            anchor,
            objective_sense="minimize",
        )


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


def test_an_old_result_cannot_make_a_later_missing_output_pass(judge_case, tmp_path):
    judge, case = judge_case
    judge.repository = ScriptRepository(GOOD)
    assert judge._run_case(tmp_path, case, CodeState.BASE, 0, 5).valid
    judge.repository = ScriptRepository("pass")
    assert (
        judge._run_case(tmp_path, case, CodeState.BASE, 0, 5).status
        == RunStatus.OUTPUT_ERROR
    )


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


def test_highs_time_limit_does_not_export_fractional_relaxation(
    tmp_path, monkeypatch
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
            0,
            "Primal bound 5\nModel status: Time limit reached\n",
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
    assert not output.with_suffix(".solution.json").exists()


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


def test_evaluator_uses_base_cache_when_available(tmp_path: Path):
    task_id = "test_task"
    base_cache_file = tmp_path / "cache" / f"{task_id}_base.parquet"
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
    base_cache_file = cache_dir / f"{task_id}_base.parquet"

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
