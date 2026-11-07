from __future__ import annotations

import json
import shutil
import subprocess
import types
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from pitbench.research import prepare_generalization_study
from pitbench.schema.research import GeneralizationStudyPlan
from pitbench.schema.task import PitBenchTask

ROOT = Path(__file__).resolve().parents[3]
PLAN = ROOT / "configs/research/archive/generalization.json"


def test_model_repeats_use_one_shared_condition_without_abc_interventions():
    value = json.loads(
        (ROOT / "configs/research/latest_high_replicates_v3.json").read_text()
    )
    plan = GeneralizationStudyPlan.model_validate(value)
    assert plan.agent_attempts_per_condition == 3
    assert [arm.id for arm in plan.arms] == ["standard"]
    assert not plan.arms[0].require_candidate_hypotheses
    value["arms"][0]["require_candidate_hypotheses"] = True
    with pytest.raises(ValidationError, match="does not add an intervention"):
        GeneralizationStudyPlan.model_validate(value)


def test_all_latest_augmentation_plan_is_problem_scoped_and_controlled():
    value = json.loads((ROOT / "configs/research/latest_high_algorithm_space_v4.json").read_text())
    plan = GeneralizationStudyPlan.model_validate(value)
    assert len(plan.task_configs) == len(plan.task_targets) == 8
    assert plan.agent_attempts_per_condition == 3
    assert len({a.development_view for a in plan.arms}) == 1
    assert len({a.instruction for a in plan.arms}) == 1
    assert sum(a.agent_import_path is not None for a in plan.arms) == 1
    assert not any(a.require_candidate_hypotheses for a in plan.arms)
    value["arms"][1]["development_view"] = "different_inputs"
    with pytest.raises(ValidationError, match="share data"):
        GeneralizationStudyPlan.model_validate(value)


def test_targeted_views_preserve_primary_scoring_and_share_public_inputs(tmp_path):
    from pitbench.research import prepare_targeted_study_views

    plan = GeneralizationStudyPlan.model_validate_json((ROOT / "configs/research/latest_high_algorithm_space_v4.json").read_text())
    path = prepare_targeted_study_views(plan=plan, root=ROOT, destination=tmp_path / "views")
    views = json.loads(path.read_text())
    for relative in plan.task_configs:
        original = PitBenchTask.from_yaml(ROOT / relative)
        row = views["tasks"][original.task_id]
        task = PitBenchTask.from_yaml(Path(row["evaluator_task_config"]))
        assert task.release == original.release
        assert task.evaluation == original.evaluation
        assert task.instance_sets == original.instance_sets
        assert task.problem_variant == plan.task_targets[task.task_id].problem_variant
        assert row["cross_problem_performance_role"] == "separate_transfer_reporting"
        assert Path(views["arms"]["standard"][task.task_id]["agent_task_config_path"]).read_bytes() == Path(views["arms"]["algorithm_space"][task.task_id]["agent_task_config_path"]).read_bytes()


def test_task_seed_policy_preserves_deterministic_tasks(tmp_path):
    from pitbench.research import (
        prepare_generalization_seed_views,
        study_evaluation_matches,
    )

    value = json.loads((ROOT / "configs/research/latest_high_algorithm_space_v4.json").read_text())
    selected = {"release_cbc_v2_10_13", "release_vroom_v1_15_0"}
    value["task_configs"] = [p for p in value["task_configs"] if Path(p).stem in selected]
    value["task_targets"] = {k: v for k, v in value["task_targets"].items() if k in selected}
    value["task_evaluation_seed_policies"] = {k: v for k, v in value["task_evaluation_seed_policies"].items() if k in selected}
    plan = GeneralizationStudyPlan.model_validate(value)
    views = {"tasks": {}, "arms": {arm.id: {} for arm in plan.arms}}
    originals = {}
    for name in plan.task_configs:
        task = PitBenchTask.from_yaml(ROOT / name)
        originals[task.task_id] = task
        views["tasks"][task.task_id] = {"evaluator_task_config": str(ROOT / name)}
        for arm in plan.arms:
            views["arms"][arm.id][task.task_id] = {"agent_task_config_path": str(ROOT / name)}
    source = tmp_path / "source.json"
    source.write_text(json.dumps(views))
    result = json.loads(prepare_generalization_seed_views(
        plan=plan, views_path=source, private_root=tmp_path / "private", destination=tmp_path / "seeds"
    ).read_text())
    for task_id in selected:
        frozen = PitBenchTask.from_yaml(Path(result["tasks"][task_id]["evaluator_task_config"]))
        assert study_evaluation_matches(originals[task_id], frozen, plan)
        if task_id == "release_vroom_v1_15_0":
            assert frozen.evaluation == originals[task_id].evaluation
            assert frozen.evaluation.solver_seeds == [0]
        else:
            assert frozen.evaluation.seed_robustness.seed_selection.seed_count == 30


def test_run_balance_keeps_seeds_budgets_and_all_panel_conditions():
    from pitbench.research import balance_generalization_task
    from pitbench.schema.task import InstanceSetSpec

    original = PitBenchTask.from_yaml(ROOT / "configs/tasks/pyvrp_v0_14_0.yaml")
    names = {"distribution-a", "distribution-b", "distribution-c"}
    native_count = balance_generalization_task(
        original, panel_names=set(), target_runs=100000
    )[2]
    for name in sorted(names):
        original.instance_sets.append(
            InstanceSetSpec(
                name=name,
                kind="judge_shift",
                instance_set_config="private://panel.yaml",
                size=10,
                budgets_sec=[10, 60],
                primary_budget_sec=60,
            )
        )
    balanced, before, after = balance_generalization_task(
        original, panel_names=names, target_runs=native_count + 7 * 120
    )
    assert before == native_count + 30 * 120
    assert after == native_count + 7 * 120
    assert sorted(p.size for p in balanced.instance_sets if p.name in names) == [
        2,
        2,
        3,
    ]
    assert balanced.evaluation == original.evaluation
    assert balanced.instance_sets[:3] == original.instance_sets[:3]
    assert [p.budgets_sec for p in balanced.instance_sets] == [
        p.budgets_sec for p in original.instance_sets
    ]
    assert [p.size for p in original.instance_sets if p.name in names] == [10] * 3
    with pytest.raises(ValueError, match="retain every panel"):
        balance_generalization_task(
            original, panel_names=names, target_runs=native_count
        )


@pytest.mark.parametrize("reduced", [False, True])
def test_balanced_views_freeze_matching_private_and_agent_panel_configs(
    tmp_path, reduced
):
    import hashlib

    from pitbench.research import prepare_generalization_run_views
    from pitbench.schema.task import InstanceSetSpec

    task = PitBenchTask.from_yaml(ROOT / "configs/tasks/pyvrp_v0_14_0.yaml")
    private = tmp_path / "private"
    private.mkdir()
    panel_path = private / "panel.yaml"
    panel_path.write_text(
        yaml.safe_dump(
            {
                "visibility": "judge",
                "instances": [
                    {"id": f"case-{i}", "uri": f"private://{i}.json"} for i in range(4)
                ],
            }
        )
    )
    task.instance_sets.append(
        InstanceSetSpec(
            name="distribution",
            kind="judge_shift",
            instance_set_config="private://panel.yaml",
            instance_set_config_sha256=hashlib.sha256(
                panel_path.read_bytes()
            ).hexdigest(),
            size=4,
            budgets_sec=[5, 10],
            primary_budget_sec=10,
        )
    )
    task_path = tmp_path / "task.yaml"
    task_path.write_text(yaml.safe_dump(task.model_dump(mode="json")))
    from pitbench.research import balance_generalization_task

    before = balance_generalization_task(task, panel_names=set(), target_runs=100000)[2]
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "tasks": {
                    task.task_id: {
                        "evaluator_task_config": str(task_path),
                        "scientific_panels": {"distribution": {"instances": 4}},
                    }
                },
                "arms": {
                    "original": {
                        task.task_id: {"agent_task_config_path": str(task_path)}
                    }
                },
            }
        )
    )
    kwargs = dict(
        views_path=source,
        private_root=private,
        destination=tmp_path / "balanced",
        target_runs=before - 240,
        supplemental_seed_count=5 if reduced else None,
        supplemental_primary_budget_only=reduced,
    )
    result_path = prepare_generalization_run_views(**kwargs)
    result = json.loads(result_path.read_text())
    metadata = result["tasks"][task.task_id]
    balanced = PitBenchTask.from_yaml(Path(metadata["evaluator_task_config"]))
    panel = balanced.instance_sets[-1]
    assert panel.size == 2
    config = private / panel.instance_set_config.removeprefix("private://")
    assert [r["id"] for r in yaml.safe_load(config.read_text())["instances"]] == [
        "case-0",
        "case-1",
    ]
    assert (
        hashlib.sha256(config.read_bytes()).hexdigest()
        == panel.instance_set_config_sha256
    )
    agent = PitBenchTask.from_yaml(
        Path(result["arms"]["original"][task.task_id]["agent_task_config_path"])
    )
    assert agent.instance_sets == balanced.instance_sets
    assert metadata["run_balance"]["balanced_runs"] == before - 240
    assert metadata["run_balance"]["selected_runs"] == before - (
        460 if reduced else 240
    )
    assert panel.solver_seed_count == (5 if reduced else None)
    assert panel.budgets_sec == ([10] if reduced else [5, 10])
    assert balanced.evaluation == task.evaluation
    assert balanced.instance_sets[:3] == task.instance_sets[:3]
    assert prepare_generalization_run_views(**kwargs) == result_path
    assert PitBenchTask.from_yaml(task_path).instance_sets[-1].size == 4


def test_study_seed_views_keep_development_and_freeze_private_holdout(tmp_path):
    import hashlib

    from pitbench.evaluator.judge import _development_seeds, _evaluation_seeds
    from pitbench.evaluator.private_assets import (
        PrivateAssetResolver,
        load_private_seed_robustness_config,
    )
    from pitbench.research import (
        prepare_generalization_seed_views,
        study_evaluation_matches,
    )

    plan = GeneralizationStudyPlan.model_validate_json(
        (ROOT / "configs/research/generalization_latest_high_v2.json").read_text()
    )
    original_path = ROOT / plan.task_configs[0]
    plan.task_configs = [plan.task_configs[0]]
    original = PitBenchTask.from_yaml(original_path)
    before = hashlib.sha256(original_path.read_bytes()).hexdigest()
    task_id = original.task_id
    views = {
        "tasks": {task_id: {"evaluator_task_config": str(original_path)}},
        "arms": {
            arm.id: {task_id: {"agent_task_config_path": str(original_path)}}
            for arm in plan.arms
        },
    }
    views_path = tmp_path / "source.json"
    views_path.write_text(json.dumps(views))
    private = tmp_path / "private"
    output = tmp_path / "frozen"
    result_path = prepare_generalization_seed_views(
        plan=plan, views_path=views_path, private_root=private, destination=output
    )
    result = json.loads(result_path.read_text())
    task = PitBenchTask.from_yaml(
        Path(result["tasks"][task_id]["evaluator_task_config"])
    )
    resolver = PrivateAssetResolver(private)
    hidden = load_private_seed_robustness_config(
        resolver, task_id=task_id, public_config=task.evaluation.seed_robustness
    )
    assert len(_development_seeds(task)) == 5
    assert len(_evaluation_seeds(task, resolver)) == 30
    assert not set(hidden.evaluation_seeds) & set(_development_seeds(task))
    assert min(hidden.evaluation_seeds) > 0
    assert task.evaluation.solver_seeds is None
    assert study_evaluation_matches(original, task, plan)
    for arm in plan.arms:
        public = PitBenchTask.from_yaml(
            Path(result["arms"][arm.id][task_id]["agent_task_config_path"])
        )
        assert public.evaluation == task.evaluation
    assert (
        prepare_generalization_seed_views(
            plan=plan, views_path=views_path, private_root=private, destination=output
        )
        == result_path
    )
    assert hashlib.sha256(original_path.read_bytes()).hexdigest() == before
    changed = task.model_copy(deep=True)
    changed.evaluation.primary_budget_sec *= 2
    assert not study_evaluation_matches(original, changed, plan)
    seed_file = resolver.resolve(task.evaluation.seed_robustness.evaluation_seeds_file)
    seed_file.write_text(seed_file.read_text() + "\n")
    with pytest.raises(ValueError, match="frozen seed view changed"):
        prepare_generalization_seed_views(
            plan=plan, views_path=views_path, private_root=private, destination=output
        )


def test_difficulty_selection_uses_discovery_and_retains_missing_profiles(tmp_path):
    from pitbench.research import analyze_population_difficulty

    population = tmp_path / "population"
    population.mkdir()
    entries = [
        {
            "id": f"case-{i}",
            "sha256": "a" * 64,
            "replicate_index": i,
            "family": "routing",
            "condition_id": "cell",
            "generator": {"kind": "euclidean_cvrp"},
        }
        for i in [0, 5]
    ]
    split = population / "calibration.json"
    split.write_text(json.dumps({"instances": entries}))
    (population / "population-construction.json").write_text(
        json.dumps({"splits": {"protocol_calibration": {"manifest": str(split)}}})
    )
    spec = tmp_path / "spec.yaml"
    spec.write_text(
        yaml.safe_dump({"families": {"routing": {"problem_family": "cvrp"}}})
    )
    protocol = {
        "specification": str(spec),
        "selection_seed_by_backend": {"pyvrp": 0, "absent": 0},
        "backend_problem_family": {"pyvrp": "cvrp", "absent": "cvrp"},
        "budget_grid_sec": [1, 10],
        "discovery_replicates": [0],
        "qualification_replicates": [5],
        "budget_selection_quantile": 0.9,
        "routing_target_relative_gap": 0.01,
        "no_hit_tier": "censored",
        "tier_by_first_hit_budget": {"1": "short", "10": "long"},
    }
    proto = tmp_path / "protocol.yaml"
    proto.write_text(yaml.safe_dump(protocol))
    results = tmp_path / "results"
    folder = results / "hosts/host"
    folder.mkdir(parents=True)
    jobs, rows = [], []
    for entry in entries:
        for budget in [1, 10]:
            identity = f"{entry['id']}/{budget}"
            jobs.append(
                {
                    "id": identity,
                    "instance": entry,
                    "seconds": budget,
                    "seed": 0,
                    "purpose": "calibration",
                    "use_warm_start": False,
                    "backend": {"name": "pyvrp"},
                }
            )
            rows.append(
                {
                    "job_id": identity,
                    "execution_status": "completed",
                    "verified_feasible": True,
                    "objective": 12
                    if entry["replicate_index"] == 5 and budget == 1
                    else 10,
                    "proven_optimal": False,
                    "solver_status": "time_limit",
                }
            )
    (results / "distribution.json").write_text(
        json.dumps({"original_manifest": {"jobs": jobs}})
    )
    (results / "bks.json").write_text(
        json.dumps(
            {
                "references": {
                    entry["id"]: {
                        "objective": 10,
                        "proven_optimal": False,
                        "locally_verified": True,
                        "instance_sha256": entry["sha256"],
                    }
                    for entry in entries
                }
            }
        )
    )
    (folder / "attempts.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    report = analyze_population_difficulty(
        population_root=population,
        result_roots=[results],
        protocol_path=proto,
        destination=tmp_path / "report.json",
    )
    assert report["status"] == "partial"
    actual = next(p for p in report["profiles"] if p["backend"] == "pyvrp")
    assert actual["selected_budget_sec"] == 1
    assert actual["qualification_at_selected_budget"]["target_hits"] == 0
    missing = next(p for p in report["profiles"] if p["backend"] == "absent")
    assert missing["by_budget"]["1"]["discovery"]["missing"] == 1


@pytest.fixture
def study_root(tmp_path: Path) -> Path:
    plan = GeneralizationStudyPlan.model_validate_json(PLAN.read_text())
    for relative in [
        plan.release_manifest,
        plan.model_matrix,
        plan.population_plan,
        "configs/research/budgets.json",
        *plan.task_configs,
    ]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    return tmp_path


def test_study_materializes_requested_matrix_without_running_experiments(
    study_root: Path,
) -> None:
    plan = GeneralizationStudyPlan.model_validate_json(PLAN.read_text())
    population_path = study_root / plan.population_plan
    populations = json.loads(population_path.read_text())
    populations["headroom_evidence"] = {"private_probe": "hidden-reference-sentinel"}
    population_path.write_text(json.dumps(populations))
    destination = study_root / "design"

    result = prepare_generalization_study(
        plan, root=study_root, destination=destination
    )

    assert result["planned_agent_runs"] == 90
    assert result["maximum_agent_wall_hours"] == 360
    assert result["agent_attempts_per_condition"] == 1
    assert result["ready_for_agent_execution"] is False
    assert result["new_agent_calls"] == result["new_solver_runs"] == 0
    assert len({job["condition_id"] for job in result["jobs"]}) == 90
    assert {(j["model"]["model"], j["model"]["setting"]) for j in result["jobs"]} == {
        ("gpt-6.1-sol", "reasoning_effort=xhigh"),
        ("gpt-6.1-sol", "reasoning_effort=high"),
        ("gpt-6-luna", "reasoning_effort=medium"),
        ("gpt-6.1-terra", "reasoning_effort=xhigh"),
        ("gpt-6.1-terra", "reasoning_effort=high"),
    }
    assert json.loads((destination / "study.json").read_text()) == result
    assert (
        "population_manifest_missing:unseen_within_family" in result["readiness_gaps"]
    )
    assert any(
        gap.startswith("source_bound_scenario_missing:")
        for gap in result["readiness_gaps"]
    )
    assert (
        "evaluator_revalidation_of_population_reference_and_scenario_evidence_required"
        in result["readiness_gaps"]
    )
    views = list((destination / "agent_views").glob("*/*/case.json"))
    assert len(views) == 18
    for path in views:
        view = json.loads(path.read_text())
        assert view["research_budget"] == plan.research_budget.model_dump(mode="json")
        assert view["agent_tools"] == plan.shared_agent_tools
        assert "hidden-reference-sentinel" not in path.read_text()
        assert (
            not {"solver_seeds", "headroom_evidence", "scenario_bindings"} & view.keys()
        )
    for task in result["tasks"]:
        assert task["upstream_suite_command_declared"] is True
        frozen = PitBenchTask.from_yaml(study_root / task["task_config"])
        assert task["source_commit"] == frozen.release.base_commit
        assert task["source_tree"] == frozen.release.tree_sha
        assert task["solve_budgets_sec"] == frozen.evaluation.budgets_sec
        assert task["solver_seeds"] == frozen.evaluation.solver_seeds
        b = json.loads(
            (
                destination / "agent_views" / task["task_id"] / "diverse/case.json"
            ).read_text()
        )
        c = json.loads(
            (
                destination
                / "agent_views"
                / task["task_id"]
                / "mechanism_aware/case.json"
            ).read_text()
        )
        assert b["development_view"] == c["development_view"]
        assert b["agent_tools"] == c["agent_tools"]
        assert b["candidate_hypotheses_required"] is False
        assert c["candidate_hypotheses_required"] is True
    with pytest.raises(FileExistsError):
        prepare_generalization_study(plan, root=study_root, destination=destination)


@pytest.mark.parametrize(
    "change", ["information", "scope", "intervention", "duplicate_task"]
)
def test_study_rejects_confounding_or_unsupported_inference(change: str) -> None:
    value = json.loads(PLAN.read_text())
    if change == "information":
        value["arms"][2]["development_view"] = "extra_information"
    elif change == "scope":
        value["claim_scope"] = "replicated_study"
    elif change == "intervention":
        value["arms"][1]["require_candidate_hypotheses"] = True
    else:
        value["task_configs"].append(value["task_configs"][0])
    with pytest.raises(ValidationError):
        GeneralizationStudyPlan.model_validate(value)


@pytest.mark.parametrize("change", ["budget", "source"])
def test_study_rejects_drift_from_shared_budget_or_source_pin(
    study_root: Path, change: str
) -> None:
    value = json.loads(PLAN.read_text())
    if change == "budget":
        value["research_budget"]["cpus"] = 4
        expected = "shared named resource profile"
    else:
        path = study_root / value["task_configs"][0]
        task = yaml.safe_load(path.read_text())
        task["release"]["base_commit"] = "0" * 40
        path.write_text(yaml.safe_dump(task))
        expected = "approved release pin"
    with pytest.raises(ValueError, match=expected):
        prepare_generalization_study(
            GeneralizationStudyPlan.model_validate(value),
            root=study_root,
            destination=study_root / "design",
        )
    assert not (study_root / "design").exists()


def test_source_binding_rejects_changed_native_artifacts_and_changed_source(
    tmp_path, monkeypatch
):
    import sys

    from pitbench.solver_drivers.common import (
        record_source_binding,
        source_binding_identity,
    )

    source = tmp_path / "source"
    package = source / "fixture_binding"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("value = 1\n")
    (source / ".gitignore").write_text("*.so\n")
    for argv in [
        ["git", "init", "-q"],
        ["git", "add", "."],
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "Base",
        ],
    ]:
        subprocess.run(argv, cwd=source, check=True, capture_output=True)
    binary = package / "native.so"
    binary.write_bytes(b"original native artifact")
    module = types.ModuleType("fixture_binding")
    module.__file__ = str(package / "__init__.py")
    monkeypatch.setitem(sys.modules, "fixture_binding", module)
    manifest = source / ".pitbench/binding.json"
    record_source_binding(
        source=source,
        package="fixture_binding",
        repository_plugin="pitbench.repositories.plugins:PyVRPRepositoryPlugin",
        version="1",
        destination=manifest,
    )
    monkeypatch.setenv("PITBENCH_SCENARIO_BINDING", str(manifest))

    def probe():
        return source_binding_identity(
            type("Fixture", (), {}),
            package="fixture_binding",
            repository_plugin="pitbench.repositories.plugins:PyVRPRepositoryPlugin",
            representations=[],
        )

    assert probe()["version"] == "1"
    binary.write_bytes(b"replaced artifact")
    with pytest.raises(ValueError, match="package changed"):
        probe()
    binary.write_bytes(b"original native artifact")
    (package / "__init__.py").write_text("value = 2\n")
    with pytest.raises(ValueError, match="source changed"):
        probe()


@pytest.mark.parametrize(
    "tests,failures,command_failed", [(0, 0, False), (4, 1, True), (4, 0, False)]
)
def test_upstream_validation_retains_failure_counts_and_rejects_zero_tests(
    tmp_path, monkeypatch, tests, failures, command_failed
):
    from types import SimpleNamespace

    from pitbench.evaluator.judge import LocalProcessJudge
    from pitbench.schema.observation import CodeState

    judge = object.__new__(LocalProcessJudge)
    judge.output_dir = tmp_path / "validation"
    judge.task = SimpleNamespace(
        task_id="fixture", release=SimpleNamespace(base_commit="a" * 40)
    )
    judge.code_states = (CodeState.BASE,)

    def workspace(state, root, kind, **kwargs):
        path = root / state.value
        path.mkdir(parents=True)
        (path / ".pitbench-upstream-tests.xml").write_text(
            f'<testsuite tests="{tests}" failures="{failures}" errors="0" skipped="0"/>'
        )
        if command_failed:
            raise RuntimeError("upstream test command returned failure")
        return path

    monkeypatch.setattr(judge, "_workspace", workspace)
    result = judge.validate_sources(include_excluded_tests=True)
    assert result["new_solver_benchmark_runs"] == 0
    assert result["states"]["base"]["test_reports"][0]["tests"] == tests
    assert result["states"]["base"]["test_reports"][0]["failures"] == failures
    assert result["status"] == ("passed" if tests and not failures else "failed")


def test_judge_source_copy_does_not_reuse_relocated_build_caches(tmp_path):
    from types import SimpleNamespace

    from pitbench.evaluator.judge import LocalProcessJudge
    from pitbench.repositories.base import BuildKind
    from pitbench.schema.observation import CodeState

    source = tmp_path / "source"
    (source / "build").mkdir(parents=True)
    (source / "build/CMakeCache.txt").write_text("CMAKE_HOME_DIRECTORY=/old/source\n")
    (source / "src/generated").mkdir(parents=True)
    (source / "src/generated/old.o").write_bytes(b"stale compiled object")
    (source / "src/main.cpp").write_text("int main() { return 0; }\n")
    judge = object.__new__(LocalProcessJudge)
    judge.base_repository, judge.output_dir = source, tmp_path / "logs"
    judge.repository = SimpleNamespace(agent_build_paths=("build", "src/generated"))
    judge.validation_test_patch = None
    workspace = judge._workspace(
        CodeState.BASE, tmp_path / "work", BuildKind.VALIDATION, commands=[]
    )
    assert not (workspace / "build").exists()
    assert not (workspace / "src/generated").exists()
    assert (workspace / "src/main.cpp").read_text() == (
        source / "src/main.cpp"
    ).read_text()
    assert (source / "build/CMakeCache.txt").exists()


def test_incremental_qualification_changes_the_optimal_assignment():
    import itertools

    specification = yaml.safe_load(
        (ROOT / "configs/research/scenario_qualification.yaml").read_text()
    )
    for case in specification["representations"]["row_column_permutation"][
        "cases"
    ].values():
        optima = []
        for record in case["models"][:2]:
            model = record["model"]
            feasible = [
                assignment
                for assignment in itertools.product((0, 1), repeat=2)
                if all(
                    lower
                    <= sum(a * x for a, x in zip(row, assignment, strict=True))
                    <= upper
                    for row, lower, upper in zip(
                        model["matrix"],
                        model["row_lower"],
                        model["row_upper"],
                        strict=True,
                    )
                )
            ]

            def objective(x):
                return sum(a * b for a, b in zip(model["cost"], x, strict=True))

            best = min(objective(assignment) for assignment in feasible)
            assert best == record["target"]
            optima.append(
                {assignment for assignment in feasible if objective(assignment) == best}
            )
        assert not optima[0] & optima[1]


@pytest.mark.parametrize("ignored_parameter", [False, True])
def test_population_construction_checks_real_shifts_and_retains_failures(
    tmp_path, ignored_parameter
):
    from pitbench.research import construct_generalization_populations

    specification = {
        "roles": {
            "development": {"visibility": "agent", "count": 2, "seed_namespace": 11},
            "hidden": {"visibility": "judge", "count": 2, "seed_namespace": 22},
        },
        "families": {
            "packing": {
                "generator": {
                    "kind": "multidimensional_knapsack_mip",
                    "items": [8],
                    "dimensions": [2],
                    "weight_max": 20,
                    "randomness": {"instance_seed": 0},
                }
            }
        },
        "conditions": [
            {"id": "baseline", "family": "packing", "roles": ["development", "hidden"]},
            {
                "id": "sparse",
                "family": "packing",
                "roles": ["development", "hidden"],
                "overrides": {
                    "density" if ignored_parameter else "nonzero_density": 0.5
                },
                "expected_features": {"nonzero_density": 0.5},
            },
        ],
    }
    path = tmp_path / "specification.yaml"
    path.write_text(yaml.safe_dump(specification))
    output = tmp_path / "populations"
    if ignored_parameter:
        with pytest.raises(ValueError, match="not realized"):
            construct_generalization_populations(
                specification_path=path, destination=output
            )
        assert (
            json.loads((output / "population-construction.json").read_text())["status"]
            == "failed"
        )
        return
    result = construct_generalization_populations(
        specification_path=path, destination=output
    )
    assert result["instances"] == 8
    assert result["new_agent_calls"] == result["new_solver_runs"] == 0
    development = json.loads((output / "development/manifest.json").read_text())[
        "instances"
    ]
    hidden = json.loads((output / "hidden/manifest.json").read_text())["instances"]
    assert not {r["semantic_sha256"] for r in development} & {
        r["semantic_sha256"] for r in hidden
    }
    assert all(
        r["features"]["nonzero_density"] == 0.5
        for r in hidden
        if r["condition_id"] == "sparse"
    )


def test_headroom_preparation_keeps_budgets_seeds_and_replayable_private_inputs(
    tmp_path, monkeypatch
):
    from pitbench.evaluator.judge import LocalProcessJudge
    from pitbench.research import (
        construct_generalization_populations,
        measure_population_headroom,
        prepare_population_references,
    )

    spec = {
        "roles": {
            "protocol_calibration": {
                "visibility": "judge",
                "count": 2,
                "seed_namespace": 99,
            }
        },
        "families": {
            "packing": {
                "generator": {
                    "kind": "multidimensional_knapsack_mip",
                    "items": [8],
                    "dimensions": [2],
                    "weight_max": 20,
                    "randomness": {"instance_seed": 0},
                }
            }
        },
        "conditions": [
            {
                "id": "baseline",
                "family": "packing",
                "roles": ["protocol_calibration"],
                "axes": {"axis": "baseline", "coordinate": None},
            }
        ],
    }
    specification = tmp_path / "generation.yaml"
    specification.write_text(yaml.safe_dump(spec))
    populations = tmp_path / "populations"
    construct_generalization_populations(
        specification_path=specification, destination=populations
    )
    task_path = ROOT / "configs/tasks/release_cbc_v2_10_12.yaml"
    original = PitBenchTask.from_yaml(task_path)
    private = tmp_path / "private"
    refs = private / "references"
    prepare_population_references(
        population_root=populations,
        task_path=task_path,
        private_root=private,
        destination=refs,
    )
    patch = tmp_path / "reference.patch"
    patch.write_text("reference source patch")

    def inspect_before_solve(judge, cases):
        assert judge.task.evaluation.budgets_sec == original.evaluation.budgets_sec
        assert list(judge.evaluation_seeds) == original.evaluation.solver_seeds
        for case in cases:
            assert case.verifier._check_oracle(case.path) is None
            assert case.path.is_relative_to(private)
        for panel in judge.task.instance_sets:
            if panel.kind.value != "agent_dev":
                assert (
                    private / panel.instance_set_config.removeprefix("private://")
                ).exists()
        raise RuntimeError("stop after checking the real preparation boundary")

    monkeypatch.setattr(LocalProcessJudge, "run", inspect_before_solve)
    output = tmp_path / "headroom"
    with pytest.raises(RuntimeError, match="real preparation boundary"):
        measure_population_headroom(
            task_path=task_path,
            base_repository=tmp_path / "unused-source",
            public_root=ROOT,
            private_root=private,
            population_root=populations,
            references_path=refs / "references.json",
            candidate_patch=patch,
            destination=output,
        )
    assert json.loads((output / "headroom.json").read_text())["status"] == "failed"
