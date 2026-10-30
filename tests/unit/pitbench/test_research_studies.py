from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from pitbench.research import prepare_generalization_study
from pitbench.schema.research import GeneralizationStudyPlan
from pitbench.schema.task import PitBenchTask

ROOT = Path(__file__).resolve().parents[3]
PLAN = ROOT / "configs/research/generalization.json"


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
