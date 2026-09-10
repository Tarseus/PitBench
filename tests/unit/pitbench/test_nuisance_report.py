import json

import pytest

from pitbench.metrics.nuisance_report import report_nuisance_results
from pitbench.schema.observation import CodeState, RunObservation, RunStatus


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def directory_panel(source):
    jobs = [
        {
            "run_id": f"case/seed/{seed}/budget-{budget}",
            "instance": "case",
            "axis": "seed",
            "replicate": seed,
            "solver_seed": seed,
            "budget_sec": budget,
        }
        for budget in [5, 15]
        for seed in [0, 1]
    ]
    write(
        source / "experiment.json",
        {
            "experiment": "Another solver <release>",
            "code_state": "base",
            "instances": [{"name": "case"}],
            "budgets_sec": [5, 15],
            "jobs": jobs,
        },
    )
    write(
        source / "runs" / jobs[0]["run_id"] / "result.json",
        {
            **jobs[0],
            "execution_status": "completed",
            "verified_feasible": True,
            "verification": {"feasible": True, "objective": 0},
            "solver_runtime_sec": 3,
            "solution_value_valid": True,
        },
    )
    return jobs


def embedded(output):
    text = (output / "report.html").read_text()
    return json.loads(text.split("const DATA=", 1)[1].split(";\nconst names=", 1)[0])


def test_directory_report_uses_manifest_counts_budgets_and_identity(tmp_path):
    source, output = tmp_path / "source", tmp_path / "report"
    directory_panel(source)
    summary = report_nuisance_results(source, output)
    assert summary["expected_runs"] == 4
    assert summary["completed_runs"] == 1
    assert summary["missing_runs"] == 3
    assert not summary["complete"]
    assert [(item["expected"], item["completed"]) for item in summary["groups"]] == [
        (2, 1),
        (2, 0),
    ]
    assert summary["groups"][0]["verified_objective"] == {
        "count": 1,
        "min": 0,
        "max": 0,
    }
    assert summary["groups"][1]["verified_objective"] is None
    data = embedded(output)
    assert data["manifest"]["budgets_sec"] == [5, 15]
    assert data["records"][0]["verification"]["objective"] == 0
    page = (output / "report.html").read_text()
    assert "Another solver &lt;release&gt;" in page
    assert "HiGHS" not in page and "2026-09-09" not in page
    assert "（30 次）" not in page and "/29*" not in page


def test_directory_report_rejects_wrong_run_in_result_file(tmp_path):
    source = tmp_path / "source"
    jobs = directory_panel(source)
    path = source / "runs" / jobs[0]["run_id"] / "result.json"
    record = json.loads(path.read_text())
    record["solver_seed"] = 999
    write(path, record)
    with pytest.raises(ValueError, match="outside the declared run grid"):
        report_nuisance_results(source, tmp_path / "report")


@pytest.mark.parametrize("manifest_name", ["experiment.json", "details.json"])
def test_judge_report_separates_states_and_preserves_failure(tmp_path, manifest_name):
    source, output = tmp_path / "source", tmp_path / "report"
    write(
        source / manifest_name,
        {
            "task_id": "a_configured_task",
            "code_states": ["base", "agent"],
            "budgets_sec": [5, 10],
            "solver_seed": 9,
            "configuration": {"solver_seed": 9},
        },
    )
    write(
        source / "transformations.json",
        {
            "case__transform": {
                "original_instance_id": "case",
                "transform_id": "permutation_0",
            },
        },
    )
    records = []
    for state in CodeState:
        valid = state is CodeState.BASE
        observation = RunObservation(
            task_id="a_configured_task",
            code_state=state,
            instance_set="agent_dev",
            instance_set_kind="agent_dev",
            instance_id="case__transform",
            solver_seed=9,
            budget_sec=5,
            status=RunStatus.COMPLETED if valid else RunStatus.CRASHED,
            valid=valid,
            objective=4 if valid else None,
        )
        records.append(
            {
                "observation": observation.model_dump(mode="json"),
                "verification": {
                    "mapped_original": {
                        "feasible": valid,
                        "objective": 4 if valid else None,
                    },
                    "objective_preserved": valid,
                },
                "artifacts": {"solution": "solution.json" if valid else None},
            }
        )
    checkpoint = source / "results.jsonl"
    checkpoint.write_text("".join(json.dumps(item) + "\n" for item in records))
    original = checkpoint.read_bytes()
    summary = report_nuisance_results(source, output)
    assert summary["expected_runs"] == 4 and summary["completed_runs"] == 2
    assert summary["execution_statuses"] == {"completed": 1, "crashed": 1}
    groups = {
        (item["code_state"], item["budget_sec"]): item for item in summary["groups"]
    }
    assert groups[("base", 5)]["verified_feasible"] == 1
    assert groups[("agent", 5)]["verified_objective"] is None
    assert groups[("base", 10)]["missing"] == 1
    assert groups[("base", 5)]["solver_gap"] is None
    assert len(embedded(output)["records"]) == 2
    assert checkpoint.read_bytes() == original
    checkpoint.write_text(original.decode() + json.dumps(records[0]) + "\n")
    with pytest.raises(ValueError, match="duplicate nuisance observation"):
        report_nuisance_results(source, output)
