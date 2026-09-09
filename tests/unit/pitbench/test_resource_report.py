from __future__ import annotations

import json
import math
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pitbench.evaluator.storage import ObservationStore
from pitbench.metrics.resource_report import (
    compute_resource_reports,
    format_resource_report,
)
from pitbench.repositories.base import NormalizedSolverOutput
from pitbench.schema.observation import CodeState, RunObservation, RunStatus
from pitbench.solver_drivers.common import process_resources, write_result


def observation(state, instance, seed, memory, **kwargs):
    values = dict(
        task_id="resource-test",
        code_state=state,
        instance_set="judge_id",
        instance_id=instance,
        solver_seed=seed,
        budget_sec=5,
        status=RunStatus.COMPLETED,
        valid=True,
        peak_rss_bytes=memory,
        cpu_time_sec=2.0,
        normalized_gap=0.1,
        resource_scope="worker_process_start_through_solver_return",
    )
    values.update(kwargs)
    return RunObservation(**values)


def test_ratios_use_thirty_paired_seed_means_then_equal_instance_geometric_mean():
    observations = []
    for seed in range(30):
        observations.extend(
            [
                observation(CodeState.BASE, "small", seed, 1 if seed % 2 else 9),
                observation(CodeState.AGENT, "small", seed, 1, normalized_gap=0.3),
                observation(CodeState.BASE, "large", seed, 100),
                observation(CodeState.AGENT, "large", seed, 500),
            ]
        )
    report, details = compute_resource_reports(observations[::-1], primary_budget_sec=5)
    instances = {
        item.instance_id: item for item in details.by_instance_set["judge_id"]["5"]
    }
    # Ratio of seed means is 1/5, not mean(1/9, 1/1); larger absolute
    # memory instances do not dominate the cross-instance relative ratio.
    assert instances["small"].memory.agent_base_ratio == pytest.approx(0.2)
    assert instances["large"].memory.agent_base_ratio == pytest.approx(5)
    aggregate = report.by_instance_set["judge_id"]["5"]
    assert aggregate.memory.geometric_mean_ratio == pytest.approx(1)
    assert aggregate.memory.complete_instance_count == 2
    assert aggregate.memory.paired_seed_count == 60
    assert instances["small"].mean_gap_change == pytest.approx(0.2)
    assert "small" not in report.model_dump_json()
    assert "30" in format_resource_report(report)

    swapped = [
        item.model_copy(
            update={
                "code_state": CodeState.AGENT
                if item.code_state == CodeState.BASE
                else CodeState.BASE
            }
        )
        for item in observations
    ]
    _, swapped_details = compute_resource_reports(swapped, primary_budget_sec=5)
    for item in swapped_details.by_instance_set["judge_id"]["5"]:
        assert item.memory.agent_base_ratio == pytest.approx(
            1 / instances[item.instance_id].memory.agent_base_ratio
        )


def test_missing_and_failed_runs_preserve_pairing_and_quality_regressions():
    observations = [
        observation(CodeState.BASE, "one", 0, 100),
        observation(
            CodeState.AGENT,
            "one",
            0,
            80,
            normalized_gap=0.4,
            solver_status="Time limit reached",
        ),
        observation(CodeState.BASE, "one", 1, 10000),  # unpaired, excluded
        observation(CodeState.BASE, "one", 2, 100),
        observation(
            CodeState.AGENT, "one", 2, 1, status=RunStatus.CRASHED, valid=False
        ),
        observation(CodeState.BASE, "one", 3, 100),
        observation(CodeState.AGENT, "one", 3, None),
    ]
    report, details = compute_resource_reports(
        observations, primary_budget_sec=5, budgets_sec=[5, 10]
    )
    instance = details.by_instance_set["judge_id"]["5"][0]
    assert instance.memory.base_mean == 100
    assert instance.memory.agent_base_ratio == 0.8
    assert instance.memory.paired_seed_count == 1
    assert instance.memory.excluded_pairs == {
        "missing_both": 26,
        "missing_agent": 1,
        "invalid_agent": 1,
        "missing_or_nonpositive_measurement": 1,
    }
    assert instance.agent_status_counts["crashed"] == 1
    assert instance.agent_status_counts["missing"] == 27
    assert instance.paired_quality_seed_count == 2
    assert instance.cpu.paired_seed_count == 2
    assert report.by_instance_set["judge_id"]["5"].memory.complete_instance_count == 0
    assert report.by_instance_set["judge_id"]["10"].memory.geometric_mean_ratio is None
    assert details.by_instance_set["judge_id"]["10"][0].missing_seed_count == 30


@pytest.mark.parametrize(
    "update,reason",
    [
        ({"resource_scope": None}, "unqualified_measurement_scope"),
        (
            {"resource_scope": "legacy_driver_and_children_at_output"},
            "unqualified_measurement_scope",
        ),
        (
            {"resource_scope": "single_native_child_lifetime"},
            "measurement_scope_mismatch",
        ),
        ({"threads": 2}, "thread_count_mismatch"),
        ({"peak_rss_bytes": 0}, "missing_or_nonpositive_measurement"),
        ({"peak_rss_bytes": -1}, "missing_or_nonpositive_measurement"),
    ],
)
def test_incompatible_or_invalid_resource_values_are_not_savings(update, reason):
    values = [
        observation(CodeState.BASE, "one", 0, 100),
        observation(CodeState.AGENT, "one", 0, 80, **update),
    ]
    report, details = compute_resource_reports(values, primary_budget_sec=5)
    assert report.by_instance_set["judge_id"]["5"].memory.geometric_mean_ratio is None
    assert (
        details.by_instance_set["judge_id"]["5"][0].memory.excluded_pairs[reason] == 1
    )


def test_cpu_nonfinite_values_do_not_break_memory_report():
    values = [
        observation(CodeState.BASE, "one", 0, 100),
        observation(CodeState.AGENT, "one", 0, 80, cpu_time_sec=math.nan),
    ]
    report, details = compute_resource_reports(values, primary_budget_sec=5)
    assert report.by_instance_set["judge_id"][
        "5"
    ].memory.geometric_mean_ratio == pytest.approx(0.8)
    assert details.by_instance_set["judge_id"]["5"][0].cpu.agent_base_ratio is None
    json.loads(report.model_dump_json())


def test_budgets_instance_sets_and_transformations_do_not_mix():
    values = []
    for instance_set, budget, ratio in [
        ("judge_id", 5, 0.5),
        ("judge_id", 10, 2),
        ("agent_dev", 5, 3),
    ]:
        values.extend(
            [
                observation(
                    CodeState.BASE,
                    "one",
                    0,
                    100,
                    instance_set=instance_set,
                    budget_sec=budget,
                ),
                observation(
                    CodeState.AGENT,
                    "one",
                    0,
                    int(100 * ratio),
                    instance_set=instance_set,
                    budget_sec=budget,
                ),
            ]
        )
    original, _ = compute_resource_reports(values, primary_budget_sec=5)
    values.append(
        observation(CodeState.AGENT, "transformed", 0, 1, equivalence_parent_id="one")
    )
    report, _ = compute_resource_reports(values, primary_budget_sec=5)
    assert report == original
    assert report.by_instance_set["judge_id"][
        "5"
    ].memory.geometric_mean_ratio == pytest.approx(0.5)
    assert report.by_instance_set["judge_id"][
        "10"
    ].memory.geometric_mean_ratio == pytest.approx(2)
    assert report.by_instance_set["agent_dev"][
        "5"
    ].memory.geometric_mean_ratio == pytest.approx(3)


def test_duplicate_or_mixed_task_records_are_rejected():
    record = observation(CodeState.BASE, "one", 0, 100)
    with pytest.raises(ValueError, match="duplicate"):
        compute_resource_reports([record, record], primary_budget_sec=5)
    with pytest.raises(ValueError, match="exactly one task"):
        compute_resource_reports(
            [record, record.model_copy(update={"task_id": "other"})],
            primary_budget_sec=5,
        )


def test_entire_missing_instances_and_instance_sets_remain_in_coverage():
    values = [
        observation(state, "present", seed, 100)
        for seed in range(30)
        for state in CodeState
    ]
    report, _ = compute_resource_reports(
        values,
        primary_budget_sec=5,
        expected_instance_counts={"judge_id": 3, "agent_dev": 10},
    )
    cell = report.by_instance_set["judge_id"]["5"]
    assert cell.instance_count == 1
    assert cell.expected_instance_count == 3
    assert cell.memory.complete_instance_count == 1
    assert report.by_instance_set["agent_dev"]["5"].memory.geometric_mean_ratio is None
    assert "memory pairs 30/90" in format_resource_report(report)


def test_native_driver_records_child_lifetime_peak(tmp_path):
    solver = tmp_path / "solver"
    solver.write_text(
        f"#!{sys.executable}\n"
        "import sys\nfrom pathlib import Path\n"
        "data = bytearray(32 * 1024 * 1024)\n"
        "for index in range(0, len(data), 4096): data[index] = 1\n"
        "for argument in sys.argv:\n"
        "    if argument.startswith('--solution_file='): Path(argument.split('=', 1)[1]).write_text('solution')\n"
        "print('Model status : Optimal\\nPrimal bound 1\\nDual bound 1\\nNodes 1')\n"
    )
    solver.chmod(0o755)
    output = tmp_path / "result.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pitbench.solver_drivers.run",
            "highs",
            "--solver",
            str(solver),
            "--instance",
            str(tmp_path / "instance.mps"),
            "--output",
            str(output),
            "--trajectory",
            str(tmp_path / "trajectory.jsonl"),
            "--seed",
            "0",
            "--budget",
            "1",
            "--threads",
            "1",
        ],
        check=True,
    )
    result = json.loads(output.read_text())
    assert result["resource_scope"] == "single_native_child_lifetime"
    assert result["peak_rss_bytes"] >= 32 * 1024 * 1024
    assert result["cpu_time_sec"] > 0
    assert result["solver_status"] == "Optimal"


def test_snapshots_keep_single_process_peak_and_freeze_before_output(tmp_path):
    own = SimpleNamespace(ru_utime=2, ru_stime=3, ru_maxrss=100)
    child = SimpleNamespace(ru_utime=7, ru_stime=11, ru_maxrss=500)
    with patch(
        "pitbench.solver_drivers.common.resource.getrusage", side_effect=[own, child]
    ):
        snapshot = process_resources()
        child_snapshot = process_resources(child_process=True)
    assert snapshot["peak_rss_bytes"] == 100 * 1024
    assert snapshot["cpu_time_sec"] == 5
    assert child_snapshot["peak_rss_bytes"] == 500 * 1024
    assert child_snapshot["cpu_time_sec"] == 18
    with patch("pitbench.solver_drivers.common.resource.getrusage", return_value=child):
        write_result(tmp_path / "result.json", started=0, valid=True, **snapshot)
    parsed = NormalizedSolverOutput.model_validate_json(
        (tmp_path / "result.json").read_text()
    )
    assert parsed.peak_rss_bytes == snapshot["peak_rss_bytes"]
    record = observation(
        CodeState.BASE,
        "one",
        0,
        parsed.peak_rss_bytes,
        resource_scope=parsed.resource_scope,
    )
    ObservationStore.write(tmp_path / "observations.parquet", [record])
    assert ObservationStore.read(tmp_path / "observations.parquet") == [record]
