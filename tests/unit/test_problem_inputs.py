"""Functional unit/component contracts consolidated by responsibility."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from pitbench.agent_cli import _public_verification
from pitbench.evaluator.scenarios import task_digest
from pitbench.instances.generate import (
    binary_clique_cover_data,
    binary_clique_cover_fme_extensions,
    materialize_generated_instance_set,
)
from pitbench.metrics.objective_native import native_objective_measurement
from pitbench.metrics.objective_qualification import compute_objective_qualification
from pitbench.metrics.scenario_report import compute_scenario_report
from pitbench.problem_families.verification import (
    CPFamily,
    MIPFamily,
    TrustedOptimumFamily,
    TrustedOptimumOracle,
    TrustedOptimumRecord,
    _public_multidimensional_knapsack_data,
)
from pitbench.schema.objectives import (
    qualification_protocols,
    qualification_protocols_sha256,
)
from pitbench.schema.observation import (
    CodeState,
    ExpectedRun,
    ExpectedRunGrid,
    RunObservation,
    RunStatus,
    SolverTermination,
)
from pitbench.schema.scenarios import SCENARIO_METRICS
from pitbench.schema.task import PitBenchTask
from pitbench.tasks import TaskCatalog
from tests.support import ROOT, campaign


def task_for(metric, protocols=(), **plan_changes):
    payload = yaml.safe_load((ROOT / "configs/tasks/highs_v1_15_1.yaml").read_text())
    payload["task_id"] = "scenario-test"
    hidden = dict(next(s for s in payload["instance_sets"] if s["name"] == "judge_id"))
    hidden.update(name="judge_shift", kind="judge_shift")
    payload["instance_sets"].append(hidden)
    payload["evaluation"].update(
        budgets_sec=[10],
        primary_budget_sec=10,
        solver_seeds=[0, 1],
        seed_robustness=None,
        representation_robustness=None,
        operational_reliability=False,
    )
    for panel in payload["instance_sets"]:
        panel["size"] = 2
    objective = next(
        k
        for k, v in qualification_protocols("1.1")["objectives"].items()
        if metric in v["metric_ids"]
    )
    plan = dict(
        protocol_version="1.1",
        protocols_sha256=qualification_protocols_sha256("1.1"),
        qualification_instance_sets=["judge_id"],
        reporting_instance_sets=["judge_shift"],
        scenario_protocols_sha256={p.id: p.digest() for p in protocols},
        guardrails=[
            dict(
                metric_id=m, comparison=c, max_degradation=0, instance_sets=["judge_id"]
            )
            for m, c in [
                ("exact_solved_coverage", "absolute"),
                ("exact_penalized_runtime_sec", "absolute"),
                ("peak_rss_ratio", "relative"),
            ]
        ],
    )
    plan.update(plan_changes)
    if metric in {"boundary_pass_counts", "operational_failure_counts"}:
        plan["reliability_scope"] = "declared_boundary_suite"
        payload["evaluation"]["operational_reliability"] = True
    payload["metric_scenarios"] = [p.model_dump(mode="json") for p in protocols]
    payload["optimization_profile"] = dict(
        objectives=[objective], primary_metric_id=metric, scoring=plan
    )
    return PitBenchTask.model_validate(payload)


def normal_evidence(task, seeds=(0, 1)):
    rows = [
        RunObservation(
            task_id=task.task_id,
            code_state=state,
            instance_set="judge_id",
            instance_set_kind="judge_id",
            instance_id=iid,
            solver_seed=seed,
            budget_sec=10,
            status=RunStatus.COMPLETED,
            valid=True,
            objective=1,
            reported_objective=1,
            optimal_or_bks=1,
            normalized_gap=0,
            cpu_time_sec=1,
            wall_time_sec=1,
            peak_rss_bytes=1000,
            resource_scope="worker_process_start_through_solver_return",
            solver_termination=SolverTermination.OPTIMAL,
        )
        for state in CodeState
        for iid in ("a", "b")
        for seed in seeds
    ]
    grid = ExpectedRunGrid(
        task_id=task.task_id,
        runs=[
            ExpectedRun(**{k: getattr(row, k) for k in ExpectedRun.model_fields})
            for row in rows
        ],
    )
    return rows, grid


def fixture(tmp_path, kind="setup", metric="setup_time_sec"):
    data = campaign(tmp_path, kind)
    task = task_for(metric, data.protocols)
    data.task_sha256 = task_digest(task)
    return task, data, *normal_evidence(task)


def test_protocols_preserve_adapter_coverage_with_single_pass_evaluation():
    old, new = qualification_protocols(), qualification_protocols("1.1")
    assert old["protocol_version"] == "1.0"
    assert "confirmation" not in old["decision_rules"]
    assert "confirmation" not in new["scenario_protocol"]
    assert old["metrics"]["setup_time_sec"]["status"] == "requires_scenario_protocol"
    assert sum(v["status"] == "implemented" for v in new["metrics"].values()) == 25
    assert new["metrics"]["search_work"]["status"] == "diagnostic_only"
    assert {m for group in SCENARIO_METRICS.values() for m in group} == {
        m for m, v in new["metrics"].items() if v["adapter"] == "declared_scenario"
    }


def test_scenario_primary_uses_one_complete_independently_verified_campaign(tmp_path):
    task, data, rows, grid = fixture(tmp_path)
    result = compute_objective_qualification(
        task=task,
        observations=rows,
        expected_run_grid=grid,
        scenario_report=compute_scenario_report(data),
        candidate_sha256=data.candidate_sha256,
        validity_accepted=True,
    )
    assert result.classification == "improved"
    assert result.initial.primary[0].upper < 0
    assert result.performance_confirmed
    assert "repeat" not in result.model_dump()


def test_budget_capped_reanalysis_does_not_replace_frozen_scaling_qualification(
    tmp_path,
):
    task, data, rows, grid = fixture(tmp_path, "scaling", "parallel_efficiency")
    result = compute_objective_qualification(
        task=task,
        observations=rows,
        expected_run_grid=grid,
        scenario_report=compute_scenario_report(
            data, scaling_time_policy="budget_capped_v1"
        ),
        candidate_sha256=data.candidate_sha256,
        validity_accepted=True,
    )
    assert not result.performance_confirmed
    assert not result.initial.primary[0].complete
    assert "reanalysis" in result.initial.primary[0].detail


def test_reported_bounds_do_not_become_independently_certified_qualification(tmp_path):
    from tests.support import reported_traces

    task, data, rows, grid = fixture(tmp_path, "trajectory", "dual_progress_integral")
    result = compute_objective_qualification(
        task=task,
        observations=rows,
        expected_run_grid=grid,
        scenario_report=compute_scenario_report(
            data,
            dual_bound_source="solver_reported",
            reported_bound_traces=reported_traces(data),
        ),
        candidate_sha256=data.candidate_sha256,
        validity_accepted=True,
    )
    assert not result.performance_confirmed
    assert "diagnostic" in result.initial.primary[0].detail


@pytest.mark.parametrize("validity", [False, None])
def test_single_pass_still_requires_independent_validity(tmp_path, validity):
    task, data, rows, grid = fixture(tmp_path)
    result = compute_objective_qualification(
        task=task,
        observations=rows,
        expected_run_grid=grid,
        scenario_report=compute_scenario_report(data),
        candidate_sha256=data.candidate_sha256,
        validity_accepted=validity,
    )
    assert result.classification == ("invalid" if validity is False else "incomplete")
    assert not result.performance_confirmed


def test_missing_or_wrong_candidate_scenario_cannot_be_scored(tmp_path):
    task, data, rows, grid = fixture(tmp_path)
    result = compute_objective_qualification(
        task=task,
        observations=rows,
        expected_run_grid=grid,
        scenario_report=compute_scenario_report(data),
        candidate_sha256="f" * 64,
    )
    assert result.classification == "incomplete"


def test_faster_setup_with_worse_final_quality_fails_scenario_guardrail(tmp_path):
    task, data, rows, grid = fixture(tmp_path)
    next(r for r in data.runs if r.key.code_state == "agent").steps[
        0
    ].final_solution_values = [2, 0]
    report = compute_scenario_report(data)
    assert report.complete and not report.invalid_runs
    result = compute_objective_qualification(
        task=task,
        observations=rows,
        expected_run_grid=grid,
        scenario_report=report,
        candidate_sha256=data.candidate_sha256,
    )
    assert (
        result.classification == "guardrail_failed" and not result.performance_confirmed
    )


def test_invalid_intermediate_solution_disqualifies_scenario_primary(tmp_path):
    task, data, rows, grid = fixture(
        tmp_path, "trajectory", "time_to_first_verified_solution_sec"
    )
    next(r for r in data.runs if r.key.code_state == "agent").steps[0].events[
        0
    ].solution_values = [0, 0]
    result = compute_objective_qualification(
        task=task,
        observations=rows,
        expected_run_grid=grid,
        scenario_report=compute_scenario_report(data),
        candidate_sha256=data.candidate_sha256,
    )
    assert result.classification == "invalid"


@pytest.mark.parametrize(
    "metric", ["boundary_pass_counts", "operational_failure_counts"]
)
def test_native_reliability_uses_declared_complete_boundary_pairs(metric):
    task = task_for(metric)
    rows, grid = normal_evidence(task)
    for state in CodeState:
        for iid in ("boundary-a", "boundary-b"):
            row = rows[0].model_copy(
                update=dict(
                    code_state=state,
                    instance_id=iid,
                    solver_seed=0,
                    test_suite="operational_reliability",
                    instance_set="boundary_cases",
                    status=RunStatus.CRASHED
                    if state == CodeState.BASE
                    else RunStatus.COMPLETED,
                    valid=state == CodeState.AGENT,
                )
            )
            rows.append(row)
            grid.runs.append(
                ExpectedRun(**{k: getattr(row, k) for k in ExpectedRun.model_fields})
            )
    value = native_objective_measurement(task, rows, grid, metric, "judge_id")
    assert value["complete"] and value["upper"] < 0
    assert not native_objective_measurement(
        task, rows[:-1], grid, metric, "judge_id"
    ).get("complete", False)


def test_native_representation_preserves_type7_iqr_and_expected_grid():
    task = task_for("representation_dispersion")
    payload = task.model_dump(mode="json")
    source = yaml.safe_load((ROOT / "configs/tasks/choco_v6_0_1.yaml").read_text())
    for key in ("release", "repository", "problem_family", "oracle"):
        payload[key] = source[key]
    payload["evaluation"]["representation_robustness"] = dict(
        kind="item_relabeling",
        instance_set="judge_id",
        solver_seed=0,
        relabeling_generation_seed=1,
        relabelings_per_instance=30,
    )
    task = PitBenchTask.model_validate(payload)
    rows, grid = normal_evidence(task)
    for state in CodeState:
        for iid in ("a", "b"):
            for variant in range(30):
                row = rows[0].model_copy(
                    update=dict(
                        code_state=state,
                        instance_id=f"{iid}-{variant}",
                        solver_seed=0,
                        equivalence_parent_id=iid,
                        equivalence_transform="row_column_permutation",
                        cpu_time_sec=1
                        + variant * (0.1 if state == CodeState.BASE else 0.05),
                        wall_time_sec=1
                        + variant * (0.1 if state == CodeState.BASE else 0.05),
                    )
                )
                rows.append(row)
                grid.runs.append(
                    ExpectedRun(
                        **{k: getattr(row, k) for k in ExpectedRun.model_fields}
                    )
                )
    value = native_objective_measurement(
        task, rows, grid, "representation_dispersion", "judge_id"
    )
    assert value["complete"] and value["degradation"] < 0
    assert not native_objective_measurement(
        task, rows[:-1], grid, "representation_dispersion", "judge_id"
    ).get("complete", False)


def test_native_seed_adapter_retains_existing_99_percent_interval():
    task = task_for("seed_dispersion", seed_dispersion_measure="exact_capped_time_iqr")
    payload = task.model_dump(mode="json")
    payload["evaluation"].update(
        solver_seeds=None,
        seed_robustness=dict(
            development_seeds=[0, 1, 2, 3],
            evaluation_seeds_file="private://fixture/seeds.yaml",
            evaluation_seeds_file_sha256="e" * 64,
            seed_selection=dict(seed_min=0, seed_max=100, seed_count=4),
        ),
    )
    task = PitBenchTask.model_validate(payload)
    rows, grid = normal_evidence(task, seeds=(4, 5, 6, 7))
    for row in rows:
        row.cpu_time_sec = (
            (row.solver_seed - 3) if row.code_state == CodeState.BASE else 2.5
        )
    value = native_objective_measurement(
        task, rows, grid, "seed_dispersion", "judge_id"
    )
    assert value["complete"] and value["confidence_level"] == 0.99
    assert value["degradation"] < 0


def test_native_controlled_retention_rejects_a_missing_condition():
    from tests.support import (
        _expected_grid,
        _heuristic_observations,
        _heuristic_task,
    )

    task = _heuristic_task()
    rows = _heuristic_observations(task)
    grid = _expected_grid(task)
    measured = native_objective_measurement(
        task, rows, grid, "controlled_retention", "judge_shift"
    )
    assert measured["complete"] and measured["classification"] == "improved"
    assert not native_objective_measurement(
        task, rows[:-1], grid, "controlled_retention", "judge_shift"
    ).get("complete", False)


def test_cli_recomputes_scenarios_and_qualification_from_raw_campaign(tmp_path):
    from typer.testing import CliRunner

    from pitbench.cli.main import app
    from pitbench.evaluator.storage import ObservationStore

    task, data, rows, grid = fixture(tmp_path)
    task_file = tmp_path / "task.yaml"
    task_file.write_text(yaml.safe_dump(task.model_dump(mode="json")))
    patch = tmp_path / "candidate.patch"
    patch.write_text("")
    data.candidate_sha256 = hashlib.sha256(patch.read_bytes()).hexdigest()
    path = tmp_path / "metric-scenarios"
    path.mkdir()
    (path / "campaign.json").write_text(data.model_dump_json())
    ObservationStore.write(tmp_path / "trials.parquet", rows)
    (tmp_path / "expected-run-grid.json").write_text(grid.model_dump_json())
    result = CliRunner().invoke(
        app, ["report", str(tmp_path), "--task-config", str(task_file), "--json"]
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["metric_scenarios"]["complete"]
    assert (
        report["objective_measurements"]["qualification"]["classification"]
        == "incomplete"
    )
    assert (
        report["objective_measurements"]["metrics"]["setup_time_sec"]["status"]
        == "reported"
    )


def test_evaluator_collects_declared_scenarios_and_invalidates_bad_solution(
    tmp_path, monkeypatch
):
    from pitbench.evaluator.evaluator import PitBenchEvaluator
    from pitbench.harness.evaluation import EvaluationRequest

    task, data, _, _ = fixture(tmp_path)
    task_file = tmp_path / "task.yaml"
    task_file.write_text(yaml.safe_dump(task.model_dump(mode="json")))
    patch = tmp_path / "candidate.patch"
    patch.write_text("")
    data.candidate_sha256 = hashlib.sha256(patch.read_bytes()).hexdigest()
    next(r for r in data.runs if r.key.code_state == "agent").steps[
        0
    ].final_solution_values = [0, 0]
    calls = []

    def collect(selected_task, candidate, output, config, **kwargs):
        calls.append((selected_task.task_id, candidate))
        destination = output / "metric-scenarios"
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / "campaign.json"
        path.write_text(data.model_dump_json())
        return compute_scenario_report(data), path

    monkeypatch.setattr("pitbench.evaluator.scenarios.collect_for_evaluation", collect)
    request = EvaluationRequest(
        task_id=task.task_id,
        task_path=ROOT,
        candidate_patch_path=patch,
        output_dir=tmp_path / "evaluation",
        agent_name="fixture",
        evaluator_config={
            "task_config_path": str(task_file),
            "fixture_mode": True,
            "fixture_instances_per_instance_set": 2,
            "metric_scenario_config": "configured.yaml",
        },
    )
    result = PitBenchEvaluator().evaluate(request)
    assert calls and result.summary.metric_scenarios.invalid_runs
    assert result.artifacts.metric_scenario_campaign is not None
    assert not result.validity.accepted
    assert any("metric scenarios" in check.detail for check in result.validity.checks)


def test_preflight_requires_scenario_configuration_for_declared_task(tmp_path):
    from pitbench.cli.doctor import CheckStatus, _metric_scenario_checks
    from pitbench.cli.evaluate_config import EvaluationConfig

    task, _, _, _ = fixture(tmp_path)
    context = SimpleNamespace(
        task=SimpleNamespace(task=task), config=EvaluationConfig(), repository_root=ROOT
    )
    (check,) = _metric_scenario_checks(context)
    assert check.status == CheckStatus.FAIL


def test_best_found_linear_reference_cannot_be_used_as_an_exact_solve_target(tmp_path):
    import pytest

    from pitbench.evaluator.judge import JudgePlan, _exact_target_verifier
    from pitbench.evaluator.private_assets import PrivateAssetResolver
    from pitbench.schema.task import InstanceSetSpec, PerformanceProtocol, PitBenchTask

    instance = tmp_path / "cover.mps"
    instance.write_text(
        "NAME Cover\nROWS\n N OBJ\n G a\n G b\n G c\nCOLUMNS\n"
        " x0 OBJ 1 a 1\n x0 b 1\n x1 OBJ 1 a 1\n x1 c 1\n x2 OBJ 1 b 1\n x2 c 1\n"
        "RHS\n rhs a 1 b 1\n rhs c 1\nBOUNDS\n BV bd x0\n BV bd x1\n BV bd x2\nENDATA\n"
    )
    solution = tmp_path / "reference.json"
    solution.write_text(
        json.dumps({"values": {"x0": 1, "x1": 1, "x2": 1}, "objective": 3})
    )
    record = TrustedOptimumRecord(
        instance_set="scientific",
        instance_id="cover",
        instance_sha256=hashlib.sha256(instance.read_bytes()).hexdigest(),
        target_kind="best_found",
        problem_kind="linear_mip",
        objective_sense="minimize",
        optimal_objective=3,
        reference_solution_uri="private://reference.json",
        reference_solution_sha256=hashlib.sha256(solution.read_bytes()).hexdigest(),
        optimality_basis={"kind": "feasible_search_reference"},
        generation_provenance={},
        verification_provenance="pitbench.problem_families.verification:TrustedOptimumFamily",
    )
    resolver = PrivateAssetResolver(tmp_path)
    with pytest.raises(ValueError, match="not exact solved coverage"):
        _exact_target_verifier(record, resolver)
    verifier = _exact_target_verifier(
        record,
        resolver,
        performance_protocol=PerformanceProtocol.HEURISTIC_FIXED_BUDGET,
    )
    assert verifier.verify(instance, solution).feasible
    assert (
        _public_verification(
            {"problem_family": "mip"}, instance, tmp_path / "reference.output.json"
        )["verification"]
        == "no_solution"
    )
    public_output = tmp_path / "public.json"
    public_output.with_suffix(".solution.json").write_text(solution.read_text())
    checked = _public_verification({"problem_family": "mip"}, instance, public_output)
    assert checked["verified_valid"] is True
    assert checked["verified_objective"] == 3
    forged = record.model_copy(update={"target_kind": "trusted_optimum"})
    assert TrustedOptimumFamily(forged, solution)._check_oracle(instance) is not None
    task = PitBenchTask.from_yaml(ROOT / "configs/tasks/release_cbc_v2_10_12.yaml")
    oracle_path = tmp_path / "oracle.yaml"
    oracle_path.write_text(
        yaml.safe_dump(
            TrustedOptimumOracle(task_id=task.task_id, records=[record]).model_dump(
                mode="json"
            )
        )
    )
    panel_path = tmp_path / "panel.yaml"
    panel_path.write_text(
        yaml.safe_dump(
            {
                "instances": [
                    {"id": "cover", "uri": "private://cover.mps", "optimal_or_bks": 3}
                ]
            }
        )
    )
    panel = InstanceSetSpec(
        name="scientific",
        kind="judge_shift",
        instance_set_config="private://panel.yaml",
        size=1,
        performance_protocol="heuristic_fixed_budget",
        budgets_sec=[60],
        primary_budget_sec=60,
    )
    task.instance_sets = [task.instance_sets[0], panel]
    task.oracle.source = "private://oracle.yaml"
    task.oracle.source_sha256 = hashlib.sha256(oracle_path.read_bytes()).hexdigest()
    plan = JudgePlan.from_instance_set_configs(
        task, resolver, public_root=ROOT, evaluation_seeds=(1,)
    )
    assert len(plan.cases) == 1
    assert plan.cases[0].budgets_sec == (60,)
    task.instance_sets[-1] = panel.model_copy(update={"performance_protocol": None})
    with pytest.raises(ValueError, match="exact target kind"):
        JudgePlan.from_instance_set_configs(
            task, resolver, public_root=ROOT, evaluation_seeds=(1,)
        )


def test_set_covering_has_an_independent_lower_bound_and_original_model_checks(
    tmp_path,
):
    import itertools

    from pitbench.instances.generate import prepare_trusted_optimum_oracle

    generator = {
        "kind": "set_covering_mip",
        "count": 1,
        "groups": [3],
        "group_sizes": [2],
        "extra_rows": [4],
        "extra_row_density": 0.5,
        "randomness": {"instance_seed": 7103},
    }
    payload = {"schema_version": "1.0", "visibility": "judge", "generator": generator}
    instance = materialize_generated_instance_set(
        payload, tmp_path / "instances", expected_visibility="judge"
    )[0]
    config = tmp_path / "generator.yaml"
    config.write_text(yaml.safe_dump(payload))
    private = tmp_path / "private"
    oracle = prepare_trusted_optimum_oracle(
        task_config_path=ROOT / "configs/tasks/release_cbc_v2_10_12.yaml",
        instance_set_name="set_covering",
        instance_set_config_path=config,
        private_root=private,
        output_path=private / "oracle.yaml",
        reference_solution_dir=private / "solutions",
        materialized_instances=[instance],
    )
    record = TrustedOptimumRecord.model_validate(oracle["records"][0])
    reference = private / record.reference_solution_uri.removeprefix("private://")
    verifier = TrustedOptimumFamily(record, reference)
    assert record.problem_kind == "set_covering"
    assert record.optimal_objective == 3
    assert verifier._check_oracle(instance) is None
    assert verifier.verify(instance, reference).feasible
    objectives = []
    for assignment in itertools.product((0, 1), repeat=6):
        solution = {
            "values": {f"x{i}": value for i, value in enumerate(assignment)},
            "objective": sum(assignment),
        }
        checked = MIPFamily._verify_numeric_model_solution(instance, solution)
        if checked.feasible:
            objectives.append(checked.objective)
    assert min(objectives) == record.optimal_objective
    invalid = tmp_path / "fractional.json"
    invalid.write_text(
        json.dumps({"values": {f"x{i}": 0.5 for i in range(6)}, "objective": 3})
    )
    assert not verifier.verify(instance, invalid).feasible
    fake = record.model_copy(deep=True)
    fake.optimality_basis["groups"][1][0] = fake.optimality_basis["groups"][0][0]
    assert TrustedOptimumFamily(fake, reference)._check_oracle(instance) is not None
    fake = record.model_copy(update={"optimal_objective": 2})
    assert TrustedOptimumFamily(fake, reference)._check_oracle(instance) is not None


def test_sparse_weighted_knapsack_proof_is_checked_against_original_coefficients(
    tmp_path,
):
    from pitbench.instances.generate import (
        _knapsack_optimum_basis,
        multidimensional_knapsack_data,
    )

    generator = {
        "kind": "multidimensional_knapsack_mip",
        "count": 1,
        "items": [14],
        "dimensions": [3],
        "weight_max": 20,
        "nonzero_density": 0.4,
        "selected_fraction": 0.3,
        "profit_mode": "weighted_sum",
        "profit_row_weight_base": 3,
        "randomness": {"instance_seed": 9831},
    }
    instance = materialize_generated_instance_set(
        {"visibility": "judge", "generator": generator},
        tmp_path / "instances",
        expected_visibility="judge",
        stem_prefix="judge_id",
    )[0]
    weights, capacities, selected = multidimensional_knapsack_data(generator, 0)
    target, basis = _knapsack_optimum_basis(generator, weights, capacities)
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"selected_items": selected}))
    record = TrustedOptimumRecord(
        instance_set="judge_id",
        instance_id=instance.stem,
        instance_sha256=hashlib.sha256(instance.read_bytes()).hexdigest(),
        problem_kind="multidimensional_knapsack",
        objective_sense="minimize",
        optimal_objective=target,
        reference_solution_uri="private://reference.json",
        reference_solution_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        optimality_basis=basis,
        generation_provenance={},
        verification_provenance="pitbench.problem_families.verification:TrustedOptimumFamily",
    )
    verifier = TrustedOptimumFamily(record, reference)
    assert verifier._check_oracle(instance) is None
    assert len(selected) == 4
    solution = tmp_path / "candidate.json"
    solution.write_text(
        json.dumps(
            {
                "values": {f"x{i}": int(i in selected) for i in range(14)},
                "objective": target,
            }
        )
    )
    assert verifier.verify(instance, solution).feasible
    bad = record.model_copy(deep=True)
    bad.optimality_basis["row_multipliers"][0] += 1
    assert TrustedOptimumFamily(bad, reference)._check_oracle(instance) is not None


def test_binary_clique_cover_mip_has_a_hash_bound_optimality_proof(
    tmp_path: Path,
) -> None:
    generator = {
        "kind": "binary_clique_cover_mip",
        "count": 1,
        "groups": [4],
        "group_sizes": [5],
        "coefficient_min": 120,
        "coefficient_max": 190,
        "cross_edge_density": [0.1],
        "randomness": {"instance_seed": 910000},
    }
    instance_config = {
        "schema_version": "1.0",
        "visibility": "judge",
        "format": "mps",
        "generator": generator,
    }
    instance = materialize_generated_instance_set(
        instance_config,
        tmp_path / "instances",
        expected_visibility="judge",
        stem_prefix="judge_id",
    )[0]
    groups, planted, _ = binary_clique_cover_data(generator, 0)
    values = {
        name: int(offset == planted[group_index])
        for group_index, group in enumerate(groups)
        for offset, name in enumerate(group)
    }
    reference = {"values": values, "objective": -len(groups)}
    reference_path = tmp_path / "reference.solution.json"
    reference_path.write_text(json.dumps(reference))
    record = TrustedOptimumRecord(
        instance_set="judge_id",
        instance_id=instance.stem,
        instance_sha256=hashlib.sha256(instance.read_bytes()).hexdigest(),
        problem_kind="binary_clique_cover",
        objective_sense="minimize",
        optimal_objective=-len(groups),
        reference_solution_uri="private://reference.solution.json",
        reference_solution_sha256=hashlib.sha256(
            reference_path.read_bytes()
        ).hexdigest(),
        optimality_basis={"kind": "binary_clique_cover", "groups": groups},
        generation_provenance={},
        verification_provenance=(
            "pitbench.problem_families.verification:TrustedOptimumFamily"
        ),
    )
    verifier = TrustedOptimumFamily(record, reference_path)

    assert verifier._check_oracle(instance) is None
    assert verifier.verify(instance, reference_path).objective == -len(groups)

    invalid_groups = [*groups[:-1], groups[-1][:-1]]
    invalid_record = record.model_copy(
        update={
            "optimality_basis": {
                "kind": "binary_clique_cover",
                "groups": invalid_groups,
            }
        }
    )
    assert (
        TrustedOptimumFamily(invalid_record, reference_path)
        .verify(instance, reference_path)
        .infrastructure_error
    )

    invalid_solution = dict(reference)
    invalid_values = dict(values)
    invalid_values[groups[0][0]] = 1
    invalid_values[groups[0][1]] = 1
    invalid_solution["values"] = invalid_values
    invalid_solution["objective"] = -len(groups) - 1
    invalid_solution_path = tmp_path / "invalid.solution.json"
    invalid_solution_path.write_text(json.dumps(invalid_solution))
    assert verifier.verify(instance, invalid_solution_path).feasible is False


def test_binary_clique_cover_fme_extension_keeps_a_checkable_optimum(
    tmp_path: Path,
) -> None:
    generator = {
        "kind": "binary_clique_cover_mip",
        "count": 1,
        "groups": [4],
        "group_sizes": [5],
        "coefficient_min": 120,
        "coefficient_max": 190,
        "cross_edge_density": [0.1],
        "fme_auxiliary_pairs": 2,
        "randomness": {"instance_seed": 910001},
    }
    instance_config = {
        "schema_version": "1.0",
        "visibility": "judge",
        "format": "mps",
        "generator": generator,
    }
    instance = materialize_generated_instance_set(
        instance_config,
        tmp_path / "instances",
        expected_visibility="judge",
        stem_prefix="judge_id",
    )[0]
    groups, planted, _ = binary_clique_cover_data(generator, 0)
    extensions = binary_clique_cover_fme_extensions(generator, 0, groups, planted)
    values = {
        name: int(offset == planted[group_index])
        for group_index, group in enumerate(groups)
        for offset, name in enumerate(group)
    }
    values.update({extension["column"]: 0 for extension in extensions})
    objective = -len(groups)
    reference = {"values": values, "objective": objective}
    reference_path = tmp_path / "reference-fme.solution.json"
    reference_path.write_text(json.dumps(reference))
    record = TrustedOptimumRecord(
        instance_set="judge_id",
        instance_id=instance.stem,
        instance_sha256=hashlib.sha256(instance.read_bytes()).hexdigest(),
        problem_kind="binary_clique_cover",
        objective_sense="minimize",
        optimal_objective=objective,
        reference_solution_uri="private://reference-fme.solution.json",
        reference_solution_sha256=hashlib.sha256(
            reference_path.read_bytes()
        ).hexdigest(),
        optimality_basis={
            "kind": "binary_clique_cover",
            "groups": groups,
            "fme_extensions": extensions,
        },
        generation_provenance={
            "kind": "binary_clique_cover_fme_extension",
            "fme_extension_count": len(extensions),
            "optimality_proof": (
                "the clique-cover bound holds for every feasible assignment and "
                "the planted assignment extends with each auxiliary column zero"
            ),
        },
        verification_provenance=(
            "pitbench.problem_families.verification:TrustedOptimumFamily"
        ),
    )
    verifier = TrustedOptimumFamily(record, reference_path)

    assert len(extensions) == 2
    assert all(
        row["rhs"] == 0
        for extension in extensions
        for row in extension["positive_rows"] + extension["negative_rows"]
    )
    assert verifier._check_oracle(instance) is None
    checked = verifier.verify(instance, reference_path)
    assert checked.feasible is True
    assert checked.objective == objective

    bad_extensions = [dict(item) for item in extensions]
    bad_extensions[0] = {
        **bad_extensions[0],
        "positive_rows": [
            {
                **bad_extensions[0]["positive_rows"][0],
                "binary_coefficients": {
                    "not_a_model_column": 1,
                },
            },
            *bad_extensions[0]["positive_rows"][1:],
        ],
    }
    bad_record = record.model_copy(
        update={
            "optimality_basis": {
                "kind": "binary_clique_cover",
                "groups": groups,
                "fme_extensions": bad_extensions,
            }
        }
    )
    assert (
        TrustedOptimumFamily(bad_record, reference_path)
        ._check_oracle(instance)
        .infrastructure_error
    )


def test_binary_clique_cover_geometric_edge_sampler_is_seeded_and_respects_witness() -> (
    None
):
    generator = {
        "kind": "binary_clique_cover_mip",
        "count": 1,
        "groups": [12],
        "group_sizes": [8],
        "coefficient_min": 120,
        "coefficient_max": 190,
        "cross_edge_density": [0.08],
        "cross_edge_sampler": "geometric_skip_v1",
        "randomness": {"instance_seed": 910017},
    }
    groups, planted, edges = binary_clique_cover_data(generator, 0)
    repeated = binary_clique_cover_data(generator, 0)
    group_by_column = {
        column: group_index
        for group_index, group in enumerate(groups)
        for column in group
    }
    witness = {group[selected] for group, selected in zip(groups, planted, strict=True)}

    assert repeated == (groups, planted, edges)
    assert edges
    assert all(
        group_by_column[first] != group_by_column[second] for first, second in edges
    )
    assert all(not (first in witness and second in witness) for first, second in edges)


def _knapsack_solution(selected: list[int], item_count: int, objective: int) -> dict:
    raw = (
        f"Model status\nOptimal\nObjective {objective}\n"
        f"# Columns {item_count}\n"
        + "".join(f"x{item} {int(item in selected)}\n" for item in range(item_count))
        + "# Rows 0\n"
    )
    return {"raw_solution": raw}


def test_new_exact_task_panels_have_hash_bound_verified_targets() -> None:
    catalog = TaskCatalog(ROOT)
    for task_id in (
        "highs_multidimensional_knapsack_v1_15_1",
        "highs_binary_clique_presolve_20260921",
        "choco_graph_coloring_v6_0_1",
    ):
        record = catalog.validate_one(task_id)
        oracle_path = record.task.oracle.source.removeprefix("private://")
        oracle = TrustedOptimumOracle.from_yaml(ROOT / "private" / oracle_path)
        judge_config = next(
            instance_set.instance_set_config.removeprefix("private://")
            for instance_set in record.task.instance_sets
            if instance_set.kind.value == "judge_id"
        )
        payload = yaml.safe_load((ROOT / "private" / judge_config).read_text())
        with tempfile.TemporaryDirectory() as temporary:
            paths = materialize_generated_instance_set(
                payload,
                Path(temporary),
                expected_visibility="judge",
                stem_prefix="judge_id",
            )
            records = [
                record for record in oracle.records if record.instance_set == "judge_id"
            ]
            assert len(paths) == len(records) == 30
            for path, record in zip(paths, records, strict=True):
                reference = (
                    ROOT
                    / "private"
                    / record.reference_solution_uri.removeprefix("private://")
                )
                verifier = TrustedOptimumFamily(record, reference)
                assert verifier._check_oracle(path) is None
                if record.problem_kind == "graph_coloring":
                    assert verifier.verify(path, reference).feasible


def test_new_exact_tasks_reject_semantically_invalid_solutions(tmp_path: Path) -> None:
    knapsack = (
        ROOT
        / "configs/instance_sets/highs_multidimensional_knapsack_v1_15_1_dev"
        / "instances/dev_0000.lp"
    )
    reference = json.loads(
        (
            ROOT
            / "configs/instance_sets/highs_multidimensional_knapsack_v1_15_1_dev"
            / "bks_solutions/dev_0000.solution.json"
        ).read_text()
    )
    weights, capacities = _public_multidimensional_knapsack_data(knapsack)
    solution_path = tmp_path / "knapsack.solution.json"
    solution_path.write_text(
        json.dumps(
            _knapsack_solution(
                reference["selected_items"], len(weights[0]), -sum(capacities)
            )
        )
    )
    assert MIPFamily().verify(knapsack, solution_path).feasible
    development_output = tmp_path / "knapsack.json"
    development_output.with_suffix(".solution.json").write_bytes(
        solution_path.read_bytes()
    )
    feedback = _public_verification(
        {
            "problem_family": "mip",
            "development_targets": {
                "dev_0000.lp": {
                    "sha256": hashlib.sha256(knapsack.read_bytes()).hexdigest()
                }
            },
        },
        knapsack,
        development_output,
    )
    assert feedback["verification"] == "independent_public"
    assert feedback["verified_valid"] is True
    solution_path.write_text(
        json.dumps(
            _knapsack_solution(
                list(range(len(weights[0]))), len(weights[0]), -sum(capacities)
            )
        )
    )
    assert not MIPFamily().verify(knapsack, solution_path).feasible

    wrapped = tmp_path / "wrapped_knapsack.lp"
    wrapped_lines = []
    for line in knapsack.read_text().splitlines():
        while len(line) > 120:
            split = line.rfind(" ", 0, 120)
            assert split > 0
            wrapped_lines.append(line[:split])
            line = "  " + line[split + 1 :]
        wrapped_lines.append(line)
    wrapped.write_text("\n".join(wrapped_lines) + "\n")
    assert _public_multidimensional_knapsack_data(wrapped) == (weights, capacities)
    solution_path.write_text(
        json.dumps(
            _knapsack_solution(
                reference["selected_items"], len(weights[0]), -sum(capacities)
            )
        )
    )
    assert MIPFamily().verify(wrapped, solution_path).feasible
    changed = tmp_path / "changed_knapsack.lp"
    changed.write_text(wrapped.read_text().replace("<= 1274", "<= 1275", 1))
    changed_data = _public_multidimensional_knapsack_data(changed)
    assert changed_data is not None
    assert changed_data != (weights, capacities)
    assert changed_data[1][0] == 1275

    mps = ROOT / "tests/fixtures/mip/highs_pr577_knapsack_dev_0000.mps"
    assert _public_multidimensional_knapsack_data(mps) == (weights, capacities)
    selected = set(reference["selected_items"])
    legacy_solution = (
        "60 3 : Number of columns and rows for primal or dual solution or basis\n"
        "T Primal solution\nF Dual solution\nF Basis\nColumns\n"
        + "".join(f"{int(item in selected)}\n" for item in range(60))
    )
    mps_solution_path = tmp_path / "legacy_mps.solution.json"
    mps_solution_path.write_text(
        json.dumps({"raw_solution": legacy_solution, "objective": -4542})
    )
    assert MIPFamily().verify(mps, mps_solution_path).feasible
    hidden_oracle = TrustedOptimumOracle.from_yaml(
        ROOT / "private/oracles/highs_mip_row_aggregation_20210916.yaml"
    )
    trusted_record = next(
        record
        for record in hidden_oracle.records
        if record.instance_set == "judge_id" and record.instance_id == "judge_id_0000"
    )
    trusted_reference = (
        ROOT
        / "private"
        / trusted_record.reference_solution_uri.removeprefix("private://")
    )
    hidden_instance = (
        ROOT
        / "private/hidden_instances/highs_pr577_mps/multidimensional_knapsack/instances/judge_id_0000.mps"
    )
    assert (
        TrustedOptimumFamily(trusted_record, trusted_reference)._check_oracle(
            hidden_instance
        )
        is None
    )

    schedule_mps = ROOT / "tests/fixtures/mip/highs_pr577_schedule_dev_0000.mps"
    schedule_reference = json.loads(
        (
            ROOT
            / "configs/instance_sets/highs_v1_15_1_dev/bks_solutions/dev_0000.solution.json"
        ).read_text()
    )
    order = schedule_reference["order"]
    position = {job: index for index, job in enumerate(order)}
    start_times = schedule_reference["start_times"]
    order_variables = [
        int(position[first] < position[second])
        for first in range(len(order))
        for second in range(first + 1, len(order))
    ]
    schedule_values = [*start_times, *order_variables]
    schedule_raw_solution = (
        f"{len(schedule_values)} 72 : Number of columns and rows for primal or dual solution or basis\n"
        "T Primal solution\nF Dual solution\nF Basis\nColumns\n"
        + "".join(f"{value}\n" for value in schedule_values)
    )
    mps_solution_path.write_text(
        json.dumps(
            {
                "raw_solution": schedule_raw_solution,
                "objective": sum(start_times),
            }
        )
    )
    assert MIPFamily().verify(schedule_mps, mps_solution_path).feasible

    graph = (
        ROOT
        / "configs/instance_sets/choco_graph_coloring_v6_0_1_dev"
        / "instances/dev_0000.json"
    )
    graph_data = json.loads(graph.read_text())
    colors = json.loads(
        (
            ROOT
            / "configs/instance_sets/choco_graph_coloring_v6_0_1_dev"
            / "bks_solutions/dev_0000.solution.json"
        ).read_text()
    )["colors"]
    solution_path = tmp_path / "graph.solution.json"
    solution_path.write_text(json.dumps({"colors": colors}))
    assert CPFamily().verify(graph, solution_path).feasible
    development_output = tmp_path / "graph.json"
    development_output.with_suffix(".solution.json").write_bytes(
        solution_path.read_bytes()
    )
    feedback = _public_verification({"problem_family": "cp"}, graph, development_output)
    assert feedback["verification"] == "independent_public"
    assert feedback["verified_valid"] is True
    changed = colors.copy()
    first, second = graph_data["edge_endpoints"][:2]
    changed[second] = changed[first]
    solution_path.write_text(json.dumps({"colors": changed}))
    assert not CPFamily().verify(graph, solution_path).feasible
