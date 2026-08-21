from __future__ import annotations

import pytest

from pitbench.distribution.confounding_recovery import (
    ConfoundingRecord,
    greedy_pairs,
    matched_effect,
    maximum_standardized_imbalance,
    optimal_pairs,
)
from pitbench.distribution.paired_protocol import (
    PRIMARY_TREATMENTS,
    build_paired_cvrp_panel,
    solver_experiment_gate,
)
from pitbench.distribution.uchoa_construct_validation import (
    UchoaConstructCase,
    run_uchoa_construct_validation,
)
from pitbench.instances import make_uchoa_cvrp_instance
from pitbench.metrics.paired_response import IncumbentPoint, anytime_outcomes
from pitbench.metrics.stability import orbit_distribution_dispersion, wasserstein_1d
from pitbench.qoi.cvrp import CVRP_INSTANCE_QOI_V1_1, extract_cvrp_instance_qoi
from pitbench.qoi.schema import QoIRole


def _uchoa(**updates):
    parameters = {
        "name": "uchoa",
        "customers": 200,
        "coordinate_seed": 101,
        "demand_seed": 202,
        "depot_positioning": "C",
        "customer_positioning": "R",
        "demand_family": "small_large_variance",
        "route_size": 10,
    }
    parameters.update(updates)
    return make_uchoa_cvrp_instance(**parameters)


def test_uchoa_generator_uses_matched_latent_streams() -> None:
    central = _uchoa(depot_positioning="C")
    eccentric = _uchoa(depot_positioning="E")
    clustered = _uchoa(customer_positioning="C")

    assert central["coordinates"][1:] == eccentric["coordinates"][1:]
    assert central["demands"] == eccentric["demands"]
    assert central["demands"] == clustered["demands"]
    assert central["coordinates"][1:] != clustered["coordinates"][1:]


def test_quadrant_control_preserves_demand_multiset_and_is_detected() -> None:
    coupled = _uchoa(demand_family="quadrant")
    permuted = _uchoa(demand_family="quadrant_permuted")

    assert sorted(coupled["demands"]) == sorted(permuted["demands"])
    coupled_qoi = extract_cvrp_instance_qoi(coupled).values
    permuted_qoi = extract_cvrp_instance_qoi(permuted).values
    assert (
        coupled_qoi["demand_spatial_quadrupole_coupling"]
        > (permuted_qoi["demand_spatial_quadrupole_coupling"])
    )


def test_v1_1_marks_degenerate_geometry_as_undefined() -> None:
    instance = {
        "name": "line",
        "depot": 0,
        "coordinates": [[0, 0], [1, 0], [2, 0], [3, 0]],
        "demands": [0, 1, 2, 3],
        "capacity": 5,
    }
    observation = extract_cvrp_instance_qoi(instance)

    assert observation.values["convex_hull_area_ratio"] == 0
    assert observation.axis_defined["nearest_neighbor_clark_evans_ratio"] is False


def test_v1_1_roles_do_not_imply_a_unified_geometry() -> None:
    roles = {axis.name: axis.role for axis in CVRP_INSTANCE_QOI_V1_1.axes}

    for axis in (
        "demand_cv",
        "nearest_neighbor_clark_evans_ratio",
        "nearest_neighbor_iqr_clark_evans_ratio",
        "mst_edge_mean_n_corrected",
        "convex_hull_area_ratio",
    ):
        assert roles[axis] == QoIRole.SCALE_CONDITIONED
    assert roles["demand_spatial_quadrupole_coupling"] == QoIRole.EXPERIMENTAL


def test_anytime_outcomes_include_preincumbent_and_censoring() -> None:
    outcomes = anytime_outcomes(
        [
            IncumbentPoint(time_sec=2, objective=120),
            IncumbentPoint(time_sec=6, objective=100),
        ],
        reference=100,
        budget_sec=10,
    )

    assert outcomes.feasible is True
    assert outcomes.reference_gap == 0
    assert outcomes.time_to_target_sec == 6
    assert outcomes.primal_integral == pytest.approx(0.28)

    missing = anytime_outcomes([], reference=100, budget_sec=10)
    assert missing.feasible is False
    assert missing.primal_integral == 1
    assert missing.target_hit is False

    checkpoint = anytime_outcomes(
        [
            IncumbentPoint(time_sec=0.5, objective=110),
            IncumbentPoint(time_sec=2, objective=100),
        ],
        reference=100,
        budget_sec=1,
    )
    assert checkpoint.reference_gap == pytest.approx(0.1)


def test_distributional_stability_metrics() -> None:
    assert wasserstein_1d([0, 1], [1, 2]) == pytest.approx(1)
    result = orbit_distribution_dispersion(
        {"identity": [0, 1], "rotate": [0, 1], "relabel": [2, 3]}
    )
    assert result["identity_wasserstein_max"] == pytest.approx(2)
    assert result["orbit_pairwise_wasserstein_max"] == pytest.approx(2)


def test_exact_ot_assignment_and_recovery_diagnostics() -> None:
    source = [
        ConfoundingRecord("s0", "small", 1, {"x": 0}),
        ConfoundingRecord("s1", "small", 2, {"x": 2}),
    ]
    target = [
        ConfoundingRecord("t0", "large", 4, {"x": 2.1}),
        ConfoundingRecord("t1", "large", 3, {"x": 0.1}),
    ]
    greedy = greedy_pairs(source, target, axes=("x",))
    optimal = optimal_pairs(source, target, axes=("x",))

    assert optimal == ((0, 1), (1, 0))
    assert greedy == optimal
    assert matched_effect(source, target, optimal) == pytest.approx(2)
    assert maximum_standardized_imbalance(source, target, optimal, axes=("x",)) < 0.11


def test_uchoa_construct_validation_is_solver_free() -> None:
    cases = []
    index = 0
    for depot in ("C", "E"):
        for customer in ("R", "C"):
            for demand in ("unitary", "quadrant"):
                for route_size in (5, 20):
                    instance = make_uchoa_cvrp_instance(
                        name=f"case-{index}",
                        customers=80,
                        coordinate_seed=100 + index,
                        demand_seed=200 + index,
                        depot_positioning=depot,
                        customer_positioning=customer,
                        demand_family=demand,
                        route_size=route_size,
                    )
                    cases.append(
                        UchoaConstructCase(
                            instance_id=f"case-{index}",
                            instance=instance,
                            depot_positioning=depot,
                            customer_positioning=customer,
                            demand_family=demand,
                            route_size=route_size,
                        )
                    )
                    index += 1

    report = run_uchoa_construct_validation(cases, permutations=19)

    assert report.kind == "cvrp_uchoa_external_construct_validation"
    assert report.solver_runs_used == 0
    assert report.solver_runs_created == 0
    assert set(report.evidence) == {
        "depot_c_to_e",
        "customer_r_to_c",
        "unitary_to_variable_demand",
        "nonquadrant_to_quadrant",
        "route_size",
    }


def test_paired_panel_and_solver_gate() -> None:
    panel = build_paired_cvrp_panel((101, 202))
    assert len(panel) == 2 * len(PRIMARY_TREATMENTS)
    coupling = next(item for item in panel if item.treatment == "non_radial_coupling")
    assert sorted(coupling.source["demands"]) == sorted(coupling.target["demands"])

    artifact = {
        "qoi_spec_version": "1.1",
        "conclusion": "Falsified",
        "axioms": {
            "1_solver_independence": {"passed": True},
            "2_semantic_invariance": {"passed": True},
            "3_scale_sensitivity": {"passed": True},
            "4_structure_scale_separability": {"passed": False},
            "5_unit_representation_robustness": {"passed": True},
            "7_version_freezing": {"passed": True},
            "8_falsifiability": {"passed": True},
        },
        "controlled_directions": [
            {"name": name, "passed": True}
            for name in (
                "depot_position",
                "cluster_spread",
                "demand_dispersion",
                "non_radial_coupling",
                "route_size",
            )
        ],
    }

    ready, _ = solver_experiment_gate(artifact)
    assert ready is True
    ready, _ = solver_experiment_gate(artifact, treatment="route_size")
    assert ready is True

    artifact["controlled_directions"][-1]["passed"] = False
    blocked, reason = solver_experiment_gate(artifact, treatment="route_size")
    assert blocked is False
    assert "route_size" in reason
    unaffected, _ = solver_experiment_gate(artifact, treatment="depot")
    assert unaffected is True

    artifact["axioms"]["3_scale_sensitivity"]["passed"] = False
    blocked_scale, reason = solver_experiment_gate(artifact, treatment="scale")
    assert blocked_scale is False
    assert "scale" in reason
