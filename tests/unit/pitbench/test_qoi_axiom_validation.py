from __future__ import annotations

import json

from typer.testing import CliRunner

from pitbench.cli.main import app
from pitbench.distribution.qoi_axiom_validation import (
    RAW_UNIT_AXES,
    _version_freezing_evidence,
    run_cvrp_qoi_axiom_validation,
)
from pitbench.qoi.cvrp import (
    CVRP_INSTANCE_QOI,
    CVRP_INSTANCE_QOI_PINNED_FINGERPRINTS,
)


def _small_report():
    return run_cvrp_qoi_axiom_validation(
        seeds=(101, 211, 307, 401),
        sizes=(30, 60, 120),
        calibration_size=16,
        calibration_namespace="qoi-axiom-unit-test",
    )


def test_qoi_axiom_report_is_solver_free_and_axis_level():
    report = _small_report()

    assert report.solver_runs_used == 0
    assert report.solver_runs_created == 0
    assert report.production_geometry_changed is False
    assert "no_unified_geometry_is_defined_or_validated" in report.claim_scope
    assert set(report.axis_diagnostics) == {
        axis.name for axis in CVRP_INSTANCE_QOI.axes
    }
    assert report.axioms["1_solver_independence"]["passed"] is True
    assert report.axioms["2_semantic_invariance"]["passed"] is True
    assert report.axioms["3_scale_sensitivity"]["passed"] is True
    assert report.axioms["5_unit_representation_robustness"]["passed"] is True
    assert report.axioms["6_no_circular_learned_semantics"]["passed"] is True
    assert report.axioms["7_version_freezing"]["passed"] is True
    assert report.axioms["4_structure_scale_separability"]["scope"] == (
        "struct_core_role_only_no_unified_geometry_claim"
    )
    assert (
        "no_unified_geometry_is_defined_or_validated"
        in (report.preregistration["claim_scope"])
    )


def test_raw_unit_axes_are_reported_but_not_misclassified_as_invariant():
    report = _small_report()

    for axis in RAW_UNIT_AXES:
        assert report.axis_diagnostics[axis]["role"] == "raw"
        assert report.axis_diagnostics[axis]["unit_robust"] is None
    assert report.axis_diagnostics["demand_cv"]["unit_robust"] is True
    assert report.axis_diagnostics["customer_count"]["unit_robust"] is True


def test_controlled_directions_hold_customer_count_fixed():
    report = _small_report()

    assert {item.name for item in report.controlled_directions} == {
        "capacity_pressure",
        "cluster_spread",
        "demand_dispersion",
        "demand_location_coupling",
        "depot_position",
        "non_radial_coupling",
        "route_size",
    }
    assert all(item.scale_axis_max_delta == 0 for item in report.controlled_directions)
    assert all(item.responsive_axes for item in report.controlled_directions)


def test_qoi_axiom_report_is_reproducible():
    first = _small_report()
    second = _small_report()

    assert first.panel_fingerprint == second.panel_fingerprint
    assert first.as_dict() == second.as_dict()


def test_version_freezing_uses_independent_version_pin():
    current_fingerprint = CVRP_INSTANCE_QOI.fingerprint()
    current = _version_freezing_evidence(
        CVRP_INSTANCE_QOI,
        CVRP_INSTANCE_QOI_PINNED_FINGERPRINTS,
        {current_fingerprint},
    )
    assert current["qoi_spec_fingerprint_frozen"] is True

    changed_axis = CVRP_INSTANCE_QOI.axes[0].model_copy(
        update={"description": "Unversioned semantic change"}
    )
    changed_spec = CVRP_INSTANCE_QOI.model_copy(
        update={"axes": (changed_axis, *CVRP_INSTANCE_QOI.axes[1:])}
    )
    changed = _version_freezing_evidence(
        changed_spec,
        CVRP_INSTANCE_QOI_PINNED_FINGERPRINTS,
        {changed_spec.fingerprint()},
    )
    assert changed["version_has_pinned_fingerprint"] is True
    assert changed["spec_matches_pinned_fingerprint"] is False
    assert changed["qoi_spec_fingerprint_frozen"] is False

    bumped_spec = changed_spec.model_copy(update={"version": "1.2"})
    unpinned = _version_freezing_evidence(
        bumped_spec,
        CVRP_INSTANCE_QOI_PINNED_FINGERPRINTS,
        {bumped_spec.fingerprint()},
    )
    assert unpinned["version_has_pinned_fingerprint"] is False
    assert unpinned["qoi_spec_fingerprint_frozen"] is False

    pinned = _version_freezing_evidence(
        bumped_spec,
        {**CVRP_INSTANCE_QOI_PINNED_FINGERPRINTS, "1.2": bumped_spec.fingerprint()},
        {bumped_spec.fingerprint()},
    )
    assert pinned["qoi_spec_fingerprint_frozen"] is True


def test_qoi_axiom_cli_writes_artifact(tmp_path):
    output = tmp_path / "qoi-axioms.json"
    result = CliRunner().invoke(
        app,
        [
            "qoi",
            "validate-cvrp-axioms",
            "--calibration-size",
            "16",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text())
    assert payload["kind"] == "cvrp_instance_qoi_axiom_validation"
    assert payload["solver_runs_created"] == 0
    assert set(payload["axioms"]) == {
        f"{number}_{name}"
        for number, name in (
            (1, "solver_independence"),
            (2, "semantic_invariance"),
            (3, "scale_sensitivity"),
            (4, "structure_scale_separability"),
            (5, "unit_representation_robustness"),
            (6, "no_circular_learned_semantics"),
            (7, "version_freezing"),
            (8, "falsifiability"),
        )
    }
