from __future__ import annotations

from pitbench.distribution.cvrp_v2_matching import (
    CVRP_V2_MATCHING_BLOCKS,
    run_cvrp_v2_matching_validation,
)
from pitbench.qoi.cvrp import CVRP_INSTANCE_QOI_V2_CANDIDATE_0
from pitbench.qoi.schema import QoIRole


def test_v2_matching_blocks_cover_no_raw_axes() -> None:
    roles = {axis.name: axis.role for axis in CVRP_INSTANCE_QOI_V2_CANDIDATE_0.axes}
    flattened = tuple(
        axis for axes in CVRP_V2_MATCHING_BLOCKS.values() for axis in axes
    )
    block_axes = set(flattened)

    assert len(flattened) == len(block_axes)
    assert "capacity" not in block_axes
    assert "total_demand" not in block_axes
    assert "pairwise_distance_median" not in block_axes
    assert all(roles[axis] != QoIRole.RAW for axis in block_axes)


def test_v2_matching_validation_is_solver_free_and_deterministic() -> None:
    roles = {axis.name: axis.role for axis in CVRP_INSTANCE_QOI_V2_CANDIDATE_0.axes}
    first = run_cvrp_v2_matching_validation(pair_count=8, generator_seed=1234)
    second = run_cvrp_v2_matching_validation(pair_count=8, generator_seed=1234)

    assert first == second
    assert first.qoi_spec_version == "2.0-candidate.0"
    assert first.qoi_spec_status == "candidate.0_not_frozen"
    assert first.solver_runs_used == 0
    assert first.solver_runs_created == 0
    assert len(first.treatments) == 6
    assert first.panel_fingerprint
    for evidence in first.treatments.values():
        assert set(evidence["affected_blocks"]).isdisjoint(
            evidence["confounder_blocks"]
        )
        assert evidence["active_confounder_axes"]
        assert evidence["active_treatment_profile_axes"]
        assert evidence["methods"]["oracle_crn"]["pair_recovery_rate"] == 1
        assert all(
            roles[axis] != QoIRole.EXPERIMENTAL
            for axis in evidence["active_confounder_axes"]
        )
