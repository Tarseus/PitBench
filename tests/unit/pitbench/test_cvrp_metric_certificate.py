from __future__ import annotations

import pytest

from pitbench.metrics.cvrp_certificate import (
    CVRPMapCertificate,
    certificate_from_correspondence,
    search_cvrp_upper_bound,
    signature_cvrp_map_certificate,
    verify_cvrp_map_certificate,
)
from pitbench.metrics.cvrp_problem import (
    anchored_correspondence_cost,
    as_anchored_marked_cvrp,
    exact_cvrp_problem_metric,
)


def _instance(*, coordinates=None, demands=None, capacity: float = 5):
    return {
        "depot": 0,
        "coordinates": coordinates or [[0, 0], [2, 0], [0, 1]],
        "demands": demands or [0, 1, 2],
        "capacity": capacity,
    }


def test_map_certificate_round_trips_and_matches_its_union_relation() -> None:
    left = as_anchored_marked_cvrp(_instance())
    right = as_anchored_marked_cvrp(
        _instance(
            coordinates=[[0, 0], [1.8, 0], [0, 1], [1, 1]],
            demands=[0, 1, 2, 1],
        )
    )
    certificate = CVRPMapCertificate(forward_map=(0, 1, 2), backward_map=(0, 1, 2, 2))

    restored = CVRPMapCertificate.from_mapping(certificate.to_dict())
    verified = verify_cvrp_map_certificate(left, right, restored)

    assert restored == certificate
    assert verified.upper_bound == pytest.approx(
        anchored_correspondence_cost(left, right, verified.correspondence)
    )


def test_codistortion_rejects_individually_isometric_but_incoherent_maps() -> None:
    symmetric = _instance(coordinates=[[0, 0], [1, 0], [-1, 0]], demands=[0, 1, 1])
    incompatible = CVRPMapCertificate(forward_map=(0, 1, 2), backward_map=(0, 2, 1))

    verified = verify_cvrp_map_certificate(symmetric, symmetric, incompatible)

    assert verified.forward_distortion == pytest.approx(0)
    assert verified.backward_distortion == pytest.approx(0)
    assert verified.forward_mark_discrepancy == pytest.approx(0)
    assert verified.backward_mark_discrepancy == pytest.approx(0)
    assert verified.codistortion == pytest.approx(1)
    assert verified.upper_bound == pytest.approx(1)


def test_maps_selected_from_an_exact_correspondence_recover_the_exact_cost() -> None:
    left = _instance()
    right = _instance(coordinates=[[0, 0], [1.7, 0.1], [0.2, 1.1]], demands=[0, 2, 1])
    exact = exact_cvrp_problem_metric(left, right)

    certificate = certificate_from_correspondence(left, right, exact.correspondence)
    verified = verify_cvrp_map_certificate(left, right, certificate)

    assert verified.upper_bound == pytest.approx(exact.distance)


def test_signature_witness_is_a_certified_upper_bound_with_mixed_cardinality() -> None:
    left = _instance()
    right = _instance(
        coordinates=[[0, 0], [1.8, 0], [0, 1], [1, 1]],
        demands=[0, 1, 2, 1],
    )
    exact = exact_cvrp_problem_metric(left, right)

    certificate = signature_cvrp_map_certificate(left, right)
    verified = verify_cvrp_map_certificate(left, right, certificate)

    assert verified.upper_bound + 1e-12 >= exact.distance
    assert len(certificate.forward_map) == 3
    assert len(certificate.backward_map) == 4


def test_bounded_local_search_returns_monotone_anytime_certificates() -> None:
    symmetric = _instance(coordinates=[[0, 0], [1, 0], [-1, 0]], demands=[0, 1, 1])
    incompatible = CVRPMapCertificate(forward_map=(0, 1, 2), backward_map=(0, 2, 1))

    result = search_cvrp_upper_bound(
        symmetric,
        symmetric,
        initial_certificate=incompatible,
        candidate_pool_size=None,
        max_passes=2,
        max_evaluations=20,
    )

    assert result.upper_bound_history[0] == pytest.approx(1)
    assert result.best.upper_bound == pytest.approx(0)
    assert all(
        later <= earlier
        for earlier, later in zip(
            result.upper_bound_history, result.upper_bound_history[1:]
        )
    )
    assert result.verifier_evaluations <= 20


@pytest.mark.parametrize(
    "certificate, message",
    [
        (CVRPMapCertificate((0, 1), (0, 1, 2)), "one image"),
        (CVRPMapCertificate((1, 1, 2), (0, 1, 2)), "depot partition"),
        (CVRPMapCertificate((0, 1, 3), (0, 1, 2)), "out-of-range"),
    ],
)
def test_verifier_rejects_malformed_certificates(certificate, message) -> None:
    with pytest.raises(ValueError, match=message):
        verify_cvrp_map_certificate(_instance(), _instance(), certificate)


def test_serialized_certificate_does_not_coerce_non_integer_images() -> None:
    payload = {"forward_map": [0, 1.5, 2], "backward_map": [0, 1, 2]}

    with pytest.raises(ValueError, match="integers"):
        verify_cvrp_map_certificate(_instance(), _instance(), payload)
