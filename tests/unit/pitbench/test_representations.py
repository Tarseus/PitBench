import numpy as np
import pytest

pytest.importorskip("scipy")
highspy = pytest.importorskip("highspy")
from scipy.sparse import csc_matrix  # noqa: E402

from pitbench.evaluator.representations import LinearModelRepresentation  # noqa: E402


@pytest.fixture
def model():
    return {
        "matrix": csc_matrix([[1.0, 2.0, 0.0], [-1.0, 0.0, 1.0]]),
        "cost": np.array([3.0, -2.0, 1.0]),
        "col_lower": np.zeros(3),
        "col_upper": np.array([3.0, 2.0, 4.0]),
        "row_lower": np.array([2.0, -np.inf]),
        "row_upper": np.array([np.inf, 1.5]),
        "integrality": np.array([1, 1, 0]),
        "col_names": np.array(["a", "b", "c"]),
        "row_names": np.array(["demand", "link"]),
        "offset": 7.0,
        "sense": 1,
    }


@pytest.mark.parametrize("sense,expected", [(1, 3.0), (-1, 20.0)])
def test_permuted_mip_has_same_optimum_and_mapped_feasible_solution(
    model, sense, expected
):
    model["sense"] = sense
    rows, columns = [1, 0], [2, 0, 1]
    transformed = LinearModelRepresentation.permute(model, rows, columns)
    LinearModelRepresentation.assert_equivalent(model, transformed, rows, columns)
    for candidate, mapping in ((model, [0, 1, 2]), (transformed, columns)):
        solver = highspy.Highs()
        solver.setOptionValue("output_flag", False)
        solver.setOptionValue("threads", 1)
        assert (
            solver.passModel(LinearModelRepresentation.to_highs_lp(candidate))
            == highspy.HighsStatus.kOk
        )
        assert solver.run() == highspy.HighsStatus.kOk
        assert solver.getModelStatus() == highspy.HighsModelStatus.kOptimal
        result = LinearModelRepresentation.verify_primal(
            model,
            LinearModelRepresentation.map_solution(
                solver.getSolution().col_value, mapping
            ),
            1e-6,
        )
        assert result["feasible"]
        assert result["objective"] == pytest.approx(expected)


@pytest.mark.parametrize("field", ["cost", "col_upper", "row_lower", "integrality"])
def test_equivalence_rejects_semantic_changes(model, field):
    transformed = LinearModelRepresentation.permute(model, [1, 0], [2, 0, 1])
    transformed[field] = transformed[field].copy()
    index = int(np.flatnonzero(np.isfinite(transformed[field]))[0])
    transformed[field][index] += 1
    with pytest.raises(ValueError, match="permutation changed"):
        LinearModelRepresentation.assert_equivalent(
            model, transformed, [1, 0], [2, 0, 1]
        )


@pytest.mark.parametrize(
    "values,violation",
    [
        ([0, 0, 0], "max_row_violation"),
        ([0, 3, 0], "max_bound_violation"),
        ([0, 1.5, 0], "max_integrality_violation"),
    ],
)
def test_independent_verifier_rejects_bad_primal(model, values, violation):
    result = LinearModelRepresentation.verify_primal(model, values, 1e-6)
    assert not result["feasible"]
    assert result[violation] > 1e-6


def test_verifier_rejects_nonfinite_and_wrong_length(model):
    for values in ([0, np.nan, 0], [0, np.inf, 0], [0, 2]):
        assert not LinearModelRepresentation.verify_primal(model, values, 1e-6)[
            "feasible"
        ]


def test_storage_preserves_objective_sense_offset_and_domains(model, tmp_path):
    path = tmp_path / "model.npz"
    LinearModelRepresentation.save(path, model)
    restored = LinearModelRepresentation.load(path)
    LinearModelRepresentation.assert_equivalent(model, restored, [0, 1], [0, 1, 2])


@pytest.mark.parametrize("mapping", [[0, 0, 2], [0.0, 1.0, 2.0], [0, 1], [0, 1, 3]])
def test_nonbijective_mappings_are_rejected(model, mapping):
    with pytest.raises(ValueError, match="bijection"):
        LinearModelRepresentation.permute(model, [0, 1], mapping)
