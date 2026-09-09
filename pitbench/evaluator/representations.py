"""Equivalent input representations for routing and linear models."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class CustomerRepresentation:
    """Customer permutations with the depot fixed."""

    @staticmethod
    def permutations(
        customer_count: int, count: int, generator: random.Random
    ) -> list[list[int]]:
        if count < 1 or math.factorial(customer_count) - 1 < count:
            raise ValueError("not enough distinct nonidentity customer permutations")
        original = tuple(range(1, customer_count + 1))
        seen = {original}
        mappings = []
        while len(mappings) < count:
            customers = list(original)
            generator.shuffle(customers)
            permutation = tuple(customers)
            if permutation not in seen:
                seen.add(permutation)
                mappings.append([0, *customers])
        return mappings

    @staticmethod
    def permute(original: dict, new_to_original: list[int]) -> dict:
        node_count = len(original["coordinates"])
        if (
            original.get("depot", 0) != 0
            or original.get("distance_metric") != "EUC_2D"
            or len(original["demands"]) != node_count
        ):
            raise ValueError("expected a normalized single-depot EUC_2D CVRP instance")
        if new_to_original[0] != 0 or sorted(new_to_original) != list(
            range(node_count)
        ):
            raise ValueError("mapping must be a bijection fixing depot zero")
        transformed = dict(original)
        # node_ids remain the labels in the new representation; the saved mapping
        # records their relationship to the original customers.
        transformed["coordinates"] = [
            original["coordinates"][i] for i in new_to_original
        ]
        transformed["demands"] = [original["demands"][i] for i in new_to_original]
        if "node_ids" in original:
            transformed["node_ids"] = list(range(1, node_count + 1))
        return transformed

    @staticmethod
    def map_solution(solution: dict, new_to_original: list[int]) -> dict:
        routes = []
        for route in solution["routes"]:
            if any(
                type(node) is not int or not 0 < node < len(new_to_original)
                for node in route
            ):
                raise ValueError("solution contains an invalid customer index")
            routes.append([new_to_original[node] for node in route])
        return {"routes": routes}


VECTOR_FIELDS = ("cost", "col_lower", "col_upper", "integrality", "col_names")

ROW_FIELDS = ("row_lower", "row_upper", "row_names")


class LinearModelRepresentation:
    """MILP row/column permutations and independent primal arithmetic checks."""

    @staticmethod
    def read(path: Path) -> dict:
        import highspy
        import numpy as np
        from scipy.sparse import csc_matrix

        solver = highspy.Highs()
        solver.setOptionValue("output_flag", False)
        if solver.readModel(str(path)) != highspy.HighsStatus.kOk:
            raise ValueError(f"could not read model: {path}")
        model = solver.getLp()
        if model.a_matrix_.format_ != highspy.MatrixFormat.kColwise:
            raise ValueError("expected column-wise model matrix")
        types = np.asarray(model.integrality_, dtype=np.int32)
        if len(types) != model.num_col_ or not set(types) <= {0, 1}:
            raise ValueError("collector supports continuous and integer MILP variables")
        matrix = csc_matrix(
            (model.a_matrix_.value_, model.a_matrix_.index_, model.a_matrix_.start_),
            shape=(model.num_row_, model.num_col_),
        )
        matrix.sort_indices()
        return {
            "matrix": matrix,
            "cost": np.asarray(model.col_cost_),
            "col_lower": np.asarray(model.col_lower_),
            "col_upper": np.asarray(model.col_upper_),
            "row_lower": np.asarray(model.row_lower_),
            "row_upper": np.asarray(model.row_upper_),
            "integrality": types,
            "col_names": np.asarray(model.col_names_, dtype=str),
            "row_names": np.asarray(model.row_names_, dtype=str),
            "sense": int(model.sense_),
            "offset": model.offset_,
        }

    @staticmethod
    def save(path: Path, model: dict) -> None:
        import numpy as np

        matrix = model["matrix"].tocsc()
        np.savez_compressed(
            path,
            **{key: value for key, value in model.items() if key != "matrix"},
            matrix_data=matrix.data,
            matrix_indices=matrix.indices,
            matrix_indptr=matrix.indptr,
            matrix_shape=matrix.shape,
        )

    @staticmethod
    def load(path: Path) -> dict:
        import numpy as np
        from scipy.sparse import csc_matrix

        with np.load(path, allow_pickle=False) as stored:
            result = {
                key: stored[key]
                for key in stored.files
                if not key.startswith("matrix_")
            }
            result["matrix"] = csc_matrix(
                (
                    stored["matrix_data"],
                    stored["matrix_indices"],
                    stored["matrix_indptr"],
                ),
                shape=tuple(stored["matrix_shape"]),
            )
        result["sense"] = int(result["sense"])
        result["offset"] = float(result["offset"])
        return result

    @staticmethod
    def check_permutation(mapping, size: int) -> np.ndarray:
        import numpy as np

        values = np.asarray(mapping)
        if values.dtype.kind not in "iu" or not np.array_equal(
            np.sort(values), np.arange(size)
        ):
            raise ValueError("mapping must be a bijection of the original indices")
        return values

    @staticmethod
    def permute(model: dict, rows, columns) -> dict:
        rows = LinearModelRepresentation.check_permutation(
            rows, model["matrix"].shape[0]
        )
        columns = LinearModelRepresentation.check_permutation(
            columns, model["matrix"].shape[1]
        )
        result = dict(model)
        result["matrix"] = model["matrix"][rows, :][:, columns].tocsc()
        result["matrix"].sort_indices()
        for key in VECTOR_FIELDS:
            result[key] = model[key][columns]
        for key in ROW_FIELDS:
            result[key] = model[key][rows]
        return result

    @staticmethod
    def assert_equivalent(original: dict, transformed: dict, rows, columns) -> None:
        import numpy as np

        restored = LinearModelRepresentation.permute(
            transformed, np.argsort(rows), np.argsort(columns)
        )
        if (original["matrix"] != restored["matrix"]).nnz:
            raise ValueError("permutation changed matrix coefficients")
        for key in (*VECTOR_FIELDS, *ROW_FIELDS, "sense", "offset"):
            if not np.array_equal(original[key], restored[key]):
                raise ValueError(f"permutation changed {key}")

    @staticmethod
    def to_highs_lp(model: dict):
        import highspy

        matrix = model["matrix"].tocsc()
        result = highspy.HighsLp()
        result.num_row_, result.num_col_ = matrix.shape
        for field, key in (
            ("col_cost_", "cost"),
            ("col_lower_", "col_lower"),
            ("col_upper_", "col_upper"),
            ("row_lower_", "row_lower"),
            ("row_upper_", "row_upper"),
            ("col_names_", "col_names"),
            ("row_names_", "row_names"),
        ):
            setattr(result, field, model[key].tolist())
        result.integrality_ = [
            highspy.HighsVarType(int(value)) for value in model["integrality"]
        ]
        result.sense_ = highspy.ObjSense(model["sense"])
        result.offset_ = model["offset"]
        result.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        result.a_matrix_.num_row_, result.a_matrix_.num_col_ = matrix.shape
        result.a_matrix_.start_ = matrix.indptr
        result.a_matrix_.index_ = matrix.indices
        result.a_matrix_.value_ = matrix.data
        return result

    @staticmethod
    def map_solution(values, columns) -> np.ndarray:
        import numpy as np

        values = np.asarray(values, dtype=float)
        columns = LinearModelRepresentation.check_permutation(columns, len(values))
        mapped = np.empty_like(values)
        mapped[columns] = values
        return mapped

    @staticmethod
    def verify_primal(model: dict, values, tolerance: float) -> dict:
        import numpy as np

        values = np.asarray(values, dtype=float)
        if values.shape != model["cost"].shape or not np.isfinite(values).all():
            return {
                "feasible": False,
                "reason": "missing, nonfinite or wrong-sized solution",
            }
        activity = model["matrix"] @ values
        if not np.isfinite(activity).all():
            return {"feasible": False, "reason": "nonfinite constraint activity"}
        bound_error = float(
            max(
                np.max(model["col_lower"] - values, initial=0),
                np.max(values - model["col_upper"], initial=0),
            )
        )
        row_error = float(
            max(
                np.max(model["row_lower"] - activity, initial=0),
                np.max(activity - model["row_upper"], initial=0),
            )
        )
        integer_values = values[model["integrality"] == 1]
        integer_error = float(
            np.max(np.abs(integer_values - np.rint(integer_values)), initial=0)
        )
        objective = (
            math.fsum(float(a) * float(b) for a, b in zip(model["cost"], values))
            + model["offset"]
        )
        return {
            "feasible": max(bound_error, row_error, integer_error) <= tolerance,
            "tolerance": tolerance,
            "max_bound_violation": bound_error,
            "max_row_violation": row_error,
            "max_integrality_violation": integer_error,
            "objective": objective,
        }
