"""Equivalent input representations for routing and linear models."""

from __future__ import annotations

import json
import math
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import numpy as np


class RepresentationPlugin(ABC):
    name: str
    family: str
    requires_tolerance: bool
    suffix: str
    _plugins: ClassVar[dict[str, type[RepresentationPlugin]]] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        name = getattr(cls, "name", None)
        if name:
            existing = cls._plugins.get(name)
            if existing is not None and existing is not cls:
                raise ValueError(f"duplicate representation plugin: {name}")
            cls._plugins[name] = cls

    @staticmethod
    @abstractmethod
    def verifier(tolerance=None): ...

    @staticmethod
    @abstractmethod
    def assert_same(original, restored) -> None: ...

    @staticmethod
    @abstractmethod
    def generator(seed: int): ...

    @staticmethod
    @abstractmethod
    def read_input(path: Path): ...

    @staticmethod
    @abstractmethod
    def write_input(path: Path, model) -> None: ...

    @staticmethod
    @abstractmethod
    def mappings(model, count: int, generator) -> list: ...

    @staticmethod
    @abstractmethod
    def transform(model, mapping): ...

    @staticmethod
    @abstractmethod
    def verify_files(
        original, transformed, solution, mapped_path, mapping, tolerance=None
    ) -> dict: ...


class CustomerRepresentation(RepresentationPlugin):
    """Customer permutations with the depot fixed."""

    name = "customer_relabeling"
    family = "cvrp"
    requires_tolerance = False

    @staticmethod
    def verifier(tolerance=None):
        from pitbench.problem_families.verification import CVRPFamily

        return CVRPFamily()

    @staticmethod
    def assert_same(original, restored):
        if original != restored:
            raise ValueError("representation input changed")

    suffix = ".json"
    generator = staticmethod(random.Random)

    @staticmethod
    def read_input(path: Path) -> dict:
        return json.loads(path.read_text())

    @staticmethod
    def write_input(path: Path, model: dict) -> None:
        path.write_text(json.dumps(model, indent=2, allow_nan=False) + "\n")

    @staticmethod
    def mappings(model: dict, count: int, generator) -> list[dict]:
        return [
            {"new_to_original": mapping}
            for mapping in CustomerRepresentation.permutations(
                len(model["coordinates"]) - 1, count, generator
            )
        ]

    @staticmethod
    def transform(model: dict, mapping: dict) -> dict:
        return CustomerRepresentation.permute(model, mapping["new_to_original"])

    @staticmethod
    def verify_files(
        original: Path,
        transformed: Path,
        solution: Path,
        mapped_path: Path,
        mapping: dict,
        tolerance: float | None = None,
    ) -> dict:
        from pitbench.problem_families.verification import CVRPFamily

        family = CVRPFamily()
        checked = family.verify(transformed, solution)
        CustomerRepresentation.write_input(
            mapped_path,
            CustomerRepresentation.map_solution(
                json.loads(solution.read_text()), mapping["new_to_original"]
            ),
        )
        mapped = family.verify(original, mapped_path)
        return {
            "transformed": checked.model_dump(),
            "mapped_original": mapped.model_dump(),
            "objective_preserved": checked.objective == mapped.objective
            if checked.objective is not None and mapped.objective is not None
            else None,
        }

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


class LinearModelRepresentation(RepresentationPlugin):
    """MILP row/column permutations and independent primal arithmetic checks."""

    name = "row_column_permutation"
    family = "mip"
    requires_tolerance = True

    @staticmethod
    def verifier(tolerance):
        from pitbench.problem_families.verification import NumericModelFamily

        return NumericModelFamily(tolerance)

    @staticmethod
    def assert_same(original, restored):
        LinearModelRepresentation.assert_equivalent(
            original,
            restored,
            list(range(original["matrix"].shape[0])),
            list(range(len(original["cost"]))),
        )

    suffix = ".npz"

    @staticmethod
    def generator(seed: int):
        import numpy as np

        return np.random.default_rng(seed)

    @staticmethod
    def read_input(path: Path) -> dict:
        return (
            LinearModelRepresentation.load(path)
            if path.suffix == ".npz"
            else LinearModelRepresentation.read(path)
        )

    @staticmethod
    def write_input(path: Path, model: dict) -> None:
        LinearModelRepresentation.save(path, model)

    @staticmethod
    def mappings(model: dict, count: int, generator) -> list[dict]:
        import numpy as np

        row_count, column_count = model["matrix"].shape
        if (
            count < 1
            or math.factorial(row_count) * math.factorial(column_count) - 1 < count
        ):
            raise ValueError("not enough distinct nonidentity row/column permutations")
        seen = set()
        mappings = []
        while len(mappings) < count:
            rows = generator.permutation(row_count)
            columns = generator.permutation(column_count)
            signature = (tuple(rows), tuple(columns))
            if signature in seen or (
                np.array_equal(rows, np.arange(row_count))
                and np.array_equal(columns, np.arange(column_count))
            ):
                continue
            seen.add(signature)
            mappings.append({"rows": rows.tolist(), "columns": columns.tolist()})
        return mappings

    @staticmethod
    def transform(model: dict, mapping: dict) -> dict:
        transformed = LinearModelRepresentation.permute(
            model, mapping["rows"], mapping["columns"]
        )
        LinearModelRepresentation.assert_equivalent(
            model, transformed, mapping["rows"], mapping["columns"]
        )
        return transformed

    @staticmethod
    def verify_files(
        original: Path,
        transformed: Path,
        solution: Path,
        mapped_path: Path,
        mapping: dict,
        tolerance: float | None = None,
    ) -> dict:
        if tolerance is None:
            raise ValueError(
                "numeric representation verification requires an explicit tolerance"
            )
        from pitbench.problem_families.verification import NumericModelFamily

        values = json.loads(solution.read_text())["values"]
        mapped_values = LinearModelRepresentation.map_solution(
            values, mapping["columns"]
        )
        mapped_path.write_text(json.dumps({"values": mapped_values.tolist()}) + "\n")
        checked = NumericModelFamily.check_model(
            LinearModelRepresentation.read_input(transformed), values, tolerance
        )
        mapped = NumericModelFamily.check_model(
            LinearModelRepresentation.read_input(original), mapped_values, tolerance
        )
        return {
            "transformed": checked,
            "mapped_original": mapped,
            "objective_preserved": checked.get("objective") == mapped.get("objective")
            if "objective" in checked and "objective" in mapped
            else None,
        }

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


def representation_type(name: str) -> type[RepresentationPlugin]:
    try:
        return RepresentationPlugin._plugins[name]
    except KeyError:
        raise ValueError(f"unsupported representation: {name}") from None
