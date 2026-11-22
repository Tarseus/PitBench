"""Equivalent input representations for routing and linear models."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import random
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import numpy as np


class RepresentationPlugin(ABC):
    name: str
    family: str
    requires_tolerance: bool
    suffix: str
    runtime_dependencies: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def check_runtime(cls) -> None:
        for name in cls.runtime_dependencies:
            importlib.import_module(name)

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


class NativeCPRepresentation(RepresentationPlugin):
    """Identity representation for native CP scenario inputs."""

    name = "native_instance"
    family = "cp"
    requires_tolerance = False
    suffix = ".json"
    generator = staticmethod(random.Random)

    @staticmethod
    def verifier(tolerance=None):
        from pitbench.problem_families.verification import NativeCPFamily
        return NativeCPFamily()

    @staticmethod
    def read_input(path: Path) -> dict:
        if path.suffix.lower() not in {".json", ".txt"}:
            raise ValueError("native CP scenario requires JSON or job-shop text")
        return {"content": path.read_text(), "suffix": path.suffix.lower()}

    @staticmethod
    def write_input(path: Path, model: dict) -> None:
        path.write_text(model["content"])

    @staticmethod
    def assert_same(original, restored) -> None:
        if original != restored:
            raise ValueError("native CP input changed")

    @staticmethod
    def mappings(model, count: int, generator) -> list:
        raise ValueError("native CP identity has no relabeling map")

    @staticmethod
    def transform(model, mapping):
        raise ValueError("native CP identity has no relabeling transform")

    @staticmethod
    def verify_files(original, transformed, solution, mapped_path, mapping, tolerance=None) -> dict:
        raise ValueError("native CP identity does not define a relabeling")


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
        if model.get("problem_kind") in {
            "heterogeneous_fleet_vrp", "multi_depot_vrp", "open_vrp"
        }:
            node_count = len(model["coordinates"])
            depots, customers = model.get("depots"), model.get("customers")
            if (
                not isinstance(depots, list)
                or not depots
                or not isinstance(customers, list)
                or len(set(depots)) != len(depots)
                or len(set(customers)) != len(customers)
                or set(depots) & set(customers)
                or set(depots) | set(customers) != set(range(node_count))
            ):
                raise ValueError("routing customers and depots must partition the nodes")
            if count < 1 or math.factorial(len(customers)) - 1 < count:
                raise ValueError("not enough distinct nonidentity customer permutations")
            original = tuple(customers)
            seen = {original}
            result = []
            while len(result) < count:
                permutation = list(customers)
                generator.shuffle(permutation)
                key = tuple(permutation)
                if key in seen:
                    continue
                seen.add(key)
                mapping = list(range(node_count))
                for position, source in zip(customers, permutation, strict=True):
                    mapping[position] = source
                result.append({"new_to_original": mapping})
            return result
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
        depots = original.get("depots", [original.get("depot", 0)])
        if (
            not isinstance(depots, list)
            or not depots
            or any(type(depot) is not int or not 0 <= depot < node_count for depot in depots)
            or original.get("distance_metric") != "EUC_2D"
            or len(original["demands"]) != node_count
        ):
            raise ValueError("expected a normalized EUC_2D routing instance")
        if sorted(new_to_original) != list(range(node_count)) or any(
            new_to_original[depot] != depot for depot in depots
        ):
            raise ValueError("mapping must be a bijection fixing depot zero")
        transformed = dict(original)
        # node_ids remain the labels in the new representation; the saved mapping
        # records their relationship to the original customers.
        transformed["coordinates"] = [
            original["coordinates"][i] for i in new_to_original
        ]
        transformed["demands"] = [original["demands"][i] for i in new_to_original]
        for field in ("time_windows", "service_times"):
            if field in original:
                if len(original[field]) != node_count:
                    raise ValueError(f"{field} must cover every node")
                transformed[field] = [original[field][i] for i in new_to_original]
        if "node_ids" in original:
            transformed["node_ids"] = list(range(1, node_count + 1))
        return transformed

    @staticmethod
    def map_solution(solution: dict, new_to_original: list[int]) -> dict:
        routes = []
        for route in solution["routes"]:
            if any(
                type(node) is not int or not 0 <= node < len(new_to_original)
                for node in route
            ):
                raise ValueError("solution contains an invalid customer index")
            routes.append([new_to_original[node] for node in route])
        return {**solution, "routes": routes}


class BinPackingItemRepresentation(RepresentationPlugin):
    """Permute item order while preserving bin-packing weights and capacity."""

    name = "item_relabeling"
    family = "cp"
    requires_tolerance = False
    suffix = ".json"
    generator = staticmethod(random.Random)

    @staticmethod
    def verifier(tolerance=None):
        from pitbench.problem_families.verification import CPFamily

        return CPFamily()

    @staticmethod
    def assert_same(original, restored) -> None:
        if original != restored:
            raise ValueError("representation input changed")

    @staticmethod
    def read_input(path: Path) -> dict:
        return json.loads(path.read_text())

    @staticmethod
    def write_input(path: Path, model: dict) -> None:
        path.write_text(json.dumps(model, indent=2, allow_nan=False) + "\n")

    @staticmethod
    def mappings(model: dict, count: int, generator: random.Random) -> list[dict]:
        weights = model["weights"]
        if not isinstance(weights, list) or len(weights) < 2 or count < 1:
            raise ValueError("item relabeling needs at least two items and one mapping")
        distinct_inputs = math.factorial(len(weights))
        for repeated in Counter(weights).values():
            distinct_inputs //= math.factorial(repeated)
        if distinct_inputs - 1 < count:
            raise ValueError("not enough distinct item-order representations")
        original = tuple(range(len(weights)))
        seen = {original}
        seen_weights = {tuple(weights)}
        mappings = []
        while len(mappings) < count:
            permutation = list(original)
            generator.shuffle(permutation)
            signature = tuple(permutation)
            reordered_weights = tuple(weights[index] for index in permutation)
            if signature not in seen and reordered_weights not in seen_weights:
                seen.add(signature)
                seen_weights.add(reordered_weights)
                mappings.append({"new_to_original": permutation})
        return mappings

    @staticmethod
    def transform(model: dict, mapping: dict) -> dict:
        weights = model["weights"]
        new_to_original = mapping["new_to_original"]
        if sorted(new_to_original) != list(range(len(weights))):
            raise ValueError("item mapping must be a bijection")
        if type(model.get("capacity")) is not int or model["capacity"] <= 0:
            raise ValueError("invalid bin-packing capacity")
        return {**model, "weights": [weights[index] for index in new_to_original]}

    @staticmethod
    def verify_files(
        original: Path,
        transformed: Path,
        solution: Path,
        mapped_path: Path,
        mapping: dict,
        tolerance: float | None = None,
    ) -> dict:
        from pitbench.problem_families.verification import CPFamily

        family = CPFamily()
        checked = family.verify(transformed, solution)
        new_to_original = mapping["new_to_original"]
        payload = json.loads(solution.read_text())
        mapped_bins = [
            [new_to_original[item] for item in bin_items]
            for bin_items in payload["bins"]
        ]
        BinPackingItemRepresentation.write_input(mapped_path, {"bins": mapped_bins})
        mapped = family.verify(original, mapped_path)
        return {
            "transformed": checked.model_dump(),
            "mapped_original": mapped.model_dump(),
            "objective_preserved": (
                checked.objective == mapped.objective
                if checked.objective is not None and mapped.objective is not None
                else None
            ),
        }


class SubsetSumRepresentation(BinPackingItemRepresentation):
    """Permute items in a balanced subset-sum model."""

    name = "subset_item_relabeling"

    @staticmethod
    def transform(model: dict, mapping: dict) -> dict:
        weights, order = model["weights"], mapping["new_to_original"]
        if sorted(order) != list(range(len(weights))):
            raise ValueError("subset item mapping must be a bijection")
        return {**model, "weights": [weights[index] for index in order]}

    @staticmethod
    def verify_files(original, transformed, solution, mapped_path, mapping, tolerance=None) -> dict:
        from pitbench.problem_families.verification import CPFamily

        family = CPFamily()
        checked = family.verify(transformed, solution)
        payload = json.loads(solution.read_text())
        order, selected = mapping["new_to_original"], payload.get("selected_items")
        if not isinstance(selected, list) or any(type(i) is not int or not 0 <= i < len(order) for i in selected):
            raise ValueError("subset solution contains invalid item indices")
        mapped_path.write_text(json.dumps({"selected_items": [order[i] for i in selected]}) + "\n")
        mapped = family.verify(original, mapped_path)
        return {
            "transformed": checked.model_dump(), "mapped_original": mapped.model_dump(),
            "objective_preserved": checked.objective == mapped.objective
            if checked.objective is not None and mapped.objective is not None else None,
        }


class ProjectActivityRepresentation(BinPackingItemRepresentation):
    """Relabel activities using a topological permutation of the project DAG."""

    name = "activity_relabeling"

    @staticmethod
    def mappings(model: dict, count: int, generator: random.Random) -> list[dict]:
        activities, arcs = model.get("activities"), model.get("precedence_arcs")
        if not isinstance(activities, list) or not activities or not isinstance(arcs, list) or count < 1:
            raise ValueError("invalid project activity model")
        size = len(activities)
        successors, indegree = [[] for _ in range(size)], [0] * size
        for arc in arcs:
            if not isinstance(arc, list) or len(arc) != 2 or any(type(v) is not int or not 0 <= v < size for v in arc):
                raise ValueError("invalid project precedence arc")
            successors[arc[0]].append(arc[1])
            indegree[arc[1]] += 1
        seen, result = {tuple(range(size))}, []
        for _ in range(max(1000, count * 100)):
            remaining = indegree.copy()
            ready = [i for i, degree in enumerate(remaining) if degree == 0]
            order = []
            while ready:
                index = generator.randrange(len(ready))
                node = ready.pop(index)
                order.append(node)
                for successor in successors[node]:
                    remaining[successor] -= 1
                    if remaining[successor] == 0:
                        ready.append(successor)
            if len(order) != size:
                raise ValueError("project precedence graph has a cycle")
            if tuple(order) not in seen:
                seen.add(tuple(order))
                result.append({"new_to_original": order})
                if len(result) == count:
                    return result
        raise ValueError("not enough distinct topological activity permutations")

    @staticmethod
    def transform(model: dict, mapping: dict) -> dict:
        activities, order = model["activities"], mapping["new_to_original"]
        if sorted(order) != list(range(len(activities))):
            raise ValueError("activity mapping must be a bijection")
        inverse = {old: new for new, old in enumerate(order)}
        arcs = sorted([inverse[a], inverse[b]] for a, b in model["precedence_arcs"])
        if any(a >= b for a, b in arcs):
            raise ValueError("activity mapping must preserve topological order")
        reordered = [{**activities[old], "id": new} for new, old in enumerate(order)]
        return {
            **model,
            "activities": reordered,
            "precedence_arcs": arcs,
            "durations": [x["duration"] for x in reordered],
            "resource_demands_flat": [v for x in reordered for v in x["resource_demands"]],
            "precedence_endpoints": [v for arc in arcs for v in arc],
        }

    @staticmethod
    def verify_files(original, transformed, solution, mapped_path, mapping, tolerance=None) -> dict:
        from pitbench.problem_families.verification import CPFamily

        family = CPFamily()
        checked = family.verify(transformed, solution)
        order = mapping["new_to_original"]
        starts = json.loads(solution.read_text()).get("start_times")
        if not isinstance(starts, list) or len(starts) != len(order):
            raise ValueError("project solution has incomplete start times")
        restored = [None] * len(order)
        for new, old in enumerate(order):
            restored[old] = starts[new]
        mapped_path.write_text(json.dumps({"start_times": restored}) + "\n")
        mapped = family.verify(original, mapped_path)
        return {
            "transformed": checked.model_dump(), "mapped_original": mapped.model_dump(),
            "objective_preserved": checked.objective == mapped.objective
            if checked.objective is not None and mapped.objective is not None else None,
        }


class GraphColoringRepresentation(RepresentationPlugin):
    """Permute graph vertices and map color assignments back to the source."""

    name = "vertex_relabeling"
    family = "cp"
    requires_tolerance = False
    suffix = ".json"
    generator = staticmethod(random.Random)

    @staticmethod
    def verifier(tolerance=None):
        from pitbench.problem_families.verification import CPFamily
        return CPFamily()

    @staticmethod
    def assert_same(original, restored) -> None:
        if original != restored:
            raise ValueError("graph input changed")

    @staticmethod
    def read_input(path: Path) -> dict:
        return json.loads(path.read_text())

    @staticmethod
    def write_input(path: Path, model: dict) -> None:
        path.write_text(json.dumps(model, indent=2, allow_nan=False) + "\n")

    @staticmethod
    def mappings(model: dict, count: int, generator: random.Random) -> list[dict]:
        size = model.get("vertex_count")
        if type(size) is not int or size < 2 or count < 1 or math.factorial(size) - 1 < count:
            raise ValueError("not enough distinct graph vertex permutations")
        seen = {tuple(range(size))}
        result = []
        while len(result) < count:
            order = list(range(size))
            generator.shuffle(order)
            if tuple(order) not in seen:
                seen.add(tuple(order))
                result.append({"new_to_original": order})
        return result

    @staticmethod
    def transform(model: dict, mapping: dict) -> dict:
        size, order = model["vertex_count"], mapping["new_to_original"]
        endpoints = model["edge_endpoints"]
        if sorted(order) != list(range(size)) or len(endpoints) % 2:
            raise ValueError("invalid graph or vertex permutation")
        inverse = {old: new for new, old in enumerate(order)}
        edges = []
        for first, second in zip(endpoints[::2], endpoints[1::2], strict=True):
            if type(first) is not int or type(second) is not int or not 0 <= first < second < size:
                raise ValueError("invalid graph edge")
            edges.append(tuple(sorted((inverse[first], inverse[second]))))
        return {**model, "edge_endpoints": [v for edge in sorted(edges) for v in edge]}

    @staticmethod
    def verify_files(original, transformed, solution, mapped_path, mapping, tolerance=None) -> dict:
        from pitbench.problem_families.verification import CPFamily

        family = CPFamily()
        checked = family.verify(transformed, solution)
        payload = json.loads(solution.read_text())
        order = mapping["new_to_original"]
        colors = payload.get("colors")
        if not isinstance(colors, list) or len(colors) != len(order):
            raise ValueError("color assignment does not cover the graph")
        restored = [None] * len(order)
        for new, old in enumerate(order):
            restored[old] = colors[new]
        mapped_path.write_text(json.dumps({"colors": restored}) + "\n")
        mapped = family.verify(original, mapped_path)
        return {
            "transformed": checked.model_dump(),
            "mapped_original": mapped.model_dump(),
            "objective_preserved": checked.objective == mapped.objective
            if checked.objective is not None and mapped.objective is not None else None,
        }


class JobOrderRepresentation(RepresentationPlugin):
    """Reorder jobs in a native text job-shop instance."""

    name = "job_order_relabeling"
    family = "cp"
    requires_tolerance = False
    suffix = ".txt"
    generator = staticmethod(random.Random)

    @staticmethod
    def verifier(tolerance=None):
        return NativeCPRepresentation.verifier(tolerance)

    @staticmethod
    def assert_same(original, restored) -> None:
        if original != restored:
            raise ValueError("job-shop input changed")

    @staticmethod
    def read_input(path: Path) -> dict:
        from pitbench.instances.generate import _job_shop_operations

        jobs = _job_shop_operations(path)
        return {"jobs": jobs, "machines": len(jobs[0])}

    @staticmethod
    def write_input(path: Path, model: dict) -> None:
        jobs = model["jobs"]
        lines = [f"{len(jobs)} {model['machines']}"]
        lines.extend(" ".join(str(value) for operation in job for value in operation) for job in jobs)
        path.write_text("\n".join(lines) + "\n")

    @staticmethod
    def mappings(model: dict, count: int, generator: random.Random) -> list[dict]:
        size = len(model["jobs"])
        if size < 2 or count < 1 or math.factorial(size) - 1 < count:
            raise ValueError("not enough distinct job permutations")
        seen = {tuple(range(size))}
        result = []
        while len(result) < count:
            order = list(range(size))
            generator.shuffle(order)
            if tuple(order) not in seen:
                seen.add(tuple(order))
                result.append({"new_to_original": order})
        return result

    @staticmethod
    def transform(model: dict, mapping: dict) -> dict:
        jobs, order = model["jobs"], mapping["new_to_original"]
        if sorted(order) != list(range(len(jobs))):
            raise ValueError("job mapping must be a bijection")
        return {**model, "jobs": [jobs[index] for index in order]}

    @staticmethod
    def verify_files(original, transformed, solution, mapped_path, mapping, tolerance=None) -> dict:
        from pitbench.problem_families.verification import _check_job_shop_start_times

        original_jobs = JobOrderRepresentation.read_input(original)["jobs"]
        transformed_jobs = JobOrderRepresentation.read_input(transformed)["jobs"]
        values = json.loads(solution.read_text()).get("start_times")
        if not isinstance(values, list):
            raise ValueError("job-shop solution has no start times")
        checked = _check_job_shop_start_times(values, transformed_jobs)
        order = mapping["new_to_original"]
        if sorted(order) != list(range(len(original_jobs))):
            raise ValueError("job mapping must be a bijection")
        by_job, cursor = [None] * len(order), 0
        for new, old in enumerate(order):
            count = len(transformed_jobs[new])
            by_job[old] = values[cursor:cursor + count]
            cursor += count
        mapped_values = [value for job in by_job for value in job]
        mapped_path.write_text(json.dumps({"start_times": mapped_values}) + "\n")
        mapped = _check_job_shop_start_times(mapped_values, original_jobs)
        return {
            "transformed": checked.model_dump(), "mapped_original": mapped.model_dump(),
            "objective_preserved": checked.objective == mapped.objective
            if checked.objective is not None and mapped.objective is not None else None,
        }


class JobShopRepresentation(RepresentationPlugin):
    """Verify precomputed job permutations of a fixed CP-SAT model."""

    name = "job_relabeling"
    family = "cp"
    requires_tolerance = False
    suffix = ".pb"
    generator = staticmethod(random.Random)

    @staticmethod
    def verifier(tolerance=None):
        # The repository plugin prepares the independent fixed-model verifier.
        return None

    @staticmethod
    def assert_same(original, restored) -> None:
        if original != restored:
            raise ValueError("representation input changed")

    @staticmethod
    def read_input(path: Path) -> bytes:
        return path.read_bytes()

    @staticmethod
    def write_input(path: Path, model: bytes) -> None:
        path.write_bytes(model)

    @staticmethod
    def mappings(model, count: int, generator) -> list:
        raise ValueError("job relabeling requires a verified precomputed manifest")

    @staticmethod
    def transform(model, mapping):
        raise ValueError("job relabeling requires a verified precomputed manifest")

    @staticmethod
    def verify_reference(reference_path: Path, anchor: float, jobs: list) -> None:
        from pitbench.problem_families.verification import _check_job_shop_start_times

        start_times = json.loads(reference_path.read_text())["start_times"]
        result = _check_job_shop_start_times(start_times, jobs)
        if not result.feasible or result.objective != anchor:
            raise ValueError("job-shop reference schedule differs from its target")

    @staticmethod
    def verify_files(
        original: Path,
        transformed: Path,
        solution: Path,
        mapped_path: Path,
        mapping: dict,
        tolerance: float | None = None,
    ) -> dict:
        from pitbench.problem_families.verification import _check_job_shop_start_times

        if (
            hashlib.sha256(original.read_bytes()).hexdigest()
            != mapping["original_sha256"]
        ):
            raise ValueError("original job-shop model hash differs")
        if (
            hashlib.sha256(transformed.read_bytes()).hexdigest()
            != mapping["transformed_sha256"]
        ):
            raise ValueError("transformed job-shop model hash differs")
        original_jobs = mapping["jobs"]
        new_to_original = mapping["new_to_original"]
        transformed_jobs = [original_jobs[index] for index in new_to_original]
        start_indices = mapping["start_variable_indices"]
        values = json.loads(solution.read_text())["values"]
        if any(type(value) is not int for value in values):
            raise ValueError("job-shop solution contains noninteger values")
        if any(
            type(index) is not int or not 0 <= index < len(values)
            for index in start_indices
        ):
            raise ValueError("job-shop start-variable mapping is invalid")
        transformed_starts = [values[index] for index in start_indices]
        checked = _check_job_shop_start_times(transformed_starts, transformed_jobs)
        mapped_by_job = [None] * len(original_jobs)
        cursor = 0
        for new_job, old_job in enumerate(new_to_original):
            operation_count = len(transformed_jobs[new_job])
            mapped_by_job[old_job] = transformed_starts[
                cursor : cursor + operation_count
            ]
            cursor += operation_count
        mapped_starts = [start for job in mapped_by_job for start in job]
        mapped_path.write_text(json.dumps({"start_times": mapped_starts}) + "\n")
        mapped = _check_job_shop_start_times(mapped_starts, original_jobs)
        return {
            "transformed": checked.model_dump(),
            "mapped_original": mapped.model_dump(),
            "objective_preserved": (
                checked.objective == mapped.objective
                if checked.objective is not None and mapped.objective is not None
                else None
            ),
        }


VECTOR_FIELDS = ("cost", "col_lower", "col_upper", "integrality", "col_names")

ROW_FIELDS = ("row_lower", "row_upper", "row_names")


class MPSRowRepresentation(RepresentationPlugin):
    """Reorder MPS constraints while retaining original row and column names."""

    name = "mps_row_permutation"
    family = "mip"
    requires_tolerance = False
    suffix = ".mps"
    generator = staticmethod(random.Random)

    @staticmethod
    def verifier(tolerance=None):
        from pitbench.problem_families.verification import MIPFamily
        return MIPFamily()

    @staticmethod
    def assert_same(original, restored) -> None:
        if original != restored:
            raise ValueError("MPS input changed")

    @staticmethod
    def read_input(path: Path) -> str:
        return path.read_text()

    @staticmethod
    def write_input(path: Path, model: str) -> None:
        path.write_text(model)

    @staticmethod
    def _row_positions(model: str) -> list[int]:
        lines = model.splitlines()
        rows = [i for i, line in enumerate(lines) if line.strip().upper() == "ROWS"]
        columns = [i for i, line in enumerate(lines) if line.strip().upper() == "COLUMNS"]
        if len(rows) != 1 or len(columns) != 1 or rows[0] >= columns[0]:
            raise ValueError("MPS must contain one ROWS section before COLUMNS")
        positions = []
        for index in range(rows[0] + 1, columns[0]):
            line = lines[index].strip()
            if not line or line.startswith("*"):
                continue
            fields = line.split()
            if len(fields) != 2 or fields[0].upper() not in {"N", "L", "G", "E"}:
                raise ValueError("unsupported MPS row declaration")
            if fields[0].upper() != "N":
                positions.append(index)
        return positions

    @staticmethod
    def mappings(model: str, count: int, generator: random.Random) -> list[dict]:
        size = len(MPSRowRepresentation._row_positions(model))
        if size < 2 or count < 1 or math.factorial(size) - 1 < count:
            raise ValueError("not enough distinct MPS row permutations")
        seen, result = {tuple(range(size))}, []
        while len(result) < count:
            order = list(range(size))
            generator.shuffle(order)
            if tuple(order) not in seen:
                seen.add(tuple(order))
                result.append({"new_to_original": order})
        return result

    @staticmethod
    def transform(model: str, mapping: dict) -> str:
        positions = MPSRowRepresentation._row_positions(model)
        order = mapping["new_to_original"]
        if sorted(order) != list(range(len(positions))):
            raise ValueError("MPS row mapping must be a bijection")
        lines = model.splitlines(keepends=True)
        rows = [lines[index] for index in positions]
        for index, old in zip(positions, order, strict=True):
            lines[index] = rows[old]
        return "".join(lines)

    @staticmethod
    def verify_files(original, transformed, solution, mapped_path, mapping, tolerance=None) -> dict:
        from pitbench.problem_families.verification import MIPFamily

        family = MIPFamily()
        checked = family.verify(transformed, solution)
        mapped_path.write_bytes(solution.read_bytes())
        mapped = family.verify(original, mapped_path)
        return {
            "transformed": checked.model_dump(), "mapped_original": mapped.model_dump(),
            "objective_preserved": checked.objective == mapped.objective
            if checked.objective is not None and mapped.objective is not None else None,
        }


class LinearModelRepresentation(RepresentationPlugin):
    """MILP row/column permutations and independent primal arithmetic checks."""

    runtime_dependencies = ("numpy", "scipy")
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
        if len(types) == 0:
            types = np.zeros(model.num_col_, dtype=np.int32)
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
