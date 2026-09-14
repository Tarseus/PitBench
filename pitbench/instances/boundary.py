"""Small, feasible boundary cases with explicit input and output contracts."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True)
class BoundaryCase:
    name: str
    description: str
    data: dict
    reference_solution: dict


class BoundarySuite(ABC):
    problem_family: str
    _suites: ClassVar[dict[str, type[BoundarySuite]]] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        family = getattr(cls, "problem_family", None)
        if family:
            existing = cls._suites.get(family)
            if existing is not None and existing is not cls:
                raise ValueError(f"duplicate boundary suite: {family}")
            cls._suites[family] = cls

    @staticmethod
    @abstractmethod
    def cases() -> list[BoundaryCase]: ...

    @staticmethod
    @abstractmethod
    def write(case: BoundaryCase, directory: Path) -> Path: ...

    @staticmethod
    @abstractmethod
    def verifier(): ...


class RoutingBoundarySuite(BoundarySuite):
    problem_family = "cvrp"

    @staticmethod
    def verifier():
        from pitbench.problem_families.verification import CVRPFamily

        return CVRPFamily()

    """Exercise route representation and capacity boundaries on integer distances."""

    @staticmethod
    def cases() -> list[BoundaryCase]:
        def case(name, description, coordinates, demands, capacity, routes):
            return BoundaryCase(
                name,
                description,
                {
                    "name": name,
                    "depot": 0,
                    "coordinates": coordinates,
                    "demands": demands,
                    "capacity": capacity,
                    "distance_metric": "EUC_2D",
                },
                {"routes": routes},
            )

        return [
            case(
                "single_customer",
                "One customer requiring one route.",
                [[0, 0], [3, 4]],
                [0, 1],
                1,
                [[1]],
            ),
            case(
                "coincident_customers",
                "Distinct customers at the depot; zero-cost routes.",
                [[0, 0], [0, 0], [0, 0], [0, 0]],
                [0, 1, 1, 1],
                3,
                [[1, 2, 3]],
            ),
            case(
                "exact_capacity",
                "Each customer exactly fills a vehicle.",
                [[0, 0], [3, 4], [-3, 4], [0, -5]],
                [0, 5, 5, 5],
                5,
                [[1], [2], [3]],
            ),
            case(
                "symmetric_routes",
                "Symmetric customers and tied route costs.",
                [[0, 0], [5, 0], [0, 5], [-5, 0], [0, -5]],
                [0, 1, 1, 1, 1],
                2,
                [[1, 2], [3, 4]],
            ),
        ]

    @staticmethod
    def write(case: BoundaryCase, directory: Path) -> Path:
        path = directory / f"{case.name}.json"
        path.write_text(json.dumps(case.data, indent=2) + "\n")
        return path


class LinearBoundarySuite(BoundarySuite):
    problem_family = "mip"

    @staticmethod
    def verifier():
        from pitbench.problem_families.verification import IntegerBoundaryFamily

        return IntegerBoundaryFamily()

    """Tiny integer models; feasibility and objective checks need no solver parser."""

    @staticmethod
    def cases() -> list[BoundaryCase]:
        def case(name, description, costs, lower, upper, constraints, solution):
            return BoundaryCase(
                name,
                description,
                {
                    "costs": costs,
                    "lower": lower,
                    "upper": upper,
                    "constraints": constraints,
                    "known_optimum": sum(
                        a * b for a, b in zip(costs, solution, strict=True)
                    ),
                },
                {"values": solution},
            )

        return [
            case(
                "fixed_variables",
                "Every integer variable has identical lower and upper bounds.",
                [1, 2],
                [1, 2],
                [1, 2],
                [([1, 1], "=", 3)],
                [1, 2],
            ),
            case(
                "presolve_empty",
                "An equality fixes all variables during presolve.",
                [1, 2],
                [0, 0],
                [1, 1],
                [([1, 1], "=", 0)],
                [0, 0],
            ),
            case(
                "redundant_constraints",
                "Duplicate and scaled copies of a redundant row.",
                [1, 2],
                [0, 0],
                [1, 1],
                [([1, 1], ">=", 1), ([1, 1], ">=", 1), ([2, 2], ">=", 2)],
                [1, 0],
            ),
            case(
                "equivalent_optima",
                "Several distinct integer assignments have the same optimum.",
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [([1, 1, 1, 1], "=", 2)],
                [1, 1, 0, 0],
            ),
        ]

    @staticmethod
    def write(case: BoundaryCase, directory: Path) -> Path:
        model = case.data

        def expression(coefficients):
            return " + ".join(
                f"{value} x{index}" for index, value in enumerate(coefficients)
            )

        lines = ["Minimize", f" obj: {expression(model['costs'])}", "Subject To"]
        for index, (coefficients, sense, rhs) in enumerate(model["constraints"]):
            lines.append(f" c{index}: {expression(coefficients)} {sense} {rhs}")
        lines.append("Bounds")
        for index, (lower, upper) in enumerate(
            zip(model["lower"], model["upper"], strict=True)
        ):
            lines.append(f" {lower} <= x{index} <= {upper}")
        lines.extend(
            [
                "Generals",
                " " + " ".join(f"x{i}" for i in range(len(model["costs"]))),
                "End",
            ]
        )
        path = directory / f"{case.name}.lp"
        path.write_text("\n".join(lines) + "\n")
        path.with_suffix(".model.json").write_text(json.dumps(model, indent=2) + "\n")
        return path


def boundary_suite(problem_family: str):
    family_name = getattr(problem_family, "value", problem_family)
    try:
        return BoundarySuite._suites[str(family_name)]
    except KeyError:
        raise ValueError(f"no boundary suite for {problem_family}") from None
