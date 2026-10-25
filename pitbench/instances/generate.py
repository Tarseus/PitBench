from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import platform
import random
import re
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

from pitbench.problem_families.public_verification import (
    check_cvrp,
    format_multidimensional_knapsack_lp,
)


def _choice(values: list[int], index: int) -> int:
    return values[index % len(values)]


def _instance_seed(config: dict[str, Any], index: int) -> int:
    selected_seeds = config.get("selected_instance_seeds")
    if selected_seeds is not None:
        if len(selected_seeds) != int(config["count"]):
            raise ValueError("selected_instance_seeds must match generator count")
        seed = selected_seeds[index]
        if type(seed) is not int:
            raise ValueError("selected instance seed must be an integer")
        return seed
    return int(config["randomness"]["instance_seed"]) + index


def _integer_samples(
    rng: random.Random,
    count: int,
    distribution: str,
    parameters: dict[str, Any],
    maximum: int,
) -> list[int]:
    """Shared positive-integer marginals for weights and processing times."""

    def interval(value: Any) -> tuple[int, int]:
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or any(type(item) is not int for item in value)
            or not 1 <= value[0] <= value[1]
        ):
            raise ValueError("ranges require two positive ordered integers")
        return value[0], value[1]

    if distribution == "uniform_integer":
        if set(parameters) - {"low", "high"}:
            raise ValueError("unknown uniform integer parameter")
        low, high = interval(
            [parameters.get("low", 1), parameters.get("high", maximum)]
        )
        return [rng.randint(low, high) for _ in range(count)]
    if distribution == "bimodal":
        if set(parameters) - {"low", "high", "high_probability"}:
            raise ValueError("unknown bimodal integer parameter")
        low = interval(parameters["low"])
        high = interval(parameters["high"])
        probability = parameters.get("high_probability", 0.5)
        if not 0 <= probability <= 1:
            raise ValueError("high_probability must be in [0, 1]")
        return [
            rng.randint(*(low if rng.random() < 1 - probability else high))
            for _ in range(count)
        ]
    if distribution == "discrete":
        if set(parameters) - {"values", "weights"}:
            raise ValueError("unknown discrete integer parameter")
        values, weights = parameters["values"], parameters["weights"]
        if (
            not isinstance(values, list)
            or not values
            or any(type(value) is not int or value < 1 for value in values)
            or not isinstance(weights, list)
            or len(values) != len(weights)
            or any(
                type(weight) not in {int, float}
                or not math.isfinite(weight)
                or weight < 0
                for weight in weights
            )
            or not 0 < sum(weights) < math.inf
        ):
            raise ValueError(
                "discrete integers require positive values and finite weights"
            )
        return rng.choices(values, weights=weights, k=count)
    raise ValueError(f"unsupported integer distribution: {distribution}")


def make_euclidean_cvrp_instance(
    *,
    name: str,
    customers: int,
    coordinate_seed: int,
    demand_seed: int,
    capacity_ratio: float | None = None,
    capacity: int | None = None,
    coordinate_distribution: str = "uniform",
    cluster_count: int = 4,
    cluster_spread: float = 9.0,
    uniform_fraction: float = 0.5,
    depot_mode: str = "center",
    demand_distribution: str = "uniform_integer",
    demand_parameters: dict[str, Any] | None = None,
    distance_metric: str | None = None,
) -> dict[str, Any]:
    """Generate a deterministic normalized CVRP instance.

    Independent spatial and demand streams allow matched distribution experiments.
    Existing defaults preserve historical generated instances byte for byte.
    """
    if type(customers) is not int or customers <= 0:
        raise ValueError("customers must be a positive integer")
    if (capacity is None) == (capacity_ratio is None):
        raise ValueError("specify exactly one of capacity and capacity_ratio")
    if capacity_ratio is not None and not 0 < capacity_ratio <= 1:
        raise ValueError("capacity_ratio must be in (0, 1]")
    if capacity is not None and (type(capacity) is not int or capacity <= 0):
        raise ValueError("capacity must be a positive integer")
    if distance_metric not in {None, "EXACT_2D", "EUC_2D"}:
        raise ValueError("unsupported CVRP distance metric")
    coordinate_rng = random.Random(coordinate_seed)
    demand_rng = random.Random(demand_seed)
    if coordinate_distribution == "uniform":
        customer_coordinates = [
            [coordinate_rng.uniform(0, 100), coordinate_rng.uniform(0, 100)]
            for _ in range(customers)
        ]
    elif coordinate_distribution in {"clustered", "mixed"}:
        if (
            type(cluster_count) is not int
            or cluster_count <= 0
            or not math.isfinite(cluster_spread)
            or cluster_spread <= 0
        ):
            raise ValueError("cluster_count and cluster_spread must be positive")
        if not 0 <= uniform_fraction <= 1:
            raise ValueError("uniform_fraction must be in [0, 1]")
        centers = [
            [coordinate_rng.uniform(15, 85), coordinate_rng.uniform(15, 85)]
            for _ in range(cluster_count)
        ]
        customer_coordinates = []
        mixture_rng = random.Random(f"cvrp-mixture:{coordinate_seed}")
        for index in range(customers):
            if (
                coordinate_distribution == "mixed"
                and mixture_rng.random() < uniform_fraction
            ):
                customer_coordinates.append(
                    [coordinate_rng.uniform(0, 100), coordinate_rng.uniform(0, 100)]
                )
                continue
            center_x, center_y = centers[index % cluster_count]
            customer_coordinates.append(
                [
                    min(
                        100.0, max(0.0, coordinate_rng.gauss(center_x, cluster_spread))
                    ),
                    min(
                        100.0, max(0.0, coordinate_rng.gauss(center_y, cluster_spread))
                    ),
                ]
            )
    else:
        raise ValueError(f"unknown coordinate_distribution: {coordinate_distribution}")
    if depot_mode == "center":
        depot_coordinates = [50.0, 50.0]
    elif depot_mode == "corner":
        depot_coordinates = [0.0, 0.0]
    elif depot_mode == "random":
        # A separate stream keeps the existing coordinate realization unchanged.
        depot_rng = random.Random(f"cvrp-depot:{coordinate_seed}")
        depot_coordinates = [
            depot_rng.uniform(0, 100),
            depot_rng.uniform(0, 100),
        ]
    else:
        raise ValueError(f"unknown depot_mode: {depot_mode}")

    parameters = demand_parameters or {}

    def positive_range(value: Any) -> tuple[int, int]:
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or any(type(item) is not int for item in value)
            or not 1 <= value[0] <= value[1]
        ):
            raise ValueError("demand ranges require two positive ordered integers")
        return value[0], value[1]

    if demand_distribution in {
        "uniform_integer",
        "depot_correlated",
        "depot_anticorrelated",
    }:
        if set(parameters) - {"low", "high"}:
            raise ValueError("unknown uniform demand parameter")
        low, high = positive_range(
            [parameters.get("low", 1), parameters.get("high", 10)]
        )
        customer_demands = [demand_rng.randint(low, high) for _ in range(customers)]
    elif demand_distribution == "bimodal":
        if set(parameters) - {"low", "high", "high_probability"}:
            raise ValueError("unknown bimodal demand parameter")
        low = positive_range(parameters.get("low", [1, 3]))
        high = positive_range(parameters.get("high", [8, 10]))
        high_probability = parameters.get("high_probability", 0.5)
        if not 0 <= high_probability <= 1:
            raise ValueError("high_probability must be in [0, 1]")
        customer_demands = [
            demand_rng.randint(*low)
            if demand_rng.random() < 1 - high_probability
            else demand_rng.randint(*high)
            for _ in range(customers)
        ]
    elif demand_distribution == "discrete":
        if set(parameters) - {"values", "weights"}:
            raise ValueError("unknown discrete demand parameter")
        values = parameters.get("values")
        weights = parameters.get("weights")
        if (
            not isinstance(values, list)
            or not values
            or any(type(value) is not int or value <= 0 for value in values)
            or not isinstance(weights, list)
            or len(weights) != len(values)
            or any(
                type(weight) not in {int, float}
                or not math.isfinite(weight)
                or weight < 0
                for weight in weights
            )
            or not 0 < sum(weights) < math.inf
        ):
            raise ValueError(
                "discrete demands require positive values and finite weights"
            )
        customer_demands = demand_rng.choices(values, weights=weights, k=customers)
    else:
        raise ValueError(f"unknown demand_distribution: {demand_distribution}")

    if demand_distribution in {"depot_correlated", "depot_anticorrelated"}:
        # Keep the sampled marginal while coupling its order to depot distance.
        sampled = sorted(customer_demands)
        by_distance = sorted(
            range(customers),
            key=lambda index: (
                math.dist(customer_coordinates[index], depot_coordinates),
                index,
            ),
            reverse=demand_distribution == "depot_anticorrelated",
        )
        customer_demands = [0] * customers
        for index, demand in zip(by_distance, sampled, strict=True):
            customer_demands[index] = demand
    demands = [0, *customer_demands]
    if capacity is None:
        assert capacity_ratio is not None
        capacity = max(10, math.ceil(sum(demands) * capacity_ratio))
    if max(customer_demands) > capacity:
        raise ValueError("customer demand exceeds vehicle capacity")
    instance = {
        "name": name,
        "depot": 0,
        "coordinates": [depot_coordinates, *customer_coordinates],
        "demands": demands,
        "capacity": capacity,
    }
    if distance_metric is not None:
        instance["distance_metric"] = distance_metric
    return instance


def _cvrp(config: dict[str, Any], index: int, destination: Path) -> None:
    seeds = config["randomness"]
    instance = make_euclidean_cvrp_instance(
        name=destination.stem,
        customers=_choice(config["customers"], index),
        coordinate_seed=seeds["coordinate_seed"] + index,
        demand_seed=seeds["demand_seed"] + index,
        capacity_ratio=config.get("capacity_ratio"),
        capacity=config.get("capacity"),
        coordinate_distribution=config.get("coordinate_distribution", "uniform"),
        cluster_count=int(config.get("cluster_count", 4)),
        cluster_spread=float(config.get("cluster_spread", 9.0)),
        uniform_fraction=float(config.get("uniform_fraction", 0.5)),
        depot_mode=config.get("depot_mode", "center"),
        demand_distribution=config.get("demand_distribution", "uniform_integer"),
        demand_parameters=config.get("demand_parameters"),
        distance_metric=config.get("distance_metric"),
    )
    destination.write_text(json.dumps(instance, indent=2))


def make_bin_packing_instance(
    *,
    name: str,
    item_count: int,
    capacity: int,
    instance_seed: int,
    weight_distribution: str = "uniform_integer",
    weight_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if item_count <= 0 or capacity <= 0:
        raise ValueError("item count and capacity must be positive")
    rng = random.Random(instance_seed)
    weights = _integer_samples(
        rng, item_count, weight_distribution, weight_parameters or {}, capacity
    )
    if max(weights) > capacity:
        raise ValueError("bin packing item exceeds capacity")
    return {
        "name": name,
        "capacity": capacity,
        "weights": weights,
    }


def _bin_packing(config: dict[str, Any], index: int, destination: Path) -> None:
    item_count = _choice(config["items"], index)
    capacity = int(config["capacity"])
    destination.write_text(
        json.dumps(
            make_bin_packing_instance(
                name=destination.stem,
                item_count=item_count,
                capacity=capacity,
                instance_seed=_instance_seed(config, index),
                weight_distribution=config.get(
                    "weight_distribution", "uniform_integer"
                ),
                weight_parameters=config.get("weight_parameters"),
            ),
            indent=2,
        )
    )


def _ortools(config: dict[str, Any], index: int, destination: Path) -> None:
    destination.write_text(
        json.dumps(
            {
                "name": destination.stem,
                "intervals": _choice(config["intervals"], index),
                "constant_reuse_factor": _choice(
                    config["constant_reuse_factor"], index
                ),
                "seed": config["randomness"]["instance_seed"] + index,
            },
            indent=2,
        )
    )


def scheduling_processing_times(
    config: dict[str, Any],
    index: int,
) -> list[int]:
    rng = random.Random(_instance_seed(config, index))
    jobs = _choice(config["jobs"], index)
    return _integer_samples(
        rng,
        jobs,
        config.get("processing_distribution", "uniform_integer"),
        config.get("processing_parameters", {}),
        20,
    )


def _scheduling_lp(config: dict[str, Any], index: int, destination: Path) -> None:
    processing = scheduling_processing_times(config, index)
    jobs = len(processing)
    horizon = sum(processing)
    lines = ["Minimize", " obj: " + " + ".join(f"s{i}" for i in range(jobs))]
    lines.append("Subject To")
    constraint = 0
    binaries: list[str] = []
    for first in range(jobs):
        for second in range(first + 1, jobs):
            order = f"z{first}_{second}"
            binaries.append(order)
            lines.append(
                f" c{constraint}: s{first} - s{second} + {horizon} {order} <= "
                f"{horizon - processing[first]}"
            )
            constraint += 1
            lines.append(
                f" c{constraint}: s{second} - s{first} - {horizon} {order} <= "
                f"-{processing[second]}"
            )
            constraint += 1
    starts = [f"s{index}" for index in range(jobs)]
    lines.extend(["Bounds", *[f" 0 <= {name} <= {horizon}" for name in starts]])
    lines.extend(
        [
            "General",
            *[f" {name}" for name in starts],
            "Binary",
            *[f" {name}" for name in binaries],
            "End",
        ]
    )
    destination.write_text("\n".join(lines) + "\n")


def multidimensional_knapsack_data(
    config: dict[str, Any], index: int
) -> tuple[list[list[int]], list[int], list[int]]:
    """Plant a feasible packing whose value meets a linear dual bound."""

    rng = random.Random(_instance_seed(config, index))
    item_count = _choice(config["items"], index)
    dimensions = _choice(config["dimensions"], index)
    weight_max = int(config["weight_max"])
    if item_count < 2 or dimensions < 2 or weight_max < 2:
        raise ValueError("invalid multidimensional knapsack generator settings")
    weights = [
        _integer_samples(
            rng,
            item_count,
            config.get("weight_distribution", "uniform_integer"),
            config.get("weight_parameters", {}),
            weight_max,
        )
        for _ in range(dimensions)
    ]
    correlation = config.get("dimension_correlation", "independent")
    if correlation == "shared":
        weights = [weights[0].copy() for _ in range(dimensions)]
    elif correlation != "independent":
        raise ValueError("unsupported dimension_correlation")
    selected = sorted(rng.sample(range(item_count), item_count // 2))
    mode = config.get("capacity_mode", "planted")
    if mode == "planted":
        capacities = [sum(row[item] for item in selected) for row in weights]
    elif mode == "ratio":
        ratio = config.get("capacity_ratio", 0.5)
        if not 0 < ratio < 1:
            raise ValueError("knapsack capacity_ratio must be in (0, 1)")
        capacities = [max(1, int(sum(row) * ratio)) for row in weights]
        selected = []  # No optimal witness is planted in this mode.
    else:
        raise ValueError("unsupported knapsack capacity_mode")
    return weights, capacities, selected


def _multidimensional_knapsack_lp(
    config: dict[str, Any], index: int, destination: Path
) -> None:
    weights, capacities, _ = multidimensional_knapsack_data(config, index)
    destination.write_text(format_multidimensional_knapsack_lp(weights, capacities))


def binary_clique_cover_data(
    config: dict[str, Any], index: int
) -> tuple[list[list[str]], list[int], list[tuple[str, str]]]:
    """Build a binary set-packing MIP with a matching clique-cover certificate.

    Each group is a clique encoded by a positively weighted binary row. The
    planted assignment selects one variable from every group. Extra conflicts
    are sampled only between non-planted variables, so the clique cover proves
    the planted solution optimal while retaining a nontrivial independent-set
    search problem.
    """
    seed = _instance_seed(config, index)
    rng = random.Random(seed)
    group_count = _choice(config["groups"], index)
    group_size = _choice(config["group_sizes"], index)
    coefficient_min = int(config["coefficient_min"])
    coefficient_max = int(config["coefficient_max"])
    cross_edge_density = float(_choice(config["cross_edge_density"], index))
    cross_edge_sampler = config.get("cross_edge_sampler", "scan_v1")
    if (
        group_count < 2
        or group_size < 2
        or coefficient_min <= 0
        or coefficient_max < coefficient_min
        or 2 * coefficient_min <= coefficient_max
        or not 0 <= cross_edge_density < 1
        or not isinstance(cross_edge_sampler, str)
        or cross_edge_sampler not in {"scan_v1", "geometric_skip_v1"}
    ):
        raise ValueError("invalid binary clique-cover generator settings")

    groups = [
        [f"x{group * group_size + offset:06d}" for offset in range(group_size)]
        for group in range(group_count)
    ]
    planted = [rng.randrange(group_size) for _ in range(group_count)]
    planted_names = {
        group[chosen] for group, chosen in zip(groups, planted, strict=True)
    }
    cross_edges = []
    for first_group in range(group_count):
        for second_group in range(first_group + 1, group_count):
            if cross_edge_sampler == "scan_v1":
                for first in groups[first_group]:
                    for second in groups[second_group]:
                        if first in planted_names and second in planted_names:
                            continue
                        if rng.random() < cross_edge_density:
                            cross_edges.append((first, second))
            elif cross_edge_density > 0:
                # Geometric gaps generate the same independent Bernoulli edge
                # process as scanning every pair, while making sparse large
                # graph instances proportional to emitted edges.
                log_failure = math.log1p(-cross_edge_density)
                position = 0
                pair_count = group_size * group_size
                while position < pair_count:
                    position += int(math.log1p(-rng.random()) / log_failure)
                    if position >= pair_count:
                        break
                    first_offset, second_offset = divmod(position, group_size)
                    first = groups[first_group][first_offset]
                    second = groups[second_group][second_offset]
                    if first not in planted_names or second not in planted_names:
                        cross_edges.append((first, second))
                    position += 1
    return groups, planted, cross_edges


def binary_clique_cover_fme_extensions(
    config: dict[str, Any], index: int, groups: list[list[str]], planted: list[int]
) -> list[dict[str, str]]:
    """Describe redundant continuous extensions that expose FME presolve.

    Each continuous column ``z`` appears in two positive- and two
    negative-coefficient inequality rows. The rows share a common binary core,
    with one cross-shared and two row-specific columns. This keeps the FME
    nonzero-reduction estimate positive while avoiding identical row supports
    that other default presolve rules can remove first. Mixed-sign rows with a
    zero RHS constrain non-witness assignments, while all planted witness
    variables are outside their supports. The clique-cover lower bound and
    planted feasible solution therefore continue to certify the same optimum.
    """
    pair_count = config.get("fme_auxiliary_pairs", 0)
    if type(pair_count) is not int or pair_count < 0:
        raise ValueError("fme_auxiliary_pairs must be a nonnegative integer")
    rows_per_sign = config.get("fme_rows_per_sign", 2)
    shared_count = config.get("fme_shared_binary_count", 5)
    coefficient_max = config.get("fme_coefficient_max", 3)
    if (
        type(rows_per_sign) is not int
        or rows_per_sign != 2
        or type(shared_count) is not int
        or shared_count < 5
        or type(coefficient_max) is not int
        or coefficient_max < 1
    ):
        raise ValueError("invalid FME auxiliary row structure")
    nonplanted_columns = [
        column
        for group, selected in zip(groups, planted, strict=True)
        for offset, column in enumerate(group)
        if offset != selected
    ]
    support_size = shared_count + 3
    if pair_count * support_size > len(nonplanted_columns):
        raise ValueError("FME extensions require disjoint nonplanted binary supports")
    if pair_count == 0:
        return []
    rng = random.Random(f"clique-fme-extension:{_instance_seed(config, index)}")
    selected = rng.sample(nonplanted_columns, pair_count * support_size)
    extensions = []
    for pair in range(pair_count):
        offset = pair * support_size
        shared = selected[offset : offset + shared_count]
        extras = selected[offset + shared_count : offset + support_size]
        positive_rows = []
        negative_rows = []
        for row_index in range(rows_per_sign):
            positive_columns = [*shared, extras[row_index]]
            negative_columns = [*shared, extras[0 if row_index == 0 else 2]]

            def signed_row(columns: list[str]) -> dict[str, float]:
                signs = [-1] * (len(columns) // 2) + [1] * (
                    len(columns) - len(columns) // 2
                )
                rng.shuffle(signs)
                result = {}
                for column, sign in zip(columns, signs, strict=True):
                    magnitude = rng.randint(1, coefficient_max * 10 - 1) / 10
                    if magnitude.is_integer():
                        magnitude += 0.1
                    result[column] = sign * magnitude
                return result

            positive_coefficients = signed_row(positive_columns)
            negative_coefficients = signed_row(negative_columns)
            positive_rows.append(
                {
                    "row": f"FMEUP{pair:06d}_{row_index:02d}",
                    "binary_coefficients": positive_coefficients,
                    "rhs": 0,
                }
            )
            negative_rows.append(
                {
                    "row": f"FMEDN{pair:06d}_{row_index:02d}",
                    "binary_coefficients": negative_coefficients,
                    "rhs": 0,
                }
            )
        extensions.append(
            {
                "column": f"zFME{pair:06d}",
                "positive_rows": positive_rows,
                "negative_rows": negative_rows,
            }
        )
    return extensions


def _binary_clique_cover_mps(
    config: dict[str, Any], index: int, destination: Path
) -> None:
    groups, planted, cross_edges = binary_clique_cover_data(config, index)
    fme_extensions = binary_clique_cover_fme_extensions(config, index, groups, planted)
    fme_paper_block = config.get("fme_paper_block", False)
    if type(fme_paper_block) is not bool:
        raise ValueError("fme_paper_block must be a boolean")
    rows: list[tuple[str, list[tuple[str, float]], float]] = []
    for group_index, group in enumerate(groups):
        values = [
            random.Random(
                f"clique-coefficient:{_instance_seed(config, index)}:{group_index}:{name}"
            ).randint(int(config["coefficient_min"]), int(config["coefficient_max"]))
            for name in group
        ]
        rhs = max(values)
        rows.append((f"C{group_index:06d}", list(zip(group, values, strict=True)), rhs))
    for edge_index, (first, second) in enumerate(cross_edges):
        rows.append((f"E{edge_index:06d}", [(first, 1), (second, 1)], 1))
    for extension in fme_extensions:
        rows.extend(
            (
                row["row"],
                [(extension["column"], 1), *row["binary_coefficients"].items()],
                row["rhs"],
            )
            for row in extension["positive_rows"]
        )
        rows.extend(
            (
                row["row"],
                [(extension["column"], -1), *row["binary_coefficients"].items()],
                row["rhs"],
            )
            for row in extension["negative_rows"]
        )
    if fme_paper_block:
        rows.extend(
            [
                ("FMEPAPER1", [("fme_i", -1), ("fme_x", 1), ("fme_z", -1)], -30),
                ("FMEPAPER2", [("fme_x", 2), ("fme_y", 1), ("fme_z", 2)], 50),
                ("FMEPAPER3", [("fme_x", 3), ("fme_y", -1), ("fme_z", 3)], 40),
            ]
        )

    lines = [
        f"NAME          {destination.stem}",
        "ROWS",
        " N  OBJ",
        *(f" L  {row_name}" for row_name, _, _ in rows),
        "COLUMNS",
    ]
    row_by_variable: dict[str, list[tuple[str, int]]] = {
        name: [] for group in groups for name in group
    }
    row_by_variable.update({item["column"]: [] for item in fme_extensions})
    if fme_paper_block:
        row_by_variable.update(
            {name: [] for name in ("fme_i", "fme_x", "fme_y", "fme_z")}
        )
    extension_columns = {item["column"] for item in fme_extensions}
    for row_name, terms, _ in rows:
        for name, value in terms:
            row_by_variable[name].append((row_name, value))
    for name, terms in row_by_variable.items():
        if name in extension_columns or name in {"fme_x", "fme_y", "fme_z"}:
            entries = []
        elif name == "fme_i":
            entries = [("OBJ", 1)]
        else:
            entries = [("OBJ", -1)]
        entries.extend(terms)
        for offset in range(0, len(entries), 2):
            fields = entries[offset : offset + 2]
            formatted = " ".join(f"{row} {value}" for row, value in fields)
            lines.append(f"    {name} {formatted}")
    lines.extend(["RHS"])
    for row_name, _, rhs in rows:
        lines.append(f"    RHS1 {row_name} {rhs}")
    lines.append("BOUNDS")
    for group in groups:
        for name in group:
            lines.append(f" BV BND1 {name}")
    for extension in fme_extensions:
        lines.append(f" FR BND1 {extension['column']}")
    if fme_paper_block:
        lines.extend([" LI BND1 fme_i 0", " UI BND1 fme_i 40"])
    lines.append("ENDATA")
    destination.write_text("\n".join(lines) + "\n")


def mycielski_graph(chromatic_number: int) -> tuple[set[tuple[int, int]], list[int]]:
    """Build the canonical graph M^(k-2)(K2) and its k-coloring."""

    if not 2 <= chromatic_number <= 9:
        raise ValueError("Mycielski chromatic number must be in [2, 9]")
    edges = {(0, 1)}
    colors = [0, 1]
    for new_color in range(2, chromatic_number):
        previous_count = len(colors)
        new_edges = set(edges)
        for first, second in edges:
            new_edges.add(
                (
                    min(first + previous_count, second),
                    max(first + previous_count, second),
                )
            )
            new_edges.add(
                (
                    min(second + previous_count, first),
                    max(second + previous_count, first),
                )
            )
        for vertex in range(previous_count):
            new_edges.add((vertex + previous_count, 2 * previous_count))
        edges = new_edges
        colors = [*colors, *colors, new_color]
    return edges, colors


def graph_coloring_data(
    config: dict[str, Any], index: int
) -> tuple[dict[str, Any], list[int], dict[str, Any]]:
    """Relabel a graph with a theorem-backed chromatic lower bound."""

    rng = random.Random(_instance_seed(config, index))
    model = config.get("graph_model", "mycielski")
    if model in {"erdos_renyi", "planted_partition"}:
        vertices = _choice(config["vertex_counts"], index)
        probability = float(config["edge_probability"])
        if vertices < 2 or not 0 <= probability <= 1:
            raise ValueError("invalid random graph dimensions or density")
        if model == "planted_partition":
            partitions = _choice(config["colors"], index)
            if not 1 <= partitions <= vertices:
                raise ValueError("invalid graph partition count")
            colors = [vertex % partitions for vertex in range(vertices)]
            rng.shuffle(colors)
        else:
            colors = list(range(vertices))  # Feasible witness, no optimality claim.
        edges = [
            (a, b)
            for a in range(vertices)
            for b in range(a + 1, vertices)
            if colors[a] != colors[b] and rng.random() < probability
        ]
        return (
            {
                "name": f"graph_{index:04d}",
                "problem_kind": "graph_coloring",
                "vertex_count": vertices,
                "edge_endpoints": [value for edge in edges for value in edge],
            },
            colors,
            {"kind": "feasible_coloring_only", "graph_model": model},
        )
    if model != "mycielski":
        raise ValueError(f"unsupported graph_model: {model}")
    chromatic_number = _choice(config["colors"], index)
    probability_config = config.get("extra_edge_probability", 0)
    extra_edge_probability = float(
        _choice(probability_config, index)
        if isinstance(probability_config, list)
        else probability_config
    )
    if not 0 <= extra_edge_probability < 1:
        raise ValueError("invalid graph coloring edge probability")
    canonical_edges, canonical_colors = mycielski_graph(chromatic_number)
    vertex_count = len(canonical_colors)
    permutation = list(range(vertex_count))
    rng.shuffle(permutation)
    old_to_new = {old: new for new, old in enumerate(permutation)}
    colors = [canonical_colors[old] for old in permutation]
    edges = {
        (
            min(old_to_new[first], old_to_new[second]),
            max(old_to_new[first], old_to_new[second]),
        )
        for first, second in canonical_edges
    }
    edges.update(
        (first, second)
        for first in range(vertex_count)
        for second in range(first + 1, vertex_count)
        if colors[first] != colors[second] and rng.random() < extra_edge_probability
    )
    return (
        {
            "name": f"graph_{index:04d}",
            "problem_kind": "graph_coloring",
            "vertex_count": vertex_count,
            "edge_endpoints": [endpoint for edge in sorted(edges) for endpoint in edge],
        },
        colors,
        {
            "kind": "mycielski_lower_bound",
            "chromatic_number": chromatic_number,
            "permutation": permutation,
        },
    )


def _graph_coloring(config: dict[str, Any], index: int, destination: Path) -> None:
    instance, _, _ = graph_coloring_data(config, index)
    instance["name"] = destination.stem
    destination.write_text(json.dumps(instance, indent=2) + "\n")


def balanced_subset_sum_instance(config: dict[str, Any], index: int) -> dict[str, Any]:
    """Generate data without allocating a reachability proof over its weight sum."""
    rng = random.Random(_instance_seed(config, index))
    item_count = int(_choice(config["item_counts"], index))
    maximum = int(config["max_weight"])
    step = int(config.get("weight_step", 1))
    if item_count < 2 or maximum < 1 or step < 1 or maximum // step < 1:
        raise ValueError("invalid balanced subset-sum generator dimensions")
    cardinality = int(config.get("cardinality", item_count // 2))
    if not 0 < cardinality < item_count:
        raise ValueError("invalid balanced subset-sum cardinality")
    domain_mode = _choice(config.get("domain_modes", ["bounded"]), index)
    if domain_mode not in {"bounded", "enumerated"}:
        raise ValueError("invalid balanced subset-sum domain mode")
    weights = [
        step * value
        for value in _integer_samples(
            rng,
            item_count,
            config.get("weight_distribution", "uniform_integer"),
            config.get("weight_parameters", {}),
            maximum // step,
        )
    ]
    total = sum(weights)
    offset = int(_choice(config.get("target_offsets", [0]), index))
    target = max(0, min(total, total * cardinality // item_count + offset))
    return {
        "name": f"subset_{index:04d}",
        "problem_kind": "balanced_subset_sum",
        "weights": weights,
        "cardinality": cardinality,
        "target": target,
        "domain_mode": domain_mode,
    }


def balanced_subset_sum_data(
    config: dict[str, Any], index: int
) -> tuple[dict[str, Any], list[int], dict[str, Any]]:
    """Generate a fixed-cardinality subset-sum objective with an exact DP proof."""
    instance = balanced_subset_sum_instance(config, index)
    weights, cardinality, target = (
        instance["weights"],
        instance["cardinality"],
        instance["target"],
    )
    total, item_count = sum(weights), len(weights)

    layers = [1] + [0] * cardinality
    snapshots = [layers.copy()]
    for position, weight in enumerate(weights):
        for count in range(min(cardinality, position + 1), 0, -1):
            layers[count] |= layers[count - 1] << weight
        snapshots.append(layers.copy())
    reachable = layers[cardinality]
    best_sum = min(
        (value for value in range(total + 1) if (reachable >> value) & 1),
        key=lambda value: (abs(value - target), value),
    )
    selected = []
    remaining_sum = best_sum
    remaining_count = cardinality
    for position in range(item_count - 1, -1, -1):
        if (snapshots[position][remaining_count] >> remaining_sum) & 1:
            continue
        weight = weights[position]
        if (
            remaining_count == 0
            or remaining_sum < weight
            or not (
                snapshots[position][remaining_count - 1] >> (remaining_sum - weight)
            )
            & 1
        ):
            raise ValueError("subset-sum proof reconstruction failed")
        selected.append(position)
        remaining_sum -= weight
        remaining_count -= 1
    if remaining_count or remaining_sum:
        raise ValueError("subset-sum proof reconstruction is incomplete")
    selected.sort()
    basis = {
        "kind": "fixed_cardinality_subset_sum_dp",
        "total_weight": total,
        "cardinality": cardinality,
        "target": target,
        "reachable_bitset_sha256": hashlib.sha256(
            reachable.to_bytes((total + 8) // 8, "little")
        ).hexdigest(),
    }
    return instance, selected, basis


def _balanced_subset_sum(config: dict[str, Any], index: int, destination: Path) -> None:
    instance = balanced_subset_sum_instance(config, index)
    instance["name"] = destination.stem
    destination.write_text(json.dumps(instance, indent=2) + "\n")


def job_shop_data(config: dict[str, Any], index: int) -> list[list[list[int]]]:
    """Random job-shop routes; a shared route produces a flow-shop structure."""
    jobs = _choice(config["jobs"], index)
    machines = _choice(config["machines"], index)
    if jobs < 1 or machines < 1:
        raise ValueError("job-shop dimensions must be positive")
    seed = _instance_seed(config, index)
    order_rng = random.Random(f"job-shop-order:{seed}")
    duration_rng = random.Random(seed)
    mode = config.get("machine_order", "random_per_job")
    if mode not in {"random_per_job", "shared"}:
        raise ValueError("unsupported job-shop machine_order")
    shared = list(range(machines))
    order_rng.shuffle(shared)
    result = []
    for _ in range(jobs):
        route = shared.copy()
        if mode == "random_per_job":
            order_rng.shuffle(route)
        durations = _integer_samples(
            duration_rng,
            machines,
            config.get("processing_distribution", "uniform_integer"),
            config.get("processing_parameters", {}),
            20,
        )
        result.append(
            [
                [machine, duration]
                for machine, duration in zip(route, durations, strict=True)
            ]
        )
    return result


def _job_shop(config: dict[str, Any], index: int, destination: Path) -> None:
    jobs = job_shop_data(config, index)
    destination.write_text(
        f"{len(jobs)} {len(jobs[0])}\n"
        + "\n".join(
            " ".join(str(value) for operation in job for value in operation)
            for job in jobs
        )
        + "\n"
    )


_GENERATORS = {
    "euclidean_cvrp": ("json", _cvrp),
    "bin_packing": ("json", _bin_packing),
    "repeated_constants_cumulative": ("json", _ortools),
    "single_machine_scheduling_mip": ("lp", _scheduling_lp),
    "multidimensional_knapsack_mip": ("lp", _multidimensional_knapsack_lp),
    "binary_clique_cover_mip": ("mps", _binary_clique_cover_mps),
    "graph_coloring": ("json", _graph_coloring),
    "balanced_subset_sum": ("json", _balanced_subset_sum),
    "job_shop_scheduling": ("txt", _job_shop),
}


def _cvrp_instance_features(path: Path) -> dict[str, Any]:
    instance = json.loads(path.read_text())
    depot = instance.get("depot", 0)
    customers = [i for i in range(len(instance["coordinates"])) if i != depot]
    demands = [instance["demands"][i] for i in customers]
    coordinates = [instance["coordinates"][i] for i in customers]
    distances = [
        math.dist(point, instance["coordinates"][depot]) for point in coordinates
    ]
    checked = check_cvrp(instance, {"routes": [[i] for i in customers]})
    if not checked["feasible"]:
        raise ValueError(f"generated CVRP instance is infeasible: {checked['detail']}")
    correlation = (
        statistics.correlation(demands, distances)
        if len(demands) > 1 and len(set(demands)) > 1 and len(set(distances)) > 1
        else None
    )
    return {
        "customers": len(customers),
        "capacity": instance["capacity"],
        "total_demand": sum(demands),
        "demand_min": min(demands),
        "demand_max": max(demands),
        "demand_mean": statistics.fmean(demands),
        "demand_std": statistics.pstdev(demands),
        "capacity_to_total_demand": instance["capacity"] / sum(demands),
        "route_count_capacity_lower_bound": math.ceil(
            sum(demands) / instance["capacity"]
        ),
        "coordinate_x_mean": statistics.fmean(p[0] for p in coordinates),
        "coordinate_y_mean": statistics.fmean(p[1] for p in coordinates),
        "coordinate_x_std": statistics.pstdev(p[0] for p in coordinates),
        "coordinate_y_std": statistics.pstdev(p[1] for p in coordinates),
        "distance_to_depot_mean": statistics.fmean(distances),
        "demand_depot_distance_correlation": correlation,
        "distance_metric": instance.get("distance_metric", "EXACT_2D"),
        "feasibility_witness": "one_customer_per_route",
        "feasibility_witness_objective": checked["objective"],
    }


_INSTANCE_FEATURES = {"euclidean_cvrp": _cvrp_instance_features}


def materialize_instance_suite(
    plan: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    """Generate several problem-family grids with the same manifest format."""
    panels = plan["panels"]
    if not panels or any(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) is None for name in panels
    ):
        raise ValueError("suite panels require safe nonempty names")
    output_dir.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": "1.0", "status": "incomplete", "panels": {}}
    try:
        for name, panel in panels.items():
            result["panels"][name] = {
                "status": "started",
                "manifest": f"{name}/manifest.json",
            }
            manifest = materialize_instance_grid(panel, output_dir / name)
            result["panels"][name].update(
                status=manifest["status"],
                generated_instances=manifest["generated_instances"],
            )
        result["status"] = "completed"
    finally:
        (output_dir / "suite.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _require_planted_optimum(config: dict[str, Any]) -> None:
    if (
        config["kind"] == "multidimensional_knapsack_mip"
        and config.get("capacity_mode", "planted") != "planted"
    ):
        raise ValueError("capacity-dual optimum requires planted capacities")
    if (
        config["kind"] == "graph_coloring"
        and config.get("graph_model", "mycielski") != "mycielski"
    ):
        raise ValueError("chromatic optimum requires a Mycielski lower-bound proof")


def materialize_generated_instance_set(
    payload: dict[str, Any],
    output_dir: Path,
    *,
    expected_visibility: str,
    stem_prefix: str = "dev",
) -> list[Path]:
    """Materialize a frozen instance-set config in its authorized environment."""

    if payload.get("visibility") != expected_visibility:
        raise ValueError(
            f"instance-set visibility must be {expected_visibility!r}, "
            f"got {payload.get('visibility')!r}"
        )
    config = payload["generator"]
    kind = config["kind"]
    if kind not in _GENERATORS:
        raise ValueError(f"unknown instance-set generator: {kind}")
    suffix, generator = _GENERATORS[kind]
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index in range(int(config["count"])):
        path = output_dir / f"{stem_prefix}_{index:04d}.{suffix}"
        generator(config, index, path)
        paths.append(path)
    index_path = output_dir / "instance_set_config.yaml"
    index_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "visibility": expected_visibility,
                "instances": [{"id": path.stem, "path": path.name} for path in paths],
            },
            sort_keys=False,
        )
    )
    return paths


def materialize_instance_grid(plan: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Cross configured generator overrides, preserving paired replicate seeds.

    Each axis maps level names to generator parameter overrides. All cells reuse
    the declared randomness; the existing generator adds the replicate index.
    Generated data and descriptors do not establish an optimum or a BKS.
    """
    if plan.get("schema_version", "1.0") != "1.0":
        raise ValueError("unsupported instance generation plan version")
    visibility = plan.get("visibility")
    if visibility not in {"agent", "judge"}:
        raise ValueError("generation plan visibility must be agent or judge")
    base = plan["generator"]
    axes = plan.get("axes", {})
    names = list(axes)
    assigned: set[str] = set()
    for name, levels in axes.items():
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) is None
            or not isinstance(levels, dict)
            or not levels
        ):
            raise ValueError("axes require nonempty named levels")
        keys: set[str] = set()
        for label, overrides in levels.items():
            if (
                not isinstance(label, str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", label) is None
                or not isinstance(overrides, dict)
            ):
                raise ValueError("axis levels require a name and parameter overrides")
            keys.update(overrides)
        if keys & assigned:
            raise ValueError(
                "different axes cannot override the same generator parameter"
            )
        assigned.update(keys)
    conditions = []
    for labels in itertools.product(*(axes[name] for name in names)):
        selected = dict(zip(names, labels, strict=True))
        config = dict(base)
        for name, label in selected.items():
            config.update(axes[name][label])
        if config.get("kind") not in _GENERATORS:
            raise ValueError(f"unknown instance-set generator: {config.get('kind')}")
        if type(config.get("count")) is not int or config["count"] <= 0:
            raise ValueError("each condition requires a positive integer count")
        identity = (
            "__".join(f"{name}-{label}" for name, label in selected.items())
            or "baseline"
        )
        conditions.append(
            {
                "condition_id": identity,
                "axes": selected,
                "generator": config,
                "status": "pending",
                "instances": [],
            }
        )
    if len({cell["condition_id"] for cell in conditions}) != len(conditions):
        raise ValueError("generation conditions must have distinct directory names")
    manifest = {
        "schema_version": "1.0",
        "visibility": visibility,
        "plan": plan,
        "plan_sha256": hashlib.sha256(
            json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "generator_module_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "python_version": platform.python_version(),
        "reference_status": "optimum_and_bks_not_computed",
        "solver_runs": 0,
        "expected_instances": sum(cell["generator"]["count"] for cell in conditions),
        "conditions": conditions,
        "status": "incomplete",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "generation_plan.yaml").write_text(
        yaml.safe_dump(plan, sort_keys=False)
    )
    rows = []
    try:
        for cell in conditions:
            cell["status"] = "started"
            payload = {"visibility": visibility, "generator": cell["generator"]}
            destination = output_dir / "conditions" / cell["condition_id"]
            paths = materialize_generated_instance_set(
                payload, destination, expected_visibility=visibility
            )
            (destination / "generation_config.yaml").write_text(
                yaml.safe_dump(payload, sort_keys=False)
            )
            cell["instance_set_config"] = str(
                (destination / "instance_set_config.yaml").relative_to(output_dir)
            )
            describe = _INSTANCE_FEATURES.get(cell["generator"]["kind"])
            for index, path in enumerate(paths):
                features = describe(path) if describe is not None else {}
                entry = {
                    "id": path.stem,
                    "replicate_index": index,
                    "path": str(path.relative_to(output_dir)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "features": features,
                }
                cell["instances"].append(entry)
                rows.append(
                    {
                        "condition_id": cell["condition_id"],
                        "instance_id": entry["id"],
                        "replicate_index": index,
                        "path": entry["path"],
                        "sha256": entry["sha256"],
                        **{
                            f"axis_{name}": label
                            for name, label in cell["axes"].items()
                        },
                        **features,
                    }
                )
            cell["status"] = "completed"
        manifest["status"] = "completed"
    except Exception as error:
        cell["status"] = "failed"
        cell["error"] = {"type": type(error).__name__, "message": str(error)}
        manifest["status"] = "failed"
        raise
    finally:
        manifest["generated_instances"] = sum(
            len(cell["instances"]) for cell in conditions
        )
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (output_dir / "features.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return manifest


def materialize_instance_set(
    instance_set_config: Path,
    output_dir: Path,
) -> list[Path]:
    payload = yaml.safe_load(instance_set_config.read_text())
    if "generator" in payload:
        return materialize_generated_instance_set(
            payload,
            output_dir,
            expected_visibility="agent",
            stem_prefix="dev",
        )
    if payload.get("visibility") != "agent":
        raise ValueError("fixed development instance set must be agent-visible")

    config_directory = instance_set_config.parent.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bks_output_dir = output_dir / "bks_solutions"
    bks_output_dir.mkdir(exist_ok=True)
    paths: list[Path] = []
    output_instances: list[dict[str, Any]] = []
    instance_ids: set[str] = set()
    for index, item in enumerate(payload["instances"]):
        instance_id = item["id"]
        if instance_id in instance_ids:
            raise ValueError(f"duplicate development instance ID: {instance_id}")
        instance_ids.add(instance_id)

        source_instance = verify_public_file(
            config_directory,
            item["instance_file"],
            item["instance_file_sha256"],
        )
        instance_path = output_dir / f"dev_{index:04d}{source_instance.suffix}"
        shutil.copyfile(source_instance, instance_path)
        paths.append(instance_path)

        source_bks_solution = verify_public_file(
            config_directory,
            item["bks_solution_file"],
            item["bks_solution_file_sha256"],
        )
        bks_solution_path = bks_output_dir / source_bks_solution.name
        shutil.copyfile(source_bks_solution, bks_solution_path)
        output_instances.append(
            {
                "id": instance_id,
                "path": instance_path.name,
                "bks": item["bks"],
                "bks_solution_file": str(bks_solution_path.relative_to(output_dir)),
                "proven_optimal": item["proven_optimal"],
            }
        )

    index_path = output_dir / "instance_set_config.yaml"
    index_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "visibility": "agent",
                "instances": output_instances,
            },
            sort_keys=False,
        )
    )
    return paths


def verify_public_file(
    config_directory: Path,
    relative_path: str,
    expected_sha256: str | None,
) -> Path:
    path = (config_directory / relative_path).resolve()
    if config_directory not in path.parents:
        raise ValueError(f"development file escapes config directory: {relative_path}")
    if not path.is_file():
        raise ValueError(f"missing development file: {relative_path}")
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"development file hash mismatch for {relative_path}: "
            f"{actual_sha256} != {expected_sha256}"
        )
    return path


def prepare_collection_instances(
    source: Path,
    destination: Path,
    *,
    path_template: str | None = None,
    generated_stem_prefix: str | None = None,
) -> list[dict]:
    """Copy a fixed input panel using explicit paths, including archived indices.

    A template belongs to the experiment config, not the solver implementation.
    It permits reading old manifests without rewriting their retained artifacts.
    """
    import shutil

    payload = yaml.safe_load(source.read_text())
    private_root = Path(__file__).resolve().parents[2] / "private"
    destination.mkdir(parents=True, exist_ok=True)
    if "generator" in payload:
        paths = materialize_generated_instance_set(
            payload,
            destination,
            expected_visibility=payload.get("visibility", "agent"),
            stem_prefix=generated_stem_prefix or source.stem,
        )
        return [
            {"id": path.stem, "path": str(path.resolve()), "bks": None}
            for path in paths
        ]
    instances = []
    for item in payload["instances"]:
        identity = item.get("id", item.get("name"))
        relative = (
            path_template.format(**item)
            if path_template is not None
            else item.get("instance_file", item.get("path"))
        )
        private_uri = item.get("instance_uri")
        if identity is None or (relative is None and private_uri is None):
            raise ValueError("each instance requires an ID and explicit input path")
        if private_uri is not None:
            if not private_uri.startswith("private://"):
                raise ValueError("instance_uri must use private:// storage")
            original = private_root / private_uri.removeprefix("private://")
        else:
            original = source.parent / relative
        if not original.is_file():
            raise FileNotFoundError(f"missing collection instance: {original}")
        target = destination / f"{identity}{''.join(original.suffixes)}"
        if target.exists():
            if target.read_bytes() != original.read_bytes():
                raise ValueError(f"prepared input changed: {target}")
        else:
            shutil.copy2(original, target)
        bks_solution = (
            verify_public_file(
                source.parent,
                item["bks_solution_file"],
                item["bks_solution_file_sha256"],
            )
            if "bks_solution_file" in item
            else None
        )
        proof_path = (
            verify_public_file(
                source.parent,
                item["proof_file"],
                item["proof_file_sha256"],
            )
            if "proof_file" in item
            else None
        )
        instances.append(
            {
                "id": identity,
                "path": str(target.resolve()),
                "bks": item.get("bks"),
                "bks_solution_path": str(bks_solution) if bks_solution else None,
                "proof_path": str(proof_path) if proof_path else None,
                "proven_optimal": item.get("proven_optimal"),
            }
        )
    return instances


def first_fit_decreasing_packing(
    weights: list[int],
    capacity: int,
) -> list[list[int]]:
    bins: list[list[int]] = []
    loads: list[int] = []
    for item_index in sorted(
        range(len(weights)),
        key=lambda index: (-weights[index], index),
    ):
        weight = weights[item_index]
        for bin_index, load in enumerate(loads):
            if load + weight <= capacity:
                bins[bin_index].append(item_index)
                loads[bin_index] += weight
                break
        else:
            bins.append([item_index])
            loads.append(weight)
    return bins


def _job_shop_operations(instance_path: Path) -> list[list[list[int]]]:
    values = [
        int(token)
        for line in instance_path.read_text().splitlines()
        if not line.strip().startswith("#")
        for token in line.split()
    ]
    if len(values) < 2 or values[0] <= 0 or values[1] <= 0:
        raise ValueError(
            "job-shop instance must declare positive job and machine counts"
        )
    job_count, machine_count = values[:2]
    if len(values) != 2 + 2 * job_count * machine_count:
        raise ValueError("job-shop operation count does not match instance header")
    operations = []
    cursor = 2
    for _ in range(job_count):
        job = []
        for _ in range(machine_count):
            machine, duration = values[cursor : cursor + 2]
            cursor += 2
            if machine < 0 or machine >= machine_count or duration <= 0:
                raise ValueError("job-shop operation is outside its declared domain")
            job.append([machine, duration])
        operations.append(job)
    return operations


def prepare_cp_sat_job_shop_panel(
    *,
    task_id: str,
    instance_set_name: str,
    visibility: str,
    source_root: Path,
    instance_ids: list[str],
    model_preparer_image: str,
    instance_output_dir: Path,
    reference_solution_dir: Path | None,
    instance_set_config_path: Path,
    private_root: Path | None = None,
    oracle_output_path: Path | None = None,
    reference_solution_source_dirs: list[Path] | None = None,
) -> dict[str, Any]:
    """Prepare fixed CP-SAT JSSP proto inputs and their exact targets."""
    from pitbench.problem_families.verification import (
        TrustedOptimumOracle,
        TrustedOptimumRecord,
    )

    if visibility not in {"agent", "judge"}:
        raise ValueError("job-shop panel visibility must be agent or judge")
    if len(instance_ids) != len(set(instance_ids)) or not instance_ids:
        raise ValueError("job-shop panel requires unique instance IDs")
    if (private_root is None) != (oracle_output_path is None):
        raise ValueError(
            "trusted oracle output and private root must be supplied together"
        )
    if visibility == "agent" and not reference_solution_source_dirs:
        raise ValueError("job-shop panel requires independent reference schedules")
    if visibility == "agent" and reference_solution_dir is None:
        raise ValueError("agent job-shop panel requires a reference solution directory")
    if not model_preparer_image:
        raise ValueError("job-shop panel requires a CP-SAT model preparer image")
    source_root = source_root.resolve()
    index_path = source_root / "instances.json"
    indexed = {item["name"]: item for item in json.loads(index_path.read_text())}
    missing = [
        name for name in instance_ids if indexed.get(name, {}).get("optimum") is None
    ]
    if missing:
        raise ValueError(f"job-shop instances lack fixed optima: {missing}")

    instance_output_dir.mkdir(parents=True, exist_ok=True)
    if reference_solution_dir is not None:
        reference_solution_dir.mkdir(parents=True, exist_ok=True)
    records = []
    config_instances = []
    for instance_id in instance_ids:
        source = source_root / indexed[instance_id]["path"]
        model_path = instance_output_dir / f"{instance_id}.pb"
        prepared = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--mount",
                f"type=bind,src={source_root},dst=/input,readonly",
                "--mount",
                f"type=bind,src={instance_output_dir.resolve()},dst=/output",
                model_preparer_image,
                "python3",
                "/opt/pitbench-jvm/runner.py",
                "prepare-jssp",
                "--solver",
                "ortools_cp_sat",
                "--instance",
                str(Path("/input") / source.relative_to(source_root)),
                "--output",
                str(Path("/output") / model_path.name),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        metadata = json.loads(prepared.stdout)
        instance_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
        operations = _job_shop_operations(source)
        relative_model = model_path.relative_to(instance_set_config_path.parent)
        if visibility == "agent":
            reference_schedule = None
            for reference_dir in reference_solution_source_dirs or []:
                source_path = reference_dir / f"{instance_id}.solution.json"
                if not source_path.is_file():
                    continue
                candidate = json.loads(source_path.read_text())
                if isinstance(candidate.get("start_times"), list) and all(
                    type(value) is int for value in candidate["start_times"]
                ):
                    reference_schedule = candidate
                    break
            if reference_schedule is None:
                raise ValueError(
                    f"missing independent reference schedule for {instance_id}"
                )
            start_times = reference_schedule["start_times"]
            expected_operation_count = len(metadata["start_variable_indices"])
            if len(start_times) != expected_operation_count:
                raise ValueError(
                    f"reference schedule has the wrong operation count for {instance_id}"
                )
            assert reference_solution_dir is not None
            reference_path = reference_solution_dir / f"{instance_id}.solution.json"
            reference_path.write_text(
                json.dumps({"start_times": start_times}, indent=2) + "\n"
            )
            reference_sha256 = hashlib.sha256(reference_path.read_bytes()).hexdigest()
            config_instances.append(
                {
                    "id": instance_id,
                    "instance_file": relative_model.as_posix(),
                    "instance_file_sha256": instance_sha256,
                    "bks": indexed[instance_id]["optimum"],
                    "bks_solution_file": reference_path.relative_to(
                        instance_set_config_path.parent
                    ).as_posix(),
                    "bks_solution_file_sha256": reference_sha256,
                    "proven_optimal": True,
                }
            )
            continue
        assert private_root is not None
        assert oracle_output_path is not None
        records.append(
            TrustedOptimumRecord(
                instance_set=instance_set_name,
                instance_id=instance_id,
                instance_sha256=instance_sha256,
                target_kind="published_bks",
                problem_kind="job_shop_scheduling",
                objective_sense="minimize",
                optimal_objective=indexed[instance_id]["optimum"],
                optimality_basis={
                    "kind": "published_job_shop_bks",
                    "jobs": operations,
                    "start_variable_indices": metadata["start_variable_indices"],
                },
                generation_provenance={
                    "published_bks_source": {
                        "repository": "https://github.com/tamy0612/JSPLIB",
                        "artifact_path": "instances.json",
                        "artifact_sha256": hashlib.sha256(
                            index_path.read_bytes()
                        ).hexdigest(),
                        "instance_path": indexed[instance_id]["path"],
                    },
                    "source_instance_sha256": hashlib.sha256(
                        source.read_bytes()
                    ).hexdigest(),
                    "model_preparer": "adapters.jvm.runner:prepare-jssp",
                    "model_preparer_image": model_preparer_image,
                },
                verification_provenance=(
                    "pitbench.problem_families.verification:TrustedOptimumFamily"
                ),
            )
        )
        config_instances.append(
            {
                "id": instance_id,
                "uri": "private://" + model_path.relative_to(private_root).as_posix(),
                "optimal_or_bks": indexed[instance_id]["optimum"],
            }
        )

    instance_set_config_path.parent.mkdir(parents=True, exist_ok=True)
    instance_set_config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "visibility": visibility,
                "format": "cp_sat_model_proto",
                "instances": config_instances,
            },
            sort_keys=False,
        )
    )
    if oracle_output_path is not None:
        oracle = TrustedOptimumOracle(task_id=task_id, records=records)
        oracle_output_path.parent.mkdir(parents=True, exist_ok=True)
        oracle_output_path.write_text(
            yaml.safe_dump(
                oracle.model_dump(mode="json", exclude_none=True), sort_keys=False
            )
        )
        return oracle.model_dump(mode="json", exclude_none=True)
    return {"instances": config_instances}


def prepare_trusted_optimum_oracle(
    *,
    task_config_path: Path,
    instance_set_name: str,
    instance_set_config_path: Path,
    private_root: Path,
    output_path: Path,
    reference_solution_dir: Path,
    bin_packing_proof_dir: Path | None = None,
    proof_solver_commit: str | None = None,
) -> dict[str, Any]:
    from pitbench.problem_families.verification import (
        TrustedOptimumOracle,
        TrustedOptimumRecord,
        validate_bin_packing_optimum_record,
    )
    from pitbench.schema.task import PitBenchTask

    task = PitBenchTask.from_yaml(task_config_path)
    instance_set_config = yaml.safe_load(instance_set_config_path.read_text())
    generator = instance_set_config["generator"]
    _require_planted_optimum(generator)
    if instance_set_config.get("visibility") != "judge":
        raise ValueError("trusted optimum generation requires a judge instance set")
    if generator["kind"] not in {
        "single_machine_scheduling_mip",
        "bin_packing",
        "multidimensional_knapsack_mip",
        "binary_clique_cover_mip",
        "graph_coloring",
        "balanced_subset_sum",
    }:
        raise ValueError("unsupported trusted optimum problem kind")
    private_root = private_root.resolve()
    output_path = output_path.resolve()
    reference_solution_dir = reference_solution_dir.resolve()
    if not output_path.is_relative_to(private_root) or not (
        reference_solution_dir == private_root
        or reference_solution_dir.is_relative_to(private_root)
    ):
        raise ValueError("trusted optimum artifacts must remain in private storage")
    reference_solution_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with tempfile.TemporaryDirectory(prefix="pitbench-trusted-optimum-") as temporary:
        instance_paths = materialize_generated_instance_set(
            instance_set_config,
            Path(temporary),
            expected_visibility="judge",
            stem_prefix=instance_set_name,
        )
        for index, instance_path in enumerate(instance_paths):
            instance_id = instance_path.stem
            proof_file_sha256 = None
            if generator["kind"] == "single_machine_scheduling_mip":
                processing_times = scheduling_processing_times(generator, index)
                order = sorted(
                    range(len(processing_times)),
                    key=lambda job: (processing_times[job], job),
                )
                start_times = [0] * len(processing_times)
                elapsed = 0
                for job in order:
                    start_times[job] = elapsed
                    elapsed += processing_times[job]
                optimal_objective = sum(start_times)
                problem_kind = "single_machine_scheduling"
                reference_solution = {
                    "order": order,
                    "start_times": start_times,
                }
                optimality_basis = {
                    "kind": "shortest_processing_time",
                    "processing_times": processing_times,
                }
            elif generator["kind"] == "multidimensional_knapsack_mip":
                weights, capacities, selected = multidimensional_knapsack_data(
                    generator, index
                )
                optimal_objective = -sum(capacities)
                problem_kind = "multidimensional_knapsack"
                reference_solution = {"selected_items": selected}
                optimality_basis = {
                    "kind": "capacity_dual_bound",
                    "dimensions": len(weights),
                    "items": len(weights[0]),
                }
            elif generator["kind"] == "binary_clique_cover_mip":
                groups, planted, _ = binary_clique_cover_data(generator, index)
                values = {
                    name: int(offset == planted[group_index])
                    for group_index, group in enumerate(groups)
                    for offset, name in enumerate(group)
                }
                fme_extensions = binary_clique_cover_fme_extensions(
                    generator, index, groups, planted
                )
                values.update({extension["column"]: 0 for extension in fme_extensions})
                optimal_objective = -len(groups)
                problem_kind = "binary_clique_cover"
                optimality_basis = {
                    "kind": "binary_clique_cover",
                    "groups": groups,
                }
                if generator.get("fme_paper_block", False):
                    problem_kind = "binary_clique_cover_fme"
                    optimal_objective += 12
                    values.update({"fme_i": 12, "fme_x": 0, "fme_y": 14, "fme_z": 18})
                    optimality_basis = {
                        "kind": "binary_clique_cover_fme",
                        "groups": groups,
                        "fme_paper_block": {
                            "integer_variable": "fme_i",
                            "continuous_variables": ["fme_x", "fme_y", "fme_z"],
                            "rows": ["FMEPAPER1", "FMEPAPER2", "FMEPAPER3"],
                            "lower_bound": 12,
                            "witness": {
                                "fme_i": 12,
                                "fme_x": 0,
                                "fme_y": 14,
                                "fme_z": 18,
                            },
                        },
                    }
                reference_solution = {
                    "values": values,
                    "objective": optimal_objective,
                }
                if fme_extensions:
                    optimality_basis["fme_extensions"] = fme_extensions
            elif generator["kind"] == "graph_coloring":
                instance, colors, basis = graph_coloring_data(generator, index)
                instance["name"] = instance_id
                if json.loads(instance_path.read_text()) != instance:
                    raise ValueError("graph coloring generated input changed")
                optimal_objective = basis["chromatic_number"]
                problem_kind = "graph_coloring"
                reference_solution = {"colors": colors}
                optimality_basis = basis
            elif generator["kind"] == "balanced_subset_sum":
                instance, selected, basis = balanced_subset_sum_data(generator, index)
                instance["name"] = instance_id
                if json.loads(instance_path.read_text()) != instance:
                    raise ValueError("balanced subset-sum generated input changed")
                optimal_objective = abs(
                    sum(instance["weights"][item] for item in selected)
                    - instance["target"]
                )
                problem_kind = "balanced_subset_sum"
                reference_solution = {"selected_items": selected}
                optimality_basis = basis
            else:
                instance = json.loads(instance_path.read_text())
                weights = instance["weights"]
                capacity = instance["capacity"]
                if bin_packing_proof_dir is not None:
                    proof_path = bin_packing_proof_dir / f"{instance_id}.proof.json"
                    proof_bytes = proof_path.read_bytes()
                    proof = json.loads(proof_bytes)
                    if not isinstance(proof, dict):
                        raise ValueError(
                            f"bin packing proof is not an object: {instance_id}"
                        )
                    if (
                        proof_solver_commit is not None
                        and proof.get("solver_commit") != proof_solver_commit
                    ):
                        raise ValueError(
                            f"bin packing proof solver commit differs: {instance_id}"
                        )
                    reference_bins = proof.get("bins")
                    optimal_objective = validate_bin_packing_optimum_record(
                        proof,
                        instance_sha256=hashlib.sha256(
                            instance_path.read_bytes()
                        ).hexdigest(),
                        weights=weights,
                        capacity=capacity,
                        reference_bins=reference_bins,
                    )
                    optimality_basis = {
                        "kind": "independent_solver_optimal",
                        "solver": "ortools_cp_sat",
                        "proof": proof,
                    }
                    proof_file_sha256 = hashlib.sha256(proof_bytes).hexdigest()
                else:
                    reference_bins = first_fit_decreasing_packing(weights, capacity)
                    capacity_lower_bound = math.ceil(sum(weights) / capacity)
                    if len(reference_bins) != capacity_lower_bound:
                        raise ValueError(
                            f"instance {instance_id} reference packing does not match "
                            "its capacity lower bound"
                        )
                    optimal_objective = capacity_lower_bound
                    optimality_basis = {
                        "kind": "capacity_lower_bound",
                        "total_weight": sum(weights),
                        "capacity": capacity,
                        "lower_bound": capacity_lower_bound,
                    }
                problem_kind = "bin_packing"
                reference_solution = {"bins": reference_bins}

            reference_solution_path = (
                reference_solution_dir / f"{instance_id}.solution.json"
            )
            reference_solution_path.write_text(
                json.dumps(reference_solution, indent=2) + "\n"
            )
            records.append(
                TrustedOptimumRecord(
                    instance_set=instance_set_name,
                    instance_id=instance_id,
                    instance_sha256=hashlib.sha256(
                        instance_path.read_bytes()
                    ).hexdigest(),
                    problem_kind=problem_kind,
                    objective_sense=task.oracle.objective_sense,
                    optimal_objective=optimal_objective,
                    reference_solution_uri=(
                        "private://"
                        + reference_solution_path.relative_to(private_root).as_posix()
                    ),
                    reference_solution_sha256=hashlib.sha256(
                        reference_solution_path.read_bytes()
                    ).hexdigest(),
                    optimality_basis=optimality_basis,
                    generation_provenance={
                        "instance_generator": generator["kind"],
                        "instance_seed": _instance_seed(generator, index),
                        "instance_set_config_sha256": hashlib.sha256(
                            instance_set_config_path.read_bytes()
                        ).hexdigest(),
                        "task_release_commit": task.release.base_commit,
                        "oracle_builder": (
                            "pitbench.instances.generate:prepare_trusted_optimum_oracle"
                        ),
                        **(
                            {"proof_file_sha256": proof_file_sha256}
                            if proof_file_sha256 is not None
                            else {}
                        ),
                    },
                    verification_provenance=(
                        "pitbench.problem_families.verification:TrustedOptimumFamily"
                    ),
                )
            )

    oracle = TrustedOptimumOracle(
        task_id=task.task_id,
        records=records,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(oracle.model_dump(mode="json"), sort_keys=False)
    )
    return oracle.model_dump(mode="json")


def prepare_generated_development_panel(
    generator_config_path: Path,
    instance_set_config_path: Path,
) -> dict[str, Any]:
    """Freeze generated exact-task inputs and visible reference solutions."""

    payload = yaml.safe_load(generator_config_path.read_text())
    if payload.get("visibility") != "agent":
        raise ValueError("development panel generator must be agent-visible")
    config = payload["generator"]
    _require_planted_optimum(config)
    kind = config["kind"]
    if kind not in {
        "multidimensional_knapsack_mip",
        "binary_clique_cover_mip",
        "graph_coloring",
    }:
        raise ValueError("unsupported generated development panel")
    suffix, generator = _GENERATORS[kind]
    panel_dir = instance_set_config_path.with_suffix("")
    instances_dir = panel_dir / "instances"
    solutions_dir = panel_dir / "bks_solutions"
    instances_dir.mkdir(parents=True, exist_ok=True)
    solutions_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index in range(int(config["count"])):
        instance_id = f"dev_{index:04d}"
        instance_path = instances_dir / f"{instance_id}.{suffix}"
        generator(config, index, instance_path)
        if kind == "multidimensional_knapsack_mip":
            _, capacities, selected = multidimensional_knapsack_data(config, index)
            optimum = -sum(capacities)
            solution = {"selected_items": selected}
        elif kind == "binary_clique_cover_mip":
            groups, planted, _ = binary_clique_cover_data(config, index)
            fme_extensions = binary_clique_cover_fme_extensions(
                config, index, groups, planted
            )
            optimum = -len(groups)
            if config.get("fme_paper_block", False):
                optimum += 12
            solution = {
                "values": {
                    name: int(offset == planted[group_index])
                    for group_index, group in enumerate(groups)
                    for offset, name in enumerate(group)
                }
                | {extension["column"]: 0 for extension in fme_extensions},
                "objective": optimum,
            }
            if config.get("fme_paper_block", False):
                solution["values"].update(
                    {"fme_i": 12, "fme_x": 0, "fme_y": 14, "fme_z": 18}
                )
        else:
            _, colors, basis = graph_coloring_data(config, index)
            optimum = basis["chromatic_number"]
            solution = {"colors": colors}
        solution_path = solutions_dir / f"{instance_id}.solution.json"
        solution_path.write_text(json.dumps(solution, indent=2) + "\n")
        records.append(
            {
                "id": instance_id,
                "instance_file": instance_path.relative_to(
                    instance_set_config_path.parent
                ).as_posix(),
                "instance_file_sha256": hashlib.sha256(
                    instance_path.read_bytes()
                ).hexdigest(),
                "bks": optimum,
                "bks_solution_file": solution_path.relative_to(
                    instance_set_config_path.parent
                ).as_posix(),
                "bks_solution_file_sha256": hashlib.sha256(
                    solution_path.read_bytes()
                ).hexdigest(),
                "proven_optimal": True,
            }
        )
    manifest = {
        "schema_version": "1.0",
        "visibility": "agent",
        "format": suffix,
        "instances": records,
        "selection": {"source_generator": kind, **config},
    }
    instance_set_config_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return manifest
