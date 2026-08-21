from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import yaml

_UCHOA_DEMAND_FAMILIES = {
    "unitary",
    "small_large_variance",
    "small_small_variance",
    "large_large_variance",
    "large_small_variance",
    "quadrant",
    "quadrant_permuted",
    "many_small_few_large",
}


def _stream_seed(namespace: str, seed: int) -> int:
    digest = hashlib.sha256(f"{namespace}:{seed}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _uchoa_customer_coordinates(
    customers: int, coordinate_seed: int, mode: str
) -> list[list[float]]:
    random_rng = random.Random(_stream_seed("uchoa-random", coordinate_seed))
    random_points = [
        [random_rng.uniform(0, 100), random_rng.uniform(0, 100)]
        for _ in range(customers)
    ]
    cluster_rng = random.Random(_stream_seed("uchoa-cluster", coordinate_seed))
    cluster_count = 3 + cluster_rng.randrange(6)
    centers = [
        [cluster_rng.uniform(10, 90), cluster_rng.uniform(10, 90)]
        for _ in range(cluster_count)
    ]
    clustered_points = []
    for index in range(customers):
        center_x, center_y = centers[index % cluster_count]
        radius = cluster_rng.expovariate(1 / 7.5)
        angle = cluster_rng.uniform(0, 2 * math.pi)
        clustered_points.append(
            [
                min(100.0, max(0.0, center_x + radius * math.cos(angle))),
                min(100.0, max(0.0, center_y + radius * math.sin(angle))),
            ]
        )
    if mode == "R":
        return random_points
    if mode == "C":
        return clustered_points
    if mode == "RC":
        return [
            clustered_points[index] if index % 2 == 0 else random_points[index]
            for index in range(customers)
        ]
    raise ValueError(f"unknown Uchoa customer positioning: {mode}")


def _uchoa_demands(
    coordinates: list[list[float]], demand_seed: int, family: str
) -> list[int]:
    if family not in _UCHOA_DEMAND_FAMILIES:
        raise ValueError(f"unknown Uchoa demand family: {family}")
    rng = random.Random(_stream_seed("uchoa-demand", demand_seed))
    uniforms = [rng.random() for _ in coordinates]

    def discrete(low: int, high: int, value: float) -> int:
        return low + min(int(value * (high - low + 1)), high - low)

    if family == "unitary":
        return [1] * len(coordinates)
    if family == "small_large_variance":
        return [discrete(1, 10, value) for value in uniforms]
    if family == "small_small_variance":
        return [discrete(5, 10, value) for value in uniforms]
    if family == "large_large_variance":
        return [discrete(1, 100, value) for value in uniforms]
    if family == "large_small_variance":
        return [discrete(50, 100, value) for value in uniforms]
    if family == "many_small_few_large":
        return [
            discrete(1, 10, value / 0.85)
            if value < 0.85
            else discrete(50, 100, (value - 0.85) / 0.15)
            for value in uniforms
        ]
    quadrant = [
        discrete(1, 50, value) if ((x >= 50) == (y >= 50)) else discrete(51, 100, value)
        for (x, y), value in zip(coordinates, uniforms, strict=True)
    ]
    if family == "quadrant":
        return quadrant
    permutation_rng = random.Random(
        _stream_seed("uchoa-demand-permutation", demand_seed)
    )
    permutation = list(range(len(quadrant)))
    permutation_rng.shuffle(permutation)
    return [quadrant[index] for index in permutation]


def make_uchoa_cvrp_instance(
    *,
    name: str,
    customers: int,
    coordinate_seed: int,
    demand_seed: int,
    depot_positioning: str,
    customer_positioning: str,
    demand_family: str,
    route_size: float,
) -> dict[str, Any]:
    """Generate a deterministic Uchoa-X-style instance from shared latent streams."""

    if customers <= 0:
        raise ValueError("customers must be positive")
    if route_size <= 0:
        raise ValueError("route_size must be positive")
    customer_coordinates = _uchoa_customer_coordinates(
        customers, coordinate_seed, customer_positioning
    )
    if depot_positioning == "C":
        depot_coordinates = [50.0, 50.0]
    elif depot_positioning == "E":
        depot_coordinates = [0.0, 0.0]
    elif depot_positioning == "R":
        depot_rng = random.Random(_stream_seed("uchoa-depot", coordinate_seed))
        depot_coordinates = [depot_rng.uniform(0, 100), depot_rng.uniform(0, 100)]
    else:
        raise ValueError(f"unknown Uchoa depot positioning: {depot_positioning}")
    customer_demands = _uchoa_demands(customer_coordinates, demand_seed, demand_family)
    capacity = max(
        max(customer_demands),
        math.ceil(route_size * sum(customer_demands) / customers),
    )
    return {
        "name": name,
        "depot": 0,
        "coordinates": [depot_coordinates, *customer_coordinates],
        "demands": [0, *customer_demands],
        "capacity": capacity,
        "generator_attributes": {
            "family": "uchoa_x_style",
            "depot_positioning": depot_positioning,
            "customer_positioning": customer_positioning,
            "demand_family": demand_family,
            "requested_route_size": route_size,
            "coordinate_seed": coordinate_seed,
            "demand_seed": demand_seed,
        },
    }


def _choice(values: list[int], index: int) -> int:
    return values[index % len(values)]


def make_euclidean_cvrp_instance(
    *,
    name: str,
    customers: int,
    coordinate_seed: int,
    demand_seed: int,
    capacity_ratio: float,
    coordinate_distribution: str = "uniform",
    cluster_count: int = 4,
    cluster_spread: float = 9.0,
    depot_mode: str = "center",
    demand_distribution: str = "uniform_integer",
) -> dict[str, Any]:
    """Generate a deterministic normalized CVRP instance.

    Uniform and clustered coordinates share the same RNG dimensions so validation
    panels can hold demand realization fixed while changing only spatial structure.
    """
    if customers <= 0:
        raise ValueError("customers must be positive")
    if not 0 < capacity_ratio <= 1:
        raise ValueError("capacity_ratio must be in (0, 1]")
    coordinate_rng = random.Random(coordinate_seed)
    demand_rng = random.Random(demand_seed)
    if coordinate_distribution == "uniform":
        customer_coordinates = [
            [coordinate_rng.uniform(0, 100), coordinate_rng.uniform(0, 100)]
            for _ in range(customers)
        ]
    elif coordinate_distribution == "clustered":
        if cluster_count <= 0 or cluster_spread <= 0:
            raise ValueError("cluster_count and cluster_spread must be positive")
        centers = [
            [coordinate_rng.uniform(15, 85), coordinate_rng.uniform(15, 85)]
            for _ in range(cluster_count)
        ]
        customer_coordinates = []
        for index in range(customers):
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

    if demand_distribution == "uniform_integer":
        customer_demands = [demand_rng.randint(1, 10) for _ in range(customers)]
    elif demand_distribution == "bimodal":
        customer_demands = [
            demand_rng.randint(1, 3)
            if demand_rng.random() < 0.5
            else demand_rng.randint(8, 10)
            for _ in range(customers)
        ]
    elif demand_distribution in {"depot_correlated", "depot_anticorrelated"}:
        # Preserve the uniform-integer marginal while coupling its order to location.
        sampled = sorted(demand_rng.randint(1, 10) for _ in range(customers))
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
    else:
        raise ValueError(f"unknown demand_distribution: {demand_distribution}")

    demands = [0, *customer_demands]
    capacity = max(10, math.ceil(sum(demands) * capacity_ratio))
    return {
        "name": name,
        "depot": 0,
        "coordinates": [depot_coordinates, *customer_coordinates],
        "demands": demands,
        "capacity": capacity,
    }


def _cvrp(config: dict[str, Any], index: int, destination: Path) -> None:
    seeds = config["randomness"]
    instance = make_euclidean_cvrp_instance(
        name=destination.stem,
        customers=_choice(config["customers"], index),
        coordinate_seed=seeds["coordinate_seed"] + index,
        demand_seed=seeds["demand_seed"] + index,
        capacity_ratio=float(config["capacity_ratio"]),
        coordinate_distribution=config.get("coordinate_distribution", "uniform"),
        cluster_count=int(config.get("cluster_count", 4)),
        cluster_spread=float(config.get("cluster_spread", 9.0)),
        depot_mode=config.get("depot_mode", "center"),
        demand_distribution=config.get("demand_distribution", "uniform_integer"),
    )
    destination.write_text(json.dumps(instance, indent=2))


def _bin_packing(config: dict[str, Any], index: int, destination: Path) -> None:
    rng = random.Random(config["randomness"]["instance_seed"] + index)
    item_count = _choice(config["items"], index)
    capacity = int(config["capacity"])
    destination.write_text(
        json.dumps(
            {
                "name": destination.stem,
                "capacity": capacity,
                "weights": [rng.randint(1, capacity) for _ in range(item_count)],
            },
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


def _scheduling_lp(config: dict[str, Any], index: int, destination: Path) -> None:
    rng = random.Random(config["randomness"]["instance_seed"] + index)
    jobs = _choice(config["jobs"], index)
    processing = [rng.randint(1, 20) for _ in range(jobs)]
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
    lines.extend(["Bounds", *[f" 0 <= s{i} <= {horizon}" for i in range(jobs)]])
    lines.extend(["Binary", *[f" {name}" for name in binaries], "End"])
    destination.write_text("\n".join(lines) + "\n")


_GENERATORS = {
    "euclidean_cvrp": ("json", _cvrp),
    "bin_packing": ("json", _bin_packing),
    "repeated_constants_cumulative": ("json", _ortools),
    "single_machine_scheduling_mip": ("lp", _scheduling_lp),
}


def materialize_population(manifest: Path, output_dir: Path) -> list[Path]:
    payload = yaml.safe_load(manifest.read_text())
    if payload.get("visibility") != "agent":
        raise ValueError("only agent-visible populations may be materialized here")
    config = payload["generator"]
    kind = config["kind"]
    if kind not in _GENERATORS:
        raise ValueError(f"unknown population generator: {kind}")
    suffix, generator = _GENERATORS[kind]
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index in range(int(config["count"])):
        path = output_dir / f"dev_{index:04d}.{suffix}"
        generator(config, index, path)
        paths.append(path)
    index_path = output_dir / "population.yaml"
    index_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "visibility": "agent",
                "instances": [{"id": path.stem, "path": path.name} for path in paths],
            },
            sort_keys=False,
        )
    )
    return paths
