from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

import yaml


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
    distance_metric: str | None = None,
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
        capacity_ratio=float(config["capacity_ratio"]),
        coordinate_distribution=config.get("coordinate_distribution", "uniform"),
        cluster_count=int(config.get("cluster_count", 4)),
        cluster_spread=float(config.get("cluster_spread", 9.0)),
        depot_mode=config.get("depot_mode", "center"),
        demand_distribution=config.get("demand_distribution", "uniform_integer"),
        distance_metric=config.get("distance_metric"),
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
                "visibility": "agent",
                "instances": [{"id": path.stem, "path": path.name} for path in paths],
            },
            sort_keys=False,
        )
    )
    return paths


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
