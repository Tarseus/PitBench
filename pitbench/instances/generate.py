from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import yaml


def _choice(values: list[int], index: int) -> int:
    return values[index % len(values)]


def _cvrp(config: dict[str, Any], index: int, destination: Path) -> None:
    seeds = config["randomness"]
    coordinate_rng = random.Random(seeds["coordinate_seed"] + index)
    demand_rng = random.Random(seeds["demand_seed"] + index)
    customers = _choice(config["customers"], index)
    coordinates = [[50.0, 50.0]] + [
        [coordinate_rng.uniform(0, 100), coordinate_rng.uniform(0, 100)]
        for _ in range(customers)
    ]
    demands = [0] + [demand_rng.randint(1, 10) for _ in range(customers)]
    capacity = max(10, math.ceil(sum(demands) * config["capacity_ratio"]))
    destination.write_text(
        json.dumps(
            {
                "name": destination.stem,
                "depot": 0,
                "coordinates": coordinates,
                "demands": demands,
                "capacity": capacity,
            },
            indent=2,
        )
    )


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
