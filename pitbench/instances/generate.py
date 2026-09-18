from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml


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


def make_bin_packing_instance(
    *,
    name: str,
    item_count: int,
    capacity: int,
    instance_seed: int,
) -> dict[str, Any]:
    if item_count <= 0 or capacity <= 0:
        raise ValueError("item count and capacity must be positive")
    rng = random.Random(instance_seed)
    return {
        "name": name,
        "capacity": capacity,
        "weights": [rng.randint(1, capacity) for _ in range(item_count)],
    }


def _bin_packing(config: dict[str, Any], index: int, destination: Path) -> None:
    if config.get("weight_distribution", "uniform_integer") != "uniform_integer":
        raise ValueError("unsupported bin packing weight distribution")
    item_count = _choice(config["items"], index)
    capacity = int(config["capacity"])
    destination.write_text(
        json.dumps(
            make_bin_packing_instance(
                name=destination.stem,
                item_count=item_count,
                capacity=capacity,
                instance_seed=_instance_seed(config, index),
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
    return [rng.randint(1, 20) for _ in range(jobs)]


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


def prepare_collection_instances(
    source: Path,
    destination: Path,
    *,
    path_template: str | None = None,
) -> list[dict]:
    """Copy a fixed input panel using explicit paths, including archived indices.

    A template belongs to the experiment config, not the solver implementation.
    It permits reading old manifests without rewriting their retained artifacts.
    """
    import shutil

    payload = yaml.safe_load(source.read_text())
    destination.mkdir(parents=True, exist_ok=True)
    instances = []
    for item in payload["instances"]:
        identity = item.get("id", item.get("name"))
        relative = (
            path_template.format(**item)
            if path_template is not None
            else item.get("instance_file", item.get("path"))
        )
        if identity is None or relative is None:
            raise ValueError("each instance requires an ID and explicit input path")
        original = source.parent / relative
        target = destination / f"{identity}{''.join(original.suffixes)}"
        if target.exists():
            if target.read_bytes() != original.read_bytes():
                raise ValueError(f"prepared input changed: {target}")
        else:
            shutil.copy2(original, target)
        instances.append(
            {"id": identity, "path": str(target.resolve()), "bks": item.get("bks")}
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
) -> dict[str, Any]:
    from pitbench.problem_families.verification import (
        TrustedOptimumOracle,
        TrustedOptimumRecord,
    )
    from pitbench.schema.task import PitBenchTask

    task = PitBenchTask.from_yaml(task_config_path)
    instance_set_config = yaml.safe_load(instance_set_config_path.read_text())
    generator = instance_set_config["generator"]
    if instance_set_config.get("visibility") != "judge":
        raise ValueError("trusted optimum generation requires a judge instance set")
    if generator["kind"] not in {
        "single_machine_scheduling_mip",
        "bin_packing",
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
            else:
                instance = json.loads(instance_path.read_text())
                weights = instance["weights"]
                capacity = instance["capacity"]
                reference_bins = first_fit_decreasing_packing(weights, capacity)
                capacity_lower_bound = math.ceil(sum(weights) / capacity)
                if len(reference_bins) != capacity_lower_bound:
                    raise ValueError(
                        f"instance {instance_id} reference packing does not match "
                        "its capacity lower bound"
                    )
                optimal_objective = capacity_lower_bound
                problem_kind = "bin_packing"
                reference_solution = {"bins": reference_bins}
                optimality_basis = {
                    "kind": "capacity_lower_bound",
                    "total_weight": sum(weights),
                    "capacity": capacity,
                    "lower_bound": capacity_lower_bound,
                }

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
