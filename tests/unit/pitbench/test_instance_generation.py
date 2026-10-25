from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from pitbench.instances import make_euclidean_cvrp_instance, materialize_instance_grid
from pitbench.instances.generate import (
    _job_shop_operations,
    _require_planted_optimum,
    graph_coloring_data,
    job_shop_data,
    make_bin_packing_instance,
    materialize_instance_suite,
    multidimensional_knapsack_data,
    scheduling_processing_times,
)

ROOT = Path(__file__).resolve().parents[3]


def test_all_current_problem_families_generate_reproducible_suites(tmp_path):
    plan = yaml.safe_load(
        (ROOT / "configs/experiments/problem_generation.yaml").read_text()
    )
    a = materialize_instance_suite(plan, tmp_path / "a")
    b = materialize_instance_suite(plan, tmp_path / "b")
    assert a == b
    assert sum(p["generated_instances"] for p in a["panels"].values()) == 114
    assert {panel["generator"]["kind"] for panel in plan["panels"].values()} == {
        "euclidean_cvrp",
        "bin_packing",
        "single_machine_scheduling_mip",
        "multidimensional_knapsack_mip",
        "binary_clique_cover_mip",
        "graph_coloring",
        "balanced_subset_sum",
        "job_shop_scheduling",
    }
    for name in a["panels"]:
        first = json.loads((tmp_path / "a" / name / "manifest.json").read_text())
        second = json.loads((tmp_path / "b" / name / "manifest.json").read_text())
        assert first == second
        for cell in first["conditions"]:
            for entry in cell["instances"]:
                path = tmp_path / "a" / name / entry["path"]
                assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]


def test_high_precision_subset_generation_does_not_allocate_optimum_dp(
    tmp_path, monkeypatch
):
    import pitbench.instances.generate as generation

    def no_proof(*args, **kwargs):
        raise AssertionError("generation must not run an optimum DP")

    monkeypatch.setattr(generation, "balanced_subset_sum_data", no_proof)
    manifest = materialize_instance_grid(
        {
            "visibility": "agent",
            "generator": {
                "kind": "balanced_subset_sum",
                "count": 1,
                "item_counts": [60],
                "max_weight": 100000000,
                "randomness": {"instance_seed": 37},
            },
        },
        tmp_path / "grid",
    )
    instance = json.loads(
        (
            tmp_path / "grid" / manifest["conditions"][0]["instances"][0]["path"]
        ).read_text()
    )
    assert len(instance["weights"]) == 60
    assert sum(instance["weights"]) > 100000000


def test_distribution_controls_and_legacy_integer_streams():
    import random

    packing = make_bin_packing_instance(
        name="packing", item_count=20, capacity=100, instance_seed=37
    )
    rng = random.Random(37)
    assert packing["weights"] == [rng.randint(1, 100) for _ in range(20)]
    config = {"jobs": [20], "randomness": {"instance_seed": 37}}
    rng = random.Random(37)
    assert scheduling_processing_times(config, 0) == [
        rng.randint(1, 20) for _ in range(20)
    ]
    packing = make_bin_packing_instance(
        name="packing",
        item_count=20,
        capacity=100,
        instance_seed=37,
        weight_distribution="discrete",
        weight_parameters={"values": [20, 80], "weights": [1, 0]},
    )
    assert packing["weights"] == [20] * 20
    with pytest.raises(ValueError, match="exceeds capacity"):
        make_bin_packing_instance(
            name="bad",
            item_count=2,
            capacity=5,
            instance_seed=1,
            weight_parameters={"low": 6, "high": 6},
        )


def test_random_graph_size_is_independent_of_partition_count():
    config = {
        "graph_model": "planted_partition",
        "vertex_counts": [60],
        "colors": [3],
        "edge_probability": 0.6,
        "randomness": {"instance_seed": 37},
    }
    instance, colors, basis = graph_coloring_data(config, 0)
    assert instance["vertex_count"] == 60
    assert len(set(colors)) == 3
    assert basis["kind"] == "feasible_coloring_only"
    endpoints = instance["edge_endpoints"]
    assert all(colors[a] != colors[b] for a, b in zip(endpoints[::2], endpoints[1::2]))


def test_random_generators_cannot_inherit_planted_optimum_claims():
    for config in (
        {"kind": "graph_coloring", "graph_model": "erdos_renyi"},
        {"kind": "multidimensional_knapsack_mip", "capacity_mode": "ratio"},
    ):
        with pytest.raises(ValueError, match="requires"):
            _require_planted_optimum(config)
    weights, caps, selected = multidimensional_knapsack_data(
        {
            "items": [12],
            "dimensions": [3],
            "weight_max": 20,
            "capacity_mode": "ratio",
            "capacity_ratio": 0.3,
            "dimension_correlation": "shared",
            "randomness": {"instance_seed": 37},
        },
        0,
    )
    assert selected == []
    assert weights[0] == weights[1] == weights[2]
    assert caps == [int(sum(weights[0]) * 0.3)] * 3


def test_job_shop_generator_machine_orders_and_text_bridge(tmp_path):
    config = {
        "kind": "job_shop_scheduling",
        "count": 1,
        "jobs": [5],
        "machines": [4],
        "randomness": {"instance_seed": 37},
    }
    jobs = job_shop_data(config, 0)
    assert all(sorted(machine for machine, _ in job) == list(range(4)) for job in jobs)
    shared = job_shop_data({**config, "machine_order": "shared"}, 0)
    assert len({tuple(machine for machine, _ in job) for job in shared}) == 1
    assert [[duration for _, duration in job] for job in jobs] == [
        [duration for _, duration in job] for job in shared
    ]
    manifest = materialize_instance_grid(
        {"visibility": "judge", "generator": config}, tmp_path / "grid"
    )
    path = tmp_path / "grid" / manifest["conditions"][0]["instances"][0]["path"]
    assert _job_shop_operations(path) == jobs


def _instance(**changes):
    return make_euclidean_cvrp_instance(
        **{
            "name": "frozen",
            "customers": 25,
            "coordinate_seed": 11501,
            "demand_seed": 11502,
            "capacity_ratio": 0.18,
            **changes,
        }
    )


@pytest.mark.parametrize(
    "spatial,demand,expected",
    [
        (
            "uniform",
            "uniform_integer",
            "6de4a8269b1e8293892e70e73f1a5c1c663f7e72de86813790f17dbb4cbd07bb",
        ),
        (
            "uniform",
            "bimodal",
            "21d9e5510b55227907e2bb64fb4514847c33110528914aa6bb82fb09485bc7a7",
        ),
        (
            "uniform",
            "depot_correlated",
            "33587ad5b731fe6c5679eb6de4b55a65ac145ba1928a995c860b47d76b653a57",
        ),
        (
            "uniform",
            "depot_anticorrelated",
            "da081d7622e588cb8e7fcacf332092ab90186dece02c5fabb974c8e3492a3f24",
        ),
        (
            "clustered",
            "uniform_integer",
            "18e14bd3167a47d10e041c9d0e392e1d482bc58de71822709973606ea0e33696",
        ),
        (
            "clustered",
            "bimodal",
            "8993917ba889e1ade28e1ae094bbdd3770b7cf4957b171e888489da9d75cd580",
        ),
        (
            "clustered",
            "depot_correlated",
            "846d8d36003971a90e9dacfa3d7571bf2e6961967cf86b666cf6e2415b331cb4",
        ),
        (
            "clustered",
            "depot_anticorrelated",
            "e1b2876f4e72d7641b7db02be715d5743385712531ecc93ae7844d65f762f9ad",
        ),
    ],
)
def test_existing_cvrp_generation_preserves_frozen_inputs(spatial, demand, expected):
    instance = _instance(coordinate_distribution=spatial, demand_distribution=demand)
    assert (
        hashlib.sha256(json.dumps(instance, indent=2).encode()).hexdigest() == expected
    )


@pytest.mark.parametrize("spatial", ["uniform", "clustered", "mixed"])
def test_cvrp_scale_changes_preserve_sample_prefix_and_fixed_capacity(spatial):
    small = _instance(
        customers=30, capacity_ratio=None, capacity=40, coordinate_distribution=spatial
    )
    large = _instance(
        customers=90, capacity_ratio=None, capacity=40, coordinate_distribution=spatial
    )
    assert small["coordinates"] == large["coordinates"][:31]
    assert small["demands"] == large["demands"][:31]
    assert small["capacity"] == large["capacity"] == 40


def test_cvrp_spatial_and_demand_controls_are_independent():
    uniform = _instance()
    clustered = _instance(coordinate_distribution="clustered")
    bimodal = _instance(demand_distribution="bimodal")
    assert uniform["demands"] == clustered["demands"]
    assert uniform["coordinates"] != clustered["coordinates"]
    assert uniform["coordinates"] == bimodal["coordinates"]
    assert uniform["demands"] != bimodal["demands"]
    mixed = _instance(coordinate_distribution="mixed", uniform_fraction=0)
    assert mixed["coordinates"] == clustered["coordinates"]


def test_cvrp_custom_demands_and_spatial_correlations():
    discrete = _instance(
        demand_distribution="discrete",
        demand_parameters={"values": [2, 7], "weights": [0, 1]},
    )
    assert discrete["demands"][1:] == [7] * 25
    bimodal = _instance(
        demand_distribution="bimodal",
        demand_parameters={"low": [2, 2], "high": [9, 9], "high_probability": 0},
    )
    assert bimodal["demands"][1:] == [2] * 25
    uniform = _instance(demand_parameters={"low": 3, "high": 8})
    assert all(3 <= d <= 8 for d in uniform["demands"][1:])
    for mode in ["depot_correlated", "depot_anticorrelated"]:
        linked = _instance(
            demand_distribution=mode, demand_parameters={"low": 3, "high": 8}
        )
        assert sorted(linked["demands"]) == sorted(uniform["demands"])
        by_distance = sorted(
            range(1, 26),
            key=lambda i: math.dist(linked["coordinates"][i], linked["coordinates"][0]),
        )
        ordered = [linked["demands"][i] for i in by_distance]
        assert ordered == sorted(ordered, reverse=mode == "depot_anticorrelated")


@pytest.mark.parametrize(
    "changes",
    [
        {"capacity": 50},
        {"capacity": 1, "capacity_ratio": None},
        {"capacity": 50.5, "capacity_ratio": None},
        {"demand_parameters": {"low": 0}},
        {"demand_parameters": {"unknown": 1}},
        {
            "demand_distribution": "bimodal",
            "demand_parameters": {"high_probability": 2},
        },
        {
            "demand_distribution": "discrete",
            "demand_parameters": {"values": [1], "weights": [0]},
        },
        {
            "demand_distribution": "discrete",
            "demand_parameters": {"values": [1], "weights": [float("nan")]},
        },
        {"coordinate_distribution": "mixed", "uniform_fraction": 2},
    ],
)
def test_invalid_distribution_or_capacity_does_not_generate_mislabeled_data(changes):
    with pytest.raises(ValueError):
        _instance(**changes)


def _plan():
    return {
        "visibility": "agent",
        "generator": {
            "kind": "euclidean_cvrp",
            "count": 2,
            "capacity": 50,
            "randomness": {"coordinate_seed": 100, "demand_seed": 200},
        },
        "axes": {
            "spatial": {
                "uniform": {"coordinate_distribution": "uniform"},
                "clustered": {"coordinate_distribution": "clustered"},
            },
            "scale": {"small": {"customers": [12]}, "large": {"customers": [24]}},
        },
    }


def test_generation_grid_keeps_all_combinations_and_reproducible_instances(tmp_path):
    result = materialize_instance_grid(_plan(), tmp_path / "one")
    repeated = materialize_instance_grid(_plan(), tmp_path / "two")
    assert result == repeated
    assert result["status"] == "completed"
    assert result["generated_instances"] == result["expected_instances"] == 8
    assert len(result["conditions"]) == 4
    assert result["reference_status"] == "optimum_and_bks_not_computed"
    assert result["solver_runs"] == 0
    inputs = {}
    for cell in result["conditions"]:
        assert cell["status"] == "completed"
        for entry in cell["instances"]:
            path = tmp_path / "one" / entry["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
            assert entry["features"]["customers"] == cell["generator"]["customers"][0]
            assert entry["features"]["feasibility_witness"] == "one_customer_per_route"
            inputs[
                (
                    cell["axes"]["spatial"],
                    cell["axes"]["scale"],
                    entry["replicate_index"],
                )
            ] = json.loads(path.read_text())
    for scale in ["small", "large"]:
        for replicate in [0, 1]:
            assert (
                inputs[("uniform", scale, replicate)]["demands"]
                == inputs[("clustered", scale, replicate)]["demands"]
            )
    with pytest.raises(FileExistsError):
        materialize_instance_grid(_plan(), tmp_path / "one")


def test_generated_judge_index_preserves_visibility(tmp_path):
    plan = _plan()
    plan["visibility"] = "judge"
    result = materialize_instance_grid(plan, tmp_path / "judge")
    for cell in result["conditions"]:
        index = yaml.safe_load(
            (tmp_path / "judge" / cell["instance_set_config"]).read_text()
        )
        assert index["visibility"] == "judge"


def test_generation_failure_retains_completed_failed_and_pending_conditions(tmp_path):
    plan = _plan()
    plan["generator"]["customers"] = [12]
    plan["generator"]["demand_parameters"] = {"low": 10, "high": 10}
    plan["axes"] = {
        "capacity": {
            "feasible": {"capacity": 20},
            "invalid": {"capacity": 1},
            "pending": {"capacity": 30},
        }
    }
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="exceeds vehicle capacity"):
        materialize_instance_grid(plan, output)
    result = json.loads((output / "manifest.json").read_text())
    assert result["status"] == "failed"
    assert [c["status"] for c in result["conditions"]] == [
        "completed",
        "failed",
        "pending",
    ]
    assert result["generated_instances"] == 2
    assert result["expected_instances"] == 6
    assert result["conditions"][1]["error"]["type"] == "ValueError"
    assert (output / "features.csv").is_file()


def test_generation_axes_cannot_overwrite_each_other(tmp_path):
    plan = _plan()
    plan["axes"]["conflicting"] = {"other": {"customers": [100]}}
    with pytest.raises(ValueError, match="same generator parameter"):
        materialize_instance_grid(plan, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_generation_cli_uses_the_configured_generator(tmp_path):
    config = tmp_path / "input.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "visibility": "agent",
                "generator": {
                    "kind": "bin_packing",
                    "count": 2,
                    "items": [5],
                    "capacity": 20,
                    "randomness": {"instance_seed": 500},
                },
            }
        )
    )
    output = tmp_path / "output"
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.generate_instances",
            "--config",
            str(config),
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(process.stdout)["instances"] == 2
    instances = list((output / "conditions" / "baseline").glob("dev_*.json"))
    assert len(instances) == 2
    assert all(len(json.loads(p.read_text())["weights"]) == 5 for p in instances)
