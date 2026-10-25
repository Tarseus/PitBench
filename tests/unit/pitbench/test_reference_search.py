from __future__ import annotations

import itertools
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml

from pitbench.evaluator.collection import (
    ReferenceCollection,
    run_collection_jobs,
    run_collection_process,
)
from pitbench.instances import materialize_instance_grid
from pitbench.problem_families.public_verification import check_cvrp
from pitbench.solver_drivers.reference import (
    SearchBudget,
    _bin_packing,
    _cvrp,
    _graph_coloring,
    _job_shop,
    collect_reference_solutions,
    solve_reference,
)

ROOT = Path(__file__).resolve().parents[3]


def test_weighted_cpu_scheduling_never_overlaps_allocated_cores():
    cpus = sorted(os.sched_getaffinity(0))[:3]
    if len(cpus) < 3:
        pytest.skip("requires three available CPU IDs")
    active = set()
    lock = threading.Lock()

    def execute(job, allocated):
        assert len(allocated) == job["threads"]
        with lock:
            assert not active.intersection(allocated)
            active.update(allocated)
        time.sleep(0.02)
        with lock:
            active.difference_update(allocated)
        return allocated

    results = run_collection_jobs(
        [{"threads": n} for n in [2, 1, 2, 3]],
        cpus,
        execute,
        cpu_request=lambda job: job["threads"],
    )
    assert [len(ids) for ids in results] == [2, 1, 2, 3]
    assert not active


def test_collection_watchdog_terminates_child_process_group(tmp_path):
    child_pid = tmp_path / "child.pid"
    program = (
        "import subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        f"open({str(child_pid)!r},'w').write(str(child.pid)); time.sleep(30)"
    )
    result = run_collection_process(
        [sys.executable, "-c", program], tmp_path / "job", {}, timeout=1
    )
    assert result["execution_status"] == "watchdog_timeout"
    if child_pid.exists():
        state = Path(f"/proc/{child_pid.read_text()}/stat")
        assert not state.exists() or state.read_text().split()[2] == "Z"


def test_reference_campaign_runs_and_resumes_without_duplicate_attempts(tmp_path):
    materialize_instance_grid(
        {
            "visibility": "agent",
            "generator": {
                "kind": "bin_packing",
                "count": 2,
                "items": [5],
                "capacity": 10,
                "randomness": {"instance_seed": 37},
            },
        },
        tmp_path / "grid",
    )
    config = {
        "cpus": sorted(os.sched_getaffinity(0))[:2],
        "manifests": [str(tmp_path / "grid/manifest.json")],
        "seeds": [0],
        "backends": [{"name": "native", "kinds": ["bin_packing"]}],
        "phases": [{"name": "proof", "seconds": 1}],
    }
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(yaml.safe_dump(config))
    output = tmp_path / "campaign"
    first = ReferenceCollection.run(config_path, output)
    second = ReferenceCollection.run(config_path, output)
    assert first["finished_jobs"] == second["finished_jobs"] == 2
    assert second["execution_counts"] == {"completed": 2}
    assert second["proven_optimal"] == 2
    assert len(list((output / "jobs").glob("**/attempt-*"))) == 2
    manifest = json.loads((output / "campaign.json").read_text())
    assert "pitbench/solver_drivers/reference.py" in manifest["source_hashes"]
    assert manifest["cpus"] == config["cpus"]


def test_bin_packing_search_improves_ffd_and_proves_optimum():
    result = _bin_packing(
        {"weights": [4, 4, 3, 3, 3, 3], "capacity": 10}, SearchBudget(1)
    )
    assert result.proven
    assert len(result.solution["bins"]) == 2


def test_graph_search_distinguishes_clique_bound_from_chromatic_optimum():
    result = _graph_coloring(
        {"vertex_count": 5, "edge_endpoints": [0, 1, 1, 2, 2, 3, 3, 4, 4, 0]},
        SearchBudget(1),
    )
    assert result.proven
    assert len(result.basis["clique_lower_bound"]) == 2
    assert max(result.solution["colors"]) + 1 == 3


def test_cvrp_dp_matches_independent_permutation_and_cut_enumeration():
    instance = {
        "depot": 0,
        "coordinates": [[0, 0], [2, 1], [3, 4], [7, 3], [1, 7]],
        "demands": [0, 2, 2, 3, 1],
        "capacity": 4,
        "distance_metric": "EUC_2D",
    }
    best = math.inf
    for permutation in itertools.permutations(range(1, 5)):
        for cuts in range(1 << 3):
            routes, route = [], []
            for i, customer in enumerate(permutation):
                route.append(customer)
                if i == 3 or cuts & (1 << i):
                    routes.append(route)
                    route = []
            checked = check_cvrp(instance, {"routes": routes})
            if checked["feasible"]:
                best = min(best, checked["objective"])
    result = _cvrp(instance, SearchBudget(1), 10)
    assert result.proven
    assert check_cvrp(instance, result.solution)["objective"] == best


def test_job_shop_search_matches_machine_order_enumeration():
    # Enumerate both possible orders independently on each machine.
    jobs = [[[0, 3], [1, 2]], [[1, 4], [0, 1]]]
    best = math.inf
    durations = [3, 2, 4, 1]
    for first, second in itertools.product([(0, 3), (3, 0)], [(1, 2), (2, 1)]):
        edges = [(0, 1), (2, 3), first, second]
        starts = [0] * 4
        for iteration in range(5):
            changed = False
            for a, b in edges:
                if starts[b] < starts[a] + durations[a]:
                    starts[b] = starts[a] + durations[a]
                    changed = True
            if not changed:
                best = min(best, max(starts[i] + durations[i] for i in range(4)))
                break
    result = _job_shop(jobs, SearchBudget(1))
    assert result.proven
    assert max(s + d for s, d in zip(result.solution["start_times"], durations)) == best


def test_time_limits_preserve_feasible_incumbents_without_optimality_claim():
    result = _bin_packing(
        {"weights": [4, 4, 3, 3, 3, 3], "capacity": 10}, SearchBudget(0)
    )
    assert result.search_status == "time_limit"
    assert not result.proven
    assert len(result.solution["bins"]) == 3
    instance = {
        "coordinates": [[0, 0], [1, 2], [3, 4]],
        "demands": [0, 2, 2],
        "capacity": 3,
        "distance_metric": "EUC_2D",
    }
    result = _cvrp(instance, SearchBudget(1), 1)
    assert not result.proven
    assert result.search_status == "exact_size_limit"
    assert check_cvrp(instance, result.solution)["feasible"]


@pytest.mark.parametrize(
    "panel_name",
    [
        "routing",
        "packing",
        "scheduling",
        "knapsack",
        "conflicts",
        "coloring",
        "subset",
        "jobshop",
    ],
)
def test_all_problem_families_have_independently_verified_references(
    tmp_path, panel_name
):
    plan = yaml.safe_load(
        (ROOT / "configs/experiments/problem_generation.yaml").read_text()
    )["panels"][panel_name]
    plan["generator"]["count"] = 1
    plan["axes"]["scale"] = {"tiny": plan["axes"]["scale"]["tiny"]}
    manifest = materialize_instance_grid(plan, tmp_path / "grid")
    for cell in manifest["conditions"]:
        entry = cell["instances"][0]
        result, checked = solve_reference(
            tmp_path / "grid" / entry["path"], cell["generator"], 0, seconds=2
        )
        assert checked["feasible"]
        assert result.proven


def test_reference_collection_retains_failed_attempts(tmp_path):
    plan = {
        "visibility": "agent",
        "generator": {
            "kind": "bin_packing",
            "count": 2,
            "items": [5],
            "capacity": 10,
            "randomness": {"instance_seed": 37},
        },
    }
    materialize_instance_grid(plan, tmp_path / "grid")
    # An unsupported backend kind simulates a solver setup failure while keeping inputs intact.
    manifest_path = tmp_path / "grid" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["conditions"][0]["generator"]["kind"] = "unsupported"
    manifest_path.write_text(json.dumps(manifest))
    summary = collect_reference_solutions(manifest_path, tmp_path / "refs", seconds=1)
    assert summary["counts"] == {"missing": 2}
    assert summary["failed_attempts"] == 2
    records = [
        json.loads(line)
        for line in (tmp_path / "refs/attempts.jsonl").read_text().splitlines()
    ]
    assert all(r["status"] == "failed" and r["instance_sha256"] for r in records)
