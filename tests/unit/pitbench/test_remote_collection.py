from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import pitbench.evaluator.collection as collection
import pitbench.evaluator.remote as remote


def test_partition_preserves_jobs_seeds_and_budgets_without_overlap():
    jobs = [
        {
            "id": f"job-{i}",
            "instance": {"id": f"instance-{i % 5}"},
            "seconds": 3600 if i % 2 else 14400,
            "threads": 8 if i % 3 else 1,
            "seed": i % 6,
        }
        for i in range(40)
    ]
    original = copy.deepcopy(jobs)
    hosts = [
        {"host": "a", "cpus": list(range(32))},
        {"host": "b", "cpus": list(range(16))},
    ]
    assigned, omitted = remote.partition_reference_jobs(
        jobs, hosts, {"job-1"}, {"instance-3": {"proven_optimal": True}}
    )
    flattened = [j for group in assigned.values() for j in group]
    identities = [j["id"] for j in flattened]
    assert len(set(identities)) == len(identities)
    assert set(identities) | {r["job_id"] for r in omitted} == {j["id"] for j in jobs}
    assert all(
        j == next(old for old in original if old["id"] == j["id"]) for j in flattened
    )
    assert jobs == original
    assert (
        assigned
        == remote.partition_reference_jobs(
            jobs, hosts, {"job-1"}, {"instance-3": {"proven_optimal": True}}
        )[0]
    )


def test_partition_respects_host_capacity_and_rejects_impossible_request():
    jobs = [
        {"id": str(i), "instance": {"id": str(i)}, "seconds": 100, "threads": 1}
        for i in range(48)
    ]
    hosts = [
        {"host": "a", "cpus": list(range(32))},
        {"host": "b", "cpus": list(range(16))},
    ]
    assigned, _ = remote.partition_reference_jobs(jobs, hosts, set(), {})
    assert len(assigned["a"]) == 32
    assert len(assigned["b"]) == 16
    with pytest.raises(ValueError, match="no host fits"):
        remote.partition_reference_jobs([{**jobs[0], "threads": 64}], hosts, set(), {})


def test_host_template_relocates_inputs_and_warm_starts_without_mutating_source():
    instance = {
        "id": "routing/scale-tiny/one",
        "path": "/source/instance.json",
        "sha256": "a" * 64,
        "generator": {"kind": "euclidean_cvrp"},
        "replicate_index": 0,
    }
    backend = {"name": "pyvrp", "threads": 1, "kinds": ["euclidean_cvrp"]}
    job = {
        "id": "long/one/seed-2",
        "instance": instance,
        "backend": backend,
        "seed": 2,
        "seconds": 3600,
        "threads": 1,
    }
    record = {
        "instance": instance["id"],
        "instance_path": instance["path"],
        "solution": "/source/solution.json",
        "solution_sha256": "b" * 64,
    }
    distribution = {
        "source_campaign": "/source/campaign",
        "source_manifest_sha256": "c" * 64,
        "files": {
            instance["path"]: "data/one.json",
            record["solution"]: "data/warm.json",
        },
        "assignments": {"host-a": [job]},
        "instances": [instance],
        "references": {instance["id"]: record},
        "original_manifest": {
            "config": {},
            "backends": [backend],
            "python_version": "3.12",
        },
        "config": {"resource_guard": {"max_load1": 64}, "native_builds": {}},
        "bundle_sha256": "d" * 64,
    }
    before = copy.deepcopy(distribution)
    result = remote.RemoteReferenceCollection.host_template(
        distribution,
        {"host": "host-a", "root": "/var/tmp/reference", "cpus": list(range(32))},
    )
    assert distribution == before
    assert result["manifest"]["jobs"][0]["seed"] == 2
    assert result["manifest"]["jobs"][0]["seconds"] == 3600
    assert (
        result["manifest"]["jobs"][0]["instance"]["path"]
        == "/var/tmp/reference/project/data/one.json"
    )
    assert (
        result["references"][instance["id"]]["solution"]
        == "/var/tmp/reference/project/data/warm.json"
    )


def test_shared_solution_cache_verifies_each_instance_and_reported_objective(tmp_path):
    solution = tmp_path / "solution.json"
    solution.write_text(json.dumps({"routes": [[1]]}))
    digest = hashlib.sha256(solution.read_bytes()).hexdigest()
    instances = []
    for distance in [1, 3]:
        path = tmp_path / f"instance-{distance}.json"
        path.write_text(
            json.dumps(
                {
                    "coordinates": [[0, 0], [distance, 0]],
                    "demands": [0, 1],
                    "capacity": 1,
                    "distance_metric": "EUC_2D",
                }
            )
        )
        instances.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "generator": {"kind": "euclidean_cvrp"},
            }
        )
    record = {"solution_sha256": digest, "objective": 2}
    assert (
        remote.verify_collected_solution(
            instances[0], solution, record, tmp_path / "cache"
        )["objective"]
        == 2
    )
    with pytest.raises(ValueError, match="verification"):
        remote.verify_collected_solution(
            instances[1], solution, record, tmp_path / "cache"
        )
    right = {**record, "objective": 6}
    assert (
        remote.verify_collected_solution(
            instances[1], solution, right, tmp_path / "cache"
        )["objective"]
        == 6
    )
    with pytest.raises(ValueError, match="does not match"):
        remote.verify_collected_solution(
            instances[1], solution, record, tmp_path / "cache"
        )


def test_credential_file_requires_private_permissions(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "ROOT", tmp_path)
    path = tmp_path / "credentials.json"
    path.write_text(
        json.dumps(
            {"hosts": [{"host": "host-a", "username": "user", "password": "test-only"}]}
        )
    )
    path.chmod(0o644)
    with pytest.raises(ValueError, match="0600"):
        remote.RemoteReferenceCollection._credentials({"credentials_file": path.name})
    path.chmod(0o600)
    assert (
        remote.RemoteReferenceCollection._credentials({"credentials_file": path.name})[
            "host-a"
        ]["username"]
        == "user"
    )


def test_resource_guard_blocks_admission_on_load_memory_and_disk(tmp_path, monkeypatch):
    original = Path.read_text

    def read(path, *args, **kwargs):
        return (
            "MemAvailable: 8388608 kB\n"
            if str(path) == "/proc/meminfo"
            else original(path, *args, **kwargs)
        )

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(Path, "glob", lambda *args: iter([]))
    monkeypatch.setattr(os, "getloadavg", lambda: (90, 80, 70))
    monkeypatch.setattr(
        "shutil.disk_usage", lambda path: SimpleNamespace(free=3 * 1024**3)
    )
    state = collection.reference_resource_state(
        {
            "max_load1": 64,
            "min_memory_available_gib": 16,
            "min_disk_free_gib": 4,
            "max_temperature_celsius": 75,
        },
        tmp_path,
    )
    assert set(state["blocked_by"]) == {"load", "memory", "disk"}


def test_reboot_recovery_retains_checkpoint_and_marks_interruption(tmp_path):
    directory = tmp_path / "jobs/one/attempt-0000"
    directory.mkdir(parents=True)
    (directory / "running.json").write_text(
        json.dumps(
            {"job_id": "one", "pid": os.getpid(), "worker_boot_id": "previous-boot"}
        )
    )
    (directory / "incumbent.json").write_text(
        json.dumps({"objective": 12, "solution": "/retained/solution.json"})
    )
    recovered = collection.ReferenceCollection.recover_interrupted(tmp_path)
    record = json.loads((directory / "result.json").read_text())
    assert len(recovered) == 1
    assert record["execution_status"] == "interrupted"
    assert record["objective"] == 12
    assert record["solution"] == "/retained/solution.json"
    assert collection.ReferenceCollection.recover_interrupted(tmp_path) == []
