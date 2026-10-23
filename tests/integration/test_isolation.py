"""B1 acceptance in pinned, existing solver images and actual private assets."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import pytest

from adapters.pitbench.adapter import PitBenchAdapter
from pitbench.evaluator.docker_judge import DockerJudge
from pitbench.evaluator.isolation import PRIVATE_ASSET_MOUNT
from pitbench.instances import materialize_instance_set
from pitbench.schema.task import InstanceSetKind
from pitbench.tasks import TaskCatalog

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.docker


@pytest.mark.parametrize(
    "task_id",
    [
        "pyvrp_v0_14_0",
        "highs_v1_15_1",
        "choco_v6_0_1",
        "ortools_cp_sat_solve_v9_15",
    ],
)
def test_real_image_isolation_and_existing_solver(task_id, tmp_path):
    record = TaskCatalog(ROOT).validate_one(task_id)
    task = record.task
    panel = next(
        item for item in task.instance_sets if item.kind == InstanceSetKind.AGENT_DEV
    )
    paths = materialize_instance_set(
        ROOT / panel.instance_set_config, tmp_path / "public"
    )
    instance = paths[0]
    private = ROOT / "private"
    secret = next(iter(sorted((private / "oracles").glob("*.yaml"))), None)
    assert secret is not None, "B1 acceptance requires an actual private oracle"
    repo = PitBenchAdapter.source_cache_path(ROOT, task)
    if repo.is_dir():
        PitBenchAdapter.validate_repository(task, repo)
    else:
        repo = tmp_path / "base"
        repo.mkdir()
    patch = tmp_path / "candidate.patch"
    patch.write_text("")
    output = tmp_path / "output"
    output.mkdir()
    judge = DockerJudge(
        image=task.repository.judge_image,
        task_config_path=record.task_config_path,
        base_repository=repo,
        private_root=private,
        candidate_patch=patch,
        output_dir=output,
        cpus=8,
        memory="8g",
    )
    command = judge.command()
    image_index = command.index(task.repository.judge_image)
    input_path = "/probe/instance" + instance.suffix
    command[image_index:image_index] = [
        "--pull",
        "never",
        "--volume",
        f"{instance}:{input_path}:ro",
        "--volume",
        f"{ROOT / 'tests/fixtures/isolation_probe.py'}:/probe/acceptance.py:ro",
    ]
    start = command.index("python3")
    command[start:] = [
        "python3",
        "/probe/acceptance.py",
        PRIVATE_ASSET_MOUNT + "/" + secret.relative_to(private).as_posix(),
        input_path,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=900)
    evidence_dir = ROOT / "report/b1-isolation-20261022"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    attempt_dir = evidence_dir / task_id / str(time.time_ns())
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "stdout.log").write_text(completed.stdout)
    (attempt_dir / "stderr.log").write_text(completed.stderr)
    (attempt_dir / "attempt.json").write_text(
        json.dumps(
            {
                "returncode": completed.returncode,
                "command": command,
            },
            indent=2,
        )
        + "\n"
    )
    assert completed.returncode == 0, completed.stderr[-8000:]
    result = json.loads(completed.stdout)
    result.update(
        {
            "image": task.repository.judge_image,
            "task_config_sha256": record.task_config_sha256,
            "instance_sha256": hashlib.sha256(instance.read_bytes()).hexdigest(),
            "private_asset_sha256": hashlib.sha256(secret.read_bytes()).hexdigest(),
            "private_asset": secret.relative_to(private).as_posix(),
            "command": command,
        }
    )
    (evidence_dir / f"{task_id}.json").write_text(json.dumps(result, indent=2) + "\n")
    (attempt_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    assert len(result["checks"]) == 10
    assert result["solver"]["independent_verification"]["feasible"]
