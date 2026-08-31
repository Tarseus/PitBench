import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from adapters.pitbench.agent_tooling import write_agent_tooling
from pitbench.instances import materialize_population
from pitbench.schema.task import PopulationKind
from pitbench.tasks import TaskCatalog

ROOT = Path(__file__).resolve().parents[3]


def test_pyvrp_runner_check_imports_native_extension(monkeypatch) -> None:
    from pitbench import agent_cli

    monkeypatch.setattr(
        agent_cli.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ModuleNotFoundError("pyvrp._pyvrp")),
    )

    available, detail = agent_cli._runner_available(
        {"runner_requirement": "pyvrp_import"}
    )

    assert available is False
    assert "pyvrp._pyvrp" in detail


@pytest.mark.parametrize(
    "record", TaskCatalog(ROOT).records(), ids=lambda item: item.task.task_id
)
def test_agent_cli_contract_for_every_task(record, tmp_path: Path) -> None:
    task_dir = tmp_path / record.task.task_id
    task_dir.mkdir()
    development = next(
        item
        for item in record.task.populations
        if item.kind == PopulationKind.AGENT_DEV
    )
    dev = task_dir / "dev_instances"
    instances = materialize_population(ROOT / development.manifest, dev)
    write_agent_tooling(repository_root=ROOT, task=record.task, task_dir=task_dir)

    repository = task_dir / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=repository, check=True)
    (repository / "README.md").write_text("fixture\n")
    subprocess.run(["git", "add", "README.md"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=PitBench",
            "-c",
            "user.email=pitbench.invalid",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
        cwd=repository,
        check=True,
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PITBENCH_AGENT_CONFIG": str(task_dir / "agent_config.json"),
            "PITBENCH_DEV_INSTANCES": str(dev),
            "PITBENCH_REPO": str(repository),
            "PITBENCH_RUNS": str(task_dir / "runs"),
            "PYTHONPATH": str(task_dir / "agent_tooling"),
        }
    )
    executable = task_dir / "agent_bin" / "pitbench"

    inspected = subprocess.run(
        [str(executable), "inspect"],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(inspected.stdout)
    assert payload["task_id"] == record.task.task_id
    assert len(payload["development_instances"]) == development.size
    assert payload["editable_paths"] == record.task.repository.editable_paths
    assert payload["validation_available"] is True

    instance = instances[0].stem
    for command in (
        ["bench", "--split", "dev", "--instance", instance, "--dry-run"],
        ["profile", "--instance", instance, "--dry-run"],
        ["check"],
        ["diff"],
    ):
        subprocess.run(
            [str(executable), *command],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )

    visible = "\n".join(
        path.read_text(errors="replace")
        for path in task_dir.rglob("*")
        if path.is_file()
    )
    assert "private://" not in visible


def test_agent_cli_check_reports_workspace_state_without_gating(
    monkeypatch, tmp_path, capsys
):
    from pitbench import agent_cli

    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=repository, check=True)
    (repository / "README.md").write_text("fixture\n")
    subprocess.run(["git", "add", "README.md"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=PitBench",
            "-c",
            "user.email=pitbench.invalid",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
        cwd=repository,
        check=True,
    )
    (repository / "outside.txt").write_text("visible but not editable\n")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"editable_paths": ["src"]}))
    monkeypatch.setenv("PITBENCH_REPO", str(repository))
    monkeypatch.setenv("PITBENCH_AGENT_CONFIG", str(config))

    assert agent_cli.check_command(SimpleNamespace()) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "editable_paths": ["src"],
        "changed_paths": ["outside.txt"],
        "outside_editable_paths": ["outside.txt"],
    }


def test_agent_cli_workspace_verifies_capability_boundary(
    monkeypatch, tmp_path, capsys
):
    from pitbench import agent_cli

    repository = tmp_path / "repo"
    repository.mkdir()
    git_dir = repository / ".git"
    git_dir.mkdir()
    editable = repository / "src"
    editable.mkdir()
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"editable_paths": ["src"]}))
    monkeypatch.setenv("PITBENCH_REPO", str(repository))
    monkeypatch.setenv("PITBENCH_AGENT_CONFIG", str(config))

    git_dir.chmod(0o555)
    repository.chmod(0o555)
    try:
        assert agent_cli.workspace_command(SimpleNamespace()) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["enforced"] is True
        assert payload["repository_root_writable"] is False
        assert payload["git_metadata_writable"] is False
        assert payload["editable_paths_writable"] == {"src": True}
    finally:
        repository.chmod(0o755)
        git_dir.chmod(0o755)


def test_agent_cli_validate_uses_clean_clone_and_candidate_patch(monkeypatch, tmp_path):
    from pitbench import agent_cli

    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=repository, check=True)
    (repository / "README.md").write_text("fixture\n")
    subprocess.run(["git", "add", "README.md"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=PitBench",
            "-c",
            "user.email=pitbench.invalid",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
        cwd=repository,
        check=True,
    )
    (repository / "candidate.txt").write_text("changed\n")
    generated = repository / ".pitbench/runs"
    generated.mkdir(parents=True)
    (generated / "result.json").write_text("{}\n")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "editable_paths": ["src"],
                "validation_commands": [
                    {
                        "argv": [
                            "python",
                            "-c",
                            (
                                "from pathlib import Path; "
                                "assert Path('candidate.txt').read_text() == "
                                "'changed\\n'; "
                                "assert not Path('.pitbench/runs/result.json').exists()"
                            ),
                        ],
                        "cwd": ".",
                        "env": {},
                        "timeout_sec": 30,
                    }
                ],
            }
        )
    )
    monkeypatch.setenv("PITBENCH_REPO", str(repository))
    monkeypatch.setenv("PITBENCH_AGENT_CONFIG", str(config))

    index_before = (repository / ".git/index").read_bytes()
    assert agent_cli.validate_command(SimpleNamespace()) == 0
    assert (repository / ".git/index").read_bytes() == index_before
    assert "?? candidate.txt" in subprocess.check_output(
        ["git", "status", "--short"], cwd=repository, text=True
    )
