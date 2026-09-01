from pathlib import Path

from ruamel.yaml import YAML

from adapters.pitbench.adapter import PitBenchAdapter
from pitbench.tasks import TaskCatalog

ROOT = Path(__file__).resolve().parents[3]


def _solver(task_plugin: str) -> str:
    return task_plugin.split(".")[2].split(":")[0]


def test_publication_matrix_covers_every_release_task_config() -> None:
    workflow = YAML(typ="safe").load(
        (ROOT / ".github/workflows/publish-solver-images.yml").read_text()
    )
    entries = workflow["jobs"]["publish"]["strategy"]["matrix"]["include"]
    published = {(item["solver"], item["base_commit"]) for item in entries}

    tasks = [
        record.task
        for record in TaskCatalog(ROOT).validate_all()
        if record.task.task_id.startswith("pyvrp_")
    ]
    expected = {
        (_solver(task.repository.plugin), task.release.base_commit) for task in tasks
    }
    assert published == expected

    workflow_text = (ROOT / ".github/workflows/publish-solver-images.yml").read_text()
    assert "Verify downloaded-image execution path" in workflow_text
    assert "--network none" in workflow_text
    assert "pitbench inspect" in workflow_text
    assert "pitbench bench --split dev" in workflow_text
    assert "--agent-tool inspect" in workflow_text
    assert "--agent-tool bench" in workflow_text

    for task in tasks:
        solver = "pyvrp"
        assert task.repository.agent_image == (
            f"ghcr.io/tarseus/pitbench-{solver}:{task.release.base_commit}"
        )
        recipe = ROOT / "docker/solver-images" / solver / "Dockerfile"
        assert recipe.is_file()
        contents = recipe.read_text()
        assert "ARG BASE_COMMIT" in contents
        assert "fetch --depth=1 --no-tags" in contents
        assert "checkout --detach FETCH_HEAD" in contents
        assert "remote remove origin" in contents
        assert "import pyvrp._pyvrp" in contents
        assert "import pyvrp.search._search" in contents


def test_prepared_task_context_excludes_solver_checkout() -> None:
    dockerignore = set(
        line.strip() for line in PitBenchAdapter._dockerignore().splitlines()
    )
    assert PitBenchAdapter._dockerignore() == "*\n!Dockerfile\n"
    dockerignore = set(
        line.strip() for line in PitBenchAdapter._dockerignore(["bench"]).splitlines()
    )
    assert dockerignore >= {"*", "!agent_tooling/**", "!dev_instances/**"}
    assert "!repo/**" not in dockerignore
