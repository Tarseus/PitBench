from pathlib import Path

from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[3]


def test_judge_image_recipe_contains_only_public_environment_inputs() -> None:
    recipe = ROOT / "docker/judge-images/pyvrp/Dockerfile"
    contents = recipe.read_text()

    assert "FROM ${UBUNTU_BASE_IMAGE}" in contents
    assert "libspdlog-dev" in contents
    assert "gcc" in contents or "build-essential" in contents
    assert "python3-dev" in contents
    assert "COPY" not in contents
    assert "ADD" not in contents
    assert "private://" not in contents


def test_judge_image_workflow_verifies_before_publishing() -> None:
    workflow_path = ROOT / ".github/workflows/publish-judge-image.yml"
    workflow = YAML(typ="safe").load(workflow_path.read_text())
    steps = workflow["jobs"]["publish"]["steps"]
    step_names = [step.get("name") for step in steps]

    assert step_names.index("Verify evaluator execution path") < step_names.index(
        "Publish verified judge image"
    )

    contents = workflow_path.read_text()
    assert "--network none" in contents
    assert "--read-only" in contents
    assert "PYTHONPATH=/opt/pitbench" in contents
    assert "push: true" in contents
    assert "ghcr.io/tarseus/pitbench-pyvrp-judge" in contents
