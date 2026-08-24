import tomllib
from pathlib import Path

from pitbench.harness.registry.client import DISTRIBUTION_NAME, RegistryClient

ROOT = Path(__file__).resolve().parents[3]


def test_registry_version_lookup_matches_distribution_metadata(monkeypatch) -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    requested: list[str] = []

    def version(distribution_name: str) -> str:
        requested.append(distribution_name)
        return project["version"]

    monkeypatch.setattr("importlib.metadata.version", version)

    client = RegistryClient(local_registry_path=ROOT / "registry.json")

    assert project["name"] == DISTRIBUTION_NAME == "fc-eval"
    assert requested == [project["name"]]
    assert client._current_version == project["version"]
