import hashlib
from pathlib import Path

import pytest

from adapters.pitbench.adapter import PitBenchAdapter
from pitbench.evaluator.docker_judge import DockerJudge
from pitbench.evaluator.private_assets import PrivateAssetError, PrivateAssetResolver


def test_relocated_private_namespace_resolves_and_mounts_real_assets(tmp_path: Path):
    namespace = tmp_path / "private"
    storage = tmp_path / "organized" / "private"
    namespace.mkdir()
    storage.mkdir(parents=True)
    asset = storage / "seeds.yaml"
    asset.write_text("seeds: [1, 2]\n")
    namespace.joinpath(".asset-root").symlink_to(storage, target_is_directory=True)
    resolver = PrivateAssetResolver(namespace)
    expected_hash = hashlib.sha256(asset.read_bytes()).hexdigest()
    assert resolver.resolve("private://seeds.yaml", expected_hash) == asset

    task = tmp_path / "task.yaml"
    patch = tmp_path / "candidate.patch"
    task.write_text("task_id: example\n")
    patch.write_text("")
    judge = DockerJudge(
        image="sha256:" + "a" * 64,
        task_config_path=task,
        base_repository=tmp_path,
        private_root=namespace,
        candidate_patch=patch,
        output_dir=tmp_path / "evaluation",
    )
    assert judge.private_root == storage
    assert PitBenchAdapter(tmp_path, namespace).private_root == storage
    with pytest.raises(PrivateAssetError, match="hash mismatch"):
        resolver.resolve("private://seeds.yaml", "0" * 64)


@pytest.mark.parametrize("escape", ["../outside.yaml", "escape.yaml"])
def test_relocated_namespace_still_rejects_assets_outside_root(tmp_path, escape):
    namespace = tmp_path / "private"
    storage = tmp_path / "assets"
    namespace.mkdir()
    storage.mkdir()
    namespace.joinpath(".asset-root").symlink_to(storage, target_is_directory=True)
    outside = tmp_path / "outside.yaml"
    outside.write_text("outside\n")
    storage.joinpath("escape.yaml").symlink_to(outside)
    with pytest.raises(PrivateAssetError, match="escapes root"):
        PrivateAssetResolver(namespace).resolve("private://" + escape)
