import hashlib
from pathlib import Path

import pytest

from pitbench.evaluator.private_assets import PrivateAssetError, PrivateAssetResolver


def test_private_asset_hash_is_verified(tmp_path: Path) -> None:
    patch = tmp_path / "human.patch"
    patch.write_bytes(b"trusted")
    expected = hashlib.sha256(b"trusted").hexdigest()
    resolver = PrivateAssetResolver(tmp_path)
    assert resolver.resolve("private://human.patch", expected) == patch
    with pytest.raises(PrivateAssetError, match="hash mismatch"):
        resolver.resolve("private://human.patch", "0" * 64)
