import hashlib
from pathlib import Path

import pytest

from pitbench.evaluator.private_assets import PrivateAssetError, PrivateAssetResolver


def test_private_asset_hash_is_verified(tmp_path: Path) -> None:
    asset = tmp_path / "oracle.json"
    asset.write_bytes(b"trusted")
    expected = hashlib.sha256(b"trusted").hexdigest()
    resolver = PrivateAssetResolver(tmp_path)
    assert resolver.resolve("private://oracle.json", expected) == asset
    with pytest.raises(PrivateAssetError, match="hash mismatch"):
        resolver.resolve("private://oracle.json", "0" * 64)
