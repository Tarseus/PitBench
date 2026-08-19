from __future__ import annotations

import hashlib
from pathlib import Path


class PrivateAssetError(RuntimeError):
    pass


class PrivateAssetResolver:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def resolve(self, uri: str, sha256: str | None = None) -> Path:
        if not uri.startswith("private://"):
            raise PrivateAssetError(f"not a private URI: {uri}")
        relative = Path(uri.removeprefix("private://"))
        path = (self.root / relative).resolve()
        if self.root not in path.parents and path != self.root:
            raise PrivateAssetError(f"private URI escapes root: {uri}")
        if not path.is_file():
            raise PrivateAssetError(f"missing private asset: {uri}")
        if sha256 is not None:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != sha256:
                raise PrivateAssetError(
                    f"private asset hash mismatch for {uri}: {actual} != {sha256}"
                )
        return path
