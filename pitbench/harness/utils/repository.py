"""Repository evidence shared by agent tracing and final candidate capture."""

from __future__ import annotations

import io
import tarfile


def archive_repository_head(container, *, workdir: str | None, head: str) -> bytes:
    """Archive tracked blobs, including files excluded/rewritten by git archive.

    Reading Git objects in batches avoids changing the working tree, index or
    attributes. The archive contains source files, not the repository's history.
    """
    result = container.exec_run(
        ["timeout", "10s", "git", "ls-tree", "-rz", head],
        workdir=workdir,
    )
    if result.exit_code != 0 or not isinstance(result.output, bytes):
        raise RuntimeError("could not enumerate repository HEAD")
    entries = []
    for entry in result.output.split(b"\0"):
        if not entry:
            continue
        attributes, path = entry.split(b"\t", 1)
        mode, kind, oid = attributes.decode("ascii").split()
        if kind != "blob":
            raise ValueError("source archive cannot reconstruct submodule contents")
        entries.append((mode, oid, path.decode(errors="surrogateescape")))
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for start in range(0, len(entries), 512):
            batch = entries[start : start + 512]
            result = container.exec_run(
                [
                    "timeout",
                    "10s",
                    "sh",
                    "-c",
                    'printf "%s\\n" "$@" | git cat-file --batch',
                    "pitbench-source-archive",
                    *(oid for _, oid, _ in batch),
                ],
                workdir=workdir,
            )
            if result.exit_code != 0 or not isinstance(result.output, bytes):
                raise RuntimeError("could not read repository HEAD blobs")
            blobs = io.BytesIO(result.output)
            for mode, oid, path in batch:
                found_oid, kind, size = blobs.readline().decode("ascii").split()
                content = blobs.read(int(size))
                if found_oid != oid or kind != "blob" or blobs.read(1) != b"\n":
                    raise ValueError("invalid Git blob response")
                info = tarfile.TarInfo(path)
                if mode == "120000":
                    info.type = tarfile.SYMTYPE
                    info.linkname = content.decode(errors="surrogateescape")
                    archive.addfile(info)
                else:
                    info.mode = int(mode, 8) & 0o777
                    info.size = len(content)
                    archive.addfile(info, io.BytesIO(content))
    return output.getvalue()


def capture_repository_patch(
    container, *, workdir: str | None, timeout_sec=None
) -> bytes:
    """Capture the candidate scope without staging files or changing HEAD."""

    def git(*args, allowed=(0,)):
        prefix = ["git"]
        if timeout_sec is not None:
            prefix = ["timeout", f"{timeout_sec}s", "git", "--no-optional-locks"]
        result = container.exec_run([*prefix, *args], workdir=workdir)
        if not isinstance(result.output, bytes):
            raise TypeError("container returned non-bytes repository evidence")
        if result.exit_code not in allowed:
            detail = result.output.decode(errors="replace")
            raise RuntimeError(f"repository {args[0]} capture failed: {detail}")
        return result.output

    scope = ["--", ".", ":(exclude).pitbench", ":(exclude).pitbench/**"]
    patch = bytearray(git("diff", "--binary", "--no-ext-diff", "HEAD", *scope))
    paths = git("ls-files", "--others", "--exclude-standard", "-z", *scope)
    for raw_path in paths.split(b"\0"):
        if raw_path:
            patch.extend(
                git(
                    "diff",
                    "--binary",
                    "--no-ext-diff",
                    "--no-index",
                    "--",
                    "/dev/null",
                    raw_path.decode(errors="surrogateescape"),
                    allowed=(0, 1),
                )
            )
    return bytes(patch)
