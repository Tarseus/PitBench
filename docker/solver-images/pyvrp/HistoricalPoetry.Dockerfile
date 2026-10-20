ARG PREPARED_IMAGE=pitbench-pyvrp-history:5a8c327
FROM ${PREPARED_IMAGE}

# GCC 14 reports a new dangling-reference warning on unchanged 2023 code.
ENV CXXFLAGS="-Wno-error=dangling-reference"

# Older Meson snapshots call only `poetry version --short` during setup.
RUN printf '%s\n' \
      '#!/usr/bin/env python3' \
      'import pathlib, sys, tomllib' \
      'if sys.argv[1:] != ["version", "--short"]:' \
      '    raise SystemExit("unsupported Poetry build command")' \
      'print(tomllib.loads(pathlib.Path("pyproject.toml").read_text())["tool"]["poetry"]["version"])' \
      > /usr/local/bin/poetry \
    && chmod 755 /usr/local/bin/poetry
