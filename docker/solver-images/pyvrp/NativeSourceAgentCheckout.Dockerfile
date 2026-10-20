ARG PREPARED_IMAGE=pitbench-pyvrp-history:poetry-compat
FROM ${PREPARED_IMAGE}

RUN rm -rf /workspace/repo
COPY . /workspace/repo/
RUN cd /workspace/repo \
    && python3 -m venv --system-site-packages .pitbench-venv \
    && .pitbench-venv/bin/python -m pip install --no-build-isolation --no-deps -e . \
    && .pitbench-venv/bin/python -c "import pathlib, runpy, sys, tomllib; sys.modules.setdefault('tomli', tomllib); script = pathlib.Path('build_extensions.py'); runpy.run_path(str(script), run_name='__main__') if script.is_file() else None" \
    && .pitbench-venv/bin/python -c "import pathlib, sys, tomllib; sys.modules.setdefault('tomli', tomllib); import pyvrp; print(pyvrp.__file__)"

WORKDIR /workspace/repo
