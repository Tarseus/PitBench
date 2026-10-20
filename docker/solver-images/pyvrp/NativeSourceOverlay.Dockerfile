ARG PREPARED_IMAGE=pitbench-pyvrp-history:poetry-compat
FROM ${PREPARED_IMAGE}

RUN rm -rf /workspace/repo
COPY source/ /workspace/repo/
RUN cd /workspace/repo \
    && python3 -m venv --system-site-packages .pitbench-venv \
    && .pitbench-venv/bin/python -m pip install --no-build-isolation --no-deps -e . \
    && .pitbench-venv/bin/python -c "import runpy, sys, tomllib; sys.modules.setdefault('tomli', tomllib); runpy.run_path('build_extensions.py', run_name='__main__')" \
    && .pitbench-venv/bin/python -c "import sys, tomllib; sys.modules.setdefault('tomli', tomllib); import pyvrp; print(pyvrp.__file__)"

WORKDIR /workspace/repo
