ARG PREPARED_IMAGE=pitbench-pyvrp-history:poetry-compat
FROM ${PREPARED_IMAGE}

RUN rm -rf /workspace/repo
COPY source/ /workspace/repo/
RUN cd /workspace/repo \
    && git submodule absorbgitdirs \
    && git submodule update --init --recursive --no-fetch \
    && (git remote remove origin 2>/dev/null || true) \
    && git submodule foreach --recursive 'git remote remove origin 2>/dev/null || true'
RUN python3 -m pip install --break-system-packages --no-cache-dir tomli==2.2.1
RUN cd /workspace/repo \
    && python3 -m venv --system-site-packages .pitbench-venv \
    && .pitbench-venv/bin/python -m pip install --no-cache-dir tomli==2.2.1 \
    && .pitbench-venv/bin/python -m pip install --no-build-isolation --no-deps -e . \
    && if [ -f build_extensions.py ]; then .pitbench-venv/bin/python build_extensions.py; fi \
    && .pitbench-venv/bin/python -c "import pyvrp; print(pyvrp.__file__)"

WORKDIR /workspace/repo
