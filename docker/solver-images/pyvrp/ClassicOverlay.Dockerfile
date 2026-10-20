ARG PREPARED_IMAGE=pitbench-pyvrp-history:poetry-compat
FROM ${PREPARED_IMAGE}

RUN rm -rf /workspace/repo
COPY source/ /workspace/repo/
RUN cd /workspace/repo \
    && python3 -m venv --system-site-packages .pitbench-venv \
    && .pitbench-venv/bin/python -m pip install --no-build-isolation \
       --no-deps -e . \
    && .pitbench-venv/bin/python -c "import pyvrp._pyvrp; import pyvrp.search._search"

WORKDIR /workspace/repo
