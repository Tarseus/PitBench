ARG PREPARED_IMAGE=pitbench-pyvrp-history:5a8c327
FROM ${PREPARED_IMAGE}

RUN rm -rf /workspace/repo
COPY repo/ /workspace/repo/
RUN python3 -m pip install --break-system-packages --no-build-isolation \
      --no-deps --force-reinstall -e /workspace/repo \
    && python3 -c "import pyvrp._pyvrp; import pyvrp.search._search"

WORKDIR /workspace/repo
