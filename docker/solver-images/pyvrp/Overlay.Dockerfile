ARG PREPARED_IMAGE=pitbench-pyvrp-history:5a8c327
FROM ${PREPARED_IMAGE}

# The build context contains a git archive of the pinned PR base commit.
# Reuse the prepared compiler and Python environment across historical PRs.
RUN rm -rf /workspace/repo
COPY source/ /workspace/repo/
RUN python3 -m pip install --break-system-packages --no-build-isolation \
      --no-deps --force-reinstall -e /workspace/repo \
    && python3 -c "import pyvrp._pyvrp; import pyvrp.search._search"

WORKDIR /workspace/repo
