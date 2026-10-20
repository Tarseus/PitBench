ARG PREPARED_IMAGE=pitbench-pyvrp-history:5a8c327
FROM ${PREPARED_IMAGE}

# Historical source snapshots can require runtime wheels absent from the
# current prepared image. Supply hash-checked wheels in the build context.
COPY wheels/ /opt/pitbench-wheels/
RUN python3 -m pip install --break-system-packages --no-index --no-deps \
      /opt/pitbench-wheels/*.whl
