ARG PREPARED_IMAGE
FROM ${PREPARED_IMAGE}

ARG EXPECTED_COMMIT
RUN rm -rf /workspace/repo && mkdir -p /workspace/repo
COPY repo/ /workspace/repo/
RUN test "$(git -C /workspace/repo rev-parse HEAD)" = "${EXPECTED_COMMIT}" \
    && make -C /workspace/repo/src -j1 'CXX=g++ -include vector -include array' \
    && test -x /workspace/repo/bin/vroom

ENV PITBENCH_VROOM_SOLVER=/workspace/repo/bin/vroom
WORKDIR /workspace/repo
