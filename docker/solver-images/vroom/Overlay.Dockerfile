ARG PREPARED_IMAGE=pitbench-vroom:noise-calibration
FROM ${PREPARED_IMAGE}

RUN rm -rf /workspace/repo
COPY source/ /workspace/repo/
RUN make -C /workspace/repo/src -j1

ENV PITBENCH_VROOM_SOLVER=/workspace/repo/bin/vroom
WORKDIR /workspace/repo
