ARG BASE_IMAGE=pitbench-highs:single-worker
FROM ${BASE_IMAGE}

COPY source/ /workspace/repo/
RUN cmake --build /workspace/repo/build -j1

WORKDIR /workspace/repo
