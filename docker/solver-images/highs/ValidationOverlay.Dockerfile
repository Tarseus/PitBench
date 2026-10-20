ARG BASE_IMAGE=pitbench-highs:single-worker
FROM ${BASE_IMAGE}

COPY HPresolve.cpp /workspace/repo/highs/presolve/HPresolve.cpp
RUN cmake --build /workspace/repo/build -j1

WORKDIR /workspace/repo
