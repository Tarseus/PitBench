ARG BASE_IMAGE=pitbench-vroom:noise-calibration
FROM ${BASE_IMAGE}

COPY heuristics.cpp /workspace/repo/src/algorithms/heuristics/heuristics.cpp
RUN make -C /workspace/repo/src -j1

WORKDIR /workspace/repo
