ARG BASE_IMAGE
FROM ${BASE_IMAGE}

COPY source.patch /tmp/pitbench-gecode-pr.patch
RUN cd /workspace/repo \
    && git apply /tmp/pitbench-gecode-pr.patch \
    && cmake --build build --target gecodeint_shared gecodesearch_shared -j2 \
    && g++ -std=c++17 -O3 -I. -Ibuild \
       /opt/pitbench-adapters/ExactSolverRunner.cpp \
       -o build/pitbench-exact -Lbuild \
       -Wl,-rpath,/workspace/repo/build \
       -lgecodeint -lgecodesearch -lgecodekernel -lgecodesupport -pthread

WORKDIR /workspace/repo
