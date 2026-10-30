ARG UBUNTU_BASE_IMAGE=ubuntu:24.04@sha256:d78ab76437b1afc5f01e223d6bf0172763f404bb166441328845adbef44518cb
FROM ${UBUNTU_BASE_IMAGE}

ARG SOURCE_REPOSITORY=https://github.com/Gecode/gecode.git
ARG BASE_COMMIT

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       asciinema build-essential cmake git python3 tmux \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY adapters/cpp/ExactSolverRunner.cpp /opt/pitbench-adapters/ExactSolverRunner.cpp
COPY adapters/pitbench/exact_solver_runner.py /opt/pitbench-adapters/exact_solver_runner.py
RUN chmod -R a+rX /opt/pitbench-adapters
RUN test -n "$BASE_COMMIT" \
    && git init --quiet /workspace/repo \
    && git -C /workspace/repo remote add origin "$SOURCE_REPOSITORY" \
    && git -C /workspace/repo fetch --depth=1 --no-tags origin "$BASE_COMMIT" \
    && git -C /workspace/repo checkout --quiet --detach FETCH_HEAD \
    && git -C /workspace/repo remote remove origin \
    && git -C /workspace/repo reflog expire --expire=now --all \
    && git -C /workspace/repo gc --prune=now --quiet
RUN --network=none cmake -S /workspace/repo -B /workspace/repo/build \
      -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
      -DBUILD_SHARED_LIBS=ON -DGECODE_BUILD_SHARED=ON -DGECODE_BUILD_STATIC=OFF \
    && cmake --build /workspace/repo/build --target help > /tmp/pitbench-targets.txt \
    && if grep -q gecodeint_shared /tmp/pitbench-targets.txt; then \
         cmake --build /workspace/repo/build --target gecodeint_shared gecodesearch_shared -j2; \
       else \
         cmake --build /workspace/repo/build --target gecodeint gecodesearch -j2; \
       fi
RUN cd /workspace/repo \
    && g++ -std=c++17 -O3 -I. -Ibuild \
       /opt/pitbench-adapters/ExactSolverRunner.cpp \
       -o build/pitbench-exact -Lbuild \
       -Wl,-rpath,/workspace/repo/build \
       -lgecodeint -lgecodesearch -lgecodekernel -lgecodesupport -pthread
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --break-system-packages --no-cache-dir \
    pydantic==2.13.5 pyyaml==6.0.3

ENV PITBENCH_GECODE_RUNNER="python3 /opt/pitbench-adapters/exact_solver_runner.py --engine gecode"
ENV PITBENCH_EXACT_SOLVER="build/pitbench-exact"
WORKDIR /workspace/repo
