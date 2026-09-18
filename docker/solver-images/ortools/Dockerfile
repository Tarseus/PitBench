FROM maven:3.9-eclipse-temurin-17@sha256:880934ae394bf91bc3e57d573e4fc04774f064f3c4df7ccd7cc10b3b126737bf

ARG SOURCE_REPOSITORY=https://github.com/google/or-tools.git
ARG BASE_COMMIT
ARG PYPI_INDEX_URL=https://pypi.org/simple

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       asciinema build-essential cmake git ninja-build python3 python3-pip swig tmux \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --break-system-packages --no-cache-dir \
    --index-url "$PYPI_INDEX_URL" \
    pydantic==2.13.5 pyyaml==6.0.3
COPY adapters/jvm /opt/pitbench-jvm
RUN test -n "$BASE_COMMIT" \
    && git init --quiet /workspace/repo \
    && git -C /workspace/repo remote add origin "$SOURCE_REPOSITORY" \
    && git -C /workspace/repo fetch --depth=1 --no-tags origin "$BASE_COMMIT" \
    && git -C /workspace/repo checkout --quiet --detach FETCH_HEAD \
    && git -C /workspace/repo remote remove origin \
    && git -C /workspace/repo reflog expire --expire=now --all \
    && git -C /workspace/repo gc --prune=now --quiet
RUN cmake -S /workspace/repo -B /workspace/repo/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DFETCHCONTENT_BASE_DIR=/opt/ortools-deps \
      -DBUILD_JAVA=ON -DBUILD_DEPS=ON \
      -DBUILD_MATH_OPT=ON -DBUILD_TESTING=OFF \
      -DBUILD_SAMPLES=OFF -DBUILD_EXAMPLES=OFF \
      -DBUILD_FLATZINC=OFF \
      -DUSE_BOP=ON -DUSE_GUROBI=ON -DUSE_XPRESS=ON \
      -DUSE_COINOR=OFF -DUSE_HIGHS=OFF -DUSE_PDLP=OFF -DUSE_SCIP=OFF \
    && cmake --build /workspace/repo/build --target java_package -j6

ENV PITBENCH_ORTOOLS_CP_SAT_RUNNER="python3 /opt/pitbench-jvm/runner.py run --solver ortools_cp_sat"
WORKDIR /workspace/repo
