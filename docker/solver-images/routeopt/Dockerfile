ARG UBUNTU_BASE_IMAGE=ubuntu:24.04@sha256:d78ab76437b1afc5f01e223d6bf0172763f404bb166441328845adbef44518cb
FROM ${UBUNTU_BASE_IMAGE}

ARG SOURCE_REPOSITORY=https://github.com/Zhengzhong-You/RouteOpt.git
ARG BASE_COMMIT=c3f943f65f347d562d4eb8368bf037301f6c2914
ARG GUROBI_SDK_URL=https://packages.gurobi.com/13.0/gurobi13.0.0_linux64.tar.gz
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    asciinema build-essential ca-certificates cmake curl git ninja-build python3 tmux zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/gurobi \
    && curl -fsSL "$GUROBI_SDK_URL" | tar -xz -C /opt/gurobi --strip-components=1
ENV GUROBI_HOME=/opt/gurobi/linux64
ENV LD_LIBRARY_PATH=/opt/gurobi/linux64/lib
RUN git init --quiet /workspace/repo \
    && git -C /workspace/repo remote add origin "$SOURCE_REPOSITORY" \
    && git -C /workspace/repo -c http.version=HTTP/1.1 \
       -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=30 \
       fetch --depth=1 --no-tags origin "$BASE_COMMIT" \
    && git -C /workspace/repo checkout --quiet --detach FETCH_HEAD \
    && git -C /workspace/repo remote remove origin
WORKDIR /workspace/repo
# The upstream script builds XGBoost, HGS and CVRPSEP and configures Gurobi.
RUN printf '%s\n' "$GUROBI_HOME" | python3 -c 'import build; build.cpu_jobs = lambda: 2; build.main()' \
    && cmake -S packages/application/cvrp -B packages/application/cvrp/build \
       -DCMAKE_BUILD_TYPE=Release -DLABEL_ASSIGN_OVERRIDE=160000 \
    && cmake --build packages/application/cvrp/build --parallel 2
