FROM maven:3.9-eclipse-temurin-17@sha256:880934ae394bf91bc3e57d573e4fc04774f064f3c4df7ccd7cc10b3b126737bf

ARG UBUNTU_MIRROR=http://archive.ubuntu.com/ubuntu/
ARG UBUNTU_SECURITY_MIRROR=http://security.ubuntu.com/ubuntu/
ARG PYPI_INDEX_URL=https://pypi.org/simple
ARG SOURCE_REPOSITORY=https://github.com/chocoteam/choco-solver.git
ARG BASE_COMMIT

RUN sed -i \
       -e "s|http://archive.ubuntu.com/ubuntu/|$UBUNTU_MIRROR|g" \
       -e "s|http://security.ubuntu.com/ubuntu/|$UBUNTU_SECURITY_MIRROR|g" \
       /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       git python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --break-system-packages --no-cache-dir \
    --index-url "$PYPI_INDEX_URL" \
    pydantic==2.13.5 pyyaml==6.0.3

COPY adapters/jvm /opt/pitbench-jvm
RUN test -n "$BASE_COMMIT" \
    && git init --quiet /workspace/repo \
    && git -C /workspace/repo remote add origin "$SOURCE_REPOSITORY" \
    && git -C /workspace/repo config http.version HTTP/1.1 \
    && for attempt in 1 2 3; do \
       if git -C /workspace/repo \
         -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=30 \
         fetch --depth=1 --no-tags origin "$BASE_COMMIT"; then \
         break; \
       elif [ "$attempt" -eq 3 ]; then \
         exit 1; \
       else \
         sleep 2; \
       fi; \
       done \
    && git -C /workspace/repo checkout --detach FETCH_HEAD \
    && git -C /workspace/repo submodule update --init --depth=1 \
    && git -C /workspace/repo remote remove origin \
    && git -C /workspace/repo/solver/src/main/java/cpp remote remove origin \
    && git -C /workspace/repo/solver/src/main/java/cpp for-each-ref --format='%(refname)' | xargs -r -n1 git -C /workspace/repo/solver/src/main/java/cpp update-ref -d \
    && git -C /workspace/repo/solver/src/main/java/cpp reflog expire --expire=now --all \
    && git -C /workspace/repo/solver/src/main/java/cpp gc --prune=now \
    && git -C /workspace/repo for-each-ref --format='%(refname)' | xargs -r -n1 git -C /workspace/repo update-ref -d \
    && git -C /workspace/repo reflog expire --expire=now --all \
    && git -C /workspace/repo gc --prune=now
RUN cd /workspace/repo \
    && mvn -q package -DskipTests \
    && python3 /opt/pitbench-jvm/runner.py compile --solver choco

ENV PITBENCH_CHOCO_RUNNER="python3 /opt/pitbench-jvm/runner.py run --solver choco"
WORKDIR /workspace/repo
