ARG UBUNTU_BASE_IMAGE=ubuntu:24.04@sha256:d78ab76437b1afc5f01e223d6bf0172763f404bb166441328845adbef44518cb
FROM ${UBUNTU_BASE_IMAGE}

# This mirror retains upstream's academic/non-commercial terms.
ARG SOURCE_REPOSITORY=https://github.com/ProvideQ/lkh-3.git
ARG BASE_COMMIT=580151a4a3504193ea5964a037a351fe6cf87455
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    asciinema build-essential ca-certificates git python3 tmux \
    && rm -rf /var/lib/apt/lists/*
RUN git init --quiet /workspace/repo \
    && git -C /workspace/repo remote add origin "$SOURCE_REPOSITORY" \
    && git -C /workspace/repo -c http.version=HTTP/1.1 \
       -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=30 \
       fetch --depth=1 --no-tags origin "$BASE_COMMIT" \
    && git -C /workspace/repo checkout --quiet --detach FETCH_HEAD \
    && git -C /workspace/repo remote remove origin \
    && make -C /workspace/repo clean \
    && make -C /workspace/repo -j2
WORKDIR /workspace/repo
