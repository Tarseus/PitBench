ARG UBUNTU_BASE_IMAGE=ubuntu:24.04@sha256:d78ab76437b1afc5f01e223d6bf0172763f404bb166441328845adbef44518cb
FROM ${UBUNTU_BASE_IMAGE}

ARG SOURCE_REPOSITORY=https://github.com/coin-or/Cbc.git
ARG BASE_COMMIT

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       asciinema build-essential git pkg-config python3 python3-pip tmux \
       coinor-libclp-dev coinor-libcgl-dev coinor-libcoinutils-dev coinor-libosi-dev \
       zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --break-system-packages --no-cache-dir \
    pydantic==2.13.5 pyyaml==6.0.3
RUN test -n "$BASE_COMMIT" \
    && git init --quiet /workspace/repo \
    && git -C /workspace/repo remote add origin "$SOURCE_REPOSITORY" \
    && git -C /workspace/repo fetch --depth=1 --no-tags origin "$BASE_COMMIT" \
    && git -C /workspace/repo checkout --quiet --detach FETCH_HEAD \
    && git -C /workspace/repo remote remove origin \
    && git -C /workspace/repo reflog expire --expire=now --all \
    && git -C /workspace/repo gc --prune=now --quiet
RUN mkdir /workspace/repo/build \
    && cd /workspace/repo/build \
    && ../configure --disable-shared --enable-static \
    && make -j2

WORKDIR /workspace/repo
