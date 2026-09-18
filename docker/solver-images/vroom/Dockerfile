FROM ubuntu:24.04@sha256:d78ab76437b1afc5f01e223d6bf0172763f404bb166441328845adbef44518cb

ARG SOURCE_REPOSITORY=https://github.com/VROOM-Project/vroom.git
ARG BASE_COMMIT
ARG PYPI_INDEX_URL=https://pypi.org/simple

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       build-essential git libasio-dev libglpk-dev libssl-dev pkg-config \
       python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --break-system-packages --no-cache-dir \
    --index-url "$PYPI_INDEX_URL" \
    pydantic==2.13.5 pyyaml==6.0.3
RUN test -n "$BASE_COMMIT" \
    && git init --quiet /workspace/repo \
    && git -C /workspace/repo remote add origin "$SOURCE_REPOSITORY" \
    && git -C /workspace/repo fetch --depth=1 --no-tags origin "$BASE_COMMIT" \
    && git -C /workspace/repo checkout --quiet --detach FETCH_HEAD \
    && git -C /workspace/repo submodule update --init --depth=1 \
    && git -C /workspace/repo submodule foreach --recursive \
       'git remote remove origin || true; git for-each-ref --format="%(refname)" | xargs -r -n1 git update-ref -d; git reflog expire --expire=now --all; git gc --prune=now --quiet' \
    && git -C /workspace/repo remote remove origin \
    && git -C /workspace/repo for-each-ref --format='%(refname)' | xargs -r -n1 git -C /workspace/repo update-ref -d \
    && git -C /workspace/repo reflog expire --expire=now --all \
    && git -C /workspace/repo gc --prune=now --quiet
RUN make -C /workspace/repo/src -j1

WORKDIR /workspace/repo
