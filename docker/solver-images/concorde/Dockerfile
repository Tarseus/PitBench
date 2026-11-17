ARG UBUNTU_BASE_IMAGE=ubuntu:24.04@sha256:d78ab76437b1afc5f01e223d6bf0172763f404bb166441328845adbef44518cb
FROM ${UBUNTU_BASE_IMAGE}

# Concorde and QSopt are provided by their authors for academic research.
ARG CONCORDE_SOURCE_URL=https://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/src/co031219.tgz
ARG QSOPT_BASE_URL=https://www.math.uwaterloo.ca/~bico/qsopt/beta/codes/PIC
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    asciinema build-essential ca-certificates curl git python3 tmux \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/qsopt /workspace/repo \
    && curl -fsSL "$QSOPT_BASE_URL/qsopt.PIC.a" -o /opt/qsopt/qsopt.a \
    && curl -fsSL "$QSOPT_BASE_URL/qsopt.h" -o /opt/qsopt/qsopt.h \
    && curl -fsSL "$CONCORDE_SOURCE_URL" | tar -xz --strip-components=2 -C /workspace/repo
WORKDIR /workspace/repo
RUN git init --quiet && git add . \
    && git -c user.name=PitBench -c user.email=pitbench@localhost commit -qm 'Import Concorde source' \
    && CFLAGS='-O2 -fPIC -std=gnu89' ./configure --with-qsopt=/opt/qsopt \
    && make -j2
