ARG PYTHON_BASE_IMAGE=python:3.13-trixie
FROM ${PYTHON_BASE_IMAGE}

ARG DEBIAN_MIRROR=http://deb.debian.org/debian
ARG DEBIAN_SECURITY_MIRROR=http://deb.debian.org/debian-security
ARG PYPI_INDEX_URL=https://pypi.org/simple

RUN sed -i \
       -e "s|URIs: http://deb.debian.org/debian$|URIs: $DEBIAN_MIRROR|" \
       -e "s|URIs: http://deb.debian.org/debian-security$|URIs: $DEBIAN_SECURITY_MIRROR|" \
       /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
       asciinema build-essential cmake git libspdlog-dev ninja-build python3-dev \
       time tmux \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --break-system-packages --no-cache-dir \
    --index-url "$PYPI_INDEX_URL" \
    docblock matplotlib meson ninja numpy poetry-core pybind11 \
    pydantic pytest pytest-cov pytest-timeout pytest-xdist pyyaml setuptools \
    tqdm vrplib wheel
ARG SOURCE_REPOSITORY=https://github.com/PyVRP/PyVRP.git
ARG BASE_COMMIT
RUN test -n "$BASE_COMMIT" \
    && git init /workspace/repo \
    && git -C /workspace/repo remote add origin "$SOURCE_REPOSITORY" \
    && git -C /workspace/repo config http.version HTTP/1.1 \
    && for attempt in 1 2 3; do \
       if git -C /workspace/repo fetch --depth=1 --no-tags origin "$BASE_COMMIT"; then \
         break; \
       elif [ "$attempt" -eq 3 ]; then \
         exit 1; \
       else \
         sleep 2; \
       fi; \
       done \
    && git -C /workspace/repo checkout --detach FETCH_HEAD \
    && git -C /workspace/repo remote remove origin \
    && git -C /workspace/repo reflog expire --expire=now --all \
    && git -C /workspace/repo gc --prune=now
RUN python3 -m pip install --break-system-packages \
    --no-build-isolation --no-deps -e /workspace/repo
RUN python3 -c "import pyvrp._pyvrp; import pyvrp.search._search"

ENV PIP_NO_BUILD_ISOLATION=1
WORKDIR /workspace/repo
