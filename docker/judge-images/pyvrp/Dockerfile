ARG UBUNTU_BASE_IMAGE=ubuntu:24.04
FROM ${UBUNTU_BASE_IMAGE}

ARG DEBIAN_MIRROR=http://archive.ubuntu.com/ubuntu
ARG DEBIAN_SECURITY_MIRROR=http://security.ubuntu.com/ubuntu
ARG PYPI_INDEX_URL=https://pypi.org/simple

RUN sed -i \
       -e "s|URIs: http://archive.ubuntu.com/ubuntu/|URIs: $DEBIAN_MIRROR/|" \
       -e "s|URIs: http://security.ubuntu.com/ubuntu/|URIs: $DEBIAN_SECURITY_MIRROR/|" \
       /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
       asciinema build-essential cmake git libspdlog-dev ninja-build pkg-config \
       python-is-python3 python3-dev python3-pip python3-venv time tmux \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --break-system-packages --no-cache-dir \
    --index-url "$PYPI_INDEX_URL" \
    docblock==0.1.8 \
    matplotlib==3.11.1 \
    meson==1.12.0 \
    ninja==1.13.0 \
    numpy==2.5.2 \
    pandas==3.0.5 \
    poetry-core==2.4.1 \
    pyarrow==25.0.1 \
    pybind11==3.1.0 \
    pydantic==2.13.4 \
    pytest==9.1.1 \
    pytest-cov==7.1.0 \
    pytest-timeout==2.4.0 \
    pytest-xdist==3.8.0 \
    pyyaml==6.0.3 \
    setuptools==68.1.2 \
    tqdm==4.70.0 \
    vrplib==2.2.0 \
    wheel==0.42.0

ENV PIP_NO_BUILD_ISOLATION=1
RUN mkdir -p /input /output /private /workspace/repo
WORKDIR /workspace/repo
CMD ["sh", "-c", "sleep infinity"]
