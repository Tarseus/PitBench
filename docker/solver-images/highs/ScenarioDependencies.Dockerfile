ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Dependencies for the source-built HiGHS Python scenario binding.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3-dev python3-numpy python3-scipy pybind11-dev \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --break-system-packages --no-cache-dir pybind11==3.0.1
