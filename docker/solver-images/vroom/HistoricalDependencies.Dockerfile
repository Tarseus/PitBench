ARG PREPARED_IMAGE=pitbench-vroom:noise-calibration
FROM ${PREPARED_IMAGE}

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       libboost-dev libboost-system-dev \
    && rm -rf /var/lib/apt/lists/*
