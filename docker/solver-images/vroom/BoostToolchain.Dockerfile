FROM pitbench-vroom:noise-calibration

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       libboost-log-dev libboost-system-dev libboost-thread-dev \
    && rm -rf /var/lib/apt/lists/*
