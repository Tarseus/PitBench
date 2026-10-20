ARG PREPARED_IMAGE=pitbench-vroom:noise-calibration
FROM ${PREPARED_IMAGE}

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       libboost-log-dev libboost-system-dev libboost-thread-dev \
    && rm -rf /var/lib/apt/lists/*

RUN rm -rf /workspace/repo
COPY repo/ /workspace/repo/
RUN make -C /workspace/repo/src -j1 'CXX=g++ -include vector -include array'

ENV PITBENCH_VROOM_SOLVER=/workspace/repo/bin/vroom
WORKDIR /workspace/repo
