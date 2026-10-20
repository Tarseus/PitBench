ARG PREPARED_IMAGE
FROM ${PREPARED_IMAGE}

RUN rm -rf /workspace/repo && mkdir -p /workspace/repo
COPY source/ /workspace/repo/
RUN make -C /workspace/repo/src -j1 'CXX=g++ -include vector -include array'

ENV PITBENCH_VROOM_SOLVER=/workspace/repo/bin/vroom
WORKDIR /workspace/repo
